//! pickle_idx.rs
//! Pure Rust inert pickle opcode virtual machine.
//! Safely decodes RPA index dictionaries without executing any Python code or invoking unpicklers.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PickleError {
    #[error("Unexpected end of pickle stream at offset {0}")]
    UnexpectedEof(usize),
    #[error("Unsupported or dangerous pickle opcode 0x{0:02x} ('{1}') at offset {2}")]
    DangerousOpcode(u8, char, usize),
    #[error("Pickle stack underflow")]
    StackUnderflow,
    #[error("Pickle mark not found")]
    MarkNotFound,
    #[error("Invalid UTF-8 in pickle string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("Malformed index structure in RPA archive")]
    MalformedIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<PickleValue>),
    Tuple(Vec<PickleValue>),
    Dict(HashMap<String, PickleValue>),
    Mark,
}

#[derive(Debug, Clone)]
pub struct RpaFileEntry {
    pub offset: u64,
    pub length: u64,
    pub prefix: Vec<u8>,
}

pub struct InertPickleVm<'a> {
    reader: Cursor<&'a [u8]>,
    stack: Vec<PickleValue>,
    memo: HashMap<u32, PickleValue>,
}

impl<'a> InertPickleVm<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: Cursor::new(data),
            stack: Vec::with_capacity(64),
            memo: HashMap::new(),
        }
    }

    pub fn parse_rpa_index(mut self) -> Result<HashMap<String, Vec<RpaFileEntry>>, PickleError> {
        let value = self.execute()?;
        match value {
            PickleValue::Dict(dict) => {
                let mut result = HashMap::new();
                for (filename, val) in dict {
                    let entries = extract_file_entries(&val)?;
                    result.insert(filename, entries);
                }
                Ok(result)
            }
            _ => Err(PickleError::MalformedIndex),
        }
    }

    fn execute(&mut self) -> Result<PickleValue, PickleError> {
        loop {
            let pos = self.reader.position() as usize;
            let opcode = match self.reader.read_u8() {
                Ok(b) => b,
                Err(_) => return Err(PickleError::UnexpectedEof(pos)),
            };

            match opcode {
                // PROTO (0x80)
                0x80 => {
                    let _proto = self.read_u8()?;
                }
                // FRAME (0x95) - Python protocol 4+ frame indicator
                0x95 => {
                    let _frame_len = self.reader.read_u64::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))?;
                }
                // EMPTY_DICT (0x7d, '}')
                b'}' => {
                    self.stack.push(PickleValue::Dict(HashMap::new()));
                }
                // EMPTY_LIST (0x5d, ']')
                b']' => {
                    self.stack.push(PickleValue::List(Vec::new()));
                }
                // EMPTY_TUPLE (0x29, ')')
                b')' => {
                    self.stack.push(PickleValue::Tuple(Vec::new()));
                }
                // MARK (0x28, '(')
                b'(' => {
                    self.stack.push(PickleValue::Mark);
                }
                // NONE (0x4e, 'N')
                b'N' => {
                    self.stack.push(PickleValue::None);
                }
                // BININT (0x4a, 'J')
                b'J' => {
                    let val = self.reader.read_i32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))?;
                    self.stack.push(PickleValue::Int(val as i64));
                }
                // BININT1 (0x4b, 'K')
                b'K' => {
                    let val = self.read_u8()?;
                    self.stack.push(PickleValue::Int(val as i64));
                }
                // BININT2 (0x4d, 'M')
                b'M' => {
                    let val = self.reader.read_u16::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))?;
                    self.stack.push(PickleValue::Int(val as i64));
                }
                // LONG1 (0x8a)
                0x8a => {
                    let len = self.read_u8()? as usize;
                    let val = self.read_long_bytes(len)?;
                    self.stack.push(PickleValue::Int(val));
                }
                // LONG4 (0x8b)
                0x8b => {
                    let len = self.reader.read_u32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))? as usize;
                    let val = self.read_long_bytes(len)?;
                    self.stack.push(PickleValue::Int(val));
                }
                // BINUNICODE (0x58, 'X')
                b'X' => {
                    let len = self.reader.read_u32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    let s = String::from_utf8(buf)?;
                    self.stack.push(PickleValue::String(s));
                }
                // SHORT_BINUNICODE (0x8c)
                0x8c => {
                    let len = self.read_u8()? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    let s = String::from_utf8(buf)?;
                    self.stack.push(PickleValue::String(s));
                }
                // BINBYTES (0x42, 'B')
                b'B' => {
                    let len = self.reader.read_u32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    self.stack.push(PickleValue::Bytes(buf));
                }
                // BINBYTES8 (0x8e) - 64-bit length bytes
                0x8e => {
                    let len = self.reader.read_u64::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    self.stack.push(PickleValue::Bytes(buf));
                }
                // SHORT_BINBYTES (0x43, 'C') - 8-bit length bytes
                b'C' => {
                    let len = self.read_u8()? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    self.stack.push(PickleValue::Bytes(buf));
                }
                // BINSTRING (0x54, 'T') - Python 2
                b'T' => {
                    let len = self.reader.read_u32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    let s = String::from_utf8(buf.clone()).unwrap_or_else(|_| String::from_utf8_lossy(&buf).to_string());
                    self.stack.push(PickleValue::String(s));
                }
                // SHORT_BINSTRING (0x55, 'U') - Python 2
                b'U' => {
                    let len = self.read_u8()? as usize;
                    let mut buf = vec![0u8; len];
                    self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
                    let s = String::from_utf8(buf.clone()).unwrap_or_else(|_| String::from_utf8_lossy(&buf).to_string());
                    self.stack.push(PickleValue::String(s));
                }
                // TUPLE1 (0x85)
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a]));
                }
                // TUPLE2 (0x86)
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a, b]));
                }
                // TUPLE3 (0x87)
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a, b, c]));
                }
                // TUPLE (0x74, 't')
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(PickleValue::Tuple(items));
                }
                // APPEND (0x61, 'a')
                b'a' => {
                    let val = self.pop()?;
                    if let Some(PickleValue::List(ref mut list)) = self.stack.last_mut() {
                        list.push(val);
                    } else {
                        return Err(PickleError::MalformedIndex);
                    }
                }
                // APPENDS (0x65, 'e')
                b'e' => {
                    let items = self.pop_mark()?;
                    if let Some(PickleValue::List(ref mut list)) = self.stack.last_mut() {
                        list.extend(items);
                    } else {
                        return Err(PickleError::MalformedIndex);
                    }
                }
                // SETITEM (0x73, 's')
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    if let Some(PickleValue::Dict(ref mut dict)) = self.stack.last_mut() {
                        let key_str = match key {
                            PickleValue::String(s) => s,
                            _ => return Err(PickleError::MalformedIndex),
                        };
                        dict.insert(key_str, value);
                    } else {
                        return Err(PickleError::MalformedIndex);
                    }
                }
                // SETITEMS (0x75, 'u')
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(PickleError::MalformedIndex);
                    }
                    if let Some(PickleValue::Dict(ref mut dict)) = self.stack.last_mut() {
                        for chunk in items.chunks_exact(2) {
                            let k = match &chunk[0] {
                                PickleValue::String(s) => s.clone(),
                                _ => return Err(PickleError::MalformedIndex),
                            };
                            dict.insert(k, chunk[1].clone());
                        }
                    } else {
                        return Err(PickleError::MalformedIndex);
                    }
                }
                // DICT (0x64, 'd')
                b'd' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(PickleError::MalformedIndex);
                    }
                    let mut map = HashMap::new();
                    for chunk in items.chunks_exact(2) {
                        if let PickleValue::String(ref k) = chunk[0] {
                            map.insert(k.clone(), chunk[1].clone());
                        }
                    }
                    self.stack.push(PickleValue::Dict(map));
                }
                // BINPUT (0x71, 'q')
                b'q' => {
                    let idx = self.read_u8()? as u32;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(idx, top.clone());
                    }
                }
                // LONG_BINPUT (0x72, 'r')
                b'r' => {
                    let idx = self.reader.read_u32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))?;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(idx, top.clone());
                    }
                }
                // MEMOIZE (0x94)
                0x94 => {
                    let idx = self.memo.len() as u32;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(idx, top.clone());
                    }
                }
                // BINGET (0x68, 'h')
                b'h' => {
                    let idx = self.read_u8()? as u32;
                    let val = self.memo.get(&idx).cloned().ok_or(PickleError::MalformedIndex)?;
                    self.stack.push(val);
                }
                // LONG_BINGET (0x6a, 'j')
                b'j' => {
                    let idx = self.reader.read_u32::<LittleEndian>().map_err(|_| PickleError::UnexpectedEof(pos))?;
                    let val = self.memo.get(&idx).cloned().ok_or(PickleError::MalformedIndex)?;
                    self.stack.push(val);
                }
                // STOP (0x2e, '.')
                b'.' => {
                    return self.pop();
                }
                // Any other opcode (especially REDUCE 'R', GLOBAL 'c', INST 'i', BUILD 'b') is considered unsafe!
                _ => {
                    return Err(PickleError::DangerousOpcode(opcode, opcode as char, pos));
                }
            }
        }
    }

    fn read_u8(&mut self) -> Result<u8, PickleError> {
        let pos = self.reader.position() as usize;
        self.reader.read_u8().map_err(|_| PickleError::UnexpectedEof(pos))
    }

    fn read_long_bytes(&mut self, len: usize) -> Result<i64, PickleError> {
        let pos = self.reader.position() as usize;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf).map_err(|_| PickleError::UnexpectedEof(pos))?;
        if buf.is_empty() {
            return Ok(0);
        }
        let mut val: i64 = 0;
        for (i, &b) in buf.iter().enumerate().take(8) {
            val |= (b as i64) << (i * 8);
        }
        // Sign extension if highest bit set
        if (buf.last().unwrap() & 0x80) != 0 && buf.len() < 8 {
            for i in buf.len()..8 {
                val |= 0xffi64 << (i * 8);
            }
        }
        Ok(val)
    }

    fn pop(&mut self) -> Result<PickleValue, PickleError> {
        self.stack.pop().ok_or(PickleError::StackUnderflow)
    }

    fn pop_mark(&mut self) -> Result<Vec<PickleValue>, PickleError> {
        let mut items = Vec::new();
        while let Some(top) = self.stack.pop() {
            if top == PickleValue::Mark {
                items.reverse();
                return Ok(items);
            }
            items.push(top);
        }
        Err(PickleError::MarkNotFound)
    }
}

fn extract_file_entries(val: &PickleValue) -> Result<Vec<RpaFileEntry>, PickleError> {
    let mut entries = Vec::new();
    match val {
        PickleValue::List(chunks) => {
            for chunk in chunks {
                if let Some(entry) = parse_single_chunk(chunk) {
                    entries.push(entry);
                }
            }
        }
        PickleValue::Tuple(_) => {
            if let Some(entry) = parse_single_chunk(val) {
                entries.push(entry);
            }
        }
        _ => return Err(PickleError::MalformedIndex),
    }
    Ok(entries)
}

fn parse_single_chunk(chunk: &PickleValue) -> Option<RpaFileEntry> {
    match chunk {
        PickleValue::Tuple(fields) => {
            if fields.len() >= 2 {
                let offset = match fields[0] {
                    PickleValue::Int(i) => i as u64,
                    _ => return None,
                };
                let length = match fields[1] {
                    PickleValue::Int(i) => i as u64,
                    _ => return None,
                };
                let prefix = if fields.len() >= 3 {
                    match &fields[2] {
                        PickleValue::Bytes(b) => b.clone(),
                        PickleValue::String(s) => s.as_bytes().to_vec(),
                        _ => Vec::new(),
                    }
                } else {
                    Vec::new()
                };
                Some(RpaFileEntry { offset, length, prefix })
            } else {
                None
            }
        }
        _ => None,
    }
}
