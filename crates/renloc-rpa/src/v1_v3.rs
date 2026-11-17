//! v1_v3.rs
//! Core decompression, multi-part chunked archive stitching, and slicing logic for:
//! RPA-1.0, RPA-2.0, RPA-3.0, RPA-3.2, RPA-4.0, ALT-1.0, and ZiX-12A/12B.

use flate2::read::ZlibDecoder;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::header::{parse_rpa_header, RpaHeaderInfo, RpaVersion};
use crate::pickle_idx::{InertPickleVm, PickleError, RpaFileEntry};

#[derive(Debug, Error)]
pub enum RpaError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Header parse error: {0}")]
    Header(#[from] crate::header::HeaderError),
    #[error("Pickle parse error: {0}")]
    Pickle(#[from] PickleError),
    #[error("Decompression failed: {0}")]
    Decompression(String),
    #[error("File not found in archive: {0}")]
    FileNotFound(String),
    #[error("Corrupted slice or offset out of bounds: offset={0}, len={1}, archive_len={2}")]
    OutOfBounds(u64, u64, usize),
}

/// Splits a filename into base archive name and part identifier if matching multi-part patterns.
/// Examples:
///   "archive.rpa.001" -> Some(("archive.rpa", "001"))
///   "data.rpa.1"      -> Some(("data.rpa", "1"))
///   "archive.rpa.part02" -> Some(("archive.rpa", "part02"))
pub fn split_multipart_filename(filename: &str) -> Option<(&str, &str)> {
    if let Some(idx) = filename.rfind('.') {
        let (base, ext) = (&filename[..idx], &filename[idx + 1..]);
        if !ext.is_empty() && ext.chars().all(|c| c.is_ascii_digit()) {
            return Some((base, ext));
        }
        if let Some(rest) = ext.strip_prefix("part") {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return Some((base, ext));
            }
        }
    }
    None
}

/// Discovers sibling parts of a multi-part archive given a starting part or base name.
pub fn discover_sibling_parts(first_part: &Path) -> Vec<PathBuf> {
    let mut parts = vec![first_part.to_path_buf()];
    let filename = match first_part.file_name().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return parts,
    };

    let parent = match first_part.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };

    if let Some((base, ext)) = split_multipart_filename(filename) {
        if let Some(digits_str) = ext.strip_prefix("part") {
            if let Ok(first_num) = digits_str.parse::<usize>() {
                let width = digits_str.len();
                let mut next_num = first_num + 1;
                loop {
                    let candidate =
                        parent.join(format!("{}.part{:0width$}", base, next_num, width = width));
                    if candidate.is_file() {
                        parts.push(candidate);
                        next_num += 1;
                    } else {
                        break;
                    }
                }
            }
        } else if let Ok(first_num) = ext.parse::<usize>() {
            let width = ext.len();
            let mut next_num = first_num + 1;
            loop {
                let candidate_formatted =
                    parent.join(format!("{}.{:0width$}", base, next_num, width = width));
                if candidate_formatted.is_file() {
                    parts.push(candidate_formatted);
                    next_num += 1;
                    continue;
                }
                let candidate_raw = parent.join(format!("{}.{}", base, next_num));
                if candidate_raw.is_file() {
                    parts.push(candidate_raw);
                    next_num += 1;
                    continue;
                }
                break;
            }
        }
    } else if filename.ends_with(".rpa") || filename.ends_with(".rpi") || filename.ends_with(".arc")
    {
        // If "archive.rpa" does not exist or has sibling parts:
        let part1 = parent.join(format!("{}.001", filename));
        if part1.is_file() {
            return discover_sibling_parts(&part1);
        }
        let part1_raw = parent.join(format!("{}.1", filename));
        if part1_raw.is_file() {
            return discover_sibling_parts(&part1_raw);
        }
        let part1_part = parent.join(format!("{}.part1", filename));
        if part1_part.is_file() {
            return discover_sibling_parts(&part1_part);
        }
    }

    parts
}

/// Composite multi-part memory mapped storage supporting seamless reads across
/// arbitrary part boundaries.
pub struct MultiPartStorage {
    mmaps: Vec<Mmap>,
    part_offsets: Vec<u64>,
    total_len: u64,
}

impl MultiPartStorage {
    pub fn open(paths: &[PathBuf]) -> Result<Self, RpaError> {
        if paths.is_empty() {
            return Err(RpaError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No archive paths provided for multi-part storage",
            )));
        }

        let mut mmaps = Vec::with_capacity(paths.len());
        let mut part_offsets = Vec::with_capacity(paths.len());
        let mut total_len = 0u64;

        for path in paths {
            let file = File::open(path)?;
            let meta = file.metadata()?;
            let len = meta.len();
            if len == 0 {
                continue;
            }
            let mmap = unsafe { Mmap::map(&file)? };
            part_offsets.push(total_len);
            total_len += len;
            mmaps.push(mmap);
        }

        if mmaps.is_empty() {
            return Err(RpaError::OutOfBounds(0, 0, 0));
        }

        Ok(Self {
            mmaps,
            part_offsets,
            total_len,
        })
    }

    pub fn len(&self) -> u64 {
        self.total_len
    }

    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Maps a global offset in the virtual stitched archive to (part_index, local_offset).
    pub fn locate(&self, pos: u64) -> (usize, usize) {
        if self.part_offsets.is_empty() {
            return (0, 0);
        }
        let mut part_idx = 0;
        for (i, &offset) in self.part_offsets.iter().enumerate() {
            if offset <= pos {
                part_idx = i;
            } else {
                break;
            }
        }
        let local_offset = (pos - self.part_offsets[part_idx]) as usize;
        (part_idx, local_offset)
    }

    /// Reads `len` bytes starting at `offset`, stitching bytes across multiple parts if needed.
    pub fn read_bytes(&self, offset: u64, len: usize, dest: &mut Vec<u8>) -> Result<(), RpaError> {
        if offset + len as u64 > self.total_len {
            return Err(RpaError::OutOfBounds(
                offset,
                len as u64,
                self.total_len as usize,
            ));
        }

        let mut remaining = len;
        let mut current_offset = offset;
        while remaining > 0 {
            let (part_idx, local_offset) = self.locate(current_offset);
            let part_mmap = &self.mmaps[part_idx];
            let available = part_mmap.len() - local_offset;
            let take = remaining.min(available);
            dest.extend_from_slice(&part_mmap[local_offset..local_offset + take]);
            current_offset += take as u64;
            remaining -= take;
        }
        Ok(())
    }

    /// Returns a new vector containing exactly `len` bytes from `offset`.
    pub fn slice_exact(&self, offset: u64, len: usize) -> Result<Vec<u8>, RpaError> {
        let mut buf = Vec::with_capacity(len);
        self.read_bytes(offset, len, &mut buf)?;
        Ok(buf)
    }
}

/// Streaming reader cursor over `MultiPartStorage` that implements `std::io::Read`.
pub struct MultiPartReaderCursor<'a> {
    storage: &'a MultiPartStorage,
    position: u64,
}

impl<'a> MultiPartReaderCursor<'a> {
    pub fn new(storage: &'a MultiPartStorage, start_offset: u64) -> Self {
        Self {
            storage,
            position: start_offset,
        }
    }
}

impl<'a> std::io::Read for MultiPartReaderCursor<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.storage.total_len || buf.is_empty() {
            return Ok(0);
        }
        let available_total = (self.storage.total_len - self.position) as usize;
        let to_read = buf.len().min(available_total);
        if to_read == 0 {
            return Ok(0);
        }

        let (part_idx, local_offset) = self.storage.locate(self.position);
        let part_mmap = &self.storage.mmaps[part_idx];
        let available_in_part = part_mmap.len() - local_offset;
        let read_now = to_read.min(available_in_part);

        buf[..read_now].copy_from_slice(&part_mmap[local_offset..local_offset + read_now]);
        self.position += read_now as u64;
        Ok(read_now)
    }
}

pub struct RpaReader {
    paths: Vec<PathBuf>,
    storage: MultiPartStorage,
    header: RpaHeaderInfo,
    index: HashMap<String, Vec<RpaFileEntry>>,
}

impl RpaReader {
    /// Opens an archive file, automatically detecting sibling parts if it is a multi-part file.
    pub fn open<P: AsRef<Path>>(archive_path: P) -> Result<Self, RpaError> {
        let path = archive_path.as_ref();
        let parts = discover_sibling_parts(path);
        Self::open_multipart(&parts)
    }

    /// Opens an explicit sequence of multi-part archive files, stitching them into a single archive.
    pub fn open_multipart<P: AsRef<Path>>(part_paths: &[P]) -> Result<Self, RpaError> {
        let paths: Vec<PathBuf> = part_paths
            .iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        if paths.is_empty() {
            return Err(RpaError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No archive paths provided",
            )));
        }

        let storage = MultiPartStorage::open(&paths)?;

        // Check if there is a separate .rpi file (RPA-1.0)
        let primary_path = &paths[0];
        let rpi_candidate_1 = primary_path.with_extension("rpi");
        let rpi_candidate_2 = if let Some(base) = primary_path
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(split_multipart_filename)
            .map(|(b, _)| b)
        {
            let p = primary_path.parent().unwrap_or_else(|| Path::new("."));
            p.join(base).with_extension("rpi")
        } else {
            primary_path.with_extension("rpi")
        };

        let rpi_path = if rpi_candidate_1.is_file() {
            Some(rpi_candidate_1)
        } else if rpi_candidate_2.is_file() {
            Some(rpi_candidate_2)
        } else {
            None
        };

        let (header, index) = if let Some(rpi) = rpi_path {
            let rpi_bytes = std::fs::read(&rpi)?;
            let header = parse_rpa_header(&rpi_bytes)?;
            let mut decoder = ZlibDecoder::new(&rpi_bytes[..]);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| RpaError::Decompression(e.to_string()))?;
            let vm = InertPickleVm::new(&decompressed);
            let index = vm.parse_rpa_index()?;
            (header, index)
        } else {
            let header_bytes = storage.slice_exact(0, 128.min(storage.len() as usize))?;
            let header = parse_rpa_header(&header_bytes)?;
            let index = match header.version {
                RpaVersion::Rpa1 => Self::decompress_and_parse_index(&storage, 0, None)?,
                RpaVersion::Rpa2 => {
                    Self::decompress_and_parse_index(&storage, header.index_offset, None)?
                }
                RpaVersion::Rpa3
                | RpaVersion::Custom8Byte(_)
                | RpaVersion::Rpa32
                | RpaVersion::Rpa40
                | RpaVersion::Alt10 => Self::decompress_and_parse_index(
                    &storage,
                    header.index_offset,
                    header.obfuscation_key,
                )?,
                RpaVersion::ZiX12A | RpaVersion::ZiX12B => {
                    // ZiX variant requires Live Ren'Py In-Game VFS Fallback
                    HashMap::new()
                }
            };
            (header, index)
        };

        Ok(Self {
            paths,
            storage,
            header,
            index,
        })
    }

    fn decompress_and_parse_index(
        storage: &MultiPartStorage,
        offset: u64,
        key: Option<u32>,
    ) -> Result<HashMap<String, Vec<RpaFileEntry>>, RpaError> {
        if offset >= storage.total_len {
            return Err(RpaError::OutOfBounds(offset, 0, storage.total_len as usize));
        }

        let cursor = MultiPartReaderCursor::new(storage, offset);
        let mut decoder = ZlibDecoder::new(cursor);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| RpaError::Decompression(e.to_string()))?;

        // Parse with inert pickle VM
        let vm = InertPickleVm::new(&decompressed);
        let mut raw_index = vm.parse_rpa_index()?;

        // If RPA-3.0 with XOR key, deobfuscate offsets and lengths
        if let Some(xor_key) = key {
            let k = xor_key as u64;
            for entries in raw_index.values_mut() {
                for entry in entries {
                    entry.offset ^= k;
                    entry.length ^= k;
                }
            }
        }

        Ok(raw_index)
    }

    pub fn list_files(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    pub fn read_file(&self, filename: &str) -> Result<Vec<u8>, RpaError> {
        let entries = self
            .index
            .get(filename)
            .ok_or_else(|| RpaError::FileNotFound(filename.to_string()))?;
        let mut content = Vec::new();

        for entry in entries {
            let total_len = self.storage.len();
            if entry.offset + entry.length > total_len {
                return Err(RpaError::OutOfBounds(
                    entry.offset,
                    entry.length,
                    total_len as usize,
                ));
            }

            content.extend_from_slice(&entry.prefix);
            let slice_offset = entry.offset + entry.prefix.len() as u64;
            let slice_len = (entry.length as usize).saturating_sub(entry.prefix.len());
            self.storage
                .read_bytes(slice_offset, slice_len, &mut content)?;
        }

        Ok(content)
    }

    pub fn version(&self) -> &RpaVersion {
        &self.header.version
    }

    pub fn path(&self) -> &Path {
        &self.paths[0]
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn is_multipart(&self) -> bool {
        self.paths.len() > 1
    }

    pub fn part_count(&self) -> usize {
        self.paths.len()
    }

    pub fn is_fallback_required(&self) -> bool {
        crate::fallback::is_fallback_required(&self.header.version)
    }

    pub fn storage_len(&self) -> u64 {
        self.storage.len()
    }
}
