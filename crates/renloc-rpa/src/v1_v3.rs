//! v1_v3.rs
//! Core decompression and slicing logic for RPA-1.0, RPA-2.0, RPA-3.0, RPA-3.2, RPA-4.0, and ALT-1.0.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use flate2::read::ZlibDecoder;
use memmap2::Mmap;
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

pub struct RpaReader {
    path: PathBuf,
    mmap: Mmap,
    header: RpaHeaderInfo,
    index: HashMap<String, Vec<RpaFileEntry>>,
}

impl RpaReader {
    pub fn open<P: AsRef<Path>>(archive_path: P) -> Result<Self, RpaError> {
        let path = archive_path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Check if there is a separate .rpi file (RPA-1.0)
        let rpi_path = path.with_extension("rpi");
        let (header, index) = if rpi_path.exists() {
            let rpi_bytes = std::fs::read(&rpi_path)?;
            let header = parse_rpa_header(&rpi_bytes)?;
            let index = Self::decompress_and_parse_index(&rpi_bytes, 0, None)?;
            (header, index)
        } else {
            let header = parse_rpa_header(&mmap)?;
            let index = match header.version {
                RpaVersion::Rpa1 => {
                    Self::decompress_and_parse_index(&mmap, 0, None)?
                }
                RpaVersion::Rpa2 => {
                    Self::decompress_and_parse_index(&mmap, header.index_offset, None)?
                }
                RpaVersion::Rpa3 | RpaVersion::Custom8Byte(_) | RpaVersion::Rpa32 | RpaVersion::Rpa40 | RpaVersion::Alt10 => {
                    Self::decompress_and_parse_index(&mmap, header.index_offset, header.obfuscation_key)?
                }
                RpaVersion::ZiX12A | RpaVersion::ZiX12B => {
                    // ZiX variant will be routed through unrpa/in-game fallback
                    HashMap::new()
                }
            };
            (header, index)
        };

        Ok(Self {
            path,
            mmap,
            header,
            index,
        })
    }

    fn decompress_and_parse_index(
        bytes: &[u8],
        offset: u64,
        key: Option<u32>,
    ) -> Result<HashMap<String, Vec<RpaFileEntry>>, RpaError> {
        if offset as usize >= bytes.len() {
            return Err(RpaError::OutOfBounds(offset, 0, bytes.len()));
        }

        let slice = &bytes[offset as usize..];
        let mut decoder = ZlibDecoder::new(slice);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|e| RpaError::Decompression(e.to_string()))?;

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
        let entries = self.index.get(filename).ok_or_else(|| RpaError::FileNotFound(filename.to_string()))?;
        let mut content = Vec::new();

        for entry in entries {
            let total_len = self.mmap.len();
            if entry.offset as usize + entry.length as usize > total_len {
                return Err(RpaError::OutOfBounds(entry.offset, entry.length, total_len));
            }

            // In Ren'Py RPA format, if prefix is specified, the file starts with prefix bytes
            content.extend_from_slice(&entry.prefix);
            let slice_offset = (entry.offset as usize) + entry.prefix.len();
            let slice_len = (entry.length as usize).saturating_sub(entry.prefix.len());
            content.extend_from_slice(&self.mmap[slice_offset..slice_offset + slice_len]);
        }

        Ok(content)
    }

    pub fn version(&self) -> &RpaVersion {
        &self.header.version
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
