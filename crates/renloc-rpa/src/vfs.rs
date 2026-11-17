//! vfs.rs
//! Virtual File System (VFS) Layer combining loose game files and multiple RPA archives.
//! Follows Ren'Py priority: loose files override archives; archives load in lexicographical order.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::v1_v3::{RpaError, RpaReader};

#[derive(Debug, Error)]
pub enum VfsError {
    #[error("File not found in VFS: {0}")]
    NotFound(String),
    #[error("I/O error in VFS: {0}")]
    Io(#[from] std::io::Error),
    #[error("RPA error in VFS: {0}")]
    Rpa(#[from] RpaError),
}

pub struct VfsLayer {
    base_dir: PathBuf,
    archives: Vec<RpaReader>,
    file_map: HashMap<String, usize>, // maps normalized filename -> index in `archives`
}

impl VfsLayer {
    pub fn new<P: AsRef<Path>>(game_dir: P) -> Self {
        Self {
            base_dir: game_dir.as_ref().to_path_buf(),
            archives: Vec::new(),
            file_map: HashMap::new(),
        }
    }

    /// Discovers and mounts all .rpa archives in the game directory in lexicographical order.
    pub fn mount_game_directory(&mut self) -> Result<(), VfsError> {
        if !self.base_dir.exists() {
            return Ok(());
        }

        let mut rpa_files = Vec::new();
        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "rpa" || ext_str == "rpi" {
                        rpa_files.push(path);
                    }
                }
            }
        }

        // Sort lexicographically so archives mount in standard order (archive.rpa, scripts.rpa, etc.)
        rpa_files.sort();

        for path in rpa_files {
            if let Ok(reader) = RpaReader::open(&path) {
                let arc_idx = self.archives.len();
                for file_name in reader.list_files() {
                    let normalized = Self::normalize_path(&file_name);
                    self.file_map.insert(normalized, arc_idx);
                }
                self.archives.push(reader);
            }
        }

        Ok(())
    }

    /// Mounts a specific archive directly.
    pub fn mount_archive<P: AsRef<Path>>(&mut self, path: P) -> Result<(), VfsError> {
        let reader = RpaReader::open(path)?;
        let arc_idx = self.archives.len();
        for file_name in reader.list_files() {
            let normalized = Self::normalize_path(&file_name);
            self.file_map.insert(normalized, arc_idx);
        }
        self.archives.push(reader);
        Ok(())
    }

    /// Reads a file from the VFS: loose files take precedence over archives.
    pub fn read_file(&self, virtual_path: &str) -> Result<Vec<u8>, VfsError> {
        let normalized = Self::normalize_path(virtual_path);

        // 1. Check if loose file exists in game directory
        let loose_path = self.base_dir.join(&normalized);
        if loose_path.is_file() {
            let bytes = fs::read(loose_path)?;
            return Ok(bytes);
        }

        // 2. Check mounted archives
        if let Some(&arc_idx) = self.file_map.get(&normalized) {
            let bytes = self.archives[arc_idx].read_file(virtual_path)?;
            return Ok(bytes);
        }

        Err(VfsError::NotFound(virtual_path.to_string()))
    }

    /// Lists all available files in the VFS (both loose files and archive files).
    pub fn list_all_files(&self) -> Vec<String> {
        let mut files = Vec::new();
        for k in self.file_map.keys() {
            files.push(k.clone());
        }

        // Scan loose files in base_dir
        if let Ok(entries) = walk_dir(&self.base_dir) {
            for path in entries {
                if let Ok(rel) = path.strip_prefix(&self.base_dir) {
                    let rel_str = Self::normalize_path(&rel.to_string_lossy());
                    if !files.contains(&rel_str) {
                        files.push(rel_str);
                    }
                }
            }
        }

        files.sort();
        files
    }

    fn normalize_path(path: &str) -> String {
        path.replace('\\', "/").trim_start_matches('/').to_string()
    }
}

fn walk_dir(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();
    if !dir.is_dir() {
        return Ok(files);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(walk_dir(&path)?);
        } else {
            files.push(path);
        }
    }
    Ok(files)
}
