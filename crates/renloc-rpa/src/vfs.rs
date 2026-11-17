//! vfs.rs
//! Virtual File System (VFS) Layer combining loose game files, multi-part chunked
//! RPA archives, and live engine In-Game VFS Fallback for advanced/encrypted variants.
//!
//! Follows Ren'Py priority: loose files override archives; archives load in lexicographical order.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::header::RpaVersion;
use crate::v1_v3::{split_multipart_filename, RpaError, RpaReader};

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
    fallback_dirs: Vec<PathBuf>,
    archives: Vec<RpaReader>,
    file_map: HashMap<String, usize>, // maps normalized filename -> index in `archives`
}

impl VfsLayer {
    pub fn new<P: AsRef<Path>>(game_dir: P) -> Self {
        Self {
            base_dir: game_dir.as_ref().to_path_buf(),
            fallback_dirs: Vec::new(),
            archives: Vec::new(),
            file_map: HashMap::new(),
        }
    }

    /// Discovers and mounts all .rpa archives and multi-part chunked archives
    /// in the game directory in lexicographical order.
    pub fn mount_game_directory(&mut self) -> Result<(), VfsError> {
        if !self.base_dir.exists() {
            return Ok(());
        }

        let mut groups: HashMap<String, Vec<(usize, PathBuf)>> = HashMap::new();
        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let file_name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };

            if let Some((base, part_num)) = parse_archive_filename_and_part(file_name) {
                groups.entry(base).or_default().push((part_num, path));
            }
        }

        let mut sorted_groups: Vec<(String, Vec<PathBuf>)> = Vec::new();
        for (base_name, mut parts) in groups {
            let has_numbered_parts = parts.iter().any(|(num, _)| *num > 0);
            if has_numbered_parts {
                parts.retain(|(num, _)| *num > 0);
            }
            parts.sort_by_key(|(num, _)| *num);
            let part_paths: Vec<PathBuf> = parts.into_iter().map(|(_, p)| p).collect();
            sorted_groups.push((base_name, part_paths));
        }

        // Sort groups lexicographically by base name
        sorted_groups.sort_by(|a, b| a.0.cmp(&b.0));

        for (_base_name, part_paths) in sorted_groups {
            let res = if part_paths.len() == 1 {
                RpaReader::open(&part_paths[0])
            } else {
                RpaReader::open_multipart(&part_paths)
            };

            if let Ok(reader) = res {
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

    /// Mounts a specific archive directly, automatically discovering sibling parts if multi-part.
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

    /// Mounts an explicit sequence of multi-part archive chunks, stitching them into a single archive.
    pub fn mount_multipart_archive<P: AsRef<Path>>(&mut self, paths: &[P]) -> Result<(), VfsError> {
        let reader = RpaReader::open_multipart(paths)?;
        let arc_idx = self.archives.len();
        for file_name in reader.list_files() {
            let normalized = Self::normalize_path(&file_name);
            self.file_map.insert(normalized, arc_idx);
        }
        self.archives.push(reader);
        Ok(())
    }

    /// Mounts an extracted fallback directory containing files dumped by live engine fallback.
    pub fn mount_fallback_directory<P: AsRef<Path>>(&mut self, fallback_dir: P) -> Result<(), VfsError> {
        let dir = fallback_dir.as_ref();
        if dir.is_dir() {
            self.fallback_dirs.push(dir.to_path_buf());
        }
        Ok(())
    }

    /// Checks if any mounted archive requires live In-Game VFS Fallback (e.g., ZiX-12A/12B).
    pub fn requires_in_game_fallback(&self) -> bool {
        self.archives.iter().any(|arc| arc.is_fallback_required())
    }

    /// Returns list of archive paths and versions requiring live In-Game VFS Fallback.
    pub fn fallback_archives(&self) -> Vec<(&Path, &RpaVersion)> {
        self.archives
            .iter()
            .filter(|arc| arc.is_fallback_required())
            .map(|arc| (arc.path(), arc.version()))
            .collect()
    }

    /// Generates an In-Game VFS Fallback Python/RPY script string with `init -999 python:`.
    pub fn generate_in_game_fallback(&self) -> String {
        crate::fallback::generate_in_game_fallback_script(None)
    }

    /// Writes the In-Game VFS Fallback script to `target_game_dir/00_renloc_vfs_fallback.rpy`.
    pub fn write_in_game_fallback<P: AsRef<Path>>(&self, target_game_dir: P) -> Result<PathBuf, VfsError> {
        let path = crate::fallback::write_in_game_fallback_script(target_game_dir, None)?;
        Ok(path)
    }

    /// Reads a file from the VFS: loose files take precedence over archives.
    pub fn read_file(&self, virtual_path: &str) -> Result<Vec<u8>, VfsError> {
        let normalized = Self::normalize_path(virtual_path);

        // 1. Check if loose file exists in primary game directory
        let loose_path = self.base_dir.join(&normalized);
        if loose_path.is_file() {
            let bytes = fs::read(loose_path)?;
            return Ok(bytes);
        }

        // 2. Check fallback extraction directories (if any)
        for fb_dir in &self.fallback_dirs {
            let fb_path = fb_dir.join(&normalized);
            if fb_path.is_file() {
                let bytes = fs::read(fb_path)?;
                return Ok(bytes);
            }
        }

        // 3. Check mounted archives
        if let Some(&arc_idx) = self.file_map.get(&normalized) {
            let bytes = self.archives[arc_idx].read_file(virtual_path)?;
            return Ok(bytes);
        }

        Err(VfsError::NotFound(virtual_path.to_string()))
    }

    /// Lists all available files in the VFS (loose files, fallback files, and archive files).
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

        // Scan loose files in fallback_dirs
        for fb_dir in &self.fallback_dirs {
            if let Ok(entries) = walk_dir(fb_dir) {
                for path in entries {
                    if let Ok(rel) = path.strip_prefix(fb_dir) {
                        let rel_str = Self::normalize_path(&rel.to_string_lossy());
                        if !files.contains(&rel_str) {
                            files.push(rel_str);
                        }
                    }
                }
            }
        }

        files.sort();
        files
    }

    /// Returns list of currently mounted archives.
    pub fn archives(&self) -> &[RpaReader] {
        &self.archives
    }

    pub fn normalize_path(path: &str) -> String {
        path.replace('\\', "/").trim_start_matches('/').to_string()
    }
}

fn parse_archive_filename_and_part(file_name: &str) -> Option<(String, usize)> {
    let lower = file_name.to_lowercase();
    if lower.ends_with(".rpa") || lower.ends_with(".rpi") || lower.ends_with(".arc") {
        return Some((lower, 0));
    }

    if let Some((base, ext)) = split_multipart_filename(&lower) {
        if base.ends_with(".rpa") || base.ends_with(".rpi") || base.ends_with(".arc") {
            if let Some(digits) = ext.strip_prefix("part") {
                if let Ok(num) = digits.parse::<usize>() {
                    return Some((base.to_string(), num));
                }
            } else if let Ok(num) = ext.parse::<usize>() {
                return Some((base.to_string(), num));
            }
        }
    }

    None
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
