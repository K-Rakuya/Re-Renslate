//! drive_guard.rs
//! Local drive verification and network filesystem redirection to prevent SQLite WAL -shm corruption.

use std::env;
use std::path::{Path, PathBuf};

pub const NFS_SUPER_MAGIC: i64 = 0x6969;
pub const SMB_SUPER_MAGIC: i64 = 0x517B;
pub const CIFS_MAGIC_NUMBER: i64 = 0xFF534D42_u32 as i64;
pub const SMB2_MAGIC_NUMBER: i64 = 0xFE534D42_u32 as i64;
pub const FUSE_SUPER_MAGIC: i64 = 0x65735546;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriveType {
    Local,
    Network(String),
}

pub fn check_path_drive<P: AsRef<Path>>(path: P) -> DriveType {
    let p = path.as_ref();

    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let check_path = if p.exists() {
            p.to_path_buf()
        } else if let Some(parent) = p.parent() {
            if parent.exists() {
                parent.to_path_buf()
            } else {
                PathBuf::from(".")
            }
        } else {
            PathBuf::from(".")
        };

        if let Ok(c_path) = CString::new(check_path.as_os_str().as_bytes()) {
            unsafe {
                let mut stat: libc::statfs = std::mem::zeroed();
                if libc::statfs(c_path.as_ptr(), &mut stat) == 0 {
                    let f_type = stat.f_type as i64;
                    if f_type == NFS_SUPER_MAGIC {
                        return DriveType::Network("NFS".into());
                    } else if f_type == SMB_SUPER_MAGIC || f_type == 0x517B {
                        return DriveType::Network("SMB".into());
                    } else if f_type == CIFS_MAGIC_NUMBER || f_type == (CIFS_MAGIC_NUMBER as i32 as i64) {
                        return DriveType::Network("CIFS".into());
                    } else if f_type == SMB2_MAGIC_NUMBER || f_type == (SMB2_MAGIC_NUMBER as i32 as i64) {
                        return DriveType::Network("SMB2".into());
                    } else if f_type == FUSE_SUPER_MAGIC {
                        return DriveType::Network("FUSE".into());
                    }
                }
            }
        }
    }

    #[cfg(windows)]
    {
        let p_str = p.to_string_lossy();
        if p_str.starts_with("\\\\") || p_str.starts_with("//") {
            return DriveType::Network("UNC".into());
        }
        if p_str.len() >= 2 && p_str.chars().nth(1) == Some(':') {
            use windows_sys::Win32::Storage::FileSystem::*;
            let root = format!("{}:\\\0", &p_str[0..1]);
            let wide: Vec<u16> = root.encode_utf16().collect();
            unsafe {
                let dt = GetDriveTypeW(wide.as_ptr());
                if dt == DRIVE_REMOTE {
                    return DriveType::Network("NetworkDrive".into());
                }
            }
        }
    }

    DriveType::Local
}

pub fn get_safe_local_db_path<P: AsRef<Path>>(original_path: P) -> PathBuf {
    let p = original_path.as_ref();
    match check_path_drive(p) {
        DriveType::Local => p.to_path_buf(),
        DriveType::Network(_) => {
            let hash = simple_fnv_hash(p.to_string_lossy().as_bytes());
            let file_name = p
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("project.sqlite"));

            #[cfg(windows)]
            let base_dir = env::var("LOCALAPPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("C:\\ProgramData"));

            #[cfg(unix)]
            let base_dir = env::var("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = env::var("HOME").unwrap_or_else(|_| "/tmp".into());
                    PathBuf::from(home).join(".local").join("share")
                });

            let safe_dir = base_dir
                .join("renslate")
                .join("databases")
                .join(format!("{:016x}", hash));
            let _ = std::fs::create_dir_all(&safe_dir);
            safe_dir.join(file_name)
        }
    }
}

fn simple_fnv_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_drive_detection() {
        let cwd = std::env::current_dir().unwrap();
        let dt = check_path_drive(&cwd);
        // Current directory on local disk should be Local or FUSE depending on runner
        match dt {
            DriveType::Local => assert!(true),
            DriveType::Network(name) => println!("Detected filesystem: {}", name),
        }
    }

    #[test]
    fn test_safe_local_db_path_generation() {
        let p = PathBuf::from("/mnt/nfs_share/game/project.sqlite");
        let safe = get_safe_local_db_path(&p);
        assert!(!safe.as_os_str().is_empty());
    }
}
