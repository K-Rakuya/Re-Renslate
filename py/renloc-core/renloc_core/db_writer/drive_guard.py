"""
drive_guard.py
Local drive validation and path safety guard for Renslate SQLite storage.

Prevents SQLite WAL shared memory (-shm) corruption on network filesystems
(NFS, CIFS/SMB, FUSE cloud sync folders) by detecting unsupported filesystems
and automatically relocating the database working copy to a local fixed drive:
- Windows: %LOCALAPPDATA%/renslate/databases/<project_hash>/<filename>
- Linux/macOS: $XDG_DATA_HOME/renslate/databases/<project_hash>/<filename>
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Windows GetDriveTypeW constants
DRIVE_UNKNOWN = 0
DRIVE_NO_ROOT_DIR = 1
DRIVE_REMOVABLE = 2
DRIVE_FIXED = 3
DRIVE_REMOTE = 4
DRIVE_CDROM = 5
DRIVE_RAMDISK = 6

# Linux filesystem magic numbers (statfs f_type)
LINUX_MAGIC_NETWORK_FS: dict[int, str] = {
    0x6969: "nfs",
    0x517B: "smb",
    0xFF534D42: "cifs",
    0xFE534D42: "smb2",
    0x65735546: "fuse",
    0x73757245: "coda",
    0x5346414F: "afs",
    0x564C: "ncp",
    0x00C36400: "ceph",
    0x7655821: "glusterfs",
    0x7461636F: "ocfs2",
}

NETWORK_FS_TYPE_NAMES: set[str] = {
    "nfs",
    "nfs4",
    "cifs",
    "smb",
    "smb2",
    "smb3",
    "smbfs",
    "afpfs",
    "sshfs",
    "davfs",
    "curlftpfs",
    "glusterfs",
    "ceph",
}


@dataclass(frozen=True)
class SafePathResult:
    """Result of path validation and relocation."""
    safe_path: Path
    original_path: Path
    is_relocated: bool
    fs_type: str
    reason: str


class _StatfsLinux(ctypes.Structure):
    _fields_ = [
        ("f_type", ctypes.c_long),
        ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulonglong),
        ("f_bfree", ctypes.c_ulonglong),
        ("f_bavail", ctypes.c_ulonglong),
        ("f_files", ctypes.c_ulonglong),
        ("f_ffree", ctypes.c_ulonglong),
        ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long),
        ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long),
        ("f_spare", ctypes.c_long * 4),
    ]


def _get_drive_type_win32(drive_root: str) -> int:
    """Invokes Windows GetDriveTypeW via ctypes.windll.kernel32."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return DRIVE_UNKNOWN
    return windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive_root))


def _check_windows_drive(path: Path) -> Tuple[bool, str]:
    """Check drive type on Windows via GetDriveTypeW and UNC path detection."""
    str_path = str(path)
    # Check UNC network path (\\server\share or //server/share)
    if str_path.startswith(("\\\\", "//")):
        return True, "UNC_NETWORK_SHARE"

    drive_root = os.path.splitdrive(str_path)[0]
    if not drive_root:
        drive_root = "C:\\"
    elif not drive_root.endswith("\\"):
        drive_root = drive_root + "\\"

    try:
        drive_type = _get_drive_type_win32(drive_root)
        if drive_type == DRIVE_REMOTE:
            return True, "DRIVE_REMOTE"
        elif drive_type in (DRIVE_FIXED, DRIVE_RAMDISK):
            return False, f"DRIVE_TYPE_{drive_type}"
        elif drive_type == DRIVE_REMOVABLE:
            # Removable drives (USB flash) may not guarantee atomic POSIX/Win locks for WAL
            return True, "DRIVE_REMOVABLE"
        else:
            return True, f"DRIVE_TYPE_{drive_type}"
    except Exception as exc:
        logger.warning("Failed to invoke GetDriveTypeW: %s", exc)
        return False, "UNKNOWN_WIN32"


def _check_linux_mounts(path: Path) -> Tuple[Optional[bool], str]:
    """Find mount point in /proc/mounts and check for network/fuse filesystem."""
    try:
        resolved = str(path.resolve())
        best_match_mount = ""
        best_match_fstype = "unknown"

        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mount_point = parts[1]
                    fstype = parts[2].lower()
                    if resolved.startswith(mount_point) and len(mount_point) > len(best_match_mount):
                        best_match_mount = mount_point
                        best_match_fstype = fstype

        if best_match_fstype != "unknown":
            # Match against known network fs and fuse prefixes
            if (
                best_match_fstype in NETWORK_FS_TYPE_NAMES
                or best_match_fstype.startswith("fuse.")
                or best_match_fstype.startswith("nfs")
                or best_match_fstype.startswith("cifs")
                or best_match_fstype.startswith("smb")
            ):
                return True, best_match_fstype
            return False, best_match_fstype
    except Exception as exc:
        logger.debug("Could not parse /proc/mounts: %s", exc)

    return None, "unknown"


def _check_linux_statfs(path: Path) -> Tuple[bool, str]:
    """Check Linux filesystem type magic number using libc statfs."""
    try:
        libc = ctypes.CDLL(None)
        buf = _StatfsLinux()
        path_bytes = os.fsencode(str(path))
        if libc.statfs(path_bytes, ctypes.byref(buf)) == 0:
            raw_magic = buf.f_type
            # Handle unsigned representation on 32-bit / 64-bit systems
            magic = raw_magic & 0xFFFFFFFF
            if magic in LINUX_MAGIC_NETWORK_FS:
                return True, LINUX_MAGIC_NETWORK_FS[magic]
            return False, f"0x{magic:x}"
    except Exception as exc:
        logger.debug("Could not call libc statfs: %s", exc)
    return False, "unknown"


def _check_darwin_statfs(path: Path) -> Tuple[bool, str]:
    """Check macOS filesystem type via statfs f_fstypename."""
    try:
        # macOS struct statfs has f_fstypename at offset 68 (16 chars)
        class _StatfsDarwin(ctypes.Structure):
            _fields_ = [
                ("f_bsize", ctypes.c_uint32),
                ("f_iosize", ctypes.c_int32),
                ("f_blocks", ctypes.c_uint64),
                ("f_bfree", ctypes.c_uint64),
                ("f_bavail", ctypes.c_uint64),
                ("f_files", ctypes.c_uint64),
                ("f_ffree", ctypes.c_uint64),
                ("f_fsid", ctypes.c_int32 * 2),
                ("f_owner", ctypes.c_uint32),
                ("f_type", ctypes.c_uint32),
                ("f_flags", ctypes.c_uint32),
                ("f_fssubtype", ctypes.c_uint32),
                ("f_fstypename", ctypes.c_char * 16),
                ("f_mntonname", ctypes.c_char * 1024),
                ("f_mntfromname", ctypes.c_char * 1024),
                ("f_reserved", ctypes.c_uint32 * 8),
            ]

        libc = ctypes.CDLL(None)
        buf = _StatfsDarwin()
        path_bytes = os.fsencode(str(path))
        if libc.statfs(path_bytes, ctypes.byref(buf)) == 0:
            fstype = buf.f_fstypename.decode("utf-8", errors="replace").strip("\x00").lower()
            if (
                fstype in NETWORK_FS_TYPE_NAMES
                or "fuse" in fstype
                or fstype.startswith(("smb", "nfs", "afp"))
            ):
                return True, fstype
            return False, fstype
    except Exception as exc:
        logger.debug("Could not call macOS statfs: %s", exc)
    return False, "unknown"


def is_network_filesystem(path: Path | str) -> Tuple[bool, str]:
    """
    Determine if a path resides on a network/unsupported filesystem.
    Returns (is_network, fs_type_name).
    """
    target = Path(path).resolve()
    # Check parent directory if target file does not exist yet
    probe_dir = target if target.exists() and target.is_dir() else target.parent
    if not probe_dir.exists():
        # Fall back to first existing ancestor
        while not probe_dir.exists() and probe_dir.parent != probe_dir:
            probe_dir = probe_dir.parent

    if sys.platform == "win32":
        return _check_windows_drive(probe_dir)
    elif sys.platform == "darwin":
        return _check_darwin_statfs(probe_dir)
    else:
        # Linux / Unix
        mount_result, mount_fs = _check_linux_mounts(probe_dir)
        if mount_result is not None:
            return mount_result, mount_fs

        statfs_network, statfs_fs = _check_linux_statfs(probe_dir)
        if statfs_network:
            return True, statfs_fs
        return False, mount_fs if mount_fs != "unknown" else statfs_fs


def get_local_fallback_directory() -> Path:
    """
    Returns the root directory for relocated local working databases:
    - Windows: %LOCALAPPDATA%/renslate/databases
    - POSIX: $XDG_DATA_HOME/renslate/databases (or ~/.local/share/renslate/databases)
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_DATA_HOME")
        if not base:
            base = os.path.expanduser("~/.local/share")

    return Path(base) / "renslate" / "databases"


def resolve_safe_db_path(
    db_path: Path | str,
    fallback_base_dir: Optional[Path | str] = None,
) -> SafePathResult:
    """
    Validates a database path. If located on a network or unsafe filesystem,
    relocates it to a local fixed application directory under a deterministic project hash.
    """
    original = Path(db_path).resolve()
    is_net, fs_type = is_network_filesystem(original)

    if not is_net:
        return SafePathResult(
            safe_path=original,
            original_path=original,
            is_relocated=False,
            fs_type=fs_type,
            reason="Local fixed filesystem",
        )

    # Compute deterministic project hash from parent folder path
    parent_str = str(original.parent).encode("utf-8")
    project_hash = hashlib.sha256(parent_str).hexdigest()[:16]

    if fallback_base_dir is not None:
        target_root = Path(fallback_base_dir)
    else:
        target_root = get_local_fallback_directory()

    relocated_dir = target_root / project_hash
    relocated_dir.mkdir(parents=True, exist_ok=True)
    safe_target = relocated_dir / original.name

    logger.warning(
        "Database path '%s' is on a network/unsafe filesystem (%s). "
        "Relocating to safe local path: '%s' to prevent WAL -shm corruption.",
        original,
        fs_type,
        safe_target,
    )

    return SafePathResult(
        safe_path=safe_target,
        original_path=original,
        is_relocated=True,
        fs_type=fs_type,
        reason=f"Relocated from network/unsafe filesystem ({fs_type})",
    )
