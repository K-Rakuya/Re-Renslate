//! header.rs
//! Archive signature detection and header parser for all Ren'Py archive formats:
//! RPA-1.0, RPA-2.0, RPA-3.0, RPA-3.2, RPA-4.0, ALT-1.0, ZiX-12A/12B, and custom 8-byte headers.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HeaderError {
    #[error("File is too small to contain a valid RPA header")]
    TooSmall,
    #[error("Unrecognized or corrupted RPA header signature: {0:?}")]
    UnrecognizedSignature(Vec<u8>),
    #[error("Failed to parse hexadecimal offset/key in header: {0}")]
    HexParseError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpaVersion {
    Rpa1,
    Rpa2,
    Rpa3,
    Rpa32,
    Rpa40,
    Alt10,
    ZiX12A,
    ZiX12B,
    Custom8Byte(String),
}

#[derive(Debug, Clone)]
pub struct RpaHeaderInfo {
    pub version: RpaVersion,
    pub index_offset: u64,
    pub obfuscation_key: Option<u32>,
    pub extra_keys: Vec<u64>,
}

/// Detects and parses the header of an RPA archive from the initial bytes.
pub fn parse_rpa_header(bytes: &[u8]) -> Result<RpaHeaderInfo, HeaderError> {
    if bytes.len() < 8 {
        return Err(HeaderError::TooSmall);
    }

    // 1. RPA-1.0 Detection (ZLIB compressed index file signature 0x78 0x9c or paired .rpi)
    if bytes.starts_with(&[0x78, 0x9c]) || bytes.starts_with(&[0x78, 0x01]) || bytes.starts_with(&[0x78, 0xda]) {
        return Ok(RpaHeaderInfo {
            version: RpaVersion::Rpa1,
            index_offset: 0,
            obfuscation_key: None,
            extra_keys: Vec::new(),
        });
    }

    // Convert leading line to string for ASCII header parsing
    let line_end = bytes.iter().position(|&b| b == b'\n').unwrap_or(bytes.len().min(128));
    let line = &bytes[..line_end];
    let line_str = String::from_utf8_lossy(line);
    let trimmed = line_str.trim_end_matches(['\r', '\n']);

    // 2. RPA-2.0 ("RPA-2.0 0123456789abcdef [extra]")
    if let Some(rest) = trimmed.strip_prefix("RPA-2.0 ") {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.is_empty() {
            return Err(HeaderError::HexParseError("Missing index offset in RPA-2.0 header".to_string()));
        }
        let index_offset = u64::from_str_radix(parts[0], 16)
            .map_err(|e| HeaderError::HexParseError(format!("Invalid RPA-2.0 offset: {}", e)))?;
        return Ok(RpaHeaderInfo {
            version: RpaVersion::Rpa2,
            index_offset,
            obfuscation_key: None,
            extra_keys: Vec::new(),
        });
    }

    // 3. RPA-3.0 ("RPA-3.0 0123456789abcdef 01234567")
    if let Some(rest) = trimmed.strip_prefix("RPA-3.0 ") {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(HeaderError::HexParseError("Missing offset or key in RPA-3.0 header".to_string()));
        }
        let index_offset = u64::from_str_radix(parts[0], 16)
            .map_err(|e| HeaderError::HexParseError(format!("Invalid RPA-3.0 offset: {}", e)))?;
        let obfuscation_key = u32::from_str_radix(parts[1], 16)
            .map_err(|e| HeaderError::HexParseError(format!("Invalid RPA-3.0 key: {}", e)))?;

        let mut extra_keys = Vec::new();
        for &extra in &parts[2..] {
            if let Ok(val) = u64::from_str_radix(extra, 16) {
                extra_keys.push(val);
            }
        }

        return Ok(RpaHeaderInfo {
            version: RpaVersion::Rpa3,
            index_offset,
            obfuscation_key: Some(obfuscation_key),
            extra_keys,
        });
    }

    // 4. RPA-3.2
    if let Some(rest) = trimmed.strip_prefix("RPA-3.2 ") {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        let index_offset = parts.first().and_then(|s| u64::from_str_radix(s, 16).ok()).unwrap_or(0);
        let key = parts.get(1).and_then(|s| u32::from_str_radix(s, 16).ok());
        return Ok(RpaHeaderInfo {
            version: RpaVersion::Rpa32,
            index_offset,
            obfuscation_key: key,
            extra_keys: Vec::new(),
        });
    }

    // 5. RPA-4.0
    if let Some(rest) = trimmed.strip_prefix("RPA-4.0 ") {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        let index_offset = parts.first().and_then(|s| u64::from_str_radix(s, 16).ok()).unwrap_or(0);
        let key = parts.get(1).and_then(|s| u32::from_str_radix(s, 16).ok());
        return Ok(RpaHeaderInfo {
            version: RpaVersion::Rpa40,
            index_offset,
            obfuscation_key: key,
            extra_keys: Vec::new(),
        });
    }

    // 6. ALT-1.0
    if let Some(rest) = trimmed.strip_prefix("ALT-1.0 ") {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        let index_offset = parts.first().and_then(|s| u64::from_str_radix(s, 16).ok()).unwrap_or(0);
        let key = parts.get(1).and_then(|s| u32::from_str_radix(s, 16).ok());
        return Ok(RpaHeaderInfo {
            version: RpaVersion::Alt10,
            index_offset,
            obfuscation_key: key,
            extra_keys: Vec::new(),
        });
    }

    // 7. ZiX-12A and ZiX-12B
    if trimmed.starts_with("ZiX-12A ") || trimmed.starts_with("ZiX-12A") {
        return Ok(RpaHeaderInfo {
            version: RpaVersion::ZiX12A,
            index_offset: 0,
            obfuscation_key: None,
            extra_keys: Vec::new(),
        });
    }
    if trimmed.starts_with("ZiX-12B ") || trimmed.starts_with("ZiX-12B") {
        return Ok(RpaHeaderInfo {
            version: RpaVersion::ZiX12B,
            index_offset: 0,
            obfuscation_key: None,
            extra_keys: Vec::new(),
        });
    }

    // 8. Custom 8-byte header recovery (8-byte custom signature followed by offset and key)
    if trimmed.len() >= 18 {
        let custom_sig = trimmed[..8].to_string();
        let rest = &trimmed[8..];
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(offset), Ok(key)) = (
                u64::from_str_radix(parts[0], 16),
                u32::from_str_radix(parts[1], 16),
            ) {
                return Ok(RpaHeaderInfo {
                    version: RpaVersion::Custom8Byte(custom_sig),
                    index_offset: offset,
                    obfuscation_key: Some(key),
                    extra_keys: Vec::new(),
                });
            }
        }
    }

    Err(HeaderError::UnrecognizedSignature(bytes[..bytes.len().min(16)].to_vec()))
}
