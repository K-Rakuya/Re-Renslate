//! lib.rs
//! renloc-rpa: Safe, high-performance binary RPA archive extractor and VFS for Ren'Py.

pub mod header;
pub mod pickle_idx;
pub mod v1_v3;
pub mod vfs;

pub use header::{parse_rpa_header, RpaHeaderInfo, RpaVersion};
pub use pickle_idx::{InertPickleVm, PickleError, PickleValue, RpaFileEntry};
pub use v1_v3::{RpaError, RpaReader};
pub use vfs::{VfsError, VfsLayer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpa3_header_parsing() {
        let header_bytes = b"RPA-3.0 0000000000001234 56789abc\n";
        let info = parse_rpa_header(header_bytes).expect("Failed to parse RPA-3.0 header");
        assert_eq!(info.version, RpaVersion::Rpa3);
        assert_eq!(info.index_offset, 0x1234);
        assert_eq!(info.obfuscation_key, Some(0x56789abc));
    }

    #[test]
    fn test_custom_8byte_header_recovery() {
        let custom_bytes = b"MYARC3.0 00000000000000ff 0000beef\n";
        let info = parse_rpa_header(custom_bytes).expect("Failed to parse custom 8-byte header");
        match info.version {
            RpaVersion::Custom8Byte(sig) => assert_eq!(sig, "MYARC3.0"),
            _ => panic!("Expected Custom8Byte variant"),
        }
        assert_eq!(info.index_offset, 0xff);
        assert_eq!(info.obfuscation_key, Some(0xbeef));
    }

    #[test]
    fn test_inert_pickle_rejects_malicious_opcodes() {
        // Construct a malicious pickle stream containing GLOBAL 'c' and REDUCE 'R' (e.g. os.system('calc'))
        let malicious_pickle = b"\x80\x02cos\nsystem\n(X\x04\x00\x00\x00calcqtR.";
        let vm = InertPickleVm::new(malicious_pickle);
        let res = vm.parse_rpa_index();
        assert!(res.is_err(), "Inert pickle VM MUST reject dangerous opcodes!");
        match res.unwrap_err() {
            PickleError::DangerousOpcode(opcode, _, _) => {
                assert_eq!(opcode, b'c', "Should stop at GLOBAL opcode 'c'");
            }
            other => panic!("Expected DangerousOpcode error, got {:?}", other),
        }
    }
}
