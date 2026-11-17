//! lib.rs
//! renloc-rpa: Safe, high-performance binary RPA archive extractor and VFS for Ren'Py.

pub mod fallback;
pub mod header;
pub mod pickle_idx;
pub mod v1_v3;
pub mod vfs;

pub use fallback::{
    generate_in_game_fallback_script, is_fallback_required, write_in_game_fallback_script,
    FallbackConfig,
};
pub use header::{parse_rpa_header, RpaHeaderInfo, RpaVersion};
pub use pickle_idx::{InertPickleVm, PickleError, PickleValue, RpaFileEntry};
pub use v1_v3::{
    discover_sibling_parts, split_multipart_filename, MultiPartReaderCursor, MultiPartStorage,
    RpaError, RpaReader,
};
pub use vfs::{VfsError, VfsLayer};

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

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
        assert!(
            res.is_err(),
            "Inert pickle VM MUST reject dangerous opcodes!"
        );
        match res.unwrap_err() {
            PickleError::DangerousOpcode(opcode, _, _) => {
                assert_eq!(opcode, b'c', "Should stop at GLOBAL opcode 'c'");
            }
            other => panic!("Expected DangerousOpcode error, got {:?}", other),
        }
    }

    #[test]
    fn test_inert_pickle_protocol0_opcodes() {
        // Construct a protocol 0 dictionary with ASCII INT, STRING, TUPLE, LIST, DICT, SETITEM
        let proto0_stream = b"(dp0\nS'file.txt'\np1\n(lp2\n(I123\nI456\nS'prefix'\ntp3\nas.";
        let vm = InertPickleVm::new(proto0_stream);
        let res = vm.parse_rpa_index().expect("Should parse protocol 0 index");
        assert!(res.contains_key("file.txt"));
        let entries = &res["file.txt"];
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].offset, 123);
        assert_eq!(entries[0].length, 456);
        assert_eq!(entries[0].prefix, b"prefix");
    }

    type TestIndexEntry<'a> = (&'a str, Vec<(u64, u64, &'a [u8])>);

    /// Helper to construct a minimal zlib-compressed pickle index for RPA-2.0.
    fn make_rpa2_index_bytes(index_map: &[TestIndexEntry<'_>]) -> Vec<u8> {
        // Build pickle stream protocol 2
        let mut raw = Vec::new();
        raw.extend_from_slice(b"\x80\x02}"); // PROTO 2, EMPTY_DICT

        for (name, entries) in index_map {
            // Push key: SHORT_BINUNICODE or BINUNICODE
            raw.push(b'X');
            let name_bytes = name.as_bytes();
            raw.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            raw.extend_from_slice(name_bytes);

            // Push value: list of tuples
            raw.push(b'('); // MARK
            for (offset, length, prefix) in entries {
                raw.push(b'('); // MARK
                                // offset as int (BININT or INT)
                raw.push(b'J');
                raw.extend_from_slice(&(*offset as i32).to_le_bytes());
                // length as int
                raw.push(b'J');
                raw.extend_from_slice(&(*length as i32).to_le_bytes());
                // prefix as bytes
                raw.push(b'C');
                raw.push(prefix.len() as u8);
                raw.extend_from_slice(prefix);
                raw.push(b't'); // TUPLE
            }
            raw.push(b'l'); // LIST
            raw.push(b's'); // SETITEM
        }
        raw.push(b'.'); // STOP

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn test_multipart_chunked_archive_stitching() {
        let dir = tempdir().unwrap();
        let part1_path = dir.path().join("game_archive.rpa.001");
        let part2_path = dir.path().join("game_archive.rpa.002");

        let header_str = format!("RPA-2.0 {:016x}\n", 0); // Placeholder header, len = 25
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Data for file A: spans across part 1 and part 2!
        // Part 1 will hold: header (25 bytes) + file A part 1 (100 bytes of 'A') = 125 bytes.
        // Part 2 will hold: file A part 2 (150 bytes of 'B') + file B chunks + index.
        let file_a_part1 = vec![b'A'; 100];
        let file_a_part2 = vec![b'B'; 150];
        let mut expected_file_a = Vec::new();
        expected_file_a.extend_from_slice(&file_a_part1);
        expected_file_a.extend_from_slice(&file_a_part2);

        // Data for file B: composed of two non-contiguous slices in the archive (multi-chunk)
        let file_b_chunk1 = b"Hello from chunk 1! ";
        let file_b_chunk2 = b"And this is chunk 2!";
        let mut expected_file_b = Vec::new();
        expected_file_b.extend_from_slice(file_b_chunk1);
        expected_file_b.extend_from_slice(file_b_chunk2);

        // Offsets in the unified stitched archive:
        let file_a_offset = header_len; // 25
        let file_a_len = (file_a_part1.len() + file_a_part2.len()) as u64; // 250

        let part1_total_len = header_len + file_a_part1.len() as u64; // 125
        let file_b_chunk1_offset = part1_total_len + file_a_part2.len() as u64; // 125 + 150 = 275
        let file_b_chunk1_len = file_b_chunk1.len() as u64;

        let padding = vec![0xcc; 50];
        let file_b_chunk2_offset = file_b_chunk1_offset + file_b_chunk1_len + padding.len() as u64;
        let file_b_chunk2_len = file_b_chunk2.len() as u64;

        let index_offset = file_b_chunk2_offset + file_b_chunk2_len;

        // Prepare index data
        let index_entries = vec![
            (
                "story/chapter1.txt",
                vec![(file_a_offset, file_a_len, &b""[..])],
            ),
            (
                "scripts/multi_slice.rpy",
                vec![
                    (file_b_chunk1_offset, file_b_chunk1_len, &b""[..]),
                    (file_b_chunk2_offset, file_b_chunk2_len, &b""[..]),
                ],
            ),
        ];
        let index_bytes = make_rpa2_index_bytes(&index_entries);

        // True header with index_offset pointing into part 2
        let true_header = format!("RPA-2.0 {:016x}\n", index_offset);
        assert_eq!(true_header.len(), header_len as usize);

        // Assemble part 1
        let mut part1_data = Vec::new();
        part1_data.extend_from_slice(true_header.as_bytes());
        part1_data.extend_from_slice(&file_a_part1);
        assert_eq!(part1_data.len() as u64, part1_total_len);
        fs::write(&part1_path, &part1_data).unwrap();

        // Assemble part 2
        let mut part2_data = Vec::new();
        part2_data.extend_from_slice(&file_a_part2);
        part2_data.extend_from_slice(file_b_chunk1);
        part2_data.extend_from_slice(&padding);
        part2_data.extend_from_slice(file_b_chunk2);
        part2_data.extend_from_slice(&index_bytes);
        fs::write(&part2_path, &part2_data).unwrap();

        // 1. Open with auto-discovery on part 1
        let reader =
            RpaReader::open(&part1_path).expect("Should auto-discover sibling part 2 and stitch");
        assert!(reader.is_multipart());
        assert_eq!(reader.part_count(), 2);
        assert_eq!(reader.paths().len(), 2);

        // 2. Read file A (spans across .001 and .002 boundaries)
        let content_a = reader
            .read_file("story/chapter1.txt")
            .expect("Should read stitched cross-boundary file");
        assert_eq!(content_a, expected_file_a);

        // 3. Read file B (composed of multiple non-contiguous slices)
        let content_b = reader
            .read_file("scripts/multi_slice.rpy")
            .expect("Should read multi-slice file");
        assert_eq!(content_b, expected_file_b);

        // 4. Test VFS layer mounting
        let mut vfs = VfsLayer::new(dir.path());
        vfs.mount_game_directory()
            .expect("VfsLayer should mount chunked archives");

        let vfs_files = vfs.list_all_files();
        assert!(vfs_files.contains(&"story/chapter1.txt".to_string()));
        assert!(vfs_files.contains(&"scripts/multi_slice.rpy".to_string()));

        let vfs_a = vfs.read_file("story/chapter1.txt").unwrap();
        assert_eq!(vfs_a, expected_file_a);

        // 5. Test loose file override: loose file takes precedence over multi-part archive
        let loose_story = dir.path().join("story").join("chapter1.txt");
        fs::create_dir_all(loose_story.parent().unwrap()).unwrap();
        let loose_content = b"Overridden loose file content!";
        fs::write(&loose_story, loose_content).unwrap();

        let vfs_a_overridden = vfs.read_file("story/chapter1.txt").unwrap();
        assert_eq!(vfs_a_overridden, loose_content);
    }

    #[test]
    fn test_zix_in_game_fallback_mechanism() {
        let dir = tempdir().unwrap();
        let zix_arc_path = dir.path().join("encrypted.rpa");

        // Write a ZiX-12A header
        fs::write(&zix_arc_path, b"ZiX-12A 0000000000000000 00000000\n").unwrap();

        let reader = RpaReader::open(&zix_arc_path).expect("Should parse ZiX-12A header");
        assert_eq!(reader.version(), &RpaVersion::ZiX12A);
        assert!(reader.is_fallback_required());

        let mut vfs = VfsLayer::new(dir.path());
        vfs.mount_game_directory().unwrap();
        assert!(vfs.requires_in_game_fallback());

        let fallback_arcs = vfs.fallback_archives();
        assert_eq!(fallback_arcs.len(), 1);
        assert_eq!(fallback_arcs[0].1, &RpaVersion::ZiX12A);

        // Verify fallback script generation
        let script = vfs.generate_in_game_fallback();
        assert!(script.contains("init -999 python:"));
        assert!(script.contains("renpy.loader.load("));
        assert!(script.contains("class RenslateVfsFallback"));

        // Verify fallback writing to disk
        let written_path = vfs.write_in_game_fallback(dir.path()).unwrap();
        assert!(written_path.exists());
        let file_content = fs::read_to_string(&written_path).unwrap();
        assert_eq!(file_content, script);
    }
}
