//! cmap_audit.rs
//! Font cmap table auditor for the 134 Vietnamese NFC Unicode characters.
//! Detects missing glyphs and automatically generates Ren'Py FontGroup fallback configuration.

use thiserror::Error;
use ttf_parser::Face;

#[derive(Debug, Error)]
pub enum FontAuditError {
    #[error("Failed to parse font file: {0}")]
    FontParseError(String),
    #[error("No valid cmap table found in font")]
    MissingCmap,
}

/// The exact set of 134 Vietnamese NFC characters requiring font glyph coverage beyond basic ASCII.
/// Comprises 12 vowels with all diacritics and tones plus đ/Đ.
pub const VIETNAMESE_NFC_CHARS: &str = "\
àÀảẢãÃáÁạẠ\
ăĂằẰẳẲẵẴắẮặẶ\
âÂầẦẩẨẫẪấẤậẬ\
èÈẻẺẽẼéÉẹẸ\
êÊềỀểỂễỄếẾệỆ\
ìÌỉỈĩĨíÍịỊ\
òÒỏỎõÕóÓọỌ\
ôÔồỒổỔỗỖốỐộỘ\
ơƠờỜởỞỡỠớỚợỢ\
ùÙủỦũŨúÚụỤ\
ưƯừỪửỬữỮứỨựỰ\
ỳỲỷỶỹỸýÝỵỴ\
đĐ";

#[derive(Debug, Clone)]
pub struct FontAuditResult {
    pub font_name: String,
    pub total_required: usize,
    pub supported_count: usize,
    pub missing_chars: Vec<char>,
    pub coverage_ratio: f32,
    pub is_fully_compatible: bool,
}

pub struct CmapAuditor;

impl CmapAuditor {
    /// Audits a font's raw TTF/OTF bytes against the 134 Vietnamese NFC characters.
    pub fn audit_font_bytes(font_name: &str, font_data: &[u8]) -> Result<FontAuditResult, FontAuditError> {
        let face = Face::parse(font_data, 0).map_err(|e| FontAuditError::FontParseError(e.to_string()))?;

        let mut missing = Vec::new();
        let mut supported = 0;

        let required_chars: Vec<char> = VIETNAMESE_NFC_CHARS.chars().collect();
        let total_required = required_chars.len();

        for ch in &required_chars {
            if let Some(glyph_id) = face.glyph_index(*ch) {
                if glyph_id.0 != 0 {
                    supported += 1;
                    continue;
                }
            }
            missing.push(*ch);
        }

        let coverage_ratio = supported as f32 / total_required as f32;
        let is_fully_compatible = missing.is_empty();

        Ok(FontAuditResult {
            font_name: font_name.to_string(),
            total_required,
            supported_count: supported,
            missing_chars: missing,
            coverage_ratio,
            is_fully_compatible,
        })
    }

    /// Generates Ren'Py FontGroup DSL configuration code to seamlessly fallback missing Vietnamese characters.
    pub fn generate_renpy_fontgroup_dsl(
        group_name: &str,
        primary_font_path: &str,
        fallback_font_path: &str,
        audit: &FontAuditResult,
    ) -> String {
        let mut dsl = String::new();
        dsl.push_str("init python:\n");
        dsl.push_str(&format!("    renpy.register_font_group(\"{}\",\n", group_name));
        dsl.push_str(&format!("        FontGroup().add(\"{}\", 0x0020, 0x007e)\n", primary_font_path));

        if !audit.is_fully_compatible {
            dsl.push_str(&format!("        .add(\"{}\", 0x00c0, 0x1ef9) # Vietnamese Vowels & Tone Marks\n", fallback_font_path));
        }

        dsl.push_str(&format!("        .add(\"{}\") # Default fallback to primary\n", primary_font_path));
        dsl.push_str("    )\n");
        dsl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_missing_characters() {
        // Test auditing with empty / invalid font
        let result = CmapAuditor::audit_font_bytes("invalid.ttf", b"not a font");
        assert!(result.is_err());
    }

    #[test]
    fn test_vietnamese_fontgroup_dsl_generation() {
        let audit = FontAuditResult {
            font_name: "custom.ttf".to_string(),
            total_required: 134,
            supported_count: 50,
            missing_chars: vec!['à', 'á'],
            coverage_ratio: 50.0 / 134.0,
            is_fully_compatible: false,
        };

        let dsl = CmapAuditor::generate_renpy_fontgroup_dsl(
            "custom_viet",
            "fonts/custom.ttf",
            "fonts/fallback.ttf",
            &audit,
        );

        assert!(dsl.contains("renpy.register_font_group(\"custom_viet\""));
        assert!(dsl.contains(".add(\"fonts/fallback.ttf\", 0x00c0, 0x1ef9)"));
        assert!(dsl.contains(".add(\"fonts/custom.ttf\")"));
    }

    #[test]
    fn test_system_dejavu_font_audit() {
        if std::path::Path::new("/usr/share/fonts/TTF/DejaVuSans.ttf").exists() {
            let font_bytes = std::fs::read("/usr/share/fonts/TTF/DejaVuSans.ttf").expect("read font");
            let result = CmapAuditor::audit_font_bytes("DejaVuSans.ttf", &font_bytes).expect("audit");
            assert_eq!(result.total_required, 134);
            assert_eq!(result.supported_count, 134);
            assert!(result.is_fully_compatible);
            assert_eq!(result.missing_chars.len(), 0);
        }
    }
}
