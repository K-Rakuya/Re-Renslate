//! lib.rs
//! renloc-shaper: OpenType text shaping, HarfBuzz layout, and 134-glyph Vietnamese NFC cmap audit.

pub mod bidi_hb;
pub mod cmap_audit;

pub use bidi_hb::{ShapeError, ShapedTextMetrics, TextShaper};
pub use cmap_audit::{CmapAuditor, FontAuditError, FontAuditResult, VIETNAMESE_NFC_CHARS};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vietnamese_char_count() {
        let count = VIETNAMESE_NFC_CHARS.chars().count();
        // 134 Vietnamese NFC characters
        assert_eq!(
            count, 134,
            "Vietnamese NFC set must contain exactly 134 unique characters"
        );
    }
}
