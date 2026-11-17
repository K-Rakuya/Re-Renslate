//! bidi_hb.rs
//! Text shaping with rustybuzz (HarfBuzz port) and BiDi analysis.
//! Computes exact pixel bounding boxes to detect dialogue box overflow.

use rustybuzz::{Face, UnicodeBuffer};
use thiserror::Error;
use unicode_bidi::BidiInfo;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Error)]
pub enum ShapeError {
    #[error("Failed to load font face: {0}")]
    FaceError(String),
}

#[derive(Debug, Clone)]
pub struct ShapedTextMetrics {
    pub total_advance_pixels: f32,
    pub max_height_pixels: f32,
    pub glyph_count: usize,
    pub is_overflowing: bool,
}

pub struct TextShaper;

impl TextShaper {
    /// Shapes a text string using HarfBuzz algorithms and checks if it exceeds max_width or max_height.
    pub fn shape_and_measure(
        font_data: &[u8],
        font_size_pt: f32,
        text: &str,
        max_box_width_px: Option<f32>,
        max_box_height_px: Option<f32>,
    ) -> Result<ShapedTextMetrics, ShapeError> {
        let face = Face::from_slice(font_data, 0).ok_or_else(|| ShapeError::FaceError("Invalid font data".into()))?;
        let upem = face.units_per_em() as f32;
        let scale_factor = font_size_pt / upem;

        // Normalize text to Unicode NFC form for correct Vietnamese glyph matching
        let text_nfc: String = text.nfc().collect();

        // Bidi analysis
        let bidi_info = BidiInfo::new(&text_nfc, None);
        let mut total_advance = 0.0f32;
        let mut max_line_advance = 0.0f32;
        let mut glyph_count = 0;

        for para in &bidi_info.paragraphs {
            let line = &text_nfc[para.range.clone()];
            let mut buffer = UnicodeBuffer::new();
            buffer.push_str(line);
            buffer.guess_segment_properties();

            let glyph_buffer = rustybuzz::shape(&face, &[], buffer);
            let positions = glyph_buffer.glyph_positions();
            glyph_count += positions.len();

            let mut line_advance = 0.0f32;
            for pos in positions {
                line_advance += (pos.x_advance as f32) * scale_factor;
            }
            if line_advance > max_line_advance {
                max_line_advance = line_advance;
            }
            total_advance += line_advance;
        }

        let ascender = (face.ascender() as f32) * scale_factor;
        let descender = (face.descender() as f32).abs() * scale_factor;
        let line_height = ascender + descender;
        let num_lines = bidi_info.paragraphs.len().max(1) as f32;
        let total_rendered_height = line_height * num_lines;

        let mut is_overflowing = false;
        if let Some(max_w) = max_box_width_px {
            let eval_w = if bidi_info.paragraphs.len() > 1 {
                max_line_advance
            } else {
                total_advance
            };
            if eval_w > max_w {
                is_overflowing = true;
            }
        }
        if let Some(max_h) = max_box_height_px {
            if total_rendered_height > max_h {
                is_overflowing = true;
            }
        }

        Ok(ShapedTextMetrics {
            total_advance_pixels: total_advance,
            max_height_pixels: total_rendered_height,
            glyph_count,
            is_overflowing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_invalid_font() {
        let res = TextShaper::shape_and_measure(b"invalid data", 14.0, "Xin chào", None, None);
        assert!(res.is_err());
    }

    #[test]
    fn test_shape_and_overflow_detection() {
        if std::path::Path::new("/usr/share/fonts/TTF/DejaVuSans.ttf").exists() {
            let font_bytes = std::fs::read("/usr/share/fonts/TTF/DejaVuSans.ttf").expect("read font");
            
            // Short text, large box -> No overflow
            let m1 = TextShaper::shape_and_measure(&font_bytes, 16.0, "Xin chào", Some(500.0), Some(100.0)).expect("shape");
            assert!(!m1.is_overflowing);
            assert!(m1.total_advance_pixels > 0.0);

            // Long text, tiny box -> Overflow detected
            let m2 = TextShaper::shape_and_measure(&font_bytes, 16.0, "Đây là một đoạn hội thoại rất dài để kiểm tra tràn khung thoại trong Ren'Py.", Some(50.0), Some(100.0)).expect("shape");
            assert!(m2.is_overflowing);
        }
    }
}
