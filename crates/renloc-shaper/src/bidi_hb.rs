//! bidi_hb.rs
//! Text shaping with rustybuzz (HarfBuzz port) and BiDi analysis.
//! Computes exact pixel bounding boxes to detect dialogue box overflow.

use rustybuzz::{Face, UnicodeBuffer};
use thiserror::Error;
use unicode_bidi::BidiInfo;

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

        // Bidi analysis
        let bidi_info = BidiInfo::new(text, None);
        let mut total_advance = 0.0f32;
        let mut glyph_count = 0;

        for para in &bidi_info.paragraphs {
            let line = &text[para.range.clone()];
            let mut buffer = UnicodeBuffer::new();
            buffer.push_str(line);
            buffer.guess_segment_properties();

            let glyph_buffer = rustybuzz::shape(&face, &[], buffer);
            let positions = glyph_buffer.glyph_positions();
            glyph_count += positions.len();

            for pos in positions {
                total_advance += (pos.x_advance as f32) * scale_factor;
            }
        }

        let ascender = (face.ascender() as f32) * scale_factor;
        let descender = (face.descender() as f32).abs() * scale_factor;
        let line_height = ascender + descender;

        let mut is_overflowing = false;
        if let Some(max_w) = max_box_width_px {
            if total_advance > max_w {
                is_overflowing = true;
            }
        }
        if let Some(max_h) = max_box_height_px {
            if line_height > max_h {
                is_overflowing = true;
            }
        }

        Ok(ShapedTextMetrics {
            total_advance_pixels: total_advance,
            max_height_pixels: line_height,
            glyph_count,
            is_overflowing,
        })
    }
}
