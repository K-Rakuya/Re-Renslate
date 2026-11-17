//! lib.rs
//! renloc_native: PyO3 ABI3 Python extension module exporting high-performance
//! Rust capabilities (RPA archive reader, VFS, OpenType HarfBuzz shaper, cmap auditor).

#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use renloc_rpa::{RpaReader, VfsLayer};
use renloc_shaper::{CmapAuditor, TextShaper};

#[pyclass(name = "RpaArchive")]
pub struct PyRpaArchive {
    inner: RpaReader,
}

#[pymethods]
impl PyRpaArchive {
    #[new]
    pub fn new(path: String) -> PyResult<Self> {
        let inner = RpaReader::open(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn list_files(&self) -> Vec<String> {
        self.inner.list_files()
    }

    pub fn read_file<'py>(&self, py: Python<'py>, filename: String) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.read_file(&filename).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    pub fn is_multipart(&self) -> bool {
        self.inner.is_multipart()
    }

    pub fn part_count(&self) -> usize {
        self.inner.part_count()
    }

    pub fn is_fallback_required(&self) -> bool {
        self.inner.is_fallback_required()
    }
}

#[pyclass(name = "VfsLayer")]
pub struct PyVfsLayer {
    inner: VfsLayer,
}

#[pymethods]
impl PyVfsLayer {
    #[new]
    pub fn new(game_dir: String) -> Self {
        Self {
            inner: VfsLayer::new(game_dir),
        }
    }

    pub fn mount_game_directory(&mut self) -> PyResult<()> {
        self.inner.mount_game_directory().map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn mount_archive(&mut self, path: String) -> PyResult<()> {
        self.inner.mount_archive(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn mount_multipart_archive(&mut self, paths: Vec<String>) -> PyResult<()> {
        self.inner.mount_multipart_archive(&paths).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn mount_fallback_directory(&mut self, fallback_dir: String) -> PyResult<()> {
        self.inner.mount_fallback_directory(fallback_dir).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn requires_in_game_fallback(&self) -> bool {
        self.inner.requires_in_game_fallback()
    }

    pub fn generate_in_game_fallback(&self) -> String {
        self.inner.generate_in_game_fallback()
    }

    pub fn write_in_game_fallback(&self, target_game_dir: String) -> PyResult<String> {
        let path = self.inner.write_in_game_fallback(target_game_dir).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(path.to_string_lossy().to_string())
    }

    pub fn read_file<'py>(&self, py: Python<'py>, virtual_path: String) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.read_file(&virtual_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    pub fn list_all_files(&self) -> Vec<String> {
        self.inner.list_all_files()
    }
}

#[pyfunction]
pub fn audit_font_bytes<'py>(py: Python<'py>, font_name: String, font_data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let result = CmapAuditor::audit_font_bytes(&font_name, font_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("font_name", result.font_name)?;
    dict.set_item("total_required", result.total_required)?;
    dict.set_item("supported_count", result.supported_count)?;
    let missing_str: String = result.missing_chars.into_iter().collect();
    dict.set_item("missing_chars", missing_str)?;
    dict.set_item("coverage_ratio", result.coverage_ratio)?;
    dict.set_item("is_fully_compatible", result.is_fully_compatible)?;

    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (font_data, font_size_pt, text, max_box_width_px=None, max_box_height_px=None))]
pub fn shape_and_measure<'py>(
    py: Python<'py>,
    font_data: &[u8],
    font_size_pt: f32,
    text: String,
    max_box_width_px: Option<f32>,
    max_box_height_px: Option<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics = TextShaper::shape_and_measure(font_data, font_size_pt, &text, max_box_width_px, max_box_height_px)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("total_advance_pixels", metrics.total_advance_pixels)?;
    dict.set_item("max_height_pixels", metrics.max_height_pixels)?;
    dict.set_item("glyph_count", metrics.glyph_count)?;
    dict.set_item("is_overflowing", metrics.is_overflowing)?;

    Ok(dict)
}

#[pymodule]
fn renloc_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRpaArchive>()?;
    m.add_class::<PyVfsLayer>()?;
    m.add_function(wrap_pyfunction!(audit_font_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(shape_and_measure, m)?)?;
    Ok(())
}
