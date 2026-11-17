//! fts_vec.rs
//! High-performance Keyset Pagination and Full-Text Search (FTS5) queries.
//! Handles 300,000+ translation units without memory exhaustion in the GUI.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use crate::readonly_pool::DbError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationUnitRow {
    pub id: i64,
    pub unit_identifier: String,
    pub source_text: String,
    pub target_text: Option<String>,
    pub speaker: Option<String>,
    pub status: String,
    pub row_order: i64,
}

pub struct QueryEngine;

impl QueryEngine {
    /// Keyset pagination: retrieves up to `limit` rows starting after `cursor_row_order`.
    pub fn fetch_page_keyset(
        conn: &Connection,
        cursor_row_order: i64,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        let mut stmt = conn.prepare(
            "SELECT id, unit_identifier, source_text, target_text, speaker, status, row_order
             FROM translation_units
             WHERE row_order > ?1
             ORDER BY row_order ASC
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![cursor_row_order, limit as i64], |row| {
            Ok(TranslationUnitRow {
                id: row.get(0)?,
                unit_identifier: row.get(1)?,
                source_text: row.get(2)?,
                target_text: row.get(3)?,
                speaker: row.get(4)?,
                status: row.get(5)?,
                row_order: row.get(6)?,
            })
        })?;

        let mut results = Vec::with_capacity(limit);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Full-Text Search using FTS5 virtual table.
    pub fn search_fts(
        conn: &Connection,
        query: &str,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        let mut stmt = conn.prepare(
            "SELECT u.id, u.unit_identifier, u.source_text, u.target_text, u.speaker, u.status, u.row_order
             FROM units_fts f
             JOIN translation_units u ON f.rowid = u.id
             WHERE units_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok(TranslationUnitRow {
                id: row.get(0)?,
                unit_identifier: row.get(1)?,
                source_text: row.get(2)?,
                target_text: row.get(3)?,
                speaker: row.get(4)?,
                status: row.get(5)?,
                row_order: row.get(6)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
