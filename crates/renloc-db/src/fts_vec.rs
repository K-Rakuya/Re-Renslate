//! fts_vec.rs
//! High-performance Keyset Pagination and Full-Text Search (FTS5) queries.
//! Handles 300,000+ translation units without memory exhaustion in the GUI.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use crate::readonly_pool::{DbError, ReadOnlyPool};

pub const DEFAULT_PAGE_LIMIT: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationUnitRow {
    pub id: i64,
    pub unit_identifier: String,
    pub source_text: String,
    pub target_text: Option<String>,
    pub speaker: Option<String>,
    pub status: String,
    pub row_order: i64,
}

fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TranslationUnitRow> {
    Ok(TranslationUnitRow {
        id: row.get(0)?,
        unit_identifier: row.get(1)?,
        source_text: row.get(2)?,
        target_text: row.get(3)?,
        speaker: row.get(4)?,
        status: row.get(5)?,
        row_order: row.get(6)?,
    })
}

pub struct QueryEngine;

impl QueryEngine {
    /// Keyset pagination: retrieves up to `limit` rows starting after `cursor_row_order`.
    /// Defaults to 200 rows if `limit == 0`.
    pub fn fetch_page_keyset(
        conn: &Connection,
        cursor_row_order: i64,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        let effective_limit = if limit == 0 { DEFAULT_PAGE_LIMIT } else { limit };
        let mut stmt = conn.prepare(
            "SELECT id, unit_identifier, source_text, target_text, speaker, status, row_order
             FROM translation_units
             WHERE row_order > ?1
             ORDER BY row_order ASC
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![cursor_row_order, effective_limit as i64], map_row)?;

        let mut results = Vec::with_capacity(effective_limit);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Keyset pagination backward (optional reverse cursor):
    pub fn fetch_page_keyset_before(
        conn: &Connection,
        cursor_row_order: i64,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        let effective_limit = if limit == 0 { DEFAULT_PAGE_LIMIT } else { limit };
        let mut stmt = conn.prepare(
            "SELECT id, unit_identifier, source_text, target_text, speaker, status, row_order
             FROM translation_units
             WHERE row_order < ?1
             ORDER BY row_order DESC
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![cursor_row_order, effective_limit as i64], map_row)?;

        let mut results = Vec::with_capacity(effective_limit);
        for row in rows {
            results.push(row?);
        }
        results.reverse();
        Ok(results)
    }

    /// Count total translation units in the database.
    pub fn count_total_units(conn: &Connection) -> Result<i64, DbError> {
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM translation_units")?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        Ok(count)
    }

    /// Sanitize user input for FTS5 queries to prevent syntax errors on raw punctuation.
    pub fn sanitize_fts_query(query: &str) -> String {
        let cleaned: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() || c.is_whitespace() || c == '_' { c } else { ' ' })
            .collect();
        let tokens: Vec<&str> = cleaned.split_whitespace().collect();
        if tokens.is_empty() {
            return String::new();
        }
        // Quote each token with wildcard matching for prefix search
        tokens.iter().map(|t| format!("\"{}\"*", t)).collect::<Vec<_>>().join(" AND ")
    }

    /// Full-Text Search using FTS5 virtual table.
    pub fn search_fts(
        conn: &Connection,
        query: &str,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        let effective_limit = if limit == 0 { DEFAULT_PAGE_LIMIT } else { limit };
        let sanitized = Self::sanitize_fts_query(trimmed);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = conn.prepare(
            "SELECT u.id, u.unit_identifier, u.source_text, u.target_text, u.speaker, u.status, u.row_order
             FROM units_fts f
             JOIN translation_units u ON f.rowid = u.id
             WHERE units_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
        )?;

        if let Ok(rows) = stmt.query_map(params![sanitized, effective_limit as i64], map_row) {
            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            if !results.is_empty() {
                return Ok(results);
            }
        }

        // Fallback to simple phrase match
        let simple_query = format!("\"{}\"", trimmed.replace('"', "\"\""));
        let mut fallback_stmt = conn.prepare(
            "SELECT u.id, u.unit_identifier, u.source_text, u.target_text, u.speaker, u.status, u.row_order
             FROM units_fts f
             JOIN translation_units u ON f.rowid = u.id
             WHERE units_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
        )?;
        let rows = fallback_stmt.query_map(params![simple_query, effective_limit as i64], map_row)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Asynchronous Full-Text Search via spawn_blocking to prevent blocking the async runtime.
    pub async fn search_fts_async(
        pool: ReadOnlyPool,
        query: String,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        tokio::task::spawn_blocking(move || {
            let conn = pool.get_connection()?;
            Self::search_fts(&conn, &query, limit)
        })
        .await
        .map_err(|e| DbError::AsyncJoin(e.to_string()))?
    }

    /// Asynchronous Keyset Pagination via spawn_blocking.
    pub async fn fetch_page_keyset_async(
        pool: ReadOnlyPool,
        cursor_row_order: i64,
        limit: usize,
    ) -> Result<Vec<TranslationUnitRow>, DbError> {
        tokio::task::spawn_blocking(move || {
            let conn = pool.get_connection()?;
            Self::fetch_page_keyset(&conn, cursor_row_order, limit)
        })
        .await
        .map_err(|e| DbError::AsyncJoin(e.to_string()))?
    }

    /// Asynchronous count of total units.
    pub async fn count_total_units_async(
        pool: ReadOnlyPool,
    ) -> Result<i64, DbError> {
        tokio::task::spawn_blocking(move || {
            let conn = pool.get_connection()?;
            Self::count_total_units(&conn)
        })
        .await
        .map_err(|e| DbError::AsyncJoin(e.to_string()))?
    }
}
