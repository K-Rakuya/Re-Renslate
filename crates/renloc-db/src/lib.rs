//! lib.rs
//! renloc-db: Read-only SQLite WAL connection pooling, FTS5 search, and keyset pagination.

pub mod fts_vec;
pub mod readonly_pool;

pub use fts_vec::{QueryEngine, TranslationUnitRow};
pub use readonly_pool::{DbError, ReadOnlyPool};

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::NamedTempFile;

    #[test]
    fn test_keyset_pagination() {
        let tmp_file = NamedTempFile::new().unwrap();
        let conn = Connection::open(tmp_file.path()).unwrap();

        // Create table and insert mock data
        conn.execute_batch(
            "CREATE TABLE translation_units (
                id INTEGER PRIMARY KEY,
                unit_identifier TEXT NOT NULL,
                source_text TEXT NOT NULL,
                target_text TEXT,
                speaker TEXT,
                status TEXT NOT NULL,
                row_order INTEGER NOT NULL
            );
            INSERT INTO translation_units VALUES (1, 'say_01', 'Hello', 'Xin chao', 'eileen', 'COMMITTED', 1);
            INSERT INTO translation_units VALUES (2, 'say_02', 'Welcome', 'Chao mung', 'eileen', 'COMMITTED', 2);
            INSERT INTO translation_units VALUES (3, 'say_03', 'Goodbye', 'Tam biet', 'eileen', 'COMMITTED', 3);
            "
        ).unwrap();

        let page = QueryEngine::fetch_page_keyset(&conn, 0, 2).unwrap();
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].unit_identifier, "say_01");
        assert_eq!(page[1].unit_identifier, "say_02");

        let page2 = QueryEngine::fetch_page_keyset(&conn, page.last().unwrap().row_order, 2).unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].unit_identifier, "say_03");
    }
}
