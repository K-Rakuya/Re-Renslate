//! lib.rs
//! renloc-db: Read-only SQLite WAL connection pooling, FTS5 search, and keyset pagination.

pub mod drive_guard;
pub mod fts_vec;
pub mod readonly_pool;

pub use drive_guard::{check_path_drive, get_safe_local_db_path, DriveType};
pub use fts_vec::{QueryEngine, TranslationUnitRow, DEFAULT_PAGE_LIMIT};
pub use readonly_pool::{DbError, ReadOnlyPool};

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use tempfile::NamedTempFile;

    fn setup_test_db(path: &std::path::Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE translation_units (
                id INTEGER PRIMARY KEY,
                unit_identifier TEXT NOT NULL,
                source_text TEXT NOT NULL,
                target_text TEXT,
                speaker TEXT,
                status TEXT NOT NULL,
                row_order INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE units_fts USING fts5(
                source_text,
                target_text,
                speaker,
                content='translation_units',
                content_rowid='id'
            );
            CREATE TRIGGER trg_ai_units AFTER INSERT ON translation_units BEGIN
                INSERT INTO units_fts(rowid, source_text, target_text, speaker)
                VALUES (new.id, new.source_text, new.target_text, new.speaker);
            END;
            INSERT INTO translation_units VALUES (1, 'say_01', 'Hello world', 'Xin chao the gioi', 'eileen', 'COMMITTED', 1);
            INSERT INTO translation_units VALUES (2, 'say_02', 'Welcome home', 'Chao mung ve nha', 'eileen', 'COMMITTED', 2);
            INSERT INTO translation_units VALUES (3, 'say_03', 'Goodbye friend', 'Tam biet ban', 'eileen', 'COMMITTED', 3);
            "
        ).unwrap();
    }

    #[test]
    fn test_keyset_pagination() {
        let tmp = NamedTempFile::new().unwrap();
        setup_test_db(tmp.path());

        let conn = Connection::open(tmp.path()).unwrap();
        let page = QueryEngine::fetch_page_keyset(&conn, 0, 2).unwrap();
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].unit_identifier, "say_01");
        assert_eq!(page[1].unit_identifier, "say_02");

        let page2 =
            QueryEngine::fetch_page_keyset(&conn, page.last().unwrap().row_order, 2).unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].unit_identifier, "say_03");

        let total = QueryEngine::count_total_units(&conn).unwrap();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_large_keyset_pagination_iteration() {
        let tmp = NamedTempFile::new().unwrap();
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE translation_units (
                id INTEGER PRIMARY KEY,
                unit_identifier TEXT NOT NULL,
                source_text TEXT NOT NULL,
                target_text TEXT,
                speaker TEXT,
                status TEXT NOT NULL,
                row_order INTEGER NOT NULL
            );",
        )
        .unwrap();

        // Insert 2500 rows in a transaction
        let mut appender = conn.prepare(
            "INSERT INTO translation_units (id, unit_identifier, source_text, target_text, speaker, status, row_order)
             VALUES (?1, ?2, ?3, ?4, 'narrator', 'NEW', ?1)"
        ).unwrap();
        conn.execute("BEGIN TRANSACTION", []).unwrap();
        for i in 1..=2500 {
            appender
                .execute(rusqlite::params![
                    i,
                    format!("id_{:05}", i),
                    format!("Source dialogue {}", i),
                    format!("Target dialogue {}", i),
                ])
                .unwrap();
        }
        conn.execute("COMMIT", []).unwrap();
        drop(appender);

        let total = QueryEngine::count_total_units(&conn).unwrap();
        assert_eq!(total, 2500);

        // Page through with keyset pagination
        let mut cursor = 0;
        let mut fetched_count = 0;
        let page_size = 200;
        loop {
            let page = QueryEngine::fetch_page_keyset(&conn, cursor, page_size).unwrap();
            if page.is_empty() {
                break;
            }
            assert!(page.len() <= page_size);
            for item in &page {
                assert!(item.row_order > cursor);
                fetched_count += 1;
            }
            cursor = page.last().unwrap().row_order;
        }
        assert_eq!(fetched_count, 2500);
    }

    #[test]
    fn test_fts5_search() {
        let tmp = NamedTempFile::new().unwrap();
        setup_test_db(tmp.path());

        let conn = Connection::open(tmp.path()).unwrap();
        let results = QueryEngine::search_fts(&conn, "welcome", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unit_identifier, "say_02");

        // Test prefix / sanitized search
        let results_prefix = QueryEngine::search_fts(&conn, "Good*", 10).unwrap();
        assert_eq!(results_prefix.len(), 1);
        assert_eq!(results_prefix[0].unit_identifier, "say_03");
    }

    #[tokio::test]
    async fn test_async_queries() {
        let tmp = NamedTempFile::new().unwrap();
        setup_test_db(tmp.path());

        let pool = ReadOnlyPool::new(tmp.path(), 4).unwrap();
        let page = QueryEngine::fetch_page_keyset_async(pool.clone(), 0, 2)
            .await
            .unwrap();
        assert_eq!(page.len(), 2);

        let search = QueryEngine::search_fts_async(pool.clone(), "Hello".into(), 10)
            .await
            .unwrap();
        assert_eq!(search.len(), 1);
        assert_eq!(search[0].unit_identifier, "say_01");

        let count = QueryEngine::count_total_units_async(pool).await.unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_readonly_pool_query_only_enforced() {
        let tmp = NamedTempFile::new().unwrap();
        setup_test_db(tmp.path());

        let pool = ReadOnlyPool::new(tmp.path(), 2).unwrap();
        let conn = pool.get_connection().unwrap();

        // Attempting to write on a read-only pool MUST fail
        let res = conn.execute(
            "INSERT INTO translation_units VALUES (4, 'say_04', 'Fail', NULL, NULL, 'NEW', 4)",
            [],
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_concurrent_read_write_wal_no_busy() {
        let tmp = NamedTempFile::new().unwrap();
        setup_test_db(tmp.path());

        let path = tmp.path().to_path_buf();
        let pool = ReadOnlyPool::new(&path, 8).unwrap();
        let running = Arc::new(AtomicBool::new(true));

        // Background writer: continuously writes new rows in WAL mode
        let writer_running = Arc::clone(&running);
        let writer_path = path.clone();
        let writer_handle = thread::spawn(move || {
            let mut writer_conn = Connection::open(&writer_path).unwrap();
            writer_conn
                .execute_batch(
                    "PRAGMA journal_mode = WAL;
                 PRAGMA busy_timeout = 10000;",
                )
                .unwrap();

            let mut row_id = 100;
            while writer_running.load(Ordering::Relaxed) {
                row_id += 1;
                let tx = writer_conn
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                    .unwrap();
                tx.execute(
                    "INSERT INTO translation_units (id, unit_identifier, source_text, target_text, speaker, status, row_order)
                     VALUES (?1, ?2, 'test text', 'ban dich', 'eileen', 'COMMITTED', ?1)",
                    rusqlite::params![row_id, format!("say_{}", row_id)],
                ).unwrap();
                tx.commit().unwrap();
                thread::sleep(Duration::from_millis(2));
            }
        });

        // 4 Reader threads: continuously read from ReadOnlyPool
        let mut reader_handles = Vec::new();
        for _ in 0..4 {
            let pool_clone = pool.clone();
            let reader_running = Arc::clone(&running);
            reader_handles.push(thread::spawn(move || {
                let mut read_count = 0;
                while reader_running.load(Ordering::Relaxed) {
                    let conn = pool_clone.get_connection().unwrap();
                    let page = QueryEngine::fetch_page_keyset(&conn, 0, 50).unwrap();
                    assert!(!page.is_empty());
                    read_count += 1;
                    thread::sleep(Duration::from_millis(3));
                }
                read_count
            }));
        }

        thread::sleep(Duration::from_millis(200));
        running.store(false, Ordering::Relaxed);

        writer_handle.join().unwrap();
        for h in reader_handles {
            let count = h.join().unwrap();
            assert!(
                count > 0,
                "Reader thread should have performed multiple reads without error"
            );
        }
    }
}
