//! readonly_pool.rs
//! High-throughput read-only connection pooling with rusqlite for GUI queries.
//! Configured with PRAGMA query_only=ON and PRAGMA busy_timeout=5000 in WAL mode.

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::OpenFlags;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("Database pool error: {0}")]
    Pool(#[from] r2d2::Error),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Database path not found: {0}")]
    NotFound(String),
}

pub struct ReadOnlyPool {
    pool: Pool<SqliteConnectionManager>,
    db_path: PathBuf,
}

impl ReadOnlyPool {
    pub fn new<P: AsRef<Path>>(db_path: P, max_connections: u32) -> Result<Self, DbError> {
        let path = db_path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(DbError::NotFound(path.display().to_string()));
        }

        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let manager = SqliteConnectionManager::file(&path)
            .with_flags(flags)
            .with_init(|conn| {
                conn.execute_batch(
                    "PRAGMA query_only = ON;
                     PRAGMA busy_timeout = 5000;
                     PRAGMA mmap_size = 268435456;" // 256MB mmap
                )
            });

        let pool = Pool::builder()
            .max_size(max_connections)
            .build(manager)?;

        Ok(Self { pool, db_path: path })
    }

    pub fn get_connection(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>, DbError> {
        Ok(self.pool.get()?)
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }
}
