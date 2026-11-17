// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod job_guard;
pub mod rpc_client;

use std::sync::Mutex;
use tauri::State;
use renloc_db::{QueryEngine, ReadOnlyPool, TranslationUnitRow};

struct AppState {
    db_pool: Mutex<Option<ReadOnlyPool>>,
}

#[tauri::command]
fn open_project_db(path: String, state: State<'_, AppState>) -> Result<String, String> {
    let pool = ReadOnlyPool::new(&path, 8).map_err(|e| e.to_string())?;
    let mut lock = state.db_pool.lock().unwrap();
    *lock = Some(pool);
    Ok("Database opened successfully".into())
}

#[tauri::command]
fn fetch_translation_page(
    cursor_row_order: i64,
    limit: usize,
    state: State<'_, AppState>,
) -> Result<Vec<TranslationUnitRow>, String> {
    let lock = state.db_pool.lock().unwrap();
    let pool = lock.as_ref().ok_or("No database opened")?;
    let conn = pool.get_connection().map_err(|e| e.to_string())?;
    QueryEngine::fetch_page_keyset(&conn, cursor_row_order, limit).map_err(|e| e.to_string())
}

#[tauri::command]
fn search_units(
    query: String,
    limit: usize,
    state: State<'_, AppState>,
) -> Result<Vec<TranslationUnitRow>, String> {
    let lock = state.db_pool.lock().unwrap();
    let pool = lock.as_ref().ok_or("No database opened")?;
    let conn = pool.get_connection().map_err(|e| e.to_string())?;
    QueryEngine::search_fts(&conn, &query, limit).map_err(|e| e.to_string())
}

fn main() {
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .manage(AppState {
            db_pool: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            open_project_db,
            fetch_translation_page,
            search_units
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
