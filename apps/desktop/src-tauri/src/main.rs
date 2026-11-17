// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod job_guard;
pub mod rpc_client;
pub mod sidecar;

use std::sync::Mutex;
use tauri::State;
use renloc_db::{QueryEngine, ReadOnlyPool, TranslationUnitRow};
use crate::rpc_client::RpcClient;
use crate::sidecar::SidecarSupervisor;

pub struct AppState {
    pub db_pool: Mutex<Option<ReadOnlyPool>>,
    pub rpc_client: Mutex<Option<RpcClient>>,
    pub sidecar: Mutex<Option<SidecarSupervisor>>,
}

#[tauri::command]
fn open_project_db(path: String, state: State<'_, AppState>) -> Result<String, String> {
    let pool = ReadOnlyPool::new(&path, 8).map_err(|e| e.to_string())?;
    let mut lock = state.db_pool.lock().unwrap();
    *lock = Some(pool);
    Ok("Database opened successfully".into())
}

#[tauri::command]
fn close_project_db(state: State<'_, AppState>) -> Result<(), String> {
    let mut lock = state.db_pool.lock().unwrap();
    *lock = None;
    Ok(())
}

#[tauri::command]
async fn get_total_count(state: State<'_, AppState>) -> Result<i64, String> {
    let pool = {
        let lock = state.db_pool.lock().unwrap();
        lock.clone().ok_or_else(|| "No database opened".to_string())?
    };
    QueryEngine::count_total_units_async(pool)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn fetch_translation_page(
    cursor_row_order: i64,
    limit: usize,
    state: State<'_, AppState>,
) -> Result<Vec<TranslationUnitRow>, String> {
    let pool = {
        let lock = state.db_pool.lock().unwrap();
        lock.clone().ok_or_else(|| "No database opened".to_string())?
    };
    QueryEngine::fetch_page_keyset_async(pool, cursor_row_order, limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn search_units(
    query: String,
    limit: usize,
    state: State<'_, AppState>,
) -> Result<Vec<TranslationUnitRow>, String> {
    let pool = {
        let lock = state.db_pool.lock().unwrap();
        lock.clone().ok_or_else(|| "No database opened".to_string())?
    };
    QueryEngine::search_fts_async(pool, query, limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn call_sidecar_rpc(
    method: String,
    payload: serde_json::Value,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let client = {
        let lock = state.rpc_client.lock().unwrap();
        lock.clone()
    };
    let client = client.ok_or_else(|| "Sidecar RPC client not connected".to_string())?;
    client.call(&method, payload).await.map_err(|e| e.to_string())
}

#[tauri::command]
fn get_sidecar_status(state: State<'_, AppState>) -> Result<bool, String> {
    let mut lock = state.sidecar.lock().unwrap();
    if let Some(supervisor) = lock.as_mut() {
        Ok(supervisor.is_running())
    } else {
        Ok(false)
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .manage(AppState {
            db_pool: Mutex::new(None),
            rpc_client: Mutex::new(None),
            sidecar: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            open_project_db,
            close_project_db,
            get_total_count,
            fetch_translation_page,
            search_units,
            call_sidecar_rpc,
            get_sidecar_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
