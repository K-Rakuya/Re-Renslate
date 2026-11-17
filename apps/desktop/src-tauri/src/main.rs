// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod job_guard;
pub mod rpc_client;
pub mod sidecar;

use crate::rpc_client::RpcClient;
use crate::sidecar::SidecarSupervisor;
use renloc_db::{QueryEngine, ReadOnlyPool, TranslationUnitRow};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{Manager, State};

pub struct AppState {
    pub db_pool: Mutex<Option<ReadOnlyPool>>,
    pub rpc_client: Mutex<Option<RpcClient>>,
    pub sidecar: Mutex<Option<SidecarSupervisor>>,
}

fn get_python_path() -> PathBuf {
    if let Ok(p) = std::env::var("RENSLATE_PYTHON") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return pb;
        }
    }

    let mut candidates = Vec::new();

    // Check executable directory and its parents
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join(".venvs/core/bin/python"));
            candidates.push(exe_dir.join("resources/runtime/bin/python"));
            if let Some(p1) = exe_dir.parent() {
                candidates.push(p1.join(".venvs/core/bin/python"));
                if let Some(p2) = p1.parent() {
                    candidates.push(p2.join(".venvs/core/bin/python"));
                }
            }
        }
    }

    // Check current working directory and its parents dynamically
    if let Ok(cwd) = std::env::current_dir() {
        let mut curr = cwd.clone();
        for _ in 0..5 {
            candidates.push(curr.join(".venvs/core/bin/python"));
            candidates.push(curr.join("resources/runtime/bin/python"));
            if let Some(parent) = curr.parent() {
                curr = parent.to_path_buf();
            } else {
                break;
            }
        }
    }

    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("python3")
}

fn get_socket_path() -> PathBuf {
    std::env::temp_dir().join("renslate_sidecar_rpc.sock")
}

fn ensure_sidecar_started(state: &AppState) -> Result<RpcClient, String> {
    let mut client_lock = state.rpc_client.lock().unwrap();
    if let Some(client) = client_lock.as_ref() {
        return Ok(client.clone());
    }

    let socket_path = get_socket_path();
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let py_path = get_python_path();
    let mut supervisor = SidecarSupervisor::new(&py_path);
    let token_hex = supervisor.auth_token().as_hex().to_string();
    let sock_str = socket_path.to_string_lossy().to_string();

    supervisor
        .start(&[
            "-m",
            "renloc_core",
            "serve",
            "--socket",
            &sock_str,
            "--token",
            &token_hex,
        ])
        .map_err(|e| format!("Failed to spawn Python sidecar: {}", e))?;

    // Wait up to 3 seconds for socket file to appear
    let start = std::time::Instant::now();
    let mut socket_ready = false;
    while start.elapsed() < std::time::Duration::from_millis(3000) {
        if socket_path.exists() {
            socket_ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    if !socket_ready {
        supervisor.stop();
        return Err("Sidecar process spawned, but RPC socket file did not appear within 3 seconds.".to_string());
    }

    let client = RpcClient::with_token(socket_path, token_hex);
    *client_lock = Some(client.clone());

    let mut sidecar_lock = state.sidecar.lock().unwrap();
    *sidecar_lock = Some(supervisor);

    Ok(client)
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
        lock.clone()
            .ok_or_else(|| "No database opened".to_string())?
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
        lock.clone()
            .ok_or_else(|| "No database opened".to_string())?
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
        lock.clone()
            .ok_or_else(|| "No database opened".to_string())?
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
        let existing = {
            let lock = state.rpc_client.lock().unwrap();
            lock.clone()
        };
        match existing {
            Some(c) => c,
            None => ensure_sidecar_started(&state)?,
        }
    };
    client
        .call(&method, payload)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_sidecar_status(state: State<'_, AppState>) -> Result<bool, String> {
    let mut lock = state.sidecar.lock().unwrap();
    if let Some(supervisor) = lock.as_mut() {
        Ok(supervisor.is_running())
    } else {
        drop(lock);
        match ensure_sidecar_started(&state) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
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
        .setup(|app| {
            let state = app.state::<AppState>();
            if let Err(e) = ensure_sidecar_started(&state) {
                eprintln!("[!] Sidecar lazy start deferred: {}", e);
            } else {
                println!("[+] Sidecar auto-connected on Tauri launch!");
            }
            Ok(())
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
