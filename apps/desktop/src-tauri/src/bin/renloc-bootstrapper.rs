//! renloc-bootstrapper.rs
//! Standalone bootstrapper binary for Renslate.
//! Spawns and supervises the python-build-standalone runtime and renloc-core sidecar.

use std::env;
use std::process::{Command, Stdio};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    println!("[*] Renslate Bootstrapper starting with args: {:?}", args);

    // Look for python runtime in resources/ or system python3
    let current_exe = env::current_exe().unwrap_or_default();
    let app_dir = current_exe
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let bundled_python = app_dir
        .join("resources")
        .join("runtime")
        .join("bin")
        .join("python3");

    let python_bin = if bundled_python.exists() {
        bundled_python
    } else {
        std::path::PathBuf::from("python3")
    };

    println!("[*] Executing sidecar using: {:?}", python_bin);
    let mut cmd = Command::new(python_bin);
    cmd.args(&args);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    match cmd.status() {
        Ok(status) => {
            std::process::exit(status.code().unwrap_or(0));
        }
        Err(e) => {
            eprintln!("[!] Bootstrapper failed to spawn process: {}", e);
            std::process::exit(1);
        }
    }
}
