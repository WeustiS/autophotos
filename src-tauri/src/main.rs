// autophotos — minimal Tauri v2 shell.
//
// The real app is the FastAPI engine + browser UI. This shell just (1) spawns
// the engine as a sidecar process and (2) opens a native window pointed at it.
// So a "native app" is the same code you already run with uvicorn, wrapped.
//
// Requires the autophotos venv on PATH (or set AUTOPHOTOS_PYTHON to its python).
// Set AUTOPHOTOS_LIBRARY before launch, same as the CLI.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command};
use std::{env, thread, time::Duration};

fn spawn_engine() -> std::io::Result<Child> {
    let python = env::var("AUTOPHOTOS_PYTHON").unwrap_or_else(|_| "python".into());
    Command::new(python)
        .args([
            "-m", "uvicorn", "autophotos.api:app",
            "--host", "127.0.0.1", "--port", "8731",
        ])
        .spawn()
}

fn main() {
    // Start the Python engine; keep the handle so it is killed on exit.
    let mut child = spawn_engine().expect("failed to start autophotos engine");
    // Give uvicorn a moment to bind before the window loads localhost.
    thread::sleep(Duration::from_millis(1500));

    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running autophotos");

    let _ = child.kill();
}
