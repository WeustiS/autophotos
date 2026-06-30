# Tauri desktop wrapper (optional)

The viewer already runs as a local web app (`uvicorn autophotos.api:app`). Tauri
just wraps that exact server in a native window. **It must be built on your
machine** — a Windows app can't be cross-compiled from the Linux sandbox, and the
crates download from crates.io (which the sandbox blocks).

## One-time install (Windows)
1. Rust toolchain: https://rustup.rs  (run the installer, accept defaults)
2. WebView2 runtime — preinstalled on Windows 10/11; if missing, get the
   "Evergreen Bootstrapper" from Microsoft.
3. Tauri CLI:
   ```powershell
   cargo install tauri-cli --version "^2"
   ```

## Run (dev)
From the repo root, with your venv active and a library set:
```powershell
$env:AUTOPHOTOS_LIBRARY = "C:/Users/willc/Pictures/ukgood"
$env:AUTOPHOTOS_PYTHON  = "C:/Users/willc/code/autophotos/.venv/Scripts/python.exe"
cd src-tauri
cargo tauri dev
```
`src-tauri/src/main.rs` spawns the engine (`python -m uvicorn autophotos.api:app
--port 8731`) and opens a native window pointed at it. On exit it kills the engine.

## Build a distributable
```powershell
cd src-tauri
cargo tauri build      # produces an .msi / .exe under target/release/bundle
```

## Notes / likely tweaks
- This scaffold targets Tauri v2. If `cargo tauri dev` complains about a missing
  `frontendDist` (it isn't really used here since the window loads a URL), point
  it at any folder — `../autophotos/web` is fine.
- If the window opens before the engine is ready, increase the startup delay in
  `main.rs` (currently 1500 ms) or add a localhost health poll.
- I could not compile this in the sandbox (no Rust/network), so treat the first
  `cargo tauri dev` as the real smoke test and tell me any errors.
