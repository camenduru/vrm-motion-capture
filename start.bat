@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [!] No virtual environment found.
  echo     Run:  python -m venv .venv
  echo           .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

if not exist "web\dist\index.html" (
  echo [!] Frontend not built. Build it once with:
  echo     cd web ^&^& npm install ^&^& npm run build
  exit /b 1
)

start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m app.server %*
