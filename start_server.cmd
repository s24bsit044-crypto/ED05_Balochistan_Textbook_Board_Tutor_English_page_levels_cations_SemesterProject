@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Create it with: py -3.11 -m venv .venv
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >nul 2>&1
if %errorlevel%==0 (
  echo ED-05 server is already running at http://127.0.0.1:8000/
  start "" "http://127.0.0.1:8000/"
  exit /b 0
)

echo Starting ED-05 server at http://127.0.0.1:8000/
start "ED-05 Server" /wait ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
