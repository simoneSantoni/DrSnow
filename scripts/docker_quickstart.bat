@echo off
REM Build (if needed) and start the DrSnow web GUI with Docker Desktop.
REM The GUI is published on http://localhost:8000 (loopback only).
setlocal
cd /d "%~dp0\.."

docker compose version >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop with "docker compose" is required:
    echo https://docs.docker.com/desktop/install/windows-install/
    exit /b 1
)

docker compose up -d --build
if errorlevel 1 exit /b 1

echo Waiting for the DrSnow GUI to start (first start compiles code)...
for /l %%i in (1,1,90) do (
    curl -fsS http://localhost:8000/healthz >nul 2>&1 && goto ready
    timeout /t 2 /nobreak >nul
)
echo The GUI did not answer; check the logs with: docker compose logs drsnow-gui
exit /b 1

:ready
echo DrSnow GUI is running at http://localhost:8000/
echo Stop it with: docker compose down
start "" http://localhost:8000/
exit /b 0
