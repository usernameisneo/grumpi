@echo off
REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the server (adjust host/port/level if needed, or rely on config.toml)
echo Starting LOLLMS Server...
uvicorn lollms_server.main:app --host 0.0.0.0 --port 9600

REM Deactivate environment when server stops (optional)
call venv\Scripts\deactivate.bat
pause