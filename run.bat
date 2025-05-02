@echo off
setlocal

REM --- Configuration ---
set "VENV_PATH=venv"
set "PYTHON_EXE=%VENV_PATH%\Scripts\python.exe"
set "MAIN_SCRIPT=lollms_server\main.py"

REM --- Check for virtual environment ---
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at '%VENV_PATH%'.
    echo Please run the install.bat script first.
    pause
    exit /b 1
)

REM --- Activate virtual environment ---
echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM --- Check for main script ---
if not exist "%MAIN_SCRIPT%" (
    echo ERROR: Main server script not found at '%MAIN_SCRIPT%'.
    echo Please ensure the script exists and you are in the project root.
    pause
    exit /b 1
)

REM --- Run the server ---
echo Starting LOLLMS Server...
echo Using Python: %PYTHON_EXE%
echo Running script: %MAIN_SCRIPT%
echo (Server host/port are now loaded from the main configuration file found by the script)
echo.

"%PYTHON_EXE%" "%MAIN_SCRIPT%"

REM --- Server finished or failed ---
if errorlevel 1 (
    echo ERROR: Server exited with an error. Check logs above.
) else (
    echo Server stopped.
)

REM --- Deactivate environment ---
echo Deactivating virtual environment...
call "%VENV_PATH%\Scripts\deactivate.bat"

endlocal
pause