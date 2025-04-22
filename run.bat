@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set "CONFIG_FILE=config.toml"
set "VENV_PATH=venv"
set "DEFAULT_HOST=0.0.0.0"
set "DEFAULT_PORT=9600"

REM --- Activate virtual environment ---
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at '%VENV_PATH%'.
    echo Please run the install.bat script first.
    pause
    exit /b 1
)
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM --- Parse Config File (Improved Attempt) ---
set "HOST=%DEFAULT_HOST%"
set "PORT=%DEFAULT_PORT%"
set "IN_SERVER_SECTION=0"

echo Parsing %CONFIG_FILE% for server host and port...

for /f "usebackq tokens=* delims=" %%L in ("%CONFIG_FILE%") do (
    set "LINE=%%L"
    REM Remove leading/trailing whitespace (basic)
    for /f "tokens=* delims= " %%i in ("!LINE!") do set "LINE=%%i"

    REM Check for section headers
    if "!LINE!"=="[server]" set "IN_SERVER_SECTION=1"
    if "!LINE:~0,1!"=="[" if not "!LINE!"=="[server]" set "IN_SERVER_SECTION=0"

    REM If inside [server] section, look for host/port
    if !IN_SERVER_SECTION! == 1 (
        REM Check for host line
        echo !LINE! | findstr /i /b /c:"host *=" > nul
        if not errorlevel 1 (
            for /f "tokens=1,* delims==" %%a in ("!LINE!") do (
                set "VALUE=%%b"
                REM Trim leading spaces from value
                for /f "tokens=* delims= " %%i in ("!VALUE!") do set "VALUE=%%i"
                REM Trim quotes
                if "!VALUE:~0,1!"=="""" set "VALUE=!VALUE:~1,-1!"
                REM Remove potential trailing comments (basic)
                for /f "tokens=1" %%j in ("!VALUE!") do set "HOST=%%j"
                echo Found Host: !HOST!
            )
        )
        REM Check for port line
        echo !LINE! | findstr /i /b /c:"port *=" > nul
        if not errorlevel 1 (
            for /f "tokens=1,* delims==" %%a in ("!LINE!") do (
                set "VALUE=%%b"
                REM Trim leading spaces from value
                for /f "tokens=* delims= " %%i in ("!VALUE!") do set "VALUE=%%i"
                REM Remove potential trailing comments (basic)
                for /f "tokens=1" %%j in ("!VALUE!") do set "PORT=%%j"
                echo Found Port: !PORT!
            )
        )
    )
)

REM --- Validate Port (Basic Check) ---
set /a "PORT_CHECK=PORT + 0" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Parsed port '%PORT%' is not a valid number. Using default %DEFAULT_PORT%.
    set "PORT=%DEFAULT_PORT%"
)

REM --- Run the server ---
echo.
echo Starting LOLLMS Server on %HOST%:%PORT% (from %CONFIG_FILE%)...
echo Command: uvicorn lollms_server.main:app --host %HOST% --port %PORT%
echo.

uvicorn lollms_server.main:app --host %HOST% --port %PORT%
if errorlevel 1 (
    echo ERROR: Uvicorn failed to start. Check the host/port and previous logs.
)

REM --- Deactivate environment ---
echo.
echo Deactivating virtual environment...
call "%VENV_PATH%\Scripts\deactivate.bat"

endlocal
pause
