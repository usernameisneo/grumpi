@echo off
echo --- lollms_server Installer ---

echo Checking Python...
python --version > nul 2>&1
if %errorlevel% NEQ 0 (
    echo ERROR: python command failed or returned an error.
    echo Please check Python installation and PATH.
    pause
    exit /b 1
)
echo Found Python:
python --version

echo Checking pip...
python -m pip --version > nul 2>&1
if %errorlevel% NEQ 0 (
    echo ERROR: pip command failed or returned an error.
    pause
    exit /b 1
)
echo Found pip.

echo Running core installation script (install_core.py)...
python install_core.py
if %errorlevel% NEQ 0 (
    echo ERROR: Core installation script failed. Please check the errors above.
    pause
    exit /b 1
)

echo --- Installation Finished ---
pause
exit /b 0