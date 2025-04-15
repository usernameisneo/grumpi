@echo off
echo --- lollms_server Installer ---

REM Check for Python
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: python command could not be found.
    echo Please install Python (>= 3.9 recommended) and ensure it's added to your PATH during installation.
    goto :eof
)
echo Found Python:
python --version

REM Check for pip
python -m pip --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: pip for python could not be found or is not working.
    echo Please ensure pip is installed for your Python distribution (usually included).
    goto :eof
)
echo Found pip.

REM Run the core Python installation script
echo Running core installation script (install_core.py)...
python install_core.py

REM Check exit code of the Python script
if errorlevel 1 (
    echo ERROR: Core installation script failed. Please check the errors above.
    goto :eof
)

echo ---------------------------------
echo Installation script finished.
echo Please follow the 'Next Steps' printed above to activate the environment and run the server.
echo ---------------------------------

:eof
pause
exit /b 0