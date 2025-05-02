#!/bin/bash

VENV_DIR="venv"
MAIN_SCRIPT="lollms_server/main.py"

# --- Check for virtual environment ---
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at '$VENV_DIR'."
    echo "Please run the install.sh script first."
    exit 1
fi

# --- Activate virtual environment ---
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi
echo "Virtual environment activated."

# --- Check for main script ---
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "ERROR: Main server script not found at '$MAIN_SCRIPT'."
    echo "Please ensure the script exists and you are in the project root."
    # Deactivate before exiting on error
    deactivate
    exit 1
fi

# --- Run the server ---
echo "Starting LOLLMS Server..."
echo "Running script: $MAIN_SCRIPT"
echo "(Server host/port are now loaded from the main configuration file found by the script)"
echo ""

# Get the Python executable from the activated venv
PYTHON_EXE="$VENV_DIR/bin/python"

"$PYTHON_EXE" "$MAIN_SCRIPT"

# Capture exit code
EXIT_CODE=$?

# --- Server finished or failed ---
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Server exited with code $EXIT_CODE. Check logs above."
else
    echo ""
    echo "Server stopped."
fi

# --- Deactivate environment ---
echo "Deactivating virtual environment..."
deactivate

# Exit with the server's exit code
exit $EXIT_CODE