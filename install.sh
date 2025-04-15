#!/bin/bash

# Simple installer script for lollms_server on Linux/macOS

echo "--- lollms_server Installer ---"

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 command could not be found."
    echo "Please install Python 3 (>= 3.9 recommended) and ensure it's in your PATH."
    exit 1
fi
echo "Found Python 3: $(command -v python3)"

# Check for pip (usually comes with Python 3)
if ! python3 -m pip --version &> /dev/null
then
    echo "ERROR: pip for python3 could not be found or is not working."
    echo "Please ensure pip is installed for your Python 3 distribution."
    exit 1
fi
echo "Found pip for Python 3."

# Run the core Python installation script
echo "Running core installation script (install_core.py)..."
python3 install_core.py

# Check exit code of the Python script
if [ $? -ne 0 ]; then
    echo "ERROR: Core installation script failed. Please check the errors above."
    exit 1
fi

echo "---------------------------------"
echo "Installation script finished."
echo "Please follow the 'Next Steps' printed above to activate the environment and run the server."
echo "---------------------------------"

exit 0