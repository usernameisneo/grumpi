#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the server (adjust host/port/level if needed, or rely on config.toml)
echo "Starting LOLLMS Server..."
uvicorn lollms_server.main:app --host 0.0.0.0 --port 9600
# Deactivate environment when server stops (optional)
deactivate