#!/bin/bash

# Activate virtual environment
source venv/bin/activate

CONFIG_FILE="config.toml"
# Extract host and port using grep and sed (basic parsing, assumes simple format)
# Reads the first matching line, removes quotes, takes value after '='
HOST=$(grep -E '^\s*host\s*=' "$CONFIG_FILE" | sed -E 's/^\s*host\s*=\s*"?([^"#]*)"?.*$/\1/' | head -n 1)
PORT=$(grep -E '^\s*port\s*=' "$CONFIG_FILE" | sed -E 's/^\s*port\s*=\s*([0-9]+).*/\1/' | head -n 1)

# Use defaults if parsing failed or values are empty
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"9600"}

echo "Starting LOLLMS Server on $HOST:$PORT (from $CONFIG_FILE)..."

# Run the server using extracted host and port
uvicorn lollms_server.main:app --host "$HOST" --port "$PORT"

# Deactivate environment when server stops (optional)
deactivate
