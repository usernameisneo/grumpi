#!/bin/bash

# Simple script to run the lollms_server using Uvicorn

# Default values
HOST="0.0.0.0"
PORT="9601"
WORKERS="1"
RELOAD="" # Set to "--reload" for development
CONFIG_FILE="config.toml"

# You can extend this script to parse command line arguments
# for host, port, workers, reload, config file etc.

# Read host/port from config file if possible (using basic tools)
# This is optional, uvicorn will read from main.py which reads the config
# CONFIG_HOST=$(grep -E '^host\s*=' "$CONFIG_FILE" | sed 's/host\s*=\s*//' | tr -d '"' | tr -d "'")
# CONFIG_PORT=$(grep -E '^port\s*=' "$CONFIG_FILE" | sed 's/port\s*=\s*//')
# if [[ ! -z "$CONFIG_HOST" ]]; then HOST="$CONFIG_HOST"; fi
# if [[ ! -z "$CONFIG_PORT" ]]; then PORT="$CONFIG_PORT"; fi

echo "Starting lollms_server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Config File: $CONFIG_FILE"
if [[ ! -z "$RELOAD" ]]; then echo "Reload Mode: Enabled"; fi

# Ensure the script is run from the project root directory
cd "$(dirname "$0")/.." || exit 1

# Command to run the server
uvicorn lollms_server.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" $RELOAD --app-dir .

# Example with reload for development (install uvicorn[standard,watch]):
# uvicorn lollms_server.main:app --host "$HOST" --port "$PORT" --reload --app-dir .
