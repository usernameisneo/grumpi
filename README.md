# LOLLMS Server (Alpha)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<!-- Add PyPI version once published -->
<!-- [![PyPI version](https://badge.fury.io/py/lollms-server.svg)](https://badge.fury.io/py/lollms-server) -->
<!-- Add Test/Coverage badges when tests are implemented -->
<!-- [![Tests](https://img.shields.io/github/actions/workflow/status/ParisNeo/lollms_server/tests.yml?branch=main)](https://github.com/ParisNeo/lollms_server/actions/workflows/tests.yml) -->
<!-- [![Coverage Status](https://coveralls.io/repos/github/ParisNeo/lollms_server/badge.svg?branch=main)](https://coveralls.io/github/ParisNeo/lollms_server?branch=main) -->


**Created by:** ParisNeo ([GitHub Profile](https://github.com/ParisNeo))

**Repository:** [https://github.com/ParisNeo/lollms_server](https://github.com/ParisNeo/lollms_server)

**Warning:** This project is currently in **Alpha**. Expect potential bugs, breaking changes, and incomplete features. Use with caution, especially features involving code execution.

`lollms_server` is a versatile, asynchronous, multi-modal generation server designed to work with the [LOLLMS](https://github.com/ParisNeo/lollms) ecosystem. It provides a unified API endpoint (`/generate`) to interact with various text, image, video, and music generation backends (bindings) using configurable personalities and optional function execution.

## Features

*   **Multi-Modal Generation:** Supports Text-to-Text (TTT), Text-to-Image (TTI), Text-to-Video (TTV), Text-to-Music (TTM) through extensible bindings.
*   **Binding Agnostic:** Works with various backends like Ollama, OpenAI (including compatible APIs like Groq), DALL-E, Hugging Face Transformers, llama-cpp-python, vLLM, etc. Users can add their own bindings easily.
*   **LOLLMS Personality System:** Loads and utilizes standard LOLLMS personalities (both configuration-based and scripted `workflow.py` types).
*   **Asynchronous & Concurrent:** Built with FastAPI and `asyncio` for high throughput and responsiveness. Handles multiple requests concurrently.
*   **Resource Management:** Implements basic resource management (GPU semaphore) to prevent overloading during model loading. Requests are queued if resources are unavailable.
*   **Configuration Driven:** Server settings, paths, default models/bindings, security keys, logging levels, UI options, and personality configurations are managed via a central `config.toml` file.
*   **Dynamic Discovery:** Automatically discovers personalities, bindings, and functions placed in user-configured folders.
*   **Secure API:** Uses API key authentication for endpoints.
*   **Extensible Function Calling:** Supports adding custom Python functions that can be called during generation workflows (primarily within scripted personalities).
*   **Streaming Support:** Provides Server-Sent Events (SSE) for real-time streaming of Text-to-Text (TTT) responses.
*   **(Optional) Integrated Web UI:** Includes a basic Vue.js frontend served directly by FastAPI for easy interaction (configurable).
*   **Cross-Platform:** Includes installation scripts for Linux, macOS, and Windows.

## Installation

Follow these steps to install `lollms_server`.

### Prerequisites

*   **Python:** Version **3.9 or higher** is required. Ensure it's added to your system's PATH. Download from [python.org](https://www.python.org/).
*   **pip:** Python's package installer, usually included with Python.
*   **Git:** Required to clone the repository. Install from [git-scm.com](https://git-scm.com/).
*   **(Optional) Backend Servers:** If using bindings like Ollama, ensure the respective server is installed and running separately ([ollama.com](https://ollama.com/)).

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ParisNeo/lollms_server.git
    cd lollms_server
    ```

2.  **Run the Installation Script:**
    *   **On Linux or macOS:**
        ```bash
        chmod +x install.sh
        ./install.sh
        ```
    *   **On Windows:**
        Double-click `install.bat` or run it from Command Prompt/PowerShell:
        ```cmd
        .\install.bat
        ```

    This script will:
    *   Check your Python version.
    *   Create a virtual environment named `venv`.
    *   Install core Python packages from `requirements.txt` into `venv`.
    *   Copy `config.toml.example` to `config.toml` if it doesn't exist.

3.  **Activate the Virtual Environment:**
    *   **Linux/macOS:** `source venv/bin/activate`
    *   **Windows Cmd:** `venv\Scripts\activate`
    *   **Windows PowerShell:** `.\venv\Scripts\Activate.ps1` (You might need to adjust execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)
    You should see `(venv)` at the beginning of your terminal prompt. **All subsequent commands (pip installs, running the server) should be done with the environment active.**

4.  **Configure `config.toml`:**
    *   **This is the most important step!** Open `config.toml` in a text editor. Review `config.toml.example` for all options.
    *   **`[security]`:** Change `allowed_api_keys` to include your own secure API keys for accessing *this server*. Generate strong random keys.
    *   **`[bindings]`:** Configure instances for the backends you want to use (Ollama, OpenAI, DALL-E, etc.). Add necessary API keys (`api_key`), hosts (`host`), or other parameters specific to each binding type. The `type` must match a discovered binding (e.g., `ollama_binding`).
    *   **`[defaults]`:** Set the default `ttt_binding`, `ttt_model`, `tti_binding`, etc. These names must match instance names defined in `[bindings]`. Ensure the default models exist for the chosen bindings. Set `default_context_size` and `default_max_output_tokens`.
    *   **`[personalities_config]` (Optional):** Disable specific personalities or configure settings like `max_execution_retries`. Personalities are enabled by default if not listed.
    *   **`[logging]`:** Set the `log_level` (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
    *   **`[webui]` (Optional):** Set `enable_ui = false` to disable the built-in UI.
    *   **`[server]` (Optional):** Change `allowed_origins` for CORS if needed.

5.  **(Optional) Install Binding-Specific Dependencies:**
    Install Python packages required by the specific bindings you configured. Ensure `venv` is active.
    ```bash
    # Example: Install packages for the included example bindings
    pip install ollama openai pillow

    # Example: If adding Hugging Face support later
    # pip install transformers torch diffusers accelerate

    # Example: If adding llama-cpp support later
    # pip install llama-cpp-python
    ```

## Running the Server

1.  **Activate Environment:** Make sure your virtual environment (`venv`) is activated.
2.  **Run using Script:**
    *   **Linux/macOS:** `./run.sh`
    *   **Windows:** `.\run.bat`
    *(These scripts activate the environment and start Uvicorn, reading host/port/log-level from config.toml defaults)*
3.  **Run Manually (Alternative):**
    ```bash
    # Reads host/port/log-level from config.toml via main.py's logic
    python lollms_server/main.py
    # Or specify manually (overrides config for this run)
    # uvicorn lollms_server.main:app --host 0.0.0.0 --port 9600 --log-level debug
    ```

4.  **Access:** Open your browser or API client:
    *   **API Docs (Swagger):** `http://localhost:9600/docs` (or your configured host/port)
    *   **API Docs (ReDoc):** `http://localhost:9600/redoc`
    *   **Web UI (if enabled):** `http://localhost:9600`

## Server Configuration (`config.toml`)

The server is configured primarily through the `config.toml` file located in the project root. See `config.toml.example` for detailed comments on all sections.

*   **`[server]`:** Host, port, CORS `allowed_origins`.
*   **`[paths]`:** Locations for `personalities_folder`, `bindings_folder`, `functions_folder`, `models_folder`, and example folders.
*   **`[security]`:** `allowed_api_keys` for server access.
*   **`[defaults]`:** Default `*_binding`, `*_model`, `default_context_size`, `default_max_output_tokens`.
*   **`[webui]`:** `enable_ui` (boolean).
*   **`[logging]`:** `log_level` (string: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
*   **`[bindings]`:** Define named instances of specific binding types.
    *   `[bindings.INSTANCE_NAME]`: Choose a name (e.g., `openai_gpt35`, `local_ollama`).
    *   `type = "binding_type_name"`: Must match a discovered binding (e.g., `ollama_binding`).
    *   Other keys are specific to the binding (e.g., `api_key`, `host`).
*   **`[personalities_config]`:** Optional section to configure specific personalities.
    *   `[personalities_config.FOLDER_NAME]`: Key must match the personality's directory name.
    *   `enabled = true | false`: Enable/disable loading.
    *   `max_execution_retries = int`: Max retries for scripted execution (default 1).
*   **`[resource_manager]`:** GPU locking/semaphore settings (`gpu_strategy`, `gpu_limit`, `queue_timeout`).

## API Usage

The server exposes a REST API under `/api/v1`. Access requires a valid API key in the `X-API-Key` header (configure keys in `config.toml`). Interactive documentation is available at `/docs`.

### Key Endpoints

*   **`GET /`**: Basic server info (does not require API key).
*   **`GET /api/v1/list_bindings`**: Lists discovered binding types and configured instances.
*   **`GET /api/v1/list_personalities`**: Lists loaded and enabled personalities.
*   **`GET /api/v1/list_functions`**: Lists discovered custom Python functions.
*   **`GET /api/v1/list_models`**: Basic scan of the configured `paths.models_folder`.
*   **`GET /api/v1/list_available_models/{binding_name}`**: Asks a specific binding instance to list models it can access. Provides standardized details (name, size, format, context_size, etc.) where available.
*   **`POST /api/v1/generate`**: Main endpoint for triggering generation.

### `/generate` Request Body (JSON)

```jsonc
{
  "personality": "string | null", // Name of personality (or null for none)
  "prompt": "string",           // REQUIRED: User input prompt
  "extra_data": "object | null", // Optional: JSON data for context (RAG results, etc.)
  "binding_name": "string | null", // Optional: Override server's default binding
  "model_name": "string | null",   // Optional: Override server's or personality's default model
  "generation_type": "'ttt'|'tti'|'ttv'|'ttm'", // Optional: Defaults to 'ttt'
  "stream": "boolean",           // Optional: Defaults to false. Use true for TTT streaming via SSE.
  "parameters": {                // Optional: Override defaults and personality settings
    "max_tokens": 512,           // Example: Max generation length
    "temperature": 0.7,          // Example: Sampling temperature
    "system_message": "Override personality conditioning...", // Example
    // ... other binding-specific parameters ...
  },
  "functions": "array[string] | null" // Optional: Future use for function calling
}
```

### `/generate` Response

*   **Non-Streaming TTT:** `text/plain` response with the generated text.
*   **Streaming TTT:** `text/event-stream` response. Events are `data: JSON\n\n`. JSON matches `StreamChunk` model (`{"type": "chunk"|"final"|"error"|"info", "content": ..., "metadata": ...}`).
*   **TTI/TTV/TTM:** `application/json` response. Structure depends on binding, typically includes base64 data (e.g., `{"image_base64": "...", ...}`).
*   **Errors:** Standard HTTP error codes (401, 404, 422, 500) with `application/json` body like `{"detail": "Error message"}`.

### Example `curl` Requests

*(Replace `YOUR_API_KEY` with a valid key from your `config.toml`)*

```bash
# List Personalities
curl -X GET http://localhost:9600/api/v1/list_personalities -H "X-API-Key: YOUR_API_KEY"

# List models accessible by the 'default_ollama' binding instance
curl -X GET http://localhost:9600/api/v1/list_available_models/default_ollama -H "X-API-Key: YOUR_API_KEY"

# Simple TTT using personality 'lollms' and server defaults for binding/model
curl -X POST http://localhost:9600/api/v1/generate \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"personality": "lollms", "prompt": "What is FastAPI?"}'

# Streaming TTT, no personality, specifying binding/model and custom system message
curl -N -X POST http://localhost:9600/api/v1/generate \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Accept: text/event-stream" \
     -H "Content-Type: application/json" \
     -d '{
           "personality": null,
           "prompt": "Explain the concept of event loops.",
           "binding_name": "default_ollama",
           "model_name": "phi3:mini",
           "stream": true,
           "parameters": {"system_message": "Explain like I am five."}
         }'

# Image Generation (TTI) using DALL-E binding instance 'my_dalle_binding'
curl -X POST http://localhost:9600/api/v1/generate \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
           "personality": null,
           "prompt": "Epic cinematic photo of a corgi astronaut planting a flag on Mars",
           "binding_name": "my_dalle_binding",
           "generation_type": "tti",
           "parameters": {"size": "1792x1024", "quality": "hd"}
         }' \
     -o dalle_output.json # Save response to file
```

## Client Examples

See the `client_examples/` directory for Python scripts demonstrating API interactions:

*   `simple_client.py`: Interactive client for listing components and making non-streaming TTT requests with various options.
*   `streaming_client.py`: Demonstrates handling TTT streaming responses using SSE.
*   `image_client.py`: Shows how to request TTI generation and save the resulting base64 image.
*   `web/`: Contains a basic Vue.js frontend example.

Install client dependencies: `pip install -r client_examples/requirements.txt`
Run examples: `python client_examples/simple_client.py` (Edit API key/URL inside scripts).

## Extending the Server

`lollms_server` is designed to be easily extended.

### Adding Custom Bindings

1.  Create a Python file (e.g., `my_binding.py`) in `personal_bindings/`.
2.  Define a class inheriting from `lollms_server.core.bindings.Binding`.
3.  Implement abstract methods (`get_binding_config`, `load_model`, `unload_model`, `generate`, `list_available_models`) and optionally `generate_stream`, `health_check`. Ensure `list_available_models` returns dicts with a `'name'` key and other standardized fields where possible.
4.  Configure an instance in `config.toml` under `[bindings]` using the `type` name defined in `get_binding_config`.
5.  Restart server.

### Adding Custom Personalities

1.  Create a folder (e.g., `my_persona`) in `personal_personalities/`.
2.  Add `config.yaml` following LOLLMS format (must include `name`, `author`, `version`, `personality_description`, `personality_conditioning`).
3.  **(Scripted):** Add `scripts/workflow.py` with an `async def run_workflow(prompt, params, context)` function. Set `script_path: scripts/workflow.py` in `config.yaml`. The `context` dict contains `binding`, `personality`, `config`, etc.
4.  **(Optional):** Add `assets/logo.png`, `data/` folder.
5.  **(Optional):** Configure in `config.toml` under `[personalities_config]` (e.g., `my_persona = { enabled = true, max_execution_retries = 0 }`).
6.  Restart server.

### Adding Custom Functions

1.  Create a Python file in `personal_functions/` (e.g., `my_tools.py`).
2.  Define `async` functions (e.g., `async def my_tool(arg1: str): ...`).
3.  Restart server. Functions are discovered as `module_name.function_name`.
4.  Call these functions from within scripted personality workflows using the `function_manager` available in the `context` passed to `run_workflow`.

## Development

*   **Environment:** Use the created `venv` virtual environment. Activate it before running commands.
*   **Formatting:** Code follows the [Black](https://github.com/psf/black) style guide. Use `pip install black isort` and run `black .` and `isort .` before committing.
*   **Testing:** (TODO: Add details when tests are implemented).

## Contributing

Contributions via Pull Requests are welcome! Please ensure:
*   Code follows the Black style guide.
*   New features are documented.
*   Consider adding tests for new functionality.
*   Adhere to the Apache 2.0 License.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Copyright 2025 ParisNeo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.