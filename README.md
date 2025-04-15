# LOLLMS Server (Alpha)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<!-- Add PyPI version once published -->
<!-- [![PyPI version](https://badge.fury.io/py/lollms-server.svg)](https://badge.fury.io/py/lollms-server) -->

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
    *   **This is the most important step!** Open `config.toml` in a text editor. Review `config.toml.example` for detailed comments on all options. See the "Server Configuration" section below for more details.
    *   **Crucially:** Set your API keys in `[security]` and configure your desired generation backends in `[bindings]`.

5.  **(Optional) Install Binding-Specific Dependencies:**
    Install Python packages required by the specific bindings you configured in `config.toml`. Ensure `venv` is active.
    ```bash
    # Example: Install packages for the included example bindings
    pip install ollama openai pillow

    # Example: If using the python_builder_executor personality
    # pip install pygame numpy # Or other libraries the generated code might need
    ```

## Running the Server

1.  **Activate Environment:** Make sure your virtual environment (`venv`) is activated.
2.  **Run using Script:**
    *   **Linux/macOS:** `./run.sh`
    *   **Windows:** `.\run.bat`
3.  **Run Manually (Alternative):**
    ```bash
    # Reads host/port/log-level from config.toml via main.py's logic
    python lollms_server/main.py
    ```
4.  **Access:**
    *   **API Docs (Swagger):** `http://localhost:9600/docs` (or your configured host/port)
    *   **Web UI (if enabled):** `http://localhost:9600`

## Server Configuration (`config.toml`)

The `config.toml` file controls server behavior, paths, security, defaults, and component configurations.

*   **`[server]`**
    *   `host`: IP address to bind (e.g., `"0.0.0.0"`, `"127.0.0.1"`). Default: `"0.0.0.0"`.
    *   `port`: Port number. Default: `9600`.
    *   `allowed_origins`: List of origins for CORS (Cross-Origin Resource Sharing). Needed for web UIs accessing the API from different domains/ports. Default includes common local development ports and `"null"` (for `file://` access - use cautiously). Example: `["http://localhost:5173", "https://my-app.com"]`. Use `["*"]` to allow all (less secure).

*   **`[paths]`**
    *   Specifies directories for discovering custom components. Paths are relative to the `config.toml` location unless absolute.
    *   `personalities_folder`: Location of your custom personalities (e.g., `"personal_personalities/"`).
    *   `bindings_folder`: Location of your custom bindings (e.g., `"personal_bindings/"`).
    *   `functions_folder`: Location of your custom functions (e.g., `"personal_functions/"`).
    *   `models_folder`: Base directory for local models used by some bindings (e.g., `"models/"`). Bindings may expect subfolders like `ttt`, `tti`.
    *   `example_*_folder`: Points to the built-in examples (e.g., `"examples/personalities/"`).

*   **`[security]`**
    *   `allowed_api_keys`: **Required.** List of secret keys clients must provide in the `X-API-Key` header. Generate strong random keys. Example: `["your-secret-key-1", "another-secure-key"]`.

*   **`[defaults]`**
    *   Specifies fallback binding instances and models if not provided in the API request. The `*_binding` value must match an instance name defined under `[bindings]`.
    *   `ttt_binding`, `ttt_model`: For Text-to-Text.
    *   `tti_binding`, `tti_model`: For Text-to-Image.
    *   `ttv_binding`, `ttv_model`: For Text-to-Video.
    *   `ttm_binding`, `ttm_model`: For Text-to-Music.
    *   `default_context_size`: Default context window size (informational, used if not specified elsewhere).
    *   `default_max_output_tokens`: Default maximum generation length (maps to `max_tokens`).

*   **`[webui]` (Optional)**
    *   `enable_ui`: `true` (default) or `false`. If `true`, the server attempts to serve static files from `webui/dist/`. Requires building the UI (`cd webui && npm run build`).

*   **`[logging]` (Optional)**
    *   `log_level`: Sets the logging verbosity. Options: `"DEBUG"`, `"INFO"` (default), `"WARNING"`, `"ERROR"`, `"CRITICAL"`. `DEBUG` is very verbose.

*   **`[bindings]`**
    *   Defines named instances of specific binding types. You **must** configure instances for the bindings you want to use.
    *   **`[bindings.INSTANCE_NAME]`**: Replace `INSTANCE_NAME` with a unique identifier (e.g., `local_ollama`, `openai_gpt4o`, `dalle3_standard`).
    *   **`type = "binding_type_name"`**: **Required.** Must match the `type_name` returned by the binding's `get_binding_config()` (usually the Python filename stem, e.g., `ollama_binding`, `openai_binding`, `dalle_binding`).
    *   **Other Keys:** Parameters specific to the binding `type`. Examples:
        *   For `ollama_binding`: `host = "http://localhost:11434"`
        *   For `openai_binding`: `api_key = "sk-..."`, `base_url = "..."` (optional)
        *   For `dalle_binding`: `api_key = "sk-..."`, `model = "dall-e-3"` (optional default)

*   **`[personalities_config]` (Optional)**
    *   Override settings for specific personalities discovered in the `paths`. If a personality folder exists but isn't listed here, default settings apply (`enabled = true`, `max_execution_retries = 1`).
    *   **`[personalities_config.FOLDER_NAME]`**: Key must match the personality's directory name (e.g., `python_builder_executor`, `my_rag_bot`).
    *   `enabled = true | false`: Set to `false` to prevent this personality from being loaded.
    *   `max_execution_retries = int`: Max retries for scripted execution workflows (default 1).

*   **`[resource_manager]`**
    *   `gpu_strategy`: `"semaphore"` (limit N concurrent GPU tasks) or `"simple_lock"` (limit 1). Controls access for bindings that request GPU resources via the manager.
    *   `gpu_limit`: Max concurrent tasks if using `semaphore`.
    *   `queue_timeout`: Seconds to wait for a resource before failing the request.

## API Usage

See `/docs` on your running server for interactive API documentation (Swagger UI).

### Authentication

Send a valid API key (from `config.toml [security].allowed_api_keys`) in the `X-API-Key` HTTP header for all `/api/v1/*` endpoints.

### Key Endpoints

*   **`GET /`**: Basic server info (no key required).
*   **`GET /api/v1/list_bindings`**: Lists discovered binding types and configured instances.
*   **`GET /api/v1/list_personalities`**: Lists loaded and enabled personalities.
*   **`GET /api/v1/list_available_models/{binding_name}`**: Asks a specific binding instance (e.g., `default_ollama`) to list models it can access. Returns detailed model info.
*   **`POST /api/v1/generate`**: Main endpoint for triggering generation (see request/response details below).

### `/generate` Endpoint

**Request Body (JSON):**

```jsonc
{
  "personality": "string | null", // Name of loaded personality (or null/omit for none)
  "prompt": "string",           // REQUIRED: User input prompt
  "extra_data": "object | null", // Optional: Extra JSON data for context (RAG results, etc.) passed to scripts
  "binding_name": "string | null", // Optional: Override default binding instance from [defaults] or personality
  "model_name": "string | null",   // Optional: Override default model name from [defaults] or personality
  "generation_type": "'ttt'|'tti'|'ttv'|'ttm'", // Optional: Defaults to 'ttt' (Text-to-Text)
  "stream": "boolean",           // Optional: Defaults to false. Use true for TTT streaming via SSE.
  "parameters": {                // Optional: Override default/personality params. Passed to binding/script.
    "max_tokens": "int | null",  // Example: Max generation length (overrides default_max_output_tokens)
    "temperature": "float | null", // Example: Sampling temperature
    "system_message": "string | null", // Example: Override personality conditioning
    // ... other binding-specific parameters (e.g., size, quality for DALL-E) ...
  },
  "functions": "array[string] | null" // Reserved for future function calling features
}
```

**Response:** See "API Usage" section in previous README version for response types (text/plain, text/event-stream, application/json based on type and streaming).

### Client Examples

See the `client_examples/` directory for Python and basic Web examples demonstrating API usage.

## Extending the Server

`lollms_server` is designed for extension. Add your own components by placing files in the folders defined in `config.toml [paths]`.

### Adding Custom Bindings

Provide interfaces to different AI model backends (local or remote APIs).

1.  **Create File:** Add `your_binding_name.py` to the `personal_bindings/` folder.
2.  **Inherit:** Define a class inheriting from `lollms_server.core.bindings.Binding`.
    ```python
    from lollms_server.core.bindings import Binding
    class MyCustomBinding(Binding):
        # Required: Unique name used in config.toml 'type' field
        binding_type_name = "my_custom_binding"
        # ... implement methods ...
    ```
3.  **Implement Abstract Methods:**
    *   `get_binding_config() -> Dict`: Return metadata (`type_name`, `description`, `requirements` list, `config_template` dict).
    *   `async load_model(self, model_name: str) -> bool`: Load/prepare the specified model. Use `self.resource_manager` if needed. Idempotent.
    *   `async unload_model(self) -> bool`: Release model resources. Idempotent.
    *   `async generate(self, prompt: str, params: Dict, request_info: Dict) -> Union[str, Dict]`: Perform generation. Return `str` for TTT, `dict` for TTI/TTV/TTM (e.g., `{"image_base64": "..."}`). Access instance config via `self.config`.
    *   `async list_available_models(self) -> List[Dict]`: Return list of models this binding can use. Each dict **must** have a `'name'` key. Populate other standard keys (`size`, `format`, `context_size`, etc.) if possible.
4.  **(Optional) Implement Streaming:** `async generate_stream(self, ...) -> AsyncGenerator[Dict, None]` for TTT. Yield `StreamChunk`-like dicts.
5.  **(Optional) Implement Health Check:** `async health_check(self) -> Tuple[bool, str]`.
6.  **Configure:** Add an instance in `config.toml`:
    ```toml
    [bindings.my_instance_name]
    type = "my_custom_binding" # Matches binding_type_name
    your_specific_config_key = "value"
    ```
7.  **Restart:** The server will discover and instantiate your binding.

### Adding Custom Personalities

Define the behavior, instructions, and context for the AI.

**1. Simple Personality (Config-based):**

*   **Create Folder:** Make a new directory in `personal_personalities/` (e.g., `my_poet_persona`).
*   **Create `config.yaml`:** Add a `config.yaml` file inside the folder. Essential keys:
    *   `name`: Unique name for the personality (used in API requests).
    *   `author`: Your name or identifier.
    *   `version`: Version number/string.
    *   `personality_description`: Brief description shown in listings.
    *   `personality_conditioning`: The core system prompt/instructions given to the LLM.
    *   *(Optional but Recommended):* `category`, `tags`, `welcome_message`, `user_message_prefix`, `ai_message_prefix`, default model parameters (`model_temperature`, `model_n_predicts`, etc.), `dependencies`.
*   **(Optional) Assets:** Add an `assets/logo.png` (or other assets). Reference the icon in `config.yaml`.
*   **(Optional) Data:** Add a `data/` folder for RAG databases or other files. These are *not* automatically used; scripted personalities must load them.
*   **(Optional) Configure:** Add an entry in `config.toml [personalities_config]` if you need to disable it or set specific retries.
*   **Restart:** The server will discover and load it.

**2. Scripted Personality (Agentic):**

*   **Follow Simple Steps:** Create the folder, `config.yaml` (including essential keys), and optional `assets/`, `data/`.
*   **Create Script:** Add a `scripts/` subfolder and place your workflow file inside (e.g., `scripts/my_workflow.py`).
*   **Define `run_workflow`:** In your script file, define the main entry point:
    ```python
    # scripts/my_workflow.py
    import asyncio
    from typing import Any, Dict, Optional
    # Import binding helpers etc. if needed

    async def run_workflow(prompt: str, params: Dict, context: Dict) -> Any:
        # Access provided objects:
        binding = context.get('binding') # The selected Binding instance
        personality = context.get('personality') # The Personality object itself
        app_config = context.get('config') # The server's AppConfig
        function_manager = context.get('function_manager')
        resource_manager = context.get('resource_manager')
        extra_data = context.get('extra_data') # Data from the API request

        # Use binding helpers for interaction:
        # is_sure = await binding.ask_yes_no("Are you sure?", params)
        # code_blocks = await binding.generate_and_extract_all_codes(prompt, params, context)
        # result = await function_manager.execute_function("module.func", {"arg": ...})

        # Perform logic...

        # Return final result (string, dict, or async generator for streaming)
        return "Workflow completed successfully!"
    ```
*   **Set `script_path`:** In `config.yaml`, add the line: `script_path: scripts/my_workflow.py` (relative to the personality root).
*   **(Optional) Configure:** Add entry in `config.toml [personalities_config]` (e.g., `my_agent = { enabled = true, max_execution_retries = 3 }`).
*   **Restart:** The server finds the personality, loads the script, and will call `run_workflow` when this personality is used in a `/generate` request.

### Adding Custom Functions

1.  Create a Python file in `personal_functions/` (e.g., `my_utils.py`).
2.  Define `async` functions (e.g., `async def calculate_something(x: int): ...`).
3.  Restart server. Functions are discovered as `module_name.function_name` (e.g., `my_utils.calculate_something`).
4.  Call from scripted personalities:
    ```python
    # Inside run_workflow
    function_manager = context.get('function_manager')
    if function_manager:
        success, result = await function_manager.execute_function(
            "my_utils.calculate_something",
            {"x": 10}
        )
        if success:
            # Use result
            pass
    ```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Copyright 2025 ParisNeo

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.