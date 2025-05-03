# LOLLMS Server (Alpha)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Config](https://img.shields.io/badge/Config-ConfigGuard-blueviolet.svg)](https://github.com/ParisNeo/ConfigGuard)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<!-- [![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)] -->

**Created by:** ParisNeo ([GitHub Profile](https://github.com/ParisNeo))

**Repository:** [https://github.com/ParisNeo/lollms_server](https://github.com/ParisNeo/lollms_server)

**Warning:** This project is currently in **Alpha**. Expect potential bugs, breaking changes, and incomplete features. Use with caution, especially features involving code execution.

`lollms_server` is a versatile, asynchronous, multi-modal generation server designed to work with the [LOLLMS](https://github.com/ParisNeo/lollms) ecosystem. It provides a unified API endpoint (`/generate`) to interact with various text, image, audio, video, and music generation backends (bindings) using configurable personalities and optional function execution. Built with **ConfigGuard**, it offers robust, schema-driven configuration, validation, encryption, and versioning. It aims to provide a reliable and extensible platform for experimenting with and deploying multi-modal AI workflows.

## Key Features

*   **Multi-Modal Generation:** Supports Text-to-Text (TTT), Text-to-Image (TTI), Text-to-Video (TTV), Text-to-Music (TTM), Text-to-Speech (TTS), Speech-to-Text (STT), Image-to-Image (I2I), Audio-to-Audio, and more through extensible bindings.
*   **Multimodal Input API:** Accepts complex inputs via the `/generate` endpoint's `input_data` list, allowing combinations of text, images, audio, etc., each with assigned roles (e.g., `user_prompt`, `input_image`, `system_context`, `mask_image`).
*   **Binding Agnostic:** Easily integrate different AI backends (local or remote APIs) like Ollama, OpenAI (including compatible APIs like Groq, Together.ai), DALL-E, Gemini, llama-cpp-python, Hugging Face Transformers, Diffusers, etc.
*   **LOLLMS Personality System:** Loads and utilizes standard LOLLMS personalities, including configuration-based prompts (`config.yaml`) and scripted Python workflows (`scripts/workflow.py`).
*   **Configuration Fortress (ConfigGuard):**
    *   Schema-driven configuration (YAML, TOML, JSON, SQLite) with validation.
    *   Built-in encryption for sensitive settings (e.g., API keys).
    *   Automatic configuration versioning and migration support.
    *   Interactive setup wizard (`configuration_wizard.py`) for easy initial setup.
*   **Asynchronous & Concurrent:** Built with FastAPI and `asyncio` for high throughput and responsiveness, handling multiple requests concurrently.
*   **Resource Management:** Implements basic resource management (GPU semaphore/lock) to prevent overloading during model loading or intensive generation tasks. Requests are queued if resources are unavailable.
*   **Dynamic Discovery:** Automatically discovers personalities, bindings, and functions placed in user-configured folders upon startup.
*   **Secure API:** Uses API key authentication (`X-API-Key` header) for all functional API endpoints.
*   **Streaming Support:** Provides Server-Sent Events (SSE) for real-time streaming of Text-to-Text (TTT) and potentially other compatible modalities.
*   **Extensible Function Calling:** Allows defining custom Python functions that can be discovered and executed within scripted personality workflows.
*   **Tokenizer Utilities:** API endpoints to tokenize, detokenize, and count tokens using the active binding's model.
*   **Model Information:** API endpoint to retrieve context size and other details about the currently loaded model for a binding.
*   **(Optional) Integrated Web UI:** Includes a basic Vue.js frontend served directly by FastAPI for easy interaction and testing (configurable).
*   **Cross-Platform:** Includes installation scripts for Linux, macOS, and Windows.
*   **Testing:** Includes a `pytest`-based test suite for core components and API endpoints.

## Core Concepts

*   **Binding:** A Python class that interfaces with a specific AI model backend (e.g., OpenAI API, local Ollama server, `llama-cpp-python`). Bindings handle model loading, parameter translation, generation calls, and capability reporting. They are defined by Python code and a `binding_card.yaml` file containing metadata and the instance configuration schema. You configure specific *instances* of these bindings (e.g., `my_openai_gpt4o`, `local_llama3_8b`) in separate files within the `lollms_configs/bindings/` directory.
*   **Personality:** Defines the behavior, instructions, and context for the AI, similar to custom GPTs or system prompts. They consist of a `config.yaml` file (parsed by `PersonalityConfig`) and optional assets (`assets/`) or Python scripts (`scripts/workflow.py`). Scripted personalities allow for complex, agentic behavior.
*   **Function:** A custom Python function placed in the user's functions folder, discoverable by the server and callable from scripted personalities to extend capabilities (e.g., calling external tools, performing calculations).
*   **Multimodal Input (`input_data`):** The `/generate` endpoint accepts a list of input objects, each specifying its `type` (text, image, audio, etc.), `role` (how it should be used, e.g., `user_prompt`, `input_image`, `mask_image`), and the actual `data` (text content, base64 string, URL).
*   **API Key:** A secret key required to authenticate requests to the server's API endpoints. Defined in the main configuration file.
*   **ConfigGuard:** The library used to manage configuration loading, validation, encryption, and versioning based on defined schemas. The main configuration (e.g., `main_config.yaml`) and binding instance configurations (e.g., `my_binding.yaml`) are handled by ConfigGuard.

## Installation

Follow these steps to install `lollms_server`.

### Prerequisites

*   **Python:** Version **3.9 or higher** is required. Ensure it's added to your system's PATH. Download from [python.org](https://www.python.org/).
*   **pip:** Python's package installer, usually included with Python.
*   **Git:** Required to clone the repository. Install from [git-scm.com](https://git-scm.com/).
*   **(Optional) Backend Servers/Libraries:** Depending on the bindings you intend to use, you might need:
    *   Ollama server running ([ollama.com](https://ollama.com/))
    *   Specific Python libraries for local models (e.g., `llama-cpp-python`, `transformers`, `torch`, `diffusers`)
    *   API keys for remote services (OpenAI, Gemini, etc.)

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

    This script (`install_core.py`) will:
    *   Check your Python version.
    *   Create a Python virtual environment named `venv`.
    *   Install core Python packages (`fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `pipmaster`, `configguard`, `ascii_colors`) from `requirements.txt` into `venv`.
    *   **Run the interactive `configuration_wizard.py`**. This wizard:
        *   Prompts you to select a directory for user configurations (defaults to `lollms_configs`).
        *   Asks you to choose a format for the main configuration file (e.g., `main_config.yaml` - default, `.toml`, `.json`, `.db`).
        *   Guides you through setting up essential server settings, paths, security (API keys, optional encryption), default bindings, and resource limits.
        *   Optionally applies presets to quickly configure common binding combinations (like Ollama + DALL-E, LlamaCpp + Diffusers).
        *   Optionally guides you through creating the necessary configuration files for the binding *instances* selected in a preset (e.g., `lollms_configs/bindings/my_ollama_gpu.yaml`).
    *   Offer to install optional dependencies (for specific bindings or config file formats).
    *   Provide a CLI menu to manage binding instances (add/edit/remove) after the wizard.

3.  **Activate the Virtual Environment:**
    *   **Linux/macOS:** `source venv/bin/activate`
    *   **Windows Cmd:** `venv\Scripts\activate.bat`
    *   **Windows PowerShell:** `.\venv\Scripts\Activate.ps1` (You might need to adjust execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)

    You should see `(venv)` at the beginning of your terminal prompt. **All subsequent commands (pip installs, running the server) must be done with the environment active.**

4.  **(Optional) Install Additional Dependencies:**
    If you plan to use bindings whose dependencies weren't installed via the installer menu (option 2), install them now. Check the `requirements` list in the respective `zoos/bindings/*/binding_card.yaml` or use the extras defined in `pyproject.toml`. Ensure `venv` is active.
    ```bash
    # Example: Install extras for Ollama and Diffusers support
    (venv) pip install .[ollama,diffusers]

    # Example: Install only llama-cpp-python (if not done via installer)
    # Use pre-built wheels if available for your platform/GPU for easier setup
    # See https://github.com/abetlen/llama-cpp-python#installation
    (venv) pip install llama-cpp-python
    ```

5.  **Review Configuration:**
    *   Check the main configuration file created by the wizard (e.g., `lollms_configs/main_config.yaml`).
    *   Check the binding instance configuration files created (e.g., `lollms_configs/bindings/my_ollama_gpu.yaml`). Ensure API keys or other specific settings are correct.

## Running the Server

1.  **Activate Environment:** Make sure your virtual environment (`venv`) is activated.
2.  **Run using Script:**
    *   **Linux/macOS:** `./run.sh`
    *   **Windows:** `.\run.bat`
    These scripts activate the environment (if not already) and run `lollms_server/main.py`, which reads host/port/etc., from your main configuration file.
3.  **Run Manually (Alternative):**
    ```bash
    # Reads host/port/log-level from the main config file
    (venv) python lollms_server/main.py
    ```
4.  **Access:**
    *   **API Docs (Swagger):** `http://localhost:9601/docs` (or your configured host/port). Use the "Authorize" button here (enter your API key) to test authenticated endpoints.
    *   **API Health:** `http://localhost:9601/health` (No API key needed).
    *   **Web UI (if enabled):** `http://localhost:9601`

## Server Configuration

Configuration is primarily managed via the main configuration file (e.g., `lollms_configs/main_config.yaml`) created by the `configuration_wizard.py`. Binding-specific settings are stored in separate files within the configured `instance_bindings_folder` (e.g., `lollms_configs/bindings/`).

**Main Configuration File (`main_config.yaml` or similar):**

*   **`server`**: `host`, `port`, `allowed_origins` (for CORS, crucial for web UIs).
*   **`paths`**: Absolute paths resolved during startup for discovering `personalities`, `bindings`, `functions`, `models`, the user `config_base_dir`, the `instance_bindings_folder`, and example components (`zoos`).
*   **`security`**: `allowed_api_keys` (list of required secret keys), `encryption_key` (optional Fernet key for binding configs).
*   **`defaults`**: Fallback `ttt_binding`, `ttt_model`, `tti_binding`, etc. Values must match instance names defined in `bindings_map`. Also includes `default_context_size` and `default_max_output_tokens`.
*   **`bindings_map`**: **Crucial section** mapping your chosen instance names (e.g., `my_openai_gpt4o`) to their binding type names (e.g., `openai_binding`). This links your instance config files to the correct code.
*   **`resource_manager`**: `gpu_strategy` (`semaphore`, `simple_lock`, `none`), `gpu_limit`, `queue_timeout`.
*   **`webui`**: `enable_ui` (boolean) to serve the static Vue.js app.
*   **`logging`**: `log_level` ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
*   **`personalities_config`**: Optional overrides for specific personalities (e.g., `{'python_builder_executor': {'enabled': false}}`). Keys must match personality folder names.

**Binding Instance Configuration Files (e.g., `lollms_configs/bindings/my_openai_gpt4o.yaml`):**

*   These files contain settings specific to one instance of a binding, loaded via ConfigGuard using the schema from the binding's `binding_card.yaml`.
*   **`type`**: Must match the `type_name` from the binding's card (e.g., `openai_binding`).
*   **`binding_instance_name`**: Must match the name used in the main config's `bindings_map` (e.g., `my_openai_gpt4o`).
*   **Other keys:** Specific settings defined in the `instance_schema` of the binding's `binding_card.yaml` (e.g., `api_key`, `base_url`, `models_folder`, `n_gpu_layers`).

## API Usage

Interact with the server via its REST API. Use the Swagger UI at `/docs` for detailed endpoint information and testing.

### Authentication

All `/api/v1/*` endpoints (except `/health`) require authentication. Include a valid API key from your main configuration file in the `X-API-Key` HTTP header with every request.

### Key Endpoints

#### Server Info

*   **`GET /health`**: Checks server status, version, and if API key is required.

#### Listing

*   **`GET /api/v1/list_bindings`**: Lists discovered binding types (from cards) and configured instances (from instance files).
*   **`GET /api/v1/list_personalities`**: Lists loaded and enabled personalities.
*   **`GET /api/v1/list_functions`**: Lists discovered custom functions.
*   **`GET /api/v1/list_available_models/{binding_instance_name}`**: Asks a specific binding instance (e.g., `default_ollama`) to list models it can access. Returns detailed `ModelInfo` including capabilities.
*   **`GET /api/v1/list_models`**: Lists models found in the configured `models_folder` subdirectories (basic file scan).

#### Generation

*   **`POST /api/v1/generate`**: The main endpoint for triggering generation tasks (TTT, TTI, etc.). See details below.

#### Utilities

*   **`POST /api/v1/tokenize`**: Tokenizes text using the specified binding's currently loaded model.
    *   **Request:** `{"text": "string", "binding_name": "string", "add_bos": true, "add_eos": false}`
    *   **Response:** `{"tokens": [int], "count": int}`
    *   **Note:** Requires a model to be loaded on the binding. Not all bindings support this.
*   **`POST /api/v1/detokenize`**: Converts token IDs back to text using the specified binding's currently loaded model.
    *   **Request:** `{"tokens": [int], "binding_name": "string"}`
    *   **Response:** `{"text": "string"}`
    *   **Note:** Requires a model to be loaded on the binding. Not all bindings support this.
*   **`POST /api/v1/count_tokens`**: Counts tokens in the provided text using the specified binding's currently loaded model.
    *   **Request:** `{"text": "string", "binding_name": "string"}`
    *   **Response:** `{"count": int}`
    *   **Note:** Requires a model to be loaded on the binding. Relies on the binding's `tokenize` implementation.
*   **`GET /api/v1/get_model_info/{binding_name}`**: Retrieves info about the *currently loaded* model for a specific binding.
    *   **Response:** `{"binding_instance_name": "string", "model_name": "string|null", "context_size": int|null, "max_output_tokens": int|null, "supports_vision": bool, "supports_audio": bool, "details": {}}`
    *   **Note:** Does not load a model. Returns info only if a model is already active on the binding.

### `/generate` Endpoint Details

This is the core endpoint for all generation types.

**Request Body (`application/json`):**

```jsonc
{
  // --- Input Specification (Required) ---
  "input_data": [
    // List of input items. MUST contain at least one item.
    {
      "type": "'text'|'image'|'audio'|'video'|'document'", // Type of data
      "role": "string", // How this input is used (e.g., 'user_prompt', 'input_image', 'system_context', 'controlnet_image', 'mask_image')
      "data": "string", // The actual data (text content, base64 encoded string, URL)
      "mime_type": "string | null", // REQUIRED for binary types (e.g., 'image/png', 'audio/wav')
      "metadata": "object | null" // Optional additional info (filename, etc.)
    }
    // ... potentially more input items ...
  ],

  // --- Generation Control (Optional - Uses defaults if omitted) ---
  "personality": "string | null", // Name of loaded personality to use
  "binding_name": "string | null", // Specific binding instance name (from config files)
  "model_name": "string | null",   // Specific model name for the chosen binding
  "generation_type": "'ttt'|'tti'|'tts'|'stt'|'ttv'|'ttm'|'i2i'|'audio2audio'", // Default: 'ttt'
  "stream": "boolean",           // Default: false. Use true for streaming (TTT, TTS supported)
  "parameters": {                // Optional dict passed to binding/script. Overrides defaults/personality params.
    "max_tokens": "int | null",
    "temperature": "float | null",
    "system_message": "string | null", // Override personality conditioning/system context
    "image_size": "string | null",    // Example: for TTI
    "controlnet_scale": "float | null" // Example: for ControlNet
    // ... other binding/model specific parameters ...
  },
  "functions": "array[string] | null" // Reserved for future structured function calling
}
```

**Example TTT Request:**

```json
{
  "input_data": [
    {"type": "text", "role": "user_prompt", "data": "Explain the concept of recursion simply."}
  ],
  "personality": "lollms",
  "stream": false
}
```

**Example TTI Request:**

```json
{
  "input_data": [
    {"type": "text", "role": "user_prompt", "data": "Astronaut riding a unicorn on the moon, digital art"}
  ],
  "generation_type": "tti",
  "binding_name": "my_dalle3_api", // Instance name defined in main_config.yaml & bindings/my_dalle3_api.yaml
  "parameters": { "size": "1024x1024", "quality": "hd" }
}
```

**Example Vision (VLM) Request:**

```json
{
  "input_data": [
    {"type": "text", "role": "user_prompt", "data": "What objects are in this image?"},
    {"type": "image", "role": "input_image", "data": "BASE64_ENCODED_IMAGE_STRING", "mime_type": "image/jpeg"}
  ],
  "generation_type": "ttt", // Still TTT as output is text
  "binding_name": "gemini1" // Instance name defined in main_config.yaml & bindings/gemini1.yaml
}
```

**Response:**

*   **Non-streaming:** `application/json` containing the `GenerateResponse` model (includes `output: List[OutputData]`, `personality`, `request_id`, `execution_time`). Output list can contain multiple items (text, image, etc.). Includes extracted `thoughts` within text `OutputData` if generated by the model/script.
*   **Streaming:** `text/event-stream` with Server-Sent Events (SSE). Each event is `data: JSON_STRING\n\n`, where `JSON_STRING` represents a `StreamChunk` model (`{"type": "...", "content": ..., "thoughts": ..., "metadata": ...}`). Types include `chunk`, `info`, `error`, `final`. `thoughts` can appear in `chunk` type events.

## Extending the Server

Add your own components by placing Python code and configuration files in the directories specified in your main config's `[paths]` section (e.g., `personal_bindings/`, `personal_personalities/`, `personal_functions/`).

### Adding Custom Bindings

1.  **Create Folder & Files:** Add a folder `your_binding_type` inside `personal_bindings/`. Inside it, create:
    *   `__init__.py`: Contains your binding class implementation.
    *   `binding_card.yaml`: Defines metadata and the ConfigGuard schema (`instance_schema`) for configuration files of this binding type.
2.  **Implement Binding Class:** In `__init__.py`, define a class inheriting from `lollms_server.core.bindings.Binding`. Implement required methods:
    *   `binding_type_name`: Class attribute matching the `type_name` in `binding_card.yaml`.
    *   Implement abstract methods like `list_available_models`, `get_supported_*_modalities`, `load_model`, `unload_model`, `generate`.
    *   Handle multimodal input (`multimodal_data`) in `generate`/`generate_stream`.
    *   Optionally implement `generate_stream`, `health_check`, `tokenize`, `detokenize`, `get_current_model_info`.
3.  **Define Binding Card:** In `binding_card.yaml`, specify:
    *   `type_name`: Matches the class attribute.
    *   `display_name`, `version`, `author`, `description`, `requirements`.
    *   `instance_schema`: A ConfigGuard schema defining settings for instances of this binding (e.g., API keys, paths, specific parameters).
4.  **Configure an Instance:**
    *   Create a file like `my_new_binding_instance.yaml` inside `lollms_configs/bindings/`.
    *   Add `type: your_binding_type` and other settings defined in your `instance_schema`.
    *   Add an entry in your main config's `bindings_map`: `my_new_binding_instance: your_binding_type`.
5.  **Dependencies:** Ensure any required Python libraries (listed in `binding_card.yaml`) are installed in the `venv`.
6.  **Restart Server:** It will discover the new binding type and attempt to load your configured instance.

### Adding Custom Personalities

**1. Simple Personality (Config-based):**

*   **Create Folder:** E.g., `personal_personalities/my_story_writer`.
*   **Create `config.yaml`:** Define `name`, `author`, `version`, `personality_description`, `personality_conditioning`, and optional fields (see `PersonalityConfig` model in `core/personalities.py` or example `zoos/personalities/lollms/config.yaml`).
*   **(Optional) Add `assets/icon.png`.**
*   **Restart Server.**

**2. Scripted Personality (Agentic):**

*   **Follow Simple Steps:** Create folder, `config.yaml`.
*   **Create Script:** Add `scripts/workflow.py` (or other name) inside the personality folder.
*   **Define `run_workflow`:** Implement the core logic in the script:
    ```python
    # scripts/workflow.py
    import ascii_colors as logging
    from typing import Any, Dict, Optional, List, Union, AsyncGenerator
    # Import necessary types
    from lollms_server.api.models import InputData, OutputData

    async def run_workflow(prompt: str, params: Dict, context: Dict) -> Union[str, Dict, List[Dict], AsyncGenerator[Dict, None]]:
        # context contains: 'binding', 'personality', 'config', 'function_manager',
        # 'binding_manager', 'resource_manager', 'input_data', 'request_info'
        binding = context.get('binding')
        input_data: List[InputData] = context.get('input_data', [])
        # ... access other context items ...
        logger = logging.getLogger(__name__)

        # --- Your Workflow Logic ---
        # Example: Process images, call binding, call functions...
        text_response = f"Workflow received: '{prompt}' and {len(input_data)} input items."

        # Return List[OutputData]-like dicts, a simple string, or an async generator
        return [{"type": "text", "data": text_response}]
    ```
*   **Set `script_path`:** In `config.yaml`, add `script_path: scripts/workflow.py`.
*   **(Optional) Configure:** Use main config's `personalities_config` section to disable or set retries for this personality by its folder name.
*   **Restart Server.**

### Adding Custom Functions

1.  Create `my_functions.py` in your configured `functions_folder` (e.g., `personal_functions/`).
2.  Define `async def my_async_func(arg1: str, arg2: int) -> dict: ...`. Functions *must* be asynchronous.
3.  Restart server. Function is discovered as `my_functions.my_async_func`.
4.  Call from scripted personalities using `context['function_manager'].execute_function(...)`.

## Web UI

If `webui.enable_ui = true` in your main configuration file:

1.  **Build the UI (One-time setup):**
    ```bash
    # Navigate to the webui directory from the project root
    cd webui
    # Install Node.js dependencies
    npm install
    # Build the static assets for production
    npm run build
    # Go back to the project root
    cd ..
    ```
2.  **Run Server:** Start `lollms_server` as usual (e.g., `./run.sh`).
3.  **Access:** Open your browser to the server's address (e.g., `http://localhost:9601`).

The server will serve the built `index.html` and assets from `webui/dist/`.

## Client Examples

The `client_examples/` directory (mentioned in the original README, though not present in the provided structure - *assuming it exists*) contains scripts demonstrating API interaction. Refer to that directory for examples of making requests (including streaming and multimodal).

## Development

*   **Testing:** A test suite using `pytest` is included in the `tests/` directory.
    1.  Install development dependencies: `pip install -e .[dev]` (ensure venv is active).
    2.  Run tests from the project root: `pytest`
*   **Linting/Formatting:** Uses `ruff` for linting and `black` for formatting. Configure your IDE or run manually:
    ```bash
    (venv) ruff check .
    (venv) black .
    ```
*   **Contributing:** Contributions are welcome! Please follow standard GitHub practices (fork, feature branch, pull request). Ensure tests pass and code is formatted/linted.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Copyright 2025 ParisNeo

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Disclaimer

This software is provided "as is" without warranty of any kind. Features involving code execution (like the `python_builder_executor` personality) are experimental and **extremely dangerous**. Use them only in isolated, secure environments and at your own risk. The developers are not responsible for any damage or data loss caused by the use of this software. Always review code generated by LLMs before execution.