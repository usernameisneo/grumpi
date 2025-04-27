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

`lollms_server` is a versatile, asynchronous, multi-modal generation server designed to work with the [LOLLMS](https://github.com/ParisNeo/lollms) ecosystem. It provides a unified API endpoint (`/generate`) to interact with various text, image, audio, video, and music generation backends (bindings) using configurable personalities and optional function execution. It aims to provide a robust and extensible platform for experimenting with and deploying multi-modal AI workflows.

## Key Features

*   **Multi-Modal Generation:** Supports Text-to-Text (TTT), Text-to-Image (TTI), Text-to-Video (TTV), Text-to-Music (TTM), Text-to-Speech (TTS), Speech-to-Text (STT), Image-to-Image (I2I), Audio-to-Audio, and more through extensible bindings.
*   **Binding Agnostic:** Easily integrate different AI backends (local or remote APIs) like Ollama, OpenAI (including compatible APIs like Groq, Together.ai), DALL-E, Gemini, llama-cpp-python, Hugging Face Transformers, etc.
*   **LOLLMS Personality System:** Loads and utilizes standard LOLLMS personalities, including configuration-based prompts (`config.yaml`) and scripted workflows (`workflow.py`).
*   **Multimodal Input API:** Accepts complex inputs via the `/generate` endpoint's `input_data` list, allowing combinations of text, images, audio, etc., each with assigned roles (e.g., `user_prompt`, `input_image`, `controlnet_image`).
*   **Asynchronous & Concurrent:** Built with FastAPI and `asyncio` for high throughput and responsiveness, handling multiple requests concurrently.
*   **Resource Management:** Implements basic resource management (GPU semaphore) to prevent overloading during model loading or intensive generation tasks. Requests are queued if resources are unavailable.
*   **Configuration Driven:** Server settings, paths, default models/bindings, security keys, logging levels, UI options, and personality configurations are managed via a central `config.toml` file.
*   **Dynamic Discovery:** Automatically discovers personalities, bindings, and functions placed in user-configured folders upon startup.
*   **Secure API:** Uses API key authentication (`X-API-Key` header) for all API endpoints.
*   **Streaming Support:** Provides Server-Sent Events (SSE) for real-time streaming of Text-to-Text (TTT) and potentially other compatible modalities.
*   **Extensible Function Calling:** Allows defining custom Python functions that can be discovered and executed within scripted personality workflows.
*   **(Optional) Integrated Web UI:** Includes a basic Vue.js frontend served directly by FastAPI for easy interaction and testing (configurable).
*   **Cross-Platform:** Includes installation scripts for Linux, macOS, and Windows.
*   **Tokenizer Utilities:** API endpoints to tokenize, detokenize, and count tokens using the active binding's model.
*   **Model Information:** API endpoint to retrieve context size and other details about the currently loaded model for a binding.

## Core Concepts

*   **Binding:** A Python class that interfaces with a specific AI model backend (e.g., OpenAI API, local Ollama server, `llama-cpp-python`). Bindings handle model loading, parameter translation, generation calls, and capability reporting. You configure *instances* of bindings in `config.toml`.
*   **Personality:** Defines the behavior, instructions, and context for the AI, similar to custom GPTs or system prompts. They consist of a `config.yaml` file and optional assets or scripts. Scripted personalities (`workflow.py`) allow for complex, agentic behavior.
*   **Function:** A custom Python function placed in the functions folder, discoverable by the server and callable from scripted personalities to extend capabilities (e.g., calling external tools, performing calculations).
*   **Multimodal Input (`input_data`):** The `/generate` endpoint accepts a list of input objects, each specifying its `type` (text, image, audio, etc.), `role` (how it should be used, e.g., `user_prompt`, `input_image`, `mask_image`), and the actual `data` (text content, base64 string, URL).
*   **API Key:** A secret key required to authenticate requests to the server's API endpoints. Defined in `config.toml`.

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

    This script will:
    *   Check your Python version.
    *   Create a Python virtual environment named `venv`.
    *   Install core Python packages (`fastapi`, `uvicorn`, `pydantic`, `toml`, `pyyaml`, `pipmaster`, `ascii_colors`) from `requirements.txt` into `venv`.
    *   Copy `config.toml.example` to `config.toml` if it doesn't exist.

3.  **Activate the Virtual Environment:**
    *   **Linux/macOS:** `source venv/bin/activate`
    *   **Windows Cmd:** `venv\Scripts\activate.bat`
    *   **Windows PowerShell:** `.\venv\Scripts\Activate.ps1` (You might need to adjust execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)

    You should see `(venv)` at the beginning of your terminal prompt. **All subsequent commands (pip installs, running the server) must be done with the environment active.**

4.  **Configure `config.toml`:**
    *   **This is the most important step!** Open `config.toml` in a text editor. Review `config.toml.example` for detailed comments on all options. See the "Server Configuration" section below for key areas.
    *   **API Keys:** Add at least one strong, random key to `[security].allowed_api_keys`.
    *   **Bindings:** Configure instances under `[bindings]` for the AI backends you want to use (e.g., `[bindings.default_ollama]`, `[bindings.my_openai]`). Set the correct `type` and any required parameters like API keys or hosts.
    *   **Paths:** Verify the `[paths]` section points to the correct locations for your models, personalities, etc.

5.  **(Optional) Install Binding-Specific Dependencies:**
    Install Python packages required by the specific bindings you enabled in `config.toml`. Ensure `venv` is active. Check the `requirements` list within each binding's `get_binding_config()` method (visible via `/docs` or in the source code).
    ```bash
    # Example: Install packages for the included example bindings
    (venv) pip install ollama openai pillow google-generativeai requests llama-cpp-python

    # Example: If using a transformers-based binding (check its requirements)
    # (venv) pip install transformers torch accelerate

    # Example: If using the python_builder_executor personality
    # (venv) pip install pygame numpy # Or other libraries the generated code might need
    ```

## Running the Server

1.  **Activate Environment:** Make sure your virtual environment (`venv`) is activated.
2.  **Run using Script:**
    *   **Linux/macOS:** `./run.sh`
    *   **Windows:** `.\run.bat`
    These scripts attempt to read the host/port from `config.toml` and run `uvicorn`.
3.  **Run Manually (Alternative):**
    ```bash
    # Reads host/port/log-level from config.toml via main.py's logic
    (venv) python lollms_server/main.py
    ```
    Or directly with Uvicorn:
    ```bash
    # Replace host/port if different from config defaults
    (venv) uvicorn lollms_server.main:app --host 0.0.0.0 --port 9600 --workers 1
    ```
4.  **Access:**
    *   **API Docs (Swagger):** `http://localhost:9600/docs` (or your configured host/port). Use the "Authorize" button here to test endpoints with your API key.
    *   **Web UI (if enabled):** `http://localhost:9600`

## Server Configuration (`config.toml`)

The `config.toml` file controls server behavior. Review `config.toml.example` for comments.

*   **`[server]`**: `host`, `port`, `allowed_origins` (for CORS, crucial for web UIs).
*   **`[paths]`**: Directories for discovering `personalities`, `bindings`, `functions`, `models`, and the built-in `examples`. Paths are relative to `config.toml` unless absolute.
*   **`[security]`**: **`allowed_api_keys`**: **Required list** of secret keys clients must send in the `X-API-Key` header.
*   **`[defaults]`**: Fallback `ttt_binding`, `ttt_model`, `tti_binding`, etc., if not specified in API requests. Values must match instance names defined in `[bindings]`. Also includes `default_context_size` and `default_max_output_tokens`.
*   **`[webui]` (Optional)**: `enable_ui = true` to serve the static Vue.js app from `webui/dist/`. Requires building the UI first.
*   **`[logging]` (Optional)**: `log_level` ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
*   **`[bindings]`**: **Required section** to define instances of bindings.
    *   **`[bindings.INSTANCE_NAME]`**: Choose a unique name (e.g., `default_ollama`, `openai_gpt4o`, `my_sdxl_api`).
    *   **`type = "binding_type_name"`**: **Required.** Must match the `binding_type_name` attribute of the Binding class (e.g., `ollama_binding`, `openai_binding`, `dalle_binding`, `llamacpp_binding`).
    *   **Other Keys:** Parameters specific to that binding `type` (e.g., `host`, `api_key`, `models_folder`, `n_gpu_layers`). Check the binding's documentation or `get_binding_config()`.
*   **`[personalities_config]` (Optional)**: Override settings for specific personalities by their folder name.
    *   **`[personalities_config.FOLDER_NAME]`**: Key must match the personality's directory name.
    *   `enabled = false`: Disable loading this personality.
    *   `max_execution_retries = int`: Max retries for scripted workflow errors (default 1).
*   **`[resource_manager]`**: Controls access to limited resources (primarily GPU).
    *   `gpu_strategy`: `"semaphore"` (limit N concurrent tasks) or `"simple_lock"` (limit 1).
    *   `gpu_limit`: Max concurrent tasks for `semaphore`.
    *   `queue_timeout`: Seconds to wait for resource before failing request.

## API Usage

Interact with the server via its REST API. Use the Swagger UI at `/docs` for detailed endpoint information and testing.

### Authentication

All `/api/v1/*` endpoints require authentication. Include a valid API key from your `config.toml` in the `X-API-Key` HTTP header with every request.

### Key Endpoints

#### Listing

*   **`GET /api/v1/list_bindings`**: Lists discovered binding types and configured instances.
*   **`GET /api/v1/list_personalities`**: Lists loaded and enabled personalities.
*   **`GET /api/v1/list_functions`**: Lists discovered custom functions.
*   **`GET /api/v1/list_available_models/{binding_name}`**: Asks a specific binding instance (e.g., `default_ollama`) to list models it can access. Returns detailed model info including capabilities.
*   **`GET /api/v1/list_models`**: Lists models found in the configured `models_folder` subdirectories (basic file scan).

#### Generation

*   **`POST /api/v1/generate`**: The main endpoint for triggering generation tasks (TTT, TTI, etc.). See details below.

#### Utilities (New!)

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
    *   **Response:** `{"binding_name": "string", "model_name": "string|null", "context_size": int|null, "max_output_tokens": int|null, "supports_vision": bool, "supports_audio": bool, "details": {}}`
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
  // (Optional) Deprecated field for simple text prompts. Use input_data instead.
  // "text_prompt": "string | null",

  // --- Generation Control (Optional - Uses defaults if omitted) ---
  "personality": "string | null", // Name of loaded personality to use
  "binding_name": "string | null", // Specific binding instance name (from config.toml)
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
  "binding_name": "my_dalle_binding", // Must be configured in config.toml
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
  "binding_name": "gemini_vision" // Must be configured & support vision
}
```

**Response:**

*   **Non-streaming:** `application/json` containing the result (e.g., `{"output": {"text": "..."}}` for TTT, `{"output": {"image_base64": "...", "mime_type": "..."}}` for TTI).
*   **Streaming:** `text/event-stream` with Server-Sent Events (SSE). Each event is `data: {"type": "...", "content": ..., "metadata": ...}\n\n`. Types include `chunk`, `info`, `error`, `final`.

## Extending the Server

Add your own components by placing Python files/folders in the directories specified in `config.toml [paths]`.

### Adding Custom Bindings

1.  **Create File:** Add `your_binding_name.py` to `personal_bindings/`.
2.  **Inherit & Implement:** Define a class inheriting from `lollms_server.core.bindings.Binding`. Implement required methods:
    *   `binding_type_name`: Class attribute, unique string identifier (e.g., `"my_api_binding"`). Used in `config.toml`.
    *   `get_binding_config()`: Class method returning dict with metadata (`type_name`, `description`, `requirements`, `config_template`).
    *   `get_supported_input_modalities() -> List[str]`: Return list like `['text', 'image']`.
    *   `get_supported_output_modalities() -> List[str]`: Return list like `['text']` or `['image']`.
    *   `async list_available_models() -> List[Dict]`: List models usable by this binding instance. Include standard keys (`name`, `size`, `supports_vision`, etc.) and put others in `details`.
    *   `async load_model(model_name: str) -> bool`: Load/prepare the model. Use `self.resource_manager` for GPU access if needed.
    *   `async unload_model() -> bool`: Release model resources.
    *   `async generate(prompt: str, params: Dict, request_info: Dict, multimodal_data: List[InputData]) -> Union[str, Dict]`: Core generation logic. Process `multimodal_data` based on `get_supported_input_modalities`. Return `{"text": "..."}` or `{"image_base64": "...", ...}` etc.
    *   `(Optional)` `async generate_stream(...) -> AsyncGenerator[Dict, None]`: Implement for streaming. Yield `StreamChunk`-like dicts.
    *   `(Optional)` `async health_check() -> Tuple[bool, str]`: Check connectivity/status.
    *   `(Optional)` `async tokenize(text, add_bos, add_eos) -> List[int]`: Implement if tokenization is supported.
    *   `(Optional)` `async detokenize(tokens) -> str`: Implement if detokenization is supported.
    *   `async get_current_model_info() -> Dict[str, Any]`: Return info about the loaded model.
3.  **Configure:** Add an instance in `config.toml`:
    ```toml
    [bindings.my_instance_name]
    type = "my_api_binding" # Matches binding_type_name
    api_endpoint = "https://example.com/api"
    api_secret = "..."
    ```
4.  **Dependencies:** Ensure any required libraries are installed in the `venv`.
5.  **Restart Server:** It will discover and instantiate the binding.

### Adding Custom Personalities

**1. Simple Personality (Config-based):**

*   **Create Folder:** E.g., `personal_personalities/my_story_writer`.
*   **Create `config.yaml`:** Define `name`, `author`, `version`, `personality_description`, `personality_conditioning`, and optional fields (see `zoos/personalities/lollms/config.yaml`).
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
    from lollms_server.utils.helpers import extract_code_blocks
    from lollms_server.api.models import InputData # Access InputData

    async def run_workflow(prompt: str, params: Dict, context: Dict) -> Union[str, Dict, AsyncGenerator[Dict, None]]:
        # context contains: 'binding', 'personality', 'config', 'function_manager',
        # 'resource_manager', 'input_data', 'request_info'
        binding = context.get('binding')
        input_data: List[InputData] = context.get('input_data', [])
        function_manager = context.get('function_manager')
        logger = logging.getLogger(__name__)
        logger.info(f"Workflow received {len(input_data)} input items.")

        # Example: Call binding, call function, process images...
        # image_items = [item for item in input_data if item.type == 'image']
        # response = await binding.generate(prompt, params, context['request_info'], input_data)
        # success, func_result = await function_manager.execute_function("my_module.my_func", {})

        # Return string, dict, or async generator for streaming
        return f"Workflow executed for: {prompt}"
    ```
*   **Set `script_path`:** In `config.yaml`, add `script_path: scripts/workflow.py`.
*   **(Optional) Configure:** Use `config.toml [personalities_config]` to disable or set retries.
*   **Restart Server.**

### Adding Custom Functions

1.  Create `my_functions.py` in `personal_functions/`.
2.  Define `async def my_async_func(arg1: str, arg2: int) -> dict: ...`.
3.  Restart server. Function is discovered as `my_functions.my_async_func`.
4.  Call from scripted personalities using `function_manager.execute_function(...)`.

## Web UI

If `[webui].enable_ui = true` in `config.toml`:

1.  **Build the UI (One-time setup):**
    ```bash
    cd webui
    npm install  # Install dependencies
    npm run build # Create the 'dist' folder
    cd ..        # Go back to project root
    ```
2.  **Run Server:** Start `lollms_server` as usual.
3.  **Access:** Open your browser to the server's address (e.g., `http://localhost:9600`).

The server will serve the built `index.html` and assets from `webui/dist/`.

## Client Examples

The `client_examples/` directory contains various Python scripts and a basic web UI demonstrating how to interact with the server's API:

*   `simple_client.py`: Basic requests for listing resources and non-streaming TTT.
*   `streaming_client.py`: Demonstrates handling Server-Sent Events for TTT streaming.
*   `image_client.py`: Shows how to send TTI requests and save the resulting image.
*   `chat_app.py`: An interactive console chat application with history and settings.
*   `lord_of_discord.py`: A Discord bot integrating with the server.
*   `web/`: A simple HTML/CSS/JS web client (requires CORS configuration on the server).

See the individual `README.md` files within each example's directory for specific instructions.

## Development

*   **Testing:** A test suite using `pytest` is included in the `tests/` directory. Run tests (after installing dev dependencies: `pip install -e .[dev]`) using the `pytest` command in the project root.
*   **Linting/Formatting:** Uses `ruff` for linting and `black` for formatting. Configure your IDE or run manually:
    ```bash
    (venv) ruff check .
    (venv) black .
    ```
*   **Contributing:** Contributions are welcome! Please follow standard GitHub practices (fork, feature branch, pull request). Ensure tests pass and code is formatted.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Copyright 2025 ParisNeo

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Disclaimer

This software is provided "as is" without warranty of any kind. Features involving code execution (like the `python_builder_executor` personality) are experimental and **extremely dangerous**. Use them only in isolated, secure environments and at your own risk. The developers are not responsible for any damage or data loss caused by the use of this software. Always review code generated by LLMs before execution.
