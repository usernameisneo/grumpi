# lollms_server - Plan

## 1. Project Goal

To create a highly configurable, asynchronous, multi-modal (Text, Image, Video, Music) generation server named `lollms_server`. It aims to be compatible with the `lollms` personality ecosystem, support various generation bindings (llamacpp, HF, Ollama, OpenAI, vLLM etc.), manage resources effectively (like VRAM), and provide a secure, discoverable API using FastAPI.

## 2. Core Features

*   **Multi-Modal Generation:** Support Text-to-Text (TTT), Text-to-Image (TTI), Text-to-Video (TTV), Text-to-Music (TTM).
*   **Binding Agnostic:** Abstract the generation backend (binding) allowing users to plug in different implementations.
*   **Personality System:** Load and utilize `lollms` personalities (both scripted and non-scripted configurations).
*   **Function Calling:** Support for extending generation workflows with custom Python functions.
*   **Asynchronous & Concurrent:** Utilize `asyncio` and FastAPI with multiple workers for high throughput.
*   **Configuration Driven:** Server behavior, paths, default models/bindings, and users configured via `config.toml`.
*   **Dynamic Discovery:** Automatically discover personalities, bindings, and functions placed in configured folders.
*   **API:**
    *   Secure endpoints using API key authentication.
    *   Endpoints to list available personalities, bindings, functions, and models.
    *   `/generate` endpoint for triggering generation tasks.
    *   Support for specific model/binding selection per request, falling back to defaults.
    *   Streaming support for TTT generation.
    *   Base64 encoding for non-textual outputs (images, video, music).
*   **Resource Management:**
    *   On-demand model loading.
    *   Queuing mechanism for requests when resources (e.g., VRAM) are unavailable.
    *   Timeout for queued requests.
*   **Extensibility:** Users can easily add their own bindings, personalities, and functions.
*   **Ease of Use:** Simple installation and clear documentation.

## 3. Technology Stack

*   **Language:** Python 3.9+
*   **Web Framework:** FastAPI
*   **Async:** asyncio
*   **Configuration:** TOML (using `toml` library)
*   **Data Validation:** Pydantic
*   **Web Server:** Uvicorn
*   **Potential Binding Libraries (Examples):** `requests`, `openai`, `huggingface_hub`, `transformers`, `diffusers`, `llama-cpp-python`, `ollama`, `vllm` (dependencies managed per-binding)
*   **License:** Apache 2.0

## 4. Project Structure (High-Level)

```
lollms_server/
├── .github/             # GitHub Actions, templates (optional)
├── .gitignore
├── LICENSE
├── Plan.md              # This file
├── README.md
├── config.toml.example  # Example configuration
├── zoos/            # Example implementations
│   ├── bindings/
│   ├── functions/
│   └── personalities/
├── lollms_server/       # Source code root
│   ├── __init__.py
│   ├── api/             # FastAPI endpoints and models
│   │   ├── __init__.py
│   │   ├── endpoints.py # API route definitions
│   │   └── models.py    # Pydantic request/response models
│   ├── core/            # Core logic
│   │   ├── __init__.py
│   │   ├── bindings.py  # Binding abstraction, manager
│   │   ├── config.py    # Configuration loading/validation
│   │   ├── functions.py # Function abstraction, manager
│   │   ├── generation.py# Generation orchestration logic
│   │   ├── personalities.py # Personality handling, manager
│   │   ├── resource_manager.py # Resource locking/queuing
│   │   └── security.py  # Authentication handling
│   ├── utils/           # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py
│   │   └── helpers.py
│   └── main.py          # FastAPI application entry point
├── pyproject.toml       # Project metadata, build config (optional but recommended)
├── requirements.txt     # Core dependencies
└── scripts/             # Helper scripts (e.g., run.sh)
```

## 5. Development Phases

### Phase 1: Foundation & Core Setup

*   [x] Initialize project structure (`mkdir`, `git init`).
*   [x] Add `LICENSE` (Apache 2.0).
*   [x] Create basic `.gitignore`.
*   [x] Define `requirements.txt` (FastAPI, Uvicorn, Pydantic, Toml).
*   [x] Implement `config.py` using Pydantic to load and validate `config.toml`.
*   [x] Set up basic FastAPI app in `main.py`.
*   [x] Implement basic API key authentication in `core/security.py` and integrate it.

### Phase 2: Abstraction Layers

*   [x] Define base `Binding` class in `core/bindings.py` with abstract methods (`load_model`, `generate`, `get_resource_requirements`).
*   [x] Implement `BindingManager` in `core/bindings.py` to discover and load binding *classes* from configured folders.
*   [x] Define `Personality` representation and `PersonalityManager` in `core/personalities.py` to scan and load personality configurations.
*   [x] Define `Function` representation and `FunctionManager` in `core/functions.py` to discover and load functions.

### Phase 3: API Endpoints & Basic Generation Flow

*   [x] Define API request/response models in `api/models.py`.
*   [x] Implement listing endpoints (`/list_bindings`, `/list_personalities`, `/list_functions`, `/list_models`) in `api/endpoints.py`.
*   [x] Implement the basic structure of the `/generate` endpoint.
*   [x] Implement the core generation orchestration logic in `core/generation.py` (finding personality, selecting binding, calling binding's generate method - *without* resource management initially).

### Phase 4: Resource Management

*   [x] Implement `ResourceManager` in `core/resource_manager.py` using `asyncio.Lock` or `Semaphore` for basic resource control (e.g., a simple GPU lock).
*   [x] Implement queuing logic (`asyncio.Queue`) within the `ResourceManager` or `BindingManager` for requests waiting for resources.
*   [x] Integrate resource acquisition/release into the `generate` flow and `Binding` methods (especially `load_model`).
*   [x] Add timeout mechanism for queued requests.

### Phase 5: Binding Implementations & Examples

*   [x] Create example `DummyBinding` (TTT, TTI) for testing.
*   [x] Implement wrappers/examples for key bindings (start with OpenAI, Ollama, maybe a placeholder for llamacpp/HF). Place these in `zoos/bindings/`.
*   [x] Refine the `generate` endpoint to handle different modalities and base64 encoding.
*   [x] Implement TTT streaming using `StreamingResponse`.

### Phase 6: Personality & Function Integration

*   [x] Refine `PersonalityManager` to differentiate between scripted (`run_workflow`) and non-scripted personalities.
*   [x] Update `core/generation.py` to execute `run_workflow` for scripted personalities.
*   [x] Integrate function execution into the generation flow if functions are specified.
*   [x] Add example personalities and functions in `zoos/`.

### Phase 7: Testing, Documentation & Refinement

*   [x] Write `README.md` with installation, configuration, and usage instructions.
*   [x] Add docstrings and type hints throughout the code.
*   [x] Perform manual testing of different generation scenarios.
*   [ ] (Optional) Add automated tests (unit/integration).
*   [x] Refine error handling and logging.
*   [x] Create `config.toml.example`.
*   [x] Prepare for hosting on GitHub (ensure `LICENSE`, `README.md` are ready).

## 6. Key Considerations & Challenges

*   **Binding Complexity:** Each binding has unique setup, model loading, and generation APIs. The abstraction needs to be flexible.
*   **Resource Management:** Accurately tracking VRAM or other resource usage across different bindings can be complex. Start simple (e.g., one model loaded per GPU at a time).
*   **Dependency Management:** Bindings often have heavy or conflicting dependencies. Consider strategies like optional installs (`pip install lollms_server[huggingface]`) or guiding users to manage environments. For user-provided bindings, they manage their own dependencies.
*   **Error Handling:** Robust error handling across async tasks, binding failures, resource timeouts, etc., is crucial.
*   **Security:** The initial API key method is basic. For production, consider more robust methods if needed.
*   **Scripted Personalities:** Defining a clear and secure execution context for `run_workflow` is important.
