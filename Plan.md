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


#### **NEW Phase 5: Multimodal Input - API & Core Models**

*   [ ] **Define `InputData` Model:** Create a Pydantic model (`api/models.py`) to represent a single piece of input data (e.g., `{ "type": "image", "role": "input_image", "data": "base64_string", "mime_type": "image/png" }`). Define standard roles ('user_prompt', 'input_image', 'input_audio', 'controlnet_image', 'mask_image', 'reference_image', etc.).
*   [ ] **Modify `GenerateRequest`:** Update `api/models.py` to replace/supplement the `prompt: str` with a structure that includes the main text prompt *and* an optional list of `InputData` objects (`input_data: Optional[List[InputData]] = None`). Ensure the primary text prompt is still easily accessible.
*   [ ] **Update `Binding` Signature:** Modify the `Binding.generate` and `Binding.generate_stream` method signatures (`core/bindings.py`) to accept the structured input data. Proposal: `generate(self, request: GenerateRequest, params: Dict[str, Any]) -> ...` or `generate(self, prompt: str, input_data: Optional[List[InputData]], params: Dict[str, Any], request_info: Dict[str, Any]) -> ...`. *Let's refine the signature during implementation, leaning towards passing necessary parts rather than the whole request object.*
*   [ ] **Add Binding Capabilities:** Introduce methods in the `Binding` base class for bindings to declare their input/output capabilities (e.g., `supports_input_type(type: str, role: str) -> bool`, `get_supported_modalities() -> Dict`).

#### **NEW Phase 6: Multimodal Input - Generation Logic**

*   [ ] **Update `process_generation_request` (`core/generation.py`):**
    *   Parse the new `input_data` field from the incoming `GenerateRequest`.
    *   Perform basic validation (e.g., check if required data for the `generation_type` is present).
    *   Determine the target binding based on request/defaults.
    *   **Crucially:** Check if the selected binding *supports* the provided input data types/roles using the new capability methods. Raise HTTP errors if incompatible.
    *   Pass the validated/relevant `input_data` (or processed representation) to the binding's `generate`/`generate_stream` method.

#### **NEW Phase 7: Multimodal Input - Binding Implementation**

*   [ ] **Refactor Example Bindings (Ollama, OpenAI, Gemini, Dummy):**
    *   Implement the new capability declaration methods.
    *   Modify their `generate`/`generate_stream` methods to handle `input_data` list.
    *   **Gemini/Ollama/OpenAI (Vision):** Implement logic to process `InputData` with `type='image'` and include it in the API call payload alongside the text prompt. Handle base64 decoding.
    *   **Dummy Binding:** Enhance to simulate handling image inputs based on its configured `mode`.
*   [ ] **Update `list_available_models`:** Bindings should ideally include information about a model's multimodal capabilities (e.g., vision-capable, audio-input) in the details returned by `list_available_models`.

#### **NEW Phase 8: Audio Modality Support**

*   [ ] **Define Audio Data Structures:** Potentially add specific models or roles for audio input/output in `api/models.py`.
*   [ ] **Add Example Bindings:**
    *   **TTS Binding:** Create a simple Text-to-Speech binding (e.g., using `pyttsx3` for local testing or an API like ElevenLabs). It will take text input and return `{"audio_base64": "...", "mime_type": "audio/wav"}`.
    *   **TTM Binding (Dummy):** Create a dummy Text-to-Music binding returning placeholder audio data.
    *   **(Future) STT Binding:** Speech-to-Text would take audio input and return text.

#### **NEW Phase 9: Advanced Image Workflows**

*   [ ] **Implement Specific Roles:** Define standard roles like `controlnet_image`, `mask_image`, `reference_image`.
*   [ ] **Add Stable Diffusion Binding (Example):** Create a binding (e.g., using `diffusers` or an API) that demonstrates handling:
    *   Text-to-Image (basic).
    *   Image-to-Image (using `role='input_image'`).
    *   Inpainting (using `role='input_image'` and `role='mask_image'`).
    *   ControlNet (using `role='input_image'` and `role='controlnet_image'`).

#### **NEW Phase 10: Personality & Client Updates**

*   [ ] **Scripted Personalities:** Update `core/generation.py` to ensure the full `input_data` list is available within the `context` dictionary passed to `run_workflow`. Update example scripted personalities to show how to access and use image/audio data.
*   [ ] **Client Examples:** Refactor Python and Web client examples (`client_examples/`) to demonstrate:
    *   Sending requests with image/audio data (base64).
    *   Handling multimodal responses (displaying images, playing audio).

#### **Phase 11 (was 7): Testing, Documentation & Refinement**

*   [x] Write `README.md`.
*   [x] Add docstrings and type hints.
*   [x] Perform manual testing.
*   [ ] **Update Tests:** Add unit/integration tests specifically for multimodal input handling, binding capabilities, and new modalities.
*   [x] Refine error handling.
*   [x] Create `config.toml.example`.
*   [ ] **Update Documentation:** Explain the new multimodal API, how to implement multimodal bindings, and how to use the features.
