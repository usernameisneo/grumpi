## [0.3.1] - 2025-05-01

### Changed

*   **Default Server Port:** Updated the default server port from `9600` to `9601`. This change helps prevent potential conflicts with the default port commonly used by the `lollms-webui` application.
    *   The primary default value is updated in `lollms_server/core/config.py` (within the `MAIN_SCHEMA` definition).
    *   The interactive `configuration_wizard.py` now suggests `9601` as the default.
    *   Example configurations and documentation should reflect this new default. Users upgrading should check their existing configuration files if they were relying on the previous default port (`9600`).

## [0.3.0] - 2025-05-01

### Added

*   **Configuration System Overhaul (ConfigGuard Integration):**
    *   Replaced previous configuration loading with `ConfigGuard` (`lollms_server/core/config.py`).
    *   Introduced schema-driven configuration (`MAIN_SCHEMA`) for validation, defaults, versioning, and type safety.
    *   Added support for multiple main config file formats (YAML, TOML, JSON, SQLite).
    *   Created an interactive Configuration Wizard (`configuration_wizard.py`) for guided setup, presets, and binding instance creation.
    *   Separated binding instance configurations into individual files (e.g., `.yaml`) in `lollms_configs/bindings/`.
    *   Implemented optional configuration encryption via `ConfigGuard`.
    *   Added `personalities_config` section in main config for enabling/disabling specific personalities.
*   **Refined Binding System:**
    *   Introduced `binding_card.yaml` for binding metadata and instance configuration schemas (`zoos/bindings/*/`).
    *   Refactored `BindingManager` (`core/bindings.py`) for discovery and instantiation based on cards and instance files.
    *   Added standardized binding methods: `list_available_models`, `get_supported_*_modalities`, `get_current_model_info`, `tokenize`, `detokenize`.
    *   Included example bindings: `dalle_binding`, `diffusers_binding`, `dummy_binding`, `gemini_binding`, `hf_binding`, `llamacpp_binding`, `ollama_binding`, `openai_binding`.
    *   Added support for user-defined bindings in `personal_bindings/`.
*   **Enhanced Personality System:**
    *   Introduced `PersonalityConfig` Pydantic model (`core/personalities.py`) for validating `config.yaml`.
    *   Formalized scripted personality execution via `run_workflow` in `scripts/workflow.py`.
    *   Refactored `PersonalityManager` (`core/personalities.py`) for discovery, loading, and script execution.
    *   Added example personalities: `artbot`, `lollms`, `python_builder_executor`, `scripted_example`.
    *   Added support for user-defined personalities in `personal_personalities/`.
*   **Function Calling System:**
    *   Added `FunctionManager` (`core/functions.py`) for discovering and managing custom Python functions.
    *   Enabled execution of functions from scripted personalities (`personal_functions/`, `zoos/functions/`).
*   **Multimodal Input/Output Handling:**
    *   Updated `/generate` endpoint and `GenerateRequest` model (`api/models.py`) to use `input_data` list for multimodal inputs (text, image, etc.) with defined roles.
    *   Added `OutputData` model for standardizing multimodal responses.
    *   Updated bindings (Ollama, Gemini, OpenAI) to handle multimodal inputs.
    *   Added `<think>` tag parsing helper (`utils/helpers.py`) and integrated it into responses.
*   **Optional Integrated Web UI:**
    *   Added `webui/` directory with Vue.js frontend structure.
    *   Added Vite configuration (`webui/vite.config.ts`).
    *   Added server logic (`main.py`) to serve the static UI if `webui.enable_ui` is true.
*   **API Enhancements:**
    *   Added `/list_available_models/{binding_instance_name}` endpoint.
    *   Added `/get_model_info/{binding_instance_name}` endpoint.
    *   Added `/tokenize`, `/detokenize`, `/count_tokens` utility endpoints.
    *   Refined API response models (`ModelInfo`, `PersonalityInfo`, `GenerateResponse`, `StreamChunk`, etc.).
*   **Installation & Setup Improvements:**
    *   Added comprehensive installation script (`install_core.py`) handling venv, core deps, wizard trigger, optional deps menu, and binding instance management CLI.
    *   Added wrapper scripts `install.sh`/`.bat` and `run.sh`/`.bat`.
*   **Utilities & Dependencies:**
    *   Integrated `pipmaster` via `dependency_manager.py` for optional dependencies.
    *   Integrated `ascii_colors` for enhanced logging and interactive CLI elements.
    *   Added helper functions in `utils/file_utils.py` and `utils/helpers.py`.
    *   Updated `requirements.txt` and `pyproject.toml`.
*   **Discord Bot Integration:**
    *   Added configuration files (`bot_config.json`, `discord_settings.json`) suggesting Discord bot capabilities.
*   **Documentation & Planning:**
    *   Added `Plan.md` outlining project goals.
    *   Added library documentation in `libraries_docs/`.

### Changed

*   **Configuration Management:** Transitioned from basic TOML parsing to the robust `ConfigGuard` system with schema validation, versioning, encryption, and separate instance files.
*   **Binding System:** Refactored for discoverability via `binding_card.yaml`, standardized methods, and instance-specific configurations.
*   **Personality System:** Formalized `config.yaml` structure with `PersonalityConfig` and standardized scripted workflows.
*   **Generation Logic (`core/generation.py`):** Rewritten to orchestrate multimodal inputs, personalities, bindings, resource management, and standardized output.
*   **API (`/generate`):** Input format changed to `input_data: List[InputData]` to support multimodal inputs, replacing the deprecated `text_prompt` field. Standardized response formats.
*   **Logging:** Enhanced using the `ascii_colors` library.
*   **Installation:** Improved user experience with interactive scripts (`install_core.py`, `configuration_wizard.py`).



## [0.1.1] - 2025-04-17

### Added

*   **Comprehensive Test Suite:** Introduced a test suite using `pytest` framework.
    *   Added `tests/` directory structure mirroring the main application (`api/`, `core/`, `utils/`).
    *   Implemented initial unit and integration tests for:
        *   API Endpoints (`tests/api/test_endpoints.py`): Covering authentication, listing resources (bindings, personalities, models, functions), basic generation requests (TTT, TTI, streaming/non-streaming), and error handling (4xx, 5xx).
        *   Core Configuration (`tests/core/test_config.py`): Testing config loading, defaults, path resolution, and directory creation.
        *   Core Security (`tests/core/test_security.py`): Testing API key verification logic.
        *   Core Resource Manager (`tests/core/test_resource_manager.py`): Testing semaphore/lock strategies and timeouts.
        *   Utility Functions (`tests/utils/test_helpers.py`, `tests/utils/test_file_utils.py`): Testing code block extraction, path manipulation, and module loading.
    *   Added shared fixtures in `tests/conftest.py` for temporary directories and config setup.
    *   Introduced mock fixtures (`mock_config`, `mock_binding_manager`, etc.) in `tests/api/test_endpoints.py` using `unittest.mock`.
*   **Development Dependencies:** Added `pytest`, `pytest-asyncio`, `pytest-mock`, and `httpx` to `[project.optional-dependencies].dev` in `pyproject.toml`.
*   **Pytest Configuration:** Configured `pytest-asyncio` mode in `pyproject.toml` under `[tool.pytest.ini_options]` to automatically handle async tests.

### Fixed

*   **Code Block Extraction (`utils.helpers.extract_code_blocks`):** Corrected several bugs in the code block extraction logic related to handling incomplete blocks, language tags with surrounding whitespace, and blocks without language tags.
*   **API Testing Client (`tests/api/test_endpoints.py`):** Corrected `httpx.AsyncClient` instantiation to properly use `ASGITransport` for testing ASGI applications like FastAPI.
*   **Mocking Strategy (`tests/api/test_endpoints.py`):** Fixed issues where synchronous methods called on `AsyncMock` instances returned unexpected coroutine objects, leading to 500 errors. Switched relevant manager mocks to `MagicMock` where appropriate.
*   **API Response Validation (`tests/api/test_endpoints.py`, `api/models.py`):** Ensured mock data returned by fixtures aligns with Pydantic models (`PersonalityInfo`, `ModelInfo`) used within endpoints, resolving internal validation errors causing 500 responses. Added missing `icon` and `language` fields to `PersonalityInfo`.
*   **Resource Manager Config Validation (`tests/core/test_resource_manager.py`):** Corrected instantiation of `ResourceManagerConfig` in tests to explicitly provide all required fields, resolving Pydantic validation errors. Fixed usage of float for integer field `queue_timeout`.
*   **API Binding Not Found Error (`api/endpoints.py`):** Modified `list_available_models_for_binding` endpoint to return a 404 status code instead of 500/503 when a requested binding is not found.
*   **Pydantic Deprecation (`core/config.py`):** Updated `_resolve_paths` function to access `model_fields` via the class (`PathsConfig.model_fields`) instead of the instance, resolving `PydanticDeprecatedSince211` warning.

### Changed

*   **Setuptools Configuration (`pyproject.toml`):** Added `[tool.setuptools.packages.find]` section to explicitly specify the `lollms_server` package location, resolving build/installation errors related to multiple top-level directories.