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