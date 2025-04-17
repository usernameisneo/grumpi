# tests/api/test_endpoints.py
import pytest
import httpx
from fastapi import status
from fastapi.responses import StreamingResponse
from unittest.mock import AsyncMock, MagicMock, patch # Use MagicMock for manager
import json
import asyncio
from pathlib import Path

# Import necessary components for dependency overrides
from lollms_server.main import app as fastapi_app
from lollms_server.core.config import (
    AppConfig, ServerConfig, LoggingConfig, PathsConfig,
    SecurityConfig, DefaultsConfig, ResourceManagerConfig,
    WebUIConfig
)
from lollms_server.api.endpoints import (
    get_config_dep, get_binding_manager_dep, get_personality_manager_dep,
    get_function_manager_dep, get_resource_manager_dep
)
from lollms_server.core.security import verify_api_key
from lollms_server.api.models import GenerateRequest, StreamChunk, PersonalityInfo

# --- Fixtures ---

@pytest.fixture
def test_api_key() -> str:
    return "test_api_key_123"

@pytest.fixture
def mock_config(test_api_key: str, tmp_path: Path) -> AppConfig:
    mock_paths = PathsConfig(
        personalities_folder=tmp_path / "pers", bindings_folder=tmp_path / "bind",
        functions_folder=tmp_path / "func", models_folder=tmp_path / "mod",
        example_personalities_folder=tmp_path / "zoo_pers", example_bindings_folder=tmp_path / "zoo_bind",
        example_functions_folder=tmp_path / "zoo_func"
    )
    mock_paths.personalities_folder.mkdir(parents=True, exist_ok=True)
    mock_paths.bindings_folder.mkdir(parents=True, exist_ok=True)
    mock_paths.functions_folder.mkdir(parents=True, exist_ok=True)
    mock_paths.models_folder.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        server=ServerConfig(), logging=LoggingConfig(log_level="DEBUG", level=10), paths=mock_paths,
        security=SecurityConfig(allowed_api_keys=[test_api_key]),
        defaults=DefaultsConfig(ttt_binding="dummy_binding", ttt_model="dummy-model"),
        bindings={"dummy_instance": {"type": "dummy_binding", "mode": "ttt"}},
        resource_manager=ResourceManagerConfig(), webui=WebUIConfig(enable_ui=False),
        personalities_config={}
    )

@pytest.fixture
def mock_binding_manager() -> MagicMock: # --- FIX: Use MagicMock ---
    """Provides a mock BindingManager."""
    manager = MagicMock() # Use MagicMock as methods called are sync
    manager.list_binding_types.return_value = {"dummy_binding": {"type_name": "dummy_binding", "description": "DummyDesc"}}
    manager.list_binding_instances.return_value = {
        "dummy_instance": {"type": "dummy_binding", "mode": "ttt"}
    }
    # Mock the binding instance returned by get_binding
    # IMPORTANT: Binding methods like list_available_models ARE async, so the *returned* mock needs to be AsyncMock
    mock_binding_instance = AsyncMock()
    mock_model_data = {
        "name": "dummy-model", "size": 100, "modified_at": None,
        "quantization_level": "dummy_q", "format": "dummy_fmt", "family": "dummy_fam",
        "families": None, "parameter_size": None, "context_size": 2048,
        "max_output_tokens": 512, "template": None, "license": None,
        "homepage": None, "details": {"extra_detail": "value"},
    }
    mock_binding_instance.list_available_models = AsyncMock(return_value=[mock_model_data]) # Mock the async method
    # Configure the SYNC get_binding method on the MagicMock manager
    manager.get_binding.return_value = mock_binding_instance
    return manager

@pytest.fixture
def mock_personality_manager() -> MagicMock:
    """Provides a mock PersonalityManager."""
    manager = MagicMock()
    manager.list_personalities.return_value = {
        "test_pers": {
            "name": "test_pers", "author": "tester", "version": "1.0",
            "description": "Desc", "is_scripted": False, "path": "/fake",
            "category": "test", "tags": ["tag1"],
            "icon": "icon.png", # --- FIX: Added icon ---
            "language": "english",
        }
    }
    mock_pers_instance = MagicMock()
    mock_pers_instance.name = "test_pers"
    mock_pers_instance.is_scripted = False
    mock_pers_instance.config = MagicMock(personality_conditioning="Default conditioning")
    manager.get_personality.return_value = mock_pers_instance
    return manager

@pytest.fixture
def mock_function_manager() -> MagicMock:
    manager = MagicMock()
    manager.list_functions.return_value = ["test_module.test_func"]
    return manager

@pytest.fixture
def mock_resource_manager() -> MagicMock:
     manager = MagicMock()
     return manager

# --- Apply Dependency Overrides globally for this test module ---
@pytest.fixture(autouse=True)
async def override_dependencies(
    mock_config: AppConfig,
    mock_binding_manager: MagicMock, # Changed to MagicMock
    mock_personality_manager: MagicMock,
    mock_function_manager: MagicMock,
    mock_resource_manager: MagicMock,
    test_api_key: str
):
    config_patch1 = patch('lollms_server.core.config.get_config', return_value=mock_config)
    config_patch2 = patch('lollms_server.core.security.get_config', return_value=mock_config)
    dependency_overrides = {
        get_config_dep: lambda: mock_config,
        get_binding_manager_dep: lambda: mock_binding_manager,
        get_personality_manager_dep: lambda: mock_personality_manager,
        get_function_manager_dep: lambda: mock_function_manager,
        get_resource_manager_dep: lambda: mock_resource_manager,
        # Let real verify_api_key run with mocked config
    }
    with config_patch1, config_patch2:
        original_overrides = fastapi_app.dependency_overrides.copy()
        fastapi_app.dependency_overrides.update(dependency_overrides)
        yield
        fastapi_app.dependency_overrides = original_overrides

# --- Client Fixture ---
@pytest.fixture
async def client() -> httpx.AsyncClient:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test") as test_client:
        yield test_client

# --- Tests ---

@pytest.mark.asyncio
async def test_list_bindings_unauthorized(client: httpx.AsyncClient):
    response = await client.get("/api/v1/list_bindings")
    assert response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.asyncio
async def test_list_bindings_invalid_key(client: httpx.AsyncClient, test_api_key: str):
    headers = {"X-API-Key": "this-is-the-wrong-key"}
    response = await client.get("/api/v1/list_bindings", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid or missing API Key" in response.json()["detail"]

@pytest.mark.asyncio
async def test_list_bindings_success(
    client: httpx.AsyncClient,
    test_api_key: str,
    mock_binding_manager: MagicMock # Changed to MagicMock
):
    """Test /list_bindings with a valid API key -> 200."""
    headers = {"X-API-Key": test_api_key}
    response = await client.get("/api/v1/list_bindings", headers=headers)
    # Add print for debugging if it still fails
    if response.status_code != 200:
        print("Response Text:", response.text)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "binding_types" in data
    assert "binding_instances" in data
    assert data["binding_types"] == {"dummy_binding": {"type_name": "dummy_binding", "description": "DummyDesc"}}
    assert data["binding_instances"] == {"dummy_instance": {"type": "dummy_binding", "mode": "ttt"}}
    mock_binding_manager.list_binding_types.assert_called_once()
    mock_binding_manager.list_binding_instances.assert_called_once()

@pytest.mark.asyncio
async def test_list_personalities_success(
    client: httpx.AsyncClient,
    test_api_key: str,
    mock_personality_manager: MagicMock
):
    """Test /list_personalities success -> 200."""
    headers = {"X-API-Key": test_api_key}
    response = await client.get("/api/v1/list_personalities", headers=headers)
    # Add print for debugging if it still fails
    if response.status_code != 200:
        print("Response Text:", response.text)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "personalities" in data
    assert "test_pers" in data["personalities"]
    p_info = data["personalities"]["test_pers"]
    assert p_info["name"] == "test_pers"
    assert p_info["author"] == "tester"
    assert p_info["version"] == "1.0"
    assert p_info["description"] == "Desc"
    assert p_info["category"] == "test"
    assert p_info["tags"] == ["tag1"]
    assert p_info["is_scripted"] is False
    assert p_info["path"] == "/fake"
    assert p_info["icon"] == "icon.png" # Check icon
    assert p_info["language"] == "english" # Check language
    mock_personality_manager.list_personalities.assert_called_once()

# ... (test_list_functions_success, test_list_models_success remain the same) ...

@pytest.mark.asyncio
async def test_list_functions_success(
    client: httpx.AsyncClient, # Use client fixture
    test_api_key: str,
    mock_function_manager: MagicMock
):
    """Test /list_functions success -> 200."""
    headers = {"X-API-Key": test_api_key}
    response = await client.get("/api/v1/list_functions", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "functions" in data
    assert data["functions"] == ["test_module.test_func"]
    mock_function_manager.list_functions.assert_called_once()

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints._scan_models_folder')
async def test_list_models_success(
    mock_scan_models: MagicMock,
    client: httpx.AsyncClient, # Use client fixture
    test_api_key: str,
    mock_config: AppConfig
):
    """Test /list_models success -> 200."""
    mock_scan_models.return_value = {"ttt": ["model1.gguf"], "tti": ["sdxl.safetensors"]}
    headers = {"X-API-Key": test_api_key}
    response = await client.get("/api/v1/list_models", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "models" in data
    assert data["models"] == {"ttt": ["model1.gguf"], "tti": ["sdxl.safetensors"]}
    mock_scan_models.assert_called_once_with(mock_config.paths.models_folder)


@pytest.mark.asyncio
async def test_list_available_models_success(
    client: httpx.AsyncClient,
    test_api_key: str,
    mock_binding_manager: MagicMock # Changed to MagicMock
):
    """Test /list_available_models success -> 200."""
    binding_name = "dummy_instance"
    headers = {"X-API-Key": test_api_key}
    response = await client.get(f"/api/v1/list_available_models/{binding_name}", headers=headers)
    # Add print for debugging if it still fails
    if response.status_code != 200:
        print("Response Text:", response.text)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["binding_name"] == binding_name
    assert "models" in data
    assert len(data["models"]) == 1
    model_info = data["models"][0]
    assert model_info["name"] == "dummy-model"
    assert model_info["size"] == 100
    assert model_info["context_size"] == 2048
    assert model_info["quantization_level"] == "dummy_q"
    # Check manager was called
    mock_binding_manager.get_binding.assert_called_with(binding_name)
    # Check the returned mock binding instance's async method was awaited
    binding_instance_mock = mock_binding_manager.get_binding.return_value
    binding_instance_mock.list_available_models.assert_awaited_once()

@pytest.mark.asyncio
async def test_list_available_models_not_found(
    client: httpx.AsyncClient,
    test_api_key: str,
    mock_binding_manager: MagicMock # Changed to MagicMock
):
    """Test /list_available_models when binding doesn't exist -> 404."""
    binding_name = "non_existent_binding"
    headers = {"X-API-Key": test_api_key}
    mock_binding_manager.get_binding.return_value = None # Simulate binding not found

    response = await client.get(f"/api/v1/list_available_models/{binding_name}", headers=headers)
    # Assert based on updated endpoint logic raising 404
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert f"Binding '{binding_name}' not found" in response.json().get("detail", "")
    mock_binding_manager.get_binding.assert_called_with(binding_name)

# ... (Generation tests remain the same, ensure mock_process_generation args access is correct) ...
@pytest.mark.asyncio
async def test_generate_missing_prompt(client: httpx.AsyncClient, test_api_key: str):
    """Test /generate without providing a prompt -> 422."""
    headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = {"personality": "test_pers"}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_ttt_non_stream_success(
    mock_process_generation: AsyncMock,
    client: httpx.AsyncClient,
    test_api_key: str,
    mock_config: AppConfig,
    mock_binding_manager: MagicMock, # Changed to MagicMock
    mock_personality_manager: MagicMock,
    mock_function_manager: MagicMock,
    mock_resource_manager: MagicMock
):
    """Test successful non-streaming TTT generation -> 200 text/plain."""
    expected_response = "Generated text response."
    mock_process_generation.return_value = expected_response
    headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = { "prompt": "Tell me a joke", "stream": False, "personality": "test_pers", "parameters": {"temperature": 0.9}}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.text == expected_response
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args
    assert len(call_args) == 0
    passed_request: GenerateRequest = call_kwargs['request']
    assert isinstance(passed_request, GenerateRequest)
    assert passed_request.prompt == "Tell me a joke"
    # ... check other args ...
    assert call_kwargs['personality_manager'] is mock_personality_manager
    assert call_kwargs['binding_manager'] is mock_binding_manager
    assert call_kwargs['function_manager'] is mock_function_manager
    assert call_kwargs['resource_manager'] is mock_resource_manager
    assert call_kwargs['config'] is mock_config

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_tti_success(
    mock_process_generation: AsyncMock,
    client: httpx.AsyncClient,
    test_api_key: str,
):
    """Test successful TTI generation -> 200 application/json."""
    mock_response_dict = {"image_base64": "dummy_base64_string", "model": "dummy_tti"}
    mock_process_generation.return_value = mock_response_dict
    headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = { "prompt": "A cat painting", "generation_type": "tti", "stream": False }
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == mock_response_dict
    assert "application/json" in response.headers["content-type"]
    mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args
    passed_request: GenerateRequest = call_kwargs['request']
    assert passed_request.generation_type == "tti"

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_ttt_stream_success(
    mock_process_generation: AsyncMock,
    client: httpx.AsyncClient,
    test_api_key: str,
):
    """Test successful streaming TTT generation -> 200 text/event-stream."""
    async def mock_sse_content_generator():
        chunk1 = StreamChunk(type="chunk", content="Hello ").model_dump()
        yield f"data: {json.dumps(chunk1)}\n\n"
        await asyncio.sleep(0.01)
        chunk2 = StreamChunk(type="chunk", content="World!").model_dump()
        yield f"data: {json.dumps(chunk2)}\n\n"
        await asyncio.sleep(0.01)
        chunk3 = StreamChunk(type="final", content="Hello World!", metadata={"tokens": 2}).model_dump()
        yield f"data: {json.dumps(chunk3)}\n\n"

    mock_process_generation.return_value = StreamingResponse(
        mock_sse_content_generator(), media_type="text/event-stream"
    )
    headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = { "prompt": "Say hello", "stream": True, "generation_type": "ttt" }
    received_chunks = []
    full_text = ""
    final_chunk_data = None
    async with client.stream("POST", "/api/v1/generate", headers=headers, json=payload) as response:
        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers["content-type"]
        sse_buffer = ""
        async for line in response.aiter_lines():
            sse_buffer += line + "\n"
            if sse_buffer.endswith("\n\n"):
                message = sse_buffer.strip()
                if message.startswith("data:"):
                    try:
                        json_data = message[len("data:"):].strip()
                        chunk_data = json.loads(json_data)
                        received_chunks.append(chunk_data)
                        if chunk_data.get("type") == "chunk":
                            full_text += chunk_data.get("content", "")
                        elif chunk_data.get("type") == "final":
                            final_chunk_data = chunk_data
                    except json.JSONDecodeError:
                        pytest.fail(f"Failed to decode SSE JSON data: {json_data}")
                sse_buffer = ""
    assert len(received_chunks) == 3
    # ... check chunks ...
    assert full_text == "Hello World!"
    assert final_chunk_data is not None
    mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args
    passed_request: GenerateRequest = call_kwargs['request']
    assert passed_request.stream is True