# tests/api/test_endpoints.py
import pytest
import httpx
from fastapi import status
from fastapi.responses import StreamingResponse, JSONResponse
from unittest.mock import AsyncMock, MagicMock, patch
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import necessary components for dependency overrides and models
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
# --- IMPORT: Use updated models ---
from lollms_server.api.models import GenerateRequest, StreamChunk, PersonalityInfo, InputData, ModelInfo
# --- IMPORT: Binding for spec ---
from lollms_server.core.bindings import Binding

# --- Fixtures ---

@pytest.fixture
def test_api_key() -> str:
    return "test_api_key_123"

@pytest.fixture
def mock_config(test_api_key: str, tmp_path: Path) -> AppConfig:
    """Provides a mock AppConfig."""
    mock_paths = PathsConfig(
        personalities_folder=tmp_path / "pers", bindings_folder=tmp_path / "bind",
        functions_folder=tmp_path / "func", models_folder=tmp_path / "mod",
        example_personalities_folder=tmp_path / "zoo_pers", example_bindings_folder=tmp_path / "zoo_bind",
        example_functions_folder=tmp_path / "zoo_func"
    )
    for folder in [mock_paths.personalities_folder, mock_paths.bindings_folder, mock_paths.functions_folder, mock_paths.models_folder]:
         folder.mkdir(parents=True, exist_ok=True)
    for sub in ["ttt","tti","ttv","ttm","tts","stt", "i2i", "audio2audio"]:
        (mock_paths.models_folder / sub).mkdir(exist_ok=True)

    return AppConfig(
        server=ServerConfig(), logging=LoggingConfig(log_level="DEBUG", level=10), paths=mock_paths,
        security=SecurityConfig(allowed_api_keys=[test_api_key]),
        defaults=DefaultsConfig(ttt_binding="dummy_instance", ttt_model="dummy-model"),
        bindings={"dummy_instance": {"type": "dummy_binding", "mode": "ttt"}, "dummy_image_instance": {"type": "dummy_binding", "mode": "tti"}},
        resource_manager=ResourceManagerConfig(), webui=WebUIConfig(enable_ui=False),
        personalities_config={}
    )


@pytest.fixture
def mock_binding_manager() -> MagicMock:
    """Provides a mock BindingManager with updated Binding mock."""
    manager = MagicMock()
    manager.list_binding_types.return_value = {"dummy_binding": {"type_name": "dummy_binding", "description": "DummyDesc"}}
    manager.list_binding_instances.return_value = { "dummy_instance": {"type": "dummy_binding", "mode": "ttt"}, "dummy_image_instance": {"type": "dummy_binding", "mode": "tti"} }

    mock_binding_instance = AsyncMock(spec=Binding)
    mock_binding_instance.get_supported_input_modalities.return_value = ['text']
    mock_binding_instance.get_supported_output_modalities.return_value = ['text']

    mock_model_data = {
        "name": "dummy-model", "size": 100, "modified_at": datetime.now(),
        "quantization_level": "dummy_q", "format": "dummy_fmt", "family": "dummy_fam",
        "families": None, "parameter_size": None, "context_size": 2048,
        "max_output_tokens": 512, "template": None, "license": None,
        "homepage": None, "supports_vision": False, "supports_audio": False,
        "details": {"extra_detail": "value"},
    }
    # Return dict directly for list_available_models mock
    mock_binding_instance.list_available_models.return_value = [mock_model_data]

    async def mock_generate(prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List[InputData]] = None):
        gen_type = request_info.get("generation_type", "ttt")
        binding_name = getattr(mock_binding_instance, 'binding_name', 'unknown')
        if binding_name == "dummy_instance" and gen_type == 'ttt': return f"Mocked text response to: {prompt[:20]}"
        elif binding_name == "dummy_image_instance" and gen_type == 'tti':
            img_desc = f"cat wearing {prompt[:20]}" if not multimodal_data else f"image based on {prompt[:20]} and input"
            return {"image_base64": "dummy_base64_string", "mime_type": "image/png", "description": img_desc}
        else: return {"error": f"Mock generate for '{binding_name}' doesn't support '{gen_type}'"}
    mock_binding_instance.generate = mock_generate

    async def mock_generate_stream(prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List[InputData]] = None):
         gen_type = request_info.get("generation_type", "ttt")
         binding_name = getattr(mock_binding_instance, 'binding_name', 'unknown')
         if binding_name == "dummy_instance" and gen_type == 'ttt':
             resp_text = f"Streamed response to: {prompt[:20]}"; words = resp_text.split(); full_response = ""
             for i, word in enumerate(words):
                 content = word + " "; full_response += content
                 yield StreamChunk(type="chunk", content=content, metadata={"index": i}).model_dump(); await asyncio.sleep(0.01)
             yield StreamChunk(type="final", content=full_response.strip(), metadata={"words": len(words)}).model_dump()
         else:
             final_error_dict = {"error": f"Mock streaming for '{binding_name}' doesn't support '{gen_type}'"}
             yield StreamChunk(type="error", content=f"Streaming not supported for {gen_type} on {binding_name}").model_dump()
             yield StreamChunk(type="final", content=final_error_dict).model_dump()
    mock_binding_instance.generate_stream = mock_generate_stream

    def get_binding_side_effect(binding_name):
        if binding_name == "dummy_instance":
             mock_binding_instance.binding_name = "dummy_instance"
             mock_binding_instance.get_supported_input_modalities.return_value = ['text']
             mock_binding_instance.get_supported_output_modalities.return_value = ['text']
             # mock_binding_instance.list_available_models.reset_mock() # REMOVED RESET
             return mock_binding_instance
        elif binding_name == "dummy_image_instance":
             mock_binding_instance.binding_name = "dummy_image_instance"
             mock_binding_instance.get_supported_input_modalities.return_value = ['text', 'image']
             mock_binding_instance.get_supported_output_modalities.return_value = ['image']
             # mock_binding_instance.list_available_models.reset_mock() # REMOVED RESET
             return mock_binding_instance
        else: return None
    manager.get_binding.side_effect = get_binding_side_effect
    return manager


@pytest.fixture
def mock_personality_manager() -> MagicMock:
    """Provides a mock PersonalityManager."""
    manager = MagicMock()
    manager.list_personalities.return_value = { "test_pers": { "name": "test_pers", "author": "tester", "version": "1.0", "description": "Desc", "is_scripted": False, "path": "/fake", "category": "test", "tags": ["tag1"], "icon": "icon.png", "language": "english" } }
    mock_pers_instance = MagicMock(); mock_pers_instance.name = "test_pers"; mock_pers_instance.is_scripted = False
    mock_pers_instance.config = MagicMock(personality_conditioning="Default conditioning")
    manager.get_personality.return_value = mock_pers_instance
    return manager

@pytest.fixture
def mock_function_manager() -> MagicMock:
    """Provides a mock FunctionManager."""
    manager = MagicMock(); manager.list_functions.return_value = ["test_module.test_func"]
    return manager

@pytest.fixture
def mock_resource_manager() -> MagicMock:
    """Provides a mock ResourceManager."""
    manager = MagicMock(); return manager

@pytest.fixture(autouse=True)
async def override_dependencies( mock_config: AppConfig, mock_binding_manager: MagicMock, mock_personality_manager: MagicMock, mock_function_manager: MagicMock, mock_resource_manager: MagicMock, test_api_key: str ):
    """Overrides dependencies for API tests."""
    config_patch1 = patch('lollms_server.core.config.get_config', return_value=mock_config)
    config_patch2 = patch('lollms_server.core.security.get_config', return_value=mock_config)
    dependency_overrides = {
        get_config_dep: lambda: mock_config, get_binding_manager_dep: lambda: mock_binding_manager,
        get_personality_manager_dep: lambda: mock_personality_manager, get_function_manager_dep: lambda: mock_function_manager,
        get_resource_manager_dep: lambda: mock_resource_manager,
    }
    with config_patch1, config_patch2:
        original_overrides = fastapi_app.dependency_overrides.copy()
        fastapi_app.dependency_overrides.update(dependency_overrides)
        yield
        fastapi_app.dependency_overrides = original_overrides

@pytest.fixture
async def client() -> httpx.AsyncClient:
    """Provides an async test client."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test") as test_client: yield test_client

# --- Tests ---

@pytest.mark.asyncio
async def test_list_bindings_unauthorized(client: httpx.AsyncClient):
    """Tests listing bindings without API key -> 403."""
    response = await client.get("/api/v1/list_bindings"); assert response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.asyncio
async def test_list_bindings_invalid_key(client: httpx.AsyncClient, test_api_key: str):
    """Tests listing bindings with invalid API key -> 401."""
    headers = {"X-API-Key": "wrong-key"}; response = await client.get("/api/v1/list_bindings", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED; assert "Invalid or missing API Key" in response.json()["detail"]

@pytest.mark.asyncio
async def test_list_bindings_success(client: httpx.AsyncClient, test_api_key: str, mock_binding_manager: MagicMock):
    """Tests listing bindings successfully -> 200."""
    headers = {"X-API-Key": test_api_key}; response = await client.get("/api/v1/list_bindings", headers=headers)
    assert response.status_code == status.HTTP_200_OK; data = response.json()
    assert "binding_types" in data; assert "binding_instances" in data
    assert "dummy_instance" in data["binding_instances"]; mock_binding_manager.list_binding_types.assert_called_once(); mock_binding_manager.list_binding_instances.assert_called_once()

@pytest.mark.asyncio
async def test_list_personalities_success(client: httpx.AsyncClient, test_api_key: str, mock_personality_manager: MagicMock):
    """Tests listing personalities successfully -> 200."""
    headers = {"X-API-Key": test_api_key}; response = await client.get("/api/v1/list_personalities", headers=headers)
    assert response.status_code == status.HTTP_200_OK; data = response.json()
    assert "personalities" in data; assert "test_pers" in data["personalities"]; mock_personality_manager.list_personalities.assert_called_once()

@pytest.mark.asyncio
async def test_list_functions_success(client: httpx.AsyncClient, test_api_key: str, mock_function_manager: MagicMock):
    """Tests listing functions successfully -> 200."""
    headers = {"X-API-Key": test_api_key}; response = await client.get("/api/v1/list_functions", headers=headers)
    assert response.status_code == status.HTTP_200_OK; data = response.json()
    assert "functions" in data; assert data["functions"] == ["test_module.test_func"]; mock_function_manager.list_functions.assert_called_once()

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints._scan_models_folder')
async def test_list_models_success(mock_scan_models: MagicMock, client: httpx.AsyncClient, test_api_key: str, mock_config: AppConfig):
    """Tests listing discovered models successfully -> 200."""
    mock_scan_models.return_value = {"ttt": ["model1.gguf"], "tti": ["sdxl.safetensors"], "tts": [], "stt": [], "i2i": [], "audio2audio": []}
    headers = {"X-API-Key": test_api_key}; response = await client.get("/api/v1/list_models", headers=headers)
    assert response.status_code == status.HTTP_200_OK; data = response.json(); assert "models" in data
    assert data["models"]["ttt"] == ["model1.gguf"]; assert data["models"]["tti"] == ["sdxl.safetensors"]; mock_scan_models.assert_called_once_with(mock_config.paths.models_folder)

@pytest.mark.asyncio
async def test_list_available_models_success(client: httpx.AsyncClient, test_api_key: str, mock_binding_manager: MagicMock):
    """Tests listing available models for a binding -> 200."""
    binding_name = "dummy_instance"; headers = {"X-API-Key": test_api_key}
    # Reset mock *before* the call for this specific test case if needed
    binding_instance_mock = mock_binding_manager.get_binding(binding_name) # Get the instance first
    binding_instance_mock.list_available_models.reset_mock() # Reset just before use

    response = await client.get(f"/api/v1/list_available_models/{binding_name}", headers=headers)

    assert response.status_code == status.HTTP_200_OK; data = response.json()
    assert data["binding_name"] == binding_name; assert "models" in data; assert len(data["models"]) == 1
    model_info = data["models"][0]; assert model_info["name"] == "dummy-model"; assert model_info["supports_vision"] is False
    # Check the specific instance returned by the side effect was awaited
    binding_instance_mock.list_available_models.assert_awaited_once()

@pytest.mark.asyncio
async def test_list_available_models_not_found(client: httpx.AsyncClient, test_api_key: str, mock_binding_manager: MagicMock):
    """Tests listing available models for non-existent binding -> 404."""
    binding_name = "non_existent_binding"; headers = {"X-API-Key": test_api_key}
    response = await client.get(f"/api/v1/list_available_models/{binding_name}", headers=headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND; assert f"Binding '{binding_name}' not found" in response.json().get("detail", "")
    assert mock_binding_manager.get_binding.called

@pytest.mark.asyncio
async def test_generate_missing_input(client: httpx.AsyncClient, test_api_key: str):
    """Tests generate endpoint with missing input -> 422."""
    headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = {"input_data": [], "text_prompt": None, "personality": "test_pers"}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY; assert "Generation request must include at least one item" in response.text

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_ttt_non_stream_success_new_payload(mock_process_generation: AsyncMock, client: httpx.AsyncClient, test_api_key: str, mock_config: AppConfig, mock_binding_manager: MagicMock, mock_personality_manager: MagicMock, mock_function_manager: MagicMock, mock_resource_manager: MagicMock):
    """Tests TTT non-streaming using input_data -> 200 JSON."""
    mock_prompt = "Tell me a joke"; expected_output_dict = {"text": f"Mocked text response to: {mock_prompt[:20]}"}
    response_payload = {"personality": "test_pers", "output": expected_output_dict, "execution_time": 0.123, "request_id": None}
    mock_process_generation.return_value = JSONResponse(content=response_payload); headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = {"input_data": [{"type": "text", "role": "user_prompt", "data": mock_prompt}], "stream": False, "personality": "test_pers", "parameters": {"temperature": 0.9}}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_200_OK; assert "application/json" in response.headers["content-type"]
    data = response.json(); assert data["output"] == expected_output_dict; mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args; passed_request: GenerateRequest = call_kwargs['request']
    assert isinstance(passed_request, GenerateRequest); assert len(passed_request.input_data) == 1; assert passed_request.input_data[0].data == mock_prompt

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_ttt_non_stream_success_deprecated_prompt(mock_process_generation: AsyncMock, client: httpx.AsyncClient, test_api_key: str):
    """Tests TTT non-streaming using deprecated text_prompt -> 200 JSON."""
    mock_prompt = "Old joke format"; expected_output_dict = {"text": f"Mocked text response to: {mock_prompt[:20]}"}
    response_payload = {"personality": "test_pers", "output": expected_output_dict, "execution_time": 0.1, "request_id": None}
    mock_process_generation.return_value = JSONResponse(content=response_payload); headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = {"text_prompt": mock_prompt, "stream": False, "personality": "test_pers"}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_200_OK; assert "application/json" in response.headers["content-type"]
    data = response.json(); assert data["output"] == expected_output_dict; mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args; passed_request: GenerateRequest = call_kwargs['request']
    assert len(passed_request.input_data) == 1; assert passed_request.input_data[0].data == mock_prompt; assert passed_request.text_prompt is None

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_tti_success(mock_process_generation: AsyncMock, client: httpx.AsyncClient, test_api_key: str):
    """Tests TTI non-streaming -> 200 JSON."""
    mock_output_dict = {"image_base64": "dummy_base64_string", "mime_type": "image/png", "description": "cat wearing wizard hat"}
    response_payload = {"personality": None, "output": mock_output_dict, "execution_time": 0.5, "request_id": None}
    mock_process_generation.return_value = JSONResponse(content=response_payload); headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = {"input_data": [{"type": "text", "role": "user_prompt", "data": "wizard hat"}], "generation_type": "tti", "stream": False, "binding_name": "dummy_image_instance"}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_200_OK; assert "application/json" in response.headers["content-type"]
    data = response.json(); assert data["output"] == mock_output_dict; mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args; passed_request: GenerateRequest = call_kwargs['request']
    assert passed_request.generation_type == "tti"

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_vision_success(mock_process_generation: AsyncMock, client: httpx.AsyncClient, test_api_key: str):
    """Tests vision (image input) non-streaming -> 200 JSON."""
    mock_output_dict = {"text": "The image shows a black cat."}; response_payload = {"personality": None, "output": mock_output_dict, "execution_time": 0.6, "request_id": None}
    mock_process_generation.return_value = JSONResponse(content=response_payload); headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    payload = {"input_data": [{"type": "text", "role": "user_prompt", "data": "Describe image."}, {"type": "image", "role": "input_image", "data": dummy_b64, "mime_type": "image/png"}], "generation_type": "ttt", "stream": False, "binding_name": "dummy_image_instance"}
    response = await client.post("/api/v1/generate", headers=headers, json=payload)
    assert response.status_code == status.HTTP_200_OK; assert "application/json" in response.headers["content-type"]
    data = response.json(); assert data["output"] == mock_output_dict; mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args; passed_request: GenerateRequest = call_kwargs['request']
    assert len(passed_request.input_data) == 2; assert passed_request.input_data[1].type == "image"

@pytest.mark.asyncio
@patch('lollms_server.api.endpoints.process_generation_request', new_callable=AsyncMock)
async def test_generate_ttt_stream_success(mock_process_generation: AsyncMock, client: httpx.AsyncClient, test_api_key: str):
    """Tests TTT streaming -> 200 SSE."""
    async def mock_sse_content_generator():
        yield f"data: {json.dumps(StreamChunk(type='chunk', content='Hello ').model_dump())}\n\n"; await asyncio.sleep(0.01)
        yield f"data: {json.dumps(StreamChunk(type='chunk', content='World!').model_dump())}\n\n"; await asyncio.sleep(0.01)
        yield f"data: {json.dumps(StreamChunk(type='final', content='Hello World!', metadata={'tokens': 2}).model_dump())}\n\n"
    mock_process_generation.return_value = StreamingResponse(mock_sse_content_generator(), media_type="text/event-stream")
    headers = {"X-API-Key": test_api_key, "Content-Type": "application/json"}
    payload = {"input_data": [{"type": "text", "role": "user_prompt", "data": "Say hello"}], "stream": True, "generation_type": "ttt"}
    received_chunks = []; full_text = ""; final_chunk_data = None
    async with client.stream("POST", "/api/v1/generate", headers=headers, json=payload) as response:
        assert response.status_code == status.HTTP_200_OK; assert "text/event-stream" in response.headers["content-type"]
        sse_buffer = ""
        async for line in response.aiter_lines():
            sse_buffer += line + "\n"
            if sse_buffer.endswith("\n\n"):
                message = sse_buffer.strip()
                if message.startswith("data:"):
                    try:
                        json_data = message[len("data:"):].strip(); chunk_data = json.loads(json_data); received_chunks.append(chunk_data)
                        if chunk_data.get("type") == "chunk": full_text += chunk_data.get("content", "")
                        elif chunk_data.get("type") == "final": final_chunk_data = chunk_data
                    except json.JSONDecodeError: pytest.fail(f"Failed SSE decode: {json_data}")
                sse_buffer = ""
    assert len(received_chunks) == 3; assert full_text == "Hello World!"; assert final_chunk_data is not None
    assert final_chunk_data["content"] == "Hello World!"; mock_process_generation.assert_awaited_once()
    call_args, call_kwargs = mock_process_generation.call_args; passed_request: GenerateRequest = call_kwargs['request']
    assert passed_request.stream is True