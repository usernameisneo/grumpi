# lollms_server/api/endpoints.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Query
from fastapi.responses import JSONResponse, StreamingResponse
import ascii_colors as logging
from typing import List, Dict, Any, Union, Optional
from pydantic import ValidationError
import importlib.metadata # To get version

# --- IMPORT: Use updated models ---
from .models import (
    GenerateRequest, ListBindingsResponse, ListPersonalitiesResponse, PersonalityInfo,
    ListFunctionsResponse, ListModelsResponse, ModelInfo, ListAvailableModelsResponse, InputData,
    HealthResponse, StreamChunk, GenerateResponse, # Added GenerateResponse
    TokenizeRequest, TokenizeResponse, DetokenizeRequest, DetokenizeResponse,
    CountTokensRequest, CountTokensResponse, GetModelInfoResponse # Added new models
)
# Core components (NO direct import from main.py)
from lollms_server.core.config import AppConfig
from lollms_server.core.security import verify_api_key
from lollms_server.core.bindings import BindingManager, Binding
from lollms_server.core.personalities import PersonalityManager
from lollms_server.core.functions import FunctionManager
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.core.generation import process_generation_request, _scan_models_folder

logger = logging.getLogger(__name__)
router = APIRouter()

# --- API Dependencies (unchanged) ---
def get_config_dep(request: Request) -> AppConfig:
    if not hasattr(request.app.state, 'config'): raise HTTPException(status_code=503, detail="Server configuration not ready")
    return request.app.state.config
def get_binding_manager_dep(request: Request) -> BindingManager:
    if not hasattr(request.app.state, 'binding_manager'): raise HTTPException(status_code=503, detail="Binding Manager not ready")
    return request.app.state.binding_manager
def get_personality_manager_dep(request: Request) -> PersonalityManager:
    if not hasattr(request.app.state, 'personality_manager'): raise HTTPException(status_code=503, detail="Personality Manager not ready")
    return request.app.state.personality_manager
def get_function_manager_dep(request: Request) -> FunctionManager:
    if not hasattr(request.app.state, 'function_manager'): raise HTTPException(status_code=503, detail="Function Manager not ready")
    return request.app.state.function_manager
def get_resource_manager_dep(request: Request) -> ResourceManager:
    if not hasattr(request.app.state, 'resource_manager'): raise HTTPException(status_code=503, detail="Resource Manager not ready")
    return request.app.state.resource_manager

# === Helper Function to get Binding and check loaded model ===
async def _get_binding_and_check_loaded(binding_name: str, binding_manager: BindingManager) -> Binding:
    """Helper to get binding and check if a model is loaded."""
    binding = binding_manager.get_binding(binding_name)
    if not binding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding '{binding_name}' not found.")
    if not binding.is_model_loaded:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"No model is currently loaded for binding '{binding_name}'. Load one via /generate first.")
    return binding

# === Health Check Endpoint ===
@router.get("/health",
            response_model=HealthResponse,
            summary="Check Server Health and Configuration",
            description="Provides server status and basic configuration info.",
            tags=["Server Info"])
async def health_check(config: AppConfig = Depends(get_config_dep)):
    """Checks server health and returns basic configuration status."""
    is_key_required = bool(config.security.allowed_api_keys)
    server_version = None
    try: server_version = importlib.metadata.version("lollms_server")
    except importlib.metadata.PackageNotFoundError: logger.warning("Could not determine lollms_server version.")
    return HealthResponse( status="ok", api_key_required=is_key_required, version=server_version )

# === Listing Endpoints ===
@router.get("/list_bindings", response_model=ListBindingsResponse, summary="List Available Bindings", dependencies=[Depends(verify_api_key)], tags=["Listing"])
async def list_bindings(binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """Lists discovered binding types and configured binding instances."""
    try: return ListBindingsResponse(binding_types=binding_manager.list_binding_types(), binding_instances=binding_manager.list_binding_instances())
    except Exception as e: logger.error(f"Error listing bindings: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list bindings")

@router.get("/list_available_models/{binding_name}", response_model=ListAvailableModelsResponse, summary="List Models Available to a Specific Binding", dependencies=[Depends(verify_api_key)], tags=["Listing"])
async def list_available_models_for_binding(binding_name: str, binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """Retrieves models recognized by a specific configured binding instance."""
    logger.info(f"Request received to list available models for binding: '{binding_name}'")
    binding = binding_manager.get_binding(binding_name)
    if not binding: logger.error(f"Binding '{binding_name}' requested not found."); raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding '{binding_name}' not found.")
    try:
        models_data: List[Dict[str, Any]] = await binding.list_available_models(); validated_models = []
        for model_dict in models_data:
            try: model_dict.setdefault('details', {}); validated_models.append(ModelInfo(**model_dict))
            except ValidationError as e: logger.warning(f"Failed validate model data from '{binding_name}': {model_dict}. Error: {e}. Skip.")
            except Exception as e_inner: logger.warning(f"Error processing model data {model_dict} for '{binding_name}': {e_inner}. Skip.")
        return ListAvailableModelsResponse(binding_name=binding_name, models=validated_models)
    except (RuntimeError, ValueError, NotImplementedError) as e: logger.error(f"Error getting models from '{binding_name}': {e}"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed get models from '{binding_name}': {e}")
    except Exception as e: logger.error(f"Unexpected error list models for '{binding_name}': {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error list models for '{binding_name}'.")

@router.get("/list_personalities", response_model=ListPersonalitiesResponse, summary="List Available Personalities", dependencies=[Depends(verify_api_key)], tags=["Listing"])
async def list_personalities(personality_manager: PersonalityManager = Depends(get_personality_manager_dep)):
    """Lists loaded and enabled personalities."""
    try:
        raw_list = personality_manager.list_personalities(); personalities_info = {}
        for name, info_dict in raw_list.items():
            try:
                info_dict.setdefault('category', None); info_dict.setdefault('tags', []); info_dict.setdefault('icon', None); info_dict.setdefault('language', None)
                personalities_info[name] = PersonalityInfo(**info_dict)
            except ValidationError as e: logger.warning(f"Failed validate personality '{name}': {info_dict}. Error: {e}. Skip.")
            except Exception as e_inner: logger.warning(f"Error processing personality '{name}': {e_inner}. Skip.")
        return ListPersonalitiesResponse(personalities=personalities_info)
    except Exception as e: logger.error(f"Error listing personalities: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list personalities")

@router.get("/list_functions", response_model=ListFunctionsResponse, summary="List Available Functions", dependencies=[Depends(verify_api_key)], tags=["Listing"])
async def list_functions(function_manager: FunctionManager = Depends(get_function_manager_dep)):
    """Lists discovered custom Python functions."""
    try: return ListFunctionsResponse(functions=function_manager.list_functions())
    except Exception as e: logger.error(f"Error listing functions: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list functions")

@router.get("/list_models", response_model=ListModelsResponse, summary="List Discovered Models in Folder", dependencies=[Depends(verify_api_key)], tags=["Listing"])
async def list_models(config: AppConfig = Depends(get_config_dep)):
    """Lists models found by scanning the configured models folder."""
    try: return ListModelsResponse(models=_scan_models_folder(config.paths.models_folder))
    except Exception as e: logger.error(f"Error listing models: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list models")

# === Generation Endpoint ===
@router.post("/generate",
             summary="Generate Text, Image, etc.",
             description="Triggers a generation task using specified inputs and parameters.",
             response_model=None, # Return type varies (JSON or SSE stream)
             dependencies=[Depends(verify_api_key)],
             tags=["Generation"])
async def generate(
    request_payload: GenerateRequest,
    personality_manager: PersonalityManager = Depends(get_personality_manager_dep),
    binding_manager: BindingManager = Depends(get_binding_manager_dep),
    function_manager: FunctionManager = Depends(get_function_manager_dep),
    resource_manager: ResourceManager = Depends(get_resource_manager_dep),
    config: AppConfig = Depends(get_config_dep),
    http_request: Request = None # Keep if needed by FastAPI internally
) -> Response:
    """Handles generation requests, supporting multimodal inputs."""
    logger.info(f"Received generation request (Type: {request_payload.generation_type}, Stream: {request_payload.stream})")
    if request_payload.input_data: logger.info(f"Input Data Summary: {[f'{item.type}/{item.role}' for item in request_payload.input_data]}")
    else: logger.warning("Received generate request with empty input_data list.")
    try:
        result_response = await process_generation_request( request=request_payload, personality_manager=personality_manager, binding_manager=binding_manager, function_manager=function_manager, resource_manager=resource_manager, config=config )
        if isinstance(result_response, (JSONResponse, StreamingResponse)): return result_response
        else:
            logger.error(f"process_generation_request returned unexpected type: {type(result_response)}")
            error_content = {"error": "Internal server error: Unexpected generation result type", "details": str(result_response)}
            return JSONResponse(content=error_content, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except HTTPException as e: logger.warning(f"HTTP Exception during generation: Status={e.status_code}, Detail={e.detail}"); raise e
    except Exception as e: logger.error(f"Unhandled exception in /generate: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred.")

# === NEW: Tokenizer Endpoints ===
@router.post("/tokenize", response_model=TokenizeResponse, summary="Tokenize Text", dependencies=[Depends(verify_api_key)], tags=["Utilities"])
async def tokenize_text(request: TokenizeRequest, binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """Tokenizes text using the specified binding's currently loaded model."""
    try:
        binding = await _get_binding_and_check_loaded(request.binding_name, binding_manager)
        tokens = await binding.tokenize(request.text, request.add_bos, request.add_eos)
        return TokenizeResponse(tokens=tokens, count=len(tokens))
    except NotImplementedError: raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding '{request.binding_name}' does not support tokenization.")
    except HTTPException as e: raise e # Re-raise 404/409
    except Exception as e: logger.error(f"Error tokenizing text: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tokenization failed: {e}")

@router.post("/detokenize", response_model=DetokenizeResponse, summary="Detokenize Tokens", dependencies=[Depends(verify_api_key)], tags=["Utilities"])
async def detokenize_tokens(request: DetokenizeRequest, binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """Detokenizes a list of token IDs using the specified binding's currently loaded model."""
    try:
        binding = await _get_binding_and_check_loaded(request.binding_name, binding_manager)
        text = await binding.detokenize(request.tokens)
        return DetokenizeResponse(text=text)
    except NotImplementedError: raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding '{request.binding_name}' does not support detokenization.")
    except HTTPException as e: raise e # Re-raise 404/409
    except Exception as e: logger.error(f"Error detokenizing tokens: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Detokenization failed: {e}")

@router.post("/count_tokens", response_model=CountTokensResponse, summary="Count Tokens in Text", dependencies=[Depends(verify_api_key)], tags=["Utilities"])
async def count_tokens(request: CountTokensRequest, binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """Counts tokens in text using the specified binding's currently loaded model."""
    try:
        binding = await _get_binding_and_check_loaded(request.binding_name, binding_manager)
        tokens = await binding.tokenize(request.text, add_bos=False, add_eos=False) # Typically don't count special tokens for this
        return CountTokensResponse(count=len(tokens))
    except NotImplementedError: raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding '{request.binding_name}' does not support tokenization.")
    except HTTPException as e: raise e # Re-raise 404/409
    except Exception as e: logger.error(f"Error counting tokens: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Token counting failed: {e}")

# === NEW: Model Info Endpoint ===
@router.get("/get_model_info/{binding_name}", response_model=GetModelInfoResponse, summary="Get Current Model Info", dependencies=[Depends(verify_api_key)], tags=["Utilities"])
async def get_model_info(binding_name: str, binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """Retrieves information about the currently loaded model for a binding."""
    binding = binding_manager.get_binding(binding_name)
    if not binding: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding '{binding_name}' not found.")
    try:
        if not binding.is_model_loaded:
            logger.info(f"No model currently loaded for binding '{binding_name}'. Returning empty info.")
            return GetModelInfoResponse(binding_name=binding_name, model_name=None, context_size=None, max_output_tokens=None, supports_vision=False, supports_audio=False, details={})

        model_info = await binding.get_current_model_info()
        # Map the returned dict to the response model
        return GetModelInfoResponse(
            binding_name=binding_name,
            model_name=model_info.get("name"),
            context_size=model_info.get("context_size"),
            max_output_tokens=model_info.get("max_output_tokens"),
            supports_vision=model_info.get("supports_vision", False),
            supports_audio=model_info.get("supports_audio", False),
            details=model_info.get("details", {})
        )
    except NotImplementedError: raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding '{binding_name}' cannot provide current model info.")
    except Exception as e: logger.error(f"Error getting model info for '{binding_name}': {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get model info: {e}")
