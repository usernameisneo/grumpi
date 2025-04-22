# lollms_server/api/endpoints.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response # Keep Response
from fastapi.responses import StreamingResponse, JSONResponse # Keep JSONResponse
import logging
from typing import List, Dict, Any, Union
from pydantic import ValidationError
# --- IMPORT: Use updated models ---
from .models import (
    GenerateRequest, ListBindingsResponse, ListPersonalitiesResponse, PersonalityInfo,
    ListFunctionsResponse, ListModelsResponse, ModelInfo, ListAvailableModelsResponse, InputData # Make sure InputData is here if needed elsewhere
)
# Core components (NO direct import from main.py)
from lollms_server.core.config import AppConfig
from lollms_server.core.security import verify_api_key
from lollms_server.core.bindings import BindingManager
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


# === Listing Endpoints (Ensure ModelInfo conforms in Phase 7) ===

@router.get("/list_bindings", response_model=ListBindingsResponse, summary="List Available Bindings", dependencies=[Depends(verify_api_key)])
async def list_bindings(binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    try:
        types = binding_manager.list_binding_types()
        instances = binding_manager.list_binding_instances()
        return ListBindingsResponse(binding_types=types, binding_instances=instances)
    except Exception as e:
        logger.error(f"Error listing bindings: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list bindings")


@router.get("/list_available_models/{binding_name}", response_model=ListAvailableModelsResponse, summary="List Models Available to a Specific Binding", dependencies=[Depends(verify_api_key)])
async def list_available_models_for_binding(binding_name: str, binding_manager: BindingManager = Depends(get_binding_manager_dep)):
    """
    Retrieves the list of models recognized or usable by a specific
    configured binding instance. The returned model information includes
    standardized fields and capability flags (e.g., supports_vision).
    """
    logger.info(f"Request received to list available models for binding: '{binding_name}'")
    binding = binding_manager.get_binding(binding_name)
    if not binding:
        logger.error(f"Binding '{binding_name}' requested for model listing not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding '{binding_name}' not found or not configured.")
    try:
        # Binding's list_available_models MUST return data conforming to ModelInfo
        models_data: List[Dict[str, Any]] = await binding.list_available_models()
        validated_models = []
        for model_dict in models_data:
            try:
                # Ensure 'details' exists if not provided by binding
                if 'details' not in model_dict: model_dict['details'] = {}
                # Attempt validation with the potentially updated ModelInfo
                model_info_obj = ModelInfo(**model_dict)
                validated_models.append(model_info_obj)
            except ValidationError as e:
                logger.warning(f"Failed to validate model data from binding '{binding_name}' against ModelInfo: {model_dict}. Error: {e}. Skipping.")
            except Exception as e_inner:
                 logger.warning(f"Error processing model data {model_dict} for binding '{binding_name}': {e_inner}. Skipping.")
        return ListAvailableModelsResponse(binding_name=binding_name, models=validated_models)
    except (RuntimeError, ValueError, NotImplementedError) as e:
        logger.error(f"Error getting available models from binding '{binding_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve models from binding '{binding_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error listing available models for binding '{binding_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred while listing models for binding '{binding_name}'.")


@router.get("/list_personalities", response_model=ListPersonalitiesResponse, summary="List Available Personalities", dependencies=[Depends(verify_api_key)])
async def list_personalities(personality_manager: PersonalityManager = Depends(get_personality_manager_dep)):
    try:
        raw_list = personality_manager.list_personalities()
        personalities_info = {}
        for name, info_dict in raw_list.items():
            try:
                # Set defaults for validation if keys might be missing from manager list
                info_dict.setdefault('category', None)
                info_dict.setdefault('tags', [])
                info_dict.setdefault('icon', None)
                info_dict.setdefault('language', None)
                personalities_info[name] = PersonalityInfo(**info_dict)
            except ValidationError as e:
                 logger.warning(f"Failed to validate personality data for '{name}' against PersonalityInfo: {info_dict}. Error: {e}. Skipping.")
            except Exception as e_inner:
                 logger.warning(f"Error processing personality data {info_dict} for '{name}': {e_inner}. Skipping.")
        return ListPersonalitiesResponse(personalities=personalities_info)
    except Exception as e:
        logger.error(f"Error listing personalities: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list personalities")


@router.get("/list_functions", response_model=ListFunctionsResponse, summary="List Available Functions", dependencies=[Depends(verify_api_key)])
async def list_functions(function_manager: FunctionManager = Depends(get_function_manager_dep)):
    try:
        functions = function_manager.list_functions()
        return ListFunctionsResponse(functions=functions)
    except Exception as e:
        logger.error(f"Error listing functions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list functions")


@router.get("/list_models", response_model=ListModelsResponse, summary="List Discovered Models", dependencies=[Depends(verify_api_key)])
async def list_models(config: AppConfig = Depends(get_config_dep)):
    """Lists models found in the configured models folder."""
    try:
        models_found = _scan_models_folder(config.paths.models_folder)
        return ListModelsResponse(models=models_found)
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list models")


# === Generation Endpoint ===

@router.post("/generate",
             summary="Generate Text, Image, Video, Music, Speech etc.",
             description="Triggers a generation task using optional personality, multimodal inputs, and specified parameters. Accepts input data via the `input_data` list.",
             response_model=None, # Set to None as return type varies (JSON or SSE stream)
             dependencies=[Depends(verify_api_key)])
async def generate(
    request_payload: GenerateRequest, # Uses updated GenerateRequest model
    # Inject managers (unchanged)
    personality_manager: PersonalityManager = Depends(get_personality_manager_dep),
    binding_manager: BindingManager = Depends(get_binding_manager_dep),
    function_manager: FunctionManager = Depends(get_function_manager_dep),
    resource_manager: ResourceManager = Depends(get_resource_manager_dep),
    config: AppConfig = Depends(get_config_dep),
    http_request: Request = None # Keep if needed by FastAPI internally
) -> Response: # Return type always a Response subtype now
    """
    Handles generation requests, supporting multimodal inputs and diverse generation types.
    Returns either a JSONResponse or a StreamingResponse.
    """
    # Basic logging of the request type
    logger.info(f"Received generation request (Type: {request_payload.generation_type}, Stream: {request_payload.stream})")
    # Log input types and roles without logging the actual data
    if request_payload.input_data:
         input_summary = [f"{item.type}/{item.role}" for item in request_payload.input_data]
         logger.info(f"Input Data Summary: {input_summary}")
    else:
         logger.warning("Received generate request with empty input_data list.")


    try:
        # process_generation_request now accepts the updated GenerateRequest
        # and internally handles the new structure and binding calls.
        # It's expected to return JSONResponse or StreamingResponse directly.
        result_response = await process_generation_request(
            request=request_payload,
            personality_manager=personality_manager,
            binding_manager=binding_manager,
            function_manager=function_manager,
            resource_manager=resource_manager,
            config=config
        )

        # --- Check the type returned by process_generation_request ---
        if isinstance(result_response, (JSONResponse, StreamingResponse)):
            return result_response
        else:
            # This indicates an internal error in process_generation_request
            logger.error(f"process_generation_request returned unexpected type: {type(result_response)}")
            try:
                 # Attempt to wrap it for a consistent error response
                 error_content = {"error": "Internal server error: Unexpected generation result type", "details": str(result_response)}
                 return JSONResponse(content=error_content, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception:
                 # Fallback if the result itself is not serializable
                 return JSONResponse(content={"error": "Internal server error: Non-serializable unexpected generation result type"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except HTTPException as e:
        # Re-raise known HTTP exceptions (like 4xx/5xx errors from processing)
        logger.warning(f"HTTP Exception during generation processing: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        # Catch unexpected errors during the endpoint handling itself
        logger.error(f"Unhandled exception in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected internal server error occurred.")