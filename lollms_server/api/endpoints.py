# lollms_server/api/endpoints.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import logging
from typing import List, Dict, Any, Union
from pydantic import ValidationError
# Models
from .models import (
    GenerateRequest, ListBindingsResponse, ListPersonalitiesResponse, PersonalityInfo,
    ListFunctionsResponse, ListModelsResponse, GenerateResponse, ModelInfo, ListAvailableModelsResponse
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

# --- API Dependencies ---
# Define functions here to get managers from app state via request

def get_config_dep(request: Request) -> AppConfig:
    if not hasattr(request.app.state, 'config'):
        # This might happen if a request comes in before lifespan startup finishes
        raise HTTPException(status_code=503, detail="Server configuration not ready")
    return request.app.state.config

def get_binding_manager_dep(request: Request) -> BindingManager:
    if not hasattr(request.app.state, 'binding_manager'):
        raise HTTPException(status_code=503, detail="Binding Manager not ready")
    return request.app.state.binding_manager

def get_personality_manager_dep(request: Request) -> PersonalityManager:
    if not hasattr(request.app.state, 'personality_manager'):
        raise HTTPException(status_code=503, detail="Personality Manager not ready")
    return request.app.state.personality_manager

def get_function_manager_dep(request: Request) -> FunctionManager:
    if not hasattr(request.app.state, 'function_manager'):
        raise HTTPException(status_code=503, detail="Function Manager not ready")
    return request.app.state.function_manager

def get_resource_manager_dep(request: Request) -> ResourceManager:
    if not hasattr(request.app.state, 'resource_manager'):
        raise HTTPException(status_code=503, detail="Resource Manager not ready")
    return request.app.state.resource_manager


# === Listing Endpoints ===

@router.get("/list_bindings",
            response_model=ListBindingsResponse,
            summary="List Available Bindings",
            description="Lists all discovered binding types and configured binding instances.",
            dependencies=[Depends(verify_api_key)])
async def list_bindings(
    binding_manager: BindingManager = Depends(get_binding_manager_dep)
):
    try:
        types = binding_manager.list_binding_types()
        instances = binding_manager.list_binding_instances() # This needs to be sync or the manager method async
        # Let's assume list_binding_instances is sync for now
        return ListBindingsResponse(binding_types=types, binding_instances=instances)
    except Exception as e:
        logger.error(f"Error listing bindings: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list bindings")


@router.get("/list_available_models/{binding_name}",
            response_model=ListAvailableModelsResponse,
            summary="List Models Available to a Specific Binding",
            description="Retrieves the list of models recognized or usable by a specific configured binding instance.",
            dependencies=[Depends(verify_api_key)])
async def list_available_models_for_binding(
    binding_name: str,
    binding_manager: BindingManager = Depends(get_binding_manager_dep)
):
    logger.info(f"Request received to list available models for binding: '{binding_name}'")
    binding = binding_manager.get_binding(binding_name)
    if not binding:
        # ... (error handling) ...
        raise HTTPException(...)

    try:
        # Binding now returns List[Dict] where each dict has 'name'
        # and potentially other standardized keys plus a 'details' dict.
        models_data: List[Dict[str, Any]] = await binding.list_available_models()

        # --- Validate and Structure using Pydantic ---
        validated_models = []
        for model_dict in models_data:
            try:
                # Pydantic will automatically map keys from model_dict
                # to the fields in ModelInfo. Any extra keys not defined
                # in ModelInfo will be ignored unless captured by 'details'
                # if we added model_config = {"extra": "allow"} to ModelInfo,
                # but our bindings now put extras into the 'details' key themselves.

                # Ensure 'details' exists if not provided by binding, Pydantic needs it
                if 'details' not in model_dict:
                    model_dict['details'] = {}

                model_info_obj = ModelInfo(**model_dict)
                validated_models.append(model_info_obj)
            except ValidationError as e:
                logger.warning(f"Failed to validate model data from binding '{binding_name}' against ModelInfo model: {model_dict}. Error: {e}. Skipping.")
            except Exception as e_inner: # Catch other potential errors during Pydantic processing
                 logger.warning(f"Error processing model data {model_dict} for binding '{binding_name}': {e_inner}. Skipping.")

        # --- End Validation ---

        return ListAvailableModelsResponse(
            binding_name=binding_name,
            models=validated_models # List of Pydantic ModelInfo objects
        )
    except (RuntimeError, ValueError, NotImplementedError) as e:
        # Catch errors raised by the binding's implementation
        logger.error(f"Error getting available models from binding '{binding_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models from binding '{binding_name}': {e}"
        )
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Unexpected error listing available models for binding '{binding_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while listing models for binding '{binding_name}'."
        )

@router.get("/list_personalities",
            response_model=ListPersonalitiesResponse,
            summary="List Available Personalities",
            description="Lists all loaded personalities.",
            dependencies=[Depends(verify_api_key)])
async def list_personalities(
    personality_manager: PersonalityManager = Depends(get_personality_manager_dep)
):
    try:
        # Convert Personality objects to PersonalityInfo Pydantic models
        raw_list = personality_manager.list_personalities() # Assume this returns Dict[str, Dict] for now
        personalities_info = { name: PersonalityInfo(**info) for name, info in raw_list.items() }
        return ListPersonalitiesResponse(personalities=personalities_info)
    except Exception as e:
        logger.error(f"Error listing personalities: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list personalities")

@router.get("/list_functions",
            response_model=ListFunctionsResponse,
            summary="List Available Functions",
            description="Lists all discovered custom functions.",
            dependencies=[Depends(verify_api_key)])
async def list_functions(
    function_manager: FunctionManager = Depends(get_function_manager_dep)
):
    try:
        functions = function_manager.list_functions()
        return ListFunctionsResponse(functions=functions)
    except Exception as e:
        logger.error(f"Error listing functions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list functions")

@router.get("/list_models",
            response_model=ListModelsResponse,
            summary="List Discovered Models",
            description="Lists models found in the configured models folder.",
            dependencies=[Depends(verify_api_key)])
async def list_models(
    config: AppConfig = Depends(get_config_dep)
):
    try:
        # Scan the models folder structure
        models_found = _scan_models_folder(config.paths.models_folder)
        return ListModelsResponse(models=models_found)
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list models")


# === Generation Endpoint ===

@router.post("/generate",
             summary="Generate Text, Image, Video or Music",
             description="Triggers a generation task using a specified personality and prompt.",
             dependencies=[Depends(verify_api_key)])
async def generate(
    request_payload: GenerateRequest, # Renamed to avoid conflict with fastapi Request
    # Inject managers using new dependency functions
    personality_manager: PersonalityManager = Depends(get_personality_manager_dep),
    binding_manager: BindingManager = Depends(get_binding_manager_dep),
    function_manager: FunctionManager = Depends(get_function_manager_dep),
    resource_manager: ResourceManager = Depends(get_resource_manager_dep), # Injection is correct here
    config: AppConfig = Depends(get_config_dep),
    http_request: Request = None
):
    """
    Handles generation requests.

    Supports different generation types (TTT, TTI, TTV, TTM) and optional streaming for TTT.
    """
    logger.info(f"Received generation request for personality '{request_payload.personality}' (Type: {request_payload.generation_type}, Stream: {request_payload.stream})")

    try:
        result = await process_generation_request(
            request=request_payload,
            personality_manager=personality_manager,
            binding_manager=binding_manager,
            function_manager=function_manager,
            resource_manager=resource_manager, # <-- ENSURE THIS IS PASSED
            config=config
        )

        # process_generation_request returns:
        # - StreamingResponse for TTT stream
        # - str for TTT non-stream
        # - Dict for TTI/V/M or complex/scripted output

        if isinstance(result, StreamingResponse):
            logger.debug("Returning StreamingResponse.")
            return result
        elif isinstance(result, str): # Raw text for non-streaming TTT
                logger.debug("Returning raw text response.")
                # FastAPI automatically sets content-type to text/plain
                return Response(content=result, media_type="text/plain")
        elif isinstance(result, dict): # JSON for non-TTT or complex responses
                logger.debug("Returning JSON response.")
                return JSONResponse(content=result)
        else:
                # Should not happen if process_generation_request is correct
                logger.error(f"Unexpected return type from generation process: {type(result)}")
                raise HTTPException(status_code=500, detail="Internal server error: Unexpected generation result type.")

    except HTTPException as e:
        # Re-raise known HTTP exceptions from the generation process
        logger.warning(f"HTTP Exception during generation: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        # Catch unexpected errors during the endpoint handling itself
        logger.error(f"Unhandled exception in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred.")
