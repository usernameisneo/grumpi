# lollms_server/api/endpoints.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: Defines the FastAPI endpoints for the LoLLMs server API.

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Union, Optional
from pydantic import ValidationError
import importlib.metadata # To get version
from pathlib import Path # For Path type hint in dependencies

# Logging
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors # For potential use
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore

# ConfigGuard and Core Components Dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from configguard import ConfigGuard
    except ImportError: ConfigGuard = Any # type: ignore
    # Import core manager types for dependency hinting
    from lollms_server.core.bindings import BindingManager, Binding
    from lollms_server.core.personalities import PersonalityManager
    from lollms_server.core.functions import FunctionManager
    from lollms_server.core.resource_manager import ResourceManager

# API Models
from lollms_server.api.models import (
    GenerateRequest, ListBindingsResponse, ListPersonalitiesResponse, PersonalityInfo,
    ListFunctionsResponse, ListModelsResponse, ModelInfo, ListAvailableModelsResponse, InputData,
    HealthResponse, StreamChunk, GenerateResponse, # Added GenerateResponse
    TokenizeRequest, TokenizeResponse, DetokenizeRequest, DetokenizeResponse,
    CountTokensRequest, CountTokensResponse, GetModelInfoResponse # Added new models
)
# Core components functions for actual logic (dependencies resolved via FastAPI)
from lollms_server.core.security import verify_api_key, get_config_dependency # Use dependency getter
from lollms_server.core.generation import process_generation_request, _scan_models_folder # Import generation logic

logger = logging.getLogger(__name__)
router = APIRouter()

# --- API Dependencies (Functions to get managers from app state) ---
# These functions are used by FastAPI's dependency injection system (`Depends(...)`)

def get_binding_manager_dep(request: Request) -> 'BindingManager':
    """Dependency to get the BindingManager instance."""
    manager = getattr(request.app.state, 'binding_manager', None)
    if manager is None:
        logger.error("Binding Manager not found in app state.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Binding Manager not ready")
    return manager

def get_personality_manager_dep(request: Request) -> 'PersonalityManager':
    """Dependency to get the PersonalityManager instance."""
    manager = getattr(request.app.state, 'personality_manager', None)
    if manager is None:
        logger.error("Personality Manager not found in app state.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Personality Manager not ready")
    return manager

def get_function_manager_dep(request: Request) -> 'FunctionManager':
    """Dependency to get the FunctionManager instance."""
    manager = getattr(request.app.state, 'function_manager', None)
    if manager is None:
        logger.error("Function Manager not found in app state.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Function Manager not ready")
    return manager

def get_resource_manager_dep(request: Request) -> 'ResourceManager':
    """Dependency to get the ResourceManager instance."""
    manager = getattr(request.app.state, 'resource_manager', None)
    if manager is None:
        logger.error("Resource Manager not found in app state.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Resource Manager not ready")
    return manager

# Config dependency uses get_config_dependency from security module
# Security dependency uses verify_api_key from security module


# === Helper Function to get Binding and check loaded model ===
async def _get_binding_and_check_loaded(binding_instance_name: str, binding_manager: 'BindingManager') -> 'Binding':
    """
    Helper to get a binding instance by name and check if a model is loaded.

    Raises HTTPException 404 if binding not found, 409 if model not loaded.
    """
    binding = binding_manager.get_binding(binding_instance_name)
    if not binding:
        logger.warning(f"Binding instance '{binding_instance_name}' requested but not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding instance '{binding_instance_name}' not found.")

    # Check the binding's internal state if a model should be loaded for this operation
    if not binding.is_model_loaded:
        logger.warning(f"Operation requires loaded model, but none loaded for binding instance '{binding_instance_name}'.")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"No model is currently loaded for binding instance '{binding_instance_name}'. Load one via /generate first or check binding state.")
    return binding


# === Server Info Endpoint ===
@router.get("/health",
            response_model=HealthResponse,
            summary="Check Server Health and Configuration Status",
            description="Provides server status, version, and API key requirement.",
            tags=["Server Info"])
async def health_check(config: 'ConfigGuard' = Depends(get_config_dependency)):
    """Checks server health and returns basic configuration status."""
    is_key_required = False
    server_version = "N/A"
    try:
        # Access config via ConfigGuard attributes/items safely
        security_section = getattr(config, 'security', None)
        if security_section:
            allowed_keys = getattr(security_section, "allowed_api_keys", [])
            is_key_required = bool(allowed_keys) # True if list is not empty
    except Exception as e:
         logger.error(f"Error accessing security config during health check: {e}")
         # Default to secure assumption if config access fails
         is_key_required = True

    try:
        server_version = importlib.metadata.version("lollms_server")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("Could not determine lollms_server version via importlib.")
        # Optionally try reading from pyproject.toml or a VERSION file as fallback

    return HealthResponse( status="ok", api_key_required=is_key_required, version=server_version )


# === Listing Endpoints (Require API Key) ===
@router.get("/list_bindings",
            response_model=ListBindingsResponse,
            summary="List Discovered Binding Types and Configured Instances",
            description="Lists discovered binding types (from code/cards) and configured binding instances (from main config map and instance files).",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing"])
async def list_bindings(binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)):
    """Lists discovered binding types and configured binding instances."""
    try:
        types_info = binding_manager.list_binding_types()
        instances_info = binding_manager.list_binding_instances()
        # Remove sensitive info like api_key before sending
        for instance_config in instances_info.values():
            instance_config.pop('api_key', None)
            instance_config.pop('google_api_key', None) # Example for Gemini
        return ListBindingsResponse(binding_types=types_info, binding_instances=instances_info)
    except Exception as e:
        logger.error(f"Error listing bindings: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list bindings")

@router.get("/list_available_models/{binding_instance_name}",
            response_model=ListAvailableModelsResponse,
            summary="List Models Available to a Specific Binding Instance",
            description="Retrieves models recognized by a specific configured binding instance (e.g., local files, API models).",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing"])
async def list_available_models_for_binding(
    binding_instance_name: str,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Retrieves models recognized by a specific configured binding instance."""
    logger.info(f"Request received to list available models for binding instance: '{binding_instance_name}'")
    binding = binding_manager.get_binding(binding_instance_name)
    if not binding:
        # get_binding logs the error
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding instance '{binding_instance_name}' not found.")

    try:
        # Binding method should return List[Dict] matching ModelInfo structure
        models_data: List[Dict[str, Any]] = await binding.list_available_models()
        validated_models: List[ModelInfo] = []
        for i, model_dict in enumerate(models_data):
            try:
                # Ensure details is a dict if missing
                model_dict.setdefault('details', {})
                validated_models.append(ModelInfo(**model_dict))
            except ValidationError as e:
                logger.warning(f"Failed to validate model data #{i} from '{binding_instance_name}': {model_dict}. Error: {e}. Skipping.")
            except Exception as e_inner:
                logger.warning(f"Error processing model data #{i} {model_dict.get('name','N/A')} for '{binding_instance_name}': {e_inner}. Skipping.", exc_info=True)

        return ListAvailableModelsResponse(binding_instance_name=binding_instance_name, models=validated_models)
    except NotImplementedError:
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_instance_name}' does not support listing available models.")
    except (RuntimeError, ValueError) as e:
        logger.error(f"Error getting models from binding instance '{binding_instance_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed get models from '{binding_instance_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error listing models for binding instance '{binding_instance_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error listing models for '{binding_instance_name}'.")


@router.get("/list_personalities",
            response_model=ListPersonalitiesResponse,
            summary="List Available Personalities",
            description="Lists loaded and enabled personalities based on configuration.",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing"])
async def list_personalities(personality_manager: 'PersonalityManager' = Depends(get_personality_manager_dep)):
    """Lists loaded and enabled personalities."""
    try:
        # Manager now returns Dict[str, Dict], needs validation against PersonalityInfo
        raw_list = personality_manager.list_personalities()
        personalities_info: Dict[str, PersonalityInfo] = {}
        for name, info_dict in raw_list.items():
            try:
                # Pydantic model will handle missing optional fields with defaults or None
                personalities_info[name] = PersonalityInfo(**info_dict)
            except ValidationError as e:
                logger.warning(f"Failed validate personality info for '{name}': {info_dict}. Error: {e}. Skipping.")
            except Exception as e_inner:
                logger.warning(f"Error processing personality info '{name}': {e_inner}. Skipping.", exc_info=True)
        return ListPersonalitiesResponse(personalities=personalities_info)
    except Exception as e:
        logger.error(f"Error listing personalities: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list personalities")


@router.get("/list_functions",
            response_model=ListFunctionsResponse,
            summary="List Available Custom Functions",
            description="Lists discovered custom Python functions available for scripted personalities.",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing"])
async def list_functions(function_manager: 'FunctionManager' = Depends(get_function_manager_dep)):
    """Lists discovered custom Python functions."""
    try:
        return ListFunctionsResponse(functions=function_manager.list_functions())
    except Exception as e:
        logger.error(f"Error listing functions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list functions")


@router.get("/list_models",
            response_model=ListModelsResponse,
            summary="List Discovered Models in Folder (File Scan)",
            description="Lists models found by scanning the configured main models folder subdirectories (e.g., 'gguf', 'diffusers_models'). This is a simple file scan, not binding-specific.",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing"])
async def list_models(config: 'ConfigGuard' = Depends(get_config_dependency)):
    """Lists models found by scanning the configured models folder."""
    try:
        # Access the models_folder path from the ConfigGuard object
        models_folder_path_str = getattr(config.paths, "models_folder", None)
        if not models_folder_path_str:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Models folder path not configured.")
        models_folder_path = Path(models_folder_path_str) # Should be absolute
        # Pass the absolute path to the scanning function
        return ListModelsResponse(models=_scan_models_folder(models_folder_path))
    except HTTPException: # Re-raise
        raise
    except Exception as e:
        logger.error(f"Error listing models from folder: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list models from folder.")


# === Generation Endpoint (Requires API Key) ===
@router.post("/generate",
             # response_model=GenerateResponse, # Handled manually for streaming/non-streaming
             summary="Generate Multimodal Output",
             description="Main endpoint to trigger generation tasks (text, image, etc.) using specified inputs, personality, binding, and parameters. Supports streaming.",
             dependencies=[Depends(verify_api_key)],
             tags=["Generation"])
async def generate(
    request_payload: GenerateRequest, # Request body validated by Pydantic
    # Get managers via dependency injection
    personality_manager: 'PersonalityManager' = Depends(get_personality_manager_dep),
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep),
    function_manager: 'FunctionManager' = Depends(get_function_manager_dep),
    resource_manager: 'ResourceManager' = Depends(get_resource_manager_dep),
    config: 'ConfigGuard' = Depends(get_config_dependency),
    http_request: Request = None # Original request object if needed by logic
) -> Response: # Return type is Response for flexibility (JSON or Streaming)
    """Handles generation requests, supporting multimodal inputs and streaming."""
    # InputData validation now happens in the GenerateRequest model

    logger.info(f"Received generation request (Type: {request_payload.generation_type}, Stream: {request_payload.stream}, Binding: {request_payload.binding_name or 'Default'})")
    if request_payload.input_data:
        input_summary = [f"{item.type}/{item.role}" for item in request_payload.input_data]
        logger.debug(f"Input Data Summary: {input_summary}")
    else: # Should be caught by validator, but log defensively
        logger.error("Received generate request with invalid empty input_data list after validation.")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Internal error: input_data is empty.")

    try:
        # Call the core generation logic function
        result_response = await process_generation_request(
                                        request=request_payload,
                                        personality_manager=personality_manager,
                                        binding_manager=binding_manager,
                                        function_manager=function_manager,
                                        resource_manager=resource_manager,
                                        config=config )

        # Check the type of response returned by the core logic
        if isinstance(result_response, (JSONResponse, StreamingResponse)):
            return result_response # Return directly if already a valid FastAPI response
        else:
            # This case indicates an unexpected return type from process_generation_request
            logger.error(f"Core generation logic returned unexpected type: {type(result_response)}")
            error_content = {"error": "Internal server error: Unexpected generation result type", "details": str(result_response)}
            return JSONResponse(content=error_content, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except HTTPException as e:
        # Log and re-raise HTTP exceptions originating from core logic
        logger.warning(f"HTTP Exception during generation: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        # Catch any other unhandled exceptions during processing
        logger.error(f"Unhandled exception in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred during generation.")


# === Tokenizer Utility Endpoints (Require API Key) ===
@router.post("/tokenize",
             response_model=TokenizeResponse,
             summary="Tokenize Text using a Binding Instance",
             description="Tokenizes the provided text using the tokenizer associated with the specified binding instance's *currently loaded* model.",
             dependencies=[Depends(verify_api_key)],
             tags=["Utilities"])
async def tokenize_text(
    request: TokenizeRequest,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Tokenizes text using the specified binding instance's currently loaded model."""
    try:
        # Use helper to get binding and check if model is loaded
        binding = await _get_binding_and_check_loaded(request.binding_name, binding_manager)
        # Call the binding's tokenize method
        tokens = await binding.tokenize(request.text, request.add_bos, request.add_eos)
        return TokenizeResponse(tokens=tokens, count=len(tokens))
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{request.binding_name}' does not support tokenization.")
    except HTTPException as e: raise e # Re-raise 404/409 from helper
    except Exception as e:
        logger.error(f"Error tokenizing text with binding '{request.binding_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tokenization failed: {e}")


@router.post("/detokenize",
             response_model=DetokenizeResponse,
             summary="Detokenize Tokens using a Binding Instance",
             description="Converts a list of token IDs back to text using the tokenizer associated with the specified binding instance's *currently loaded* model.",
             dependencies=[Depends(verify_api_key)],
             tags=["Utilities"])
async def detokenize_tokens(
    request: DetokenizeRequest,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Detokenizes a list of token IDs using the specified binding instance's currently loaded model."""
    try:
        # Use helper to get binding and check if model is loaded
        binding = await _get_binding_and_check_loaded(request.binding_name, binding_manager)
        # Call the binding's detokenize method
        text = await binding.detokenize(request.tokens)
        return DetokenizeResponse(text=text)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{request.binding_name}' does not support detokenization.")
    except HTTPException as e: raise e # Re-raise 404/409 from helper
    except Exception as e:
        logger.error(f"Error detokenizing tokens with binding '{request.binding_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Detokenization failed: {e}")


@router.post("/count_tokens",
             response_model=CountTokensResponse,
             summary="Count Tokens in Text using a Binding Instance",
             description="Counts the number of tokens in the provided text using the tokenizer associated with the specified binding instance's *currently loaded* model.",
             dependencies=[Depends(verify_api_key)],
             tags=["Utilities"])
async def count_tokens(
    request: CountTokensRequest,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Counts tokens in text using the specified binding instance's currently loaded model."""
    try:
        # Use helper to get binding and check if model is loaded
        binding = await _get_binding_and_check_loaded(request.binding_name, binding_manager)
        # Call tokenize internally (typically don't need BOS/EOS for counting)
        tokens = await binding.tokenize(request.text, add_bos=False, add_eos=False)
        return CountTokensResponse(count=len(tokens))
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{request.binding_name}' does not support tokenization required for counting.")
    except HTTPException as e: raise e # Re-raise 404/409 from helper
    except Exception as e:
        logger.error(f"Error counting tokens with binding '{request.binding_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Token counting failed: {e}")


# === Model Info Utility Endpoint (Requires API Key) ===
@router.get("/get_model_info/{binding_instance_name}",
            response_model=GetModelInfoResponse,
            summary="Get Information about the Currently Loaded Model",
            description="Retrieves details (like context size) about the model currently active on the specified binding instance.",
            dependencies=[Depends(verify_api_key)],
            tags=["Utilities"])
async def get_model_info(
    binding_instance_name: str,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Retrieves information about the currently loaded model for a binding instance."""
    binding = binding_manager.get_binding(binding_instance_name)
    if not binding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding instance '{binding_instance_name}' not found.")

    try:
        # Binding method should return Dict matching GetModelInfoResponse structure
        model_info_dict = await binding.get_current_model_info()

        # Use Pydantic model for validation and response structure
        # Add binding instance name to the response explicitly
        response_data = GetModelInfoResponse(
            binding_instance_name=binding_instance_name,
            **model_info_dict # Unpack the dictionary returned by the binding
        )
        return response_data
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_instance_name}' cannot provide current model info.")
    except Exception as e:
        logger.error(f"Error getting model info for binding instance '{binding_instance_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get model info: {e}")