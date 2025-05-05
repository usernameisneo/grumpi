# lollms_server/api/endpoints.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: Defines the FastAPI endpoints for the LoLLMs server API.
# Modification Date: 2025-05-04
# Updated endpoints based on refactoring plan for model info and default handling.

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Union, Optional
from pydantic import ValidationError
import importlib.metadata # To get version
from pathlib import Path # For Path type hint in dependencies

# Logging
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

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
    HealthResponse, StreamChunk, GenerateResponse,
    TokenizeRequest, TokenizeResponse, DetokenizeRequest, DetokenizeResponse,
    CountTokensRequest, CountTokensResponse,
    GetModelInfoResponse, # NEW model info response
    ListActiveBindingsResponse,
    GetDefaultBindingsResponse,
    # GetContextLengthResponse removed
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


# === Helper Function to get Binding ===
async def _get_binding(binding_instance_name: str, binding_manager: 'BindingManager') -> 'Binding':
    """
    Helper to get a binding instance by name.

    Raises HTTPException 404 if binding not found.
    """
    binding = binding_manager.get_binding(binding_instance_name)
    if not binding:
        logger.warning(f"Binding instance '{binding_instance_name}' requested but not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Binding instance '{binding_instance_name}' not found.")
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
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list bindings")

@router.get("/list_available_models/{binding_instance_name}",
            response_model=ListAvailableModelsResponse,
            summary="List Models Available to a Specific Binding Instance",
            description="Retrieves models recognized by a specific configured binding instance (e.g., local files, API models).",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing", "Models"])
async def list_available_models_for_binding(
    binding_instance_name: str,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Retrieves models recognized by a specific configured binding instance."""
    logger.info(f"Request received to list available models for binding instance: '{binding_instance_name}'")
    binding = await _get_binding(binding_instance_name, binding_manager) # Use helper

    try:
        # Binding method should return List[Dict] matching ModelInfo structure
        models_data: List[Dict[str, Any]] = await binding.list_available_models()
        validated_models: List[ModelInfo] = []
        for i, model_dict in enumerate(models_data):
            try:
                model_dict.setdefault('details', {}) # Ensure details exists
                validated_models.append(ModelInfo(**model_dict))
            except ValidationError as e:
                logger.warning(f"Failed to validate model data #{i} from '{binding_instance_name}': {model_dict}. Error: {e}. Skipping.")
            except Exception as e_inner:
                logger.warning(f"Error processing model data #{i} {model_dict.get('name','N/A')} for '{binding_instance_name}': {e_inner}. Skipping.", exc_info=True)

        return ListAvailableModelsResponse(binding_instance_name=binding_instance_name, models=validated_models)
    except NotImplementedError:
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_instance_name}' does not support listing available models.")
    except HTTPException as e: raise e # Re-raise 404 from helper
    except (RuntimeError, ValueError) as e:
        logger.error(f"Error getting models from binding instance '{binding_instance_name}': {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed get models from '{binding_instance_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error listing models for binding instance '{binding_instance_name}': {e}", exc_info=True)
        trace_exception(e)
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
        raw_list = personality_manager.list_personalities()
        personalities_info: Dict[str, PersonalityInfo] = {}
        for name, info_dict in raw_list.items():
            try:
                personalities_info[name] = PersonalityInfo(**info_dict)
            except ValidationError as e:
                logger.warning(f"Failed validate personality info for '{name}': {info_dict}. Error: {e}. Skipping.")
            except Exception as e_inner:
                logger.warning(f"Error processing personality info '{name}': {e_inner}. Skipping.", exc_info=True)
        return ListPersonalitiesResponse(personalities=personalities_info)
    except Exception as e:
        logger.error(f"Error listing personalities: {e}", exc_info=True)
        trace_exception(e)
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
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list functions")


@router.get("/list_models",
            response_model=ListModelsResponse,
            summary="List Discovered Models in Folder (File Scan)",
            description="Lists models found by scanning the configured main models folder subdirectories (e.g., 'gguf', 'diffusers_models'). This is a simple file scan, not binding-specific.",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing", "Models"])
async def list_models(config: 'ConfigGuard' = Depends(get_config_dependency)):
    """Lists models found by scanning the configured models folder."""
    try:
        models_folder_path_str = getattr(config.paths, "models_folder", None)
        if not models_folder_path_str:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Models folder path not configured.")
        models_folder_path = Path(models_folder_path_str) # Should be absolute
        return ListModelsResponse(models=_scan_models_folder(models_folder_path))
    except HTTPException: raise # Re-raise
    except Exception as e:
        logger.error(f"Error listing models from folder: {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list models from folder.")


@router.get("/list_active_bindings",
            response_model=ListActiveBindingsResponse,
            summary="List Successfully Loaded Binding Instances",
            description="Lists only the binding instances that were successfully configured and loaded by the server.",
            dependencies=[Depends(verify_api_key)],
            tags=["Listing", "Bindings"])
async def list_active_bindings(binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)):
    """Lists only the currently active and loaded binding instances."""
    try:
        instances_info = binding_manager.list_binding_instances()
        # Optional: Cleanup sensitive keys if not already handled by list_binding_instances
        # for instance_config in instances_info.values():
        #     instance_config.pop('api_key', None)
        #     instance_config.pop('google_api_key', None)
        return ListActiveBindingsResponse(bindings=instances_info)
    except Exception as e:
        logger.error(f"Error listing active bindings: {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list active bindings")


@router.get("/get_default_bindings",
            response_model=GetDefaultBindingsResponse,
            summary="Get Current Default Binding Instances & Parameters",
            description="Retrieves the currently configured default binding instance names for each modality (TTT, TTI, etc.) and general default parameters from the server configuration.",
            dependencies=[Depends(verify_api_key)],
            tags=["Defaults", "Bindings"])
async def get_default_bindings(config: 'ConfigGuard' = Depends(get_config_dependency)):
    """Gets the configured default binding names and parameters."""
    defaults_dict: Dict[str, Optional[Union[str, int]]] = {} # Allow int for numeric defaults
    modalities = ["ttt", "tti", "tts", "stt", "ttv", "ttm"]
    numeric_defaults = ["default_context_size", "default_max_output_tokens"]

    try:
        defaults_section = getattr(config, 'defaults', None)
        if not defaults_section:
            logger.warning("Defaults configuration section not found.")
            # Return empty dict or raise 404? Let's return empty for now.
            return GetDefaultBindingsResponse(defaults={})

        # Get default binding instance names
        for mod in modalities:
            binding_attr = f"{mod}_binding"
            # NOTE: Default model names are NO LONGER stored here
            defaults_dict[binding_attr] = getattr(defaults_section, binding_attr, None)

        # Get general numeric defaults
        for num_key in numeric_defaults:
             defaults_dict[num_key] = getattr(defaults_section, num_key, None)

        return GetDefaultBindingsResponse(defaults=defaults_dict)

    except AttributeError as e:
        logger.error(f"Error accessing defaults section in config: {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error reading defaults configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error getting default bindings: {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get default bindings")


# --- REMOVED /get_default_ttt_context_length endpoint ---


# === Model Information Endpoint (NEW) ===
@router.get("/get_model_info/{binding_instance_name}",
            response_model=GetModelInfoResponse,
            summary="Get Information about a Specific Model via a Binding",
            description="Retrieves standardized details (context size, capabilities) and binding-specific info about a model accessible through the specified binding instance. If `model_name` is omitted, returns info for the instance's default or currently loaded model.",
            dependencies=[Depends(verify_api_key)],
            tags=["Utilities", "Models", "Bindings"])
async def get_model_info(
    binding_instance_name: str,
    model_name: Optional[str] = Query(None, description="Optional: The specific model name/ID to query. If omitted, info for the instance's default/current model is returned."),
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep)
):
    """Retrieves detailed information about a specific model via the specified binding instance."""
    binding = await _get_binding(binding_instance_name, binding_manager) # Use helper

    try:
        # Call the binding's get_model_info method
        model_info_dict = await binding.get_model_info(model_name=model_name)

        # Basic validation: check if essential 'name' key is present
        if not model_info_dict.get("name"):
             logger.error(f"Binding '{binding_instance_name}' returned incomplete model info (missing 'name'): {model_info_dict}")
             # Determine if it was for a specific model request or default
             target = f"model '{model_name}'" if model_name else "default/current model"
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Could not retrieve valid info for {target} from binding '{binding_instance_name}'.")

        # Use Pydantic model for validation and response structure
        # Add binding instance name explicitly to the response data
        try:
            response_data = GetModelInfoResponse(
                binding_instance_name=binding_instance_name,
                **model_info_dict # Unpack the dictionary returned by the binding
            )
            return response_data
        except ValidationError as e:
            logger.error(f"Failed to validate model info from binding '{binding_instance_name}': {model_info_dict}. Error: {e}", exc_info=True)
            trace_exception(e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Binding returned invalid model info structure.")

    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_instance_name}' does not support providing detailed model info.")
    except HTTPException as e: raise e # Re-raise 404 from helper
    except Exception as e:
        logger.error(f"Error getting model info for binding instance '{binding_instance_name}' (Model: {model_name or 'Default'}): {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get model info: {e}")


# === Generation Endpoint (Requires API Key) ===
@router.post("/generate",
             # response_model=GenerateResponse, # Handled manually for streaming/non-streaming
             summary="Generate Multimodal Output",
             description="Main endpoint to trigger generation tasks (text, image, etc.) using specified inputs, personality, binding, and parameters. Supports streaming.",
             dependencies=[Depends(verify_api_key)],
             tags=["Generation"])
async def generate(
    request_payload: GenerateRequest,
    personality_manager: 'PersonalityManager' = Depends(get_personality_manager_dep),
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep),
    function_manager: 'FunctionManager' = Depends(get_function_manager_dep),
    resource_manager: 'ResourceManager' = Depends(get_resource_manager_dep),
    config: 'ConfigGuard' = Depends(get_config_dependency),
    http_request: Request = None # Original request object if needed
) -> Response:
    """Handles generation requests, supporting multimodal inputs and streaming."""
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

        if isinstance(result_response, (JSONResponse, StreamingResponse)):
            return result_response
        else:
            logger.error(f"Core generation logic returned unexpected type: {type(result_response)}")
            error_content = {"error": "Internal server error: Unexpected generation result type", "details": str(result_response)}
            return JSONResponse(content=error_content, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except HTTPException as e:
        logger.warning(f"HTTP Exception during generation: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception in /generate endpoint: {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred during generation.")


# === Tokenizer Utility Endpoints (Require API Key) ===
@router.post("/tokenize",
             response_model=TokenizeResponse,
             summary="Tokenize Text using a Binding Instance",
             description="Tokenizes the provided text using the tokenizer associated with the specified binding instance. Uses the instance's default model if `model_name` is omitted.",
             dependencies=[Depends(verify_api_key)],
             tags=["Utilities"])
async def tokenize_text(
    request: TokenizeRequest,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep),
    config: 'ConfigGuard' = Depends(get_config_dependency)
):
    """Tokenizes text using the specified binding instance."""
    binding_name = request.binding_name
    model_name = request.model_name # Can be None

    # Determine binding instance name if not provided
    if not binding_name:
        try:
            # Get default TTT binding instance name
            default_binding_attr = "ttt_binding"
            binding_name = getattr(config.defaults, default_binding_attr, None)
            if binding_name:
                logger.debug(f"Using default TTT binding instance '{binding_name}' for tokenize.")
            else:
                logger.error("No binding instance specified and no default TTT binding configured.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No binding instance specified or configured.")
        except AttributeError as e:
            logger.error(f"Error accessing default binding config: {e}.")
            trace_exception(e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error accessing server defaults.")

    # Get the binding instance
    binding = await _get_binding(binding_name, binding_manager) # Use helper

    try:
        # Call the binding's tokenize method, passing model_name (could be None)
        tokens = await binding.tokenize(request.text, request.add_bos, request.add_eos, model_name=model_name)
        return TokenizeResponse(tokens=tokens, count=len(tokens))
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_name}' does not support tokenization.")
    except HTTPException as e: raise e # Re-raise 404 from helper
    except Exception as e:
        logger.error(f"Error tokenizing text with binding '{binding_name}': {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tokenization failed: {e}")


@router.post("/detokenize",
             response_model=DetokenizeResponse,
             summary="Detokenize Tokens using a Binding Instance",
             description="Converts a list of token IDs back to text using the tokenizer associated with the specified binding instance. Uses the instance's default model if `model_name` is omitted.",
             dependencies=[Depends(verify_api_key)],
             tags=["Utilities"])
async def detokenize_tokens(
    request: DetokenizeRequest,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep),
    config: 'ConfigGuard' = Depends(get_config_dependency)
):
    """Detokenizes a list of token IDs using the specified binding instance."""
    binding_name = request.binding_name
    model_name = request.model_name # Can be None

    # Determine binding instance name if not provided
    if not binding_name:
        try:
            default_binding_attr = "ttt_binding"
            binding_name = getattr(config.defaults, default_binding_attr, None)
            if binding_name:
                logger.debug(f"Using default TTT binding instance '{binding_name}' for detokenize.")
            else:
                logger.error("No binding instance specified and no default TTT binding configured.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No binding instance specified or configured.")
        except AttributeError as e:
            logger.error(f"Error accessing default binding config: {e}.")
            trace_exception(e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error accessing server defaults.")

    # Get the binding instance
    binding = await _get_binding(binding_name, binding_manager) # Use helper

    try:
        # Call the binding's detokenize method, passing model_name (could be None)
        text = await binding.detokenize(request.tokens, model_name=model_name)
        return DetokenizeResponse(text=text)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_name}' does not support detokenization.")
    except HTTPException as e: raise e # Re-raise 404 from helper
    except Exception as e:
        logger.error(f"Error detokenizing tokens with binding '{binding_name}': {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Detokenization failed: {e}")


@router.post("/count_tokens",
             response_model=CountTokensResponse,
             summary="Count Tokens in Text using a Binding Instance",
             description="Counts the number of tokens in the provided text using the tokenizer associated with the specified binding instance. Uses the instance's default model if `model_name` is omitted.",
             dependencies=[Depends(verify_api_key)],
             tags=["Utilities"])
async def count_tokens(
    request: CountTokensRequest,
    binding_manager: 'BindingManager' = Depends(get_binding_manager_dep),
    config: 'ConfigGuard' = Depends(get_config_dependency)
):
    """Counts tokens in text using the specified binding instance."""
    binding_name = request.binding_name
    model_name = request.model_name # Can be None

    # Determine binding instance name if not provided
    if not binding_name:
        try:
            default_binding_attr = "ttt_binding"
            binding_name = getattr(config.defaults, default_binding_attr, None)
            if binding_name:
                logger.debug(f"Using default TTT binding instance '{binding_name}' for count_tokens.")
            else:
                logger.error("No binding instance specified and no default TTT binding configured.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No binding instance specified or configured.")
        except AttributeError as e:
            logger.error(f"Error accessing default binding config: {e}.")
            trace_exception(e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error accessing server defaults.")

    # Get the binding instance
    binding = await _get_binding(binding_name, binding_manager) # Use helper

    try:
        # Call the binding's count_tokens method (or tokenize if count_tokens not implemented)
        # Passing model_name which could be None
        if hasattr(binding, 'count_tokens') and callable(binding.count_tokens):
             tokens_count = await binding.count_tokens(request.text, add_bos=request.add_bos, add_eos=request.add_eos, model_name=model_name)
        else: # Fallback to tokenize
            tokens = await binding.tokenize(request.text, add_bos=request.add_bos, add_eos=request.add_eos, model_name=model_name)
            tokens_count = len(tokens)

        return CountTokensResponse(count=tokens_count)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Binding instance '{binding_name}' does not support tokenization required for counting.")
    except HTTPException as e: raise e # Re-raise 404 from helper
    except Exception as e:
        logger.error(f"Error counting tokens with binding '{binding_name}': {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Token counting failed: {e}")