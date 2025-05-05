# encoding:utf-8
# Project: lollms_server
# File: lollms_server/core/generation.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Handles the core generation logic, orchestrating personalities, bindings, and resources.

import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator, Union, Tuple, List # Added Tuple
from contextlib import asynccontextmanager, nullcontext

# FastAPI imports for responses and exceptions
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse

# ConfigGuard import for type hinting
# Use TYPE_CHECKING to avoid runtime import errors if ConfigGuard isn't strictly needed elsewhere
from typing import TYPE_CHECKING
from configguard import ConfigGuard
import secrets


# Logging library
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

# Core components
from lollms_server.core.bindings import BindingManager, Binding
from lollms_server.core.personalities import PersonalityManager, Personality
from lollms_server.core.functions import FunctionManager
from lollms_server.core.resource_manager import ResourceManager
# API Models
from lollms_server.api.models import GenerateRequest, StreamChunk, InputData, OutputData, GenerateResponse
# Utils
from lollms_server.utils.helpers import encode_base64 # Keep relevant helpers

logger = logging.getLogger(__name__)


# --- Model Loading Context Manager ---
class ModelLoadingError(Exception):
    """Custom exception for model loading failures."""
    pass

@asynccontextmanager
async def manage_model_loading(binding: Binding, model_name: str):
    """
    Asynchronous context manager to handle model loading via the binding,
    using the binding's internal lock and resource management.

    Args:
        binding: The Binding instance responsible for the model.
        model_name: The name of the model to load.
    """
    load_success = False
    if not binding:
        raise ModelLoadingError("Binding instance is None, cannot load model.")

    # Check if the correct model is already loaded
    if binding.is_model_loaded and binding.model_name == model_name:
        logger.debug(f"Model '{model_name}' already loaded on binding '{binding.binding_instance_name}'. Skipping load.")
        try:
            yield # Proceed without loading
            return
        finally:
            pass # No unloading needed as we didn't load

    # If different model or no model loaded, proceed with load attempt
    logger.info(f"Requesting model load: Binding='{binding.binding_instance_name}', Model='{model_name}'")
    try:
        # The binding's load_model should handle its internal lock and resource manager acquisition
        load_success = await binding.load_model(model_name)
        if not load_success:
            logger.error(f"Binding '{binding.binding_instance_name}' failed to load model '{model_name}'.")
            raise ModelLoadingError(f"Binding '{binding.binding_instance_name}' failed to load model '{model_name}'.")

        logger.info(f"Model '{model_name}' confirmed loaded by binding '{binding.binding_instance_name}'.")
        yield # Yield control to the caller after successful load
    finally:
        # Decide on unloading strategy - for now, we keep models loaded within the binding lifecycle.
        # The binding's cleanup/unload_model method handles unloading later.
        # If you wanted to unload immediately after the 'with' block:
        # if load_success:
        #     logger.info(f"Requesting unload for model '{model_name}' on binding '{binding.binding_instance_name}' after use.")
        #     await binding.unload_model()
        pass # Keep loaded for now

# --- Standardize Output Helper ---
def standardize_output(raw_output: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[OutputData]:
    """Converts raw binding/script output into the standard List[OutputData] format."""
    output_list = []
    if isinstance(raw_output, str):
        # Simple text output
        output_list.append(OutputData(type="text", data=raw_output))
    elif isinstance(raw_output, dict):
        # Try to map common single-output dicts
        mapped = False
        output_type = raw_output.get("type")
        output_data = raw_output.get("data")
        output_thoughts = raw_output.get("thoughts")
        output_metadata = raw_output.get("metadata", {})
        output_mime = raw_output.get("mime_type")

        # Check if it already matches OutputData structure reasonably well
        if output_type in ["text", "image", "audio", "video", "json", "error", "info"] and output_data is not None:
             try:
                 # --- ADDED: Pass thoughts to OutputData constructor ---
                 output_list.append(OutputData(type=output_type, data=output_data, thoughts=output_thoughts, mime_type=output_mime, metadata=output_metadata))
                 # -------------------------------------------------------
                 mapped = True
             except Exception as e:
                 logger.warning(f"Failed to directly validate dict as OutputData: {raw_output}. Error: {e}. Trying fallback.")

        # Fallback mapping (add thoughts if possible)
        if not mapped:
            # --- ADDED: Include thoughts in fallbacks ---
            if "text" in raw_output:
                output_list.append(OutputData(type="text", data=raw_output["text"], thoughts=raw_output.get("thoughts"), metadata=raw_output.get("metadata", {})))
                mapped = True
            elif "image_base64" in raw_output: # Images unlikely to have thoughts
                output_list.append(OutputData(type="image", data=raw_output["image_base64"], mime_type=raw_output.get("mime_type"), metadata=raw_output.get("metadata", {})))
                mapped = True
            elif "audio_base64" in raw_output:
                output_list.append(OutputData(type="audio", data=raw_output["audio_base64"], mime_type=raw_output.get("mime_type"), metadata=raw_output.get("metadata", {})))
                mapped = True
            elif "video_base64" in raw_output:
                output_list.append(OutputData(type="video", data=raw_output["video_base64"], mime_type=raw_output.get("mime_type"), metadata=raw_output.get("metadata", {})))
                mapped = True

        if not mapped:
            # Final fallback for unknown dict structure: wrap as JSON
            logger.warning(f"Standardizing unknown dict structure as JSON: {list(raw_output.keys())}")
            output_list.append(OutputData(type="json", data=raw_output)) # Store the whole dict

    elif isinstance(raw_output, list):
        # Assume it's already a list of OutputData-like dicts
        # Perform validation/conversion for each item
        for i, item in enumerate(raw_output):
            if isinstance(item, dict) and "type" in item and "data" in item:
                 try:
                     # Pydantic will automatically pick up 'thoughts' if present in the dict item
                     output_list.append(OutputData(**item))
                 except Exception as e:
                     logger.warning(f"Failed to validate item {i} in output list as OutputData: {item}. Error: {e}. Skipping.")
            else:
                 logger.warning(f"Skipping invalid item {i} in output list (not dict or missing keys): {item}")
    else:
        # Handle unexpected types
        logger.warning(f"Standardizing unexpected output type {type(raw_output)} as text.")
        output_list.append(OutputData(type="text", data=str(raw_output)))

    # Ensure at least one output item exists if input was valid but conversion failed
    if not output_list and raw_output is not None:
         logger.error("Standardization resulted in empty list from non-empty input. Adding error output.")
         output_list.append(OutputData(type="error", data="Internal error: Failed to standardize generation output."))

    return output_list


# --- Main process_generation_request function ---
async def process_generation_request(
    request: GenerateRequest,
    personality_manager: PersonalityManager,
    binding_manager: BindingManager,
    function_manager: FunctionManager,
    resource_manager: ResourceManager,
    config: 'ConfigGuard' # Use TYPE_CHECKING hint
) -> Union[JSONResponse, StreamingResponse]:
    """
    Processes a generation request, handling multimodal input, personality selection,
    binding execution (scripted or direct), resource management, and response formatting.
    """
    request_id = secrets.token_hex(8) # Basic request ID
    logger.debug(f"--- Start process_generation_request (ID: {request_id}) --- Type: {request.generation_type}, Stream: {request.stream}")
    start_time = time.time()
    personality: Optional[Personality] = None

    # --- 0. Extract Inputs from Request ---
    input_data: List[InputData] = request.input_data
    # Find first item designated as the main user prompt
    text_prompt_items = [item for item in input_data if item.type == 'text' and item.role == 'user_prompt']
    primary_text_prompt = text_prompt_items[0].data if text_prompt_items else ""
    if not primary_text_prompt and not any(item.type != 'text' for item in input_data): # Log warning only if no text AND no other modality
        logger.warning(f"Request {request_id}: No primary text prompt found in input_data with role 'user_prompt'. Proceeding with empty prompt.")

    # Separate multimodal data for binding call
    multimodal_data_for_binding: List[InputData] = [ item for item in input_data if item.type != 'text' ]
    logger.debug(f"Request {request_id}: Primary text prompt: '{primary_text_prompt[:100]}...'")
    if multimodal_data_for_binding:
         logger.info(f"Request {request_id}: Found {len(multimodal_data_for_binding)} multimodal items: {[f'{item.type}/{item.role}' for item in multimodal_data_for_binding]}")
    # --- End Input Extraction ---


    # 1. Get Personality (If requested)
    if request.personality:
        personality = personality_manager.get_personality(request.personality)
        if not personality:
            logger.error(f"Request {request_id}: Personality '{request.personality}' requested but not found or disabled.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Personality '{request.personality}' not found or disabled.")
        logger.info(f"Request {request_id}: Using personality: {request.personality}")
    else:
        logger.info(f"Request {request_id}: No personality specified.")

    # 2. Determine Binding Instance and Model Name
    # This function now uses the ConfigGuard config object
    binding_name, model_name = _determine_binding_and_model(request, personality, config)
    logger.debug(f"Request {request_id}: Determined Binding Instance='{binding_name}', Model Name='{model_name}'")
    if not binding_name:
         # _determine_binding_and_model logs the error
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not determine binding instance for the request. Check defaults or request parameters.")

    # 3. Get Binding Instance
    binding = binding_manager.get_binding(binding_name) # Get instance by name
    if not binding:
        # Binding manager logs specific load errors
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Binding instance '{binding_name}' not available or failed to load.")
    logger.debug(f"Request {request_id}: Using binding instance: '{binding.binding_instance_name}' (Type: {binding.binding_type_name})")

    # 4. Check Binding Capabilities vs Input Data
    logger.debug(f"Request {request_id}: Checking binding capabilities for instance '{binding_name}'...")
    supported_inputs = binding.get_supported_input_modalities()
    for item in input_data:
        if item.type not in supported_inputs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Binding instance '{binding_name}' does not support input type '{item.type}'.")
        # Check role support (optional, binding can refine this)
        if not binding.supports_input_role(item.type, item.role):
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Binding instance '{binding_name}' does not support input type '{item.type}' with role '{item.role}'.")
    logger.debug(f"Request {request_id}: Binding capabilities check passed.")

    # 5. Prepare Generation Parameters
    merged_params = {}
    logger.debug(f"Request {request_id}: Applying global default parameters...")
    try: # Use safe access with getattr for ConfigGuard
        merged_params['max_tokens'] = getattr(config.defaults, "default_max_output_tokens", 1024)
        merged_params['context_size'] = getattr(config.defaults, "default_context_size", 4096)
    except AttributeError:
        logger.warning("Could not read global defaults from config. Defaults section might be missing or incomplete.")

    personality_conditioning = None
    # Extract system context provided in the request itself
    system_context_items = [item for item in input_data if item.type == 'text' and item.role == 'system_context']
    system_context_block = "\n\n".join([item.data.strip() for item in system_context_items if item.data]).strip()
    system_context_block = system_context_block + "\n\n" if system_context_block else ""

    # Apply personality defaults/conditioning
    if personality:
        logger.debug(f"Request {request_id}: Applying defaults/conditioning from personality '{personality.name}'...")
        p_config = personality.config # Access the Pydantic model
        personality_conditioning = p_config.personality_conditioning
        # Override defaults if set in personality config
        if p_config.model_temperature is not None: merged_params['temperature'] = p_config.model_temperature
        if p_config.model_n_predicts is not None: merged_params['max_tokens'] = p_config.model_n_predicts
        if p_config.model_top_k is not None: merged_params['top_k'] = p_config.model_top_k
        if p_config.model_top_p is not None: merged_params['top_p'] = p_config.model_top_p
        if p_config.model_repeat_penalty is not None: merged_params['repeat_penalty'] = p_config.model_repeat_penalty
        if p_config.model_repeat_last_n is not None: merged_params['repeat_last_n'] = p_config.model_repeat_last_n
        # Add other mappable personality params here

    # Apply request-specific parameters, potentially overriding personality/defaults
    if request.parameters:
        logger.debug(f"Request {request_id}: Applying request parameters...")
        merged_params.update(request.parameters)

    # Construct final system message (Context Block + Personality Conditioning / Request Override)
    request_system_message = merged_params.get("system_message") # Check if request params included sys msg
    final_system_message_content = None
    if request_system_message is not None:
        final_system_message_content = request_system_message # Request overrides personality
        logger.debug(f"Request {request_id}: Using system message override from request parameters.")
    elif personality_conditioning is not None:
        final_system_message_content = personality_conditioning # Use personality conditioning
        logger.debug(f"Request {request_id}: Using system message from personality '{personality.name}'.")

    final_system_message = f"{system_context_block}{final_system_message_content if final_system_message_content else ''}".strip()
    # Set the final system message in params, ensuring it's not empty string if nothing was provided
    merged_params['system_message'] = final_system_message if final_system_message else None

    # Log final parameters (excluding potentially long system message)
    loggable_params = {k:v for k,v in merged_params.items() if k != 'system_message'}
    logger.info(f"Request {request_id}: Final merged parameters: {loggable_params}")
    if merged_params.get('system_message'): logger.debug(f"Request {request_id}: Final system msg (start): {merged_params['system_message'][:200]}...")


    # 6. Prepare Context for Scripted Personalities & Binding
    request_info_dict = { "personality": personality.name if personality else None, "functions": request.functions, "generation_type": request.generation_type, "request_id": request_id }
    generation_context = { "request": request, # Pass original request object
                           "input_data": input_data, # Pass processed input data list
                           "personality": personality,
                           "binding": binding,
                           "function_manager": function_manager,
                           "binding_manager": binding_manager,
                           "resource_manager": resource_manager,
                           "config": config,
                           "request_info": request_info_dict } # Pass request details separately


    # 7. Execute Generation
    try:
        is_scripted_request = personality and personality.is_scripted
        if is_scripted_request and not personality:
             # This check is slightly redundant due to earlier check, but belts and suspenders
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Scripted generation requested but personality object is invalid.")

        output_raw: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None # Holds the raw result for non-streaming

        # Determine the effective model name for loading context manager
        # Priority: Request -> Binding's currently loaded -> Binding's default from config? -> Error?
        # Let's simplify: load_model *must* be called with a specific name.
        # _determine_binding_and_model provides the target model name.
        # If model_name is None after that, the binding must handle it (use internal default).
        effective_model_name_for_load = model_name or binding.model_name or binding.default_model_name# Best guess for context manager name
        logger.debug(f"Request {request_id}: Entering manage_model_loading for Binding='{binding.binding_instance_name}', effective_model_name='{effective_model_name_for_load}'")
        # The context manager now uses binding.load_model which handles resource management internally
        async with manage_model_loading(binding, effective_model_name_for_load):
            logger.debug(f"Request {request_id}: Inside manage_model_loading context. Binding: {binding.binding_instance_name}, Model loaded: {binding.is_model_loaded}, Loaded Name: {binding.model_name}")

            # Ensure the *actually* loaded model name is used if manage_model_loading performed a load
            actual_model_name_used = binding.model_name

            # --- Scripted Personality Path ---
            if is_scripted_request:
                logger.info(f"Request {request_id}: Executing scripted workflow: '{personality.name}'")
                # Pass the full generation context to the workflow
                script_output_raw = await personality.run_workflow( prompt=primary_text_prompt, params=merged_params, context=generation_context )

                # --- Handle Streaming Response from Script ---
                if request.stream and isinstance(script_output_raw, AsyncGenerator):
                    logger.info(f"Request {request_id}: Script returned stream. Handling SSE.")
                    async def script_event_stream():
                        final_stream_content_list: List[Dict] = [] # Accumulate content for the final message
                        try:
                            async for chunk_data in script_output_raw:
                                if not isinstance(chunk_data, dict): # Simple non-dict yield -> treat as text chunk
                                     chunk_data_dict = {"type": "chunk", "content": str(chunk_data)}
                                else: chunk_data_dict = chunk_data # Assume it's StreamChunk-like
                                try:
                                    chunk = StreamChunk(**chunk_data_dict) # Validate chunk structure
                                    # Don't add final chunk content to accumulation here
                                    if chunk.type != "final":
                                        yield f"data: {chunk.model_dump_json()}\n\n"
                                    # If it IS the final chunk from script, extract its content
                                    if chunk.type == "final" and chunk.content:
                                         # Assume chunk.content is List[OutputData]-like
                                         if isinstance(chunk.content, list): final_stream_content_list = chunk.content
                                         else: final_stream_content_list = [chunk.content] # Wrap single item
                                except Exception as pydantic_error:
                                    logger.error(f"Request {request_id}: Invalid chunk from script: {chunk_data_dict}. Error: {pydantic_error}", exc_info=True)
                                    error_chunk = StreamChunk(type="error", content=f"Script stream error: {pydantic_error}")
                                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                            # --- Send the standardized final chunk AFTER loop ---
                            # The script should yield a final chunk with List[OutputData] structure
                            # If script ended without yielding final, final_stream_content_list might be empty
                            # standardize_output ensures a valid list structure
                            standardized_final_list = standardize_output(final_stream_content_list or [])
                            final_sse_chunk = StreamChunk(type="final", content=standardized_final_list, metadata={"request_id": request_id})
                            yield f"data: {final_sse_chunk.model_dump_json()}\n\n"
                            logger.info(f"Request {request_id}: Scripted stream finished.")
                        except Exception as e:
                            logger.error(f"Request {request_id}: Error during script stream processing: {e}", exc_info=True)
                            error_chunk = StreamChunk(type="error", content=f"Script stream error: {e}")
                            try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                            except Exception: pass # Avoid error during error reporting
                    return StreamingResponse(script_event_stream(), media_type="text/event-stream")
                else: # Non-streaming script result
                    output_raw = script_output_raw

            # --- Direct Binding Path ---
            else:
                logger.info(f"Request {request_id}: Executing direct generation: Binding='{binding_name}', Type='{request.generation_type}'")
                if request.stream:
                     logger.info(f"Request {request_id}: Starting stream generation via binding '{binding_name}'...")
                     binding_stream_generator = binding.generate_stream( prompt=primary_text_prompt, params=merged_params, request_info=request_info_dict, multimodal_data=multimodal_data_for_binding )
                     async def binding_event_stream():
                         final_stream_content_list: List[Dict] = [] # Accumulate for final message
                         try:
                             async for chunk_data in binding_stream_generator:
                                 if not isinstance(chunk_data, dict): # Basic safety
                                     chunk_data_dict = {"type": "chunk", "content": str(chunk_data)}
                                 else: chunk_data_dict = chunk_data
                                 try:
                                     chunk = StreamChunk(**chunk_data_dict) # Validate
                                     if chunk.type != "final": yield f"data: {chunk.model_dump_json()}\n\n"
                                     # Extract final content if present
                                     if chunk.type == "final" and chunk.content:
                                         if isinstance(chunk.content, list): final_stream_content_list = chunk.content
                                         else: final_stream_content_list = [chunk.content] # Wrap single item
                                 except Exception as pydantic_error:
                                     logger.error(f"Request {request_id}: Invalid chunk from binding '{binding_name}': {chunk_data_dict}. Error: {pydantic_error}", exc_info=True)
                                     error_chunk = StreamChunk(type="error", content=f"Binding stream error: {pydantic_error}")
                                     yield f"data: {error_chunk.model_dump_json()}\n\n"
                             # --- Send standardized final chunk ---
                             standardized_final_list = standardize_output(final_stream_content_list or [])
                             final_sse_chunk = StreamChunk(type="final", content=standardized_final_list, metadata={"request_id": request_id})
                             yield f"data: {final_sse_chunk.model_dump_json()}\n\n"
                             logger.info(f"Request {request_id}: Binding stream '{binding_name}' finished.")
                         except Exception as e:
                             logger.error(f"Request {request_id}: Error during binding stream '{binding_name}': {e}", exc_info=True)
                             error_chunk = StreamChunk(type="error", content=f"Binding stream error: {e}")
                             try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                             except Exception: pass
                     return StreamingResponse(binding_event_stream(), media_type="text/event-stream")
                else: # Non-streaming binding call
                     logger.info(f"Request {request_id}: Starting non-stream generation via binding '{binding_name}'...")
                     binding_output_raw = await binding.generate( prompt=primary_text_prompt, params=merged_params, request_info=request_info_dict, multimodal_data=multimodal_data_for_binding )
                     output_raw = binding_output_raw

        # --- End Generation Execution ---
        execution_time = time.time() - start_time
        logger.info(f"Request {request_id}: Generation processed in {execution_time:.2f} seconds.")

        # --- Standardize Non-Streaming Response Formatting ---
        if output_raw is not None: # Only process if it wasn't a streaming request that returned early
            final_output_list: List[OutputData] = standardize_output(output_raw)
            # Ensure list is not empty after standardization
            if not final_output_list:
                 logger.error(f"Request {request_id}: Generation resulted in empty standardized output list.")
                 final_output_list.append(OutputData(type="error", data="Generation failed to produce valid output.", metadata={"request_id": request_id}))

            # Use the GenerateResponse model
            response_payload = GenerateResponse(
                personality=personality.name if personality else None,
                output=final_output_list,
                execution_time=execution_time,
                request_id=request_id
            )
            # Use model_dump for Pydantic v2+; ensure non_defaults=True if needed
            return JSONResponse(content=response_payload.model_dump(exclude_none=True))
        elif not request.stream:
             # This case indicates non-streaming finished but output_raw is still None
             logger.error(f"Request {request_id}: Non-streaming request finished but output is None.")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generation failed to produce output.")
        # If it was a streaming request, the StreamingResponse was already returned earlier, so we don't reach here.

    # --- Exception Handling ---
    except ModelLoadingError as e:
         logger.error(f"Request {request_id}: ModelLoadingError: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except asyncio.TimeoutError as e:
         logger.error(f"Request {request_id}: Operation timed out: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Operation timed out: {e}")
    except HTTPException as e: # Re-raise specific HTTP exceptions
         # Log context specific to this request
         logger.warning(f"Request {request_id}: HTTP Exception during generation: Status={e.status_code}, Detail={e.detail}")
         raise e
    except NotImplementedError as e:
         logger.error(f"Request {request_id}: Functionality not implemented: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except ValueError as e: # Catch bad params, validation, safety blocks etc.
         logger.warning(f"Request {request_id}: ValueError during generation: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad request: {e}")
    except Exception as e:
        logger.critical(f"Request {request_id}: Unhandled exception during generation: {e}", exc_info=True)
        trace_exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during generation: {e}")

    # This path should ideally not be reached if logic is correct
    logger.error(f"Request {request_id}: Reached end of process_generation_request unexpectedly.")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error processing generation request.")

# --- Helper to determine binding and model ---
def _determine_binding_and_model(
    request: GenerateRequest,
    personality: Optional[Personality],
    config: 'ConfigGuard' # Use TYPE_CHECKING hint
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determines the target binding instance name and model name based on request,
    personality (future), and global defaults from the ConfigGuard object.
    """
    gen_type = request.generation_type
    # Get parameters directly from request first
    binding_name = request.binding_name
    model_name = request.model_name

    logger.debug(f"Determining binding/model: Type='{gen_type}'. Request: Binding='{binding_name}', Model='{model_name}'.")

    # --- Personality Overrides (Placeholder - Future Enhancement) ---
    # if personality:
    #     # Logic to get binding/model preference from personality config for this gen_type
    #     # personality_binding = personality.get_preference('binding', gen_type)
    #     # personality_model = personality.get_preference('model', gen_type)
    #     # binding_name = binding_name or personality_binding
    #     # model_name = model_name or personality_model
    #     pass

    # --- Fallback to Global Defaults (using ConfigGuard object) ---
    if not binding_name:
        try:
            # Get the default binding *instance name* for the requested type (ttt_binding, tti_binding, etc.)
            default_binding_attr = f"{gen_type}_binding"
            binding_name = getattr(config.defaults, default_binding_attr, None)
            if binding_name:
                logger.debug(f"Using default binding instance '{binding_name}' for type '{gen_type}' from config.defaults.{default_binding_attr}")
            else:
                logger.warning(f"No binding instance specified in request and no default '{default_binding_attr}' found in main config.")
        except AttributeError as e:
             logger.error(f"Error accessing default binding config: {e}. Defaults section might be incomplete.")
             binding_name = None # Ensure it's None if access fails

    if not model_name:
        try:
            # Get the default model name for the requested type (ttt_model, tti_model, etc.)
            default_model_attr = f"{gen_type}_model"
            model_name = getattr(config.defaults, default_model_attr, None)
            if model_name:
                logger.debug(f"Using default model '{model_name}' for type '{gen_type}' from config.defaults.{default_model_attr}")
            else:
                # Allow model_name to be None, the binding instance might have an internal default or require one explicitly
                logger.debug(f"No model specified in request and no default '{default_model_attr}' found. Binding must handle.")
        except AttributeError as e:
             logger.error(f"Error accessing default model config: {e}. Defaults section might be incomplete.")
             model_name = None # Ensure it's None if access fails

    # Final check - we absolutely need a binding instance name
    if not binding_name:
        logger.error(f"Could not determine a binding instance for generation type '{gen_type}'. Please specify in request or configure defaults.")
        return None, None # Return None, None to indicate failure

    logger.info(f"Selected for generation: Type='{gen_type}', Binding Instance='{binding_name}', Model Name='{model_name or 'Binding Default'}'")
    return binding_name, model_name


# --- Helper to scan models folder ---
def _scan_models_folder(models_base_path: Path) -> Dict[str, List[str]]:
    """ Scans the models subfolders for model files/folders. """
    # Expanded list of potential type folders
    discovered_models = {"ttt": [], "tti": [], "ttv": [], "ttm": [], "tts": [], "stt": [], "audio2audio": [], "i2i": [], "gguf": [], "diffusers_models": []}
    if not models_base_path or not models_base_path.is_dir():
            logger.warning(f"Models base path '{models_base_path}' not found or not a directory. Cannot scan for models.")
            return discovered_models # Return empty structure

    logger.debug(f"Scanning models folder: {models_base_path}")
    for model_type in discovered_models.keys():
        type_folder = models_base_path / model_type
        if type_folder.is_dir():
            try:
                # List items, excluding hidden ones (starting with '.')
                models = [item.name for item in type_folder.iterdir() if not item.name.startswith('.')]
                discovered_models[model_type] = sorted(models) # Sort for consistency
                if models: logger.debug(f"Found models in {type_folder.name}: {models}")
            except Exception as e:
                logger.error(f"Error scanning models folder {type_folder}: {e}")
        # We don't create folders here anymore, _ensure_directories in config does that

    return discovered_models