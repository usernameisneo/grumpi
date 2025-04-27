# lollms_server/core/generation.py
import ascii_colors as logging
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator, Union, Tuple, List # Add Tuple
from contextlib import asynccontextmanager, nullcontext
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse # Add JSONResponse

from .config import AppConfig
from .bindings import BindingManager, Binding
from .personalities import PersonalityManager, Personality
from .functions import FunctionManager
from .resource_manager import ResourceManager
# --- IMPORT: InputData and GenerateRequest ---
from lollms_server.api.models import GenerateRequest, StreamChunk, InputData, OutputData, GenerateResponse # Added OutputData and GenerateResponse
from lollms_server.utils.helpers import encode_base64

logger = logging.getLogger(__name__)


@asynccontextmanager
async def manage_model_loading(binding: Binding, model_name: str):
    """Context manager to handle model loading with resource management."""
    load_success = False
    try:
        logger.info(f"Requesting model load: Binding='{binding.binding_name}', Model='{model_name}'")
        load_success = await binding.load_model(model_name)
        if not load_success:
            logger.error(f"Binding '{binding.binding_name}' failed to load model '{model_name}'.")
            raise ModelLoadingError(f"Binding '{binding.binding_name}' failed to load model '{model_name}'.")
        logger.info(f"Model '{model_name}' confirmed loaded by binding '{binding.binding_name}'.")
        yield
    finally:
        # Keep models loaded for now
        pass


class ModelLoadingError(Exception):
    """Custom exception for model loading failures."""
    pass

# --- Standardize Output Helper (Ensure it's defined HERE) ---
def standardize_output(raw_output: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[OutputData]:
    """Converts raw binding/script output into the standard List[OutputData] format."""
    output_list = []
    if isinstance(raw_output, str):
        # Simple text output
        output_list.append(OutputData(type="text", data=raw_output))
    elif isinstance(raw_output, dict):
        # Try to map common single-output dicts
        mapped = False
        if "text" in raw_output:
            output_list.append(OutputData(type="text", data=raw_output["text"], metadata=raw_output.get("metadata", {})))
            mapped = True
        elif "image_base64" in raw_output:
            output_list.append(OutputData(type="image", data=raw_output["image_base64"], mime_type=raw_output.get("mime_type"), metadata=raw_output.get("metadata", {})))
            mapped = True
        elif "audio_base64" in raw_output:
            output_list.append(OutputData(type="audio", data=raw_output["audio_base64"], mime_type=raw_output.get("mime_type"), metadata=raw_output.get("metadata", {})))
            mapped = True
        elif "video_base64" in raw_output:
            output_list.append(OutputData(type="video", data=raw_output["video_base64"], mime_type=raw_output.get("mime_type"), metadata=raw_output.get("metadata", {})))
            mapped = True
        # Add mappings for other potential single outputs (json, error, info)
        elif raw_output.get("type") in ["json", "error", "info"]:
             output_list.append(OutputData(type=raw_output["type"], data=raw_output.get("data"), metadata=raw_output.get("metadata", {})))
             mapped = True

        if not mapped:
            # Fallback for unknown dict structure: wrap as JSON
            logger.warning(f"Standardizing unknown dict structure as JSON: {list(raw_output.keys())}")
            output_list.append(OutputData(type="json", data=raw_output))
    elif isinstance(raw_output, list):
        # Assume it's already a list of OutputData-like dicts
        # Perform basic validation/conversion
        for item in raw_output:
            if isinstance(item, dict) and "type" in item and "data" in item:
                 try:
                     # Validate and create OutputData instance
                     output_list.append(OutputData(**item))
                 except Exception as e:
                     logger.warning(f"Failed to validate item in output list as OutputData: {item}. Error: {e}. Skipping.")
            else:
                 logger.warning(f"Skipping invalid item in output list: {item}")
    else:
        # Handle unexpected types
        logger.warning(f"Standardizing unexpected output type {type(raw_output)} as text.")
        output_list.append(OutputData(type="text", data=str(raw_output)))

    # Ensure at least one output item exists if input was valid
    if not output_list and raw_output is not None:
         logger.error("Standardization resulted in empty list from non-empty input. Adding error output.")
         output_list.append(OutputData(type="error", data="Internal error: Failed to standardize generation output."))

    return output_list


# --- Main process_generation_request function ---
async def process_generation_request(
    request: GenerateRequest, # Now uses the updated model
    personality_manager: PersonalityManager,
    binding_manager: BindingManager,
    function_manager: FunctionManager,
    resource_manager: ResourceManager,
    config: AppConfig
) -> Union[JSONResponse, StreamingResponse]: # Always return JSON or Stream
    """
    Processes a generation request, handling personality logic, binding selection,
    parameter merging, resource management, multimodal input placeholders, and generation execution.
    """
    start_time = time.time()
    personality: Optional[Personality] = None

    # --- 0. Extract Inputs from Request ---
    # The validator in GenerateRequest already moved text_prompt into input_data
    input_data: List[InputData] = request.input_data
    # Find the first item designated as the main user prompt
    text_prompt_items = [item for item in input_data if item.type == 'text' and item.role == 'user_prompt']
    primary_text_prompt = text_prompt_items[0].data if text_prompt_items else "" # Use first user prompt
    if not primary_text_prompt:
        logger.warning("No primary text prompt found in input_data with role 'user_prompt'. Using empty string.")

    # For Phase 1, pass all non-text items. Phase 6 will add filtering based on binding caps.
    # Let's rename this more clearly for the binding call
    multimodal_data_for_binding: List[InputData] = [
        item for item in input_data if item.type != 'text'
    ]
    logger.debug(f"Extracted primary text prompt: '{primary_text_prompt[:100]}...'")
    if multimodal_data_for_binding:
         logger.info(f"Found {len(multimodal_data_for_binding)} multimodal input items: {[f'{item.type}/{item.role}' for item in multimodal_data_for_binding]}")
    # --- End Input Extraction ---


    # 1. Get Personality (If requested)
    if request.personality:
        personality = personality_manager.get_personality(request.personality)
        if not personality:
            logger.error(f"Personality '{request.personality}' requested but not found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Personality '{request.personality}' not found.")
        logger.info(f"Using personality: {request.personality}")
    else:
        logger.info("No personality specified, proceeding with generic generation.")

    # 2. Determine Binding and Model
    binding_name, model_name = _determine_binding_and_model(request, personality, config)
    if not binding_name: # Model name can be optional for some bindings
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not determine binding for the request.")

    # 3. Get Binding Instance
    binding = binding_manager.get_binding(binding_name)
    if not binding:
        logger.error(f"Binding instance '{binding_name}' not found or failed to instantiate.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Binding '{binding_name}' not available.")

    # --- Phase 6 Placeholder: Check Binding Capabilities ---
    # logger.info(f"Binding '{binding_name}' supports inputs: {binding.get_supported_input_modalities()}")
    # logger.info(f"Binding '{binding_name}' supports outputs: {binding.get_supported_output_modalities()}")
    # for item in multimodal_data_for_binding:
    #     if not binding.supports_input_role(item.type, item.role):
    #          raise HTTPException(status_code=400, detail=f"Binding '{binding_name}' does not support input type '{item.type}' with role '{item.role}'")
    # --- End Placeholder ---

    # 4. Prepare Generation Parameters
    merged_params = {}
    logger.debug("Applying global default parameters...")
    merged_params['max_tokens'] = config.defaults.default_max_output_tokens
    merged_params['context_size'] = config.defaults.default_context_size

    personality_conditioning = None
    system_context_items = [item for item in input_data if item.type == 'text' and item.role == 'system_context']
    system_context_block = "\n\n".join([item.data.strip() for item in system_context_items if item.data]).strip()
    system_context_block = system_context_block + "\n\n" if system_context_block else ""

    if personality:
        logger.debug(f"Applying defaults/conditioning from '{personality.name}'...")
        p_config = personality.config
        personality_conditioning = p_config.personality_conditioning
        if p_config.model_temperature is not None: merged_params['temperature'] = p_config.model_temperature
        if p_config.model_n_predicts is not None: merged_params['max_tokens'] = p_config.model_n_predicts
        # ... (other personality param overrides) ...

    request_system_message = (request.parameters or {}).get("system_message")
    final_system_message_content = None
    if request_system_message is not None: final_system_message_content = request_system_message
    elif personality_conditioning is not None: final_system_message_content = personality_conditioning

    final_system_message = f"{system_context_block}{final_system_message_content if final_system_message_content else ''}"
    merged_params['system_message'] = final_system_message.strip() if final_system_message and final_system_message.strip() else None

    if request.parameters:
        logger.debug("Applying remaining request parameters...")
        for key, value in request.parameters.items():
             if key != 'system_message': merged_params[key] = value

    loggable_params = {k:v for k,v in merged_params.items() if k != 'system_message'}
    logger.info(f"Final merged parameters: {loggable_params}")
    if merged_params.get('system_message'): logger.debug(f"Final sys msg (start): {merged_params['system_message'][:200]}...")

    request_info = { "personality": personality.name if personality else None, "functions": request.functions, "generation_type": request.generation_type, "request_id": None }
    generation_context = { "request": request, "personality": personality, "binding": binding, "function_manager": function_manager, "binding_manager": binding_manager, "resource_manager": resource_manager, "config": config, "input_data": input_data }

    # 5. Execute Generation
    try:
        is_scripted_request = personality and personality.is_scripted
        if is_scripted_request and not personality: raise HTTPException(status_code=500, detail="Scripted generation needs valid personality.")

        output: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None # Holds the raw result for non-streaming

        effective_model_name = model_name or binding.model_name or "binding_default" # Ensure a name for context manager

        async with manage_model_loading(binding, effective_model_name):
            if is_scripted_request:
                logger.info(f"Executing scripted workflow: '{personality.name}'")
                script_output = await personality.run_workflow( primary_text_prompt=primary_text_prompt, params=merged_params, context=generation_context )

                # --- Handle Streaming Response from Script ---
                if request.stream and isinstance(script_output, AsyncGenerator):
                    logger.info("Script returned stream. Handling SSE.")
                    async def script_event_stream():
                        final_stream_content: List[Dict] = [] # Accumulate content for the final message
                        try:
                            async for chunk_data in script_output:
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
                                         if isinstance(chunk.content, list): final_stream_content = chunk.content # Trust script returned list
                                         else: final_stream_content = [chunk.content] # Wrap single item
                                except Exception as pydantic_error:
                                    logger.error(f"Invalid chunk from script: {chunk_data_dict}. Error: {pydantic_error}", exc_info=True)
                                    error_chunk = StreamChunk(type="error", content=f"Script stream error: {pydantic_error}")
                                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                            # --- Send the standardized final chunk AFTER loop ---
                            standardized_final_list = standardize_output(final_stream_content or []) # Standardize accumulated/extracted content
                            final_sse_chunk = StreamChunk(type="final", content=standardized_final_list, metadata={}) # TODO: Include metadata from script if possible
                            yield f"data: {final_sse_chunk.model_dump_json()}\n\n"
                            logger.info("Scripted stream finished.")
                        except Exception as e:
                            logger.error(f"Error during script stream processing: {e}", exc_info=True)
                            error_chunk = StreamChunk(type="error", content=f"Script stream error: {e}")
                            try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                            except Exception: pass
                    return StreamingResponse(script_event_stream(), media_type="text/event-stream")
                # --- End Script Streaming Handling ---
                else: # Non-streaming script result
                    output = script_output
            else:
                # --- Direct Binding Path ---
                logger.info(f"Executing direct generation: Binding='{binding_name}', Type='{request.generation_type}'")
                if request.stream:
                     logger.info("Starting stream generation via binding...")
                     binding_stream_generator = binding.generate_stream( prompt=primary_text_prompt, params=merged_params, request_info=request_info, multimodal_data=multimodal_data_for_binding )
                     async def binding_event_stream():
                         final_stream_content: List[Dict] = [] # Accumulate for final message
                         try:
                             async for chunk_data in binding_stream_generator:
                                 if not isinstance(chunk_data, dict): chunk_data_dict = {"type": "chunk", "content": str(chunk_data)}
                                 else: chunk_data_dict = chunk_data
                                 try:
                                     chunk = StreamChunk(**chunk_data_dict)
                                     if chunk.type != "final": yield f"data: {chunk.model_dump_json()}\n\n"
                                     if chunk.type == "final" and chunk.content:
                                         if isinstance(chunk.content, list): final_stream_content = chunk.content
                                         else: final_stream_content = [chunk.content]
                                 except Exception as pydantic_error:
                                     logger.error(f"Invalid chunk from binding: {chunk_data_dict}. Error: {pydantic_error}", exc_info=True)
                                     error_chunk = StreamChunk(type="error", content=f"Binding stream error: {pydantic_error}")
                                     yield f"data: {error_chunk.model_dump_json()}\n\n"
                             # --- Send standardized final chunk ---
                             standardized_final_list = standardize_output(final_stream_content or [])
                             final_sse_chunk = StreamChunk(type="final", content=standardized_final_list, metadata={}) # TODO: metadata from binding?
                             yield f"data: {final_sse_chunk.model_dump_json()}\n\n"
                             logger.info("Binding stream finished.")
                         except Exception as e:
                             logger.error(f"Error during binding stream: {e}", exc_info=True)
                             error_chunk = StreamChunk(type="error", content=f"Binding stream error: {e}")
                             try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                             except Exception: pass
                     return StreamingResponse(binding_event_stream(), media_type="text/event-stream")
                else: # Non-streaming binding call
                     logger.info("Starting non-stream generation via binding...")
                     binding_output = await binding.generate( prompt=primary_text_prompt, params=merged_params, request_info=request_info, multimodal_data=multimodal_data_for_binding )
                     output = binding_output

        execution_time = time.time() - start_time
        logger.info(f"Generation request processed in {execution_time:.2f} seconds.")

        # --- Standardize Non-Streaming Response Formatting ---
        if output is not None: # Only process if it wasn't a streaming request that returned early
            final_output_list: List[OutputData] = standardize_output(output)
            if not final_output_list:
                 logger.error("Generation resulted in empty standardized output list.")
                 final_output_list.append(OutputData(type="error", data="Generation failed to produce valid output."))

            # Use the new GenerateResponse model
            response_payload = GenerateResponse(
                personality=personality.name if personality else None,
                output=final_output_list,
                execution_time=execution_time,
                request_id=request_info.get("request_id")
            )
            return JSONResponse(content=response_payload.model_dump()) # Use model_dump for Pydantic v2+
        elif not request.stream:
             logger.error("Non-streaming request finished but output is None.")
             raise HTTPException(status_code=500, detail="Generation failed to produce output.")
        # If it was a streaming request, the StreamingResponse was already returned earlier

    # --- Exception Handling (largely unchanged) ---
    except ModelLoadingError as e:
         logger.error(f"ModelLoadingError: {e}")
         detail = str(e)
         status_code = status.HTTP_408_REQUEST_TIMEOUT if "Timeout" in detail else status.HTTP_500_INTERNAL_SERVER_ERROR
         raise HTTPException(status_code=status_code, detail=detail)
    except asyncio.TimeoutError as e:
         logger.error(f"Operation timed out: {e}")
         raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Operation timed out: {e}")
    except HTTPException: raise # Re-raise specific HTTP exceptions
    except NotImplementedError as e:
         logger.error(f"Functionality not implemented: {e}")
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except ValueError as e: # Catch bad params, validation, safety blocks etc.
         logger.warning(f"ValueError during generation: {e}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad request: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

    # This path should not be reached if logic is correct
    logger.error("Reached end of process_generation_request unexpectedly.")
    raise HTTPException(status_code=500, detail="Internal server error processing generation request.")

# --- _determine_binding_and_model (unchanged for now) ---
def _determine_binding_and_model(request: GenerateRequest, personality: Optional[Personality], config: AppConfig) -> Tuple[Optional[str], Optional[str]]:
    gen_type = request.generation_type
    binding_name = request.binding_name
    model_name = request.model_name

    if not binding_name:
        binding_name = getattr(config.defaults, f"{gen_type}_binding", None)
        logger.debug(f"Using default binding '{binding_name}' for type '{gen_type}'")
    if not model_name:
        model_name = getattr(config.defaults, f"{gen_type}_model", None)
        logger.debug(f"Using default model '{model_name}' for type '{gen_type}'")

    if not binding_name:
        logger.error(f"Could not determine a binding for generation type '{gen_type}'. Please specify in request or configure defaults.")
        return None, None
    # Allow model_name to be None, binding might have a default
    if not model_name:
        logger.info(f"No specific model determined for type '{gen_type}'. Binding '{binding_name}' might use its internal default.")

    logger.info(f"Selected for generation: Type='{gen_type}', Binding='{binding_name}', Model='{model_name or 'Binding Default'}'")
    return binding_name, model_name


# --- _scan_models_folder (Added tts/stt folders) ---
def _scan_models_folder(models_base_path: Path) -> Dict[str, List[str]]:
    """ Scans the models subfolders for model files/folders. """
    # Expanded list of potential type folders
    discovered_models = {"ttt": [], "tti": [], "ttv": [], "ttm": [], "tts": [], "stt": [], "audio2audio": [], "i2i": []}
    if not models_base_path or not models_base_path.is_dir():
            logger.warning(f"Models base path '{models_base_path}' not found or not a directory. Cannot scan for models.")
            return discovered_models

    for model_type in discovered_models.keys():
        type_folder = models_base_path / model_type
        if type_folder.is_dir():
            try:
                models = [item.name for item in type_folder.iterdir() if not item.name.startswith('.')] # Ignore hidden files/folders
                discovered_models[model_type] = models
                logger.debug(f"Found models in {type_folder}: {models}")
            except Exception as e:
                logger.error(f"Error scanning models folder {type_folder}: {e}")
        else:
            # Optionally create default folders if they don't exist
            try:
                type_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured model type folder exists: {type_folder}")
            except Exception as e:
                logger.error(f"Failed to create model sub-directory {type_folder}: {e}")

    return discovered_models