# lollms_server/core/generation.py
import logging
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
from lollms_server.api.models import GenerateRequest, StreamChunk, InputData
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
    # ... other global defaults ...

    personality_conditioning = None
    # --- Updated: Extract system context from input_data ---
    # Find items designated as system context
    system_context_items = [item for item in input_data if item.type == 'text' and item.role == 'system_context']
    # Combine their data, ensuring separation
    system_context_block = "\n\n".join([item.data.strip() for item in system_context_items if item.data]).strip()
    system_context_block = system_context_block + "\n\n" if system_context_block else "" # Add trailing space if content exists
    # --- End Update ---

    if personality:
        logger.debug(f"Applying defaults and conditioning from personality '{personality.name}'...")
        p_config = personality.config
        personality_conditioning = p_config.personality_conditioning
        # ... override global defaults with personality specifics (unchanged)...
        if p_config.model_temperature is not None: merged_params['temperature'] = p_config.model_temperature
        if p_config.model_n_predicts is not None: merged_params['max_tokens'] = p_config.model_n_predicts
        if p_config.model_top_k is not None: merged_params['top_k'] = p_config.model_top_k
        if p_config.model_top_p is not None: merged_params['top_p'] = p_config.model_top_p
        if p_config.model_repeat_penalty is not None: merged_params['repeat_penalty'] = p_config.model_repeat_penalty
        if p_config.model_repeat_last_n is not None: merged_params['repeat_last_n'] = p_config.model_repeat_last_n

    request_system_message = (request.parameters or {}).get("system_message")
    final_system_message_content = None
    if request_system_message is not None:
        logger.debug("Using system message provided in request parameters.")
        final_system_message_content = request_system_message
    elif personality_conditioning is not None:
        logger.debug("Using system message from personality conditioning.")
        final_system_message_content = personality_conditioning

    # --- Combine system context from input data and determined message ---
    final_system_message = f"{system_context_block}{final_system_message_content if final_system_message_content else ''}"
    merged_params['system_message'] = final_system_message.strip() if final_system_message and final_system_message.strip() else None
    # --- End Combination ---

    if request.parameters:
        logger.debug("Applying remaining parameters from request...")
        for key, value in request.parameters.items():
             if key != 'system_message':
                  merged_params[key] = value

    loggable_params = {k:v for k,v in merged_params.items() if k != 'system_message'}
    logger.info(f"Final merged parameters for generation: {loggable_params}")
    if merged_params.get('system_message'):
         logger.debug(f"Final system message (first 200 chars): {merged_params['system_message'][:200]}...")

    request_info = {
         "personality": personality.name if personality else None,
         "functions": request.functions,
         "generation_type": request.generation_type,
         "request_id": None # Placeholder
    }
    # --- Include input_data in context for scripts ---
    generation_context = {
        "request": request, # Pass the original request object
        "personality": personality, "binding": binding,
        "function_manager": function_manager, "binding_manager": binding_manager,
        "resource_manager": resource_manager, "config": config,
        "input_data": input_data # Pass the full list for script context
    }
    # --- End Script Context Update ---


    # 5. Execute Generation
    try:
        is_scripted_request = personality and personality.is_scripted
        if is_scripted_request and not personality:
            raise HTTPException(status_code=500, detail="Scripted generation requires a valid personality.")

        output: Any = None # Holds the final result for non-streaming

        # If model name wasn't determined, the binding might use its default,
        # but we need *a* name for the loading context manager. Use a placeholder?
        # Or require model name for loadable bindings? Let's require it if loading needed.
        effective_model_name = model_name
        if not effective_model_name:
             # Check if binding needs loading - this is tricky without knowing the binding type
             # Assume for now if no model name provided, binding handles it internally (like OpenAI)
             logger.info(f"No specific model name provided for binding '{binding_name}'. Binding will use its default.")
             # We need *something* for the context manager key if loading IS needed by the binding.
             # Let the binding handle None model_name in load_model if necessary.
             effective_model_name = binding.model_name or "binding_default" # Use loaded name or placeholder

        async with manage_model_loading(binding, effective_model_name):
            if is_scripted_request:
                logger.info(f"Executing scripted workflow for personality '{personality.name}'")
                # Pass primary text prompt and full context to workflow
                script_output = await personality.run_workflow(
                    prompt=primary_text_prompt, # Main text prompt
                    params=merged_params,
                    context=generation_context # Context now includes input_data
                )

                # --- Handle Streaming Response from Script ---
                if request.stream and isinstance(script_output, AsyncGenerator):
                    logger.info("Scripted personality returned a stream.")
                    async def script_event_stream():
                        try:
                            async for chunk_data in script_output:
                                if not isinstance(chunk_data, dict):
                                    chunk_data = {"type": "chunk", "content": str(chunk_data)}
                                try:
                                    chunk = StreamChunk(**chunk_data)
                                    yield f"data: {chunk.model_dump_json()}\n\n"
                                except Exception as pydantic_error:
                                    logger.error(f"Invalid chunk format from script stream: {chunk_data}. Error: {pydantic_error}", exc_info=True)
                                    error_chunk = StreamChunk(type="error", content=f"Invalid stream chunk format from script: {pydantic_error}")
                                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                            logger.info("Scripted stream finished.")
                        except Exception as e:
                            logger.error(f"Error during scripted stream processing: {e}", exc_info=True)
                            error_chunk = StreamChunk(type="error", content=f"Script stream error: {e}")
                            try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                            except Exception: pass
                    return StreamingResponse(script_event_stream(), media_type="text/event-stream")
                # --- End Streaming Handling ---
                else:
                    # Non-streaming script result (or stream requested but not returned)
                    output = script_output # Assign to output variable
            else:
                # --- Direct Binding Path ---
                logger.info(f"Executing direct generation: Binding='{binding_name}', Type='{request.generation_type}'")

                # --- CALL BINDING WITH NEW SIGNATURE ---
                if request.stream:
                     logger.info("Starting stream generation via binding...")
                     binding_stream_generator = binding.generate_stream(
                         prompt=primary_text_prompt,
                         params=merged_params,
                         request_info=request_info,
                         multimodal_data=multimodal_data_for_binding # Pass non-text items
                     )
                     async def event_stream():
                         try:
                             async for chunk_data in binding_stream_generator:
                                 if not isinstance(chunk_data, dict):
                                     chunk_data = {"type": "chunk", "content": str(chunk_data)}
                                 try:
                                     chunk = StreamChunk(**chunk_data)
                                     yield f"data: {chunk.model_dump_json()}\n\n"
                                 except Exception as pydantic_error:
                                     logger.error(f"Invalid chunk format from binding stream: {chunk_data}. Error: {pydantic_error}", exc_info=True)
                                     error_chunk = StreamChunk(type="error", content=f"Invalid stream chunk format from binding: {pydantic_error}")
                                     yield f"data: {error_chunk.model_dump_json()}\n\n"
                             logger.info("Binding stream finished.")
                         except Exception as e:
                             logger.error(f"Error during binding stream generation: {e}", exc_info=True)
                             error_chunk = StreamChunk(type="error", content=f"Stream generation error: {e}")
                             try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                             except Exception: pass
                     return StreamingResponse(event_stream(), media_type="text/event-stream")
                else:
                     logger.info("Starting non-stream generation via binding...")
                     binding_output = await binding.generate(
                         prompt=primary_text_prompt,
                         params=merged_params,
                         request_info=request_info,
                         multimodal_data=multimodal_data_for_binding # Pass non-text items
                     )
                     output = binding_output # Assign to output variable
                # --- END BINDING CALL ---

        execution_time = time.time() - start_time
        logger.info(f"Generation request processed in {execution_time:.2f} seconds.")

        # --- Standardize Non-Streaming Response Formatting ---
        # Check if output variable was assigned (it wouldn't be for streaming requests)
        if output is not None:
            final_output_dict = {}
            if isinstance(output, str):
                # Wrap TTT string output
                final_output_dict = {"text": output}
            elif isinstance(output, dict):
                # Assume TTI/TTS/scripted dict output is already correct
                final_output_dict = output
            else:
                # Handle unexpected types from bindings/scripts
                logger.warning(f"Non-streaming request returned unexpected type {type(output)}. Wrapping.")
                try:
                    # Attempt to serialize, fallback to string representation
                    json.dumps(output) # Test serialization
                    final_output_dict = {"data": output}
                except TypeError:
                    final_output_dict = {"data_str": str(output)}

            response_payload = {
                "personality": personality.name if personality else None,
                "output": final_output_dict,
                "execution_time": execution_time,
                "request_id": request_info["request_id"] # Add request ID if generated
            }
            return JSONResponse(content=response_payload)
        elif not request.stream:
             # If it wasn't a stream and output is still None, something went wrong
             logger.error("Non-streaming request finished but output is None.")
             raise HTTPException(status_code=500, detail="Generation failed to produce output.")
        else:
             # This path should not be reached if streaming returns StreamingResponse correctly
             logger.error("Reached unexpected state at end of process_generation_request after stream handling.")
             raise HTTPException(status_code=500, detail="Internal server error processing generation response.")


    # --- Exception Handling (largely unchanged) ---
    except ModelLoadingError as e:
         logger.error(f"ModelLoadingError: {e}")
         if "Timeout waiting for GPU resource" in str(e):
             raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=str(e))
         else:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except asyncio.TimeoutError as e:
         logger.error(f"Operation timed out: {e}")
         raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Operation timed out: {e}")
    except HTTPException:
         raise
    except NotImplementedError as e:
         logger.error(f"Functionality not implemented: {e}")
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except ValueError as e: # Catch potential value errors (bad params, validation, safety)
         logger.warning(f"ValueError during generation: {e}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad request: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal server error occurred: {e}")


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