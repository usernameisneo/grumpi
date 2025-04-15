# lollms_server/core/generation.py
import logging
import time
import json
import asyncio # Ensure asyncio is imported
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator, Union, Tuple, List # Add Tuple
from contextlib import asynccontextmanager, nullcontext # Add nullcontext
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from .config import AppConfig
from .bindings import BindingManager, Binding
from .personalities import PersonalityManager, Personality
from .functions import FunctionManager
from .resource_manager import ResourceManager # Ensure ResourceManager is imported
from lollms_server.api.models import GenerateRequest, StreamChunk
from lollms_server.utils.helpers import encode_base64

logger = logging.getLogger(__name__)


@asynccontextmanager
async def manage_model_loading(binding: Binding, model_name: str):
    """Context manager to handle model loading with resource management."""
    load_success = False
    try:
        # Check if model already loaded (inside binding's load_model)
        logger.info(f"Requesting model load: Binding='{binding.binding_name}', Model='{model_name}'")

        # --- Let Binding handle resource acquisition within its load_model ---
        # The binding's load_model implementation is now responsible for
        # calling resource_manager.acquire_gpu_resource if needed.
        # We just await the load_model call here.
        load_success = await binding.load_model(model_name)

        if not load_success:
            # The binding's load_model should have logged the specific error (timeout, load fail)
                logger.error(f"Binding '{binding.binding_name}' failed to load model '{model_name}'.")
                # Raise a specific exception to be caught by the caller
                raise ModelLoadingError(f"Binding '{binding.binding_name}' failed to load model '{model_name}'.")

        logger.info(f"Model '{model_name}' confirmed loaded by binding '{binding.binding_name}'.")
        yield # Model is loaded, proceed with generation
    finally:
        # --- Unloading Strategy ---
        # Decide when to unload. For now, we DON'T unload automatically after each request.
        # Models remain loaded until the binding is explicitly unloaded or replaced.
        # logger.info(f"Request finished. Keeping model '{model_name}' loaded in binding '{binding.binding_name}'.")
        # If unloading was desired:
        # if load_success: # Only unload if we successfully loaded it?
        #     logger.info(f"Request finished. Unloading model '{model_name}' from binding '{binding.binding_name}'.")
        #     await binding.unload_model()
        pass


class ModelLoadingError(Exception):
    """Custom exception for model loading failures."""
    pass

# --- Main process_generation_request function ---
async def process_generation_request(
    request: GenerateRequest,
    personality_manager: PersonalityManager,
    binding_manager: BindingManager,
    function_manager: FunctionManager,
    resource_manager: ResourceManager,
    config: AppConfig
) -> Union[str, Dict[str, Any], StreamingResponse]:
    """
    Processes a generation request, handling personality logic, binding selection,
    parameter merging, resource management, and generation execution.
    """
    start_time = time.time()
    personality: Optional[Personality] = None

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
    if not binding_name or not model_name:
         # Error logged in helper function
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not determine binding or model for the request.")

    # 3. Get Binding Instance
    binding = binding_manager.get_binding(binding_name)
    if not binding:
        logger.error(f"Binding instance '{binding_name}' not found or failed to instantiate.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Binding '{binding_name}' not available.")

    # 4. Prepare Generation Parameters (Merging: Global < Personality < Request)
    merged_params = {}

    # Apply Global Defaults
    logger.debug("Applying global default parameters...")
    merged_params['max_tokens'] = config.defaults.default_max_output_tokens
    merged_params['context_size'] = config.defaults.default_context_size # Informational
    # Add other global defaults here (e.g., temperature?)
    # merged_params['temperature'] = config.defaults.default_temperature # Example

    # Build System Message pieces
    personality_conditioning = None
    extra_data_block = ""

    # Include Extra Data from request (prepended)
    if request.extra_data:
        logger.debug("Including extra_data in system context.")
        try:
            extra_data_str = json.dumps(request.extra_data, indent=2)
            extra_data_block = (
                f"--- Begin Provided Data ---\n"
                f"{extra_data_str}\n"
                f"--- End Provided Data ---\n\n"
            )
        except Exception as e:
            logger.warning(f"Could not format extra_data for system prompt: {e}")
            extra_data_block = f"--- Provided Data (Error Formatting) ---\n{request.extra_data}\n--- End Provided Data ---\n\n"

    # Apply Personality Defaults & Conditioning (if personality exists)
    if personality:
        logger.debug(f"Applying defaults and conditioning from personality '{personality.name}'...")
        p_config = personality.config
        personality_conditioning = p_config.personality_conditioning # Get conditioning text

        # Override global defaults with personality specifics
        if p_config.model_temperature is not None: merged_params['temperature'] = p_config.model_temperature
        if p_config.model_n_predicts is not None: merged_params['max_tokens'] = p_config.model_n_predicts
        if p_config.model_top_k is not None: merged_params['top_k'] = p_config.model_top_k
        if p_config.model_top_p is not None: merged_params['top_p'] = p_config.model_top_p
        if p_config.model_repeat_penalty is not None: merged_params['repeat_penalty'] = p_config.model_repeat_penalty
        if p_config.model_repeat_last_n is not None: merged_params['repeat_last_n'] = p_config.model_repeat_last_n
        # Add context_size/max_output from personality if defined later

    # Determine Final System Message (Priority: Request > Personality > None)
    request_system_message = (request.parameters or {}).get("system_message")
    final_system_message_content = None
    if request_system_message is not None:
        logger.debug("Using system message provided in request parameters.")
        final_system_message_content = request_system_message
    elif personality_conditioning is not None:
        logger.debug("Using system message from personality conditioning.")
        final_system_message_content = personality_conditioning

    # Prepend extra data block if it exists
    final_system_message = f"{extra_data_block}{final_system_message_content}" if final_system_message_content else extra_data_block
    # Assign to merged_params, ensuring None if empty string
    merged_params['system_message'] = final_system_message.strip() if final_system_message and final_system_message.strip() else None

    # Apply remaining Request Parameters (overriding previous levels)
    if request.parameters:
        logger.debug("Applying remaining parameters from request...")
        for key, value in request.parameters.items():
             if key != 'system_message': # system_message already handled by priority
                  merged_params[key] = value

    # Log final parameters (excluding potentially long system message)
    loggable_params = {k:v for k,v in merged_params.items() if k != 'system_message'}
    logger.info(f"Final merged parameters for generation: {loggable_params}")
    if merged_params.get('system_message'):
         logger.debug(f"Final system message (first 200 chars): {merged_params['system_message'][:200]}...")


    # Prepare context dictionary for scripts/bindings
    generation_context = {
        "request": request, "personality": personality, "binding": binding,
        "function_manager": function_manager, "binding_manager": binding_manager,
        "resource_manager": resource_manager, "config": config,
        "extra_data": request.extra_data
    }
    request_info = { # Minimal info often needed by binding generate calls
         "personality": personality.name if personality else None,
         "functions": request.functions,
         "generation_type": request.generation_type,
         "request_id": None # Placeholder for future request tracking ID
    }


    # 5. Execute Generation
    try:
        is_scripted_request = personality and personality.is_scripted
        if is_scripted_request and not personality: # Should not happen
            raise HTTPException(status_code=500, detail="Scripted generation requires a valid personality.")

        output: Any = None # Initialize variable to hold result

        # Acquire model resources (context manager handles loading/queuing/timeouts)
        async with manage_model_loading(binding, model_name):
            # --- Model is loaded ---

            if is_scripted_request:
                # --- Scripted Personality Path ---
                logger.info(f"Executing scripted workflow for personality '{personality.name}'")
                script_output = await personality.run_workflow(
                    prompt=request.prompt,
                    params=merged_params,
                    context=generation_context
                )

                # Handle streaming if script returns an async generator
                if request.stream and isinstance(script_output, AsyncGenerator):
                    logger.info("Scripted personality returned a stream.")
                    async def script_event_stream(): # Define wrapper locally
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

                # Handle non-streaming script result (even if stream was requested)
                elif request.stream:
                    logger.warning("Streaming requested, but scripted personality returned non-streamable result. Sending as single chunk.")
                    async def non_stream_wrapper(): # Define wrapper locally
                         try:
                             yield f"data: {StreamChunk(type='chunk', content=script_output).model_dump_json()}\n\n"
                             yield f"data: {StreamChunk(type='final', content=script_output, metadata={'message':'Script finished'}).model_dump_json()}\n\n"
                         except Exception as json_err:
                              logger.error(f"Could not serialize scripted output for streaming: {json_err}")
                              yield f"data: {StreamChunk(type='error', content='Could not serialize scripted output').model_dump_json()}\n\n"
                    return StreamingResponse(non_stream_wrapper(), media_type="text/event-stream")
                else:
                    # Non-streaming request, script returned final result
                    output = script_output # Assign to output variable

            else:
                # --- Direct Binding Path (Non-Scripted or No Personality) ---
                logger.info(f"Executing direct generation: Binding='{binding_name}', Type='{request.generation_type}', Personality='{request_info['personality']}'")
                current_prompt = request.prompt

                if request.generation_type == 'ttt':
                    if request.stream:
                        # --- TTT Streaming via Binding ---
                        logger.info("Starting TTT stream generation...")
                        binding_stream_generator = binding.generate_stream(current_prompt, merged_params, request_info)

                        async def event_stream(): # Define wrapper locally
                            try:
                                async for chunk_data in binding_stream_generator:
                                    if not isinstance(chunk_data, dict): # Basic validation
                                        chunk_data = {"type": "chunk", "content": str(chunk_data)}
                                    try:
                                        chunk = StreamChunk(**chunk_data)
                                        yield f"data: {chunk.model_dump_json()}\n\n"
                                    except Exception as pydantic_error:
                                        logger.error(f"Invalid chunk format from binding stream: {chunk_data}. Error: {pydantic_error}", exc_info=True)
                                        error_chunk = StreamChunk(type="error", content=f"Invalid stream chunk format from binding: {pydantic_error}")
                                        yield f"data: {error_chunk.model_dump_json()}\n\n"
                                logger.info("TTT stream finished.")
                            except Exception as e:
                                logger.error(f"Error during TTT stream generation: {e}", exc_info=True)
                                error_chunk = StreamChunk(type="error", content=f"Stream generation error: {e}")
                                try: yield f"data: {error_chunk.model_dump_json()}\n\n"
                                except Exception: pass
                        return StreamingResponse(event_stream(), media_type="text/event-stream")

                    else:
                        # --- TTT Non-Streaming via Binding ---
                        logger.info("Starting TTT non-stream generation...")
                        binding_output = await binding.generate(current_prompt, merged_params, request_info)
                        if not isinstance(binding_output, str):
                             logger.error(f"TTT binding returned non-string output: {type(binding_output)}")
                             raise HTTPException(status_code=500, detail="Generation failed: Expected text output from binding.")
                        output = binding_output # Assign to output variable

                else:
                    # --- Non-TTT Generation via Binding ---
                    logger.info(f"Starting non-TTT ({request.generation_type}) generation...")
                    binding_output = await binding.generate(current_prompt, merged_params, request_info)
                    if not isinstance(binding_output, dict):
                         logger.error(f"Non-TTT binding returned non-dict output: {type(binding_output)}")
                         raise HTTPException(status_code=500, detail=f"Generation failed: Expected dictionary output for {request.generation_type} from binding.")
                    output = binding_output # Assign to output variable

        # --- End of manage_model_loading context ---

        # --- Return Formatting (for non-streaming requests) ---
        # At this point, if it wasn't a streaming request that already returned a StreamingResponse,
        # the final result (string or dict) should be in the 'output' variable.
        execution_time = time.time() - start_time
        logger.info(f"Generation request processed in {execution_time:.2f} seconds.")

        if isinstance(output, str):
            # Return raw text for TTT string output
            return output
        elif isinstance(output, dict):
            # Return dictionary for TTI/V/M or complex/scripted dict output
            return output
        elif output is None and not request.stream:
             # Should only happen if non-streaming request resulted in no output
             logger.warning("Non-streaming request resulted in None output.")
             raise HTTPException(status_code=500, detail="Generation failed to produce output.")
        elif not request.stream:
             # Non-streaming request but output wasn't string or dict (unexpected)
             logger.warning(f"Non-streaming request returned unexpected type {type(output)}. Wrapping.")
             return {"output": output} # Wrap for safety
        else:
             # This case should not be reached if streaming responses were handled correctly above
             logger.error("Reached unexpected state at end of process_generation_request.")
             raise HTTPException(status_code=500, detail="Internal server error processing generation response.")


    # --- Exception Handling ---
    except ModelLoadingError as e:
         logger.error(f"ModelLoadingError: {e}")
         # Check if it was a timeout during resource acquisition (from ResourceManager maybe?)
         if "Timeout waiting for GPU resource" in str(e): # Check specific error message if applicable
             raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=str(e))
         else:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except asyncio.TimeoutError as e: # Catch timeouts during generate/workflow awaits
         logger.error(f"Operation timed out: {e}")
         raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Operation timed out: {e}")
    except HTTPException:
         raise # Re-raise FastAPI HTTP exceptions directly
    except NotImplementedError as e:
         logger.error(f"Functionality not implemented: {e}")
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except ValueError as e: # Catch potential value errors (e.g., bad params, DALL-E content policy)
         logger.warning(f"ValueError during generation: {e}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad request: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal server error occurred: {e}")

def _determine_binding_and_model(request: GenerateRequest, personality: Personality, config: AppConfig) -> Tuple[Optional[str], Optional[str]]:
    """Determines the binding instance name and model name to use based on request, personality, and defaults."""

    gen_type = request.generation_type
    binding_name = request.binding_name
    model_name = request.model_name

    # Priority: Request -> Personality Config (TODO) -> Global Defaults

    # If not specified in request, check personality config (Placeholder for now)
    # TODO: Add fields like `default_ttt_binding`, `default_ttt_model` etc. to PersonalityConfig
    # if not binding_name and hasattr(personality.config, f"default_{gen_type}_binding"):
    #     binding_name = getattr(personality.config, f"default_{gen_type}_binding", None)
    # if not model_name and hasattr(personality.config, f"default_{gen_type}_model"):
    #     model_name = getattr(personality.config, f"default_{gen_type}_model", None)

    # If still not determined, use global defaults from config.toml
    if not binding_name:
        binding_name = getattr(config.defaults, f"{gen_type}_binding", None)
        logger.debug(f"Using default binding '{binding_name}' for type '{gen_type}'")
    if not model_name:
        model_name = getattr(config.defaults, f"{gen_type}_model", None)
        logger.debug(f"Using default model '{model_name}' for type '{gen_type}'")


    # Final checks
    if not binding_name:
        logger.error(f"Could not determine a binding for generation type '{gen_type}'. Please specify in request or configure defaults.")
        return None, None
    if not model_name:
        logger.error(f"Could not determine a model for generation type '{gen_type}'. Please specify in request or configure defaults.")
        return None, None

    logger.info(f"Selected for generation: Type='{gen_type}', Binding='{binding_name}', Model='{model_name}'")
    return binding_name, model_name


def _scan_models_folder(models_base_path: Path) -> Dict[str, List[str]]:
    """ Scans the models subfolders (ttt, tti, ttv, ttm) for model files/folders. """
    discovered_models = {"ttt": [], "tti": [], "ttv": [], "ttm": []}
    if not models_base_path or not models_base_path.is_dir():
            logger.warning(f"Models base path '{models_base_path}' not found or not a directory. Cannot scan for models.")
            return discovered_models

    for model_type in discovered_models.keys():
        type_folder = models_base_path / model_type
        if type_folder.is_dir():
            try:
                # List files and directories - assuming model names can be either
                models = [item.name for item in type_folder.iterdir()]
                discovered_models[model_type] = models
                logger.debug(f"Found models in {type_folder}: {models}")
            except Exception as e:
                logger.error(f"Error scanning models folder {type_folder}: {e}")
        else:
            logger.debug(f"Model type folder not found: {type_folder}")

    return discovered_models
