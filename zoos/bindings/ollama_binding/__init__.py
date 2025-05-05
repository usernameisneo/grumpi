# lollms_server/core/bindings.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: Binding implementation for the Ollama inference server.
# Modification Date: 2025-05-04
# Refactored model info endpoint, default model handling, tokenizers.

import asyncio
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List

# Check/Install dependencies using pipmaster if needed (optional but good practice)
try:
    import pipmaster as pm
    pm.ensure_packages(["ollama", "pillow", "tiktoken"])
except ImportError:
    pass # Assume installed or handle import error below

# Import core libraries
try:
    import ollama
    from PIL import Image
    import tiktoken # For token approximation
    ollama_installed = True
    pillow_installed = True
    tiktoken_installed = True
except ImportError as e:
    # Mock classes for environments where imports failed but allow basic functionality
    if 'ollama' in str(e): ollama = None; ollama_installed = False
    if 'PIL' in str(e): Image = None; BytesIO = None; pillow_installed = False # type: ignore
    if 'tiktoken' in str(e): tiktoken = None; tiktoken_installed = False

# Import lollms_server components
try:
    import ascii_colors as logging # Use logging alias
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.utils.helpers import parse_thought_tags

# Use TYPE_CHECKING for API model imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        class StreamChunk: pass # type: ignore
        class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

class OllamaBinding(Binding):
    """Binding for Ollama inference server."""
    binding_type_name = "ollama_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the OllamaBinding using the instance configuration.
        """
        super().__init__(config, resource_manager) # Sets self.config, self.resource_manager, self.binding_instance_name, self.default_model_name

        if not ollama_installed or not ollama:
             raise ImportError("Ollama binding requires the 'ollama' library. Please install it (`pip install ollama`).")

        self.host = self.config.get("host", "http://localhost:11434")
        logger.info(f"Initializing Ollama Binding instance '{self.binding_instance_name}' -> Host: '{self.host}'.")

        try:
            self.client = ollama.AsyncClient(host=self.host)
        except Exception as e:
             logger.error(f"Failed to initialize Ollama client for instance '{self.binding_instance_name}' with host '{self.host}': {e}")
             trace_exception(e)
             self.client = None

        # self.model_name is set by load_model (stores currently intended/active model)
        self.model_supports_vision: bool = False
        self.current_model_details: Dict[str, Any] = {} # Store details of the *currently active* model


    # --- Helper to parse Ollama model details ---
    def _parse_ollama_details(self, model_obj_or_dict: Union[Dict, Any]) -> Dict[str, Any]:
        """
        Parses the raw Ollama model object or dictionary from show/list API.
        Returns a dictionary formatted for GetModelInfoResponse or list_available_models.
        """
        parsed = {}
        model_dict = {}

        # Handle both dict (from list) and object (from show) response formats
        if isinstance(model_obj_or_dict, dict):
             model_dict = model_obj_or_dict
        elif hasattr(model_obj_or_dict, '__dict__'):
             model_dict = model_obj_or_dict.__dict__
        elif model_obj_or_dict is None: # Handle None case gracefully
             return {}
        else:
             logger.warning(f"Unexpected type for Ollama model details: {type(model_obj_or_dict)}")
             return {}

        # --- Extract core fields ---
        name = model_dict.get('name') or model_dict.get('model')
        if not name: return {}
        parsed['name'] = name
        parsed['model_name'] = name # Explicitly add model_name for GetModelInfoResponse

        parsed['size'] = model_dict.get('size')
        try:
            mod_time_str = model_dict.get('modified_at')
            if mod_time_str:
                if '.' in mod_time_str: # Handle varying levels of precision and timezone info
                    parts = mod_time_str.split('.')
                    ts_str = parts[0]
                    tz_part = ""
                    if '+' in parts[1]: ms_part, tz_part = parts[1].split('+'); tz_part = '+' + tz_part
                    elif '-' in parts[1]: ms_part, tz_part = parts[1].split('-'); tz_part = '-' + tz_part
                    else: ms_part = parts[1].replace('Z','')
                    final_ts_str = f"{ts_str}.{ms_part[:6]}{tz_part}" # Truncate ms, keep tz
                    try: parsed['modified_at'] = datetime.fromisoformat(final_ts_str)
                    except ValueError: # Fallback if timezone parsing fails
                         try: parsed['modified_at'] = datetime.fromisoformat(mod_time_str.replace('Z','+00:00').split('.')[0])
                         except ValueError: parsed['modified_at'] = None
                else: parsed['modified_at'] = datetime.fromisoformat(mod_time_str.replace("Z", "+00:00"))
            else: parsed['modified_at'] = None
        except Exception as time_e: logger.warning(f"Could not parse Ollama modified_at time '{model_dict.get('modified_at', '')}': {time_e}"); parsed['modified_at'] = None


        # --- Extract details sub-object ---
        details_obj = model_dict.get('details')
        generic_details = {}
        if isinstance(details_obj, dict):
            parsed['format'] = details_obj.get('format')
            parsed['families'] = details_obj.get('families')
            parsed['family'] = details_obj.get('family') or (parsed['families'][0] if parsed.get('families') else None)
            parsed['parameter_size'] = details_obj.get('parameter_size')
            parsed['quantization_level'] = details_obj.get('quantization_level')
            parsed['context_size'] = None # Try to extract from parameters below

            # Extract context size from parameters string
            params_str = details_obj.get('parameters', '')
            if params_str and isinstance(params_str, str):
                lines = params_str.split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == 'num_ctx':
                         try: parsed['context_size'] = int(parts[1]); break
                         except ValueError: pass

            # Store remaining details keys
            detail_keys_parsed = {'format', 'family', 'families', 'parameter_size', 'quantization_level', 'parameters'}
            generic_details = {k: v for k, v in details_obj.items() if k not in detail_keys_parsed}

        elif hasattr(details_obj, '__dict__'): # Handle potential object type from 'show'
            # Repeat extraction logic using getattr
            parsed['format'] = getattr(details_obj, 'format', None)
            parsed['families'] = getattr(details_obj, 'families', None)
            parsed['family'] = getattr(details_obj, 'family', None) or (parsed['families'][0] if parsed.get('families') else None)
            parsed['parameter_size'] = getattr(details_obj, 'parameter_size', None)
            parsed['quantization_level'] = getattr(details_obj, 'quantization_level', None)
            parsed['context_size'] = None
            params_str = getattr(details_obj, 'parameters', '')
            # ... (repeat params string parsing logic) ...
            if params_str and isinstance(params_str, str):
                # ... (context size extraction logic) ...
                lines = params_str.split('\n');
                for line in lines:
                    parts = line.split();
                    if len(parts) >= 2 and parts[0] == 'num_ctx':
                         try: parsed['context_size'] = int(parts[1]); break
                         except ValueError: pass
            detail_keys_parsed = {'format', 'family', 'families', 'parameter_size', 'quantization_level', 'parameters'}
            for key, value in details_obj.__dict__.items():
                 if key not in detail_keys_parsed: generic_details[key] = value

        # --- Determine Capabilities ---
        supports_vision = any(tag in name.lower() for tag in ['llava', 'vision', 'bakllava'])
        if parsed.get('family'): supports_vision = supports_vision or any(f in parsed['family'].lower() for f in ['llava', 'vision'])
        elif parsed.get('families'): supports_vision = supports_vision or any(f in fam.lower() for fam in parsed['families'] for f in ['llava', 'vision'])
        parsed['supports_vision'] = supports_vision
        parsed['supports_audio'] = False # Ollama doesn't handle audio generation/input via generate
        parsed['supports_streaming'] = True # Ollama generate supports streaming

        # Determine model_type based on vision support
        if supports_vision:
            parsed['model_type'] = 'vlm' # Vision Language Model
        else:
             parsed['model_type'] = 'ttt' # Text-to-Text

        # Add digest if present
        digest_val = model_dict.get('digest')
        if digest_val: generic_details['digest'] = digest_val

        parsed['details'] = generic_details
        # Max output tokens not directly available from Ollama info
        parsed.setdefault('max_output_tokens', None)

        return parsed

    # --- Required Binding Methods ---

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available locally via the configured Ollama instance."""
        if not self.client: logger.error(f"Ollama client not initialized for instance '{self.binding_instance_name}'. Cannot list models."); return []
        logger.info(f"Ollama '{self.binding_instance_name}': Listing local models from {self.host}...")
        formatted_models = []
        try:
            response = await self.client.list()
            ollama_models = response.get('models', [])
            for model_dict in ollama_models:
                 # Use the parser helper which handles dict format from list()
                 parsed_data = self._parse_ollama_details(model_dict)
                 if parsed_data.get('name'): formatted_models.append(parsed_data)
                 else: logger.warning(f"Skipping unparsable Ollama model data: {model_dict}")
            return formatted_models
        except ollama.ResponseError as e: logger.error(f"Ollama API Error listing models from {self.host} (Status {e.status_code}): {e}"); raise RuntimeError(f"Ollama API Error: {e}") from e
        except Exception as e: logger.error(f"Unexpected error listing Ollama models from {self.host}: {e}", exc_info=True); raise RuntimeError(f"Unexpected error contacting Ollama: {e}") from e

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on the loaded model."""
        modalities = ['text']
        # Use the state variable populated by load_model or get_model_info
        if self.model_supports_vision:
            modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text']

    async def health_check(self) -> Tuple[bool, str]:
        """Checks connection to the configured Ollama server."""
        if not self.client: return False, f"Ollama client initialization failed for host '{self.host}' on instance '{self.binding_instance_name}'."
        try:
             response = await self.client.list(); models_count = len(response.get('models', []))
             return True, f"Connection to {self.host} OK ({models_count} models found)."
        except ollama.ResponseError as e: logger.error(f"Ollama Health check failed for instance '{self.binding_instance_name}' at {self.host} ({e.status_code}): {e}"); return False, f"Ollama Response Error {e.status_code}: {e}"
        except Exception as e: logger.error(f"Ollama Health check failed for instance '{self.binding_instance_name}' at {self.host}: {e}", exc_info=True); return False, f"Connection/Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Ollama manages its own resources, but indicates GPU is usually used."""
        # Returning True for GPU helps resource manager allocate if needed, even if Ollama manages it.
        return {"gpu_required": True, "estimated_vram_mb": 0} # VRAM estimation not feasible

    async def load_model(self, model_name: str) -> bool:
        """
        Sets the target model for the binding. For Ollama, this checks if the model
        is available locally via the API and stores its details. Pulling happens during generation if needed.
        """
        if not self.client: logger.error(f"Ollama client not initialized for instance '{self.binding_instance_name}'. Cannot load model."); return False

        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                logger.info(f"Ollama '{self.binding_instance_name}': Model '{model_name}' already active.")
                return True

            # Unset previous state if switching models
            if self._model_loaded:
                logger.debug(f"Unsetting previous model '{self.model_name}' state.")
                self.model_name = None; self._model_loaded = False; self.model_supports_vision = False; self.current_model_details = {}

            logger.info(f"Ollama '{self.binding_instance_name}': Setting active model to '{model_name}' on host {self.host}.")
            try:
                # Use client.show to get details and verify local existence
                model_info_resp = await self.client.show(model=model_name)
                # Parse and store details
                parsed_details = self._parse_ollama_details(model_info_resp) # Use helper
                self.current_model_details = parsed_details
                self.model_supports_vision = parsed_details.get('supports_vision', False)
                self.model_name = model_name # Set intended model name
                self._model_loaded = True # Mark as ready to use
                logger.info(f"Ollama model '{model_name}' confirmed locally. Vision: {self.model_supports_vision}, Context: {parsed_details.get('context_size')}")
                return True
            except ollama.ResponseError as e:
                if e.status_code == 404:
                    # Model not found locally, but that's okay. Ollama will pull on first use.
                    # Set the intended model name, assume capabilities based on name, reset details.
                    logger.warning(f"Ollama model '{model_name}' not found locally on {self.host}. Will attempt pull on first generation.")
                    self.model_name = model_name
                    self.model_supports_vision = any(tag in model_name.lower() for tag in ['llava', 'vision', 'bakllava'])
                    self.current_model_details = {"name": model_name, "status": "not_local"} # Store minimal info
                    self._model_loaded = True # Mark as "ready" in the sense that the binding knows which model to use
                    return True
                else:
                    # Other API error during 'show'
                    logger.error(f"Ollama API error checking model '{model_name}' on {self.host} for instance '{self.binding_instance_name}': {e}")
                    self._reset_model_state()
                    return False
            except Exception as e:
                logger.error(f"Unexpected error setting Ollama model '{model_name}' for instance '{self.binding_instance_name}': {e}", exc_info=True)
                self._reset_model_state()
                return False

    def _reset_model_state(self):
        """Helper to reset internal model state variables."""
        self.model_name = None
        self._model_loaded = False
        self.model_supports_vision = False
        self.current_model_details = {}

    async def unload_model(self) -> bool:
        """Unsets the active model. Ollama server manages actual unloading."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Ollama '{self.binding_instance_name}': Unsetting active model '{self.model_name}'. Ollama server manages unloading.")
            self._reset_model_state()
            return True

    # --- Updated Generation Methods ---
    async def _get_effective_model_name(self) -> Optional[str]:
        """Gets the model name to use, prioritizing loaded, then instance default."""
        if self._model_loaded and self.model_name:
             return self.model_name
        elif self.default_model_name:
             logger.warning(f"No model explicitly loaded for instance '{self.binding_instance_name}'. Using instance default '{self.default_model_name}'.")
             # Attempt to "load" the default model to set state correctly
             if await self.load_model(self.default_model_name):
                 return self.default_model_name
             else:
                 logger.error(f"Failed to set instance default model '{self.default_model_name}'. Cannot proceed.")
                 return None
        else:
             logger.error(f"No model loaded or configured as default for Ollama instance '{self.binding_instance_name}'.")
             return None

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Return List[OutputData]-like
        """Generates text using the Ollama API (non-streaming)."""
        if not self.client: raise RuntimeError(f"Ollama client not initialized for instance '{self.binding_instance_name}'.")

        effective_model_name = await self._get_effective_model_name()
        if not effective_model_name:
             raise RuntimeError(f"No model available or configured for Ollama instance '{self.binding_instance_name}'.")
        # Use self.model_supports_vision which should be set by _get_effective_model_name -> load_model
        current_vision_support = self.model_supports_vision

        logger.info(f"Ollama '{self.binding_instance_name}': Generating non-stream with '{effective_model_name}'...")

        options = params.get("options", {}).copy() if isinstance(params.get("options"), dict) else {}
        # Map parameters (same as before)
        if "temperature" in params: options["temperature"] = float(params["temperature"])
        if "max_tokens" in params: options["num_predict"] = int(params["max_tokens"])
        if "top_p" in params: options["top_p"] = float(params["top_p"])
        if "top_k" in params: options["top_k"] = int(params["top_k"])
        if "repeat_penalty" in params: options["repeat_penalty"] = float(params["repeat_penalty"])
        if "seed" in params: options["seed"] = int(params["seed"])
        stop = params.get("stop_sequences") or params.get("stop")
        if stop: options["stop"] = stop if isinstance(stop, list) else [stop]
        system_message = params.get("system_message", None)
        images_b64 = []

        # Process images if model supports vision
        if current_vision_support and multimodal_data:
             if not pillow_installed: logger.warning("Cannot process images: Pillow library not found.")
             else:
                 for item in multimodal_data:
                     if item.type == 'image' and item.data and isinstance(item.data, str):
                         try:
                             # Basic check + append base64 data
                             if len(item.data) > 10: images_b64.append(item.data); logger.info(f"Included image (role: {item.role}) for Ollama non-stream request.")
                             else: logger.warning(f"Skipping image data that doesn't look like base64: role={item.role}")
                         except Exception as e: logger.error(f"Failed to process image data for Ollama non-stream: {e}")
                     elif item.type == 'image': logger.warning(f"Skipping image item with missing/invalid data: role={item.role}")
        elif multimodal_data and not current_vision_support:
            logger.warning(f"Ollama instance '{self.binding_instance_name}' received image data, but model '{effective_model_name}' does not support vision. Ignoring images.")

        try:
            logger.debug(f"Ollama non-stream generate call: model={effective_model_name}, prompt='{prompt[:50]}...', system='{str(system_message)[:50] if system_message else None}...', options={options}, images_count={len(images_b64)}")
            response = await self.client.generate(model=effective_model_name, prompt=prompt, system=system_message, options=options, images=images_b64 if images_b64 else None, stream=False)

            raw_completion = response.get('response', '')
            cleaned_completion, thoughts = parse_thought_tags(raw_completion)

            logger.info(f"Ollama '{self.binding_instance_name}': Generation successful.")
            output_metadata = {"model_used": effective_model_name, "binding_instance": self.binding_instance_name}
            # Extract usage/timing stats
            for key in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']:
                 if key in response: output_metadata[key] = response[key]
            if 'prompt_eval_count' in output_metadata and 'eval_count' in output_metadata:
                 output_metadata["usage"] = {
                     "prompt_tokens": output_metadata['prompt_eval_count'],
                     "completion_tokens": output_metadata['eval_count'],
                     "total_tokens": output_metadata['prompt_eval_count'] + output_metadata['eval_count']
                 }

            return [{"type": "text", "data": cleaned_completion.strip(), "thoughts": thoughts, "metadata": output_metadata}]

        except ollama.ResponseError as e:
            logger.error(f"Ollama API Error generating from {self.host} for instance '{self.binding_instance_name}' (Status {e.status_code}): {e}")
            if e.status_code == 404 and "model" in str(e) and "not found" in str(e): raise RuntimeError(f"Ollama model '{effective_model_name}' not found locally. Pull it first (e.g., `ollama pull {effective_model_name}`).") from e
            raise RuntimeError(f"Ollama API Error ({e.status_code}): {e}") from e
        except Exception as e:
            logger.error(f"Ollama unexpected error generating from {self.host} for instance '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Ollama error: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates text using the Ollama API (streaming)."""
        if not self.client: yield {"type": "error", "content": f"Ollama client not initialized for instance '{self.binding_instance_name}'."}; return

        effective_model_name = await self._get_effective_model_name()
        if not effective_model_name:
             yield {"type": "error", "content": f"No model available or configured for Ollama instance '{self.binding_instance_name}'."}; return
        # Use self.model_supports_vision which should be set by _get_effective_model_name -> load_model
        current_vision_support = self.model_supports_vision

        logger.info(f"Ollama '{self.binding_instance_name}': Generating stream with '{effective_model_name}' from {self.host}...")

        options = params.get("options", {}).copy() if isinstance(params.get("options"), dict) else {}
        # Map parameters (same as before)
        if "temperature" in params: options["temperature"] = float(params["temperature"])
        if "max_tokens" in params: options["num_predict"] = int(params["max_tokens"])
        if "top_p" in params: options["top_p"] = float(params["top_p"])
        if "top_k" in params: options["top_k"] = int(params["top_k"])
        if "repeat_penalty" in params: options["repeat_penalty"] = float(params["repeat_penalty"])
        if "seed" in params: options["seed"] = int(params["seed"])
        stop = params.get("stop_sequences") or params.get("stop");
        if stop: options["stop"] = stop if isinstance(stop, list) else [stop]
        system_message = params.get("system_message", None)
        images_b64 = []

        # Process images if model supports vision
        if current_vision_support and multimodal_data:
             if not pillow_installed: logger.warning("Cannot process images: Pillow library not found.")
             else:
                 for item in multimodal_data:
                     if item.type == 'image' and item.data and isinstance(item.data, str):
                         try:
                              if len(item.data) > 10: images_b64.append(item.data); logger.info(f"Included image (role: {item.role}) for Ollama stream request.")
                              else: logger.warning(f"Skipping image data that doesn't look like base64: role={item.role}")
                         except Exception as e: logger.error(f"Failed to process image data for Ollama stream: {e}")
                     elif item.type == 'image': logger.warning(f"Skipping image item with missing/invalid data: role={item.role}")
        elif multimodal_data and not current_vision_support:
             logger.warning(f"Ollama instance '{self.binding_instance_name}' received image data, but model '{effective_model_name}' does not support vision. Ignoring images.")

        full_raw_response_text = ""; accumulated_thoughts = ""; is_thinking = False # State for thought parsing
        final_metadata = {"model_used": effective_model_name, "binding_instance": self.binding_instance_name}
        last_chunk_stats = {}

        try:
            logger.debug(f"Ollama stream call: model={effective_model_name}, prompt='{prompt[:50]}...', system='{str(system_message)[:50] if system_message else None}...', options={options}, images_count={len(images_b64)}")
            stream = await self.client.generate( model=effective_model_name, prompt=prompt, system=system_message, options=options, images=images_b64 if images_b64 else None, stream=True )

            async for chunk in stream:
                chunk = chunk.__dict__
                is_done = chunk.get('done', False)
                if is_done:
                    last_chunk_stats = {k: v for k, v in chunk.items() if k in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']}
                    break # Process final chunk outside the loop

                chunk_raw_content = chunk.get('response', '')
                if chunk_raw_content:
                    full_raw_response_text += chunk_raw_content # Accumulate raw text
                    # --- Stream parsing logic for thoughts ---
                    current_text_to_process = chunk_raw_content
                    processed_text_chunk = ""; processed_thoughts_chunk = None
                    while current_text_to_process:
                        if is_thinking:
                            end_tag_pos = current_text_to_process.find("</think>")
                            if end_tag_pos != -1:
                                thought_part = current_text_to_process[:end_tag_pos]; accumulated_thoughts += thought_part
                                processed_thoughts_chunk = accumulated_thoughts; accumulated_thoughts = ""; is_thinking = False
                                current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                            else: accumulated_thoughts += current_text_to_process; current_text_to_process = ""
                        else:
                            start_tag_pos = current_text_to_process.find("<think>")
                            if start_tag_pos != -1:
                                text_part = current_text_to_process[:start_tag_pos]; processed_text_chunk += text_part
                                is_thinking = True; current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                            else: processed_text_chunk += current_text_to_process; current_text_to_process = ""
                    # Yield processed parts
                    if processed_text_chunk or processed_thoughts_chunk:
                        yield {"type": "chunk", "content": processed_text_chunk if processed_text_chunk else None, "thoughts": processed_thoughts_chunk}
                    # --- End Stream parsing ---

            # --- After the Loop ---
            if is_thinking and accumulated_thoughts: # Handle incomplete thought
                logger.warning(f"Ollama stream ended mid-thought for '{self.binding_instance_name}'."); final_metadata["incomplete_thoughts"] = accumulated_thoughts

            logger.info(f"Ollama stream finished for '{self.binding_instance_name}'.")
            final_metadata.update(last_chunk_stats) # Add final stats

            # Update usage info in metadata
            if 'prompt_eval_count' in final_metadata and 'eval_count' in final_metadata:
                 final_metadata["usage"] = {
                     "prompt_tokens": final_metadata['prompt_eval_count'],
                     "completion_tokens": final_metadata['eval_count'],
                     "total_tokens": final_metadata['prompt_eval_count'] + final_metadata['eval_count']
                 }

            # Re-parse full text to get final cleaned output and thoughts
            final_cleaned_text, final_thoughts_str = parse_thought_tags(full_raw_response_text)
            if final_metadata.get("incomplete_thoughts"): # Append incomplete thoughts
                 incomplete = final_metadata["incomplete_thoughts"]
                 final_thoughts_str = (final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + incomplete).strip() if final_thoughts_str else incomplete

            final_output_list = [{"type": "text", "data": final_cleaned_text.strip(), "thoughts": final_thoughts_str, "metadata": final_metadata}]
            yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}

        except ollama.ResponseError as e:
            logger.error(f"Ollama API Error during stream from {self.host} for instance '{self.binding_instance_name}' (Status {e.status_code}): {e}")
            if e.status_code == 404 and "model" in str(e) and "not found" in str(e): yield {"type": "error", "content": f"Ollama model '{effective_model_name}' not found locally. Pull it first (e.g., `ollama pull {effective_model_name}`)."}
            else: yield {"type": "error", "content": f"Ollama API Error ({e.status_code}): {e}"}
        except Exception as e:
            logger.error(f"Ollama stream error from {self.host} for instance '{self.binding_instance_name}': {e}", exc_info=True)
            yield {"type": "error", "content": f"Unexpected Ollama stream error: {e}"}

    # --- Tokenizer / Info Methods ---

    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        """Uses tiktoken to approximate tokenization based on the target model name."""
        target_model = model_name or self.model_name or self.default_model_name or "gpt-4" # Fallback model for tiktoken
        if not tiktoken_installed:
            logger.warning(f"Ollama '{self.binding_instance_name}': Tiktoken not installed. Cannot tokenize. Returning character count based estimate.")
            return list(range(len(text))) # Dummy token IDs based on length

        logger.info(f"Ollama '{self.binding_instance_name}': Estimating tokens for model '{target_model}' with tiktoken.")
        try:
            # Try getting encoding for the specific model name (might fail for Ollama custom names)
            encoding = tiktoken.encoding_for_model(target_model)
        except KeyError:
            # Fallback to a general-purpose encoding if specific model not found
            logger.debug(f"Tiktoken encoding for '{target_model}' not found, using 'cl100k_base'.")
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Ollama binding '{self.binding_instance_name}': Tiktoken failed ({e}). Falling back to simple estimate.")
            return list(range(len(text))) # Dummy token IDs

        tokens = encoding.encode(text)
        # BOS/EOS handling is not straightforward with tiktoken or Ollama's API
        if add_bos: logger.debug("add_bos requested but not supported by tiktoken/Ollama tokenize approximation.")
        if add_eos: logger.debug("add_eos requested but not supported by tiktoken/Ollama tokenize approximation.")
        return tokens

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        """Uses tiktoken to approximate detokenization based on the target model name."""
        target_model = model_name or self.model_name or self.default_model_name or "gpt-4"
        if not tiktoken_installed:
            logger.warning(f"Ollama '{self.binding_instance_name}': Tiktoken not installed. Cannot detokenize.")
            return f"(detokenization unavailable: {len(tokens)} tokens)"

        logger.info(f"Ollama '{self.binding_instance_name}': Approximating detokenization for model '{target_model}' with tiktoken.")
        try:
            try: encoding = tiktoken.encoding_for_model(target_model)
            except KeyError: encoding = tiktoken.get_encoding("cl100k_base")
            text = encoding.decode(tokens)
            return text
        except Exception as e:
            logger.warning(f"Ollama binding '{self.binding_instance_name}': Tiktoken detokenization failed ({e}). Returning token IDs.")
            return str(tokens) # Fallback

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns information about a specific Ollama model or the instance's default/current model.
        """
        target_model = model_name # Use specified model if provided

        # If no specific model requested, determine the active/default one
        if target_model is None:
             if self._model_loaded and self.model_name:
                 target_model = self.model_name
                 logger.debug(f"Getting info for currently loaded model: {target_model}")
                 # Use cached details if available for the loaded model
                 if self.current_model_details and self.current_model_details.get('name') == target_model:
                     return self.current_model_details # Return cached details
             elif self.default_model_name:
                 target_model = self.default_model_name
                 logger.debug(f"No model loaded, getting info for instance default: {target_model}")
             else:
                 logger.warning(f"Ollama instance '{self.binding_instance_name}': Cannot get model info - no model specified, loaded, or configured as default.")
                 # Return empty structure matching GetModelInfoResponse
                 return { "binding_instance_name": self.binding_instance_name, "model_name": None, "model_type": None, "context_size": None, "max_output_tokens": None, "supports_vision": False, "supports_audio": False, "supports_streaming": True, "details": {} }

        # Fetch details for the target model from the Ollama server
        if not self.client:
            logger.error(f"Ollama client not initialized for instance '{self.binding_instance_name}'. Cannot fetch model info.")
            return { "binding_instance_name": self.binding_instance_name, "model_name": target_model, "error": "Client not initialized", "details": {} }

        logger.info(f"Ollama '{self.binding_instance_name}': Fetching info for model '{target_model}' from {self.host}.")
        try:
            model_info_resp = await self.client.show(model=target_model)
            parsed_details = self._parse_ollama_details(model_info_resp)

            # Add binding instance name and ensure all keys for GetModelInfoResponse exist
            parsed_details["binding_instance_name"] = self.binding_instance_name
            parsed_details.setdefault("model_type", None) # Should be set by parser
            parsed_details.setdefault("context_size", None)
            parsed_details.setdefault("max_output_tokens", None)
            parsed_details.setdefault("supports_vision", False)
            parsed_details.setdefault("supports_audio", False)
            parsed_details.setdefault("supports_streaming", True) # Ollama binding supports streaming
            parsed_details.setdefault("details", {})

            # If this was the currently active model, update cache
            if self.model_name == target_model:
                 self.current_model_details = parsed_details.copy()
                 self.model_supports_vision = parsed_details.get("supports_vision", False)

            return parsed_details

        except ollama.ResponseError as e:
            if e.status_code == 404:
                 logger.warning(f"Ollama model '{target_model}' not found locally on {self.host}.")
                 # Return minimal info indicating it's not found
                 return { "binding_instance_name": self.binding_instance_name, "model_name": target_model, "error": "Model not found locally", "details": {"status": "not_found"} }
            else:
                 logger.error(f"Ollama API error fetching info for model '{target_model}': {e}")
                 return { "binding_instance_name": self.binding_instance_name, "model_name": target_model, "error": f"API Error ({e.status_code}): {e}", "details": {} }
        except Exception as e:
            logger.error(f"Unexpected error fetching Ollama model info for '{target_model}': {e}", exc_info=True)
            return { "binding_instance_name": self.binding_instance_name, "model_name": target_model, "error": f"Unexpected error: {e}", "details": {} }