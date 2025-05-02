# encoding:utf-8
# Project: lollms_server
# File: zoos/bindings/ollama_binding/__init__.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Binding implementation for the Ollama inference server.

import asyncio
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List

# Check/Install dependencies using pipmaster if needed (optional but good practice)
try:
    import pipmaster as pm
    pm.install_if_missing("ollama")
    pm.install_if_missing("pillow")
except ImportError:
    pass

# Import core libraries
try:
    import ollama
    from PIL import Image
    ollama_installed = True
except ImportError:
    ollama = None; Image = None; BytesIO = None # type: ignore
    ollama_installed = False

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
# --- ADDED HELPER IMPORT ---
from lollms_server.utils.helpers import parse_thought_tags
# --------------------------

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
        super().__init__(config, resource_manager)
        if not ollama_installed or not ollama:
             raise ImportError("Ollama binding requires the 'ollama' and 'pillow' libraries. Please install them.")
        self.host = self.config.get("host", "http://localhost:11434")
        logger.info(f"Initializing Ollama Binding instance '{self.binding_instance_name}' -> Host: '{self.host}'.")
        try:
            self.client = ollama.AsyncClient(host=self.host)
        except Exception as e:
             logger.error(f"Failed to initialize Ollama client for instance '{self.binding_instance_name}' with host '{self.host}': {e}")
             trace_exception(e)
             self.client = None
        self.model_name: Optional[str] = None
        self.model_supports_vision: bool = False
        self.current_model_details: Dict[str, Any] = {}

    # --- Helper to parse Ollama model details ---
    def _parse_ollama_details(self, model_obj: Any) -> Dict[str, Any]:
        """Parses the raw Ollama model object attributes."""
        # (Implementation remains the same as before)
        parsed = {}
        name = getattr(model_obj, 'name', getattr(model_obj,'model',None))
        if not name: return {}
        parsed['name'] = name
        parsed['size'] = getattr(model_obj, 'size', None)
        try:
            mod_time_str = getattr(model_obj, 'modified_at', None)
            if mod_time_str:
                if '.' in mod_time_str:
                    parts = mod_time_str.split('.'); ts_str = parts[0]; tz_part = ""
                    if '+' in parts[1]: ms_part, tz_part = parts[1].split('+'); tz_part = '+' + tz_part
                    elif '-' in parts[1]: ms_part, tz_part = parts[1].split('-'); tz_part = '-' + tz_part
                    else: ms_part = parts[1].replace('Z','')
                    final_ts_str = f"{ts_str}.{ms_part[:6]}{tz_part}"
                    try: parsed['modified_at'] = datetime.fromisoformat(final_ts_str)
                    except ValueError: parsed['modified_at'] = datetime.fromisoformat(mod_time_str.replace('Z','+00:00').split('.')[0])
                else: parsed['modified_at'] = datetime.fromisoformat(mod_time_str.replace("Z", "+00:00"))
            else: parsed['modified_at'] = None
        except Exception as time_e: logger.warning(f"Could not parse Ollama modified_at time '{getattr(model_obj, 'modified_at', '')}': {time_e}"); parsed['modified_at'] = None

        details_obj = getattr(model_obj, 'details', None); generic_details = {}
        if details_obj:
            parsed['format'] = getattr(details_obj, 'format', None)
            parsed['families'] = getattr(details_obj, 'families', None)
            parsed['family'] = getattr(details_obj, 'family', None) or (parsed['families'][0] if parsed['families'] else None)
            parsed['parameter_size'] = getattr(details_obj, 'parameter_size', None)
            parsed['quantization_level'] = getattr(details_obj, 'quantization_level', None)
            parsed['context_size'] = None
            params_str = getattr(details_obj, 'parameters', '')
            if params_str and isinstance(params_str, str):
                lines = params_str.split('\n');
                for line in lines:
                    parts = line.split();
                    if len(parts) >= 2 and parts[0] == 'num_ctx':
                         try: parsed['context_size'] = int(parts[1]); break
                         except ValueError: pass
            detail_keys_parsed = {'format', 'family', 'families', 'parameter_size', 'quantization_level', 'parameters'}
            try:
                for key, value in vars(details_obj).items():
                    if key not in detail_keys_parsed: generic_details[key] = value
            except TypeError: pass

        supports_vision = any(tag in name.lower() for tag in ['llava', 'vision', 'bakllava'])
        if parsed.get('family'): supports_vision = supports_vision or any(f in parsed['family'].lower() for f in ['llava', 'vision'])
        elif parsed.get('families'): supports_vision = supports_vision or any(f in fam.lower() for fam in parsed['families'] for f in ['llava', 'vision'])
        parsed['supports_vision'] = supports_vision
        parsed['supports_audio'] = False
        digest_val = getattr(model_obj, 'digest', None)
        if digest_val: generic_details['digest'] = digest_val
        parsed['details'] = generic_details
        parsed.setdefault('max_output_tokens', None)
        return parsed

    # --- Required Binding Methods ---

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available locally via the configured Ollama instance."""
        # (Implementation remains the same as before)
        if not self.client: logger.error(f"Ollama client not initialized for instance '{self.binding_instance_name}'. Cannot list models."); return []
        logger.info(f"Ollama '{self.binding_instance_name}': Listing local models from {self.host}...")
        formatted_models = []
        try:
            response = await self.client.list()
            ollama_models = response.get('models', [])
            for model_dict in ollama_models:
                 model_obj = model_dict
                 parsed_data = self._parse_ollama_details(model_obj)
                 if parsed_data.get('name'): formatted_models.append(parsed_data)
                 else: logger.warning(f"Skipping unparsable Ollama model data: {model_dict}")
            return formatted_models
        except ollama.ResponseError as e: logger.error(f"Ollama API Error listing models from {self.host} (Status {e.status_code}): {e}"); raise RuntimeError(f"Ollama API Error: {e}") from e
        except Exception as e: logger.error(f"Unexpected error listing Ollama models from {self.host}: {e}", exc_info=True); raise RuntimeError(f"Unexpected error contacting Ollama: {e}") from e

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        modalities = ['text']
        if self._model_loaded and self.model_supports_vision: modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text']

    async def health_check(self) -> Tuple[bool, str]:
        """Checks connection to the configured Ollama server."""
        # (Implementation remains the same as before)
        if not self.client: return False, f"Ollama client initialization failed for host '{self.host}' on instance '{self.binding_instance_name}'."
        try:
             response = await self.client.list(); models_count = len(response.get('models', []))
             return True, f"Connection to {self.host} OK ({models_count} models found)."
        except ollama.ResponseError as e: logger.error(f"Ollama Health check failed for instance '{self.binding_instance_name}' at {self.host} ({e.status_code}): {e}"); return False, f"Ollama Response Error {e.status_code}: {e}"
        except Exception as e: logger.error(f"Ollama Health check failed for instance '{self.binding_instance_name}' at {self.host}: {e}", exc_info=True); return False, f"Connection/Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Ollama manages its own resources."""
        return {"gpu_required": True, "estimated_vram_mb": 0}

    async def load_model(self, model_name: str) -> bool:
        """Checks if Ollama model exists locally, stores details."""
        # (Implementation remains the same as before)
        if not self.client: logger.error(f"Ollama client not initialized for instance '{self.binding_instance_name}'. Cannot load model."); return False
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name: logger.info(f"Ollama '{self.binding_instance_name}': Model '{model_name}' already considered active."); return True
            logger.info(f"Ollama '{self.binding_instance_name}': Checking/setting active model to '{model_name}' on host {self.host}.")
            try:
                model_info_resp = await self.client.show(model=model_name)
                self.current_model_details = model_info_resp.__dict__ # Use parser
                self.model_name = model_name
                self.model_supports_vision = self.current_model_details.get('supports_vision', False)
                logger.info(f"Ollama model '{model_name}' confirmed locally. Vision: {self.model_supports_vision}, Context: {self.current_model_details.get('context_size')}")
                self._model_loaded = True; return True
            except ollama.ResponseError as e:
                if e.status_code == 404:
                    logger.warning(f"Ollama model '{model_name}' not found locally on {self.host} for instance '{self.binding_instance_name}'. Will attempt pull on first use."); self.model_name = model_name
                    self.model_supports_vision = any(tag in model_name.lower() for tag in ['llava', 'vision', 'bakllava']); self.current_model_details = {} # Reset details
                    self._model_loaded = True; return True
                else: logger.error(f"Ollama API error checking model '{model_name}' on {self.host} for instance '{self.binding_instance_name}': {e}"); self.model_name = None; self._model_loaded = False; self.current_model_details = {}; return False
            except Exception as e: logger.error(f"Unexpected error setting Ollama model '{model_name}' for instance '{self.binding_instance_name}': {e}", exc_info=True); self.model_name = None; self._model_loaded = False; self.current_model_details = {}; return False

    async def unload_model(self) -> bool:
        """Ollama handles unloading internally."""
        # (Implementation remains the same as before)
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Ollama '{self.binding_instance_name}': Unsetting active model '{self.model_name}'. Ollama server manages actual unloading."); self.model_name = None; self._model_loaded = False; self.model_supports_vision = False; self.current_model_details = {}; return True

    # --- UPDATED generate ---
    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Return List[OutputData]-like
        """Generates text using the Ollama API (non-streaming)."""
        if not self.client: raise RuntimeError(f"Ollama client not initialized for instance '{self.binding_instance_name}'.")
        if not self._model_loaded or not self.model_name: raise RuntimeError(f"Model not set for Ollama instance '{self.binding_instance_name}'. Call load_model first.")

        logger.info(f"Ollama '{self.binding_instance_name}': Generating non-stream with '{self.model_name}'...")

        options = params.get("options", {}).copy() if isinstance(params.get("options"), dict) else {}
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

        if self.model_supports_vision and multimodal_data:
             if not Image or not BytesIO: logger.warning("Cannot process images: Pillow library not found.")
             else:
                 for item in multimodal_data:
                     if item.type == 'image' and item.data and isinstance(item.data, str):
                         try:
                             if len(item.data) > 10 and '=' not in item.data[-4:]: images_b64.append(item.data); logger.info(f"Included image (role: {item.role}) for Ollama non-stream request.")
                             else: logger.warning(f"Skipping image data that doesn't look like base64: role={item.role}")
                         except Exception as e: logger.error(f"Failed to process image data for Ollama non-stream: {e}")
                     elif item.type == 'image': logger.warning(f"Skipping image item with missing/invalid data: role={item.role}")
        elif multimodal_data and not self.model_supports_vision:
            logger.warning(f"Ollama instance '{self.binding_instance_name}' received image data, but model '{self.model_name}' does not support vision. Ignoring images.")

        try:
            logger.debug(f"Ollama non-stream generate call: model={self.model_name}, prompt='{prompt[:50]}...', system='{str(system_message)[:50] if system_message else None}...', options={options}, images_count={len(images_b64)}")
            response = await self.client.generate(model=self.model_name, prompt=prompt, system=system_message, options=options, images=images_b64 if images_b64 else None, stream=False)

            raw_completion = response.get('response', '') # Get raw response text

            # --- ADDED: Parse thoughts ---
            cleaned_completion, thoughts = parse_thought_tags(raw_completion)
            # --------------------------

            logger.info(f"Ollama '{self.binding_instance_name}': Generation successful.")
            output_metadata = {"model_used": self.model_name, "binding_instance": self.binding_instance_name}
            for key in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']:
                 if key in response: output_metadata[key] = response[key]

            # Return standardized list format with cleaned data and thoughts
            return [{
                "type": "text",
                "data": cleaned_completion.strip(),
                "thoughts": thoughts, # Add parsed thoughts
                "metadata": output_metadata
            }]

        except ollama.ResponseError as e:
            logger.error(f"Ollama API Error generating from {self.host} for instance '{self.binding_instance_name}' (Status {e.status_code}): {e}")
            if e.status_code == 404 and "model" in str(e) and "not found" in str(e): raise RuntimeError(f"Ollama model '{self.model_name}' not found locally. You may need to pull it first (e.g., `ollama pull {self.model_name}`).") from e
            raise RuntimeError(f"Ollama API Error ({e.status_code}): {e}") from e
        except Exception as e:
            logger.error(f"Ollama unexpected error generating from {self.host} for instance '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Ollama error: {e}") from e

    # --- UPDATED generate_stream ---
    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Generates text using the Ollama API (streaming)."""
        if not self.client: yield {"type": "error", "content": f"Ollama client not initialized for instance '{self.binding_instance_name}'."}; return
        if not self._model_loaded or not self.model_name: yield {"type": "error", "content": f"Model not set for Ollama instance '{self.binding_instance_name}'."}; return

        logger.info(f"Ollama '{self.binding_instance_name}': Generating stream with '{self.model_name}' from {self.host}...")

        options = params.get("options", {}).copy() if isinstance(params.get("options"), dict) else {}
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

        if self.model_supports_vision and multimodal_data:
             if not Image or not BytesIO: logger.warning("Cannot process images: Pillow library not found.")
             else:
                 for item in multimodal_data:
                     if item.type == 'image' and item.data and isinstance(item.data, str):
                         try:
                             if len(item.data) > 10 and '=' not in item.data[-4:]: images_b64.append(item.data); logger.info(f"Included image (role: {item.role}) for Ollama stream request.")
                             else: logger.warning(f"Skipping image data that doesn't look like base64: role={item.role}")
                         except Exception as e: logger.error(f"Failed to process image data for Ollama stream: {e}")
                     elif item.type == 'image': logger.warning(f"Skipping image item with missing/invalid data: role={item.role}")
        elif multimodal_data and not self.model_supports_vision:
             logger.warning(f"Ollama instance '{self.binding_instance_name}' received image data, but model '{self.model_name}' does not support vision. Ignoring images.")

        full_raw_response_text = ""; accumulated_thoughts = ""; is_thinking = False # State for thought parsing
        final_metadata = {"model_used": self.model_name, "binding_instance": self.binding_instance_name}
        last_chunk_stats = {} # To store stats from the final 'done' chunk

        try:
            logger.debug(f"Ollama stream call: model={self.model_name}, prompt='{prompt[:50]}...', system='{str(system_message)[:50] if system_message else None}...', options={options}, images_count={len(images_b64)}")
            stream = await self.client.generate( model=self.model_name, prompt=prompt, system=system_message, options=options, images=images_b64 if images_b64 else None, stream=True )

            async for chunk in stream:
                if not isinstance(chunk, dict): continue
                is_done = chunk.get('done', False)
                # Store stats from the final chunk before breaking
                if is_done:
                    last_chunk_stats = {k: v for k, v in chunk.items() if k in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']}
                    break # Process final chunk outside the loop

                chunk_raw_content = chunk.get('response', '')
                if chunk_raw_content:
                    full_raw_response_text += chunk_raw_content # Accumulate raw text
                    # --- ADDED: Stream parsing logic ---
                    current_text_to_process = chunk_raw_content
                    processed_text_chunk = ""
                    processed_thoughts_chunk = None
                    while current_text_to_process:
                        if is_thinking:
                            end_tag_pos = current_text_to_process.find("</think>")
                            if end_tag_pos != -1:
                                thought_part = current_text_to_process[:end_tag_pos]
                                accumulated_thoughts += thought_part
                                processed_thoughts_chunk = accumulated_thoughts
                                accumulated_thoughts = ""
                                is_thinking = False
                                current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                            else:
                                accumulated_thoughts += current_text_to_process
                                current_text_to_process = ""
                        else:
                            start_tag_pos = current_text_to_process.find("<think>")
                            if start_tag_pos != -1:
                                text_part = current_text_to_process[:start_tag_pos]
                                processed_text_chunk += text_part
                                is_thinking = True
                                current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                            else:
                                processed_text_chunk += current_text_to_process
                                current_text_to_process = ""
                    # Yield if needed
                    if processed_text_chunk or processed_thoughts_chunk:
                        yield {"type": "chunk", "content": processed_text_chunk if processed_text_chunk else None, "thoughts": processed_thoughts_chunk}
                    # --- End Stream parsing logic ---

            # After loop finishes
            if is_thinking and accumulated_thoughts: # Handle incomplete thought at stream end
                logger.warning(f"Ollama stream ended mid-thought for '{self.binding_instance_name}'. Thought content:\n{accumulated_thoughts}")
                final_metadata["incomplete_thoughts"] = accumulated_thoughts # Store incomplete thoughts

            logger.info(f"Ollama stream finished for '{self.binding_instance_name}'.")
            final_metadata.update(last_chunk_stats) # Add stats from the last chunk

            # Re-parse full raw text to get final cleaned text and thoughts
            final_cleaned_text, final_thoughts_str = parse_thought_tags(full_raw_response_text)
            if final_metadata.get("incomplete_thoughts"): # Append incomplete thoughts if they exist
                 final_thoughts_str = (final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + final_metadata["incomplete_thoughts"]).strip() if final_thoughts_str else final_metadata["incomplete_thoughts"]

            # Prepare final output list
            final_output_list = [{"type": "text", "data": final_cleaned_text.strip(), "thoughts": final_thoughts_str, "metadata": final_metadata}]
            # Yield final StreamChunk-like dict
            yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}

        except ollama.ResponseError as e:
            logger.error(f"Ollama API Error during stream from {self.host} for instance '{self.binding_instance_name}' (Status {e.status_code}): {e}")
            if e.status_code == 404 and "model" in str(e) and "not found" in str(e): yield {"type": "error", "content": f"Ollama model '{self.model_name}' not found locally. Pull it first (e.g., `ollama pull {self.model_name}`)."}
            else: yield {"type": "error", "content": f"Ollama API Error ({e.status_code}): {e}"}
        except Exception as e:
            logger.error(f"Ollama stream error from {self.host} for instance '{self.binding_instance_name}': {e}", exc_info=True)
            yield {"type": "error", "content": f"Unexpected Ollama stream error: {e}"}


    # --- Tokenizer / Info Methods ---
    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenization not directly supported via the Ollama python library."""
        if not self._model_loaded: raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for tokenization")
        logger.warning(f"Ollama binding '{self.binding_instance_name}': Tokenization requested but not implemented via Python client.")
        raise NotImplementedError("Ollama binding does not support tokenization via the current client library.")

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenization is not supported via the Ollama python library."""
        if not self._model_loaded: raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for detokenization")
        logger.warning(f"Ollama binding '{self.binding_instance_name}': Detokenization requested but not implemented via Python client.")
        raise NotImplementedError("Ollama binding does not support detokenization via the current client library.")

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently active Ollama model."""
        # (Implementation remains the same as before)
        if not self._model_loaded or not self.model_name: return { "name": None, "context_size": None, "max_output_tokens": None, "supports_vision": False, "supports_audio": False, "details": {} }
        return {
            "name": self.current_model_details.get("name", self.model_name),
            "context_size": self.current_model_details.get("context_size"),
            "max_output_tokens": self.current_model_details.get("max_output_tokens"),
            "supports_vision": self.current_model_details.get("supports_vision", self.model_supports_vision),
            "supports_audio": self.current_model_details.get("supports_audio", False),
            "details": self.current_model_details.get("details", {"info":f"Active model for instance '{self.binding_instance_name}': '{self.model_name}'."})
        }