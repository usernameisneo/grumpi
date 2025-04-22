# zoos/bindings/ollama_binding.py
import asyncio
import logging
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
import base64
from io import BytesIO
import pipmaster as pm
pm.install_if_missing("ollama")
pm.install_if_missing("pillow")
try:
    import ollama
    from PIL import Image
    ollama_installed = True
except ImportError:
    ollama = None; Image = None; BytesIO = None
    ollama_installed = False

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
try:
    from lollms_server.api.models import StreamChunk, InputData
except ImportError:
     class StreamChunk: pass # type: ignore
     class InputData: pass # type: ignore
from contextlib import nullcontext
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaBinding(Binding):
    """Binding for Ollama inference server."""
    binding_type_name = "ollama_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        if not ollama_installed: raise ImportError("Ollama binding requires 'ollama' and 'pillow'.")
        self.host = self.config.get("host", "http://localhost:11434")
        logger.info(f"Ollama Binding '{self.binding_name}': Using host '{self.host}'.")
        self.client = ollama.AsyncClient(host=self.host)
        self.model_name: Optional[str] = None
        self.model_supports_vision: bool = False

    def _parse_ollama_details(self, model_obj: Any) -> Dict[str, Any]:
        """Parses the raw Ollama model object attributes."""
        parsed = {}; generic_details = {}; name = getattr(model_obj, 'name', None)
        if not name: return {}
        parsed['name'] = name; parsed['size'] = getattr(model_obj, 'size', None)
        try: parsed['modified_at'] = datetime.fromisoformat(getattr(model_obj, 'modified_at', '').replace("Z", "+00:00")) if getattr(model_obj, 'modified_at', None) else None
        except ValueError: parsed['modified_at'] = None
        details_obj = getattr(model_obj, 'details', None)
        if details_obj:
            parsed['format'] = getattr(details_obj, 'format', None); parsed['families'] = getattr(details_obj, 'families', None)
            parsed['family'] = getattr(details_obj, 'family', None) or (parsed['families'][0] if parsed['families'] else None)
            parsed['parameter_size'] = getattr(details_obj, 'parameter_size', None); parsed['quantization_level'] = getattr(details_obj, 'quantization_level', None)
            parsed['context_size'] = None; parsed['max_output_tokens'] = None
            detail_keys_parsed = {'format', 'family', 'families', 'parameter_size', 'quantization_level'}
            try:
                for key, value in vars(details_obj).items():
                    if key not in detail_keys_parsed: generic_details[key] = value
            except TypeError: pass
        supports_vision = any(tag in name.lower() for tag in ['llava', 'vision', 'bakllava'])
        parsed['supports_vision'] = supports_vision; parsed['supports_audio'] = False
        standard_keys = {'name', 'size', 'modified_at', 'details'}
        try:
            for key, value in vars(model_obj).items():
                if key not in standard_keys: generic_details[key] = value
        except TypeError:
             digest_val = getattr(model_obj, 'digest', None)
             if digest_val: generic_details['digest'] = digest_val
        parsed['details'] = generic_details; return parsed

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available locally via Ollama."""
        logger.info(f"Ollama '{self.binding_name}': Listing local models..."); formatted_models = []
        try:
            response_obj = await self.client.list(); ollama_models = getattr(response_obj, 'models', [])
            if not isinstance(ollama_models, list): logger.error(f"Ollama list gave unexpected type: {type(ollama_models)}"); return []
            logger.info(f"Ollama '{self.binding_name}': Found {len(ollama_models)} models.")
            for model_obj in ollama_models:
                parsed_data = self._parse_ollama_details(model_obj)
                if parsed_data.get('name'): formatted_models.append(parsed_data)
                else: logger.warning(f"Skipping unparsable Ollama model data: {model_obj}")
            return formatted_models
        except ollama.ResponseError as e: logger.error(f"Ollama API Error list (Status {e.status_code}): {e}"); raise RuntimeError(f"Ollama API Error: {e}") from e
        except Exception as e: logger.error(f"Unexpected error listing Ollama models: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the Ollama binding."""
        return { "type_name": cls.binding_type_name, "version": "1.1", "description": "Binding for local Ollama server (TTT, Vision).", "supports_streaming": True, "requirements": ["ollama>=0.1.7", "pillow"], "config_template": { "type": cls.binding_type_name, "host": "http://localhost:11434" } }

    # --- IMPLEMENTED CAPABILITIES ---
    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        modalities = ['text']
        if self.model_supports_vision: modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text']
    # --- END IMPLEMENTED CAPABILITIES ---

    async def health_check(self) -> Tuple[bool, str]:
        """Checks connection to the Ollama server."""
        try: response = await self.client.list(); models = getattr(response, 'models', []); return True, f"Connection OK ({len(models)} models found)."
        except ollama.ResponseError as e: logger.error(f"Ollama Health check fail ({e.status_code}): {e}"); return False, f"Ollama Response Error {e.status_code}: {e}"
        except Exception as e: logger.error(f"Ollama Health check fail: {e}", exc_info=True); return False, f"Connection/Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Ollama manages its own resources."""
        return {"gpu_required": True, "estimated_vram_mb": 0}

    async def load_model(self, model_name: str) -> bool:
        """Checks if Ollama model exists, sets internal state."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name: return True
            logger.info(f"Ollama '{self.binding_name}': Checking/setting model '{model_name}'.")
            try:
                model_info_resp = await self.client.show(model=model_name)
                self.model_name = model_name
                details = getattr(model_info_resp, 'details', None)
                families = getattr(details, 'families', []) if details else []
                self.model_supports_vision = any(f in ['clip', 'llava'] for f in families); logger.info(f"Ollama model '{model_name}' found. Vision: {self.model_supports_vision}")
                self._model_loaded = True; return True
            except ollama.ResponseError as e:
                if e.status_code == 404: logger.warning(f"Ollama model '{model_name}' not found locally. Will attempt pull.")
                else: logger.error(f"Ollama error checking model '{model_name}': {e}")
                self.model_name = model_name; self.model_supports_vision = any(tag in model_name.lower() for tag in ['llava','vision']); self._model_loaded = True; return True
            except Exception as e: logger.error(f"Unexpected error setting Ollama model '{model_name}': {e}", exc_info=True); self.model_name = None; self._model_loaded = False; return False

    async def unload_model(self) -> bool:
        """Ollama handles unloading internally."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Ollama '{self.binding_name}': Unsetting model '{self.model_name}'.")
            self.model_name = None; self._model_loaded = False; self.model_supports_vision = False; return True

    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> Union[str, Dict[str, Any]]:
        """Generates text using the Ollama API (non-streaming)."""
        if not self._model_loaded or not self.model_name: raise RuntimeError("Model not set for Ollama.")
        logger.info(f"Ollama '{self.binding_name}': Generating non-stream with '{self.model_name}'...")
        options = params.get("options", {})
        if "temperature" in params: options["temperature"] = params["temperature"]
        if "max_tokens" in params: options["num_predict"] = params["max_tokens"]
        if "top_p" in params: options["top_p"] = params["top_p"]
        stop = params.get("stop_sequences") or params.get("stop")
        if stop: options["stop"] = stop if isinstance(stop, list) else [stop]
        system_message = params.get("system_message", None); images_b64 = []
        if self.model_supports_vision and multimodal_data:
             if not Image or not BytesIO or not base64: logger.warning("Cannot process images: Pillow/base64 not available.");
             else:
                 for item in multimodal_data:
                     if item.type == 'image' and item.data:
                         try: images_b64.append(item.data); logger.info(f"Included image (role: {item.role}) for Ollama.")
                         except Exception as e: logger.error(f"Failed to decode/validate image for Ollama: {e}")
        try:
            response = await self.client.generate( model=self.model_name, prompt=prompt, system=system_message, options=options, images=images_b64 or None, stream=False )
            completion = response.get("response"); logger.info(f"Ollama '{self.binding_name}': Generation successful.")
            return {"text": completion.strip() if completion else ""}
        except ollama.ResponseError as e: logger.error(f"Ollama API Error (Status {e.status_code}): {e}"); raise RuntimeError(f"Ollama API Error: {e}") from e
        except Exception as e: logger.error(f"Ollama unexpected error: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates text using the Ollama API (streaming)."""
        if not self._model_loaded or not self.model_name: yield {"type": "error", "content": "Model not set."}; return
        if not self.client: yield {"type": "error", "content": "Ollama client not initialized."}; return
        logger.info(f"Ollama '{self.binding_name}': Generating stream with '{self.model_name}'...")
        options = params.get("options", {})
        if "temperature" in params: options["temperature"] = params["temperature"]
        if "max_tokens" in params: options["num_predict"] = params["max_tokens"]
        if "top_p" in params: options["top_p"] = params["top_p"]
        stop = params.get("stop_sequences") or params.get("stop")
        if stop: options["stop"] = stop if isinstance(stop, list) else [stop]
        system_message = params.get("system_message", None); images_b64 = []
        if self.model_supports_vision and multimodal_data:
             if not Image or not BytesIO or not base64: logger.warning("Cannot process images: Pillow/base64 not available.");
             else:
                 for item in multimodal_data:
                     if item.type == 'image' and item.data:
                         try: images_b64.append(item.data); logger.info(f"Included image (role: {item.role}) for Ollama stream.")
                         except Exception as e: logger.error(f"Failed decode image for Ollama stream: {e}")
        full_response_content = ""; final_metadata = {}
        try:
            stream = await self.client.generate( model=self.model_name, prompt=prompt, system=system_message, options=options, images=images_b64 or None, stream=True )
            async for chunk in stream:
                chunk_content = chunk.get('response', None)
                if chunk_content: full_response_content += chunk_content; yield {"type": "chunk", "content": chunk_content}
                is_done = chunk.get('done', False)
                if is_done:
                    logger.info("Ollama stream finished."); known_attrs = [ 'model', 'created_at', 'done', 'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration' ]
                    for attr_name in known_attrs:
                        if attr_name in chunk: final_metadata[attr_name] = chunk[attr_name]
                    yield {"type": "final", "content": {"text": full_response_content}, "metadata": final_metadata}; break
        except ollama.ResponseError as e: logger.error(f"Ollama API Error stream ({e.status_code}): {e}"); yield {"type": "error", "content": f"Ollama API Error: {e}"}
        except Exception as e: logger.error(f"Ollama stream error: {e}", exc_info=True); yield {"type": "error", "content": f"Unexpected error: {e}"}