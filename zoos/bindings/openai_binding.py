# zoos/bindings/openai_binding.py
import asyncio
import ascii_colors as logging
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import nullcontext
import os
from datetime import datetime
import pipmaster as pm
pm.install_if_missing("openai")
pm.install_if_missing("pillow")

try:
    from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, NotFoundError
    from openai.types.model import Model as OpenaiModelType
    openai_installed = True
except ImportError:
    OpenaiModelType = Any # type: ignore
    openai_installed = False

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
try:
    from lollms_server.api.models import StreamChunk, InputData
except ImportError:
     class StreamChunk: pass # type: ignore
     class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

class OpenAIBinding(Binding):
    """Binding for OpenAI and compatible APIs."""
    binding_type_name = "openai_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        if not openai_installed: raise ImportError("OpenAI binding requires 'openai'. Install with: pip install openai")
        self.api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.base_url = self.config.get("base_url")
        if not self.api_key: logger.warning(f"OpenAI Binding '{self.binding_name}': API key not found.")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name: Optional[str] = None
        self.model_supports_vision: bool = False

    def _parse_openai_details(self, model: OpenaiModelType) -> Dict[str, Any]:
        """Parses the raw OpenAI model data."""
        parsed = {}; name = model.id; parsed['name'] = name
        try: parsed['modified_at'] = datetime.fromtimestamp(model.created) if model.created else None
        except Exception: parsed['modified_at'] = None
        parsed['context_size'] = None; parsed['max_output_tokens'] = None; parsed['size'] = None
        parsed['quantization_level'] = None; parsed['format'] = "api"; parsed['family'] = None
        parsed['families'] = None; parsed['parameter_size'] = None; parsed['template'] = None; parsed['license'] = None
        supports_vision = any(tag in name.lower() for tag in ['vision', 'gpt-4v', 'gpt-4o'])
        parsed['supports_vision'] = supports_vision; parsed['supports_audio'] = False
        parsed['details'] = { "original_id": model.id, "object": model.object, "owned_by": model.owned_by, }
        return parsed

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available via API."""
        logger.info(f"OpenAI '{self.binding_name}': Listing models..."); available_models = []
        if not self.api_key: raise ValueError("API key not configured.")
        try:
            models_response = await self.client.models.list()
            for model in models_response.data:
                parsed_data = self._parse_openai_details(model)
                if parsed_data.get('name'): available_models.append(parsed_data)
            logger.info(f"OpenAI '{self.binding_name}': Found {len(available_models)} models.")
            return available_models
        except OpenAIError as e: logger.error(f"OpenAI API Error listing models: {e}"); raise RuntimeError(f"OpenAI API Error: {e}") from e
        except Exception as e: logger.error(f"Unexpected error listing OpenAI models: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the OpenAI binding."""
        return { "type_name": cls.binding_type_name, "version": "1.2", "description": "Binding for OpenAI & compatible APIs.", "supports_streaming": True, "requirements": ["openai>=1.0.0"], "config_template": { "type": {"type": "string", "value": cls.binding_type_name, "required":True}, "api_key": {"type": "string", "value": "", "required":False}, "base_url": {"type": "string", "value": None, "required":False} } }

    # --- IMPLEMENTED CAPABILITIES ---
    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        modalities = ['text']
        if self.model_supports_vision: modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text'] # Chat completions only output text
    # --- END IMPLEMENTED CAPABILITIES ---

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity."""
        if not self.api_key: return False, "API key not configured."
        try: models = await self.client.models.list(); return True, f"Connection OK ({len(models.data)} models)."
        except APIConnectionError as e: logger.error(f"Health check fail (Connection Error): {e}"); return False, f"API Connection Error: {e}"
        except RateLimitError as e: logger.warning(f"Health check rate limit: {e}"); return True, f"Rate limit hit: {e}"
        except OpenAIError as e:
            logger.error(f"Health check fail (OpenAI Error): {e}")
            if hasattr(e, 'status_code') and e.status_code == 401: return False, f"Authentication Error: {e}"
            return False, f"OpenAI API Error: {e}"
        except Exception as e: logger.error(f"Health check fail (Error): {e}", exc_info=True); return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """OpenAI is remote, no local GPU needed."""
        return {"gpu_required": False}

    async def load_model(self, model_name: str) -> bool:
        """Sets the model name and checks vision support."""
        async with self._load_lock:
            logger.info(f"OpenAI '{self.binding_name}': Setting model to '{model_name}'.")
            self.model_supports_vision = any(tag in model_name.lower() for tag in ['vision', 'gpt-4v', 'gpt-4o'])
            logger.info(f"Model '{model_name}' vision support: {self.model_supports_vision}")
            self.model_name = model_name; self._model_loaded = True; return True

    async def unload_model(self) -> bool:
        """Unsets the active model name."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"OpenAI '{self.binding_name}': Unsetting model '{self.model_name}'.")
            self.model_name = None; self._model_loaded = False; self.model_supports_vision = False; return True

    def _prepare_openai_messages(self, prompt: str, system_message: Optional[str], multimodal_data: Optional[List['InputData']]) -> List[Dict[str, Any]]:
        """Constructs the message list for OpenAI API."""
        messages = []
        if system_message: messages.append({"role": "system", "content": system_message})
        user_content_parts = [{"type": "text", "text": prompt}] if prompt else []
        if self.model_supports_vision and multimodal_data:
            image_items = [item for item in multimodal_data if item.type == 'image']
            if image_items:
                logger.info(f"Found {len(image_items)} image(s) for OpenAI request.")
                for item in image_items:
                    if item.data and item.mime_type:
                        image_url = f"data:{item.mime_type};base64,{item.data}"
                        user_content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                    else: logger.warning(f"Skipping image item due to missing data/mime_type: {item.role}")
            elif multimodal_data: logger.warning("Multimodal data provided but no images.")
        elif multimodal_data and not self.model_supports_vision: logger.warning(f"Image data ignored: model '{self.model_name}' lacks vision.")
        if user_content_parts: messages.append({"role": "user", "content": user_content_parts})
        else: logger.error("No user prompt or valid image data for OpenAI request.")
        return messages

    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> Union[str, Dict[str, Any]]:
        """Generates text using the OpenAI API (non-streaming)."""
        if not self._model_loaded or not self.model_name: raise RuntimeError("Model not set.")
        if not self.api_key: raise RuntimeError("API key not configured.")
        logger.info(f"OpenAI '{self.binding_name}': Generating non-stream with '{self.model_name}'...")
        max_tokens = params.get("max_tokens"); temperature = params.get("temperature", 0.7); top_p = params.get("top_p", 1.0)
        system_message = params.get("system_message", None)
        messages = self._prepare_openai_messages(prompt, system_message, multimodal_data)
        if not messages or messages[-1]['role'] != 'user': raise ValueError("Invalid message structure.")
        try:
            api_params = { "model": self.model_name, "messages": messages, "temperature": temperature, "top_p": top_p, "stream": False }
            if max_tokens is not None: api_params["max_tokens"] = max_tokens
            response = await self.client.chat.completions.create(**api_params) # type: ignore
            completion = response.choices[0].message.content
            logger.info("OpenAI generation successful.")
            return {"text": completion.strip() if completion else ""}
        except OpenAIError as e: logger.error(f"OpenAI API Error: {e}"); raise RuntimeError(f"OpenAI API Error: {e}") from e
        except Exception as e: logger.error(f"OpenAI unexpected error: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates text using the OpenAI API (streaming)."""
        if not self._model_loaded or not self.model_name: yield {"type": "error", "content": "Model not set."}; return
        if not self.api_key: yield {"type": "error", "content": "API key not configured."}; return
        logger.info(f"OpenAI '{self.binding_name}': Generating stream with '{self.model_name}'...")
        max_tokens = params.get("max_tokens"); temperature = params.get("temperature", 0.7); top_p = params.get("top_p", 1.0)
        system_message = params.get("system_message", None)
        messages = self._prepare_openai_messages(prompt, system_message, multimodal_data)
        if not messages or messages[-1]['role'] != 'user': yield {"type": "error", "content": "Invalid message structure"}; return
        full_response_content = ""
        try:
            api_params = { "model": self.model_name, "messages": messages, "temperature": temperature, "top_p": top_p, "stream": True }
            if max_tokens is not None: api_params["max_tokens"] = max_tokens
            stream = await self.client.chat.completions.create(**api_params) # type: ignore
            async for chunk in stream:
                delta = chunk.choices[0].delta; chunk_content = delta.content
                if chunk_content: full_response_content += chunk_content; yield {"type": "chunk", "content": chunk_content}
            yield {"type": "final", "content": {"text": full_response_content}, "metadata": {"reason": "completed"}}
            logger.info("OpenAI stream finished.")
        except OpenAIError as e: logger.error(f"OpenAI API Error stream: {e}"); yield {"type": "error", "content": f"OpenAI API Error: {e}"}
        except Exception as e: logger.error(f"OpenAI stream error: {e}", exc_info=True); yield {"type": "error", "content": f"Unexpected error: {e}"}