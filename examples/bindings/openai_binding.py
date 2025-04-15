# examples/bindings/openai_binding.py
import asyncio
import logging
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List 
from contextlib import nullcontext

try:
    from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError
    from openai.types.model import Model as OpenaiModelType
except ImportError:
    raise ImportError("OpenAI binding requires 'openai' package. Install it using 'pip install openai'")

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.api.models import StreamChunk
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAIBinding(Binding):
    """Binding implementation for OpenAI compatible APIs (like OpenAI, Groq, etc.)."""
    binding_type_name = "openai_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        # Configuration expects 'api_key' and optionally 'base_url'
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url") # Optional, defaults to OpenAI
        # Fallback: Check environment variables if not found in config
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                logger.info(f"OpenAI Binding '{self.binding_name}': Loaded API key from OPENAI_API_KEY environment variable.")
        if not self.api_key:
                # Consider raising error or logging warning? For now, log warning.
                logger.warning(f"OpenAI Binding '{self.binding_name}': Missing 'api_key' in configuration.")
                # Allow initialization but health/generate will fail

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.current_model_name: Optional[str] = None # OpenAI doesn't require explicit loading/unloading like local models
    def _parse_openai_details(self, model: OpenaiModelType) -> Dict[str, Any]:
        """Parses the raw OpenAI model data into standardized fields."""
        parsed = {}
        parsed['name'] = model.id # Use the model ID as the name

        # Parse datetime from Unix timestamp
        try:
            parsed['modified_at'] = datetime.fromtimestamp(model.created) if model.created else None
        except Exception:
            parsed['modified_at'] = None

        # OpenAI API doesn't provide these directly in list response
        parsed['context_size'] = None
        parsed['max_output_tokens'] = None # Not directly provided        
        parsed['size'] = None
        parsed['quantization_level'] = None
        parsed['format'] = "api" # Indicate it's an API model
        parsed['family'] = None # Could try to infer from name? Risky.
        parsed['families'] = None
        parsed['parameter_size'] = None # Not provided
        parsed['context_size'] = None # Often needs separate check or documentation lookup
        parsed['template'] = None
        parsed['license'] = None

        # Include other fields in generic details
        parsed['details'] = {
            "original_id": model.id,
            "object": model.object,
            "owned_by": model.owned_by,
            # Add other fields from model object if needed
        }
        return parsed


    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available via API, extracting standardized details."""
        logger.info(f"OpenAI Binding '{self.binding_name}': Listing available models via API...")
        if not self.api_key:
             # ... (error handling) ...
             raise ValueError("API key not configured for OpenAI binding.")
        try:
            models_response = await self.client.models.list()

            # --- Format the output using helper ---
            available_models = []
            for model in models_response.data:
                parsed_data = self._parse_openai_details(model)
                if parsed_data.get('name'):
                    available_models.append(parsed_data)
            # --- End Formatting ---

            logger.info(f"OpenAI Binding '{self.binding_name}': Found {len(available_models)} models.")
            return available_models
        except OpenAIError as e:
            logger.error(f"OpenAI Binding '{self.binding_name}': API Error listing models: {e}")
            raise RuntimeError(f"OpenAI API Error: {e}") from e
        except Exception as e:
            logger.error(f"OpenAI Binding '{self.binding_name}': Unexpected error listing models: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error listing models: {e}") from e


    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the OpenAI binding."""
        return {
            "type_name": cls.binding_type_name,
            "version": "1.1",
            "description": "Binding for OpenAI and compatible APIs (GPT models, etc.). Supports TTT.",
            "supports_streaming": True,
            "requirements": ["openai>=1.0.0"],
            "config_template": {
                "type": {"type": "string", "value": cls.binding_type_name, "description":"The type name for the binding", "required":True},
                "api_key": {"type": "string", "value": "YOUR_API_KEY_HERE", "description":"The API key for the service. Can be omitted if OPENAI_API_KEY environment variable is set.", "required":False},
                "base_url": {"type": "string", "value": None, "description":"Optional base URL for OpenAI compatible APIs (e.g., Groq, Azure)", "required":False}
            }
        }
    async def health_check(self) -> Tuple[bool, str]:
        """Checks if the API key is valid and connection is possible."""
        if not self.api_key:
            return False, "API key is not configured."
        try:
            logger.info(f"OpenAI Binding '{self.binding_name}': Performing health check (listing models)...")
            # A simple way to test the connection and key
            models = await self.client.models.list()
            # logger.debug(f"Available models: {[m.id for m in models.data]}") # Might be verbose
            logger.info(f"OpenAI Binding '{self.binding_name}': Health check successful (found {len(models.data)} models).")
            return True, f"Connection successful. Found {len(models.data)} models."
        except APIConnectionError as e:
                logger.error(f"OpenAI Binding '{self.binding_name}': Health check failed (Connection Error): {e}")
                return False, f"API Connection Error: {e}"
        except RateLimitError as e:
                logger.warning(f"OpenAI Binding '{self.binding_name}': Health check hit rate limit: {e}")
                # Consider this healthy? Or potentially unhealthy if persistent? Let's say healthy.
                return True, f"Rate limit hit during check, but connection seems ok: {e}"
        except OpenAIError as e:
                logger.error(f"OpenAI Binding '{self.binding_name}': Health check failed (OpenAI Error): {e}")
                # Check if it's an authentication error
                if hasattr(e, 'status_code') and e.status_code == 401:
                    return False, f"Authentication Error: Invalid API Key? ({e})"
                return False, f"OpenAI API Error: {e}"
        except Exception as e:
                logger.error(f"OpenAI Binding '{self.binding_name}': Health check failed (Unexpected Error): {e}", exc_info=True)
                return False, f"Unexpected Error: {e}"


    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """OpenAI binding does not use local GPU resources."""
        return {"gpu_required": False}

    async def load_model(self, model_name: str) -> bool:
        """OpenAI models are managed remotely, no local loading needed."""
        # We just store the intended model name.
        async with self._load_lock:
                logger.info(f"OpenAI Binding '{self.binding_name}': Setting active model to '{model_name}'. No local loading required.")
                # Optionally, check if model exists via API?
                # try:
                #     await self.client.models.retrieve(model_name)
                # except NotFoundError:
                #     logger.error(f"OpenAI Binding '{self.binding_name}': Model '{model_name}' not found via API.")
                #     return False
                # except Exception as e:
                #      logger.warning(f"OpenAI Binding '{self.binding_name}': Could not verify model '{model_name}' exists: {e}")
                #      # Proceed anyway, generation will fail if it doesn't exist

                self.model_name = model_name
                self._model_loaded = True # Mark as "loaded" conceptually
                return True


    async def unload_model(self) -> bool:
        """No local unloading needed for OpenAI models."""
        async with self._load_lock:
            if not self._model_loaded:
                return True
            logger.info(f"OpenAI Binding '{self.binding_name}': Unsetting active model '{self.model_name}'. No local unloading required.")
            self.model_name = None
            self._model_loaded = False
            return True


    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> Union[str, Dict[str, Any]]:
        """Generates text using the OpenAI API (non-streaming)."""
        if not self._model_loaded or not self.model_name:
                logger.error(f"OpenAI Binding '{self.binding_name}': Cannot generate, model not set.")
                raise RuntimeError("Model not set for OpenAI binding.")
        if not self.api_key:
                logger.error(f"OpenAI Binding '{self.binding_name}': Cannot generate, API key not configured.")
                raise RuntimeError("API key not configured.")

        logger.info(f"OpenAI Binding '{self.binding_name}': Generating text with model '{self.model_name}' (non-streaming)...")

        # Adapt parameters for OpenAI API
        # Example: support 'max_tokens', 'temperature', 'top_p', 'system_message'
        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)
        # Handle system message - requires messages format
        system_message = params.get("system_message", None) # Get from params or personality later
        messages = []
        if system_message:
                messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False # Ensure non-streaming
            )
            completion = response.choices[0].message.content
            logger.info(f"OpenAI Binding '{self.binding_name}': Generation successful.")
            # TODO: Extract usage stats from response.usage if needed
            return completion.strip() if completion else ""

        except OpenAIError as e:
            logger.error(f"OpenAI Binding '{self.binding_name}': API Error during generation: {e}")
            raise RuntimeError(f"OpenAI API Error: {e}") from e
        except Exception as e:
            logger.error(f"OpenAI Binding '{self.binding_name}': Unexpected error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error: {e}") from e


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates text using the OpenAI API (streaming)."""
        if not self._model_loaded or not self.model_name:
            logger.error(f"OpenAI Binding '{self.binding_name}': Cannot generate stream, model not set.")
            yield StreamChunk(type="error", content="Model not set for OpenAI binding.").model_dump()
            return
        if not self.api_key:
            logger.error(f"OpenAI Binding '{self.binding_name}': Cannot generate stream, API key not configured.")
            yield StreamChunk(type="error", content="API key not configured.").model_dump()
            return

        logger.info(f"OpenAI Binding '{self.binding_name}': Generating text stream with model '{self.model_name}'...")

        # Adapt parameters
        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)
        system_message = params.get("system_message", None)
        messages = []
        if system_message:
                messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        full_response_content = ""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                chunk_content = delta.content
                if chunk_content:
                    full_response_content += chunk_content
                    # Yield in StreamChunk format
                    yield StreamChunk(type="chunk", content=chunk_content).model_dump()

            # Yield final message after stream ends
            yield StreamChunk(type="final", content=full_response_content, metadata={"reason": "completed"}).model_dump()
            logger.info(f"OpenAI Binding '{self.binding_name}': Stream finished.")

        except OpenAIError as e:
            logger.error(f"OpenAI Binding '{self.binding_name}': API Error during stream generation: {e}")
            yield StreamChunk(type="error", content=f"OpenAI API Error: {e}").model_dump()
        except Exception as e:
            logger.error(f"OpenAI Binding '{self.binding_name}': Unexpected error during stream generation: {e}", exc_info=True)
            yield StreamChunk(type="error", content=f"Unexpected error: {e}").model_dump()
