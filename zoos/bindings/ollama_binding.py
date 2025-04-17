# zoos/bindings/ollama_binding.py
import asyncio
import logging
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple

try:
    import ollama
except ImportError:
    raise ImportError("Ollama binding requires 'ollama' package. Install it using 'pip install ollama'")


from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.api.models import StreamChunk
from contextlib import nullcontext
from typing import List
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaBinding(Binding):
    """Binding implementation for Ollama inference server."""
    binding_type_name = "ollama_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        # Configuration expects 'host' (e.g., "http://localhost:11434")
        self.host = self.config.get("host")
        if not self.host:
                logger.warning(f"Ollama Binding '{self.binding_name}': Missing 'host' in configuration. Assuming default 'http://localhost:11434'.")
                self.host = "http://localhost:11434" # Default Ollama host

        # Ollama client doesn't store API keys, connection is direct
        self.client = ollama.AsyncClient(host=self.host)
        self.current_model_name: Optional[str] = None

    # --- Helper to parse Ollama details safely ---
    # Change type hint for model_obj back to Any or object
    def _parse_ollama_details(self, model_obj: Any) -> Dict[str, Any]:
        """Parses the raw Ollama model object attributes into standardized fields."""
        parsed = {}
        generic_details = {}

        # --- Access attributes directly using getattr (remains the same) ---
        parsed['name'] = getattr(model_obj, 'model', None)
        parsed['size'] = getattr(model_obj, 'size', None)
        parsed['modified_at'] = getattr(model_obj, 'modified_at', None)
        parsed['details'] = getattr(model_obj, 'details', None)
        # ... (rest of the getattr calls for size, modified_at, details attributes) ...

        details_obj = getattr(model_obj, 'details', None)
        logger.info(f"ModelObj: {model_obj}")
        logger.info(f"Details: {details_obj}")
        if details_obj:
                parsed['format'] = getattr(details_obj, 'format', None)
                parsed['families='] = getattr(details_obj, 'families', None)
                parsed['quantization_level'] = getattr(details_obj, 'quantization_level', None)
                # ... (rest of getattr for details_obj attributes) ...
                parsed['context_size'] = None
                parsed['max_output_tokens'] = None
                # Add other attributes from details_obj to generic_details
                detail_keys_parsed = {'format', 'family', 'families', 'parameter_size', 'quantization_level'}
                try:
                    for key, value in vars(details_obj).items():
                        if key not in detail_keys_parsed:
                            generic_details[key] = value
                except TypeError:
                    logger.debug("Could not use vars() on details object, skipping extra detail attributes.")


        # Add other top-level attributes from model_obj to generic_details
        standard_keys = {'model', 'name', 'size', 'modified_at', 'details'}
        try:
                for key, value in vars(model_obj).items():
                    if key not in standard_keys:
                        generic_details[key] = value
        except TypeError:
                logger.debug("Could not use vars() on model object, skipping extra top-level attributes.")
                digest_val = getattr(model_obj, 'digest', None)
                if digest_val:
                    generic_details['digest'] = digest_val


        parsed['details'] = generic_details

        return parsed


    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available locally, extracting standardized details."""
        # ... (client check and API call) ...
        try:
            response = await self.client.list()
            ollama_models = getattr(response, 'models', [])
            if not isinstance(ollama_models, list):
                    logger.error(f"Ollama API list returned unexpected type for 'models': {type(ollama_models)}")
                    return []

            logger.info(f"Ollama Binding '{self.binding_name}': Found {len(ollama_models)} models.")

            formatted_models = []
            for model_obj in ollama_models:
                # No specific type check here now, rely on getattr in parser
                parsed_data = self._parse_ollama_details(model_obj)
                if parsed_data.get('name'):
                    formatted_models.append(parsed_data)
                else:
                    logger.warning(f"Skipping Ollama model data because 'model' attribute was missing or data unparsable: {model_obj}")
            return formatted_models

            # --- End Formatting ---
        except ollama.ResponseError as e:
            logger.error(f"Ollama Binding '{self.binding_name}': API Error listing models (Status {e.status_code}): {e}")
            raise RuntimeError(f"Ollama API Error {e.status_code}: {e}") from e
        except Exception as e:
            logger.error(f"Ollama Binding '{self.binding_name}': Unexpected error listing models: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error listing models: {e}") from e
        
    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the Ollama binding."""
        return {
            "type_name": cls.binding_type_name,
            "version": "1.0", # Use ollama package version?
            "description": "Binding for local Ollama inference server. Supports TTT.",
            "supports_streaming": True,
            "requirements": ["ollama>=0.1.7"],
            "config_template": {
                "type": cls.binding_type_name,
                "host": "http://localhost:11434" # Default host
            }
        }

    async def health_check(self) -> Tuple[bool, str]:
        """Checks connection to the Ollama server."""
        try:
            logger.info(f"Ollama Binding '{self.binding_name}': Performing health check (listing local models)...")
            response = await self.client.list()
            models = response.get("models", [])
            logger.info(f"Ollama Binding '{self.binding_name}': Health check successful (found {len(models)} local models).")
            return True, f"Connection successful. Found {len(models)} local models."
        except ollama.ResponseError as e:
                logger.error(f"Ollama Binding '{self.binding_name}': Health check failed (Ollama Response Error {e.status_code}): {e}")
                return False, f"Ollama Response Error {e.status_code}: {e}"
        except Exception as e: # Catch connection errors etc.
                logger.error(f"Ollama Binding '{self.binding_name}': Health check failed (Connection/Unexpected Error): {e}", exc_info=True)
                return False, f"Connection/Unexpected Error: {e}"


    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Ollama manages its own resources, but conceptually it uses GPU if available."""
        # This binding doesn't directly control GPU, Ollama server does.
        # We can report it as potentially needing GPU, but won't use the resource manager here.
        return {"gpu_required": True, "estimated_vram_mb": 0} # VRAM managed by Ollama

    async def load_model(self, model_name: str) -> bool:
        """Ollama loads models on demand, but we can check if it exists."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                    return True

            logger.info(f"Ollama Binding '{self.binding_name}': Checking/setting model '{model_name}'.")
            try:
                # Check if model is available locally, or pull it?
                # Let's try pulling it - Ollama handles download if needed.
                # This might take time. Should we use resource manager if pull is long? Maybe not.
                logger.info(f"Ollama Binding '{self.binding_name}': Ensuring model '{model_name}' is available via pull...")
                # Pull is synchronous in current library version, run in thread?
                # Or just check existence first? Let's check first.
                try:
                    await self.client.show(model=model_name)
                    logger.info(f"Ollama Binding '{self.binding_name}': Model '{model_name}' found locally.")
                except ollama.ResponseError as e:
                    if e.status_code == 404:
                        logger.warning(f"Ollama Binding '{self.binding_name}': Model '{model_name}' not found locally. Generation will attempt to pull it.")
                        # Optionally trigger a pull here, but it might block
                        # await self.client.pull(model=model_name) # This needs async version or thread
                    else:
                        raise # Re-raise other errors

                self.model_name = model_name
                self._model_loaded = True # Mark as conceptually loaded
                return True
            except ollama.ResponseError as e:
                    logger.error(f"Ollama Binding '{self.binding_name}': Error checking/setting model '{model_name}': {e}")
                    self.model_name = None
                    self._model_loaded = False
                    return False
            except Exception as e:
                    logger.error(f"Ollama Binding '{self.binding_name}': Unexpected error setting model '{model_name}': {e}", exc_info=True)
                    self.model_name = None
                    self._model_loaded = False
                    return False

    async def unload_model(self) -> bool:
        """Ollama handles unloading internally based on usage/config."""
        async with self._load_lock:
            if not self._model_loaded:
                return True
            logger.info(f"Ollama Binding '{self.binding_name}': Unsetting active model '{self.model_name}'. Ollama manages actual unloading.")
            self.model_name = None
            self._model_loaded = False
            return True


    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> Union[str, Dict[str, Any]]:
        """Generates text using the Ollama API (non-streaming)."""
        if not self._model_loaded or not self.model_name:
            logger.error(f"Ollama Binding '{self.binding_name}': Cannot generate, model not set.")
            raise RuntimeError("Model not set for Ollama binding.")

        logger.info(f"Ollama Binding '{self.binding_name}': Generating text with model '{self.model_name}' (non-streaming)...")

        # Adapt parameters for Ollama /generate endpoint
        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        options = params.get("options", {}) # Passthrough options directly if provided
        # Or map common params:
        if "temperature" in params: options["temperature"] = params["temperature"]
        if "max_tokens" in params: options["num_predict"] = params["max_tokens"] # Ollama uses num_predict
        if "top_p" in params: options["top_p"] = params["top_p"]
        # Add more mappings as needed (stop sequences, etc.)

        system_message = params.get("system_message", None)
        if system_message:
            logger.debug(f"Using system message: {system_message[:100]}...") # Log beginning
        # Format specification? Ollama might use specific format strings

        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_message,
                options=options,
                stream=False # Ensure non-streaming
            )
            completion = response.get("response")
            logger.info(f"Ollama Binding '{self.binding_name}': Generation successful.")
            # TODO: Extract context, timings etc from response if needed
            return completion.strip() if completion else ""

        except ollama.ResponseError as e:
            logger.error(f"Ollama Binding '{self.binding_name}': API Error during generation (Status {e.status_code}): {e}")
            raise RuntimeError(f"Ollama API Error {e.status_code}: {e}") from e
        except Exception as e:
            logger.error(f"Ollama Binding '{self.binding_name}': Unexpected error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error: {e}") from e


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates text using the Ollama API (streaming)."""
        if not self._model_loaded or not self.model_name:
                logger.error(f"Ollama Binding '{self.binding_name}': Cannot generate stream, model not set.")
                yield StreamChunk(type="error", content="Model not set for Ollama binding.").model_dump()
                return
        if not self.client:
                logger.error(f"Ollama Binding '{self.binding_name}': Client not initialized.")
                yield StreamChunk(type="error", content="Ollama client not initialized.").model_dump()
                return

        logger.info(f"Ollama Binding '{self.binding_name}': Generating text stream with model '{self.model_name}'...")

        options = params.get("options", {})
        if "temperature" in params: options["temperature"] = params["temperature"]
        if "max_tokens" in params: options["num_predict"] = params["max_tokens"]
        if "top_p" in params: options["top_p"] = params["top_p"]
        # Add other options like stop sequences etc. if needed
        # options['stop'] = params.get('stop_sequences', None)

        system_message = params.get("system_message", None)
        if system_message:
                logger.debug(f"Using system message: {system_message[:100]}...") # Log beginning

        full_response_content = ""
        try:
            stream = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_message,
                options=options,
                stream=True
            )
            async for chunk in stream:
                # Chunk is likely an object, access attributes
                chunk_content = getattr(chunk, 'response', None)
                if chunk_content:
                    full_response_content += chunk_content
                    yield StreamChunk(type="chunk", content=chunk_content).model_dump()

                # Check if stream is done using getattr
                is_done = getattr(chunk, 'done', False)
                if is_done:
                    logger.info(f"Ollama Binding '{self.binding_name}': Stream finished (done flag received).")

                    # --- FIX: Extract final metadata using getattr ---
                    final_metadata = {}
                    # Iterate over known attributes of the final chunk object
                    known_attrs = [
                        'model', 'created_at', 'done', 'total_duration', 'load_duration',
                        'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration', 'context'
                        # Add others if the library provides more
                    ]
                    for attr_name in known_attrs:
                        attr_value = getattr(chunk, attr_name, None)
                        if attr_value is not None:
                            final_metadata[attr_name] = attr_value
                    # --- END FIX ---

                    yield StreamChunk(type="final", content=full_response_content, metadata=final_metadata).model_dump()
                    break # Exit loop once done signal is received

        except ollama.ResponseError as e:
            logger.error(f"Ollama Binding '{self.binding_name}': API Error during stream generation (Status {e.status_code}): {e}")
            yield StreamChunk(type="error", content=f"Ollama API Error {e.status_code}: {e}").model_dump()
        except Exception as e:
            logger.error(f"Ollama Binding '{self.binding_name}': Unexpected error during stream generation: {e}", exc_info=True)
            yield StreamChunk(type="error", content=f"Unexpected error: {e}").model_dump()
