# lollms_server/bindings/dalle_binding/__init__.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Binding implementation for OpenAI's DALL-E image generation.
# Modification Date: 2025-05-04
# Refactored model info endpoint, default model handling.

import asyncio
import os
import base64
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from io import BytesIO # Optional: for working with image bytes
from datetime import datetime # For model info listing

# Use pipmaster if needed, or rely on main installation
try:
    import pipmaster as pm
    pm.ensure_packages(["openai", "pillow"])
except ImportError:
    pass # Assume installed or handle import error below

try:
    from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, NotFoundError, BadRequestError
    from PIL import Image # Optional: for potentially validating base64 data
    openai_installed = True
    pillow_installed = True
except ImportError as e:
    # Define placeholders if import fails
    AsyncOpenAI = None; OpenAIError = Exception; APIConnectionError = Exception # type: ignore
    RateLimitError = Exception; NotFoundError = Exception; BadRequestError = Exception # type: ignore
    Image = None; BytesIO = None # type: ignore
    openai_installed = False
    pillow_installed = False
    _import_error_msg = str(e)

try:
    import ascii_colors as logging # Use logging alias
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.utils.helpers import parse_thought_tags # Although not used here, keep for consistency

# Use TYPE_CHECKING for API model imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        class StreamChunk: pass # type: ignore
        class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

# Known DALL-E models and their typical supported sizes/features
DALLE_MODELS_INFO = {
    "dall-e-3": {
        "sizes": ["1024x1024", "1792x1024", "1024x1792"],
        "qualities": ["standard", "hd"],
        "styles": ["vivid", "natural"],
        "max_n": 1 # DALL-E 3 only supports n=1
    },
    "dall-e-2": {
        "sizes": ["256x256", "512x512", "1024x1024"],
        "qualities": ["standard"], # Only standard quality
        "styles": [None], # Style not applicable
        "max_n": 10 # Can generate multiple images
    }
}


class DallEBinding(Binding):
    """Binding implementation for OpenAI's DALL-E image generation models."""
    binding_type_name = "dalle_binding" # MUST match type_name in binding_card.yaml

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the DallEBinding using the instance configuration.

        Args:
            config: The configuration dictionary loaded from the instance's config file,
                    validated against the schema in binding_card.yaml.
            resource_manager: The shared resource manager instance.
        """
        super().__init__(config, resource_manager) # Sets self.config etc.
        if not openai_installed:
            raise ImportError(f"DALL-E binding requires the 'openai' library (>=1.0.0). Error: {_import_error_msg}")
        if not pillow_installed:
             logger.warning("Pillow library not found. Image validation disabled.")


        # --- Configuration Loading from instance config dict (self.config) ---
        self.api_key = self.config.get("api_key") # Get from instance config first
        self.base_url = self.config.get("base_url") # Optional for Azure etc.

        # Fallback: Check environment variables if not in instance config
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                logger.info(f"DALL-E Binding '{self.binding_instance_name}': Loaded API key from OPENAI_API_KEY environment variable.")

        # --- Default Parameters from Instance Config ---
        # self.default_model_name is already set by the parent class __init__
        # from self.config.get("model")
        self.default_model_name= self.config.get("default_model", "dall-e-3")
        self.default_size = self.config.get("default_size", "1024x1024")
        self.default_quality = self.config.get("default_quality", "standard")
        self.default_style = self.config.get("default_style", "vivid")

        # --- Validation and Client Initialization ---
        if not self.api_key:
             logger.warning(f"DALL-E Binding '{self.binding_instance_name}': API key not found in config or environment variables. Health check and generation will likely fail.")
             self.client = None
        else:
             try:
                 self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                 logger.info(f"DALL-E Binding '{self.binding_instance_name}': Initialized client (Base URL: {self.base_url or 'Default OpenAI'}).")
             except Exception as e:
                  logger.error(f"Failed to initialize OpenAI client for DALL-E instance '{self.binding_instance_name}': {e}")
                  trace_exception(e)
                  self.client = None # Mark client as failed

        # self.model_name stores the currently *active* model for this instance (set by load_model)


    # --- Binding Capabilities ---
    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        return ['text'] # DALL-E takes text prompts

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['image']
    # --- End Capabilities ---

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity by trying to list models."""
        # (Implementation remains the same)
        if not self.client: return False, f"OpenAI client initialization failed for DALL-E instance '{self.binding_instance_name}'."
        if not self.api_key: return False, "API key is not configured for this instance."
        try:
            logger.info(f"DALL-E Binding '{self.binding_instance_name}': Performing health check (listing models)...")
            await asyncio.wait_for(self.client.models.list(), timeout=10.0)
            logger.info(f"DALL-E Binding '{self.binding_instance_name}': Health check successful (connection OK).")
            return True, "Connection successful."
        except APIConnectionError as e: logger.error(f"DALL-E '{self.binding_instance_name}': Health check failed (Conn Err): {e}"); return False, f"API Connection Error: {e}"
        except RateLimitError as e: logger.warning(f"DALL-E '{self.binding_instance_name}': Health check rate limit: {e}"); return True, f"Rate limit hit, but connection ok: {e}"
        except OpenAIError as e:
             logger.error(f"DALL-E '{self.binding_instance_name}': Health check failed (OpenAI Err): {e}")
             status = getattr(e, 'status_code', 'N/A')
             if status == 401: return False, f"Authentication Error: Invalid API Key? ({e})"
             return False, f"OpenAI API Error ({status}): {e}"
        except asyncio.TimeoutError: logger.error(f"DALL-E Health check timed out for '{self.binding_instance_name}'."); return False, "Connection timed out."
        except Exception as e: logger.error(f"DALL-E '{self.binding_instance_name}': Health check failed (Unexpected Err): {e}", exc_info=True); return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """DALL-E binding does not use local GPU resources."""
        return {"gpu_required": False}

    async def load_model(self, model_name: str) -> bool:
        """
        Sets the DALL-E model version to use for this instance. Validates against known models.
        Uses instance default ('model' field in config) as fallback if requested model is invalid.
        """
        async with self._load_lock:
            logger.info(f"DALL-E Binding '{self.binding_instance_name}': Setting active model to '{model_name}'.")
            target_model = model_name

            if target_model not in DALLE_MODELS_INFO:
                 # Use instance default from config if the requested one is invalid
                 instance_default = self.default_model_name # From parent __init__
                 logger.warning(f"Requested model '{target_model}' not in known DALL-E models {list(DALLE_MODELS_INFO.keys())}. Trying instance default '{instance_default}'.")
                 target_model = instance_default # Fallback to instance default

                 # Check if instance default is valid
                 if not target_model or target_model not in DALLE_MODELS_INFO:
                     logger.error(f"Instance default model '{target_model}' is also invalid or missing. Cannot activate model.")
                     await self.unload_model() # Ensure state is reset
                     return False

            # Set the validated model name
            self.model_name = target_model
            self._model_loaded = True # Mark as "loaded" (ready to use)
            logger.info(f"DALL-E Binding '{self.binding_instance_name}': Active model set to '{self.model_name}'.")
            return True

    async def unload_model(self) -> bool:
        """Unsets the active DALL-E model."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"DALL-E Binding '{self.binding_instance_name}': Unsetting active model '{self.model_name}'.")
            self.model_name = None
            self._model_loaded = False
            return True

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns a list of known DALL-E models supported by this binding."""
        # (Implementation remains the same)
        logger.info(f"DALL-E Binding '{self.binding_instance_name}': Listing supported models.")
        model_list = []
        for name, details in DALLE_MODELS_INFO.items():
            model_list.append({
                "name": name, "format": "api", "size": None, "modified_at": None,
                "supports_vision": True, "supports_audio": False, "details": details,
                "context_size": None, "max_output_tokens": None, "family": "dall-e",
                "families": ["dall-e"], "parameter_size": None, "quantization_level": None,
            })
        return model_list

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None # Usually None for TTI
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]: # Return List[OutputData]-like
        """Generates an image using the DALL-E API."""
        if not self.client: raise RuntimeError(f"OpenAI client not initialized for DALL-E instance '{self.binding_instance_name}'.")
        if not self.api_key: raise RuntimeError(f"API key not configured for DALL-E instance '{self.binding_instance_name}'.")

        # Use the currently active model set by load_model
        target_model_name = self.model_name
        if not self._model_loaded or not target_model_name:
            # Try loading the instance default model automatically if none active
            instance_default = self.default_model_name
            if instance_default:
                logger.warning(f"No model active for '{self.binding_instance_name}'. Attempting to activate instance default '{instance_default}'.")
                if await self.load_model(instance_default):
                    target_model_name = instance_default
                else:
                    raise RuntimeError(f"Failed to activate default model '{instance_default}' for DALL-E instance '{self.binding_instance_name}'.")
            else:
                 raise RuntimeError(f"No model active or configured as default for DALL-E instance '{self.binding_instance_name}'. Call load_model or set default.")

        # Verify generation type is appropriate
        gen_type = request_info.get("generation_type", "tti")
        if gen_type != 'tti':
             logger.warning(f"DALL-E Binding '{self.binding_instance_name}': Received non-TTI request type '{gen_type}'. Proceeding as TTI.")

        model_info = DALLE_MODELS_INFO.get(target_model_name, {})
        if not model_info: raise RuntimeError(f"Internal error: Active model '{target_model_name}' has no info.")

        # --- Determine Parameters (remains the same, uses instance defaults as fallback) ---
        n = params.get("n", 1); max_n = model_info.get("max_n", 1)
        if n > max_n: logger.warning(f"Clamping n={n} to max {max_n} for {target_model_name}."); n = max_n
        size = params.get("size", self.default_size); supported_sizes = model_info.get("sizes", [self.default_size])
        if size not in supported_sizes: logger.warning(f"Size {size} invalid for {target_model_name}. Using default: {self.default_size}"); size = self.default_size
        quality = params.get("quality", self.default_quality); supported_qualities = model_info.get("qualities", [self.default_quality])
        if quality not in supported_qualities: logger.warning(f"Quality {quality} invalid for {target_model_name}. Using default: {self.default_quality}"); quality = self.default_quality
        style = params.get("style", self.default_style); supported_styles = model_info.get("styles", [self.default_style])
        if target_model_name == "dall-e-2": style = None
        elif style not in supported_styles: logger.warning(f"Style {style} invalid for {target_model_name}. Using default: {self.default_style}"); style = self.default_style

        logger.info(f"DALL-E Binding '{self.binding_instance_name}': Generating image with model '{target_model_name}', size={size}, quality='{quality}', style='{style}', n={n}...")

        try:
            # --- API Call ---
            api_call_task = self.client.images.generate( model=target_model_name, prompt=prompt, n=n, size=size, quality=quality, style=style, response_format="b64_json") # type: ignore
            timeout_seconds = 180
            response = await asyncio.wait_for(api_call_task, timeout=timeout_seconds)

            if not response.data or not all(img.b64_json for img in response.data): raise RuntimeError("Invalid response received from DALL-E API")

            # --- Process Response (remains the same) ---
            output_list = []
            for image_data in response.data:
                if not image_data.b64_json: continue
                image_base64 = image_data.b64_json
                image_metadata = {
                        "prompt_used": prompt, "revised_prompt": image_data.revised_prompt, "model": target_model_name,
                        "size": size, "quality": quality, "style": style, "binding_instance": self.binding_instance_name,
                }
                if pillow_installed and Image and BytesIO:
                    try: img_bytes = base64.b64decode(image_base64); img = Image.open(BytesIO(img_bytes)); logger.debug(f" -> Validated image ({img.format}, {img.size})."); image_metadata["image_format"] = img.format; image_metadata["image_size_pixels"] = img.size; img.close()
                    except Exception as img_err: logger.warning(f"Could not validate generated base64 data as image: {img_err}")
                output_list.append({ "type": "image", "data": image_base64, "mime_type": "image/png", "metadata": image_metadata })

            logger.info(f"DALL-E Binding '{self.binding_instance_name}': Generated {len(output_list)} image(s) successfully.")
            return output_list # Return the list directly

        except BadRequestError as e: logger.error(f"DALL-E '{self.binding_instance_name}': Bad Request Error (e.g., content policy): {e}"); raise ValueError(f"DALL-E Request Error: {e}") from e
        except asyncio.TimeoutError: logger.error(f"DALL-E '{self.binding_instance_name}': Generation timed out after {timeout_seconds}s."); raise RuntimeError("DALL-E API call timed out.")
        except OpenAIError as e: logger.error(f"DALL-E '{self.binding_instance_name}': API Error during generation: {e}"); raise RuntimeError(f"DALL-E API Error: {e}") from e
        except Exception as e: logger.error(f"DALL-E '{self.binding_instance_name}': Unexpected error during generation: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Simulates streaming for DALL-E image generation."""
        # (Implementation remains the same)
        logger.info(f"DALL-E Binding '{self.binding_instance_name}': Simulating stream for image generation.")
        target_model = self.model_name or self.default_model_name or "dall-e-unknown"
        yield {"type": "info", "content": {"status": "starting_image_generation", "model": target_model, "prompt": prompt}}
        try:
            image_result_list = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
            if isinstance(image_result_list, list):
                final_metadata = {"status": "success"};
                if image_result_list: final_metadata.update(image_result_list[0].get("metadata", {}))
                yield {"type": "final", "content": image_result_list, "metadata": final_metadata}
                logger.info(f"DALL-E '{self.binding_instance_name}': Simulated stream finished successfully.")
            else: raise TypeError(f"generate() returned unexpected type: {type(image_result_list)}")
        except (ValueError, RuntimeError, OpenAIError, Exception) as e:
            logger.error(f"DALL-E '{self.binding_instance_name}': Error during simulated stream's generate call: {e}", exc_info=True)
            error_content = f"Image generation failed: {str(e)}"
            yield {"type": "error", "content": error_content}
            yield {"type": "final", "content": [{"type": "error", "data": error_content}], "metadata": {"status": "failed"}}

    # --- Tokenizer / Info Methods (Not Applicable for DALL-E) ---
    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        """Tokenization not applicable for DALL-E."""
        raise NotImplementedError("DALL-E binding does not support tokenization.")

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        """Detokenization not applicable for DALL-E."""
        raise NotImplementedError("DALL-E binding does not support detokenization.")

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns information about a specific DALL-E model or the instance's active/default model.
        """
        target_model = model_name

        if target_model is None:
            # If no specific model requested, use the currently active one for the instance
            if self._model_loaded and self.model_name:
                target_model = self.model_name
                logger.debug(f"Getting info for currently active model: {target_model}")
            # If none is active, use the instance's configured default
            elif self.default_model_name:
                target_model = self.default_model_name
                logger.debug(f"No model active, getting info for instance default: {target_model}")
            else:
                logger.warning(f"DALL-E instance '{self.binding_instance_name}': Cannot get model info - no model specified, active, or configured as default.")
                return { "binding_instance_name": self.binding_instance_name, "model_name": None, "error": "No active or default model set for instance.", "details": {} }

        # Check if the target model is known
        if target_model not in DALLE_MODELS_INFO:
            logger.error(f"DALL-E model '{target_model}' is not recognized by this binding.")
            return { "binding_instance_name": self.binding_instance_name, "model_name": target_model, "error": "Model name not recognized.", "details": {} }

        # Get details from the constant
        model_details = DALLE_MODELS_INFO[target_model]

        # Populate the standardized response structure
        info = {
            "binding_instance_name": self.binding_instance_name,
            "model_name": target_model,
            "model_type": 'tti', # DALL-E is primarily Text-to-Image
            "context_size": None, # Not applicable
            "max_output_tokens": None, # Not applicable
            "supports_vision": True, # It generates images
            "supports_audio": False,
            "supports_streaming": False, # Based on binding card
            "details": {
                "info": f"Details for DALL-E model '{target_model}' via instance '{self.binding_instance_name}'.",
                "supported_sizes": model_details.get("sizes"),
                "supported_qualities": model_details.get("qualities"),
                "supported_styles": model_details.get("styles"),
                "max_images_per_request": model_details.get("max_n")
            }
        }
        return info