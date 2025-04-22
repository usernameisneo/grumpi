# zoos/bindings/dalle_binding.py
import asyncio
import logging
import os
import base64
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List

try:
    from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, NotFoundError, BadRequestError
    from PIL import Image # Optional: for potentially validating base64 data
    from io import BytesIO # Optional: for working with image bytes
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("DALL-E binding requires 'openai' and 'pillow' packages. Install them using 'pip install openai pillow'")
    raise ImportError("DALL-E binding requires 'openai' and 'pillow' packages.")


from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
# We don't directly use StreamChunk here

logger = logging.getLogger(__name__)

# Known DALL-E models and their typical supported sizes
DALLE_MODELS = {
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
    binding_type_name = "dalle_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)

        # --- Configuration Loading ---
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url") # Optional for Azure etc.
        # Fallback: Check environment variables
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                logger.info(f"DALL-E Binding '{self.binding_name}': Loaded API key from OPENAI_API_KEY environment variable.")

        # --- Validation and Client Initialization ---
        if not self.api_key:
             logger.warning(f"DALL-E Binding '{self.binding_name}': API key not found in config.toml or environment variables. Health check and generation will likely fail.")
             self.client = AsyncOpenAI(api_key=None, base_url=self.base_url)
        else:
             self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
             logger.info(f"DALL-E Binding '{self.binding_name}': Initialized client (Base URL: {self.base_url or 'Default OpenAI'}).")

        self.model_name: Optional[str] = None # Will be set by load_model

        # Default generation parameters from config
        self.default_model = self.config.get("model", "dall-e-3")
        self.default_size = self.config.get("default_size", "1024x1024")
        self.default_quality = self.config.get("default_quality", "standard")
        self.default_style = self.config.get("default_style", "vivid") # Only applies to DALL-E 3

    # --- IMPLEMENTED CAPABILITIES ---
    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types (text, potentially image)."""
        modalities = ['text']
        # Check vision support based on the *currently loaded* model
        if self.model_supports_vision: modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['image'] # Gemini chat models output text
    # --- END IMPLEMENTED CAPABILITIES ---

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the DALL-E binding."""
        return {
            "type_name": cls.binding_type_name,
            "version": "1.0",
            "description": "Binding for OpenAI DALL-E 2 and DALL-E 3 text-to-image generation.",
            "supports_streaming": False, # Image generation is not streamed
            "requirements": ["openai>=1.0.0", "pillow>=9.0.0"],
            "config_template": {
                "type": {"type": "string", "value": cls.binding_type_name, "description":"Binding type", "required":True},
                "api_key": {"type": "string", "value": "YOUR_OPENAI_API_KEY_HERE", "description":"OpenAI API Key. Can be omitted if OPENAI_API_KEY env var is set.", "required":False},
                "base_url": {"type": "string", "value": None, "description":"Optional OpenAI-compatible base URL.", "required":False},
                "model": {"type": "string", "value": "dall-e-3", "description":"Default DALL-E model to use ('dall-e-3' or 'dall-e-2').", "required":False},
                "default_size": {"type": "string", "value": "1024x1024", "description":"Default image size (e.g., '1024x1024', '1792x1024'). Check model compatibility.", "required":False},
                "default_quality": {"type": "string", "value": "standard", "description":"Default image quality ('standard' or 'hd' for DALL-E 3).", "required":False},
                "default_style": {"type": "string", "value": "vivid", "description":"Default image style ('vivid' or 'natural' for DALL-E 3).", "required":False},
            }
        }

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity by trying to list models."""
        if not self.api_key:
            return False, "API key is not configured."
        try:
            logger.info(f"DALL-E Binding '{self.binding_name}': Performing health check (listing models)...")
            await self.client.models.list() # Simple check to verify connection and key
            logger.info(f"DALL-E Binding '{self.binding_name}': Health check successful.")
            return True, "Connection successful."
        except APIConnectionError as e:
             logger.error(f"DALL-E Binding '{self.binding_name}': Health check failed (Connection Error): {e}")
             return False, f"API Connection Error: {e}"
        except RateLimitError as e:
             logger.warning(f"DALL-E Binding '{self.binding_name}': Health check hit rate limit: {e}")
             return True, f"Rate limit hit during check, but connection seems ok: {e}"
        except OpenAIError as e:
             logger.error(f"DALL-E Binding '{self.binding_name}': Health check failed (OpenAI Error): {e}")
             if hasattr(e, 'status_code') and e.status_code == 401:
                  return False, f"Authentication Error: Invalid API Key? ({e})"
             return False, f"OpenAI API Error: {e}"
        except Exception as e:
             logger.error(f"DALL-E Binding '{self.binding_name}': Health check failed (Unexpected Error): {e}", exc_info=True)
             return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """DALL-E binding does not use local GPU resources."""
        return {"gpu_required": False}

    async def load_model(self, model_name: str) -> bool:
        """Sets the DALL-E model version to use."""
        async with self._load_lock:
            logger.info(f"DALL-E Binding '{self.binding_name}': Setting active model to '{model_name}'.")
            if model_name not in DALLE_MODELS:
                 logger.error(f"DALL-E Binding '{self.binding_name}': Unsupported model name '{model_name}'. Supported: {list(DALLE_MODELS.keys())}")
                 return False

            # No actual loading, just set the name
            self.model_name = model_name
            self._model_loaded = True
            return True

    async def unload_model(self) -> bool:
        """Unsets the active DALL-E model."""
        async with self._load_lock:
            if not self._model_loaded:
                 return True
            logger.info(f"DALL-E Binding '{self.binding_name}': Unsetting active model '{self.model_name}'.")
            self.model_name = None
            self._model_loaded = False
            return True

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns a list of known DALL-E models supported by this binding."""
        logger.info(f"DALL-E Binding '{self.binding_name}': Listing supported models.")
        model_list = []
        for name, details in DALLE_MODELS.items():
            model_list.append({
                "name": name,
                "details": details, # Include supported sizes, qualities etc.
                # Add None for the new fields as they don't apply
                "context_size": None,
                "max_output_tokens": None,
            })
        return model_list

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any], # Contains personality, generation_type ('tti') etc.
        multimodal_data: Optional[List['InputData']] = None # Use forward reference for type hint
    ) -> Union[str, Dict[str, Any]]:
        """Generates an image using the DALL-E API."""
        if not self._model_loaded or not self.model_name:
            logger.error(f"DALL-E Binding '{self.binding_name}': Cannot generate, model not selected via load_model.")
            raise RuntimeError(f"DALL-E model not selected in binding '{self.binding_name}'.")
        if not self.api_key:
            logger.error(f"DALL-E Binding '{self.binding_name}': Cannot generate, API key not configured.")
            raise RuntimeError("API key not configured for DALL-E binding.")
        if request_info.get("generation_type") != 'tti':
             logger.warning(f"DALL-E Binding '{self.binding_name}': Received non-TTI request type '{request_info.get('generation_type')}'. Ignoring.")
             # Or raise error? Let's ignore for now, TTI is implied.

        model_info = DALLE_MODELS.get(self.model_name, {})

        # --- Determine Parameters ---
        n = params.get("n", 1)
        max_n = model_info.get("max_n", 1)
        if n > max_n:
             logger.warning(f"Requested n={n} images, but model {self.model_name} supports max {max_n}. Clamping to {max_n}.")
             n = max_n

        size = params.get("size", self.default_size)
        if size not in model_info.get("sizes", [self.default_size]):
             logger.warning(f"Requested size {size} not supported by {self.model_name}. Supported: {model_info.get('sizes')}. Using default: {self.default_size}")
             size = self.default_size # Fallback to default or first supported? Default for now.

        quality = params.get("quality", self.default_quality)
        if quality not in model_info.get("qualities", [self.default_quality]):
             logger.warning(f"Requested quality {quality} not supported by {self.model_name}. Supported: {model_info.get('qualities')}. Using default: {self.default_quality}")
             quality = self.default_quality

        style = params.get("style", self.default_style)
        if self.model_name == "dall-e-2": # Style only applies to DALL-E 3
             style = None
        elif style not in model_info.get("styles", [self.default_style]):
             logger.warning(f"Requested style {style} not supported by {self.model_name}. Supported: {model_info.get('styles')}. Using default: {self.default_style}")
             style = self.default_style

        logger.info(f"DALL-E Binding '{self.binding_name}': Generating image with model '{self.model_name}', size={size}, quality='{quality}', style='{style}', n={n}...")

        try:
            response = await self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                style=style,
                response_format="b64_json" # Force base64 response
            )

            if not response.data or not response.data[0].b64_json:
                 logger.error(f"DALL-E Binding '{self.binding_name}': API response did not contain base64 image data.")
                 raise RuntimeError("Invalid response received from DALL-E API")

            # Extract base64 data - return only the first image for simplicity now
            image_base64 = response.data[0].b64_json
            revised_prompt = response.data[0].revised_prompt # DALL-E 3 might revise prompts

            # Optional: Validate if it's valid image data using Pillow
            try:
                img_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(img_data))
                logger.info(f"DALL-E Binding '{self.binding_name}': Image generated successfully ({img.format}, {img.size}). Revised prompt: {revised_prompt}")
            except Exception as img_err:
                 logger.warning(f"DALL-E Binding '{self.binding_name}': Could not validate generated base64 data as image: {img_err}")
                 # Proceed anyway, return the data received

            # Return the required format
            return {
                "image_base64": image_base64,
                "prompt_used": prompt,
                "revised_prompt": revised_prompt,
                "model": self.model_name,
                "size": size,
                "quality": quality,
                "style": style,
                # Add other metadata if needed
            }

        except BadRequestError as e: # Catch specific errors like disallowed prompts
             logger.error(f"DALL-E Binding '{self.binding_name}': Bad Request Error during generation (e.g., content policy): {e}")
             raise ValueError(f"DALL-E Request Error: {e}") from e # Raise ValueError for client error
        except OpenAIError as e:
             logger.error(f"DALL-E Binding '{self.binding_name}': API Error during generation: {e}")
             raise RuntimeError(f"DALL-E API Error: {e}") from e
        except Exception as e:
             logger.error(f"DALL-E Binding '{self.binding_name}': Unexpected error during generation: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error: {e}") from e


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming is not supported for DALL-E image generation."""
        logger.error(f"DALL-E Binding '{self.binding_name}': generate_stream called, but not supported.")
        raise NotImplementedError("DALL-E binding does not support streaming generation.")
        # Need this yield to make it a generator, even though it shouldn't be reached
        yield {} # pragma: no cover