# encoding:utf-8
# Project: lollms_server
# File: zoos/bindings/gemini_binding/__init__.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Binding implementation for Google's Gemini API.

import asyncio
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from io import BytesIO

# Use pipmaster if needed
try:
    import pipmaster as pm
    pm.install_if_missing("google-generativeai")
    pm.install_if_missing("pillow")
except ImportError:
    pass

try:
    import google.generativeai as genai
    import google.api_core.exceptions
    from google.generativeai.types import (
        GenerationConfig,
        ContentDict,
        PartDict,
        Model as GeminiModelType,
        HarmCategory,
        HarmBlockThreshold,
    )
    from PIL import Image
    gemini_installed = True
except ImportError:
    genai = None
    google = None
    Image = None
    BytesIO = None  # type: ignore
    # Mock types for environments where imports failed
    class MockGenConfig: pass
    GenerationConfig = MockGenConfig  # type: ignore
    class MockGeminiModel: pass
    GeminiModelType = MockGeminiModel  # type: ignore
    class MockHarmCategory: pass
    HarmCategory = MockHarmCategory # type: ignore
    class MockHarmBlockThreshold: pass
    HarmBlockThreshold = MockHarmBlockThreshold # type: ignore
    gemini_installed = False

try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass  # type: ignore
    def trace_exception(e): logging.exception(e)

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.utils.helpers import parse_thought_tags # --- ADDED HELPER IMPORT ---

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        class StreamChunk: pass
        class InputData: pass  # type: ignore

logger = logging.getLogger(__name__)

# Constants
SAFETY_CATEGORIES = {
    "HARM_CATEGORY_HARASSMENT": "Harassment",
    "HARM_CATEGORY_HATE_SPEECH": "Hate Speech",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "Sexually Explicit",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "Dangerous Content"
}
SAFETY_OPTIONS = [
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE"
]
DEFAULT_SAFETY_SETTING = "BLOCK_MEDIUM_AND_ABOVE"


class GeminiBinding(Binding):
    """Binding for Google's Gemini API."""
    binding_type_name = "gemini_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the GeminiBinding."""
        super().__init__(config, resource_manager)

        if not gemini_installed or not genai:
            raise ImportError("Gemini binding requires 'google-generativeai' (>=0.4.0) and 'pillow'.")

        self.api_key = self.config.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            logger.info(f"Gemini Binding '{self.binding_instance_name}': Loaded API key.")

        self.auto_detect_limits = self.config.get("auto_detect_limits", True)
        self.manual_ctx_size = self.config.get("ctx_size", 30720)
        self.manual_max_output_tokens = self.config.get("max_output_tokens", 2048)

        self.safety_settings_config: Dict[str, str] = {}
        for cat_key in SAFETY_CATEGORIES.keys():
            setting_key_in_config = f"safety_setting_{cat_key.split('_')[-1].lower()}"
            self.safety_settings_config[cat_key] = self.config.get(setting_key_in_config, DEFAULT_SAFETY_SETTING)
        logger.debug(f"Gemini '{self.binding_instance_name}': Safety settings configured: {self.safety_settings_config}")

        self.genai = genai
        self.model: Optional[genai.GenerativeModel] = None
        self.current_ctx_size: int = self.manual_ctx_size
        self.current_max_output_tokens: int = self.manual_max_output_tokens
        self.model_supports_vision: bool = False

        if self.api_key:
            try:
                self.genai.configure(api_key=self.api_key)
                logger.info(f"Gemini Binding '{self.binding_instance_name}': Configured Google API key.")
            except Exception as e:
                logger.error(f"Gemini Binding '{self.binding_instance_name}': Failed configure genai library: {e}", exc_info=True)
        else:
            logger.warning(f"Gemini Binding '{self.binding_instance_name}': API key not found. API calls will fail.")

    def _parse_gemini_details(self, model: GeminiModelType) -> Dict[str, Any]:
        """Parses the raw Gemini model object attributes into a standardized dictionary."""
        model_id = getattr(model, 'name', 'unknown').split('/')[-1]
        parsed = {'name': model_id}
        parsed['size'] = None  # API doesn't provide this directly
        parsed['modified_at'] = None # API doesn't provide this
        parsed['quantization_level'] = None # Not applicable
        parsed['format'] = "api"
        parsed['family'] = "gemini"
        parsed['families'] = ["gemini"]
        parsed['parameter_size'] = None # API doesn't provide this
        parsed['template'] = None # Not applicable
        parsed['license'] = "Commercial API"
        parsed['homepage'] = "https://ai.google.dev/"

        parsed['context_size'] = getattr(model, 'input_token_limit', None)
        parsed['max_output_tokens'] = getattr(model, 'output_token_limit', None)

        supported_methods = getattr(model, 'supported_generation_methods', [])
        display_name = getattr(model, 'display_name', '')
        # Heuristic for vision support: check name, display name, or if it's 1.5 (which implies vision)
        # Also requires 'embedContent' method, though 'generateContent' handles multimodal input
        supports_vision = (
            ('vision' in model_id.lower() or 'Vision' in display_name or "1.5" in model_id.lower())
            # and 'embedContent' in supported_methods # generateContent handles multimodal now
        )
        parsed['supports_vision'] = supports_vision
        parsed['supports_audio'] = False # Currently not supported via generateContent for audio input

        parsed['details'] = {
            "original_name": getattr(model, 'name', None),
            "display_name": display_name or model_id,
            "version": getattr(model, 'version', None),
            "description": getattr(model, 'description', None),
            "supported_generation_methods": supported_methods,
            "temperature": getattr(model, 'temperature', None), # Note: These are often None/default
            "top_p": getattr(model, 'top_p', None),
            "top_k": getattr(model, 'top_k', None),
        }
        return parsed

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available via the Gemini API that support content generation."""
        if not self.api_key:
            logger.warning(f"Gemini '{self.binding_instance_name}': Cannot list models, API key not configured.")
            return []

        logger.info(f"Gemini '{self.binding_instance_name}': Listing models from Google API...")
        available_models = []
        try:
            # Run synchronous SDK call in a thread
            api_models = await asyncio.to_thread(self.genai.list_models)

            for model in api_models:
                # Filter for models that can actually generate content
                if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                    parsed_data = self._parse_gemini_details(model)
                    if parsed_data.get('name'):  # Ensure we have a valid name
                        available_models.append(parsed_data)

            logger.info(f"Gemini '{self.binding_instance_name}': Found {len(available_models)} usable models.")
            return available_models

        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API Error listing models for '{self.binding_instance_name}': {e}")
            raise RuntimeError(f"Gemini API Error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error listing Gemini models for '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error contacting Google API: {e}") from e

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on the loaded model."""
        modalities = ['text']
        if self._model_loaded and self.model_supports_vision:
            modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text']

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity and connection by attempting to list models."""
        if not self.api_key:
            return False, "API key is not configured."

        try:
            # Use list_models as a simple check for API key validity and connectivity
            await asyncio.to_thread(self.genai.list_models)
            return True, "Connection successful."
        except google.api_core.exceptions.PermissionDenied as e:
            logger.error(f"Gemini Health check failed for '{self.binding_instance_name}' (Permission Denied): {e}")
            return False, f"Permission Denied/Invalid API Key: {e}"
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Gemini Health check failed for '{self.binding_instance_name}' (API Error): {e}")
            return False, f"Google API Error: {e}"
        except Exception as e:
            logger.error(f"Gemini Health check failed for '{self.binding_instance_name}' (Unexpected Error): {e}", exc_info=True)
            return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Gemini is a remote API, so no local GPU is required."""
        return {"gpu_required": False}

    async def load_model(self, model_name: str) -> bool:
        """Prepares the binding to use the specified Gemini model."""
        if not self.api_key:
            logger.error(f"Gemini '{self.binding_instance_name}': Cannot load model, API key missing.")
            return False

        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                logger.info(f"Gemini '{self.binding_instance_name}': Model '{model_name}' already active.")
                return True

            if self._model_loaded:
                await self.unload_model() # Ensure clean state

            logger.info(f"Gemini '{self.binding_instance_name}': Preparing model '{model_name}'.")
            try:
                # Verify model exists and get its details
                model_info: GeminiModelType = await asyncio.to_thread(self.genai.get_model, f"models/{model_name}")
                logger.info(f"Verified model '{model_name}' exists via API.")

                # Determine context size and max output tokens
                detected_ctx = getattr(model_info, 'input_token_limit', None)
                detected_max_out = getattr(model_info, 'output_token_limit', None)

                if self.auto_detect_limits and detected_ctx and detected_max_out:
                    self.current_ctx_size = detected_ctx
                    self.current_max_output_tokens = detected_max_out
                    logger.info(f" -> Auto-detected limits: Context={detected_ctx}, Max Output={detected_max_out}")
                else:
                    if self.auto_detect_limits:
                        logger.warning(f" -> Failed to auto-detect limits for '{model_name}'. Using manual values.")
                    else:
                        logger.info(" -> Using manual limits from instance config.")
                    self.current_ctx_size = self.manual_ctx_size
                    self.current_max_output_tokens = self.manual_max_output_tokens
                logger.info(f" -> Effective Limits: Context={self.current_ctx_size}, Max Output={self.current_max_output_tokens}")

                # Determine vision support based on parsed details
                parsed_details = self._parse_gemini_details(model_info)
                self.model_supports_vision = parsed_details.get('supports_vision', False)
                logger.info(f" -> Vision Support: {self.model_supports_vision}")

                # Initialize the GenerativeModel instance
                self.model = await asyncio.to_thread(self.genai.GenerativeModel, model_name)
                self.model_name = model_name
                self._model_loaded = True
                logger.info(f"Gemini Binding '{self.binding_instance_name}': Prepared successfully for model '{model_name}'.")
                return True

            except google.api_core.exceptions.NotFound:
                logger.error(f"Gemini model 'models/{model_name}' not found via API for instance '{self.binding_instance_name}'.")
                self.model = None
                self.model_name = None
                self._model_loaded = False
                self.model_supports_vision = False
                return False
            except google.api_core.exceptions.GoogleAPIError as e:
                logger.error(f"Gemini API error preparing '{model_name}' for instance '{self.binding_instance_name}': {e}")
                self.model = None
                self.model_name = None
                self._model_loaded = False
                self.model_supports_vision = False
                return False
            except Exception as e:
                logger.error(f"Unexpected error preparing Gemini model '{model_name}' for instance '{self.binding_instance_name}': {e}", exc_info=True)
                self.model = None
                self.model_name = None
                self._model_loaded = False
                self.model_supports_vision = False
                return False

    async def unload_model(self) -> bool:
        """Unsets the active Gemini model and resets state."""
        async with self._load_lock:
            if not self._model_loaded:
                return True # Already unloaded

            logger.info(f"Gemini '{self.binding_instance_name}': Unsetting active model '{self.model_name}'.")
            self.model = None
            self.model_name = None
            self._model_loaded = False
            self.model_supports_vision = False
            # Reset limits to manual defaults when unloaded
            self.current_ctx_size = self.manual_ctx_size
            self.current_max_output_tokens = self.manual_max_output_tokens
            return True

    def _prepare_generation_config(self, params: Dict[str, Any]) -> Optional[GenerationConfig]:
        """Helper to build the GenerationConfig object from request parameters."""
        if not self.model:
            logger.error(f"Gemini model not initialized for instance '{self.binding_instance_name}'. Cannot prepare config.")
            return None

        # Determine max_output_tokens, respecting model limits
        requested_max_tokens = params.get("max_tokens")
        effective_max_tokens = self.current_max_output_tokens # Start with the model's limit (or manual override)

        if requested_max_tokens is not None:
            try:
                n_predict_int = int(requested_max_tokens)
                if 0 < n_predict_int <= self.current_max_output_tokens:
                    effective_max_tokens = n_predict_int
                    logger.debug(f"Using requested max_tokens: {effective_max_tokens}")
                elif n_predict_int > self.current_max_output_tokens:
                    logger.warning(
                        f"Requested max_tokens ({n_predict_int}) exceeds model limit "
                        f"({self.current_max_output_tokens}). Capping."
                    )
                    effective_max_tokens = self.current_max_output_tokens
                else: # n_predict_int <= 0
                    logger.warning(
                        f"Requested max_tokens ({n_predict_int}) is invalid. "
                        f"Using model default/limit: {effective_max_tokens}"
                    )
            except ValueError:
                logger.warning(
                    f"Invalid max_tokens value: {requested_max_tokens}. "
                    f"Using model default/limit: {effective_max_tokens}"
                )
        else:
            logger.debug(f"No max_tokens requested. Using model default/limit: {effective_max_tokens}")

        # Build config dictionary
        gen_config_params: Dict[str, Any] = {
            'candidate_count': 1, # We only support one candidate for now
            'max_output_tokens': effective_max_tokens,
        }
        if "temperature" in params:
            gen_config_params['temperature'] = float(params['temperature'])
        if "top_p" in params:
            gen_config_params['top_p'] = float(params['top_p'])
        if "top_k" in params:
            gen_config_params['top_k'] = int(params['top_k'])

        # Handle stop sequences (up to 5 allowed by Gemini API)
        stop = params.get("stop_sequences") or params.get("stop")
        if stop:
            valid_stop = []
            if isinstance(stop, str):
                valid_stop = [s.strip() for s in stop.split(',') if s.strip()]
            elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
                valid_stop = [s for s in stop if s] # Filter out empty strings
            else:
                logger.warning(f"Invalid stop_sequences format received: {stop}. Ignoring.")

            if valid_stop:
                if len(valid_stop) > 5:
                    logger.warning("Gemini supports max 5 stop sequences. Truncating.")
                    valid_stop = valid_stop[:5]
                gen_config_params['stop_sequences'] = valid_stop
                logger.debug(f"Using stop sequences: {gen_config_params['stop_sequences']}")

        try:
            return GenerationConfig(**gen_config_params)
        except Exception as e:
            logger.error(f"Failed to create Gemini GenerationConfig: {e}", exc_info=True)
            return None

    async def _process_content(
        self, prompt: str, multimodal_data: Optional[List['InputData']]
    ) -> Union[str, List[Union[str, Image.Image]]]:
        """Prepares the content payload (text and/or images) for the Gemini API."""
        content_parts: List[Union[str, Image.Image]] = []
        loaded_pil_images: List[Image.Image] = [] # Keep track to close them later

        if prompt:
            content_parts.append(prompt)

        image_items = [item for item in (multimodal_data or []) if item.type == 'image']

        if not image_items:
            # If no images, return just the prompt string if it exists, or empty string
            return content_parts[0] if content_parts else ""

        # Handle images if present
        if not self.model_supports_vision:
            logger.warning(
                f"Gemini '{self.binding_instance_name}': Image data provided, "
                f"but model '{self.model_name}' does not support vision. Ignoring images."
            )
            return content_parts[0] if content_parts else ""

        if not Image or not BytesIO:
            logger.error("Pillow library not found, cannot process images for Gemini vision model.")
            return content_parts[0] if content_parts else ""

        successful_image_loads = 0
        logger.info(f"Gemini '{self.binding_instance_name}': Processing {len(image_items)} image item(s)...")

        for i, item in enumerate(image_items):
            img = None
            if not item.data or not isinstance(item.data, str):
                logger.warning(f"Skipping image item {i+1} (role: {item.role}): Missing or invalid data.")
                continue

            try:
                # Basic check if data looks like base64 (heuristic)
                # Assumes typical base64 padding or length characteristics
                if len(item.data) > 10: # and '=' not in item.data[-4:]: # Removed padding check, can be absent
                    image_bytes = base64.b64decode(item.data)
                    img = Image.open(BytesIO(image_bytes))
                    # IMPORTANT: Keep the PIL Image object, the SDK handles conversion
                    loaded_pil_images.append(img)
                    content_parts.append(img) # Add the PIL Image directly to parts
                    successful_image_loads += 1
                    logger.debug(f" -> Successfully loaded image {i+1} (role: {item.role})")
                else:
                    logger.warning(f"Skipping image item {i+1} (role: {item.role}): Data doesn't look like base64.")
                    continue
            except Exception as e:
                logger.error(f"Gemini '{self.binding_instance_name}': Failed load/process image index {i}: {e}", exc_info=True)
                # Attempt to close the image if it was opened but failed before appending
                if img and hasattr(img, 'close'):
                    try: img.close()
                    except Exception: pass # Ignore close errors

        if successful_image_loads == 0 and image_items:
            logger.error("All provided image data failed to load. Falling back to text-only prompt.")
            # Ensure any partially loaded images are closed (should be handled above, but belt-and-suspenders)
            for img_obj in loaded_pil_images:
                if hasattr(img_obj, 'close'):
                    try: img_obj.close()
                    except Exception: pass
            return content_parts[0] if content_parts and isinstance(content_parts[0], str) else "" # Return only text part
        elif successful_image_loads < len(image_items):
            logger.warning("Some provided image data failed to load.")

        # Return the list containing text and PIL Image objects
        return content_parts

    def _prepare_safety_settings(self) -> Dict[Any, Any]:
        """Prepares the safety settings dictionary for the API call."""
        api_safety_settings = {}
        if self.genai and hasattr(self.genai, 'types'): # Check if types module exists
            for cat_key, block_level_str in self.safety_settings_config.items():
                try:
                    # Map string keys/values to Gemini SDK enums
                    cat_enum = getattr(HarmCategory, cat_key)
                    block_enum = getattr(HarmBlockThreshold, block_level_str)
                    api_safety_settings[cat_enum] = block_enum
                except AttributeError:
                    logger.warning(f"Could not map safety setting: {cat_key} = {block_level_str}")
        else:
            logger.warning("Could not prepare safety settings: genai.types not available.")
        return api_safety_settings

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Return List[OutputData]-like structure
        """Generates text using the Gemini API (non-streaming)."""
        if not self._model_loaded or not self.model_name or not self.model:
            raise RuntimeError(f"Model not loaded for Gemini instance '{self.binding_instance_name}'.")
        if not self.api_key:
            raise RuntimeError(f"API key not configured for Gemini instance '{self.binding_instance_name}'.")

        logger.info(f"Gemini '{self.binding_instance_name}': Generating non-stream with '{self.model_name}'...")

        generation_config = self._prepare_generation_config(params)
        if generation_config is None:
            raise RuntimeError("Failed to prepare Gemini generation configuration.")

        api_safety_settings = self._prepare_safety_settings()

        # Prepare content payload (handles text and images)
        content_payload = await self._process_content(prompt, multimodal_data)
        # Keep track of PIL images to close them later
        loaded_pil_images = [part for part in content_payload if isinstance(part, Image.Image)] \
                            if isinstance(content_payload, list) else []

        try:
            # Make the API call (synchronous SDK call in thread)
            response = await asyncio.to_thread(
                self.model.generate_content,
                contents=content_payload,
                generation_config=generation_config,
                stream=False,
                safety_settings=api_safety_settings
            )

            # --- Safety Checks ---
            # 1. Prompt Feedback Block
            if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                reason = response.prompt_feedback.block_reason.name
                logger.error(f"Gemini generation blocked by prompt safety settings for '{self.binding_instance_name}': {reason}")
                raise ValueError(f"Request blocked due to prompt safety ({reason}). Adjust safety settings or prompt.")

            # 2. Check for Candidates and Content Parts
            if not response.candidates or not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts'):
                finish_reason = getattr(response.candidates[0].finish_reason, 'name', 'UNKNOWN') if response.candidates else "NO_CANDIDATES"
                logger.error(f"Gemini generation failed for '{self.binding_instance_name}': No valid content. Finish Reason: {finish_reason}")

                # 2a. Check for Safety Block in Response Candidate
                if finish_reason == "SAFETY" and response.candidates:
                    blocked_rating = next((r for r in getattr(response.candidates[0], 'safety_ratings', []) if r.blocked), None)
                    details = ""
                    if blocked_rating:
                         details = (
                            f" (Category: {getattr(blocked_rating.category, 'name', 'N/A')}, "
                            f"Threshold: {getattr(blocked_rating.probability, 'name', 'N/A')})"
                         )
                    raise ValueError(f"Response blocked due to content safety{details}. Adjust safety settings or prompt.")
                else:
                    raise RuntimeError(f"Generation failed: No content received. Finish Reason: {finish_reason}")

            # --- Extract Content ---
            # Concatenate text parts from the first candidate
            raw_completion = "".join(
                part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')
            )

            # --- Parse Thoughts ---
            cleaned_completion, thoughts = parse_thought_tags(raw_completion)
            # ---------------------

            logger.info(f"Gemini generation successful for '{self.binding_instance_name}'.")

            # Prepare metadata
            output_metadata = {
                "model_used": self.model_name,
                "binding_instance": self.binding_instance_name,
                "finish_reason": getattr(response.candidates[0].finish_reason, 'name', None),
                "usage": None # Usage data often comes with streaming or in separate metadata
            }
            if hasattr(response, 'usage_metadata'):
                 usage = response.usage_metadata
                 output_metadata["usage"] = {
                     "prompt_token_count": getattr(usage, 'prompt_token_count', None),
                     "candidates_token_count": getattr(usage, 'candidates_token_count', None),
                     "total_token_count": getattr(usage, 'total_token_count', None)
                 }


            # Return standardized list format
            return [{
                "type": "text",
                "data": cleaned_completion.strip(),
                "thoughts": thoughts,
                "metadata": output_metadata
            }]

        except (google.api_core.exceptions.PermissionDenied,
                google.api_core.exceptions.ResourceExhausted,
                google.api_core.exceptions.InvalidArgument) as e:
            logger.error(f"Gemini API Error for '{self.binding_instance_name}': {e}")
            if isinstance(e, google.api_core.exceptions.InvalidArgument) and "API key not valid" in str(e):
                raise ValueError(f"Invalid Google API Key provided for '{self.binding_instance_name}'.") from e
            raise RuntimeError(f"Gemini API Error: {e}") from e
        except ValueError as e: # Re-raise safety block errors
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during Gemini generation for '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Gemini error: {e}") from e
        finally:
            # Ensure PIL images are closed after use
            if loaded_pil_images:
                logger.debug(f"Closing {len(loaded_pil_images)} PIL images for '{self.binding_instance_name}'.")
                for img in loaded_pil_images:
                    if hasattr(img, 'close'):
                        try: img.close()
                        except Exception: pass # Ignore close errors

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Generates text using the Gemini API (streaming)."""
        if not self._model_loaded or not self.model_name or not self.model:
            yield {"type": "error", "content": f"Model not loaded for Gemini instance '{self.binding_instance_name}'."}
            return
        if not self.api_key:
            yield {"type": "error", "content": f"API key not configured for Gemini instance '{self.binding_instance_name}'."}
            return

        logger.info(f"Gemini '{self.binding_instance_name}': Generating stream with '{self.model_name}'...")

        generation_config = self._prepare_generation_config(params)
        if generation_config is None:
            yield {"type": "error", "content": "Failed to prepare Gemini generation configuration."}
            return

        api_safety_settings = self._prepare_safety_settings()

        # Prepare content payload (handles text and images)
        content_payload = await self._process_content(prompt, multimodal_data)
        loaded_pil_images = [part for part in content_payload if isinstance(part, Image.Image)] \
                            if isinstance(content_payload, list) else []

        # State variables for streaming and thought parsing
        full_raw_response_text = ""
        accumulated_thoughts = ""
        is_thinking = False
        stream = None
        block_reason = None
        finish_reason_str = None
        usage_info = None
        final_metadata = {
            "model_used": self.model_name,
            "binding_instance": self.binding_instance_name,
            "usage": None,
            "finish_reason": None
        }

        try:
            # Start the streaming generation (synchronous SDK call in thread)
            stream = await asyncio.to_thread(
                self.model.generate_content,
                contents=content_payload,
                generation_config=generation_config,
                stream=True,
                safety_settings=api_safety_settings
            )

            async for chunk in stream: # Use async for directly if supported, else wrap next()
                # Note: The native SDK stream isn't directly async iterable.
                # A helper might be needed for true async iteration, or stick to next() in thread.
                # Let's assume `stream` yields chunks synchronously and we process them.
                # The `asyncio.to_thread(next, stream, None)` approach is safer if stream blocks.

                # We'll process chunk by chunk as received from the underlying iterator
                # (potentially wrapped in asyncio.to_thread if needed in a real async loop)

                # --- Safety Checks (Prompt Feedback - usually comes first) ---
                if hasattr(chunk, 'prompt_feedback') and getattr(chunk.prompt_feedback, 'block_reason', None):
                    block_reason = chunk.prompt_feedback.block_reason.name
                    logger.error(f"Gemini stream blocked by prompt safety settings for '{self.binding_instance_name}': {block_reason}")
                    yield {"type": "error", "content": f"Blocked (prompt safety): {block_reason}"}
                    break # Stop processing stream

                # --- Check Candidate Data and Response Safety ---
                chunk_finish_reason = None
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        chunk_finish_reason = candidate.finish_reason
                        finish_reason_str = chunk_finish_reason.name # Store the final reason
                        if finish_reason_str not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]:
                            logger.warning(f"Gemini stream potentially stopped by API. Reason: {finish_reason_str}")
                            # Check for SAFETY block specifically
                            if finish_reason_str == "SAFETY":
                                block_reason = finish_reason_str
                                blocked_rating = next((r for r in getattr(candidate, 'safety_ratings', []) if r.blocked), None)
                                details = ""
                                if blocked_rating:
                                    details = (
                                        f" (Category: {getattr(blocked_rating.category, 'name', 'N/A')}, "
                                        f"Threshold: {getattr(blocked_rating.probability, 'name', 'N/A')})"
                                    )
                                error_msg = f"Blocked (response safety){details}."
                                logger.error(f"Gemini stream blocked by response safety for '{self.binding_instance_name}': {error_msg}")
                                yield {"type": "error", "content": error_msg}
                                break # Stop processing stream

                # --- Usage Metadata (often comes at the end, but check each chunk) ---
                if hasattr(chunk, 'usage_metadata'):
                    usage = chunk.usage_metadata
                    usage_info = {
                        "prompt_token_count": getattr(usage, 'prompt_token_count', None),
                        "candidates_token_count": getattr(usage, 'candidates_token_count', None),
                        "total_token_count": getattr(usage, 'total_token_count', None)
                    }
                    final_metadata["usage"] = usage_info

                # --- Extract Text and Parse Thoughts ---
                chunk_raw_content = ""
                try:
                    # Accessing chunk.text directly might raise ValueError if content is blocked/empty
                    if chunk.candidates and hasattr(chunk.candidates[0], 'content') and hasattr(chunk.candidates[0].content, 'parts'):
                         chunk_raw_content = "".join(p.text for p in chunk.candidates[0].content.parts if hasattr(p, 'text'))
                except ValueError:
                    # This can happen if the chunk represents a safety block or has no text
                    logger.debug("ValueError accessing chunk text, possibly safety related.")
                    if block_reason: break # If already blocked, exit
                    continue # Otherwise, might be an empty chunk, continue loop
                except Exception as e:
                    logger.error(f"Error accessing Gemini chunk text: {e}. Chunk: {chunk}")
                    # Decide whether to break or continue based on error type

                if chunk_raw_content:
                    full_raw_response_text += chunk_raw_content
                    current_text_to_process = chunk_raw_content
                    processed_text_chunk = ""
                    processed_thoughts_chunk = None

                    while current_text_to_process:
                        if is_thinking:
                            end_tag_pos = current_text_to_process.find("</think>")
                            if end_tag_pos != -1:
                                # Found end tag: complete the thought
                                thought_part = current_text_to_process[:end_tag_pos]
                                accumulated_thoughts += thought_part
                                processed_thoughts_chunk = accumulated_thoughts # Yield complete thought
                                accumulated_thoughts = "" # Reset accumulator
                                is_thinking = False
                                current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                            else:
                                # End tag not in this chunk part: accumulate and break inner loop
                                accumulated_thoughts += current_text_to_process
                                current_text_to_process = ""
                        else: # Not currently thinking
                            start_tag_pos = current_text_to_process.find("<think>")
                            if start_tag_pos != -1:
                                # Found start tag: yield text before it, start accumulating thought
                                text_part = current_text_to_process[:start_tag_pos]
                                processed_text_chunk += text_part
                                is_thinking = True
                                current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                            else:
                                # No start tag in this chunk part: yield as text and break inner loop
                                processed_text_chunk += current_text_to_process
                                current_text_to_process = ""

                    # Yield the processed parts for this chunk
                    if processed_text_chunk or processed_thoughts_chunk:
                        yield {
                            "type": "chunk",
                            "content": processed_text_chunk if processed_text_chunk else None,
                            "thoughts": processed_thoughts_chunk # Yields None if no thought completed in this chunk
                        }

                # If a block reason emerged during chunk processing, stop
                if block_reason:
                    logger.error(f"Stopping Gemini stream for '{self.binding_instance_name}' due to block: {block_reason}")
                    break

            # --- After the Loop ---
            if not block_reason:
                # Handle any incomplete thought at the end of the stream
                if is_thinking and accumulated_thoughts:
                    logger.warning(
                        f"Gemini stream ended mid-thought for '{self.binding_instance_name}'. "
                        f"Thought content:\n{accumulated_thoughts}"
                    )
                    final_metadata["incomplete_thoughts"] = accumulated_thoughts

                logger.info(f"Gemini stream finished for '{self.binding_instance_name}'.")
                final_metadata["finish_reason"] = finish_reason_str or ("incomplete_thought" if is_thinking else "completed")

                # Re-parse the full text to ensure correct final structure
                final_cleaned_text, final_thoughts_str = parse_thought_tags(full_raw_response_text)

                # Append incomplete thoughts if they exist
                if final_metadata.get("incomplete_thoughts"):
                    incomplete = final_metadata["incomplete_thoughts"]
                    if final_thoughts_str:
                        final_thoughts_str = (
                            final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + incomplete
                        ).strip()
                    else:
                         final_thoughts_str = incomplete

                # Yield the final complete output
                final_output_list = [{
                    "type": "text",
                    "data": final_cleaned_text.strip(),
                    "thoughts": final_thoughts_str,
                    "metadata": final_metadata # Contains usage, finish reason, etc.
                }]
                yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}
            else:
                # Stream was blocked, yield final error status
                error_final_output = [{
                    "type": "error",
                    "data": f"Stream stopped due to safety block ({block_reason}).",
                    "metadata": final_metadata # Include any partial metadata gathered
                }]
                yield {"type": "final", "content": error_final_output, "metadata": {"status": "failed", "reason": block_reason}}

        except (google.api_core.exceptions.PermissionDenied,
                google.api_core.exceptions.ResourceExhausted,
                google.api_core.exceptions.InvalidArgument) as e:
            logger.error(f"Gemini API Error during stream for '{self.binding_instance_name}': {e}")
            yield {"type": "error", "content": f"Gemini API Error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during Gemini stream for '{self.binding_instance_name}': {e}", exc_info=True)
            yield {"type": "error", "content": f"Unexpected Gemini error: {e}"}
        finally:
            # Ensure PIL images are closed after use
            if loaded_pil_images:
                logger.debug(f"Closing {len(loaded_pil_images)} PIL images for '{self.binding_instance_name}'.")
                for img in loaded_pil_images:
                    if hasattr(img, 'close'):
                        try: img.close()
                        except Exception: pass # Ignore close errors

    # --- Tokenizer / Info Methods ---

    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenization to IDs is not directly supported by the Gemini API. Raises NotImplementedError."""
        if not self._model_loaded or not self.model:
            raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for tokenization")

        # Gemini API provides count_tokens, not direct tokenization to IDs.
        logger.warning(
             f"Gemini binding '{self.binding_instance_name}': Tokenization requested. "
             f"While token counting is possible via count_tokens, returning token IDs is not supported."
         )
        # Example of how to count tokens if needed elsewhere:
        # try:
        #     token_count = await asyncio.to_thread(self.model.count_tokens, text)
        #     logger.info(f"Token count for text: {getattr(token_count, 'total_tokens', 'N/A')}")
        # except Exception as e:
        #     logger.error(f"Failed to count tokens: {e}")
        raise NotImplementedError(f"Binding '{self.binding_instance_name}' (Gemini) does not support returning token IDs.")

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenization from IDs is not supported by the Gemini API. Raises NotImplementedError."""
        if not self._model_loaded:
            raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for detokenization")
        raise NotImplementedError(f"Binding '{self.binding_instance_name}' (Gemini) does not support detokenization from token IDs.")

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded Gemini model."""
        if not self._model_loaded or not self.model_name:
            # Return defaults if no model is loaded
            return {
                "name": None,
                "context_size": self.manual_ctx_size,
                "max_output_tokens": self.manual_max_output_tokens,
                "supports_vision": False,
                "supports_audio": False,
                "details": {}
            }

        # Return info based on the loaded model state
        return {
            "name": self.model_name,
            "context_size": self.current_ctx_size,
            "max_output_tokens": self.current_max_output_tokens,
            "supports_vision": self.model_supports_vision,
            "supports_audio": False, # Audio input not supported here
            "details": {
                "info": f"Active model for instance '{self.binding_instance_name}' is '{self.model_name}'.",
                # Could potentially add more details retrieved during load_model if stored
            }
        }