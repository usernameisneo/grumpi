# lollms_server/bindings/gemini_binding/__init__.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Binding implementation for Google's Gemini API.
# Modification Date: 2025-05-04
# Refactored model info endpoint, default model handling, tokenizers, thoughts parsing.

import asyncio
import os
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List

# Use pipmaster if needed
try:
    import pipmaster as pm
    pm.ensure_packages(["google-generativeai>=0.4.0", "pillow>=9.0.0", "tiktoken"])
except ImportError:
    pass # Assume installed or handle import error below

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
        GenerateContentResponse, # Added for type hinting
    )
    from PIL import Image
    import tiktoken # For token approximation
    gemini_installed = True
    pillow_installed = True
    tiktoken_installed = True
except ImportError as e:
    # Mock classes for environments where imports failed but allow basic functionality
    genai = None; google = None; Image = None; BytesIO = None; tiktoken = None # type: ignore
    class MockGenConfig: pass; GenerationConfig = MockGenConfig # type: ignore
    class MockGeminiModel: pass; GeminiModelType = MockGeminiModel # type: ignore
    class MockHarmCategory: pass; HarmCategory = MockHarmCategory # type: ignore
    class MockHarmBlockThreshold: pass; HarmBlockThreshold = MockHarmBlockThreshold # type: ignore
    class MockGenerateContentResponse: pass; GenerateContentResponse = MockGenerateContentResponse # type: ignore
    gemini_installed = False; pillow_installed = False; tiktoken_installed = False
    _import_error_msg = str(e)


try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass  # type: ignore
    def trace_exception(e): logging.exception(e)

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.utils.helpers import parse_thought_tags # Use helper

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        class StreamChunk: pass # type: ignore
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
DEFAULT_FALLBACK_CONTEXT_SIZE = 30720
DEFAULT_FALLBACK_MAX_OUTPUT = 2048


class GeminiBinding(Binding):
    """Binding for Google's Gemini API."""
    binding_type_name = "gemini_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the GeminiBinding."""
        super().__init__(config, resource_manager) # Sets self.config, self.resource_manager, self.binding_instance_name, self.default_model_name

        if not gemini_installed or not genai:
            raise ImportError(f"Gemini binding requires 'google-generativeai' (>=0.4.0). Error: {_import_error_msg}")
        if not pillow_installed:
            logger.warning("Pillow library not found. Image processing disabled for Gemini.")
        if not tiktoken_installed:
            logger.warning("Tiktoken library not found. Tokenization/detokenization will use estimates.")


        self.api_key = self.config.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            logger.info(f"Gemini Binding '{self.binding_instance_name}': Loaded API key.")
        else:
             logger.warning(f"Gemini Binding '{self.binding_instance_name}': API key not found. API calls will fail.")

        # Instance-specific configuration
        self.auto_detect_limits = self.config.get("auto_detect_limits", True)
        # Manual fallback limits from instance config (used if auto-detect fails or is off)
        self.manual_ctx_size = self.config.get("ctx_size", DEFAULT_FALLBACK_CONTEXT_SIZE)
        self.manual_max_output_tokens = self.config.get("max_output_tokens", DEFAULT_FALLBACK_MAX_OUTPUT)
        # Note: self.default_model_name is set by parent __init__ from config['default_model']

        # Safety settings
        self.safety_settings_config: Dict[str, str] = {}
        for cat_key in SAFETY_CATEGORIES.keys():
            setting_key_in_config = f"safety_setting_{cat_key.split('_')[-1].lower()}"
            self.safety_settings_config[cat_key] = self.config.get(setting_key_in_config, DEFAULT_SAFETY_SETTING)
        logger.debug(f"Gemini '{self.binding_instance_name}': Safety settings configured: {self.safety_settings_config}")

        self.genai = genai # Store module reference

        # --- Internal State ---
        self.model: Optional[genai.GenerativeModel] = None # Actual client model object
        # self.model_name stores the currently active model ID string
        self.current_model_details: Dict[str, Any] = {} # Cache for *active* model details
        self.model_supports_vision: bool = False
        self.current_ctx_size: int = self.manual_ctx_size # Effective context size
        self.current_max_output_tokens: int = self.manual_max_output_tokens # Effective max output

        # Configure the library if API key is present
        if self.api_key and self.genai:
            try:
                self.genai.configure(api_key=self.api_key)
                logger.info(f"Gemini Binding '{self.binding_instance_name}': Configured Google API key.")
            except Exception as e:
                logger.error(f"Gemini Binding '{self.binding_instance_name}': Failed configure genai library: {e}", exc_info=True)


    # --- Helper to parse Gemini model details ---
    def _parse_gemini_details(self, model_obj: GeminiModelType) -> Dict[str, Any]:
        """
        Parses the raw Gemini model object attributes into a standardized dictionary
        suitable for GetModelInfoResponse or list_available_models.
        """
        # (Implementation remains largely the same, added model_type and supports_streaming)
        if not model_obj: return {}
        model_id = getattr(model_obj, 'name', 'unknown').split('/')[-1]
        if not model_id: return {}

        parsed = {'name': model_id, 'model_name': model_id} # Add model_name too
        parsed['size'] = None; parsed['modified_at'] = None # Not provided by API
        parsed['quantization_level'] = None; parsed['format'] = "api"
        parsed['family'] = "gemini"; parsed['families'] = ["gemini"]
        parsed['parameter_size'] = None; parsed['template'] = None
        parsed['license'] = "Commercial API"; parsed['homepage'] = "https://ai.google.dev/"

        parsed['context_size'] = getattr(model_obj, 'input_token_limit', None)
        parsed['max_output_tokens'] = getattr(model_obj, 'output_token_limit', None)

        supported_methods = getattr(model_obj, 'supported_generation_methods', [])
        display_name = getattr(model_obj, 'display_name', '')
        # Heuristic for vision support
        supports_vision = ('vision' in model_id.lower() or 'Vision' in display_name or "1.5" in model_id.lower())
        parsed['supports_vision'] = supports_vision
        parsed['supports_audio'] = False # Not supported via this API
        parsed['supports_streaming'] = 'generateContent' in supported_methods # Gemini supports streaming

        # Determine model_type
        if supports_vision: parsed['model_type'] = 'vlm'
        else: parsed['model_type'] = 'ttt'

        # Add extra details
        parsed['details'] = {
            "original_name": getattr(model_obj, 'name', None),
            "display_name": display_name or model_id,
            "version": getattr(model_obj, 'version', None),
            "description": getattr(model_obj, 'description', None),
            "supported_generation_methods": supported_methods,
            "temperature": getattr(model_obj, 'temperature', None),
            "top_p": getattr(model_obj, 'top_p', None),
            "top_k": getattr(model_obj, 'top_k', None),
        }
        return parsed

    # --- Required Binding Methods ---

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available via the Gemini API that support content generation."""
        # (Implementation remains the same)
        if not self.api_key or not self.genai: logger.warning(f"Gemini '{self.binding_instance_name}': Cannot list models, API key/library not configured."); return []
        logger.info(f"Gemini '{self.binding_instance_name}': Listing models from Google API...")
        available_models = []
        try:
            api_models = await asyncio.to_thread(self.genai.list_models)
            for model in api_models:
                if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                    parsed_data = self._parse_gemini_details(model)
                    if parsed_data.get('name'): available_models.append(parsed_data)
            logger.info(f"Gemini '{self.binding_instance_name}': Found {len(available_models)} usable models.")
            return available_models
        except google.api_core.exceptions.GoogleAPIError as e: logger.error(f"Gemini API Error listing models for '{self.binding_instance_name}': {e}"); raise RuntimeError(f"Gemini API Error: {e}") from e
        except Exception as e: logger.error(f"Unexpected error listing Gemini models for '{self.binding_instance_name}': {e}", exc_info=True); raise RuntimeError(f"Unexpected error contacting Google API: {e}") from e

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on the *currently active* model."""
        modalities = ['text']
        if self.model_supports_vision: modalities.append('image') # Use state set by load_model
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text']

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity and connection by attempting to list models."""
        # (Implementation remains the same)
        if not self.api_key or not self.genai: return False, "API key or genai library not configured."
        try:
            models = await asyncio.to_thread(self.genai.list_models)
            return True, f"Connection OK ({len(list(models))} models found)."
        except google.api_core.exceptions.PermissionDenied as e: logger.error(f"Gemini Health check '{self.binding_instance_name}' failed (Perm Denied): {e}"); return False, f"Permission Denied/Invalid API Key: {e}"
        except google.api_core.exceptions.GoogleAPIError as e: logger.error(f"Gemini Health check '{self.binding_instance_name}' failed (API Err): {e}"); return False, f"Google API Error: {e}"
        except Exception as e: logger.error(f"Gemini Health check '{self.binding_instance_name}' failed (Unexpected Err): {e}", exc_info=True); return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Gemini is a remote API, so no local GPU is required."""
        return {"gpu_required": False}

    def _reset_model_state(self):
        """Helper to reset internal model state variables."""
        self.model = None
        self.model_name = None
        self._model_loaded = False
        self.model_supports_vision = False
        self.current_model_details = {}
        # Reset limits to instance manual defaults when unloaded
        self.current_ctx_size = self.manual_ctx_size
        self.current_max_output_tokens = self.manual_max_output_tokens

    async def load_model(self, model_name: str) -> bool:
        """Prepares the binding to use the specified Gemini model and caches its info."""
        if not self.api_key or not self.genai:
            logger.error(f"Gemini '{self.binding_instance_name}': Cannot load model, API key/library missing.")
            return False

        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                logger.info(f"Gemini '{self.binding_instance_name}': Model '{model_name}' already active.")
                return True

            if self._model_loaded: await self.unload_model() # Ensure clean state

            logger.info(f"Gemini '{self.binding_instance_name}': Activating model '{model_name}'. Fetching info...")
            try:
                # --- Fetch Model Info ---
                model_info_obj: GeminiModelType = await asyncio.to_thread(self.genai.get_model, f"models/{model_name}")
                if not model_info_obj: raise ValueError("API returned invalid model info.")
                logger.info(f"Verified model '{model_name}' exists via API.")

                # --- Parse and Store Details ---
                parsed_details = self._parse_gemini_details(model_info_obj)
                self.current_model_details = parsed_details # Cache parsed details
                self.model_supports_vision = parsed_details.get('supports_vision', False)

                # --- Determine Effective Limits ---
                detected_ctx = parsed_details.get('context_size')
                detected_max_out = parsed_details.get('max_output_tokens')
                if self.auto_detect_limits and detected_ctx and detected_max_out:
                    self.current_ctx_size = detected_ctx
                    self.current_max_output_tokens = detected_max_out
                    logger.info(f" -> Auto-detected limits: Context={detected_ctx}, Max Output={detected_max_out}")
                else:
                    if self.auto_detect_limits: logger.warning(f" -> Failed to auto-detect limits for '{model_name}'. Using manual values.")
                    else: logger.info(" -> Using manual limits from instance config.")
                    self.current_ctx_size = self.manual_ctx_size
                    self.current_max_output_tokens = self.manual_max_output_tokens
                logger.info(f" -> Effective Limits: Context={self.current_ctx_size}, Max Output={self.current_max_output_tokens}")
                logger.info(f" -> Vision Support: {self.model_supports_vision}")

                # --- Initialize the SDK Model Object ---
                self.model = await asyncio.to_thread(self.genai.GenerativeModel, model_name)
                self.model_name = model_name # Set active model name
                self._model_loaded = True
                logger.info(f"Gemini Binding '{self.binding_instance_name}': Prepared successfully for model '{model_name}'.")
                return True

            except google.api_core.exceptions.NotFound: logger.error(f"Gemini model 'models/{model_name}' not found via API."); self._reset_model_state(); return False
            except google.api_core.exceptions.GoogleAPIError as e: logger.error(f"Gemini API error activating '{model_name}': {e}"); self._reset_model_state(); return False
            except Exception as e: logger.error(f"Unexpected error activating Gemini model '{model_name}': {e}", exc_info=True); self._reset_model_state(); return False

    async def unload_model(self) -> bool:
        """Unsets the active Gemini model and resets state."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Gemini '{self.binding_instance_name}': Unsetting active model '{self.model_name}'.")
            self._reset_model_state()
            return True

    def _prepare_generation_config(self, params: Dict[str, Any]) -> Optional[GenerationConfig]:
        """Helper to build the GenerationConfig object from request parameters."""
        # (Implementation remains the same)
        if not self.genai: return None # Check if library loaded
        requested_max_tokens = params.get("max_tokens")
        effective_max_tokens = self.current_max_output_tokens # Start with active model's limit
        if requested_max_tokens is not None:
            try:
                n_predict_int = int(requested_max_tokens)
                if 0 < n_predict_int <= self.current_max_output_tokens: effective_max_tokens = n_predict_int
                elif n_predict_int > self.current_max_output_tokens: logger.warning(f"Requested max_tokens ({n_predict_int}) > limit ({self.current_max_output_tokens}). Capping."); effective_max_tokens = self.current_max_output_tokens
                else: logger.warning(f"Invalid max_tokens ({n_predict_int}). Using model limit: {effective_max_tokens}")
            except ValueError: logger.warning(f"Invalid max_tokens: {requested_max_tokens}. Using model limit: {effective_max_tokens}")
        else: logger.debug(f"Using effective max_output_tokens: {effective_max_tokens}")

        gen_config_params: Dict[str, Any] = {'candidate_count': 1, 'max_output_tokens': effective_max_tokens}
        if "temperature" in params: gen_config_params['temperature'] = float(params['temperature'])
        if "top_p" in params: gen_config_params['top_p'] = float(params['top_p'])
        if "top_k" in params: gen_config_params['top_k'] = int(params['top_k'])
        stop = params.get("stop_sequences") or params.get("stop")
        if stop:
            valid_stop = []
            if isinstance(stop, str): valid_stop = [s.strip() for s in stop.split(',') if s.strip()]
            elif isinstance(stop, list) and all(isinstance(s, str) for s in stop): valid_stop = [s for s in stop if s]
            else: logger.warning(f"Invalid stop_sequences format: {stop}. Ignoring.")
            if valid_stop:
                if len(valid_stop) > 5: logger.warning("Gemini supports max 5 stop sequences. Truncating."); valid_stop = valid_stop[:5]
                gen_config_params['stop_sequences'] = valid_stop; logger.debug(f"Using stop sequences: {gen_config_params['stop_sequences']}")
        try: return GenerationConfig(**gen_config_params)
        except Exception as e: logger.error(f"Failed to create Gemini GenerationConfig: {e}", exc_info=True); return None

    async def _process_content( self, prompt: str, multimodal_data: Optional[List['InputData']] ) -> Union[str, List[Union[str, Image.Image]]]:
        """Prepares the content payload (text and/or images) for the Gemini API."""
        # (Implementation remains the same)
        content_parts: List[Union[str, Image.Image]] = []; loaded_pil_images: List[Image.Image] = []
        if prompt: content_parts.append(prompt)
        image_items = [item for item in (multimodal_data or []) if item.type == 'image']
        if not image_items: return content_parts[0] if content_parts else ""
        if not self.model_supports_vision: logger.warning(f"Gemini '{self.binding_instance_name}': Images provided but model '{self.model_name}' doesn't support vision. Ignoring."); return content_parts[0] if content_parts else ""
        if not pillow_installed: logger.error("Pillow library not found, cannot process images."); return content_parts[0] if content_parts else ""

        successful_loads = 0; logger.info(f"Gemini '{self.binding_instance_name}': Processing {len(image_items)} image item(s)...")
        for i, item in enumerate(image_items):
            img = None
            if not item.data or not isinstance(item.data, str): logger.warning(f"Skipping img {i+1}: Missing/invalid data."); continue
            try:
                if len(item.data) > 10: # Basic check
                    image_bytes = base64.b64decode(item.data); img = Image.open(BytesIO(image_bytes))
                    loaded_pil_images.append(img); content_parts.append(img); successful_loads += 1; logger.debug(f" -> Loaded image {i+1} (role: {item.role})")
                else: logger.warning(f"Skipping img {i+1}: Data looks too short for base64."); continue
            except Exception as e:
                 logger.error(f"Gemini '{self.binding_instance_name}': Failed load/process image index {i}: {e}", exc_info=True)
                 if img and hasattr(img, 'close'): 
                    try: 
                        img.close(); 
                    except Exception: 
                        pass
        if successful_loads == 0 and image_items: logger.error("All images failed load. Falling back to text only."); return content_parts[0] if content_parts and isinstance(content_parts[0], str) else ""
        elif successful_loads < len(image_items): logger.warning("Some images failed to load.")
        # Keep PIL images in list, SDK handles them
        # Make sure to close them in the calling function's finally block
        return content_parts

    def _prepare_safety_settings(self) -> Dict[Any, Any]:
        """Prepares the safety settings dictionary for the API call."""
        # (Implementation remains the same)
        api_safety_settings = {};
        if self.genai and hasattr(self.genai, 'types'):
            for cat_key, block_level_str in self.safety_settings_config.items():
                try: cat_enum = getattr(HarmCategory, cat_key); block_enum = getattr(HarmBlockThreshold, block_level_str); api_safety_settings[cat_enum] = block_enum
                except AttributeError: logger.warning(f"Could not map safety setting: {cat_key} = {block_level_str}")
        else: logger.warning("Could not prepare safety settings: genai.types not available.")
        return api_safety_settings

    async def _get_effective_model_name(self) -> Optional[str]:
        """Gets the model name to use, prioritizing loaded, then instance default."""
        # (Implementation remains the same as Ollama version)
        if self._model_loaded and self.model_name:
             return self.model_name
        elif self.default_model_name:
             logger.warning(f"No model explicitly loaded for instance '{self.binding_instance_name}'. Using instance default '{self.default_model_name}'.")
             if await self.load_model(self.default_model_name): return self.default_model_name
             else: logger.error(f"Failed to set instance default model '{self.default_model_name}'."); return None
        else: logger.error(f"No model loaded or configured as default for Gemini instance '{self.binding_instance_name}'."); return None
    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Return List[OutputData]-like structure
        """Generates text using the Gemini API (non-streaming)."""
        if not self.genai: raise RuntimeError("Gemini library not loaded.")
        effective_model_name = await self._get_effective_model_name()
        if not effective_model_name or not self.model: raise RuntimeError(f"Model '{effective_model_name or 'None'}' not available/loaded for Gemini instance '{self.binding_instance_name}'.")
        if not self.api_key: raise RuntimeError(f"API key not configured for Gemini instance '{self.binding_instance_name}'.")

        logger.info(f"Gemini '{self.binding_instance_name}': Generating non-stream with '{effective_model_name}'...")

        generation_config = self._prepare_generation_config(params);
        if generation_config is None: raise RuntimeError("Failed to prepare Gemini generation configuration.")
        api_safety_settings = self._prepare_safety_settings()
        content_payload = await self._process_content(prompt, multimodal_data)
        loaded_pil_images = [p for p in content_payload if isinstance(p, Image.Image)] if isinstance(content_payload, list) else []

        # --- Extract system_message from params ---
        system_message = params.get("system_message", None)
        # -----------------------------------------

        try:
            # --- Pass system_instruction to the API call ---
            logger.debug(f"Gemini non-stream call: model={effective_model_name}, system='{str(system_message)[:50] if system_message else None}...', options={generation_config}")
            response: GenerateContentResponse = await asyncio.to_thread(
                self.model.generate_content,
                contents=f"system:{system_message}\n"+content_payload if system_message and type(content_payload)==str else [f"system:{system_message}\n"]+content_payload if system_message and type(content_payload)==list else content_payload,
                generation_config=generation_config,
                stream=False,
                safety_settings=api_safety_settings,
            )
            # ----------------------------------------------

            # --- Safety Checks ---
            if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                reason = response.prompt_feedback.block_reason.name; logger.error(f"Gemini prompt blocked: {reason}"); raise ValueError(f"Request blocked (prompt safety: {reason}).")
            if not response.candidates or not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts'):
                reason = getattr(response.candidates[0].finish_reason, 'name', 'UNKNOWN') if response.candidates else "NO_CANDIDATES"; logger.error(f"Gemini failed: No content. Finish: {reason}")
                if reason == "SAFETY":
                    rating = next((r for r in getattr(response.candidates[0], 'safety_ratings', []) if r.blocked), None)
                    details = f" (Cat: {getattr(rating.category, 'name', 'N/A')}, Prob: {getattr(rating.probability, 'name', 'N/A')})" if rating else ""
                    raise ValueError(f"Response blocked (content safety{details}).")
                else: raise RuntimeError(f"Generation failed: No content. Finish: {reason}")

            # --- Extract Content & Thoughts ---
            raw_completion = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text'))
            cleaned_completion, thoughts = parse_thought_tags(raw_completion)
            logger.info(f"Gemini generation successful for '{self.binding_instance_name}'.")

            # --- Prepare Metadata ---
            output_metadata = {"model_used": effective_model_name, "binding_instance": self.binding_instance_name, "finish_reason": getattr(response.candidates[0].finish_reason, 'name', None), "usage": None}
            if hasattr(response, 'usage_metadata'):
                 usage = response.usage_metadata; output_metadata["usage"] = {"prompt_token_count": getattr(usage, 'prompt_token_count', None), "candidates_token_count": getattr(usage, 'candidates_token_count', None), "total_token_count": getattr(usage, 'total_token_count', None)}

            return [{"type": "text", "data": cleaned_completion.strip(), "thoughts": thoughts, "metadata": output_metadata}]

        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.InvalidArgument) as e:
            logger.error(f"Gemini API Error '{self.binding_instance_name}': {e}")
            if isinstance(e, google.api_core.exceptions.InvalidArgument) and "API key not valid" in str(e): raise ValueError(f"Invalid Google API Key '{self.binding_instance_name}'.") from e
            raise RuntimeError(f"Gemini API Error: {e}") from e
        except ValueError as e: raise e # Re-raise safety blocks
        except Exception as e: logger.error(f"Unexpected Gemini error '{self.binding_instance_name}': {e}", exc_info=True); raise RuntimeError(f"Unexpected Gemini error: {e}") from e
        finally:
            if loaded_pil_images: logger.debug(f"Closing {len(loaded_pil_images)} PIL images."); [img.close() for img in loaded_pil_images if hasattr(img,'close')]

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Generates text using the Gemini API (streaming)."""
        if not self.genai: yield {"type": "error", "content": "Gemini library not loaded."}; return
        effective_model_name = await self._get_effective_model_name()
        if not effective_model_name or not self.model: yield {"type": "error", "content": f"Model '{effective_model_name or 'None'}' not available/loaded for instance '{self.binding_instance_name}'."}; return
        if not self.api_key: yield {"type": "error", "content": f"API key not configured for instance '{self.binding_instance_name}'."}; return

        logger.info(f"Gemini '{self.binding_instance_name}': Generating stream with '{effective_model_name}'...")
        generation_config = self._prepare_generation_config(params);
        if generation_config is None: yield {"type": "error", "content": "Failed prepare generation config."}; return
        api_safety_settings = self._prepare_safety_settings()
        content_payload = await self._process_content(prompt, multimodal_data)
        loaded_pil_images = [p for p in content_payload if isinstance(p, Image.Image)] if isinstance(content_payload, list) else []

        # --- Extract system_message from params ---
        system_message = params.get("system_message", None)
        # -----------------------------------------

        full_raw_response_text = ""; accumulated_thoughts = ""; is_thinking = False
        block_reason = None; finish_reason_str = None; usage_info = None
        final_metadata = {"model_used": effective_model_name, "binding_instance": self.binding_instance_name, "usage": None, "finish_reason": None}
        stream = None

        try:
            # --- Pass system_instruction to the API call ---
            logger.debug(f"Gemini stream call: model={effective_model_name}, system='{str(system_message)[:50] if system_message else None}...', options={generation_config}")
            stream = await asyncio.to_thread(
                self.model.generate_content,
                contents=f"system:{system_message}\n"+content_payload if system_message and type(content_payload)==str else [f"system:{system_message}\n"]+content_payload if system_message and type(content_payload)==list else content_payload,
                generation_config=generation_config,
                stream=True,
                safety_settings=api_safety_settings,
            )
            # ----------------------------------------------

            # Process stream chunks
            # (Rest of the streaming logic for parsing chunks, thoughts, safety, etc. remains the same)
            async for chunk in stream:
                # ... (Safety checks for prompt_feedback) ...
                if hasattr(chunk, 'prompt_feedback') and getattr(chunk.prompt_feedback, 'block_reason', None):
                    block_reason = chunk.prompt_feedback.block_reason.name; logger.error(f"Gemini stream blocked (prompt safety): {block_reason}"); yield {"type": "error", "content": f"Blocked (prompt safety): {block_reason}"}; break

                # ... (Check Candidate Data and Response Safety) ...
                chunk_finish_reason = None
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        chunk_finish_reason = candidate.finish_reason; finish_reason_str = chunk_finish_reason.name
                        if finish_reason_str == "SAFETY":
                            block_reason = finish_reason_str; rating = next((r for r in getattr(candidate, 'safety_ratings', []) if r.blocked), None)
                            details = f" (Cat: {getattr(rating.category, 'name', 'N/A')}, Prob: {getattr(rating.probability, 'name', 'N/A')})" if rating else ""
                            error_msg = f"Blocked (response safety{details})."; logger.error(f"Gemini stream blocked (response safety): {error_msg}"); yield {"type": "error", "content": error_msg}; break

                # ... (Usage Metadata) ...
                if hasattr(chunk, 'usage_metadata'):
                     usage = chunk.usage_metadata; usage_info = {"prompt_token_count": getattr(usage, 'prompt_token_count', None), "candidates_token_count": getattr(usage, 'candidates_token_count', None), "total_token_count": getattr(usage, 'total_token_count', None)}; final_metadata["usage"] = usage_info

                # ... (Extract Text and Parse Thoughts - same logic as before) ...
                chunk_raw_content = ""
                try:
                     if chunk.candidates and hasattr(chunk.candidates[0], 'content') and hasattr(chunk.candidates[0].content, 'parts'): chunk_raw_content = "".join(p.text for p in chunk.candidates[0].content.parts if hasattr(p, 'text'))
                except ValueError: 
                    logger.debug("ValueError accessing chunk text (safety?)")
                    if block_reason: break; continue
                except Exception as e: logger.error(f"Error accessing chunk text: {e}"); break

                if chunk_raw_content:
                    full_raw_response_text += chunk_raw_content
                    # --- Stream parsing logic ---
                    current_text_to_process = chunk_raw_content; processed_text_chunk = ""; processed_thoughts_chunk = None
                    while current_text_to_process:
                        if is_thinking:
                            end_tag_pos = current_text_to_process.find("</think>");
                            if end_tag_pos != -1: thought_part = current_text_to_process[:end_tag_pos]; accumulated_thoughts += thought_part; processed_thoughts_chunk = accumulated_thoughts; accumulated_thoughts = ""; is_thinking = False; current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                            else: accumulated_thoughts += current_text_to_process; current_text_to_process = ""
                        else:
                            start_tag_pos = current_text_to_process.find("<think>");
                            if start_tag_pos != -1: text_part = current_text_to_process[:start_tag_pos]; processed_text_chunk += text_part; is_thinking = True; current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                            else: processed_text_chunk += current_text_to_process; current_text_to_process = ""
                    if processed_text_chunk or processed_thoughts_chunk: yield {"type": "chunk", "content": processed_text_chunk if processed_text_chunk else None, "thoughts": processed_thoughts_chunk}
                    # --- End Stream parsing ---

                if block_reason: logger.error(f"Stopping Gemini stream due to block: {block_reason}"); break

            # --- After the Loop ---
            # (Final chunk processing logic remains the same)
            if not block_reason:
                if is_thinking and accumulated_thoughts: logger.warning(f"Stream ended mid-thought: {accumulated_thoughts}"); final_metadata["incomplete_thoughts"] = accumulated_thoughts
                logger.info(f"Gemini stream finished. Reason: {finish_reason_str or 'completed'}")
                final_metadata["finish_reason"] = finish_reason_str or ("incomplete_thought" if is_thinking else "completed")
                final_cleaned_text, final_thoughts_str = parse_thought_tags(full_raw_response_text)
                if final_metadata.get("incomplete_thoughts"): incomplete = final_metadata["incomplete_thoughts"]; final_thoughts_str = (final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + incomplete).strip() if final_thoughts_str else incomplete
                final_output_list = [{"type": "text", "data": final_cleaned_text.strip(), "thoughts": final_thoughts_str, "metadata": final_metadata}]
                yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}
            else:
                error_final_output = [{"type": "error", "data": f"Stream stopped (safety block: {block_reason}).", "metadata": final_metadata}]
                yield {"type": "final", "content": error_final_output, "metadata": {"status": "failed", "reason": block_reason}}

        # (Error handling remains the same)
        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.InvalidArgument) as e: logger.error(f"Gemini API Error during stream '{self.binding_instance_name}': {e}"); yield {"type": "error", "content": f"Gemini API Error: {e}"}
        except Exception as e: logger.error(f"Unexpected Gemini stream error '{self.binding_instance_name}': {e}", exc_info=True); yield {"type": "error", "content": f"Unexpected Gemini stream error: {e}"}
        finally:
            # (Closing PIL images remains the same)
            if loaded_pil_images: logger.debug(f"Closing {len(loaded_pil_images)} PIL images."); [img.close() for img in loaded_pil_images if hasattr(img,'close')]

    # --- Tokenizer / Info Methods ---

    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        """Approximates tokenization using tiktoken."""
        target_model = model_name or self.model_name or self.default_model_name or "gpt-4" # Fallback model for tiktoken
        if not tiktoken_installed: logger.warning(f"Gemini '{self.binding_instance_name}': Tiktoken not installed. Cannot tokenize."); return list(range(len(text)))
        logger.info(f"Gemini '{self.binding_instance_name}': Estimating tokens for model '{target_model}' with tiktoken.")
        try:
            try: encoding = tiktoken.encoding_for_model(target_model)
            except KeyError: logger.debug(f"Tiktoken encoding for '{target_model}' not found, using 'cl100k_base'."); encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if add_bos: logger.debug("add_bos not supported by tiktoken.")
            if add_eos: logger.debug("add_eos not supported by tiktoken.")
            return tokens
        except Exception as e: logger.warning(f"Gemini binding '{self.binding_instance_name}': Tiktoken failed ({e})."); return list(range(len(text)))

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        """Approximates detokenization using tiktoken."""
        target_model = model_name or self.model_name or self.default_model_name or "gpt-4"
        if not tiktoken_installed: logger.warning(f"Gemini '{self.binding_instance_name}': Tiktoken not installed. Cannot detokenize."); return f"(detokenization unavailable: {len(tokens)} tokens)"
        logger.info(f"Gemini '{self.binding_instance_name}': Approximating detokenization for model '{target_model}' with tiktoken.")
        try:
            try: encoding = tiktoken.encoding_for_model(target_model)
            except KeyError: encoding = tiktoken.get_encoding("cl100k_base")
            text = encoding.decode(tokens)
            return text
        except Exception as e: logger.warning(f"Gemini binding '{self.binding_instance_name}': Tiktoken detokenization failed ({e})."); return str(tokens)

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns information about a specific Gemini model or the instance's active/default model."""
        target_model_id = model_name

        if target_model_id is None:
            if self._model_loaded and self.model_name:
                target_model_id = self.model_name
                logger.debug(f"Getting info for currently active model: {target_model_id}")
                # Use cached details if available and match
                if self.current_model_details.get('name') == target_model_id:
                    # Ensure all required keys are present before returning cache
                    required_keys = ["model_type", "context_size", "max_output_tokens", "supports_vision", "supports_audio", "supports_streaming", "details"]
                    if all(key in self.current_model_details for key in required_keys):
                        # Add binding instance name before returning
                        self.current_model_details["binding_instance_name"] = self.binding_instance_name
                        return self.current_model_details
                    else:
                         logger.warning("Cached model details are incomplete. Refetching.")
            elif self.default_model_name:
                target_model_id = self.default_model_name
                logger.debug(f"No model active, getting info for instance default: {target_model_id}")
            else:
                logger.warning(f"Gemini instance '{self.binding_instance_name}': Cannot get model info - no model specified, active, or configured as default.")
                return { "binding_instance_name": self.binding_instance_name, "model_name": None, "error": "No active or default model set", "details": {} }

        # Fetch details for the target model from the Gemini server
        if not self.genai or not self.api_key:
            logger.error(f"Gemini client/API key not configured for instance '{self.binding_instance_name}'. Cannot fetch model info.")
            return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": "Client/API key not configured", "details": {} }

        logger.info(f"Gemini '{self.binding_instance_name}': Fetching info for model 'models/{target_model_id}'...")
        try:
            model_info_obj = await asyncio.to_thread(self.genai.get_model, f"models/{target_model_id}")
            parsed_details = self._parse_gemini_details(model_info_obj)

            # Add binding instance name and ensure all keys for GetModelInfoResponse exist
            parsed_details["binding_instance_name"] = self.binding_instance_name
            parsed_details.setdefault("model_type", None)
            parsed_details.setdefault("context_size", None)
            parsed_details.setdefault("max_output_tokens", None)
            parsed_details.setdefault("supports_vision", False)
            parsed_details.setdefault("supports_audio", False)
            parsed_details.setdefault("supports_streaming", True) # Gemini generateContent supports streaming
            parsed_details.setdefault("details", {})

            # Update cache if this was the currently active model
            if self.model_name == target_model_id:
                 self.current_model_details = parsed_details.copy()
                 # Update internal state based on fetched info IF auto-detect is on
                 if self.auto_detect_limits:
                      if parsed_details.get("context_size"): self.current_ctx_size = parsed_details["context_size"]
                      if parsed_details.get("max_output_tokens"): self.current_max_output_tokens = parsed_details["max_output_tokens"]
                 self.model_supports_vision = parsed_details.get("supports_vision", False)

            return parsed_details

        except google.api_core.exceptions.NotFound:
             logger.warning(f"Gemini model 'models/{target_model_id}' not found via API.")
             return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": "Model not found via API.", "details": {"status": "not_found"} }
        except google.api_core.exceptions.GoogleAPIError as e:
             logger.error(f"Gemini API error fetching info for model 'models/{target_model_id}': {e}")
             return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": f"API Error: {e}", "details": {} }
        except Exception as e:
            logger.error(f"Unexpected error fetching Gemini model info for 'models/{target_model_id}': {e}", exc_info=True)
            return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": f"Unexpected error: {e}", "details": {} }