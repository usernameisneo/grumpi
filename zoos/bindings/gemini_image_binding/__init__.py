# zoos/bindings/gemini_image_binding/__init__.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: Your Name/Team
# Date: 2025-05-04
# Description: Binding implementation for Google Gemini/Imagen Image Generation API.
# Correction: Uses generate_content for both models, adapting parameters.

import asyncio
import os
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List

# Use pipmaster if needed (optional dependency check)
try:
    import pipmaster as pm
    pm.ensure_packages(["google-generativeai>=0.4.0", "pillow>=9.0.0"])
except ImportError:
    pass

# --- Core Library Imports ---
try:
    import google.generativeai as genai
    import google.api_core.exceptions
    # --- Corrected Types ---
    # GenerationConfig is used for generate_content
    # GenerateImagesConfig and GenerateImagesResponse are typically NOT in this namespace
    from google.generativeai.types import (
        GenerationConfig,
        HarmCategory, HarmBlockThreshold,
        GenerateContentResponse # generate_content returns this
    )
    from PIL import Image
    gemini_installed = True
    pillow_installed = True
except ImportError as e:
    genai = None; google = None; Image = None; BytesIO = None # type: ignore
    # Mock necessary types for hinting
    class MockGenConfig: pass; GenerationConfig = MockGenConfig # type: ignore
    class MockHarmCategory: pass; HarmCategory = MockHarmCategory # type: ignore
    class MockHarmBlockThreshold: pass; HarmBlockThreshold = MockHarmBlockThreshold # type: ignore
    class MockGenContentResp: pass; GenerateContentResponse = MockGenContentResp # type: ignore
    gemini_installed = False; pillow_installed = False
    _import_error_msg = str(e)


# --- Lollms Imports ---
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        class StreamChunk: pass # type: ignore
        class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

# --- Known Models and their Methods/Capabilities ---
# Now assuming BOTH use generate_content, but we store their capabilities
SUPPORTED_MODELS = {
    "gemini-2.0-flash-exp-image-generation": {
        "method": "generate_content", # Use standard method
        "inputs": ["text", "image"],
        "outputs": ["text", "image"],
        "family": "gemini"
    },
    "imagen-3.0-generate-002": {
        "method": "generate_content", # Assume standard method
        "inputs": ["text"], # Only text prompt according to docs
        "outputs": ["image"],
        "family": "imagen"
    }
    # Add future Gemini/Imagen image models here
}

class GeminiImageBinding(Binding):
    """Binding for Google Gemini/Imagen Image Generation API."""
    binding_type_name = "gemini_image_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the GeminiImageBinding."""
        super().__init__(config, resource_manager)

        if not gemini_installed or not genai:
            raise ImportError(f"Gemini Image binding requires 'google-generativeai'. Error: {_import_error_msg}")
        if not pillow_installed:
            logger.warning("Pillow library not found. Image processing will be limited.")

        self.api_key = self.config.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning(f"Gemini Image Binding '{self.binding_instance_name}': API key not found. API calls will fail.")
        else:
            try:
                # Configure the library once
                genai.configure(api_key=self.api_key)
                logger.info(f"Gemini Image Binding '{self.binding_instance_name}': Configured Google API key.")
            except Exception as e:
                logger.error(f"Gemini Image Binding '{self.binding_instance_name}': Failed configure genai library: {e}", exc_info=True)

        # --- Load Defaults from Instance Config ---
        self.default_num_images = self.config.get("default_number_of_images", 1)
        self.default_aspect_ratio = self.config.get("default_aspect_ratio", "1:1")
        self.default_person_gen = self.config.get("default_person_generation", "ALLOW_ADULT")

        # --- Gemini Flash Safety Settings (similar to gemini_binding) ---
        self.safety_settings_config: Dict[str, str] = {}
        for cat_key in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]:
            setting_key_in_config = f"safety_setting_{cat_key.split('_')[-1].lower()}"
            self.safety_settings_config[cat_key] = self.config.get(setting_key_in_config, "BLOCK_MEDIUM_AND_ABOVE")
        logger.debug(f"Gemini Image '{self.binding_instance_name}': Safety settings configured: {self.safety_settings_config}")

        # --- Internal State ---
        self.model: Optional[genai.GenerativeModel] = None # Store the SDK model object
        self.model_capabilities: Dict[str, Any] = {}

    # --- Capability Reporting Methods (Unchanged) ---
    def get_supported_input_modalities(self) -> List[str]:
        return self.model_capabilities.get("inputs", ["text"])

    def get_supported_output_modalities(self) -> List[str]:
        return self.model_capabilities.get("outputs", ["image"])

    # --- Required Binding Methods (Most implementations unchanged, just adapting generate) ---
    async def list_available_models(self) -> List[Dict[str, Any]]:
        # (Implementation unchanged - lists models from SUPPORTED_MODELS constant)
        logger.info(f"Gemini Image Binding '{self.binding_instance_name}': Listing supported image models.")
        models_list = []
        for model_id, details in SUPPORTED_MODELS.items():
            models_list.append({
                "name": model_id,
                "format": "api",
                "family": details.get("family"),
                "families": [details.get("family")] if details.get("family") else None,
                "supports_vision": "image" in details.get("inputs", []),
                "supports_audio": False,
                "details": {
                    "api_method": details.get("method"),
                    "outputs": details.get("outputs"),
                }
            })
        return models_list

    async def health_check(self) -> Tuple[bool, str]:
        # (Implementation unchanged - lists models to check connection/auth)
        if not self.api_key or not genai:
            return False, "API key or genai library not configured."
        try:
            logger.info(f"Gemini Image Binding '{self.binding_instance_name}': Running health check (listing models)...")
            models = await asyncio.to_thread(genai.list_models)
            return True, f"Connection OK ({len(list(models))} models found)."
        except google.api_core.exceptions.PermissionDenied as e:
            logger.error(f"Gemini Image Health check '{self.binding_instance_name}' failed (Perm Denied): {e}")
            return False, f"Permission Denied/Invalid API Key: {e}"
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Gemini Image Health check '{self.binding_instance_name}' failed (API Err): {e}")
            return False, f"Google API Error: {e}"
        except Exception as e:
            logger.error(f"Gemini Image Health check '{self.binding_instance_name}' failed (Unexpected Err): {e}", exc_info=True)
            return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        # (Implementation unchanged)
        return {"gpu_required": False}

    def _reset_model_state(self):
        # (Implementation unchanged)
        self.model_name = None
        self._model_loaded = False
        self.model_capabilities = {}
        self.model = None # Also reset the SDK model object

    async def load_model(self, model_name: str) -> bool:
        """Sets the active model, capabilities, and initializes the SDK model object."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                logger.info(f"Gemini Image '{self.binding_instance_name}': Model '{model_name}' already active.")
                return True

            if model_name not in SUPPORTED_MODELS:
                logger.error(f"Gemini Image '{self.binding_instance_name}': Model '{model_name}' is not supported by this binding.")
                await self.unload_model()
                return False

            # Store the name and capabilities
            self.model_name = model_name
            self.model_capabilities = SUPPORTED_MODELS[model_name]
            self._model_loaded = True

            # --- Initialize the SDK Model Object ---
            if not genai:
                logger.error("Cannot initialize SDK model object: genai library not loaded.")
                await self.unload_model()
                return False
            try:
                self.model = await asyncio.to_thread(genai.GenerativeModel, model_name)
                logger.info(f"Gemini Image Binding '{self.binding_instance_name}': SDK Model object initialized for '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize SDK model object for '{model_name}': {e}", exc_info=True)
                await self.unload_model()
                return False

            logger.info(f"Gemini Image Binding '{self.binding_instance_name}': Active model set to '{self.model_name}'. Capabilities: {self.model_capabilities}")
            return True

    async def unload_model(self) -> bool:
        # (Implementation unchanged)
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Gemini Image '{self.binding_instance_name}': Unsetting active model '{self.model_name}'.")
            self._reset_model_state()
            return True

    # --- Helper Methods (Content/Safety Prep - Unchanged) ---
    def _prepare_gemini_flash_content(self, prompt: str, multimodal_data: Optional[List['InputData']]) -> Tuple[List[Union[str, Image.Image]], List[Image.Image]]:
        # (Implementation unchanged)
        content_parts: List[Union[str, Image.Image]] = []; loaded_pil_images: List[Image.Image] = []
        if prompt: content_parts.append(prompt)
        image_items = [item for item in (multimodal_data or []) if item.type == 'image']
        if image_items:
            if not pillow_installed: logger.error("Pillow library not found, cannot process images for Gemini Flash.")
            else:
                logger.info(f"Gemini Image '{self.binding_instance_name}': Processing {len(image_items)} image item(s) for Gemini Flash...")
                successful_loads = 0
                for i, item in enumerate(image_items):
                    img = None
                    if not item.data or not isinstance(item.data, str): logger.warning(f"Skipping img {i+1}: Missing/invalid data."); continue
                    try:
                        if len(item.data) > 10:
                            image_bytes = base64.b64decode(item.data); img = Image.open(BytesIO(image_bytes))
                            img = img.convert("RGB"); loaded_pil_images.append(img); content_parts.append(img)
                            successful_loads += 1; logger.debug(f" -> Loaded image {i+1} (role: {item.role}) for Gemini Flash")
                        else: logger.warning(f"Skipping img {i+1}: Data looks too short for base64."); continue
                    except Exception as e:
                        logger.error(f"Gemini Image '{self.binding_instance_name}': Failed load/process image index {i} for Flash: {e}", exc_info=True)
                        if img and hasattr(img, 'close'): 
                            try: 
                                img.close()
                            except Exception:
                                pass
                if successful_loads < len(image_items): logger.warning("Some images failed to load for Gemini Flash.")
        return content_parts, loaded_pil_images

    def _prepare_gemini_flash_safety_settings(self) -> Dict[Any, Any]:
        # (Implementation unchanged)
        api_safety_settings = {};
        if genai and hasattr(genai, 'types'):
            for cat_key, block_level_str in self.safety_settings_config.items():
                try: cat_enum = getattr(HarmCategory, cat_key); block_enum = getattr(HarmBlockThreshold, block_level_str); api_safety_settings[cat_enum] = block_enum
                except AttributeError: logger.warning(f"Could not map safety setting: {cat_key} = {block_level_str}")
        else: logger.warning("Could not prepare safety settings: genai.types not available.")
        return api_safety_settings

    # --- Corrected Generation Logic ---
    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Always return List[OutputData]-like
        """Generates images or text+images using generate_content API call."""
        if not genai: raise RuntimeError("Gemini library not loaded.")
        if not self.model: raise RuntimeError(f"SDK Model object not initialized for '{self.binding_instance_name}'.")
        if not self._model_loaded or not self.model_name: raise RuntimeError(f"Model not active for Gemini Image instance '{self.binding_instance_name}'.")
        if not self.api_key: raise RuntimeError(f"API key not configured for Gemini Image instance '{self.binding_instance_name}'.")

        logger.info(f"Gemini Image '{self.binding_instance_name}': Generating with model '{self.model_name}' using generate_content...")

        output_list: List[Dict[str, Any]] = []
        loaded_pil_images: List[Image.Image] = []

        try:
            # --- Prepare Common Generation Config ---
            # Use GenerationConfig for BOTH models when calling generate_content
            gen_config_options: Dict[str, Any] = {}
            # Common params (temp, top_p, top_k) apply mostly to text generation part of Flash
            if "temperature" in params: gen_config_options['temperature'] = float(params['temperature'])
            if "top_p" in params: gen_config_options['top_p'] = float(params['top_p'])
            if "top_k" in params: gen_config_options['top_k'] = int(params['top_k'])
            # Max tokens only relevant for Flash text output
            if self.model_capabilities.get("method") == "generate_content":
                gen_config_options['max_output_tokens'] = params.get("max_tokens", 2048)

            # --- Specific parameters based on model type ---
            if self.model_capabilities.get("family") == "imagen":
                # Imagen 3 specific params - need to be passed differently,
                # potentially via a dedicated config object if the SDK supports it
                # within generate_content, or maybe as direct arguments?
                # Let's try passing them within generation_config dict as a *guess*
                # as the generate_images method is not standard.
                # THIS IS SPECULATIVE - API might ignore these or require different format.
                num_img = params.get("number_of_images", self.default_num_images)
                gen_config_options['number_of_images'] = max(1, min(int(num_img), 4))

                aspect = params.get("aspect_ratio", self.default_aspect_ratio)
                supported_ratios = ["1:1", "3:4", "4:3", "9:16", "16:9"]
                gen_config_options['aspect_ratio'] = aspect if aspect in supported_ratios else self.default_aspect_ratio

                person_gen = params.get("person_generation", self.default_person_gen)
                supported_person_gen = ["DONT_ALLOW", "ALLOW_ADULT"]
                gen_config_options['person_generation'] = person_gen if person_gen in supported_person_gen else self.default_person_gen

                # Imagen 3 doesn't output text, so ensure response modality is IMAGE only
                gen_config_options['response_modalities'] = ['IMAGE']
                logger.debug(f"Imagen 3 specific config params added: {gen_config_options}")

            elif self.model_capabilities.get("family") == "gemini":
                # Gemini Flash Exp requires specifying modalities
                gen_config_options['response_modalities'] = ['TEXT', 'IMAGE']
                logger.debug("Gemini Flash specific config: response_modalities=['TEXT', 'IMAGE']")


            # Create the final GenerationConfig object
            generation_config = GenerationConfig(**gen_config_options)

            # --- Prepare Content and Safety ---
            content_payload: List[Union[str, Image.Image]] = []
            if self.model_capabilities.get("family") == "imagen":
                # Imagen only takes text prompt
                if prompt: content_payload = [prompt]
                if multimodal_data and any(item.type == 'image' for item in multimodal_data):
                    logger.warning(f"Imagen 3 ('{self.model_name}') received image input, ignoring.")
            else: # Gemini Flash
                content_payload, loaded_pil_images = self._prepare_gemini_flash_content(prompt, multimodal_data)

            if not content_payload:
                raise ValueError("No valid prompt or image content could be prepared for the API call.")

            safety_settings = self._prepare_gemini_flash_safety_settings() # Use same safety for both for now

            # Extract system message (optional for image models, primarily for Flash text part)
            system_message = params.get("system_message", None)

            # --- API Call using generate_content ---
            logger.debug(f"Gemini generate_content call: model={self.model_name}, system='{str(system_message)[:50] if system_message else 'None'}...', config={generation_config}")
            response: GenerateContentResponse = await asyncio.to_thread(
                self.model.generate_content,
                contents=f"system:{system_message}\n"+content_payload if system_message and type(content_payload)==str else [f"system:{system_message}\n"]+content_payload if system_message and type(content_payload)==list else content_payload,
                generation_config=generation_config,
                stream=False,
                safety_settings=safety_settings,
            )

            # --- Process Response ---
            # (Safety/Candidate checking logic remains the same as Gemini Flash part before)
            if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                 reason = response.prompt_feedback.block_reason.name; logger.error(f"Gemini prompt blocked: {reason}"); raise ValueError(f"Request blocked (prompt safety: {reason}).")
            if not response.candidates: raise RuntimeError("Generation failed: No candidates returned.")

            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name != "STOP":
                finish_reason_str = candidate.finish_reason.name
                if finish_reason_str == "SAFETY":
                    rating = next((r for r in getattr(candidate, 'safety_ratings', []) if r.blocked), None)
                    details = f" (Cat: {getattr(rating.category, 'name', 'N/A')})" if rating else ""
                    raise ValueError(f"Response blocked (content safety{details}).")
                else: logger.warning(f"Model '{self.model_name}' finished with reason: {finish_reason_str}")

            # --- Extract Parts (Text and/or Images) ---
            full_text_response = ""
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                         full_text_response += part.text
                    elif hasattr(part, 'inline_data') and part.inline_data:
                         try:
                             img_bytes = part.inline_data.data
                             img_mime_type = part.inline_data.mime_type
                             if pillow_installed and Image and BytesIO:
                                 pil_img = Image.open(BytesIO(img_bytes)); output_buffer = BytesIO()
                                 pil_img.save(output_buffer, format="PNG"); img_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                                 pil_img.close()
                                 output_list.append({ "type": "image", "data": img_base64, "mime_type": "image/png", "metadata": {"source_model": self.model_name, "original_mime": img_mime_type} })
                                 logger.debug(" -> Appended image from response.")
                             else:
                                 img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                                 output_list.append({ "type": "image", "data": img_base64, "mime_type": img_mime_type, "metadata": {"source_model": self.model_name, "warning": "pillow not installed, validation skipped"} })
                         except Exception as img_err: logger.error(f"Failed process inline image data: {img_err}")

            # Add text output if generated (primarily from Gemini Flash)
            cleaned_text, thoughts = parse_thought_tags(full_text_response)
            if cleaned_text.strip():
                 output_list.append({"type": "text", "data": cleaned_text.strip(), "thoughts": thoughts, "metadata": {"source_model": self.model_name}})
            elif thoughts: # If only thoughts were generated
                 output_list.append({"type": "thoughts", "data": thoughts, "metadata": {"source_model": self.model_name}})

            # If no image or text was produced, add an error/info message
            if not output_list:
                logger.warning(f"Gemini Image generation for '{self.model_name}' produced no usable output parts.")
                output_list.append({"type": "error", "data": "Generation produced no output.", "metadata": {"source_model": self.model_name}})


            logger.info(f"Gemini Image generation successful for '{self.binding_instance_name}'. Returned {len(output_list)} items.")
            # Add common metadata
            for item in output_list: item.setdefault("metadata", {}); item["metadata"]["binding_instance"] = self.binding_instance_name; item["metadata"]["model_used"] = self.model_name
            return output_list

        except ValueError as e: # Catch safety blocks etc.
            logger.error(f"Gemini Image '{self.binding_instance_name}': Value error: {e}")
            raise e
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API Error '{self.binding_instance_name}': {e}")
            raise RuntimeError(f"Gemini API Error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected Gemini Image error '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Gemini Image error: {e}") from e
        finally:
            if loaded_pil_images: logger.debug(f"Closing {len(loaded_pil_images)} PIL images."); [img.close() for img in loaded_pil_images if hasattr(img,'close')]

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Simulates streaming for image generation by yielding info then final result."""
        # (Implementation unchanged)
        logger.info(f"Gemini Image Binding '{self.binding_instance_name}': Simulating stream...")
        active_model = self.model_name or self.default_model_name or "Unknown"
        yield {"type": "info", "content": {"status": "starting_generation", "model": active_model, "prompt": prompt}}
        try:
            result_list = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
            yield {"type": "final", "content": result_list, "metadata": {"status": "complete"}}
            logger.info(f"Gemini Image Binding '{self.binding_instance_name}': Simulated stream finished successfully.")
        except Exception as e:
            logger.error(f"Gemini Image Binding '{self.binding_instance_name}': Error during simulated stream's generate call: {e}", exc_info=True)
            error_content = f"Generation failed: {str(e)}"
            yield {"type": "error", "content": error_content}
            yield {"type": "final", "content": [{"type": "error", "data": error_content}], "metadata": {"status": "failed"}}

    # --- Tokenizer / Info Methods (Unchanged stubs) ---
    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        raise NotImplementedError("Gemini Image binding does not support tokenization.")

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        raise NotImplementedError("Gemini Image binding does not support detokenization.")

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        # (Implementation unchanged - returns info based on SUPPORTED_MODELS constant)
        target_model_id = model_name
        if target_model_id is None:
            if self._model_loaded and self.model_name: target_model_id = self.model_name
            elif self.default_model_name: target_model_id = self.default_model_name
            else:
                logger.warning(f"Gemini Image instance '{self.binding_instance_name}': Cannot get model info - no model specified, active, or default.")
                return { "binding_instance_name": self.binding_instance_name, "model_name": None, "error": "No active or default model set", "details": {} }

        if target_model_id not in SUPPORTED_MODELS:
            logger.error(f"Model '{target_model_id}' is not recognized/supported by the Gemini Image binding.")
            return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": "Model not supported by this binding.", "details": {} }

        capabilities = SUPPORTED_MODELS[target_model_id]
        can_input_image = "image" in capabilities.get("inputs", [])
        can_output_text = "text" in capabilities.get("outputs", [])

        model_info = {
            "binding_instance_name": self.binding_instance_name,
            "model_name": target_model_id,
            "model_type": 'tti' if not can_output_text else 'multimodal',
            "context_size": None, "max_output_tokens": None,
            "supports_vision": can_input_image, "supports_audio": False,
            "supports_streaming": False, # Simulates streaming
            "details": { "family": capabilities.get("family"), "api_method": capabilities.get("method"), "supported_inputs": capabilities.get("inputs"), "supported_outputs": capabilities.get("outputs"), **( {"imagen3_defaults": {"number_of_images": self.default_num_images, "aspect_ratio": self.default_aspect_ratio, "person_generation": self.default_person_gen} } if capabilities.get("family") == "imagen" else {} ) }
        }
        return model_info