# zoos/bindings/gemini_binding.py
import asyncio
import ascii_colors as logging
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
import pipmaster as pm

pm.install_if_missing("google-generativeai")
pm.install_if_missing("pillow")
try:
    import google.generativeai as genai
    import google.api_core.exceptions
    from google.generativeai.types import GenerationConfig, ContentDict, PartDict, Model as GeminiModelType
    from PIL import Image
    from io import BytesIO
    gemini_installed = True
except ImportError:
    class MockGenerationConfig: pass
    class MockGeminiModelType: pass
    GenerationConfig = MockGenerationConfig # type: ignore
    GeminiModelType = MockGeminiModelType # type: ignore
    genai = None; google = None; Image = None; BytesIO = None
    gemini_installed = False

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
try:
    from lollms_server.api.models import StreamChunk, InputData
except ImportError:
     class StreamChunk: pass # type: ignore
     class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

SAFETY_CATEGORIES = { "HARM_CATEGORY_HARASSMENT": "Harassment", "HARM_CATEGORY_HATE_SPEECH": "Hate Speech", "HARM_CATEGORY_SEXUALLY_EXPLICIT": "Sexually Explicit", "HARM_CATEGORY_DANGEROUS_CONTENT": "Dangerous Content" }
SAFETY_OPTIONS = [ "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE" ]
DEFAULT_SAFETY_SETTING = "BLOCK_MEDIUM_AND_ABOVE"

class GeminiBinding(Binding):
    """Binding for Google's Gemini API."""
    binding_type_name = "gemini_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the GeminiBinding."""
        super().__init__(config, resource_manager)
        if not gemini_installed: raise ImportError("Gemini binding requires 'google-generativeai' and 'pillow'.")
        self.api_key = self.config.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
        self.auto_detect_limits = self.config.get("auto_detect_limits", True)
        self.manual_ctx_size = self.config.get("ctx_size", 30720)
        self.manual_max_output_tokens = self.config.get("max_output_tokens", 2048)
        self.safety_settings: Dict[str, str] = { cat_key: self.config.get(f"safety_setting_{cat_key.split('_')[-1].lower()}", DEFAULT_SAFETY_SETTING) for cat_key in SAFETY_CATEGORIES }
        logger.debug(f"Gemini '{self.binding_name}': Safety settings: {self.safety_settings}")
        self.genai = genai; self.model: Optional[genai.GenerativeModel] = None; self.model_name: Optional[str] = None
        self.current_ctx_size: int = self.manual_ctx_size; self.current_max_output_tokens: int = self.manual_max_output_tokens
        self.model_supports_vision: bool = False
        if self.api_key:
             try: self.genai.configure(api_key=self.api_key); logger.info(f"Gemini '{self.binding_name}': Configured Google API key.")
             except Exception as e: logger.error(f"Gemini '{self.binding_name}': Failed config: {e}", exc_info=True)
        else: logger.warning(f"Gemini '{self.binding_name}': API key not found.")

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the Gemini binding."""
        safety_templates = [ { "name": f"safety_setting_{cat.split('_')[-1].lower()}", "type": "str", "value": DEFAULT_SAFETY_SETTING, "options": SAFETY_OPTIONS, "help": f"Safety setting for {display_name}." } for cat, display_name in SAFETY_CATEGORIES.items() ]
        return { "type_name": cls.binding_type_name, "version": "1.2", "description": "Binding for Google Gemini API (text and vision).", "supports_streaming": True, "requirements": ["google-generativeai>=0.4.0", "pillow>=9.0.0"], "config_template": { "type": {"type": "string", "value": cls.binding_type_name, "required":True}, "google_api_key": {"type": "string", "value": "", "required":False}, "auto_detect_limits": {"type": "bool", "value": True, "required":False}, "ctx_size": {"type": "int", "value": 30720, "required":False}, "max_output_tokens": {"type": "int", "value": 2048, "required":False}, **{tmpl["name"]: tmpl for tmpl in safety_templates} } }

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types (text, potentially image)."""
        modalities = ['text']
        if self.model_supports_vision: modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text']

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity by listing models."""
        if not self.api_key: return False, "API key not configured."
        try: self.genai.list_models(); return True, "Connection successful."
        except google.api_core.exceptions.PermissionDenied as e: logger.error(f"Health check fail (Permission Denied): {e}"); return False, f"Permission Denied: {e}"
        except google.api_core.exceptions.GoogleAPIError as e: logger.error(f"Health check fail (API Error): {e}"); return False, f"Google API Error: {e}"
        except Exception as e: logger.error(f"Health check fail (Error): {e}", exc_info=True); return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Gemini is remote, no local GPU needed."""
        return {"gpu_required": False}

    def _parse_gemini_details(self, model: GeminiModelType) -> Dict[str, Any]:
        """Parses the raw Gemini model data."""
        model_id = model.name.split('/')[-1]; parsed = {'name': model_id}
        parsed['size'] = None; parsed['modified_at'] = None; parsed['quantization_level'] = None; parsed['format'] = "api";
        parsed['family'] = "gemini"; parsed['families'] = ["gemini"]; parsed['parameter_size'] = None; parsed['template'] = None;
        parsed['license'] = "Commercial API"; parsed['homepage'] = "https://ai.google.dev/";
        parsed['context_size'] = getattr(model, 'input_token_limit', None); parsed['max_output_tokens'] = getattr(model, 'output_token_limit', None)
        supports_vision = ('vision' in model_id.lower() or 'Vision' in getattr(model, 'display_name', '') or "1.5" in model_id.lower()) and 'embedContent' in getattr(model, 'supported_generation_methods', [])
        parsed['supports_vision'] = supports_vision; parsed['supports_audio'] = False
        parsed['details'] = { "original_name": model.name, "display_name": getattr(model, 'display_name', model_id), "version": getattr(model, 'version', None), "description": getattr(model, 'description', None), "supported_generation_methods": getattr(model, 'supported_generation_methods', []), "temperature": getattr(model, 'temperature', None), "top_p": getattr(model, 'top_p', None), "top_k": getattr(model, 'top_k', None) }
        return parsed

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available via API."""
        logger.info(f"Gemini '{self.binding_name}': Listing models..."); available_models = []
        if not self.api_key: raise ValueError("API key not configured.")
        try:
            api_models = self.genai.list_models()
            for model in api_models:
                if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                    parsed_data = self._parse_gemini_details(model)
                    if parsed_data.get('name'): available_models.append(parsed_data)
            logger.info(f"Gemini '{self.binding_name}': Found {len(available_models)} usable models.")
            return available_models
        except google.api_core.exceptions.GoogleAPIError as e: logger.error(f"API Error listing models: {e}"); raise RuntimeError(f"Gemini API Error: {e}") from e
        except Exception as e: logger.error(f"Unexpected error listing models: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    async def load_model(self, model_name: str) -> bool:
        """Prepares the binding to use the specified Gemini model."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name: return True
            logger.info(f"Gemini '{self.binding_name}': Preparing model '{model_name}'.")
            if not self.api_key: logger.error("Cannot load model, API key missing."); return False
            try:
                model_info = self.genai.get_model(f"models/{model_name}")
                logger.info(f"Verified model '{model_name}' exists.")
                detected_ctx = getattr(model_info, 'input_token_limit', None); detected_max_out = getattr(model_info, 'output_token_limit', None)
                if self.auto_detect_limits and detected_ctx and detected_max_out:
                    self.current_ctx_size = detected_ctx; self.current_max_output_tokens = detected_max_out
                    logger.info(f"Auto-detected limits: ctx={detected_ctx}, max_out={detected_max_out}")
                else:
                    if self.auto_detect_limits: logger.warning(f"Failed auto-detect limits for '{model_name}'. Using manual.")
                    else: logger.info(f"Using manual limits: ctx={self.manual_ctx_size}, max_out={self.manual_max_output_tokens}")
                    self.current_ctx_size = self.manual_ctx_size; self.current_max_output_tokens = self.manual_max_output_tokens
                self.model_supports_vision = ('vision' in model_name.lower() or 'Vision' in getattr(model_info, 'display_name', '') or "1.5" in model_name.lower()) and 'embedContent' in getattr(model_info, 'supported_generation_methods', [])
                logger.info(f"Model '{model_name}' vision support: {self.model_supports_vision}")
                self.model = self.genai.GenerativeModel(model_name); self.model_name = model_name; self._model_loaded = True
                logger.info(f"Prepared for model '{model_name}'."); return True
            except google.api_core.exceptions.NotFound: logger.error(f"Model 'models/{model_name}' not found."); self.model=None; self.model_name=None; self._model_loaded=False; return False
            except google.api_core.exceptions.GoogleAPIError as e: logger.error(f"API error preparing '{model_name}': {e}"); self.model=None; self.model_name=None; self._model_loaded=False; return False
            except Exception as e: logger.error(f"Unexpected error preparing '{model_name}': {e}", exc_info=True); self.model=None; self.model_name=None; self._model_loaded=False; return False

    async def unload_model(self) -> bool:
        """Unsets the active Gemini model."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Gemini '{self.binding_name}': Unsetting model '{self.model_name}'.")
            self.model = None; self.model_name = None; self._model_loaded = False; self.model_supports_vision = False; return True

    def _prepare_generation_config(self, params: Dict[str, Any]) -> Optional[GenerationConfig]:
        """Helper to build the GenerationConfig object."""
        if not self.model: logger.error("Model not initialized."); return None
        n_predict = params.get("max_tokens"); generate_max_tokens = self.current_max_output_tokens
        if n_predict is not None:
            try:
                n_predict_int = int(n_predict)
                if 0 < n_predict_int <= self.current_max_output_tokens: generate_max_tokens = n_predict_int
                elif n_predict_int > self.current_max_output_tokens: logger.warning(f"max_tokens ({n_predict_int}) exceeds limit ({self.current_max_output_tokens}). Capping."); generate_max_tokens = self.current_max_output_tokens
            except ValueError: logger.warning(f"Invalid max_tokens: {n_predict}. Using default: {generate_max_tokens}")
        logger.debug(f"Effective max_output_tokens: {generate_max_tokens}")
        gen_config_params = { 'candidate_count': 1, 'max_output_tokens': generate_max_tokens, 'temperature': float(params.get('temperature', 0.7)), 'top_p': float(params.get('top_p', 0.95)), 'top_k': int(params.get('top_k', 40)), }
        stop_sequences = params.get('stop_sequences') or params.get('stop')
        if stop_sequences:
            valid_stop = [];
            if isinstance(stop_sequences, str): valid_stop = [s.strip() for s in stop_sequences.split(',') if s.strip()] or [stop_sequences]
            elif isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences): valid_stop = [s for s in stop_sequences if s]
            else: logger.warning(f"Invalid stop_sequences format: {stop_sequences}. Ignoring.")
            if valid_stop: gen_config_params['stop_sequences'] = valid_stop; logger.debug(f"Using stop sequences: {valid_stop}")
        try: return GenerationConfig(**gen_config_params)
        except Exception as e: logger.error(f"Failed to create GenerationConfig: {e}", exc_info=True); return None

    async def _process_content(self, prompt: str, multimodal_data: Optional[List['InputData']]) -> Union[str, List[Union[str, Image.Image]]]:
        """Prepares the content payload, handling images."""
        image_items = [item for item in (multimodal_data or []) if item.type == 'image']
        if not image_items or not self.model_supports_vision:
            if image_items: logger.warning(f"Gemini '{self.binding_name}': Image data provided, but model '{self.model_name}' ignores it.")
            return prompt
        if not Image or not BytesIO: raise RuntimeError("Pillow library needed for image processing.")
        content_parts: List[Union[str, Image.Image]] = [prompt]; loaded_images: List[Image.Image] = []; successful_loads = 0
        for i, item in enumerate(image_items):
            img = None
            try:
                if not isinstance(item.data, str): logger.warning(f"Skipping non-string image data at index {i}."); continue
                image_bytes = base64.b64decode(item.data); img = Image.open(BytesIO(image_bytes))
                loaded_images.append(img); content_parts.append(img); successful_loads += 1
                logger.info(f"Gemini '{self.binding_name}': Processed image {i+1} (role: {item.role}) from base64.")
            except Exception as e:
                logger.error(f"Gemini '{self.binding_name}': Failed to load image data at index {i}: {e}", exc_info=True)
                if img: 
                    try:
                        img.close()
                    except Exception:
                        pass
        if successful_loads == 0:
             logger.error("All image data failed to load. Falling back to text-only.")
             for img_obj in loaded_images: 
                try: 
                    img_obj.close(); 
                except Exception: 
                    pass
             return prompt
        elif successful_loads < len(image_items): logger.warning("Some image data failed to load.")
        return content_parts

    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> Union[str, Dict[str, Any]]:
        """Generates text using the Gemini API (non-streaming)."""
        if not self._model_loaded or not self.model_name or not self.model: raise RuntimeError("Model not loaded.")
        if not self.api_key: raise RuntimeError("API key not configured.")
        logger.info(f"Gemini '{self.binding_name}': Generating non-stream with '{self.model_name}'...")
        generation_config = self._prepare_generation_config(params)
        if generation_config is None: raise RuntimeError("Failed generation config.")
        content_payload = await self._process_content(prompt, multimodal_data)
        loaded_images = [part for part in content_payload if isinstance(part, Image.Image)] if isinstance(content_payload, list) else []
        try:
            response = await self.model.generate_content( content_payload, generation_config=generation_config, stream=False, safety_settings=self.safety_settings )
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name; logger.error(f"Generation blocked (prompt safety): {reason}"); raise ValueError(f"Blocked (prompt safety): {reason}")
            if not response.candidates or not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts'):
                 finish_reason = response.candidates[0].finish_reason.name if response.candidates and hasattr(response.candidates[0], 'finish_reason') else "UNKNOWN"
                 logger.error(f"No content in response. Reason: {finish_reason}")
                 if finish_reason == "SAFETY": raise ValueError(f"Blocked (response safety): {finish_reason}")
                 else: raise RuntimeError(f"Generation failed: No content. Reason: {finish_reason}")
            completion = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            logger.info("Gemini generation successful.")
            return {"text": completion.strip() if completion else ""}
        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.InvalidArgument) as e:
            logger.error(f"Gemini API Error: {e}")
            if isinstance(e, google.api_core.exceptions.InvalidArgument): raise ValueError(f"Gemini Invalid Argument: {e}") from e
            else: raise RuntimeError(f"Gemini API Error: {e}") from e
        except ValueError as e: raise e
        except Exception as e: logger.error(f"Unexpected error generation: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e
        finally:
             if loaded_images: logger.debug(f"Closing {len(loaded_images)} images."); [img.close() for img in loaded_images if hasattr(img, 'close')]

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates text using the Gemini API (streaming)."""
        if not self._model_loaded or not self.model_name or not self.model: yield {"type": "error", "content": "Model not loaded."}; return
        if not self.api_key: yield {"type": "error", "content": "API key not configured."}; return
        logger.info(f"Gemini '{self.binding_name}': Generating stream with '{self.model_name}'...")
        generation_config = self._prepare_generation_config(params)
        if generation_config is None: yield {"type": "error", "content": "Failed generation config."}; return
        content_payload = await self._process_content(prompt, multimodal_data)
        loaded_images = [part for part in content_payload if isinstance(part, Image.Image)] if isinstance(content_payload, list) else []
        full_response_content = ""; stream = None; block_reason = None; finish_reason_str = None
        try:
            stream = await self.model.generate_content( content_payload, generation_config=generation_config, stream=True, safety_settings=self.safety_settings )
            async for chunk in stream:
                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                    block_reason = chunk.prompt_feedback.block_reason.name; logger.error(f"Blocked (prompt safety): {block_reason}"); yield {"type": "error", "content": f"Blocked (prompt safety): {block_reason}"}; break
                if chunk.candidates:
                     candidate = chunk.candidates[0]
                     if candidate.finish_reason:
                         finish_reason = candidate.finish_reason; finish_reason_str = finish_reason.name
                         if finish_reason_str not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]:
                              logger.warning(f"API stop? Reason: {finish_reason_str}")
                              if finish_reason_str == "SAFETY": block_reason = finish_reason_str; yield {"type": "error", "content": f"Blocked (response safety): {block_reason}"}; break
                word = "";
                try:
                    if hasattr(chunk, 'text'): word = chunk.text
                except Exception as e: logger.error(f"Error accessing chunk text: {e}. Chunk: {chunk}")
                if word: full_response_content += word; yield {"type": "chunk", "content": word}
                if block_reason: logger.error(f"Stopping stream due to block: {block_reason}"); break
            if not block_reason: logger.info("Gemini stream finished."); final_metadata = {"reason": finish_reason_str or "completed"}; yield {"type": "final", "content": {"text": full_response_content}, "metadata": final_metadata}
        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.InvalidArgument) as e: logger.error(f"Gemini API Error stream: {e}"); yield {"type": "error", "content": f"Gemini API Error: {e}"}
        except Exception as e: logger.error(f"Unexpected stream error: {e}", exc_info=True); yield {"type": "error", "content": f"Unexpected error: {e}"}
        finally:
             if loaded_images: logger.debug(f"Closing {len(loaded_images)} images."); [img.close() for img in loaded_images if hasattr(img, 'close')]


    # --- NEW: Tokenizer / Info ---
    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenizes text using the Gemini API (count_tokens)."""
        if not self._model_loaded or not self.model_name or not self.model: raise RuntimeError("Model not loaded for tokenization")
        logger.info(f"Gemini '{self.binding_name}': Simulating tokenize via count_tokens...")
        # Gemini API doesn't directly expose token IDs easily.
        # We can use count_tokens to get the count, but not the IDs.
        # Raise NotImplementedError as we cannot return the required List[int].
        raise NotImplementedError("Gemini binding does not support direct tokenization.")
        # If we just wanted the count:
        # try:
        #     response = await self.model.count_tokens(text)
        #     return response.total_tokens # Incorrect return type!
        # except Exception as e:
        #     logger.error(f"Error counting tokens: {e}", exc_info=True)
        #     raise RuntimeError(f"Failed to count tokens: {e}") from e

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenization is not supported by the Gemini API."""
        if not self._model_loaded: raise RuntimeError("Model not loaded for detokenization")
        raise NotImplementedError("Gemini binding does not support detokenization.")

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded Gemini model."""
        if not self._model_loaded or not self.model_name: return {}
        return {
            "name": self.model_name,
            "context_size": self.current_ctx_size,
            "max_output_tokens": self.current_max_output_tokens,
            "supports_vision": self.model_supports_vision,
            "supports_audio": False,
            "details": {"info": f"Currently loaded Gemini model {self.model_name}"}
        }