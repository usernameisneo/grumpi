# encoding:utf-8
# Project: lollms_server
# File: zoos/bindings/dummy_binding/__init__.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Dummy binding implementation for testing purposes.

import asyncio
import time
import base64
import random # For dummy tokenization
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import nullcontext
from datetime import datetime, timedelta
from io import BytesIO

try:
    from PIL import Image
    pillow_installed = True
except ImportError:
    Image = None; BytesIO = None # type: ignore
    pillow_installed = False

try:
    import ascii_colors as logging # Use logging alias
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)


from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
# Use TYPE_CHECKING for API model imports to avoid runtime errors if they aren't loaded yet
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        # Define placeholders if import fails during type checking
        class StreamChunk: pass # type: ignore
        class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def generate_dummy_image_base64(width=256, height=256):
    """Generates a simple black image encoded as base64."""
    if not pillow_installed:
        logger.error("Pillow not installed, cannot generate dummy image.")
        # Return a minimal valid transparent PNG base64 as fallback
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    try:
        img = Image.new('RGB', (width, height), color = 'black')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating dummy image: {e}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def generate_dummy_audio_base64():
    """Returns a minimal valid WAV header base64 encoded."""
    # Minimal RIFF header for a WAV file (empty data chunk)
    return "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA=="
# --- End Helper Functions ---


class DummyBinding(Binding):
    """Dummy binding for testing purposes."""
    binding_type_name = "dummy_binding" # MUST match type_name in binding_card.yaml

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the DummyBinding using the instance configuration.

        Args:
            config: The configuration dictionary loaded from the instance's config file,
                    validated against the schema in binding_card.yaml.
            resource_manager: The shared resource manager instance.
        """
        # Pass the loaded instance config dictionary to the parent class
        super().__init__(config, resource_manager)

        # Load settings directly from the validated instance config dictionary (self.config)
        # Use .get() with defaults matching the schema defaults for robustness
        self.mode = self.config.get("mode", "ttt")
        self.delay = float(self.config.get("delay", 0.1))
        self.stream_chunk_delay = float(self.config.get("stream_chunk_delay", 0.05))
        self.requires_gpu = bool(self.config.get("requires_gpu", False))
        self.load_delay = float(self.config.get("load_delay", 0.5))
        self.context_size = int(self.config.get("context_size", 4096))
        self.max_output_tokens = int(self.config.get("max_output_tokens", 1024))
        # Use the 'model' setting from the config for the dummy model name
        self.default_model_name = self.config.get("model", "dummy-model")

        # Log initialization with instance name (available from parent class)
        logger.info(f"Initialized DummyBinding instance '{self.binding_instance_name}': Mode='{self.mode}', Delay={self.delay}, GPU_Sim={self.requires_gpu}")

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns a dummy list containing only the configured model name."""
        logger.info(f"Dummy Binding '{self.binding_instance_name}': Listing dummy models.")
        now = datetime.now()
        is_vision = self.mode in ['tti', 'i2i']
        is_audio = self.mode in ['tts', 'ttm', 'stt', 'audio2audio']
        # Return a list containing only the model name specified in the instance config
        return [
            {
                "name": self.default_model_name, # Use name from config
                "size": 500*1024*1024,
                "modified_at": now - timedelta(days=1),
                "quantization_level": "Q_DUMMY",
                "format": "dummy",
                "family": "dummy",
                "context_size": self.context_size,
                "max_output_tokens": self.max_output_tokens,
                "supports_vision": is_vision,
                "supports_audio": is_audio,
                "details": {"info": f"Simulated model '{self.default_model_name}' for instance '{self.binding_instance_name}' (mode: {self.mode})"}
            },
        ]

    # get_binding_config method is removed as metadata is in binding_card.yaml

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on mode."""
        if self.mode in ['tti', 'i2i']: return ['text', 'image']
        if self.mode in ['stt', 'audio2audio']: return ['audio']
        return ['text']

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types based on mode."""
        if self.mode in ['tti', 'i2i']: return ['image']
        if self.mode in ['tts', 'ttm', 'audio2audio']: return ['audio']
        if self.mode in ['ttv']: return ['video'] # Needs dummy video generation
        return ['text']

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return simulated resource requirements based on instance config."""
        return {"gpu_required": self.requires_gpu, "estimated_vram_mb": 512 if self.requires_gpu else 0}

    async def load_model(self, model_name: str) -> bool:
        """Simulates loading a model. Only loads the model specified in instance config."""
        # Only "load" the model name defined in the instance config
        if model_name != self.default_model_name:
            logger.error(f"Dummy Binding '{self.binding_instance_name}': Cannot load requested model '{model_name}', this instance is configured only for '{self.default_model_name}'.")
            return False

        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                 logger.info(f"Dummy Binding '{self.binding_instance_name}': Model '{model_name}' already loaded.")
                 return True
            # If currently loaded model is different (shouldn't happen if check above works, but safety)
            if self._model_loaded:
                 await self.unload_model() # Unload previous within lock

            logger.info(f"Dummy Binding '{self.binding_instance_name}': Simulating load for '{model_name}' (GPU_Sim: {self.requires_gpu})...")
            resource_context = self.resource_manager.acquire_gpu_resource(task_name=f"load_{self.binding_instance_name}") if self.requires_gpu else nullcontext()
            try:
                async with resource_context:
                    if self.requires_gpu: logger.info(f"Dummy Binding '{self.binding_instance_name}': GPU resource acquired for load.")
                    await asyncio.sleep(self.load_delay) # Simulate load time
                    self.model_name = model_name # Set the loaded model name
                    self._model_loaded = True
                    logger.info(f"Dummy Binding '{self.binding_instance_name}': Model '{model_name}' loaded.")
                    return True
            except asyncio.TimeoutError:
                logger.error(f"Dummy Binding '{self.binding_instance_name}': Timeout waiting for GPU resource during load.")
                self.model_name=None; self._model_loaded=False; return False
            except Exception as e:
                 logger.error(f"Dummy Binding '{self.binding_instance_name}': Error simulating load for '{model_name}': {e}", exc_info=True)
                 self.model_name=None; self._model_loaded=False; return False

    async def unload_model(self) -> bool:
        """Simulates unloading a model."""
        async with self._load_lock:
            if not self._model_loaded: return True # Already unloaded
            logger.info(f"Dummy Binding '{self.binding_instance_name}': Simulating unload for '{self.model_name}'...");
            await asyncio.sleep(self.delay / 2) # Simulate unload time
            logger.info(f"Dummy Binding '{self.binding_instance_name}': Model '{self.model_name}' unloaded.")
            self.model_name = None
            self._model_loaded = False
            return True

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        # ... (loading check) ...
        logger.info(f"Dummy Binding '{self.binding_instance_name}': Simulating non-stream generation...")
        await asyncio.sleep(self.delay)
        output_type = self.get_supported_output_modalities()[0]
        output_metadata = {"model_used": self.model_name, "binding_instance": self.binding_instance_name}

        if output_type == "text":
            max_tokens = params.get("max_tokens", 50)
            raw_response_text = f"<think>Thinking about prompt: {prompt[:15]}...</think>\nDummy response to '{prompt[:20]}...' (instance {self.binding_instance_name}). " * (max_tokens // 10)
            raw_response_text = raw_response_text[:max_tokens].strip() # Apply max_tokens before parsing

            # --- ADDED: Parse thoughts ---
            cleaned_text, thoughts = parse_thought_tags(raw_response_text)
            # --------------------------
            output_metadata["usage"] = {"prompt_tokens": 5, "completion_tokens": len(cleaned_text.split()), "total_tokens": 5 + len(cleaned_text.split())} # Dummy usage
            return [{"type": "text", "data": cleaned_text, "thoughts": thoughts, "metadata": output_metadata}]
        # ... (other modes remain the same) ...
        else:
             return [{"type": "error", "data": f"Unknown output mode '{self.mode}'", "metadata": output_metadata}]


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # ... (loading check) ...
        logger.info(f"Dummy Binding '{self.binding_instance_name}': Simulating stream (Mode: {self.mode})...")
        output_type = self.get_supported_output_modalities()[0]
        full_response_text = "" # Track raw text including tags
        accumulated_thoughts = ""
        is_thinking = False

        if output_type == "text":
            max_tokens = params.get("max_tokens", 50)
            words_and_tags = ["<think>Starting thought..."] + [f"word{i}" for i in range(max_tokens // 2)] + ["...thought complete.</think>", "Final"] + [f"word{i}" for i in range(max_tokens//2, max_tokens)]

            for i, word_or_tag in enumerate(words_and_tags):
                await asyncio.sleep(self.stream_chunk_delay)
                # --- Simulate parsing logic ---
                current_text_to_process = word_or_tag + (" " if i < len(words_and_tags) - 1 else "")
                processed_text_chunk = ""
                processed_thoughts_chunk = None

                while current_text_to_process:
                    if is_thinking:
                        end_tag_pos = current_text_to_process.find("</think>")
                        if end_tag_pos != -1:
                            accumulated_thoughts += current_text_to_process[:end_tag_pos]
                            processed_thoughts_chunk = accumulated_thoughts
                            accumulated_thoughts = ""
                            is_thinking = False
                            current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                        else:
                            accumulated_thoughts += current_text_to_process
                            current_text_to_process = ""
                    else:
                        start_tag_pos = current_text_to_process.find("<think>")
                        if start_tag_pos != -1:
                            processed_text_chunk += current_text_to_process[:start_tag_pos]
                            is_thinking = True
                            current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                        else:
                            processed_text_chunk += current_text_to_process
                            current_text_to_process = ""

                # Yield if needed
                if processed_text_chunk or processed_thoughts_chunk:
                    full_response_text += word_or_tag + (" " if i < len(words_and_tags) - 1 else "") # Store raw for final parse
                    yield {
                        "type": "chunk",
                        "content": processed_text_chunk if processed_text_chunk else None,
                        "thoughts": processed_thoughts_chunk
                    }
                # --- End simulation logic ---

            # Final chunk
            final_cleaned_text, final_thoughts_str = parse_thought_tags(full_response_text)
            final_metadata = {
                "model_used": self.model_name,
                "binding_instance": self.binding_instance_name,
                "finish_reason": "completed",
                "usage": {"prompt_tokens": 5, "completion_tokens": len(final_cleaned_text.split()), "total_tokens": 5 + len(final_cleaned_text.split())} # Dummy usage
            }
            final_output_list = [{"type": "text", "data": final_cleaned_text.strip(), "thoughts": final_thoughts_str, "metadata": final_metadata}]
            yield {"type": "final", "content": final_output_list, "metadata": {"status":"complete"}}

        # ... (handle other modes like image/audio - they don't have thoughts) ...
        else:
            logger.warning(f"Dummy stream called in unhandled mode ({self.mode}).")
            result_list = await self.generate(prompt, params, request_info, multimodal_data) # Call non-stream
            yield {"type": "final", "content": result_list, "metadata": {"status":"complete"}}

        logger.info(f"Dummy Binding '{self.binding_instance_name}': Stream finished.")


    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Simulates tokenization by assigning random IDs to words."""
        if not self._model_loaded:
             raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for tokenization")
        logger.info(f"Dummy Binding '{self.binding_instance_name}': Simulating tokenize for '{text[:50]}...'")
        words = text.split()
        tokens = [random.randint(1000, 30000) for _ in words]
        if add_bos: tokens.insert(0, 1) # Simulate BOS token
        if add_eos: tokens.append(2)    # Simulate EOS token
        return tokens

    async def detokenize(self, tokens: List[int]) -> str:
        """Simulates detokenization."""
        if not self._model_loaded:
             raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for detokenization")
        logger.info(f"Dummy Binding '{self.binding_instance_name}': Simulating detokenize for {len(tokens)} tokens.")
        # Ignore BOS/EOS simulation
        words = [f"word{i+1}" for i, token in enumerate(tokens) if token > 2]
        return " ".join(words)

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns dummy info for the loaded model."""
        if not self._model_loaded or not self.model_name:
             return { "name": None, "context_size": self.context_size, "max_output_tokens": self.max_output_tokens, "supports_vision": False, "supports_audio": False, "details": {} }

        is_vision = self.mode in ['tti', 'i2i']
        is_audio = self.mode in ['tts', 'ttm', 'stt', 'audio2audio']
        return {
            "name": self.model_name, # Should match default_model_name from config
            "context_size": self.context_size,
            "max_output_tokens": self.max_output_tokens,
            "supports_vision": is_vision,
            "supports_audio": is_audio,
            "details": {"dummy_info": f"Instance '{self.binding_instance_name}' simulated mode {self.mode}"}
        }