# examples/bindings/dummy_binding.py
import asyncio
import logging
import time
import base64
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta # Import datetime

# Assuming lollms_server is installed or project root is in PYTHONPATH
from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.api.models import StreamChunk # Use if needed for formatting

logger = logging.getLogger(__name__)

# Dummy data generation
def generate_dummy_image_base64(width=256, height=256):
    # Creates a simple black PNG as base64
    from io import BytesIO
    from PIL import Image
    img = Image.new('RGB', (width, height), color = 'black')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_dummy_video_base64():
    # Placeholder - return minimal data or empty string
    return ""

def generate_dummy_music_base64():
    # Placeholder - return minimal data or empty string
    return ""


class DummyBinding(Binding):
    """
    A dummy binding implementation for testing purposes.
    It simulates different generation modes (TTT, TTI, TTV, TTM)
    without using actual models.
    """
    binding_type_name = "dummy_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        self.mode = self.config.get("mode", "ttt") # Configurable mode via config.toml
        self.delay = float(self.config.get("delay", 0.1)) # Simulate processing time
        self.stream_chunk_delay = float(self.config.get("stream_chunk_delay", 0.05))
        self.requires_gpu = bool(self.config.get("requires_gpu", False)) # Simulate resource need
        self.load_delay = float(self.config.get("load_delay", 0.5)) # Simulate load time
        logger.info(f"Initialized DummyBinding: Name='{self.binding_name}', Mode='{self.mode}', Delay={self.delay}, RequiresGPU={self.requires_gpu}, LoadDelay={self.load_delay}")

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns a dummy list of models with some standardized fields."""
        logger.info(f"Dummy Binding '{self.binding_name}': Listing dummy models.")
        now = datetime.now()
        model_list = [
            {
                "name": self.config.get("model", f"dummy_{self.mode}_model"),
                "size": 1024 * 1024 * 500, # Dummy size 500MB
                "modified_at": now - timedelta(days=1),
                "quantization_level": "Q_DUMMY",
                "format": "dummy",
                "family": "dummy_family",
                # --- Add dummy values ---
                "context_size": 4096,
                "max_output_tokens": 1024,
                "details": {"info": "Simulated model for dummy binding", "mode": self.mode}
             },
            {
                "name": "another_dummy_model",
                "size": 1024 * 1024 * 1000, # 1GB
                "modified_at": now - timedelta(hours=5),
                "quantization_level": "F_DUMMY",
                "format": "dummy",
                "family": "dummy_family",
                # --- Add dummy values ---
                "context_size": 8192,
                "max_output_tokens": 2048,
                "details": {"info": "Another simulated model"}
            }
        ]
        return model_list
    
    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the Dummy binding."""
        return {
            "type_name": cls.binding_type_name,
            "version": "1.0",
            "description": "A dummy binding for testing. Simulates TTT, TTI, TTV, TTM.",
            "supports_streaming": True,
            "config_template": {
                "type": cls.binding_type_name,
                "mode": "ttt", # 'ttt', 'tti', 'ttv', 'ttm'
                "delay": 0.1, # Simulate generation time per call/chunk
                "requires_gpu": False, # Simulate if loading needs GPU resource
                "load_delay": 0.5 # Simulate model load time
            }
        }

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return simulated resource requirements."""
        return {"gpu_required": self.requires_gpu, "estimated_vram_mb": 512 if self.requires_gpu else 0}


    async def load_model(self, model_name: str) -> bool:
        """Simulates loading a model."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                logger.debug(f"DummyBinding '{self.binding_name}': Model '{model_name}' already loaded.")
                return True

            logger.info(f"DummyBinding '{self.binding_name}': Simulating load for model '{model_name}' (Mode: {self.mode}, Requires GPU: {self.requires_gpu})...")

            # --- Simulate Resource Acquisition ---
            acquired_context = nullcontext()
            if self.requires_gpu and self.resource_manager:
                    logger.info(f"DummyBinding '{self.binding_name}': Acquiring GPU resource...")
                    try:
                        acquired_context = self.resource_manager.acquire_gpu_resource(
                            task_name=f"load_dummy_{self.binding_name}_{model_name}"
                        )
                    except Exception as e:
                        logger.error(f"DummyBinding '{self.binding_name}': Failed to create acquire context: {e}")
                        return False # Should not happen with nullcontext or semaphore usually

            try:
                async with acquired_context:
                    if self.requires_gpu: logger.info(f"DummyBinding '{self.binding_name}': GPU resource acquired.")
                    # Simulate load time
                    await asyncio.sleep(self.load_delay)
                    self.model_name = model_name
                    self._model_loaded = True
                    logger.info(f"DummyBinding '{self.binding_name}': Model '{model_name}' loaded successfully.")
                    return True
            except asyncio.TimeoutError:
                logger.error(f"DummyBinding '{self.binding_name}': Timeout waiting for GPU resource to load model '{model_name}'.")
                self.model_name = None
                self._model_loaded = False
                return False
            except Exception as e:
                logger.error(f"DummyBinding '{self.binding_name}': Error during simulated model loading '{model_name}': {e}", exc_info=True)
                self.model_name = None
                self._model_loaded = False
                return False

    async def unload_model(self) -> bool:
        """Simulates unloading a model."""
        async with self._load_lock:
            if not self._model_loaded:
                logger.debug(f"DummyBinding '{self.binding_name}': No model loaded, unload successful.")
                return True
            logger.info(f"DummyBinding '{self.binding_name}': Simulating unload for model '{self.model_name}'...")
            await asyncio.sleep(self.delay / 2) # Simulate unload time
            logger.info(f"DummyBinding '{self.binding_name}': Model '{self.model_name}' unloaded.")
            self.model_name = None
            self._model_loaded = False
            return True


    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any] # Contains personality, functions etc.
    ) -> Union[str, Dict[str, Any]]:
        """Simulates generation based on the configured mode."""
        if not self._model_loaded:
                logger.error(f"DummyBinding '{self.binding_name}': Cannot generate, model '{self.model_name}' not loaded.")
                raise RuntimeError("Model not loaded") # Or return error dict/string?

        logger.info(f"DummyBinding '{self.binding_name}': Simulating generation for prompt: '{prompt[:50]}...' (Mode: {self.mode})")
        await asyncio.sleep(self.delay) # Simulate generation time

        if self.mode == "ttt":
            max_tokens = params.get("max_tokens", 50)
            response_text = f"Dummy response to '{prompt[:20]}...' using model {self.model_name}. " * (max_tokens // 10)
            return response_text[:max_tokens]

        elif self.mode == "tti":
                width = params.get("width", 256)
                height = params.get("height", 256)
                logger.info(f"DummyBinding '{self.binding_name}': Generating dummy image ({width}x{height}).")
                img_b64 = generate_dummy_image_base64(width, height)
                return {"image_base64": img_b64, "prompt_used": prompt, "model": self.model_name}

        elif self.mode == "ttv":
                logger.info(f"DummyBinding '{self.binding_name}': Generating dummy video.")
                vid_b64 = generate_dummy_video_base64()
                return {"video_base64": vid_b64, "prompt_used": prompt, "model": self.model_name}

        elif self.mode == "ttm":
                logger.info(f"DummyBinding '{self.binding_name}': Generating dummy music.")
                mus_b64 = generate_dummy_music_base64()
                return {"audio_base64": mus_b64, "prompt_used": prompt, "model": self.model_name}

        else:
            return f"Error: Unknown mode '{self.mode}' for DummyBinding '{self.binding_name}'"


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any] # Contains personality, functions etc.
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulates streaming generation for TTT mode."""
        if not self._model_loaded:
                logger.error(f"DummyBinding '{self.binding_name}': Cannot generate stream, model '{self.model_name}' not loaded.")
                # Yield an error chunk?
                yield {"type": "error", "content": "Model not loaded", "metadata": {"binding": self.binding_name}}
                return # Stop generation

        if self.mode != "ttt":
            logger.warning(f"DummyBinding '{self.binding_name}': Streaming called in non-TTT mode ({self.mode}). Simulating non-stream result.")
            result = await self.generate(prompt, params, request_info)
            yield {"type": "chunk", "content": result} # Send the whole dict/error string as one chunk
            yield {"type": "final", "content": result, "metadata": {"binding": self.binding_name}}
            return

        logger.info(f"DummyBinding '{self.binding_name}': Simulating stream generation for prompt: '{prompt[:50]}...'")
        max_tokens = params.get("max_tokens", 50)
        words = [f"word{i}" for i in range(max_tokens)]
        full_response = ""

        for i, word in enumerate(words):
            await asyncio.sleep(self.stream_chunk_delay)
            chunk_content = word + (" " if i < len(words) - 1 else "")
            full_response += chunk_content
            yield {
                "type": "chunk",
                "content": chunk_content,
                "metadata": {"index": i, "binding": self.binding_name}
            }

        # Yield final message
        yield {
            "type": "final",
            "content": full_response,
            "metadata": {"total_words": len(words), "binding": self.binding_name}
        }
        logger.info(f"DummyBinding '{self.binding_name}': Stream finished.")