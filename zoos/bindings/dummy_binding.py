# zoos/bindings/dummy_binding.py
import asyncio
import logging
import time
import base64
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
try:
    from lollms_server.api.models import StreamChunk, InputData
except ImportError:
    class StreamChunk: pass # type: ignore
    class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

def generate_dummy_image_base64(width=256, height=256):
    img = Image.new('RGB', (width, height), color = 'black'); buffered = BytesIO()
    img.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
def generate_dummy_audio_base64(): return "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA=="

class DummyBinding(Binding):
    """Dummy binding for testing purposes."""
    binding_type_name = "dummy_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        self.mode = self.config.get("mode", "ttt")
        self.delay = float(self.config.get("delay", 0.1))
        self.stream_chunk_delay = float(self.config.get("stream_chunk_delay", 0.05))
        self.requires_gpu = bool(self.config.get("requires_gpu", False))
        self.load_delay = float(self.config.get("load_delay", 0.5))
        logger.info(f"Initialized DummyBinding: Name='{self.binding_name}', Mode='{self.mode}', GPU={self.requires_gpu}")

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns a dummy list of models."""
        logger.info(f"Dummy Binding '{self.binding_name}': Listing dummy models.")
        now = datetime.now()
        is_vision = self.mode in ['tti', 'i2i']
        is_audio = self.mode in ['tts', 'ttm', 'stt', 'audio2audio']
        return [
            { "name": self.config.get("model", f"dummy_{self.mode}_model"), "size": 500*1024*1024, "modified_at": now - timedelta(days=1), "quantization_level": "Q_DUMMY", "format": "dummy", "family": "dummy", "context_size": 4096, "max_output_tokens": 1024, "supports_vision": is_vision, "supports_audio": is_audio, "details": {"info": f"Simulated model for mode {self.mode}"} },
            { "name": "another_dummy", "size": 1000*1024*1024, "modified_at": now - timedelta(hours=5), "quantization_level": "F_DUMMY", "format": "dummy", "family": "dummy", "context_size": 8192, "max_output_tokens": 2048, "supports_vision": False, "supports_audio": False, "details": {"info": "Another simulated model"} }
        ]

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the Dummy binding."""
        return { "type_name": cls.binding_type_name, "version": "1.1", "description": "Dummy binding for testing.", "supports_streaming": True, "config_template": { "type": cls.binding_type_name, "mode": "ttt", "delay": 0.1, "requires_gpu": False, "load_delay": 0.5 } }

    # --- IMPLEMENTED CAPABILITIES ---
    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on mode."""
        if self.mode in ['tti', 'i2i']: return ['text', 'image']
        if self.mode in ['stt', 'audio2audio']: return ['audio']
        return ['text']

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types based on mode."""
        if self.mode in ['tti', 'i2i']: return ['image']
        if self.mode in ['tts', 'ttm', 'audio2audio']: return ['audio']
        if self.mode in ['ttv']: return ['video']
        return ['text']
    # --- END IMPLEMENTED CAPABILITIES ---

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return simulated resource requirements."""
        return {"gpu_required": self.requires_gpu, "estimated_vram_mb": 512 if self.requires_gpu else 0}

    async def load_model(self, model_name: str) -> bool:
        """Simulates loading a model."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name: return True
            logger.info(f"Dummy '{self.binding_name}': Simulating load for '{model_name}' (GPU: {self.requires_gpu})...")
            acquired_context = self.resource_manager.acquire_gpu_resource(task_name=f"load_{self.binding_name}") if self.requires_gpu else nullcontext()
            try:
                async with acquired_context:
                    if self.requires_gpu: logger.info(f"Dummy '{self.binding_name}': GPU resource acquired.")
                    await asyncio.sleep(self.load_delay); self.model_name = model_name; self._model_loaded = True
                    logger.info(f"Dummy '{self.binding_name}': Model '{model_name}' loaded."); return True
            except asyncio.TimeoutError: logger.error(f"Dummy '{self.binding_name}': Timeout waiting for GPU."); self.model_name=None; self._model_loaded=False; return False
            except Exception as e: logger.error(f"Dummy '{self.binding_name}': Error loading '{model_name}': {e}", exc_info=True); self.model_name=None; self._model_loaded=False; return False

    async def unload_model(self) -> bool:
        """Simulates unloading a model."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Dummy '{self.binding_name}': Simulating unload for '{self.model_name}'..."); await asyncio.sleep(self.delay / 2)
            logger.info(f"Dummy '{self.binding_name}': Model '{self.model_name}' unloaded."); self.model_name = None; self._model_loaded = False; return True

    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> Union[str, Dict[str, Any]]:
        """Simulates generation based on the configured mode."""
        if not self._model_loaded: raise RuntimeError("Model not loaded")
        logger.info(f"Dummy '{self.binding_name}': Simulating gen for prompt: '{prompt[:50]}...' (Mode: {self.mode})")
        if multimodal_data: logger.info(f"Dummy '{self.binding_name}': Received {len(multimodal_data)} multimodal items.") # Log received data

        await asyncio.sleep(self.delay) # Simulate generation time

        output_type = self.get_supported_output_modalities()[0] # Get primary output type

        if output_type == "text":
            max_tokens = params.get("max_tokens", 50); response_text = f"Dummy response to '{prompt[:20]}...' (model {self.model_name}). " * (max_tokens // 10)
            return {"text": response_text[:max_tokens]} # Return consistent dict
        elif output_type == "image":
            width = params.get("width", 256); height = params.get("height", 256); img_b64 = generate_dummy_image_base64(width, height)
            return {"image_base64": img_b64, "mime_type": "image/png", "prompt_used": prompt, "model": self.model_name}
        elif output_type == "audio":
            aud_b64 = generate_dummy_audio_base64(); return {"audio_base64": aud_b64, "mime_type": "audio/wav", "prompt_used": prompt, "model": self.model_name}
        elif output_type == "video":
            return {"video_base64": "", "mime_type": "video/mp4", "prompt_used": prompt, "model": self.model_name} # Placeholder
        else:
            return {"error": f"Unknown output mode '{self.mode}' for DummyBinding"}

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulates streaming generation for TTT/TTS modes."""
        if not self._model_loaded: yield {"type": "error", "content": "Model not loaded"}; return
        if multimodal_data: logger.info(f"Dummy '{self.binding_name}' (stream): Received {len(multimodal_data)} multimodal items.")

        output_type = self.get_supported_output_modalities()[0]
        if output_type not in ["text", "audio"]: # Only simulate stream for text/audio
            logger.warning(f"Dummy stream called in non-streamable mode ({self.mode}). Simulating non-stream.")
            result = await self.generate(prompt, params, request_info, multimodal_data)
            yield {"type": "final", "content": result, "metadata": {"binding": self.binding_name}}; return

        logger.info(f"Dummy '{self.binding_name}': Simulating stream (Mode: {self.mode}) for prompt: '{prompt[:50]}...'")
        full_response_content: Union[str, Dict] = "" # Adjusted type

        if output_type == "text":
            max_tokens = params.get("max_tokens", 50); words = [f"word{i}" for i in range(max_tokens)]
            text_accumulator = ""
            for i, word in enumerate(words):
                await asyncio.sleep(self.stream_chunk_delay); chunk_content = word + (" " if i < len(words) - 1 else ""); text_accumulator += chunk_content
                yield {"type": "chunk", "content": chunk_content, "metadata": {"index": i}}
            full_response_content = {"text": text_accumulator} # Final content is dict
            yield {"type": "final", "content": full_response_content, "metadata": {"total_words": len(words)}}
        elif output_type == "audio":
             audio_data = generate_dummy_audio_base64()
             # Simulate audio stream: send metadata first, then data, then final
             yield {"type": "info", "content": {"mime_type": "audio/wav", "status": "starting"}} # Example info chunk
             await asyncio.sleep(self.stream_chunk_delay)
             yield {"type": "chunk", "content": {"audio_chunk_base64": audio_data[:len(audio_data)//2] }} # Example data chunk
             await asyncio.sleep(self.stream_chunk_delay)
             yield {"type": "chunk", "content": {"audio_chunk_base64": audio_data[len(audio_data)//2:] }}
             full_response_content = {"audio_base64": audio_data, "mime_type": "audio/wav"} # Final content dict
             yield {"type": "final", "content": full_response_content, "metadata": {"prompt_used": prompt}}

        logger.info(f"Dummy Binding '{self.binding_name}': Stream finished.")