# encoding:utf-8
# Project: lollms_server
# File: zoos/bindings/hf_binding/__init__.py
# Author: lollms_server Team
# Date: 2025-05-01
# Description: Binding implementation for local Hugging Face Transformers models.

import asyncio
import sys
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from io import BytesIO
import threading
import importlib.util

# --- Dependency Check ---
# Use pipmaster if needed, but primarily rely on requirements from card
import pipmaster as pm
pm.ensure_packages(["transformers","huggingface_hub", "accelerate", "pillow", "bitsandbytes", "flash_attn"])


# --- Core Library Imports ---
try:
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        TextIteratorStreamer,
        GenerationConfig,
        BitsAndBytesConfig
    )
    from huggingface_hub import HfApi, hf_hub_download, list_models, ModelFilter
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from PIL import Image
    hf_installed = True
    torch_installed = True
except ImportError as e:
    # Mock imports if core libraries are missing
    torch = None
    transformers = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoProcessor = None
    TextIteratorStreamer = None
    GenerationConfig = None
    BitsAndBytesConfig = None # type: ignore
    HfApi = None
    hf_hub_download = None
    list_models = None
    ModelFilter = None # type: ignore
    RepositoryNotFoundError = Exception # type: ignore
    GatedRepoError = Exception # type: ignore
    init_empty_weights = None # type: ignore
    load_checkpoint_and_dispatch = None # type: ignore
    Image = None # type: ignore
    BytesIO = None # type: ignore
    hf_installed = False
    torch_installed = False
    _import_error_msg = str(e)

# --- Optional Dependency Checks ---
try:
    import bitsandbytes
    bitsandbytes_installed = True
except ImportError:
    bitsandbytes = None # type: ignore
    bitsandbytes_installed = False
try:
    import flash_attn
    flash_attn_installed = True
except ImportError:
    flash_attn = None # type: ignore
    flash_attn_installed = False

# --- Lollms Imports ---
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.core.config import get_server_root
from lollms_server.utils.helpers import parse_thought_tags # --- ADDED HELPER IMPORT ---

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData
    except ImportError:
        class StreamChunk: pass
        class InputData: pass # type: ignore

logger = logging.getLogger(__name__)

# === Constants ===
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1

class HuggingFaceBinding(Binding):
    """Binding for local models using Hugging Face Transformers library."""
    binding_type_name = "hf_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the HuggingFaceBinding."""
        super().__init__(config, resource_manager)

        if not hf_installed or not torch_installed:
            raise ImportError(f"HF binding requires 'torch' and 'transformers'. Error: {_import_error_msg}")

        self.model_identifier = self.config.get("model_name_or_path")
        if not self.model_identifier:
            raise ValueError(f"Missing required 'model_name_or_path' in HF config for '{self.binding_instance_name}'.")

        # --- Configuration Parameters ---
        self.device_str = self.config.get("device", "auto").lower()
        self.use_fp16 = self.config.get("use_fp16", True)
        self.use_bf16 = self.config.get("use_bf16", False)
        self.quantization_str = self.config.get("quantization", "none").lower()
        self.use_safetensors = self.config.get("use_safetensors", True)
        self.use_flash_attention_2 = self.config.get("use_flash_attention_2", False)
        self.trust_remote_code = self.config.get("trust_remote_code", False)

        # Default generation parameters from config
        self.default_max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.default_temperature = self.config.get("temperature", DEFAULT_TEMPERATURE)
        self.default_top_k = self.config.get("top_k", DEFAULT_TOP_K)
        self.default_top_p = self.config.get("top_p", DEFAULT_TOP_P)
        self.default_repeat_penalty = self.config.get("repeat_penalty", DEFAULT_REPETITION_PENALTY)

        if self.trust_remote_code:
            logger.warning(f"**SECURITY RISK** Instance '{self.binding_instance_name}': 'trust_remote_code' is enabled.")

        # --- Internal State ---
        self.model: Optional[transformers.PreTrainedModel] = None
        self.tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None
        self.processor: Optional[transformers.processing_utils.ProcessorMixin] = None
        self.device: Optional[torch.device] = None
        self.torch_dtype: Optional[torch.dtype] = None
        self.loaded_model_identifier: Optional[str] = None
        self.model_supports_vision: bool = False

        # Determine compute device and data type
        self._determine_device_and_dtype()

        logger.info(
            f"Initialized HF Binding '{self.binding_instance_name}': "
            f"Model='{self.model_identifier}', Device='{self.device}', DType='{self.torch_dtype}', "
            f"Quant='{self.quantization_str}'"
        )

    def _determine_device_and_dtype(self):
        """Sets the torch device and dtype based on config and hardware availability."""
        # Determine device
        if self.device_str == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Check for MPS (Apple Silicon)
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif self.device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            if self.device_str not in ["cpu", "auto"]:
                logger.warning(f"Requested device '{self.device_str}' not available. Falling back to CPU.")
            self.device = torch.device("cpu")

        # Determine dtype based on device and config
        if self.device.type != 'cpu':
            # Use bfloat16 if requested and supported (CUDA Ampere+ or MPS)
            if self.use_bf16 and \
               ((self.device.type == 'cuda' and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) or \
                self.device.type == 'mps'): # MPS seems to support bf16 reasonably well
                self.torch_dtype = torch.bfloat16
                logger.info("Using bfloat16 precision.")
            # Use float16 if requested (and bf16 not chosen)
            elif self.use_fp16:
                self.torch_dtype = torch.float16
                logger.info("Using float16 precision.")
            # Default to float32 on GPU if no precision specified
            else:
                self.torch_dtype = torch.float32
                logger.info("Using float32 precision.")
        else: # CPU
            self.torch_dtype = torch.float32
            if self.use_fp16 or self.use_bf16:
                logger.warning("fp16/bf16 not recommended on CPU. Using float32.")
            logger.info("Using CPU device with float32 precision.")

    # --- Binding Capabilities ---

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types (text, potentially image)."""
        modalities = ['text']
        if self._model_loaded and self.model_supports_vision:
            modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types (currently only text)."""
        return ['text']

    # --- Model Listing & Loading ---

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists potentially relevant models from Hugging Face Hub."""
        logger.info(f"HF Binding '{self.binding_instance_name}': Listing models from Hub...")
        if not HfApi:
            logger.warning("huggingface_hub library not available, cannot list models.")
            return []

        try:
            # Run synchronous Hub call in a separate thread
            def get_models():
                api = HfApi()
                # Fetch top text generation models
                text_models = api.list_models(
                    filter=ModelFilter(task="text-generation"),
                    sort="downloads", direction=-1, limit=50
                )
                # Fetch top vision models (often have text generation capabilities)
                vision_models = api.list_models(
                    filter=ModelFilter(task="image-to-text"), # A common task for vision+text models
                    sort="downloads", direction=-1, limit=20
                )
                return list(text_models) + list(vision_models)

            hub_models = await asyncio.to_thread(get_models)
            formatted_models = []
            seen_ids = set()

            for model_info in hub_models:
                if model_info.modelId in seen_ids:
                    continue
                seen_ids.add(model_info.modelId)

                # Heuristics to guess vision support from tags/name
                tags = getattr(model_info, 'tags', [])
                is_vision = any(tag in tags for tag in ['llava', 'vision', 'image-to-text', 'visual']) or \
                            any(kw in model_info.modelId.lower() for kw in ['vision', 'llava', 'bakllava', 'idefics', 'phi-3-vision'])

                formatted_models.append({
                    "name": model_info.modelId,
                    "size": None, # Size often not directly available, requires more API calls/downloads
                    "modified_at": model_info.lastModified,
                    "format": "transformers", # Indicates HF Transformers format
                    "family": None, # Difficult to determine reliably
                    "context_size": None, # Requires loading config
                    "supports_vision": is_vision,
                    "supports_audio": False, # Audio support not handled here
                    "details": {
                        "hub_tags": tags,
                        "hub_pipeline": getattr(model_info, 'pipeline_tag', None),
                        "hub_author": getattr(model_info, 'author', None)
                    }
                })

            logger.info(f"HF Binding '{self.binding_instance_name}': Found {len(formatted_models)} potential models on Hub.")
            return formatted_models

        except Exception as e:
            logger.error(f"Error listing models from Hub for instance '{self.binding_instance_name}': {e}", exc_info=True)
            return []

    async def health_check(self) -> Tuple[bool, str]:
        """Checks if core libraries are installed and the configured device is accessible."""
        if not torch_installed:
            return False, "PyTorch not installed."
        if not hf_installed:
            return False, "Transformers not installed."

        try:
            # Check device availability
            if self.device and self.device.type == 'cuda':
                if not torch.cuda.is_available():
                    return False, "CUDA specified but not available."
                torch.cuda.get_device_name(0) # Test CUDA interaction
            elif self.device and self.device.type == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return False, "MPS specified but not available."
                torch.tensor([1], device=self.device) # Test MPS interaction
            elif self.device and self.device.type == 'cpu':
                pass # CPU is always available
            else:
                return False, f"Configured device '{self.device}' not recognized or available."

            # Check optional dependencies based on config
            if self.quantization_str in ["4bit", "8bit"] and not bitsandbytes_installed:
                return False, f"Quantization '{self.quantization_str}' requested but 'bitsandbytes' not installed."
            if self.use_flash_attention_2 and not flash_attn_installed:
                # Also check CUDA capability for Flash Attention
                cuda_ok_for_flash = False
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability(0)
                    cuda_ok_for_flash = capability[0] >= 8 # Requires Ampere or newer
                if not flash_attn_installed or not cuda_ok_for_flash:
                    return False, f"Flash Attention 2 requested but 'flash-attn' not installed or GPU not compatible (Ampere+ required)."

            # If all checks pass
            return True, f"Torch ({torch.__version__}) & Transformers ({transformers.__version__}) available. Device '{self.device}' OK."

        except Exception as e:
            logger.error(f"HF health check failed for instance '{self.binding_instance_name}': {e}", exc_info=True)
            return False, f"Health check failed: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns estimated resource requirements (GPU needed if not CPU)."""
        return {
            "gpu_required": self.device is not None and self.device.type != 'cpu',
            "estimated_vram_mb": 0 # Accurate estimation is very difficult
        }

    async def load_model(self, model_name_or_id: str) -> bool:
        """Loads the specified Hugging Face model, tokenizer, and potentially processor."""
        target_model_id = model_name_or_id

        async with self._load_lock:
            # Check if already loaded
            if self._model_loaded and self.loaded_model_identifier == target_model_id:
                logger.info(f"HF Binding '{self.binding_instance_name}': Model '{target_model_id}' already loaded.")
                return True
            # If another model is loaded, unload it first
            elif self._model_loaded:
                logger.info(f"HF Binding '{self.binding_instance_name}': Switching model. Unloading '{self.loaded_model_identifier}' first...")
                await self.unload_model()

            logger.info(f"HF Binding '{self.binding_instance_name}': Loading model '{target_model_id}'...")
            if not self.device or not self.torch_dtype:
                logger.error("Device or dtype not determined before loading.")
                return False

            # --- Prepare loading arguments ---
            load_kwargs: Dict[str, Any] = {
                "pretrained_model_name_or_path": target_model_id,
                "torch_dtype": self.torch_dtype,
                "use_safetensors": self.use_safetensors,
                "trust_remote_code": self.trust_remote_code,
                 # Let HF distribute across GPUs if available and no specific device requested
                "device_map": "auto" if self.device.type == 'cuda' else self.device
            }

            # --- Quantization ---
            if self.quantization_str == "4bit":
                if not bitsandbytes_installed:
                    logger.warning("4-bit quantization requested but bitsandbytes not found. Loading without quantization.")
                else:
                    logger.info("Applying 4-bit quantization.")
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.torch_dtype, # Use configured dtype for computation
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4" # Recommended type
                    )
            elif self.quantization_str == "8bit":
                if not bitsandbytes_installed:
                    logger.warning("8-bit quantization requested but bitsandbytes not found. Loading without quantization.")
                else:
                    logger.info("Applying 8-bit quantization.")
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            # --- Flash Attention 2 ---
            if self.use_flash_attention_2:
                if flash_attn_installed and self.device and self.device.type == 'cuda':
                    capability = torch.cuda.get_device_capability(0)
                    if capability[0] >= 8: # Requires Ampere or newer
                        load_kwargs["attn_implementation"] = "flash_attention_2"
                        logger.info("Enabling Flash Attention 2.")
                    else:
                         logger.warning("Flash Attention 2 requested but GPU architecture is not supported (Ampere+ required).")
                else:
                    logger.warning("Flash Attention 2 requested but 'flash-attn' not installed or not using CUDA.")

            # --- Resource Acquisition ---
            needs_gpu = self.device.type != 'cpu'
            resource_context = self.resource_manager.acquire_gpu_resource(f"load_{self.binding_instance_name}_{target_model_id.replace('/','_')}") \
                               if needs_gpu else nullcontext()

            # --- Load Process ---
            try:
                async with resource_context:
                    if needs_gpu:
                        logger.info(f"HF '{self.binding_instance_name}': GPU resource acquired for model load.")

                    # Load Processor (handles text+image) or Tokenizer (text only)
                    logger.info("Loading tokenizer and processor (if applicable)...")
                    try:
                        # Try loading AutoProcessor first (for multimodal models)
                        self.processor = await asyncio.to_thread(
                            AutoProcessor.from_pretrained, target_model_id, trust_remote_code=self.trust_remote_code
                        )
                        # If processor has a tokenizer attribute, use it
                        self.tokenizer = getattr(self.processor, 'tokenizer', None)
                        if not self.tokenizer:
                            # If processor doesn't have tokenizer, load it separately
                            self.tokenizer = await asyncio.to_thread(
                                AutoTokenizer.from_pretrained, target_model_id, trust_remote_code=self.trust_remote_code
                            )
                        logger.info("Processor (and possibly Tokenizer) loaded.")
                        # Assume vision support if processor isn't just a wrapper around a tokenizer
                        self.model_supports_vision = not isinstance(self.processor, transformers.PreTrainedTokenizerBase)
                    except Exception as proc_err:
                        # Fallback to just loading tokenizer if processor fails
                        logger.warning(f"Could not load AutoProcessor for '{target_model_id}': {proc_err}. Falling back to AutoTokenizer.")
                        self.processor = None
                        self.tokenizer = await asyncio.to_thread(
                            AutoTokenizer.from_pretrained, target_model_id, trust_remote_code=self.trust_remote_code
                        )
                        self.model_supports_vision = False
                        if not self.tokenizer:
                             # If even tokenizer fails, it's a critical error
                            raise RuntimeError("Failed to load Tokenizer.") from proc_err

                    # Load Model Weights
                    logger.info("Loading model weights...")
                    # Use asyncio.to_thread to run the blocking from_pretrained call
                    self.model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, **load_kwargs)

                    if not self.model:
                        raise RuntimeError("Model loading returned None.")

                    # Set model to evaluation mode
                    await asyncio.to_thread(self.model.eval)

                    # Update state
                    self.loaded_model_identifier = target_model_id
                    self.model_name = target_model_id # Use the loaded ID as the current name
                    self._model_loaded = True

                    logger.info(
                        f"HF model '{target_model_id}' loaded successfully onto device '{self.device}' "
                        f"with dtype '{self.torch_dtype}'. Vision support: {self.model_supports_vision}"
                    )
                    return True

            except asyncio.TimeoutError:
                logger.error(f"HF '{self.binding_instance_name}': Timeout waiting for GPU to load model {target_model_id}")
                await self.unload_model() # Attempt cleanup
                return False
            except (RepositoryNotFoundError, OSError) as e:
                # Handle model not found errors (local path or hub)
                logger.error(f"HF '{self.binding_instance_name}': Model not found at '{target_model_id}'. Error: {e}")
                await self.unload_model()
                return False
            except GatedRepoError as e:
                # Handle access denied for private/gated models
                logger.error(f"HF '{self.binding_instance_name}': Access denied for gated model '{target_model_id}'. Ensure token is set. Error: {e}")
                await self.unload_model()
                return False
            except Exception as e:
                # Catch-all for other loading errors
                logger.error(f"HF '{self.binding_instance_name}': Failed to load model '{target_model_id}': {e}", exc_info=True)
                trace_exception(e)
                await self.unload_model()
                return False
            # `finally` block not strictly needed as resource_context handles release

    async def unload_model(self) -> bool:
        """Unloads the model, tokenizer, processor and clears GPU cache."""
        async with self._load_lock:
            if not self._model_loaded:
                logger.info(f"HF '{self.binding_instance_name}': No model loaded.")
                return True

            logger.info(f"HF '{self.binding_instance_name}': Unloading model '{self.loaded_model_identifier}'...")
            try:
                # Delete references to allow garbage collection
                del self.model
                del self.tokenizer
                del self.processor
                self.model = None
                self.tokenizer = None
                self.processor = None

                # Reset state variables
                self.loaded_model_identifier = None
                self.model_name = None
                self._model_loaded = False
                self.model_supports_vision = False

                # Clear GPU memory cache if applicable
                if self.device and self.device.type == 'cuda':
                    logger.info("Clearing CUDA cache...")
                    await asyncio.to_thread(torch.cuda.empty_cache)
                elif self.device and self.device.type == 'mps':
                    if hasattr(torch.mps, "empty_cache"): # Check if available
                        await asyncio.to_thread(torch.mps.empty_cache)

                logger.info(f"HF model unloaded successfully for instance '{self.binding_instance_name}'.")
                return True
            except Exception as e:
                # Log error but still attempt to reset state
                logger.error(f"Error during HF unload for '{self.binding_instance_name}': {e}", exc_info=True)
                trace_exception(e)
                self.model = None
                self.tokenizer = None
                self.processor = None
                self.loaded_model_identifier = None
                self.model_name = None
                self._model_loaded = False
                self.model_supports_vision = False
                return False # Indicate unload encountered an error

    def _prepare_hf_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps lollms parameters to Hugging Face `generate()` arguments."""
        gen_kwargs: Dict[str, Any] = {}

        # Core sampling parameters
        gen_kwargs["max_new_tokens"] = params.get("max_tokens", self.default_max_tokens)
        gen_kwargs["temperature"] = params.get("temperature", self.default_temperature)
        gen_kwargs["top_k"] = params.get("top_k", self.default_top_k)
        gen_kwargs["top_p"] = params.get("top_p", self.default_top_p)
        gen_kwargs["repetition_penalty"] = params.get("repeat_penalty", self.default_repeat_penalty)

        # Handle stop sequences (Note: Direct support in HF `generate` is limited/complex)
        stop = params.get("stop_sequences") or params.get("stop")
        if stop:
            # HF generate doesn't have a simple list of stop strings.
            # It uses stopping criteria objects, which are harder to map directly.
            # We log a warning and ignore them for now. Post-processing might be needed.
            logger.warning("Stop sequences provided but not directly supported by basic HF generate. Ignoring.")
            # Example: If tokenizer available, could convert stop strings to token IDs:
            # if self.tokenizer:
            #     stop_token_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in stop]
            #     # ... need to integrate these into a StoppingCriteria object ...

        # Control sampling behavior
        # Do sample if temperature > 0 (or if top_k/top_p imply it)
        gen_kwargs["do_sample"] = params.get("temperature", self.default_temperature) > 0

        # Set EOS and PAD token IDs
        if self.tokenizer:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            # Use EOS token ID for padding if PAD token is not set
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
                                         else gen_kwargs.get("eos_token_id")

        logger.debug(f"Prepared HF Generate Params: {gen_kwargs}")
        return gen_kwargs

    async def _prepare_hf_inputs(
        self, prompt: str, multimodal_data: Optional[List['InputData']]
    ) -> Dict[str, Any]:
        """Prepares the input dictionary for model.generate() using tokenizer or processor."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")

        input_args = {}
        image_inputs = None # Flag if images were processed

        # --- Handle Multimodal Input (if supported and processor available) ---
        if self.model_supports_vision and self.processor and multimodal_data:
            image_items = [item for item in multimodal_data if item.type == 'image']
            if image_items:
                if not Image or not BytesIO:
                    logger.error("Cannot process images: Pillow library not found.")
                else:
                    pil_images = []
                    # Decode base64 images
                    for item in image_items:
                        if item.data and isinstance(item.data, str):
                            try:
                                img_bytes = base64.b64decode(item.data)
                                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                                pil_images.append(img)
                            except Exception as img_e:
                                logger.error(f"Failed to decode/load image for HF input (role: {item.role}): {img_e}")

                    # If images successfully loaded, use the processor
                    if pil_images:
                        logger.info(f"Processing {len(pil_images)} image(s) with HF processor...")
                        # Run synchronous processing in a thread
                        def process_sync():
                            # Processor typically handles combining text and images
                            return self.processor(text=prompt, images=pil_images, return_tensors="pt")
                        input_args = await asyncio.to_thread(process_sync)
                        image_inputs = True # Mark that images were processed

                        # Close PIL images after processing
                        for img in pil_images:
                            if hasattr(img, 'close'):
                                try: img.close()
                                except Exception: pass # Ignore close errors


        # --- Handle Text Input (if no images processed or no vision support) ---
        if not image_inputs: # If processor wasn't used (or no images)
            # Tokenize the text prompt
            def tokenize_sync():
                return self.tokenizer(prompt, return_tensors="pt")
            input_args = await asyncio.to_thread(tokenize_sync)

        # Move inputs to the target device
        if self.device:
            input_args = {k: v.to(self.device) for k, v in input_args.items()}

        return input_args

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Return List[OutputData]-like structure
        """Generates output using the loaded Hugging Face model (non-streaming)."""
        if not self._model_loaded or not self.model or not self.tokenizer:
            raise RuntimeError(f"Model not loaded properly in HF binding instance '{self.binding_instance_name}'.")

        logger.info(f"HF Binding '{self.binding_instance_name}': Generating non-stream with '{self.loaded_model_identifier}'...")

        # Prepare parameters and inputs
        generation_kwargs = self._prepare_hf_generation_params(params)
        input_dict = await self._prepare_hf_inputs(prompt, multimodal_data)

        # Set random seed if provided
        seed = params.get("seed")
        if isinstance(seed, int) and seed != -1:
            torch.manual_seed(seed)
        else:
            seed = torch.seed() # Get the seed actually used

        # Acquire GPU resource if needed
        needs_gpu = self.device is not None and self.device.type != 'cpu'
        resource_context = self.resource_manager.acquire_gpu_resource(f"generate_{self.binding_instance_name}") \
                           if needs_gpu else nullcontext()

        try:
            async with resource_context:
                if needs_gpu:
                    logger.info(f"HF '{self.binding_instance_name}': GPU resource acquired for generation.")

                # Run synchronous generation in a thread
                def generate_sync():
                    with torch.no_grad(): # Disable gradient calculations for inference
                        # Ensure inputs are on the same device as the model
                        model_device = next(self.model.parameters()).device
                        inputs_on_device = {k: v.to(model_device) for k, v in input_dict.items()}

                        # Perform generation
                        output_ids = self.model.generate(**inputs_on_device, **generation_kwargs)

                        # Handle potential batch dimension in output_ids (usually shape [1, N])
                        if output_ids.ndim > 1 and output_ids.shape[0] == 1:
                            output_ids = output_ids[0] # Remove batch dimension
                        return output_ids

                logger.info("Starting HF model generation...")
                output_ids = await asyncio.to_thread(generate_sync)
                logger.info("HF model generation finished.")

            # Decode the generated tokens, skipping input tokens
            input_token_length = input_dict["input_ids"].shape[1]
            # Decode only the newly generated tokens
            raw_output_text = self.tokenizer.decode(output_ids[input_token_length:], skip_special_tokens=True)

            # --- ADDED: Parse thoughts ---
            cleaned_output_text, thoughts = parse_thought_tags(raw_output_text)
            # --------------------------

            # Determine finish reason and usage stats
            num_output_tokens = len(output_ids) - input_token_length
            # If max_new_tokens was reached
            finish_reason = "length" if num_output_tokens >= generation_kwargs['max_new_tokens'] else "stop"

            output_metadata = {
                "model_used": self.loaded_model_identifier,
                "binding_instance": self.binding_instance_name,
                "seed_used": seed,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": input_token_length,
                    "completion_tokens": num_output_tokens,
                    "total_tokens": len(output_ids) # Total tokens including prompt
                }
            }

            # Return standardized list format
            return [{
                "type": "text",
                "data": cleaned_output_text.strip(),
                "thoughts": thoughts,
                "metadata": output_metadata
            }]

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"HF '{self.binding_instance_name}': CUDA OOM during generation: {e}")
            trace_exception(e)
            raise RuntimeError("CUDA Out of Memory.") from e
        except asyncio.TimeoutError:
            logger.error(f"HF '{self.binding_instance_name}': Timeout waiting for GPU resource for generation.")
            raise RuntimeError("Timeout waiting for GPU resource.") from e
        except Exception as e:
            logger.error(f"HF '{self.binding_instance_name}': Unexpected error during generation: {e}", exc_info=True)
            trace_exception(e)
            raise RuntimeError(f"Unexpected error during generation: {e}") from e
        # `finally` block not needed as resource_context handles release

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Generates text using the Hugging Face model with streaming."""
        if not self._model_loaded or not self.model or not self.tokenizer or not TextIteratorStreamer:
            yield {"type": "error", "content": f"Model/Tokenizer/Streamer not loaded properly in HF binding instance '{self.binding_instance_name}'."}
            return

        logger.info(f"HF Binding '{self.binding_instance_name}': Generating stream with '{self.loaded_model_identifier}'...")

        # Initialize streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Prepare parameters and inputs
        generation_kwargs = self._prepare_hf_generation_params(params)
        input_dict = await self._prepare_hf_inputs(prompt, multimodal_data)

        # Set random seed
        seed = params.get("seed")
        if isinstance(seed, int) and seed != -1:
            torch.manual_seed(seed)
        else:
            seed = torch.seed()

        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=self._run_generation_in_thread,
            args=(input_dict, generation_kwargs, streamer)
        )

        # Acquire GPU resource if needed
        needs_gpu = self.device is not None and self.device.type != 'cpu'
        resource_context = self.resource_manager.acquire_gpu_resource(f"generate_stream_{self.binding_instance_name}") \
                           if needs_gpu else nullcontext()

        # State variables for streaming and thought parsing
        full_raw_response_text = ""
        accumulated_thoughts = ""
        is_thinking = False
        finish_reason = "unknown"
        usage_info = None # Usage info harder to get accurately in stream
        final_metadata = {
            "model_used": self.loaded_model_identifier,
            "binding_instance": self.binding_instance_name,
            "seed_used": seed
        }

        try:
            async with resource_context:
                if needs_gpu:
                    logger.info(f"HF '{self.binding_instance_name}': GPU resource acquired for stream.")

                # Start the generation thread
                generation_thread.start()
                logger.info("HF stream generation thread started.")

                # Asynchronously iterate through the streamer
                async for raw_text_chunk in self._async_iter_streamer(streamer):
                    if raw_text_chunk:
                        full_raw_response_text += raw_text_chunk # Accumulate raw text

                        # --- ADDED: Stream parsing logic for thoughts ---
                        current_text_to_process = raw_text_chunk
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
                        # --- End Stream parsing logic ---

                # After consuming the streamer, wait for the generation thread to finish
                # Use a reasonable timeout based on max tokens
                join_timeout = generation_kwargs.get('max_new_tokens', DEFAULT_MAX_TOKENS) * 2 # Heuristic timeout
                await asyncio.to_thread(generation_thread.join, timeout=join_timeout)

                if generation_thread.is_alive():
                    logger.warning(f"HF stream generation thread did not finish within timeout ({join_timeout}s) for instance '{self.binding_instance_name}'.")
                    finish_reason = "timeout"
                else:
                    # Estimate finish reason based on output length vs max_new_tokens
                    max_new = generation_kwargs.get('max_new_tokens', float('inf'))
                    # Estimate output tokens (less accurate than non-stream)
                    approx_output_tokens = len(self.tokenizer.encode(full_raw_response_text))
                    if approx_output_tokens >= max_new:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop" # Assume normal stop if not length limited

            # --- Final Chunk Processing ---
            # Handle incomplete thought tag at the very end
            if is_thinking and accumulated_thoughts:
                logger.warning(f"HF stream ended mid-thought for '{self.binding_instance_name}'. Thought content:\n{accumulated_thoughts}")
                final_metadata["incomplete_thoughts"] = accumulated_thoughts

            logger.info(f"HF stream finished for '{self.binding_instance_name}'. Reason: {finish_reason}")
            final_metadata["finish_reason"] = finish_reason
            final_metadata["usage"] = usage_info # Likely None for stream

            # Re-parse the full accumulated text for the final output
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

            # Yield the final result
            final_output_list = [{
                "type": "text",
                "data": final_cleaned_text.strip(),
                "thoughts": final_thoughts_str,
                "metadata": final_metadata
            }]
            yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"HF '{self.binding_instance_name}': CUDA OOM during stream: {e}")
            trace_exception(e)
            yield {"type": "error", "content": "CUDA Out of Memory during generation."}
        except asyncio.TimeoutError:
            logger.error(f"HF '{self.binding_instance_name}': Timeout waiting for GPU resource for stream.")
            yield {"type": "error", "content": "Timeout waiting for GPU resource."}
        except Exception as e:
            logger.error(f"HF '{self.binding_instance_name}': Unexpected error during stream: {e}", exc_info=True)
            trace_exception(e)
            yield {"type": "error", "content": f"Unexpected error during generation: {e}"}
        finally:
            # Ensure thread is not left running if an error occurred outside the context manager
            if generation_thread.is_alive():
                 logger.warning("Generation thread still alive after stream processing finished unexpectedly.")
                 # Attempting to force stop might be complex/unsafe. Log it.

    def _run_generation_in_thread(self, input_dict, generation_kwargs, streamer):
        """Target function for the generation thread (runs model.generate)."""
        try:
            with torch.no_grad():
                # Ensure inputs are on the correct device (might be different due to device_map='auto')
                model_device = next(self.model.parameters()).device
                inputs_on_device = {k: v.to(model_device) for k, v in input_dict.items()}
                # Run generation with the streamer
                self.model.generate(**inputs_on_device, streamer=streamer, **generation_kwargs)
        except Exception as e:
            # Log errors occurring within the thread
            logger.error(f"Error in HF generation thread for instance '{self.binding_instance_name}': {e}", exc_info=True)
        finally:
            # Note: Streamer needs to be handled by the consumer loop.
            # No specific cleanup here unless streamer needs explicit closing.
            pass

    async def _async_iter_streamer(self, streamer: TextIteratorStreamer):
        """Wraps the TextIteratorStreamer iterator in an async generator."""
        while True:
            try:
                # Run the blocking `next(streamer)` call in a thread
                # Use a sentinel StopIteration to signal the end
                next_item = await asyncio.to_thread(next, streamer, StopIteration)
                if next_item is StopIteration:
                    break # End of stream
                yield next_item
            except StopIteration:
                # This exception should be caught by the sentinel logic above,
                # but include it for robustness.
                break

    # --- Tokenizer / Info Methods ---

    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenizes text using the loaded tokenizer."""
        if not self.tokenizer:
            raise RuntimeError(f"Tokenizer not loaded for HF instance '{self.binding_instance_name}'.")

        logger.info(f"HF '{self.binding_instance_name}': Tokenizing text...")
        # Run synchronous tokenization in a thread
        def tokenize_sync():
            # `add_special_tokens=False` prevents BOS/EOS during basic encode
            return self.tokenizer.encode(text, add_special_tokens=False)
        tokens = await asyncio.to_thread(tokenize_sync)

        # Manually add BOS/EOS if requested and available
        if add_bos and self.tokenizer.bos_token_id is not None:
            tokens.insert(0, self.tokenizer.bos_token_id)
        if add_eos and self.tokenizer.eos_token_id is not None:
            tokens.append(self.tokenizer.eos_token_id)

        logger.info(f"HF '{self.binding_instance_name}': Tokenized into {len(tokens)} tokens.")
        return tokens

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenizes tokens using the loaded tokenizer."""
        if not self.tokenizer:
            raise RuntimeError(f"Tokenizer not loaded for HF instance '{self.binding_instance_name}'.")

        logger.info(f"HF '{self.binding_instance_name}': Detokenizing {len(tokens)} tokens...")
        # Run synchronous decoding in a thread
        def decode_sync():
            # `skip_special_tokens=True` removes BOS/EOS etc. from the decoded string
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        text = await asyncio.to_thread(decode_sync)

        logger.info(f"HF '{self.binding_instance_name}': Detokenization successful.")
        return text

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded Hugging Face model."""
        if not self._model_loaded or not self.model or not self.loaded_model_identifier:
            # Return empty info if nothing is loaded
            return {
                "name": None, "context_size": None, "max_output_tokens": None,
                "supports_vision": False, "supports_audio": False, "details": {}
            }

        # Try to get context size from model config
        context_size = None
        if hasattr(self.model, 'config'):
            ctx_len = getattr(self.model.config, 'max_position_embeddings', None) or \
                      getattr(self.model.config, 'n_ctx', None) # Common config names
            if isinstance(ctx_len, int):
                context_size = ctx_len

        # Check if Flash Attention 2 is actively being used by the loaded model
        flash_attn_active = False
        if hasattr(self.model.config, "_attn_implementation") and self.model.config._attn_implementation == "flash_attention_2":
             flash_attn_active = True

        # Compile details about the loaded setup
        details = {
            "model_identifier": self.loaded_model_identifier,
            "instance_name": self.binding_instance_name,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__ if self.tokenizer else None,
            "processor_class": type(self.processor).__name__ if self.processor else None,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "quantization": self.quantization_str if self.quantization_str != "none" else "none",
            "flash_attention_2_enabled": flash_attn_active,
            "trust_remote_code": self.trust_remote_code,
        }

        # Return standardized info dictionary
        return {
            "name": self.model_name or self.loaded_model_identifier, # Use model_name if set, else identifier
            "context_size": context_size,
            "max_output_tokens": None, # Max output tokens is dynamic, not fixed property
            "supports_vision": self.model_supports_vision,
            "supports_audio": False, # Audio not supported by this binding
            "details": details
        }