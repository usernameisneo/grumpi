# zoos/bindings/diffusers_binding/__init__.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: lollms_server Team
# Creation Date: 2025-05-01
# Modification Date: 2025-05-04
# Description: Binding implementation for local Hugging Face Transformers models.
# Enhanced with I2I, ControlNet, Inpainting support and model suggestions.

import asyncio
import sys
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List, Type # Added Type
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from io import BytesIO
import threading
import importlib.util
import random # Added for seed generation

# --- Dependency Check ---
# Use pipmaster if needed, but primarily rely on requirements from card
try:
    import pipmaster as pm
    # Specify requirements from the binding card here or rely on user installing extras
    # Example: pm.ensure_packages(["torch", "diffusers", "transformers", ...])
    # It's generally better to guide users via install scripts/extras for heavy deps like torch.
    # Only ensure optional libraries if absolutely necessary here.
    pm.ensure_packages(["accelerate", "safetensors", "pillow>=9.0.0"])
    # Optionally check/install compel, bitsandbytes, flash_attn if config requires them
    # (better to do this check lazily during load_model if needed)
except ImportError:
    print("WARNING: pipmaster not found. Cannot ensure diffusers optional dependencies.")
    pass

# --- Core Library Imports ---
try:
    import torch
    import transformers
    from diffusers import (
        DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline,
        StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, # Added I2I pipelines
        StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline, # Added Inpaint pipelines
        StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, # Added CN pipelines
        ControlNetModel, # Added ControlNetModel
        DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
        LCMScheduler, DDIMScheduler, DDPMScheduler, PNDMScheduler # Added more schedulers
    )
    from diffusers.utils.import_utils import is_invisible_watermark_available
    from PIL import Image
    diffusers_installed = True
    torch_installed = True
except ImportError as e:
    print(f"ERROR: Missing core dependency for DiffusersBinding: {e}")
    # Define mocks for type hinting and basic checks
    torch = None; DiffusionPipeline = None; AutoencoderKL = None; StableDiffusionPipeline = None; StableDiffusionXLPipeline = None; StableDiffusionImg2ImgPipeline = None; StableDiffusionXLImg2ImgPipeline = None; StableDiffusionInpaintPipeline = None; StableDiffusionXLInpaintPipeline = None; StableDiffusionControlNetPipeline = None; StableDiffusionXLControlNetPipeline = None; ControlNetModel = None # type: ignore
    DPMSolverMultistepScheduler = None; EulerAncestralDiscreteScheduler = None; EulerDiscreteScheduler = None; LCMScheduler=None; DDIMScheduler = None; DDPMScheduler = None; PNDMScheduler = None; Image = None # type: ignore
    is_invisible_watermark_available = lambda: False
    diffusers_installed = False
    torch_installed = False
    _import_error_msg = str(e)

# --- Optional Dependency Checks ---
try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig # Need this for quantization
    bitsandbytes_installed = True
except ImportError:
    bitsandbytes = None; BitsAndBytesConfig = None # type: ignore
    bitsandbytes_installed = False
try:
    import flash_attn # Use actual name for check
    flash_attn_installed = True
except ImportError:
    flash_attn = None # type: ignore
    flash_attn_installed = False
try:
    from compel import Compel, ReturnedEmbeddingsType
    compel_installed = True
except ImportError:
    Compel = None; ReturnedEmbeddingsType = None # type: ignore
    compel_installed = False

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
from lollms_server.core.config import get_server_root
from lollms_server.utils.helpers import parse_thought_tags # Keep for consistency

# Use TYPE_CHECKING for API model imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData, ModelInfo
    except ImportError:
        class StreamChunk: pass # type: ignore
        class InputData: pass # type: ignore
        class ModelInfo: pass # type: ignore

from PIL import Image
logger = logging.getLogger(__name__)

# --- Constants ---
SCHEDULER_MAP = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "LCMScheduler": LCMScheduler,
    "DDIMScheduler": DDIMScheduler,
    "DDPMScheduler": DDPMScheduler,
    "PNDMScheduler": PNDMScheduler,
    # Add other diffusers schedulers here by their class name string if they exist
}

# List of suggested models (Hub ID, Description) - Can be moved to separate config later
SUGGESTED_MODELS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "Stable Diffusion XL (General Purpose, High Quality)"),
    ("stabilityai/stable-diffusion-xl-refiner-1.0", "SDXL Refiner (Use with Base)"),
    ("runwayml/stable-diffusion-v1-5", "Stable Diffusion 1.5 (Lower VRAM, Faster)"),
    ("stabilityai/stable-diffusion-2-1", "Stable Diffusion 2.1 (Alternative Base)"),
    ("stabilityai/sdxl-turbo", "SDXL Turbo (Very Fast Generation, Lower Detail)"),
    ("ByteDance/SDXL-Lightning", "SDXL Lightning (Fast Generation, Various Steps)"),
    ("Lykon/dreamshaper-xl-1024-v2-baked-vae", "DreamShaper XL (Semi-Realistic/Fantasy)"),
    ("playgroundai/playground-v2.5-1024px-aesthetic", "Playground v2.5 (Aesthetic focus)"),
    ("kandinsky-community/kandinsky-2-2-decoder", "Kandinsky 2.2 (Requires Prior)"),
    ("cagliostrolab/animagine-xl-3.1", "Animagine XL 3.1 (Anime Style)"),
    # Add more diverse suggestions (inpainting, specific styles etc.)
    ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL Base (Good starting point)"), # Example duplicates are ok
    ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "SDXL Inpainting"),
]

# Suggested ControlNet models (Hub ID, Description)
SUGGESTED_CONTROLNETS = [
    ("lllyasviel/sd-controlnet-canny", "ControlNet Canny (SD 1.5)"),
    ("diffusers/controlnet-canny-sdxl-1.0", "ControlNet Canny (SDXL)"),
    ("lllyasviel/sd-controlnet-depth", "ControlNet Depth (SD 1.5)"),
    ("diffusers/controlnet-depth-sdxl-1.0", "ControlNet Depth (SDXL)"),
    ("lllyasviel/sd-controlnet-openpose", "ControlNet OpenPose (SD 1.5)"),
    ("thibaud/controlnet-openpose-sdxl-1.0", "ControlNet OpenPose (SDXL)"),
    ("lllyasviel/sd-controlnet-scribble", "ControlNet Scribble (SD 1.5)"),
    ("diffusers/controlnet-scribble-sdxl-1.0", "ControlNet Scribble (SDXL)"),
    ("lllyasviel/controlnet-canny-sdxl-1.0-small","ControlNet Canny SDXL small"),
    ("diffusers/controlnet-depth-sdxl-1.0-small","ControlNet Depth SDXL small"),
    ("diffusers/controlnet-openpose-sdxl-1.0","ControlNet Openpose SDXL"),
    ("diffusers/controlnet-scribble-sdxl-1.0","ControlNet Scribble SDXL"),
    ("diffusers/controlnet-canny-sdxl-1.0","ControlNet Canny SDXL"),
    ("diffusers/controlnet-lineart-sdxl-1.0","ControlNet Lineart SDXL"),
    ("diffusers/controlnet-segmentation-sdxl-1.0","ControlNet Segmentation SDXL"),
    ("diffusers/controlnet-tile-sdxl-1.0","ControlNet Tile SDXL"),
    ("diffusers/controlnet-softedge-sdxl-1.0","ControlNet Softedge SDXL"),
]

# --- Binding Class ---
class DiffusersBinding(Binding):
    """Binding for Stable Diffusion models using the diffusers library."""
    binding_type_name = "diffusers_binding" # Match type_name in card

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the DiffusersBinding."""
        super().__init__(config, resource_manager)

        if not diffusers_installed or not torch_installed:
            raise ImportError(f"Diffusers binding requires 'torch' and 'diffusers'. Error: {_import_error_msg}")

        # --- Configuration Loading ---
        models_folder_str = self.config.get("models_folder", "models/diffusers_models/")
        self.models_folder = self._resolve_path(models_folder_str, "models folder")
        # NEW: ControlNet Folder
        cn_folder_str = self.config.get("controlnet_models_folder", "models/controlnet/")
        self.controlnet_models_folder = self._resolve_path(cn_folder_str, "ControlNet models folder")

        self.device_str = self.config.get("device", "auto").lower()
        self.use_fp16 = self.config.get("use_fp16", True)
        self.use_bf16 = self.config.get("use_bf16", False)
        self.default_scheduler_type = self.config.get("scheduler_type", "DPMSolverMultistepScheduler")
        self.vae_path = self.config.get("vae_path") # Can be None or Hub ID/path
        self.lora_paths = self.config.get("lora_paths", []) # Default LoRAs to load
        self.controlnet_model_paths = self.config.get("controlnet_model_paths", []) # Default ControlNets
        self.enable_safety_checker = self.config.get("enable_safety_checker", True)
        self.use_compel = self.config.get("use_compel", False) and compel_installed
        self.default_strength = self.config.get("default_strength", 0.8)
        self.default_controlnet_scale = self.config.get("default_controlnet_scale", 0.75)

        self.default_model_name= self.config.get("default_model", None)


        # --- Internal State Initialization ---
        self.pipeline: Optional[DiffusionPipeline] = None
        self.compel_proc: Optional[Compel] = None
        self.loaded_controlnets: Dict[str, Any] = {} # Store loaded ControlNet models {path_or_id: model_obj}
        self.loaded_model_path: Optional[Path] = None
        self.loaded_vae_path: Optional[str] = None
        self.loaded_loras: List[str] = []
        self.torch_dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None
        self.is_sdxl = False # Flag to track if loaded model is SDXL

        # Determine Device and Dtype
        self._determine_device_and_dtype()
        if self.use_compel and not compel_installed:
             logger.warning(f"Compel requested for '{self.binding_instance_name}' but library not found. Weighting disabled.")

        logger.info(f"Initialized DiffusersBinding '{self.binding_instance_name}': Device='{self.device}', DType='{self.torch_dtype}', Models='{self.models_folder}', ControlNets='{self.controlnet_models_folder}'")

    def _resolve_path(self, path_str: Optional[str], description: str) -> Optional[Path]:
        """Resolves a path relative to server root if not absolute and ensures directory."""
        if not path_str:
            logger.debug(f"Path for '{description}' not configured.")
            return None
        path_obj = Path(path_str)
        if not path_obj.is_absolute():
            path_obj = (get_server_root() / path_obj).resolve()
            logger.info(f"Diffusers '{self.binding_instance_name}': Resolved {description} path to {path_obj}")

        # Ensure directory exists
        if not path_obj.exists():
            logger.warning(f"{description} path does not exist: {path_obj}. Creating it.")
            try: path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e: logger.error(f"Failed to create {description} path: {e}"); return None
        elif not path_obj.is_dir():
            logger.error(f"Configured {description} path is not a directory: {path_obj}")
            return None
        return path_obj

    def _determine_device_and_dtype(self):
        """Sets the torch device and dtype based on config and availability."""
        # (Implementation remains the same)
        if self.device_str == "auto":
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        elif self.device_str == "cuda" and torch.cuda.is_available(): self.device = torch.device("cuda")
        elif self.device_str == "mps" and torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")

        if self.device != torch.device("cpu"):
            if self.use_bf16 and ((self.device.type == 'cuda' and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) or self.device.type == 'mps'):
                 self.torch_dtype = torch.bfloat16; logger.info("Using bfloat16 precision.")
            elif self.use_fp16: self.torch_dtype = torch.float16; logger.info("Using float16 precision.")
            else: self.torch_dtype = torch.float32; logger.info("Using float32 precision.")
        else: # CPU
            self.torch_dtype = torch.float32
            if self.use_fp16 or self.use_bf16: logger.warning("fp16/bf16 not recommended on CPU. Using float32.")
            logger.info("Using CPU device with float32 precision.")

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types (text and image)."""
        # Diffusers binding inherently supports text and can support image (I2I, CN, Inpaint)
        return ["text", "image"]

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types (image)."""
        return ["image"]

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Scans the models_folder and adds suggestions from Hub."""
        logger.info(f"Diffusers '{self.binding_instance_name}': Scanning for models in {self.models_folder}")
        local_models = []
        if self.models_folder and self.models_folder.is_dir():
            try:
                for item in self.models_folder.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        # Basic check for diffuser structure
                        if (item / "model_index.json").exists() or ((item / "unet").is_dir() and (item / "unet" / "config.json").exists()):
                            try:
                                stat_info = item.stat()
                                # Infer capabilities (basic)
                                is_xl = "xl" in item.name.lower()
                                is_inpaint = "inpaint" in item.name.lower()
                                model_data = {
                                    "name": item.name, "size": None, "modified_at": datetime.fromtimestamp(stat_info.st_mtime),
                                    "format": "diffusers", "family": "sdxl" if is_xl else "sd", "supports_vision": True,
                                    "supports_audio": False,
                                    "details": {"path": str(item.resolve()), "type": "local", "is_xl": is_xl, "is_inpainting": is_inpaint}
                                }
                                local_models.append(model_data)
                            except Exception as e: logger.warning(f"Could not process potential model dir {item.name}: {e}")
            except Exception as scan_err: logger.error(f"Error scanning models folder {self.models_folder}: {scan_err}", exc_info=True)
        else:
             logger.warning(f"Models folder not found or invalid: {self.models_folder}")

        # Add suggestions
        suggestions = []
        local_names = {m["name"] for m in local_models}
        for hub_id, desc in SUGGESTED_MODELS:
             if hub_id not in local_names: # Only suggest if not already present locally
                 is_xl = "xl" in hub_id.lower()
                 is_inpaint = "inpaint" in hub_id.lower()
                 suggestions.append({
                     "name": hub_id, "size": None, "modified_at": None,
                     "format": "diffusers", "family": "sdxl" if is_xl else "sd", "supports_vision": True,
                     "supports_audio": False,
                     "details": {"type": "suggestion", "description": desc, "is_xl": is_xl, "is_inpainting": is_inpaint}
                 })

        logger.info(f"Diffusers '{self.binding_instance_name}': Found {len(local_models)} local models, {len(suggestions)} suggestions.")
        # Prioritize local models in the list
        return local_models + suggestions

    async def health_check(self) -> Tuple[bool, str]:
        """Checks if core libraries are installed and device is accessible."""
        # (Implementation remains the same)
        if not torch_installed: return False, "PyTorch library not installed."
        if not diffusers_installed: return False, "Diffusers library not installed."
        try:
            if self.device and self.device.type == 'cuda':
                if not torch.cuda.is_available(): return False, "CUDA specified but not available."; torch.cuda.get_device_name(0)
            elif self.device and self.device.type == 'mps':
                 if not torch.backends.mps.is_available(): return False, "MPS specified but not available."; torch.tensor([1], device=self.device)
            elif self.device and self.device.type == 'cpu': pass
            else: return False, f"Configured device '{self.device}' not recognized or available."
            return True, f"Torch ({torch.__version__}) & Diffusers available. Device '{self.device}' OK."
        except Exception as e: logger.error(f"Diffusers health check failed for '{self.binding_instance_name}': {e}", exc_info=True); return False, f"Health check failed: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Diffusers models typically require GPU unless device is CPU."""
        # (Implementation remains the same)
        return {"gpu_required": self.device != torch.device("cpu"), "estimated_vram_mb": 4096}

    async def load_model(self, model_name_or_path: str) -> bool:
        """Loads the diffusion pipeline, VAE, LoRAs, and default ControlNets."""
        target_path_obj = self.models_folder / model_name_or_path if self.models_folder else None
        load_from: Union[str, Path]

        if target_path_obj and target_path_obj.is_dir():
             load_from = target_path_obj
             logger.info(f"Diffusers '{self.binding_instance_name}': Found model locally at {load_from}")
        else:
             if "/" in model_name_or_path or Path(model_name_or_path).is_absolute():
                 load_from = model_name_or_path # Treat as Hub ID or absolute path
                 logger.info(f"Diffusers '{self.binding_instance_name}': Attempting load '{load_from}' as Hub ID or absolute path.")
             else:
                 logger.error(f"Diffusers model dir not found locally: '{model_name_or_path}' in {self.models_folder}, and not Hub ID.")
                 return False

        async with self._load_lock:
            current_identifier = Path(load_from).resolve() if isinstance(load_from, Path) else load_from
            previous_identifier = self.loaded_model_path.resolve() if isinstance(self.loaded_model_path, Path) else self.loaded_model_path
            if self._model_loaded and previous_identifier == current_identifier:
                logger.info(f"Diffusers '{self.binding_instance_name}': Model '{model_name_or_path}' already loaded.")
                return True
            elif self._model_loaded:
                logger.info(f"Diffusers '{self.binding_instance_name}': Switching model. Unloading '{self.model_name}' first...")
                await self.unload_model()

            logger.info(f"Diffusers '{self.binding_instance_name}': Loading model '{model_name_or_path}'...")
            needs_gpu = self.device != torch.device("cpu")
            model_id_for_log = Path(load_from).name if isinstance(load_from, Path) else load_from
            resource_context = self.resource_manager.acquire_gpu_resource(f"load_{self.binding_instance_name}_{model_id_for_log}") if needs_gpu else nullcontext()

            try:
                async with resource_context:
                    if needs_gpu: logger.info(f"Diffusers '{self.binding_instance_name}': GPU resource acquired.")
                    load_args = {"pretrained_model_name_or_path": str(load_from), "torch_dtype": self.torch_dtype}
                    vae = None # --- Load VAE ---
                    if self.vae_path:
                        try:
                            logger.info(f"Loading custom VAE from: {self.vae_path}")
                            vae = await asyncio.to_thread(AutoencoderKL.from_pretrained, self.vae_path, torch_dtype=self.torch_dtype)
                            load_args["vae"] = vae; self.loaded_vae_path = self.vae_path
                        except Exception as e: logger.error(f"Failed load custom VAE '{self.vae_path}': {e}. Using default.", exc_info=True); self.loaded_vae_path = None
                    else: self.loaded_vae_path = None

                    # --- Load Pipeline ---
                    logger.info("Instantiating diffusion pipeline...")
                    # Determine pipeline type (basic heuristic) - Could use AutoPipeline for more robustness
                    pipeline_class = StableDiffusionPipeline
                    self.is_sdxl = "xl" in model_id_for_log.lower()
                    if self.is_sdxl:
                        # Need to check if specific task pipelines are better, but XL base often works
                        pipeline_class = StableDiffusionXLPipeline # Default to base XL
                        # Could add checks here for 'img2img' or 'inpaint' in name if needed
                    elif "inpaint" in model_id_for_log.lower():
                         pipeline_class = StableDiffusionInpaintPipeline
                    elif "img2img" in model_id_for_log.lower(): # Less common naming
                         pipeline_class = StableDiffusionImg2ImgPipeline

                    logger.info(f"Attempting to load using pipeline class: {pipeline_class.__name__}")
                    self.pipeline = await asyncio.to_thread(pipeline_class.from_pretrained, **load_args)
                    if not self.pipeline: raise RuntimeError("Pipeline loading returned None.")
                    logger.info(f"Pipeline loaded. Type: {type(self.pipeline).__name__}")

                    # --- Load Scheduler ---
                    scheduler_cls = SCHEDULER_MAP.get(self.default_scheduler_type);
                    if scheduler_cls:
                        logger.info(f"Setting scheduler to {self.default_scheduler_type}");
                        try: self.pipeline.scheduler = await asyncio.to_thread(scheduler_cls.from_config, self.pipeline.scheduler.config)
                        except Exception as e: logger.error(f"Failed to set scheduler {self.default_scheduler_type}: {e}", exc_info=True)
                    else: logger.warning(f"Scheduler type '{self.default_scheduler_type}' not found. Using pipeline default.")

                    # --- Move to Device ---
                    logger.info(f"Moving pipeline to device: {self.device}"); await asyncio.to_thread(self.pipeline.to, self.device)

                    # --- Load Default LoRAs ---
                    self.loaded_loras = [];
                    if self.lora_paths:
                        logger.info(f"Loading {len(self.lora_paths)} default LoRA(s)...")
                        for lora in self.lora_paths:
                            try:
                                logger.info(f" -> Loading LoRA: {lora}")
                                if hasattr(self.pipeline, "load_lora_weights"): await asyncio.to_thread(self.pipeline.load_lora_weights, lora); self.loaded_loras.append(lora)
                                else: logger.warning(f"Pipeline {type(self.pipeline)} lacks load_lora_weights."); break
                            except Exception as e: logger.error(f"Failed load default LoRA '{lora}': {e}", exc_info=True)

                    # --- Load Default ControlNets ---
                    self.loaded_controlnets = {}
                    if self.controlnet_model_paths and ControlNetModel:
                        logger.info(f"Loading {len(self.controlnet_model_paths)} default ControlNet(s)...")
                        for cn_path_or_id in self.controlnet_model_paths:
                            try:
                                cn_model = None
                                logger.info(f" -> Loading ControlNet: {cn_path_or_id}")
                                # Check local path first
                                local_cn_path = self.controlnet_models_folder / cn_path_or_id if self.controlnet_models_folder else None
                                if local_cn_path and local_cn_path.is_dir():
                                    logger.debug(f"   Loading ControlNet locally from {local_cn_path}")
                                    cn_model = await asyncio.to_thread(ControlNetModel.from_pretrained, str(local_cn_path), torch_dtype=self.torch_dtype)
                                else:
                                    logger.debug(f"   Loading ControlNet from Hub ID: {cn_path_or_id}")
                                    cn_model = await asyncio.to_thread(ControlNetModel.from_pretrained, cn_path_or_id, torch_dtype=self.torch_dtype)

                                if cn_model:
                                    # Store with original path/id as key for easy reference
                                    self.loaded_controlnets[cn_path_or_id] = cn_model.to(self.device)
                                    logger.info(f"   ControlNet '{cn_path_or_id}' loaded.")
                                else:
                                    logger.error(f"   ControlNet model loading returned None for '{cn_path_or_id}'")

                            except Exception as e:
                                logger.error(f"Failed to load default ControlNet '{cn_path_or_id}': {e}", exc_info=True)
                    else:
                         if self.controlnet_model_paths: logger.warning("ControlNet paths specified but ControlNetModel class not loaded.")

                    # --- Safety Checker ---
                    if not self.enable_safety_checker:
                         if hasattr(self.pipeline, "safety_checker") and self.pipeline.safety_checker is not None: logger.info("Disabling safety checker."); self.pipeline.safety_checker = None
                         if hasattr(self.pipeline, "requires_safety_checker"): self.pipeline.requires_safety_checker = False
                    else: logger.info("Safety checker enabled (if available).")

                    # --- Compel Initialization ---
                    self.compel_proc = None;
                    if self.use_compel and Compel and self.pipeline:
                        logger.info("Initializing Compel...");
                        try:
                            tok1=getattr(self.pipeline, 'tokenizer', None); tok2=getattr(self.pipeline, 'tokenizer_2', None)
                            enc1=getattr(self.pipeline, 'text_encoder', None); enc2=getattr(self.pipeline, 'text_encoder_2', None)
                            if tok1 and enc1:
                                tokenizers = [tok1]; text_encoders = [enc1]; pooled = [False]
                                if self.is_sdxl and tok2 and enc2: tokenizers.append(tok2); text_encoders.append(enc2); pooled = [False, True]
                                self.compel_proc = await asyncio.to_thread( Compel, tokenizer=tokenizers, text_encoder=text_encoders, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=pooled); logger.info("Compel initialized.")
                            else: logger.error("Compel requires tokenizer/text_encoder. Disabling."); self.use_compel = False
                        except Exception as e: logger.error(f"Failed Compel init: {e}. Disabling.", exc_info=True); self.compel_proc = None; self.use_compel = False

                    # --- Finalize Load ---
                    self.loaded_model_path = Path(load_from) if isinstance(load_from, Path) else Path(load_from)
                    self.model_name = model_id_for_log
                    self._model_loaded = True
                    logger.info(f"Diffusers model '{self.model_name}' loaded successfully.")
                    return True

            except asyncio.TimeoutError: logger.error(f"Timeout loading {model_id_for_log}"); await self.unload_model(); return False
            except Exception as e: logger.error(f"Failed load model '{model_name_or_path}': {e}", exc_info=True); await self.unload_model(); return False

    async def unload_model(self) -> bool:
        """Unloads pipeline, ControlNets, clears memory."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Diffusers '{self.binding_instance_name}': Unloading model '{self.model_name}'...")
            try:
                del self.pipeline; del self.compel_proc; del self.loaded_controlnets
                self.pipeline = None; self.compel_proc = None; self.loaded_controlnets = {}
                self.loaded_model_path = None; self.model_name = None; self.loaded_vae_path = None; self.loaded_loras = []
                self.is_sdxl = False; self._model_loaded = False
                if self.device and self.device.type == 'cuda': logger.info("Clearing CUDA cache..."); await asyncio.to_thread(torch.cuda.empty_cache)
                elif self.device and self.device.type == 'mps': logger.info("Clearing MPS cache..."); await asyncio.to_thread(torch.mps.empty_cache)
                logger.info(f"Diffusers model unloaded for '{self.binding_instance_name}'.")
                return True
            except Exception as e: logger.error(f"Error during Diffusers unload '{self.binding_instance_name}': {e}", exc_info=True); self._model_loaded = False; return False

    def _prepare_pil_image(self, input_data: 'InputData') -> Optional[Image.Image]:
        """Decodes base64 and loads PIL image, returns None on failure."""
        if not input_data or not input_data.data or not isinstance(input_data.data, str):
            logger.warning(f"Invalid image data provided (role: {input_data.role if input_data else 'N/A'}).")
            return None
        try:
            img_bytes = base64.b64decode(input_data.data)
            img = Image.open(BytesIO(img_bytes))
            # Ensure RGB for most pipelines, L for masks
            if input_data.role == 'mask_image':
                img = img.convert("L")
            else:
                 img = img.convert("RGB")
            logger.debug(f"Successfully loaded PIL image (role: {input_data.role})")
            return img
        except Exception as e:
             logger.error(f"Failed to decode/load PIL image (role: {input_data.role}): {e}")
             return None

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]: # Return List[OutputData]-like
        """Generates an image using the loaded diffusers pipeline, handling I2I, Inpainting, ControlNet."""
        if not self.pipeline or not self._model_loaded:
            raise RuntimeError(f"Diffusers model '{self.model_name}' not loaded in binding '{self.binding_instance_name}'.")

        logger.info(f"Diffusers '{self.binding_instance_name}': Starting generation (Prompt: '{prompt[:50]}...')...")
        gen_type = request_info.get("generation_type", "tti") # Get task type

        # --- Prepare Base Parameters ---
        pipe_params: Dict[str, Any] = {}
        pipe_params["num_inference_steps"] = params.get("num_inference_steps", 30)
        pipe_params["guidance_scale"] = params.get("guidance_scale", 7.5)
        pipe_params["negative_prompt"] = params.get("negative_prompt", None)
        pipe_params["width"] = params.get("width", 512)
        pipe_params["height"] = params.get("height", 512)
        seed = params.get("seed", -1); actual_seed = seed
        if isinstance(seed, int) and seed != -1: pipe_params["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        else: actual_seed = random.randint(0, 2**32 - 1); pipe_params["generator"] = torch.Generator(device=self.device).manual_seed(actual_seed)

        # --- Handle Compel Prompt Weighting ---
        if self.use_compel and self.compel_proc:
            logger.info("Using Compel for prompt weighting.")
            try:
                conditioning, pooled = await asyncio.to_thread(self.compel_proc, prompt)
                if self.is_sdxl and pooled is not None: pipe_params["prompt_embeds"] = conditioning[0]; pipe_params["pooled_prompt_embeds"] = pooled[0]
                else: pipe_params["prompt_embeds"] = conditioning
                neg_prompt = pipe_params.get("negative_prompt", None)
                if neg_prompt:
                     neg_conditioning, neg_pooled = await asyncio.to_thread(self.compel_proc, neg_prompt)
                     if self.is_sdxl and neg_pooled is not None: pipe_params["negative_prompt_embeds"] = neg_conditioning[0]; pipe_params["negative_pooled_prompt_embeds"] = neg_pooled[0]
                     else: pipe_params["negative_prompt_embeds"] = neg_conditioning
                pipe_params.pop("prompt", None); pipe_params.pop("negative_prompt", None) # Remove raw prompts
            except Exception as e: logger.error(f"Compel processing failed: {e}. Falling back to standard prompt.", exc_info=True); pipe_params["prompt"] = prompt
        else: pipe_params["prompt"] = prompt

        # --- Process Multimodal Inputs ---
        input_pil_image: Optional[Image.Image] = None
        mask_pil_image: Optional[Image.Image] = None
        control_pil_image: Optional[Image.Image] = None # Support single CN image for now

        if multimodal_data:
            for item in multimodal_data:
                if item.type == 'image':
                    if item.role == 'input_image' and not input_pil_image:
                         input_pil_image = self._prepare_pil_image(item)
                         if input_pil_image: pipe_params["image"] = input_pil_image; logger.info("Added 'input_image'.")
                    elif item.role == 'mask_image' and not mask_pil_image:
                         mask_pil_image = self._prepare_pil_image(item)
                         if mask_pil_image: pipe_params["mask_image"] = mask_pil_image; logger.info("Added 'mask_image'.")
                    elif item.role == 'controlnet_image' and not control_pil_image:
                         control_pil_image = self._prepare_pil_image(item)
                         if control_pil_image: pipe_params["image"] = control_pil_image; logger.info("Added 'controlnet_image' (used as pipeline 'image').")
                    # Add other roles like 'reference_image' etc. if needed

        # --- Set Task-Specific Parameters ---
        is_i2i = bool(input_pil_image) and not mask_pil_image # Basic I2I
        is_inpaint = bool(input_pil_image) and bool(mask_pil_image) # Inpainting
        is_controlnet = bool(control_pil_image) and bool(self.loaded_controlnets) # ControlNet

        if is_i2i or is_inpaint:
            pipe_params["strength"] = float(params.get("strength", self.default_strength))
            logger.info(f"Setting strength to {pipe_params['strength']} for I2I/Inpaint.")
            # Pipeline type should ideally match, but Diffusers often handles this if base model supports it.
            # We might need more robust pipeline type checking here if issues arise.

        if is_controlnet:
            if not self.loaded_controlnets:
                 logger.warning("ControlNet image provided, but no ControlNet models loaded. Ignoring ControlNet.")
            else:
                pipe_params["controlnet_conditioning_scale"] = float(params.get("controlnet_scale", self.default_controlnet_scale))
                logger.info(f"Setting controlnet_conditioning_scale to {pipe_params['controlnet_conditioning_scale']}.")
                # If multiple CNs were loaded, may need to pass a list of scales/images.
                # Assuming pipeline takes single `image` and single `controlnet_conditioning_scale` for now.
                # Newer diffusers might handle multiple CNs via `controlnet_model` arg in pipeline call.

        # --- Acquire Resource and Generate ---
        needs_gpu = self.device != torch.device("cpu")
        resource_context = self.resource_manager.acquire_gpu_resource(f"generate_{self.binding_instance_name}_{self.model_name}") if needs_gpu else nullcontext()

        try:
            async with resource_context:
                if needs_gpu: logger.info(f"Diffusers '{self.binding_instance_name}': GPU resource acquired for generation.")
                logger.info("Starting image generation inference...")
                # Remove internal 'seed_used' before calling pipeline
                gen_kwargs = {k: v for k, v in pipe_params.items() if k != 'generator'}
                # Add generator back if present
                if "generator" in pipe_params: gen_kwargs["generator"] = pipe_params["generator"]

                # Pass loaded controlnets if any (newer diffusers style)
                # Assume single controlnet for now if multiple are loaded
                if self.loaded_controlnets:
                    gen_kwargs["controlnet"] = next(iter(self.loaded_controlnets.values()))


                # Use asyncio.to_thread for the blocking pipeline call
                output = await asyncio.to_thread(self.pipeline, **gen_kwargs)
                logger.info("Image generation inference finished.")

            # --- Process Output ---
            if not output.images: raise RuntimeError("Generation failed: No images returned.")
            image: Image.Image = output.images[0]
            safety_results = {}
            nsfw_list = getattr(output, "nsfw_content_detected", None)
            if isinstance(nsfw_list, list) and nsfw_list:
                 nsfw_flag = nsfw_list[0] if nsfw_list else False
                 safety_results["nsfw_content_detected"] = nsfw_flag
                 if nsfw_flag and self.enable_safety_checker: logger.warning("Potential NSFW content detected.")

            buffered = BytesIO(); image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.info(f"Diffusers '{self.binding_instance_name}': Image generated successfully.")

            output_metadata = {
                "prompt_used": prompt, "negative_prompt": params.get("negative_prompt", ""),
                "model": self.model_name, "vae": self.loaded_vae_path, "loras": self.loaded_loras,
                "controlnets": list(self.loaded_controlnets.keys()), # List loaded CNs
                "scheduler": type(self.pipeline.scheduler).__name__, "steps": pipe_params["num_inference_steps"],
                "cfg_scale": pipe_params["guidance_scale"], "seed": actual_seed,
                "size": f"{pipe_params['width']}x{pipe_params['height']}",
                **safety_results
            }
            if is_i2i or is_inpaint: output_metadata["strength"] = pipe_params["strength"]
            if is_controlnet and "controlnet_conditioning_scale" in pipe_params: output_metadata["controlnet_scale"] = pipe_params["controlnet_conditioning_scale"]

            return [{"type": "image", "data": image_base64, "mime_type": "image/png", "metadata": output_metadata}]

        except torch.cuda.OutOfMemoryError as e: logger.error(f"Diffusers OOM: {e}"); raise RuntimeError("CUDA Out of Memory.") from e
        except asyncio.TimeoutError: logger.error("Timeout waiting for GPU resource."); raise RuntimeError("Timeout waiting for GPU resource.") from e
        except Exception as e: logger.error(f"Diffusers generation error: {e}", exc_info=True); raise RuntimeError(f"Unexpected generation error: {e}") from e
        finally: # Ensure PIL images are closed
             if input_pil_image and hasattr(input_pil_image, 'close'): input_pil_image.close()
             if mask_pil_image and hasattr(mask_pil_image, 'close'): mask_pil_image.close()
             if control_pil_image and hasattr(control_pil_image, 'close'): control_pil_image.close()

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Simulates streaming for image generation by yielding info then final result."""
        # (Implementation remains the same)
        logger.info(f"Diffusers Binding '{self.binding_instance_name}': Simulating stream for image generation.")
        yield {"type": "info", "content": {"status": "starting_image_generation", "model": self.model_name or 'Unknown', "prompt": prompt}}
        try:
            image_result_list = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
            if isinstance(image_result_list, list):
                yield {"type": "final", "content": image_result_list, "metadata": {"status": "success"}}
                logger.info(f"Diffusers Binding '{self.binding_instance_name}': Simulated stream finished successfully.")
            else: raise TypeError(f"generate() returned unexpected type: {type(image_result_list)}")
        except (ValueError, RuntimeError, Exception) as e:
            logger.error(f"Diffusers Binding '{self.binding_instance_name}': Error during simulated stream's generate call: {e}", exc_info=True)
            error_content = f"Image generation failed: {str(e)}"
            yield {"type": "error", "content": error_content}
            yield {"type": "final", "content": [{"type": "error", "data": error_content}], "metadata": {"status": "failed"}}

    # --- Tokenizer Methods (Not Applicable) ---
    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        raise NotImplementedError("Diffusers binding does not support text tokenization.")

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        raise NotImplementedError("Diffusers binding does not support text detokenization.")

    # --- Updated Model Info Method ---
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns information about a specific model or the active/default one."""
        target_model_id = model_name
        is_active_model = False

        if target_model_id is None:
            if self._model_loaded and self.model_name:
                target_model_id = self.model_name
                is_active_model = True
                logger.debug(f"Getting info for currently active model: {target_model_id}")
            else:
                logger.warning(f"Diffusers instance '{self.binding_instance_name}': Cannot get default model info - no model specified or active.")
                # Return empty structure
                return { "binding_instance_name": self.binding_instance_name, "model_name": None, "model_type": None, "context_size": None, "max_output_tokens": None, "supports_vision": True, "supports_audio": False, "supports_streaming": False, "details": {} }

        # If requesting info for the currently loaded model, use its state
        if is_active_model and self.pipeline:
            details = {
                "pipeline_type": type(self.pipeline).__name__,
                "model_identifier": str(self.loaded_model_path) if self.loaded_model_path else target_model_id,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "vae": self.loaded_vae_path or "Default",
                "loras": self.loaded_loras,
                "controlnets": list(self.loaded_controlnets.keys()),
                "scheduler": type(self.pipeline.scheduler).__name__,
                "compel_enabled": self.use_compel and self.compel_proc is not None,
                "safety_checker_enabled": self.enable_safety_checker and hasattr(self.pipeline, 'safety_checker') and self.pipeline.safety_checker is not None
            }
            return {
                "binding_instance_name": self.binding_instance_name,
                "model_name": target_model_id,
                "model_type": 'tti', # Assume TTI/I2I
                "context_size": None,
                "max_output_tokens": None,
                "supports_vision": True,
                "supports_audio": False,
                "supports_streaming": False, # Defined in card
                "details": details
            }
        else:
            # Info requested for a model *not* currently loaded
            # Try to get info from local path or Hub without loading the full pipeline
            logger.info(f"Diffusers '{self.binding_instance_name}': Getting info for unloaded model '{target_model_id}'...")
            config_path = None
            is_local = False
            if self.models_folder:
                 local_path = self.models_folder / target_model_id
                 if local_path.is_dir() and (local_path / "model_index.json").exists():
                     config_path = local_path / "model_index.json"
                     is_local = True

            if is_local and config_path:
                 # Load config locally (less common, usually unet/scheduler configs are key)
                 # This part needs refinement - getting full info without loading is hard
                 logger.warning(f"Info retrieval for unloaded local model '{target_model_id}' is limited.")
                 return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "model_type": 'tti', "supports_vision": True, "supports_streaming": False, "details": {"status": "local_available", "path": str(local_path)} }

            elif "/" in target_model_id: # Assume Hub ID
                try:
                    logger.debug(f"Attempting to fetch config from Hub: {target_model_id}")
                    # Fetch config directly using huggingface_hub (synchronous)
                    def fetch_hub_config():
                         # Prefer model_index.json if it exists
                         try: return hf_hub_download(repo_id=target_model_id, filename="model_index.json")
                         except: pass # Fallback to scheduler config
                         try: return hf_hub_download(repo_id=target_model_id, filename="scheduler/scheduler_config.json")
                         except: return None # Failed to get config

                    config_file_path = await asyncio.to_thread(fetch_hub_config)

                    # Basic info, can't determine much without loading
                    return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "model_type": 'tti', "supports_vision": True, "supports_streaming": False, "details": {"status": "hub_available", "config_found": config_file_path is not None} }

                except (RepositoryNotFoundError, GatedRepoError) as e:
                     logger.warning(f"Model '{target_model_id}' not found or inaccessible on Hub: {e}")
                     return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": "Model not found or inaccessible on Hub.", "details": {"status": "hub_not_found"} }
                except Exception as e:
                     logger.error(f"Error fetching info for Hub model '{target_model_id}': {e}")
                     return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": f"Error fetching Hub info: {e}", "details": {} }
            else:
                 # Neither local nor looks like Hub ID
                 logger.warning(f"Model '{target_model_id}' not found locally or on Hub.")
                 return { "binding_instance_name": self.binding_instance_name, "model_name": target_model_id, "error": "Model not found locally or on Hub.", "details": {"status": "not_found"} }