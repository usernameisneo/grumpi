# zoos/bindings/diffusers_binding/__init__.py
# (or zoos/bindings/diffusers_binding/diffusers_binding.py if you prefer)

import asyncio
import sys
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from io import BytesIO
import importlib.util
import random # Added for seed generation

# Use pipmaster if needed (keep this for robustness)
try:
    import pipmaster as pm
    # Specify specific versions if known issues exist
    pm.ensure_packages({
        "torch":"",
        "torchvision":"",
        "torchaudio":""
    }) # Note: Index URL might be needed depending on GPU/OS
    pm.ensure_packages({
        "transformers":"",
        "diffusers":"", # >=0.20.0 specified in card
        "accelerate":"",
        "safetensors":"",
        "invisible_watermark":"",
        "pillow":"", # >=9.0.0 specified in card
        "compel":"", # >=2.0 specified in card
    })
except ImportError:
    print("WARNING: pipmaster not found. Cannot ensure diffusers dependencies.")
    pass # Assume installed or handle import error below

# Import core libraries and check installation
try:
    import torch
    from diffusers import (
        DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
        LCMScheduler # Add more schedulers as needed
    )
    from diffusers.utils.import_utils import is_invisible_watermark_available
    from PIL import Image
    diffusers_installed = True
    torch_installed = True
except ImportError as e:
    print(f"ERROR: Missing core dependency for DiffusersBinding: {e}")
    torch = None; DiffusionPipeline = None; AutoencoderKL = None; StableDiffusionPipeline = None; StableDiffusionXLPipeline = None; DPMSolverMultistepScheduler = None; EulerAncestralDiscreteScheduler = None; EulerDiscreteScheduler = None; LCMScheduler=None; Image = None # type: ignore
    is_invisible_watermark_available = lambda: False
    diffusers_installed = False
    torch_installed = False

# Import Compel separately as it's optional
try:
    from compel import Compel, ReturnedEmbeddingsType
    compel_installed = True
except ImportError:
    Compel = None; ReturnedEmbeddingsType = None # type: ignore
    compel_installed = False


# Import lollms_server components
try:
    import ascii_colors as logging # Use logging alias
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.core.config import get_server_root # Import helper to get server root

# Use TYPE_CHECKING for API model imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData, ModelInfo
    except ImportError:
        class StreamChunk: pass # type: ignore
        class InputData: pass # type: ignore
        class ModelInfo: pass # type: ignore


logger = logging.getLogger(__name__)

# --- Constants ---
SCHEDULER_MAP = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "LCMScheduler": LCMScheduler,
    # Add other diffusers schedulers here by their class name string if they exist
}

# --- Binding Class ---
class DiffusersBinding(Binding):
    """Binding for Stable Diffusion models using the diffusers library."""
    binding_type_name = "diffusers_binding" # Match type_name in card

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the DiffusersBinding using the provided instance configuration.

        Args:
            config: The configuration dictionary for this specific instance,
                    loaded and validated by the BindingManager against the
                    binding_card.yaml schema.
            resource_manager: The shared resource manager instance.
        """
        super().__init__(config, resource_manager) # Pass config to parent

        if not diffusers_installed or not torch_installed:
            raise ImportError("Diffusers binding requires 'torch' and 'diffusers' libraries to be installed.")

        # --- Configuration Loading from instance config dict (self.config) ---
        # Use .get(key, default) where default matches the schema default

        models_folder_str = self.config.get("models_folder", "models/diffusers_models/")
        self.models_folder = Path(models_folder_str)
        # Resolve path relative to server root if not absolute
        if not self.models_folder.is_absolute():
             self.models_folder = (get_server_root() / self.models_folder).resolve()
             logger.info(f"Diffusers '{self.binding_instance_name}': Resolved models_folder to {self.models_folder}")

        # Ensure models folder exists
        if not self.models_folder.exists():
            logger.warning(f"Diffusers models folder does not exist: {self.models_folder}. Creating it.")
            try: self.models_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e: logger.error(f"Failed to create diffusers models folder: {e}")
        elif not self.models_folder.is_dir():
             logger.error(f"Configured models_folder path is not a directory: {self.models_folder}")
             # Consider raising an error here if the path must be a directory

        self.device_str = self.config.get("device", "auto").lower()
        self.use_fp16 = self.config.get("use_fp16", True)
        self.use_bf16 = self.config.get("use_bf16", False)
        self.default_scheduler_type = self.config.get("scheduler_type", "DPMSolverMultistepScheduler")
        self.vae_path = self.config.get("vae_path") # Can be None
        self.lora_paths = self.config.get("lora_paths", []) # Defaults to empty list
        self.enable_safety_checker = self.config.get("enable_safety_checker", True)
        # Use compel only if installed AND enabled in config
        self.use_compel = self.config.get("use_compel", False) and compel_installed
        if self.config.get("use_compel", False) and not compel_installed:
             logger.warning(f"Compel requested for instance '{self.binding_instance_name}' but library not found. Prompt weighting disabled.")


        # --- Internal State Initialization ---
        self.pipeline: Optional[DiffusionPipeline] = None
        self.compel_proc: Optional[Compel] = None
        self.loaded_model_path: Optional[Path] = None # Stores Path obj of loaded model
        self.loaded_vae_path: Optional[str] = None
        self.loaded_loras: List[str] = []
        self.torch_dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

        # Determine Device and Dtype based on config
        self._determine_device_and_dtype()

        logger.info(f"Initialized DiffusersBinding '{self.binding_instance_name}': Device='{self.device}', DType='{self.torch_dtype}', Models='{self.models_folder}'")


    def _determine_device_and_dtype(self):
        """Sets the torch device and dtype based on config and availability."""
        # (Logic remains the same, uses self.device_str, self.use_fp16 etc. read in __init__)
        if self.device_str == "auto":
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        elif self.device_str == "cuda" and torch.cuda.is_available(): self.device = torch.device("cuda")
        elif self.device_str == "mps" and torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")

        # Determine dtype based on device and config preferences
        if self.device != torch.device("cpu"):
            # Prioritize bf16 if enabled and supported
            if self.use_bf16 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                 self.torch_dtype = torch.bfloat16
                 logger.info("Using bfloat16 precision (CUDA).")
            elif self.use_bf16 and self.device.type == 'mps': # MPS also supports bf16
                 self.torch_dtype = torch.bfloat16
                 logger.info("Using bfloat16 precision (MPS).")
            # Fallback to fp16 if enabled
            elif self.use_fp16:
                self.torch_dtype = torch.float16
                logger.info("Using float16 precision.")
            # Default to float32 if neither fp16 nor bf16 enabled/supported
            else:
                self.torch_dtype = torch.float32
                logger.info("Using float32 precision.")
        else: # CPU
            self.torch_dtype = torch.float32
            if self.use_fp16 or self.use_bf16:
                logger.warning("fp16/bf16 not recommended or supported on CPU. Using float32.")
            logger.info("Using CPU device with float32 precision.")


    # get_binding_config is REMOVED - Handled by binding_card.yaml

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        # TODO: Enhance this based on the specific pipeline loaded (SDXL, Img2Img, etc.)
        # For now, assume text input is always possible, image depends on pipeline type
        modalities = ["text"]
        if isinstance(self.pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)): # Add other types that take image input
             # Placeholder: Need better check for img2img/inpaint capability
             # For now, just add 'image' if it's a standard pipeline
             # modalities.append("image") # Uncomment when img2img is handled
             pass
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ["image"] # Diffusers pipelines generate images


    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Scans the models_folder for diffusers model directories."""
        # (Implementation remains the same - relies on self.models_folder set in __init__)
        logger.info(f"Diffusers '{self.binding_instance_name}': Scanning for models in {self.models_folder}")
        if not self.models_folder.is_dir():
            logger.error(f"Models folder not found for instance '{self.binding_instance_name}': {self.models_folder}")
            return []

        available_models = []
        try:
            for item in self.models_folder.iterdir():
                if item.is_dir() and not item.name.startswith('.'): # Ignore hidden dirs
                    # More robust check for a diffusers model directory
                    if (item / "model_index.json").exists() or \
                       ((item / "unet").is_dir() and (item / "unet" / "config.json").exists()) or \
                       ((item / "scheduler").is_dir() and (item / "scheduler" / "scheduler_config.json").exists()):
                        try:
                            stat_info = item.stat()
                            model_data = {
                                "name": item.name, # Use directory name as the model ID
                                "size": None, # Directory size is hard to calculate quickly
                                "modified_at": datetime.fromtimestamp(stat_info.st_mtime),
                                "format": "diffusers",
                                "family": None, # Hard to determine reliably from directory
                                "supports_vision": True, # Assumed TTI
                                "supports_audio": False,
                                "details": {"path": str(item.resolve())} # Store absolute path
                            }
                            available_models.append(model_data)
                        except Exception as e:
                            logger.warning(f"Could not process potential model dir {item.name}: {e}")
        except Exception as scan_err:
            logger.error(f"Error scanning models folder {self.models_folder}: {scan_err}", exc_info=True)


        logger.info(f"Diffusers '{self.binding_instance_name}': Found {len(available_models)} potential model directories.")
        return available_models

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Diffusers models typically require GPU unless device is CPU."""
        # Return estimate based on initialized device
        return {"gpu_required": self.device != torch.device("cpu"), "estimated_vram_mb": 4096} # Default estimate


    async def health_check(self) -> Tuple[bool, str]:
        """Checks if torch and diffusers are installed and device is accessible."""
        # (Implementation remains the same - relies on self.device set in __init__)
        if not torch_installed: return False, "PyTorch library not installed."
        if not diffusers_installed: return False, "Diffusers library not installed."
        try:
            if self.device and self.device.type == 'cuda':
                if not torch.cuda.is_available(): return False, "CUDA specified but not available."
                torch.cuda.get_device_name(0) # Try accessing device
            elif self.device and self.device.type == 'mps':
                 if not torch.backends.mps.is_available(): return False, "MPS specified but not available."
                 torch.tensor([1], device=self.device) # Simple tensor op to check MPS
            elif self.device and self.device.type == 'cpu':
                 pass # CPU is generally available
            else:
                 return False, f"Configured device '{self.device}' is not recognized or available."
            return True, f"Torch ({torch.__version__}) & Diffusers available. Device '{self.device}' OK."
        except Exception as e:
            logger.error(f"Diffusers health check failed for instance '{self.binding_instance_name}': {e}", exc_info=True)
            return False, f"Health check failed: {e}"

    async def load_model(self, model_name_or_path: str) -> bool:
        """Loads the diffusion pipeline from local path or Hub ID."""
        # --- Determine target path (local or Hub ID) ---
        target_path_obj = self.models_folder / model_name_or_path
        load_from: Union[str, Path]

        if target_path_obj.is_dir():
             load_from = target_path_obj
             logger.info(f"Diffusers '{self.binding_instance_name}': Found model locally at {load_from}")
        else:
             # Check if it looks like a Hub ID (contains '/') or is an absolute path
             if "/" in model_name_or_path or Path(model_name_or_path).is_absolute():
                 load_from = model_name_or_path # Treat as Hub ID or absolute path
                 logger.info(f"Diffusers '{self.binding_instance_name}': Attempting load '{load_from}' as Hub ID or absolute path.")
             else:
                 logger.error(f"Diffusers model directory not found locally for '{model_name_or_path}' in {self.models_folder}, and it doesn't look like a Hub ID.")
                 return False

        async with self._load_lock:
            # Check if already loaded
            # Store Path object for local models, string for Hub IDs
            current_identifier = Path(load_from).resolve() if isinstance(load_from, Path) else load_from
            previous_identifier = self.loaded_model_path.resolve() if isinstance(self.loaded_model_path, Path) else self.loaded_model_path

            if self._model_loaded and previous_identifier == current_identifier:
                logger.info(f"Diffusers '{self.binding_instance_name}': Model '{model_name_or_path}' is already loaded.")
                return True
            elif self._model_loaded:
                logger.info(f"Diffusers '{self.binding_instance_name}': Switching model. Unloading '{self.model_name}' first...")
                await self.unload_model() # Calls unload within the lock

            logger.info(f"Diffusers '{self.binding_instance_name}': Loading model '{model_name_or_path}'...")
            needs_gpu = self.device != torch.device("cpu")
            model_id_for_log = Path(load_from).name if isinstance(load_from, Path) else load_from
            resource_context = self.resource_manager.acquire_gpu_resource(f"load_{self.binding_instance_name}_{model_id_for_log}") if needs_gpu else nullcontext()

            try:
                async with resource_context:
                    if needs_gpu: logger.info(f"Diffusers '{self.binding_instance_name}': GPU resource acquired.")

                    load_args = {
                        "pretrained_model_name_or_path": str(load_from), # Must be string for from_pretrained
                        "torch_dtype": self.torch_dtype,
                        # "use_safetensors": True, # Auto-detected
                    }

                    # --- Load VAE (uses self.vae_path from __init__) ---
                    vae = None
                    if self.vae_path:
                        try:
                            logger.info(f"Loading custom VAE from: {self.vae_path}")
                            # Use asyncio.to_thread for potentially blocking VAE load
                            vae = await asyncio.to_thread(
                                AutoencoderKL.from_pretrained, self.vae_path, torch_dtype=self.torch_dtype
                            )
                            load_args["vae"] = vae
                            self.loaded_vae_path = self.vae_path # Store what was loaded
                        except Exception as e:
                            logger.error(f"Failed load custom VAE '{self.vae_path}': {e}. Using default VAE.", exc_info=True)
                            self.loaded_vae_path = None
                    else: self.loaded_vae_path = None

                    # --- Load Pipeline ---
                    logger.info("Instantiating diffusion pipeline...")
                    # Use asyncio.to_thread for the potentially long synchronous load
                    # Use DiffusionPipeline.from_pretrained for flexibility
                    self.pipeline = await asyncio.to_thread(
                        DiffusionPipeline.from_pretrained, **load_args
                    )
                    if not self.pipeline: raise RuntimeError("Pipeline loading returned None.")
                    logger.info(f"Pipeline loaded successfully. Type: {type(self.pipeline).__name__}")

                    # --- Load Scheduler ---
                    scheduler_cls = SCHEDULER_MAP.get(self.default_scheduler_type)
                    if scheduler_cls:
                        logger.info(f"Setting scheduler to {self.default_scheduler_type}")
                        try:
                            # Use asyncio.to_thread for scheduler loading
                            self.pipeline.scheduler = await asyncio.to_thread(
                                scheduler_cls.from_config, self.pipeline.scheduler.config
                            )
                        except Exception as e:
                            logger.error(f"Failed to set scheduler to {self.default_scheduler_type}: {e}", exc_info=True)
                    else:
                        logger.warning(f"Scheduler type '{self.default_scheduler_type}' not found in SCHEDULER_MAP. Using pipeline default.")


                    # --- Move to Device ---
                    logger.info(f"Moving pipeline to device: {self.device}")
                    # Use asyncio.to_thread for moving potentially large model
                    await asyncio.to_thread(self.pipeline.to, self.device)

                    # --- Load LoRAs (uses self.lora_paths from __init__) ---
                    self.loaded_loras = []
                    if self.lora_paths:
                        logger.info(f"Loading {len(self.lora_paths)} LoRA(s)...")
                        for lora_path_or_id in self.lora_paths:
                            try:
                                logger.info(f" -> Loading LoRA: {lora_path_or_id}")
                                if hasattr(self.pipeline, "load_lora_weights"):
                                    # Use asyncio.to_thread for LoRA loading
                                    await asyncio.to_thread(self.pipeline.load_lora_weights, lora_path_or_id)
                                    self.loaded_loras.append(lora_path_or_id)
                                else:
                                     logger.warning(f"Pipeline type {type(self.pipeline)} does not support load_lora_weights.")
                                     break
                            except Exception as e:
                                logger.error(f"Failed to load LoRA '{lora_path_or_id}': {e}", exc_info=True)
                        # Optional: Fuse LoRAs after loading if desired/supported
                        # if self.loaded_loras and hasattr(self.pipeline, "fuse_lora"):
                        #    logger.info("Fusing LoRA weights...")
                        #    await asyncio.to_thread(self.pipeline.fuse_lora)

                    # --- Safety Checker (uses self.enable_safety_checker from __init__) ---
                    if not self.enable_safety_checker:
                         if hasattr(self.pipeline, "safety_checker") and self.pipeline.safety_checker is not None:
                              logger.info("Disabling safety checker.")
                              self.pipeline.safety_checker = None
                              if hasattr(self.pipeline, "requires_safety_checker"):
                                  self.pipeline.requires_safety_checker = False
                         else: logger.info("No safety checker found or already disabled.")


                    # --- Compel (uses self.use_compel from __init__) ---
                    self.compel_proc = None # Reset first
                    if self.use_compel and Compel and self.pipeline:
                        logger.info("Initializing Compel for prompt weighting...")
                        try:
                            # Prepare tokenizers and encoders, handling potential SDXL structure
                            tokenizer1 = getattr(self.pipeline, 'tokenizer', None)
                            tokenizer2 = getattr(self.pipeline, 'tokenizer_2', None)
                            text_encoder1 = getattr(self.pipeline, 'text_encoder', None)
                            text_encoder2 = getattr(self.pipeline, 'text_encoder_2', None)

                            if tokenizer1 and text_encoder1: # Check if primary exists
                                tokenizers = [tokenizer1]
                                text_encoders = [text_encoder1]
                                requires_pooled = [False] # Default for single encoder

                                if tokenizer2 and text_encoder2: # SDXL case
                                    tokenizers.append(tokenizer2)
                                    text_encoders.append(text_encoder2)
                                    requires_pooled = [False, True] # SDXL requires pooled from second encoder

                                self.compel_proc = await asyncio.to_thread(
                                    Compel,
                                    tokenizer=tokenizers,
                                    text_encoder=text_encoders,
                                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, # Common setting
                                    requires_pooled=requires_pooled
                                )
                                logger.info("Compel initialized.")
                            else:
                                 logger.error("Compel requires at least one tokenizer and text_encoder in the pipeline. Disabling.")
                                 self.compel_proc = None
                                 self.use_compel = False # Disable if components missing

                        except Exception as e:
                            logger.error(f"Failed to initialize Compel: {e}. Disabling.", exc_info=True)
                            self.compel_proc = None
                            self.use_compel = False


                    # --- Finalize Load ---
                    self.loaded_model_path = Path(load_from) if isinstance(load_from, Path) else Path(load_from) # Store identifier
                    self.model_name = model_id_for_log # Use name derived from path/id
                    self._model_loaded = True
                    logger.info(f"Diffusers model '{self.model_name}' loaded successfully.")
                    return True

            except asyncio.TimeoutError:
                logger.error(f"Diffusers '{self.binding_instance_name}': Timeout waiting for GPU to load model {model_id_for_log}")
                await self.unload_model() # Attempt cleanup
                return False
            except Exception as e:
                logger.error(f"Diffusers '{self.binding_instance_name}': Failed to load model '{model_name_or_path}': {e}", exc_info=True)
                await self.unload_model() # Attempt cleanup
                return False

    async def unload_model(self) -> bool:
        """Unloads the pipeline and clears GPU memory."""
        # (Implementation remains the same)
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Diffusers '{self.binding_instance_name}': Unloading model '{self.model_name}'...")
            try:
                # Explicitly delete references
                del self.pipeline
                del self.compel_proc
                self.pipeline = None
                self.compel_proc = None
                self.loaded_model_path = None
                self.model_name = None
                self.loaded_vae_path = None
                self.loaded_loras = []
                if self.device and self.device.type == 'cuda':
                    logger.info("Clearing CUDA cache...")
                    await asyncio.to_thread(torch.cuda.empty_cache)
                self._model_loaded = False
                logger.info(f"Diffusers model unloaded for instance '{self.binding_instance_name}'.")
                return True
            except Exception as e:
                logger.error(f"Error during Diffusers unload for '{self.binding_instance_name}': {e}", exc_info=True)
                # Still try to mark as unloaded
                self.pipeline = None; self.compel_proc = None; self.loaded_model_path = None; self.model_name = None; self._model_loaded = False
                return False


    def _prepare_diffusers_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps generation parameters to diffusers pipeline arguments."""
        # (Implementation largely the same, uses instance defaults if params missing)
        pipe_params = {}
        # Common params - use int/float directly, assume validation happened upstream
        pipe_params["num_inference_steps"] = params.get("num_inference_steps", 30)
        pipe_params["guidance_scale"] = params.get("guidance_scale", 7.5)
        pipe_params["negative_prompt"] = params.get("negative_prompt", None) # Allow None negative prompt
        pipe_params["width"] = params.get("width", 512)
        pipe_params["height"] = params.get("height", 512)

        # Handle seed
        seed = params.get("seed", -1)
        if isinstance(seed, int) and seed != -1:
            pipe_params["generator"] = torch.Generator(device=self.device).manual_seed(seed)
            pipe_params["seed_used"] = seed # Store the actual seed used
        else:
            # Generate a random seed if -1 or invalid type, but store it
            random_seed = random.randint(0, 2**32 - 1)
            pipe_params["generator"] = torch.Generator(device=self.device).manual_seed(random_seed)
            pipe_params["seed_used"] = random_seed
            logger.debug(f"Using random seed: {random_seed}")


        # Add LoRA scale if applicable and LoRAs are loaded
        if "lora_scale" in params and self.loaded_loras:
             try:
                 scale_value = float(params["lora_scale"])
                 # Diffusers <0.26 used cross_attention_kwargs, >=0.26 uses guidance_scale directly (?)
                 # Let's use cross_attention_kwargs for broader compatibility for now
                 pipe_params["cross_attention_kwargs"] = {"scale": scale_value}
                 logger.debug(f"Applying LoRA scale via cross_attention_kwargs: {scale_value}")
             except ValueError:
                  logger.warning(f"Invalid LoRA scale value: {params['lora_scale']}. Ignoring.")

        # TODO: Add mapping for img2img specific params (image, strength)
        # TODO: Add mapping for inpainting params (image, mask_image)


        # Log the parameters being used
        loggable_params = {k: v for k, v in pipe_params.items() if k != 'generator'}
        logger.debug(f"Prepared Pipeline Params for instance '{self.binding_instance_name}': {loggable_params}")
        return pipe_params

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None # For img2img etc. later
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]: # Return List[OutputData]-like
        """Generates an image using the loaded diffusers pipeline."""
        if not self.pipeline or not self._model_loaded:
            raise RuntimeError(f"Diffusers model '{self.model_name}' not loaded in binding '{self.binding_instance_name}'.")

        logger.info(f"Diffusers '{self.binding_instance_name}': Generating image...")

        pipe_params = self._prepare_diffusers_params(params)
        final_seed_used = pipe_params.pop("seed_used", -1) # Get seed and remove from pipe args

        # --- Handle prompt weighting with Compel ---
        if self.use_compel and self.compel_proc:
            logger.info("Using Compel for prompt weighting.")
            try:
                logger.debug(f"Compel Input Prompt: {prompt}")
                # Use asyncio.to_thread for Compel processing
                conditioning, pooled = await asyncio.to_thread(self.compel_proc, prompt)

                # Adapt based on pipeline type (check for SDXL structure)
                if hasattr(self.pipeline,"text_encoder_2") and pooled is not None:
                     pipe_params["prompt_embeds"] = conditioning[0]
                     pipe_params["pooled_prompt_embeds"] = pooled[0]
                     logger.debug("Applied SDXL prompt embeds via Compel.")
                else: # Standard SD 1.5/2.1
                     pipe_params["prompt_embeds"] = conditioning
                     logger.debug("Applied standard prompt embeds via Compel.")


                # Handle negative prompt weighting too
                neg_prompt = pipe_params.get("negative_prompt", None)
                if neg_prompt:
                     logger.debug(f"Compel Input Negative Prompt: {neg_prompt}")
                     neg_conditioning, neg_pooled = await asyncio.to_thread(self.compel_proc, neg_prompt)
                     if hasattr(self.pipeline,"text_encoder_2") and neg_pooled is not None:
                         pipe_params["negative_prompt_embeds"] = neg_conditioning[0]
                         pipe_params["negative_pooled_prompt_embeds"] = neg_pooled[0]
                         logger.debug("Applied SDXL negative prompt embeds via Compel.")
                     else:
                         pipe_params["negative_prompt_embeds"] = neg_conditioning
                         logger.debug("Applied standard negative prompt embeds via Compel.")
                # Remove raw prompts if embeds are used
                pipe_params.pop("prompt", None) # Remove standard prompt field
                pipe_params.pop("negative_prompt", None) # Remove standard negative prompt field

            except Exception as e:
                logger.error(f"Compel processing failed: {e}. Falling back to standard prompt.", exc_info=True)
                pipe_params["prompt"] = prompt # Fallback to raw prompt
                pipe_params.pop("prompt_embeds", None); pipe_params.pop("pooled_prompt_embeds", None)
                pipe_params.pop("negative_prompt_embeds", None); pipe_params.pop("negative_pooled_prompt_embeds", None)
        else:
            pipe_params["prompt"] = prompt # Use raw prompt if Compel disabled/failed/not installed

        # --- TODO: Handle multimodal input (image, mask) for img2img/inpainting ---
        # Example placeholder logic:
        # if multimodal_data:
        #     input_image_item = next((item for item in multimodal_data if item.role == 'input_image'), None)
        #     mask_image_item = next((item for item in multimodal_data if item.role == 'mask_image'), None)
        #     if input_image_item and Image:
        #         try:
        #             input_img_bytes = base64.b64decode(input_image_item.data)
        #             pipe_params["image"] = Image.open(BytesIO(input_img_bytes)).convert("RGB")
        #             pipe_params["strength"] = float(params.get("strength", 0.8)) # Example strength param
        #             logger.info("Added input image for Img2Img.")
        #             if mask_image_item and hasattr(self.pipeline, 'inpaint'): # Check if pipeline supports inpainting
        #                 mask_img_bytes = base64.b64decode(mask_image_item.data)
        #                 pipe_params["mask_image"] = Image.open(BytesIO(mask_img_bytes)).convert("L") # Usually needs Luminance
        #                 logger.info("Added mask image for Inpainting.")
        #         except Exception as img_e:
        #             logger.error(f"Failed to process input/mask image: {img_e}")
        #             raise ValueError("Invalid input image data provided.") from img_e


        # Acquire GPU resource only if on GPU
        needs_gpu = self.device != torch.device("cpu")
        resource_context = self.resource_manager.acquire_gpu_resource(f"generate_{self.binding_instance_name}_{self.model_name}") if needs_gpu else nullcontext()

        try:
            async with resource_context:
                if needs_gpu: logger.info(f"Diffusers '{self.binding_instance_name}': GPU resource acquired for generation.")

                # --- Run Inference ---
                logger.info("Starting image generation inference...")
                # Use asyncio.to_thread for the blocking pipeline call
                # Remove 'seed_used' before calling pipeline
                pipe_params_for_call = {k: v for k, v in pipe_params.items() if k != 'seed_used'}
                output = await asyncio.to_thread(self.pipeline, **pipe_params_for_call)
                logger.info("Image generation inference finished.")
                # --------------------

            if not output.images: raise RuntimeError("Generation failed: No images returned by pipeline.")
            image: Image.Image = output.images[0] # Get the first generated image

            # Handle safety checker results
            safety_results = {}
            # Check for the attribute existence robustly
            nsfw_detected_list = getattr(output, "nsfw_content_detected", None)
            if isinstance(nsfw_detected_list, list) and nsfw_detected_list:
                 # Check if *any* image in the batch was flagged (we only use the first image here)
                 nsfw_flag = nsfw_detected_list[0] if len(nsfw_detected_list)>0 else False
                 safety_results["nsfw_content_detected"] = nsfw_flag
                 if nsfw_flag and self.enable_safety_checker:
                     logger.warning("Potential NSFW content detected by safety checker.")
                     # Optional: return placeholder or specific error?
            elif nsfw_detected_list is not None: # Log if attribute exists but isn't a list
                logger.warning(f"Unexpected format for nsfw_content_detected: {type(nsfw_detected_list)}")


            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            logger.info(f"Diffusers '{self.binding_instance_name}': Image generated successfully.")

            # Prepare metadata
            output_metadata = {
                "prompt_used": prompt, # Store original prompt
                "negative_prompt": params.get("negative_prompt", ""),
                "model": self.model_name,
                "vae": self.loaded_vae_path,
                "loras": self.loaded_loras,
                "scheduler": type(self.pipeline.scheduler).__name__,
                "steps": pipe_params["num_inference_steps"],
                "cfg_scale": pipe_params["guidance_scale"],
                "seed": final_seed_used, # Use the seed actually used
                "size": f"{pipe_params['width']}x{pipe_params['height']}",
                **safety_results
            }

            # Return result in standard List[OutputData]-like format
            return [{
                "type": "image",
                "data": image_base64,
                "mime_type": "image/png",
                "metadata": output_metadata
            }]

        except torch.cuda.OutOfMemoryError as e:
             logger.error(f"Diffusers '{self.binding_instance_name}': CUDA Out of Memory Error during generation: {e}")
             raise RuntimeError("CUDA Out of Memory. Try reducing image size or batch size.") from e
        except asyncio.TimeoutError:
            logger.error(f"Diffusers '{self.binding_instance_name}': Timeout waiting for GPU resource for generation.")
            raise RuntimeError("Timeout waiting for GPU resource for generation.")
        except Exception as e:
            logger.error(f"Diffusers '{self.binding_instance_name}': Unexpected error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during image generation: {e}") from e


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Simulates streaming for image generation by yielding info then final result."""
        logger.info(f"Diffusers Binding '{self.binding_instance_name}': Simulating stream for image generation.")
        # Yield an initial info message
        yield {"type": "info", "content": {"status": "starting_image_generation", "model": self.model_name or 'Unknown', "prompt": prompt}}

        try:
            # Call the non-streaming method to get the result (which is List[OutputData]-like)
            image_result_list = await self.generate(
                prompt=prompt,
                params=params,
                request_info=request_info,
                multimodal_data=multimodal_data
            )

            # Check if generate returned the expected list format
            if isinstance(image_result_list, list):
                # Yield the final chunk containing the List[OutputData]-like structure
                yield {"type": "final", "content": image_result_list, "metadata": {"status": "success"}}
                logger.info(f"Diffusers Binding '{self.binding_instance_name}': Simulated stream finished successfully.")
            else: # Handle unexpected return type from generate
                 raise TypeError(f"generate() returned unexpected type: {type(image_result_list)}")

        except (ValueError, RuntimeError, Exception) as e:
            logger.error(f"Diffusers Binding '{self.binding_instance_name}': Error during simulated stream's generate call: {e}", exc_info=True)
            error_content = f"Image generation failed: {str(e)}"
            yield {"type": "error", "content": error_content}
            # Yield a final chunk indicating failure
            yield {"type": "final", "content": [{"type": "error", "data": error_content}], "metadata": {"status": "failed"}}

    # --- Tokenizer / Info Methods (Not Applicable) ---
    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenization not typically applicable/useful for image generation binding."""
        raise NotImplementedError("Diffusers binding does not support text tokenization.")

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenization not applicable for image generation binding."""
        raise NotImplementedError("Diffusers binding does not support text detokenization.")

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded model pipeline."""
        if not self._model_loaded or not self.model_name or not self.pipeline:
             # Return defaults or empty if nothing loaded
             return { "name": None, "context_size": None, "max_output_tokens": None, "supports_vision": True, "supports_audio": False, "details": {} }

        details = {
            "pipeline_type": type(self.pipeline).__name__,
            "model_identifier": str(self.loaded_model_path) if self.loaded_model_path else self.model_name, # Show path if loaded locally
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "vae": self.loaded_vae_path or "Default",
            "loras": self.loaded_loras,
            "scheduler": type(self.pipeline.scheduler).__name__,
            "compel_enabled": self.use_compel and self.compel_proc is not None,
            "safety_checker_enabled": self.enable_safety_checker and hasattr(self.pipeline, 'safety_checker') and self.pipeline.safety_checker is not None
        }
        # Safely get tokenizer/encoder class names if they exist
        for i, name in enumerate(["tokenizer", "tokenizer_2"]):
             tok = getattr(self.pipeline, name, None)
             if tok: details[f"tokenizer_{i+1}_type"] = type(tok).__name__
        for i, name in enumerate(["text_encoder", "text_encoder_2"]):
             enc = getattr(self.pipeline, name, None)
             if enc: details[f"text_encoder_{i+1}_type"] = type(enc).__name__


        return {
            "name": self.model_name,
            "context_size": None, # Diffusers models don't have a single 'context_size' like LLMs
            "max_output_tokens": None, # Not applicable
            "supports_vision": True, # Assumes TTI/I2I
            "supports_audio": False,
            "details": details
        }