# zoos/bindings/diffusers_binding.py
import asyncio
import ascii_colors as logging
import sys
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from io import BytesIO
import importlib.util

# Use pipmaster to check/install dependencies
import pipmaster as pm

pm.ensure_packages({
    "torch":"",
    "torchvision":"",
    "torchaudio ":""
}, index_url="https://download.pytorch.org/whl/cu126")

pm.ensure_packages({
    "transformers":"",
    "diffusers":"",
    "accelerate":"",
    "safetensors":"",
    "invisible_watermark":"",
    "pillow":"",
    "compel":">=2.0",
})

# Check if core libraries are available after install attempt
diffusers_installed = importlib.util.find_spec("diffusers") is not None
torch_installed = importlib.util.find_spec("torch") is not None
compel_installed = importlib.util.find_spec("compel") is not None

if diffusers_installed and torch_installed:
    import torch
    from diffusers import (
        DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
        LCMScheduler # Add more schedulers as needed
    )
    from diffusers.utils.import_utils import is_invisible_watermark_available
    from PIL import Image
    if compel_installed:
        try: from compel import Compel, ReturnedEmbeddingsType; logger.info("Compel library loaded successfully.")
        except ImportError: compel_installed = False; Compel = None; ReturnedEmbeddingsType = None; logger.warning("Compel installed but failed to import. Prompt weighting disabled.") # type: ignore
    else: Compel = None; ReturnedEmbeddingsType = None # type: ignore
else:
    # Create dummy types if core libs missing, prevents runtime errors but binding fails later
    torch = None; DiffusionPipeline = None; AutoencoderKL = None; StableDiffusionPipeline = None; StableDiffusionXLPipeline = None; DPMSolverMultistepScheduler = None; EulerAncestralDiscreteScheduler = None; EulerDiscreteScheduler = None; LCMScheduler=None; Image = None; Compel = None; ReturnedEmbeddingsType = None # type: ignore
    is_invisible_watermark_available = lambda: False


from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
try:
    from lollms_server.api.models import StreamChunk, InputData, ModelInfo
except ImportError:
     class StreamChunk: pass # type: ignore
     class InputData: pass # type: ignore
     class ModelInfo: pass # type: ignore


logger = logging.getLogger(__name__)

SCHEDULER_MAP = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "LCMScheduler": LCMScheduler,
    # Add other diffusers schedulers here by their class name string
}

class DiffusersBinding(Binding):
    """Binding for Stable Diffusion models using the diffusers library."""
    binding_type_name = "diffusers_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the DiffusersBinding."""
        super().__init__(config, resource_manager)

        if not diffusers_installed or not torch_installed:
            raise ImportError("Diffusers binding requires 'torch' and 'diffusers' libraries.")

        # Configuration Loading
        self.models_folder = Path(self.config.get("models_folder", "models/diffusers_models/"))
        if not self.models_folder.exists():
            logger.warning(f"Diffusers models folder does not exist: {self.models_folder}. Creating it.")
            try: self.models_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e: logger.error(f"Failed to create diffusers models folder: {e}")

        self.device_str = self.config.get("device", "auto").lower()
        self.use_fp16 = self.config.get("use_fp16", True)
        self.use_bf16 = self.config.get("use_bf16", False) # bf16 usually needs Ampere+ GPUs
        self.default_scheduler_type = self.config.get("scheduler_type", "DPMSolverMultistepScheduler")
        self.vae_path = self.config.get("vae_path")
        self.lora_paths = self.config.get("lora_paths", []) # List of paths/hub IDs
        self.enable_safety_checker = self.config.get("enable_safety_checker", True)
        self.use_compel = self.config.get("use_compel", False) and compel_installed # Use compel only if installed and enabled

        # Internal State
        self.pipeline: Optional[DiffusionPipeline] = None
        self.compel_proc: Optional[Compel] = None
        self.loaded_model_path: Optional[Path] = None
        self.loaded_vae_path: Optional[str] = None
        self.loaded_loras: List[str] = []
        self.torch_dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

        # Determine Device and Dtype
        self._determine_device_and_dtype()

        logger.info(f"Initialized DiffusersBinding '{self.binding_name}': Device='{self.device}', DType='{self.torch_dtype}', Models='{self.models_folder}'")


    def _determine_device_and_dtype(self):
        """Sets the torch device and dtype based on config and availability."""
        if self.device_str == "auto":
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        elif self.device_str == "cuda" and torch.cuda.is_available(): self.device = torch.device("cuda")
        elif self.device_str == "mps" and torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")

        if self.device != torch.device("cpu"):
            if self.use_bf16 and (getattr(torch.cuda, "is_bf16_supported", lambda: False)() or self.device.type == 'mps'): # bf16 check
                self.torch_dtype = torch.bfloat16
            elif self.use_fp16:
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else: # CPU only supports float32 well
            self.torch_dtype = torch.float32
            if self.use_fp16 or self.use_bf16:
                logger.warning("fp16/bf16 not recommended for CPU. Using float32.")

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the Diffusers binding."""
        return {
            "type_name": cls.binding_type_name,
            "version": "1.0",
            "description": "Binding for local Stable Diffusion models using Hugging Face Diffusers library.",
            "supports_streaming": False, # Simulates stream for TTI
            "requirements": [
                "torch", "transformers", "diffusers>=0.20.0", "accelerate",
                "safetensors", "invisible_watermark", "pillow", "compel>=2.0"
            ],
            "config_template": {
                "type": {"type": "string", "value": cls.binding_type_name, "required":True},
                "models_folder": {"type": "string", "value": "models/diffusers_models/", "description": "Folder containing downloaded diffusers model folders.", "required": True},
                "device": {"type": "string", "value": "auto", "options":["auto", "cuda", "mps", "cpu"], "description": "Device for inference.", "required": False},
                "use_fp16": {"type": "bool", "value": True, "description": "Use float16 precision (faster, less VRAM, requires compatible GPU).", "required": False},
                "use_bf16": {"type": "bool", "value": False, "description": "Use bfloat16 precision (Ampere+ GPU or MPS, good alternative to fp16).", "required": False},
                "scheduler_type": {"type": "string", "value": "DPMSolverMultistepScheduler", "options": list(SCHEDULER_MAP.keys()), "description": "Default scheduler.", "required": False},
                "vae_path": {"type": "string", "value": None, "description": "Optional path or Hub ID for a custom VAE.", "required": False},
                "lora_paths": {"type": "list", "value": [], "description": "Optional list of paths or Hub IDs for LoRAs to load by default.", "required": False},
                "enable_safety_checker": {"type": "bool", "value": True, "description": "Enable the default safety checker.", "required": False},
                "use_compel": {"type": "bool", "value": False, "description": "Enable Compel prompt weighting (requires 'compel' install).", "required": False},
            }
        }

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        # TODO: Extend for Img2Img, Inpainting based on pipeline capabilities
        return ["text"]

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ["image"]

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Scans the models_folder for diffusers model directories."""
        logger.info(f"Diffusers '{self.binding_name}': Scanning for models in {self.models_folder}")
        if not self.models_folder.is_dir():
            logger.error(f"Models folder not found: {self.models_folder}")
            return []

        available_models = []
        for item in self.models_folder.iterdir():
            if item.is_dir():
                # Basic check: does it contain common diffuser pipeline files?
                model_index_path = item / "model_index.json"
                unet_config_path = item / "unet" / "config.json"
                if model_index_path.exists() and unet_config_path.exists():
                    try:
                        stat_info = item.stat() # Get dir stats (less useful for size)
                        # Add placeholder info - getting full details requires loading config
                        model_data = {
                            "name": item.name,
                            "size": None, # Directory size is hard to calculate quickly
                            "modified_at": datetime.fromtimestamp(stat_info.st_mtime),
                            "format": "diffusers",
                            "supports_vision": True, # It's an image model
                            "supports_audio": False,
                            "details": {"path": str(item.resolve())}
                        }
                        available_models.append(model_data)
                    except Exception as e:
                        logger.warning(f"Could not process potential model dir {item.name}: {e}")

        logger.info(f"Diffusers '{self.binding_name}': Found {len(available_models)} potential model directories.")
        return available_models

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Diffusers models typically require GPU."""
        return {"gpu_required": self.device != torch.device("cpu"), "estimated_vram_mb": 4096} # Default estimate, real usage varies wildly

    async def health_check(self) -> Tuple[bool, str]:
        """Checks if torch and diffusers are installed and device is accessible."""
        if not torch_installed: return False, "PyTorch not installed."
        if not diffusers_installed: return False, "Diffusers library not installed."
        try:
            if self.device and self.device.type == 'cuda':
                if not torch.cuda.is_available(): return False, "CUDA specified but not available."
                torch.cuda.get_device_name(0) # Try accessing device
            elif self.device and self.device.type == 'mps':
                 if not torch.backends.mps.is_available(): return False, "MPS specified but not available."
                 # Simple tensor op to check MPS
                 torch.tensor([1], device=self.device)
            return True, f"Torch ({torch.__version__}) & Diffusers available. Device '{self.device}' OK."
        except Exception as e:
            logger.error(f"Diffusers health check failed: {e}", exc_info=True)
            return False, f"Health check failed: {e}"

    async def load_model(self, model_name_or_path: str) -> bool:
        """Loads the diffusion pipeline."""
        async with self._load_lock:
            target_path = self.models_folder / model_name_or_path
            if not target_path.is_dir():
                # Allow loading from Hub ID if not found locally
                if Path(model_name_or_path).is_absolute() or "/" in model_name_or_path:
                    # Assume it's a Hub ID or absolute path
                    target_path = Path(model_name_or_path)
                    logger.info(f"Attempting to load '{model_name_or_path}' as Hub ID or absolute path.")
                else:
                    logger.error(f"Diffusers model directory not found: {target_path}")
                    return False

            if self._model_loaded and self.loaded_model_path == target_path:
                logger.info(f"Model '{target_path.name}' is already loaded.")
                return True
            elif self._model_loaded:
                logger.info(f"Switching model. Unloading '{self.loaded_model_path.name}' first...")
                await self.unload_model() # Calls unload within the lock

            logger.info(f"Diffusers '{self.binding_name}': Loading model '{target_path}'...")
            needs_gpu = self.device != torch.device("cpu")
            resource_context = self.resource_manager.acquire_gpu_resource(f"load_{self.binding_name}_{target_path.name}") if needs_gpu else nullcontext()

            try:
                async with resource_context:
                    if needs_gpu: logger.info(f"Diffusers '{self.binding_name}': GPU resource acquired.")

                    load_args = {
                        "pretrained_model_name_or_path": str(target_path),
                        "torch_dtype": self.torch_dtype,
                        # "use_safetensors": True, # Generally preferred, auto-detected
                        # "variant": "fp16" if self.torch_dtype == torch.float16 else None # Handled by torch_dtype
                    }

                    # --- Load VAE if specified ---
                    vae = None
                    if self.vae_path:
                        try:
                            logger.info(f"Loading custom VAE from: {self.vae_path}")
                            vae = AutoencoderKL.from_pretrained(self.vae_path, torch_dtype=self.torch_dtype)
                            load_args["vae"] = vae
                            self.loaded_vae_path = self.vae_path
                        except Exception as e:
                            logger.error(f"Failed to load custom VAE '{self.vae_path}': {e}. Using default VAE.")
                            self.loaded_vae_path = None
                    else: self.loaded_vae_path = None

                    # --- Load Pipeline ---
                    logger.info("Instantiating diffusion pipeline...")
                    # Use asyncio.to_thread for the potentially long synchronous load
                    self.pipeline = await asyncio.to_thread(DiffusionPipeline.from_pretrained, **load_args)
                    if not self.pipeline: raise RuntimeError("Pipeline loading failed.")

                    # --- Load Scheduler ---
                    scheduler_cls = SCHEDULER_MAP.get(self.default_scheduler_type)
                    if scheduler_cls:
                        logger.info(f"Setting scheduler to {self.default_scheduler_type}")
                        try:
                            self.pipeline.scheduler = await asyncio.to_thread(
                                scheduler_cls.from_config, self.pipeline.scheduler.config
                            )
                        except Exception as e:
                            logger.error(f"Failed to set scheduler to {self.default_scheduler_type}: {e}")
                    else: logger.warning(f"Scheduler type '{self.default_scheduler_type}' not found.")

                    # --- Move to Device ---
                    logger.info(f"Moving pipeline to device: {self.device}")
                    await asyncio.to_thread(self.pipeline.to, self.device)


                    # --- Load LoRAs ---
                    self.loaded_loras = []
                    if self.lora_paths:
                        logger.info(f"Loading {len(self.lora_paths)} LoRA(s)...")
                        for lora_path_or_id in self.lora_paths:
                            try:
                                logger.info(f"Loading LoRA: {lora_path_or_id}")
                                # Assuming LoRAs are compatible and pipeline has load_lora_weights
                                if hasattr(self.pipeline, "load_lora_weights"):
                                    await asyncio.to_thread(self.pipeline.load_lora_weights, lora_path_or_id) # Adapt args if needed
                                    self.loaded_loras.append(lora_path_or_id)
                                else:
                                     logger.warning(f"Pipeline type {type(self.pipeline)} does not support load_lora_weights.")
                                     break # Stop trying if method missing
                            except Exception as e:
                                logger.error(f"Failed to load LoRA '{lora_path_or_id}': {e}")
                        # Optional: Fuse/unfuse LoRAs here if desired / pipeline supports it
                        # if self.loaded_loras and hasattr(self.pipeline, "fuse_lora"):
                        #    logger.info("Fusing LoRA weights...")
                        #    await asyncio.to_thread(self.pipeline.fuse_lora)


                    # --- Safety Checker ---
                    if not self.enable_safety_checker:
                         if hasattr(self.pipeline, "safety_checker") and self.pipeline.safety_checker is not None:
                              logger.info("Disabling safety checker.")
                              self.pipeline.safety_checker = None
                              # Need to adjust feature extractor requirement too for some pipelines
                              if hasattr(self.pipeline, "requires_safety_checker"):
                                  self.pipeline.requires_safety_checker = False
                         else: logger.info("No safety checker found or already disabled.")


                    # --- Compel ---
                    if self.use_compel and Compel:
                        logger.info("Initializing Compel for prompt weighting...")
                        try:
                            self.compel_proc = Compel(tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2] if hasattr(self.pipeline,"tokenizer_2") and self.pipeline.tokenizer_2 else self.pipeline.tokenizer ,
                                                     text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2] if hasattr(self.pipeline,"text_encoder_2") and self.pipeline.text_encoder_2 else self.pipeline.text_encoder,
                                                     returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, # Adjust based on pipeline needs
                                                     requires_pooled=[False, True] if hasattr(self.pipeline,"text_encoder_2") and self.pipeline.text_encoder_2 else False )
                            logger.info("Compel initialized.")
                        except Exception as e:
                            logger.error(f"Failed to initialize Compel: {e}. Disabling.", exc_info=True)
                            self.compel_proc = None; self.use_compel = False # Disable if init fails
                    else: self.compel_proc = None

                    # --- Finalize ---
                    self.loaded_model_path = target_path
                    self.model_name = target_path.name # Use directory name as model name
                    self._model_loaded = True
                    logger.info(f"Diffusers model '{self.model_name}' loaded successfully.")
                    return True

            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for GPU to load model {target_path.name}")
                await self.unload_model() # Attempt cleanup
                return False
            except Exception as e:
                logger.error(f"Failed to load Diffusers model '{target_path.name}': {e}", exc_info=True)
                await self.unload_model() # Attempt cleanup
                return False

    async def unload_model(self) -> bool:
        """Unloads the pipeline and clears GPU memory."""
        async with self._load_lock:
            if not self._model_loaded: return True
            logger.info(f"Diffusers '{self.binding_name}': Unloading model '{self.model_name}'...")
            try:
                del self.pipeline
                del self.compel_proc
                # Del other components if loaded separately (VAE, etc.)
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
                logger.info("Diffusers model unloaded.")
                return True
            except Exception as e:
                logger.error(f"Error during Diffusers unload: {e}", exc_info=True)
                # Still try to mark as unloaded
                self.pipeline = None; self.compel_proc = None; self.loaded_model_path = None; self.model_name = None; self._model_loaded = False
                return False


    def _prepare_diffusers_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps generation parameters to diffusers pipeline arguments."""
        pipe_params = {}
        # Common params
        pipe_params["num_inference_steps"] = int(params.get("num_inference_steps", 30))
        pipe_params["guidance_scale"] = float(params.get("guidance_scale", 7.5))
        pipe_params["negative_prompt"] = params.get("negative_prompt", "")
        pipe_params["width"] = int(params.get("width", 512))
        pipe_params["height"] = int(params.get("height", 512))
        seed = params.get("seed", -1)
        if seed != -1:
            # Ensure seed is int. Generate if needed.
            try: seed_int = int(seed)
            except ValueError: seed_int = torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32 - 1)).seed()
            pipe_params["generator"] = torch.Generator(device=self.device).manual_seed(seed_int)
            pipe_params["seed"] = seed_int # Store the seed used
        else: pipe_params["seed"] = -1 # Indicate random seed

        # TODO: Add mapping for img2img specific params (image, strength)
        # TODO: Add mapping for inpainting params (image, mask_image)

        # Add LoRA scale if applicable (example)
        if "lora_scale" in params and self.loaded_loras:
             # Diffusers handles LoRA scale differently depending on version/pipeline
             # This might need adjustment. Common arg name: cross_attention_kwargs={"scale": float(params["lora_scale"])}
             pipe_params["cross_attention_kwargs"] = {"scale": float(params["lora_scale"])}


        # Log the parameters being used
        logger.debug(f"Prepared Pipeline Params: {pipe_params}")
        return pipe_params

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any], # Contains personality, generation_type ('tti') etc.
        multimodal_data: Optional[List['InputData']] = None # For img2img etc. later
    ) -> Union[str, Dict[str, Any]]:
        """Generates an image using the loaded diffusers pipeline."""
        if not self.pipeline or not self._model_loaded:
            raise RuntimeError(f"Diffusers model '{self.model_name}' not loaded in binding '{self.binding_name}'.")

        logger.info(f"Diffusers '{self.binding_name}': Generating image...")

        pipe_params = self._prepare_diffusers_params(params)
        final_seed = pipe_params.get("seed", -1)

        # Handle prompt weighting with Compel
        if self.use_compel and self.compel_proc:
            logger.info("Using Compel for prompt weighting.")
            try:
                # Basic Compel usage - may need refinement for SDXL etc.
                prompt_embeds, pooled = await asyncio.to_thread(self.compel_proc, prompt)
                pipe_params["prompt_embeds"] = prompt_embeds
                if pooled is not None and hasattr(self.pipeline,"text_encoder_2"): # Check if pooled is needed (e.g., SDXL)
                     pipe_params["pooled_prompt_embeds"] = pooled

                # Handle negative prompt weighting too
                neg_prompt = pipe_params.get("negative_prompt", "")
                if neg_prompt:
                     neg_prompt_embeds, neg_pooled = await asyncio.to_thread(self.compel_proc, neg_prompt)
                     pipe_params["negative_prompt_embeds"] = neg_prompt_embeds
                     if neg_pooled is not None and hasattr(self.pipeline,"text_encoder_2"):
                         pipe_params["pooled_negative_prompt_embeds"] = neg_pooled # Placeholder, might need correct arg name
            except Exception as e:
                logger.error(f"Compel processing failed: {e}. Falling back to standard prompt.", exc_info=True)
                pipe_params["prompt"] = prompt # Fallback
                # Remove embeds if they failed
                pipe_params.pop("prompt_embeds", None)
                pipe_params.pop("pooled_prompt_embeds", None)
                pipe_params.pop("negative_prompt_embeds", None)
                pipe_params.pop("pooled_negative_prompt_embeds", None)
        else:
            pipe_params["prompt"] = prompt # Use raw prompt if Compel disabled/failed

        # Remove prompt if embeds are used
        if "prompt_embeds" in pipe_params: pipe_params.pop("prompt", None)
        if "negative_prompt_embeds" in pipe_params: pipe_params.pop("negative_prompt", None)

        # TODO: Handle multimodal input (image, mask) for img2img/inpainting
        if multimodal_data:
             logger.warning("Multimodal input data provided but not yet handled by this Diffusers binding.")


        # Acquire GPU resource only if on GPU (CPU generation doesn't need lock)
        needs_gpu = self.device != torch.device("cpu")
        resource_context = self.resource_manager.acquire_gpu_resource(f"generate_{self.binding_name}_{self.model_name}") if needs_gpu else nullcontext()

        try:
            async with resource_context:
                if needs_gpu: logger.info(f"Diffusers '{self.binding_name}': GPU resource acquired for generation.")

                # Run inference in a separate thread to avoid blocking asyncio loop
                logger.info("Starting image generation inference...")
                output = await asyncio.to_thread(self.pipeline, **pipe_params)
                # Example: output = self.pipeline(prompt=prompt, **pipe_params)

            if not output.images: raise RuntimeError("Generation failed: No images returned.")
            image: Image.Image = output.images[0] # Get the first generated image

            # Handle safety checker results if enabled and present
            safety_results = {}
            if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected is not None:
                safety_results["nsfw_content_detected"] = output.nsfw_content_detected
                if output.nsfw_content_detected and self.enable_safety_checker:
                    logger.warning("Potential NSFW content detected by safety checker.")
                    # Optional: return a placeholder or raise an error?
                    # For now, just log and return the image.

            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            logger.info(f"Diffusers '{self.binding_name}': Image generated successfully.")

            # Return result in standard format
            return {
                "image_base64": image_base64,
                "mime_type": "image/png",
                "metadata": {
                    "prompt_used": prompt,
                    "negative_prompt": params.get("negative_prompt", ""),
                    "model": self.model_name,
                    "vae": self.loaded_vae_path,
                    "loras": self.loaded_loras,
                    "scheduler": type(self.pipeline.scheduler).__name__,
                    "steps": pipe_params["num_inference_steps"],
                    "cfg_scale": pipe_params["guidance_scale"],
                    "seed": final_seed,
                    "size": f"{pipe_params['width']}x{pipe_params['height']}",
                    **safety_results
                }
            }

        except torch.cuda.OutOfMemoryError as e:
             logger.error(f"CUDA Out of Memory Error during generation: {e}")
             raise RuntimeError("CUDA Out of Memory. Try reducing image size or batch size.") from e
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for GPU for generation.")
            raise RuntimeError("Timeout waiting for GPU resource for generation.")
        except Exception as e:
            logger.error(f"Diffusers '{self.binding_name}': Unexpected error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during image generation: {e}") from e


    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulates streaming for image generation."""
        logger.info(f"Diffusers Binding '{self.binding_name}': Simulating stream for image generation.")
        yield {"type": "info", "content": {"status": "starting_image_generation", "model": self.model_name, "prompt": prompt}}
        try:
            image_result_dict = await self.generate( prompt, params, request_info, multimodal_data )
            if isinstance(image_result_dict, dict) and "image_base64" in image_result_dict:
                final_output_list = [{ "type": "image", "data": image_result_dict["image_base64"], "mime_type": image_result_dict.get("mime_type", "image/png"), "metadata": image_result_dict.get("metadata", {}) }]
                yield {"type": "final", "content": final_output_list, "metadata": {"status": "success"}}
                logger.info("Simulated stream finished successfully.")
            elif isinstance(image_result_dict, dict) and "error" in image_result_dict:
                 yield {"type": "error", "content": image_result_dict["error"]}
                 yield {"type": "final", "content": [{"type": "error", "data": image_result_dict["error"]}], "metadata": {"status": "failed"}}
            else: raise TypeError(f"generate() returned unexpected type: {type(image_result_dict)}")
        except (ValueError, RuntimeError, Exception) as e:
            logger.error(f"Error during simulated stream generate call: {e}", exc_info=True)
            yield {"type": "error", "content": f"Image generation failed: {str(e)}"}
            yield {"type": "final", "content": [{"type": "error", "data": f"Image generation failed: {str(e)}"}], "metadata": {"status": "failed"}}

    # --- Tokenizer / Info Methods (Not Applicable) ---
    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenization not applicable for image generation binding."""
        raise NotImplementedError("Diffusers binding does not support text tokenization.")

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenization not applicable for image generation binding."""
        raise NotImplementedError("Diffusers binding does not support text detokenization.")

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded model."""
        if not self._model_loaded or not self.model_name: return {}
        return {
            "name": self.model_name,
            "context_size": None, # Not typically applicable
            "max_output_tokens": None, # Not applicable
            "supports_vision": True, # Input is text, output is vision
            "supports_audio": False,
            "details": {
                "path": str(self.loaded_model_path),
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "vae": self.loaded_vae_path,
                "loras": self.loaded_loras,
                "scheduler": type(self.pipeline.scheduler).__name__ if self.pipeline else None,
                "compel_enabled": self.use_compel and self.compel_proc is not None,
                "safety_checker_enabled": self.enable_safety_checker and hasattr(self.pipeline, 'safety_checker') and self.pipeline.safety_checker is not None if self.pipeline else False
            }
        }