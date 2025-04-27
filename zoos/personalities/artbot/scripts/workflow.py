# zoos/personalities/artbot/scripts/workflow.py
import ascii_colors as logging
import asyncio
from typing import Any, Dict, Optional, List, Union
from pathlib import Path

# Need Binding, BindingManager, AppConfig for type hints and access
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from lollms_server.core.bindings import Binding, BindingManager
    except ImportError: Binding = Any; BindingManager = Any # type: ignore
    try: from lollms_server.core.config import AppConfig
    except ImportError: AppConfig = Any # type: ignore
    # OutputData is needed for the return type hint
    try: from lollms_server.api.models import OutputData
    except ImportError: OutputData = Dict # type: ignore

logger = logging.getLogger(__name__)

# Keywords to trigger image generation
IMAGE_KEYWORDS = ["generate image", "draw", "create a picture", "make an image", "show me a picture"]

async def run_workflow(prompt: str, params: Dict, context: Dict) -> List[Dict[str, Any]]: # Return List[OutputData]-like
    """
    ArtBot workflow: Generates text and potentially an image if requested.
    Returns a list containing text and/or image OutputData dictionaries.
    """
    logger.info(f"ArtBot workflow started for prompt: '{prompt[:50]}...'")

    # Extract necessary components from context
    binding: Optional['Binding'] = context.get('binding') # Main TTT binding
    binding_manager: Optional['BindingManager'] = context.get('binding_manager')
    config: Optional['AppConfig'] = context.get('config')
    request_info = context.get("request_info", {}) # Original request details

    if not binding or not binding_manager or not config:
        logger.error("ArtBot workflow missing required context (binding, manager, or config).")
        return [{"type": "error", "data": "Workflow context incomplete."}]

    output_list: List[Dict[str, Any]] = [] # Store results here
    generate_image = any(keyword in prompt.lower() for keyword in IMAGE_KEYWORDS)

    # --- Step 1: Generate Initial Text Response ---
    # Always generate some text, even if an image is requested
    text_prompt_for_llm = (
        f"User asked: '{prompt}'.\n"
        f"Respond conversationally. "
        f"{'You will also attempt to generate an image for this.' if generate_image else ''}"
    )
    logger.info("ArtBot: Generating text response...")
    text_response_raw = await binding.generate(
        prompt=text_prompt_for_llm,
        params=params,
        request_info=request_info # Pass original request info
        # multimodal_data is not passed here unless ArtBot needed input images
    )

    # Standardize text output (assuming binding.generate might return str or dict)
    if isinstance(text_response_raw, dict) and "text" in text_response_raw:
        text_response = text_response_raw["text"]
    elif isinstance(text_response_raw, str):
        text_response = text_response_raw
    else:
         logger.warning(f"ArtBot: Unexpected text response format: {type(text_response_raw)}. Using empty string.")
         text_response = ""

    if text_response:
        output_list.append({"type": "text", "data": text_response.strip()})
    else:
        # Add placeholder if LLM failed text generation
        output_list.append({"type": "text", "data": "(Could not generate text response)"})


    # --- Step 2: Generate Image (if requested) ---
    if generate_image:
        logger.info("ArtBot: Image generation requested.")

        # --- a) Get TTI Binding ---
        tti_binding_name = config.defaults.tti_binding
        tti_model_name = config.defaults.tti_model # Get default TTI model too
        if not tti_binding_name:
            logger.warning("ArtBot: No default TTI binding configured in server defaults.")
            output_list.append({"type": "error", "data": "Image generation failed: No TTI binding configured."})
            return output_list # Return text + error

        tti_binding = binding_manager.get_binding(tti_binding_name)
        if not tti_binding:
            logger.error(f"ArtBot: Could not find configured TTI binding '{tti_binding_name}'.")
            output_list.append({"type": "error", "data": f"Image generation failed: Binding '{tti_binding_name}' not found."})
            return output_list # Return text + error

        # --- b) Generate Image Prompt (Simple Extraction for Example) ---
        # More sophisticated logic could involve asking the main LLM to refine the prompt
        image_prompt = prompt # Use the original user prompt directly for simplicity
        for keyword in IMAGE_KEYWORDS: image_prompt = image_prompt.replace(keyword, "", 1) # Remove trigger words
        image_prompt = image_prompt.strip()
        if not image_prompt: image_prompt = "A visually interesting abstract concept" # Fallback
        logger.info(f"ArtBot: Using image prompt: '{image_prompt}'")

        # --- c) Call TTI Binding ---
        try:
            # Use default TTI model if not overridden in request params
            # (Params here are originally for the TTT model, maybe filter TTI params?)
            tti_params = {
                 "prompt": image_prompt,
                 # Pass relevant params from original request if needed, e.g., size
                 # "size": params.get("image_size", "1024x1024") # Example
            }
            # Create a request_info dict specific for the TTI call
            tti_request_info = request_info.copy()
            tti_request_info["generation_type"] = "tti"

            logger.info(f"ArtBot: Calling TTI binding '{tti_binding_name}'...")
            # We need manage_model_loading here too!
            effective_tti_model = tti_model_name or tti_binding.model_name or "tti_default"
            async with manage_model_loading(tti_binding, effective_tti_model): # Reuse context manager
                 image_result_raw = await tti_binding.generate(
                     prompt=image_prompt,
                     params=tti_params, # Pass potentially filtered params
                     request_info=tti_request_info # Pass TTI specific info
                     # multimodal_data for TTI? Maybe if user provided input image for i2i via ArtBot
                 )

            # --- d) Process Image Result ---
            if isinstance(image_result_raw, dict) and "image_base64" in image_result_raw:
                logger.info("ArtBot: Image generated successfully by TTI binding.")
                output_list.append({
                    "type": "image",
                    "data": image_result_raw["image_base64"],
                    "mime_type": image_result_raw.get("mime_type", "image/png"), # Default mime type
                    "metadata": {
                        "prompt_used": image_prompt,
                        "model": tti_binding.model_name or effective_tti_model, # Report model used
                        **(image_result_raw.get("metadata", {})) # Include any metadata from TTI
                    }
                })
            else:
                 logger.error(f"ArtBot: TTI binding returned unexpected result: {image_result_raw}")
                 output_list.append({"type": "error", "data": "Image generation failed: Unexpected result from TTI binding."})

        except Exception as e:
            logger.error(f"ArtBot: Error during image generation: {e}", exc_info=True)
            output_list.append({"type": "error", "data": f"Image generation failed: {e}"})

    # --- Step 3: Return Combined Results ---
    logger.info(f"ArtBot workflow finished. Returning {len(output_list)} output items.")
    return output_list