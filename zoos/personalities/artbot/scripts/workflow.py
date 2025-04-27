# zoos/personalities/artbot/scripts/workflow.py
import ascii_colors as logging
import asyncio
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from lollms_server.core.generation import manage_model_loading
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
    try: from lollms_server.api.models import OutputData, InputData # Add InputData
    except ImportError: OutputData = Dict; InputData = Dict # type: ignore

logger = logging.getLogger(__name__)

# --- REMOVE KEYWORDS ---
# IMAGE_KEYWORDS = ["generate image", "draw", "create a picture", "make an image", "show me a picture"] # REMOVE THIS LINE

async def run_workflow(prompt: str, params: Dict, context: Dict) -> List[Dict[str, Any]]: # Return List[OutputData]-like
    """
    ArtBot workflow: Generates text and potentially an image if requested.
    Uses LLM yes/no check to determine image generation intent.
    Returns a list containing text and/or image OutputData dictionaries.
    """
    logger.info(f"ArtBot workflow started for prompt: '{prompt[:50]}...'")

    # Extract necessary components from context
    binding: Optional['Binding'] = context.get('binding') # Main TTT binding
    binding_manager: Optional['BindingManager'] = context.get('binding_manager')
    config: Optional['AppConfig'] = context.get('config')
    request_info = context.get("request_info", {}) # Original request details
    input_data: List['InputData'] = context.get("input_data", []) # Get original input_data

    if not binding or not binding_manager or not config:
        logger.error("ArtBot workflow missing required context (binding, manager, or config).")
        return [{"type": "error", "data": "Workflow context incomplete."}]

    output_list: List[Dict[str, Any]] = [] # Store results here

    # --- Step 1: Determine Image Generation Intent ---
    generate_image = False
    yes_no_question = f"Does the following user request *explicitly* ask to generate, create, draw, or show an image/picture?\n\nUser Request: '{prompt}'"
    logger.info("ArtBot: Asking LLM if image generation is explicitly requested...")
    try:
        # Use the main binding (usually TTT) to ask the yes/no question
        # Pass original input_data in case user provided context/images relevant to the request
        image_intent_result = await binding.ask_yes_no(
            question=yes_no_question,
            params=params, # Use similar params, maybe lower temp if not already low
            request_info=request_info,
            multimodal_data=[item for item in input_data if item.type != 'text'] # Pass non-text input
        )
        if image_intent_result is True:
            generate_image = True
            logger.info("ArtBot: LLM confirmed image generation intent.")
        elif image_intent_result is False:
             generate_image = False
             logger.info("ArtBot: LLM indicated no explicit image generation intent.")
        else: # None result
             logger.warning("ArtBot: Could not determine image generation intent from LLM yes/no response. Assuming NO image generation.")
             generate_image = False
    except NotImplementedError:
        logger.warning(f"ArtBot: Binding '{binding.binding_name}' does not support ask_yes_no. Falling back to simple keyword check (less reliable).")
        # --- Fallback to keywords if ask_yes_no not implemented ---
        IMAGE_KEYWORDS_FALLBACK = ["generate image", "draw", "create a picture", "make an image", "show me a picture"]
        generate_image = any(keyword in prompt.lower() for keyword in IMAGE_KEYWORDS_FALLBACK)
        if generate_image: logger.info("ArtBot: Keyword check indicates image generation.")
    except Exception as e:
         logger.error(f"ArtBot: Error during ask_yes_no check: {e}. Assuming NO image generation.", exc_info=True)
         generate_image = False
    # --- End Intent Check ---


    # --- Step 2: Generate Initial Text Response ---
    text_prompt_for_llm = (
        f"User asked: '{prompt}'.\n"
        f"Respond conversationally. "
        f"{'You should also be generating an image based on this request.' if generate_image else 'You are NOT generating an image for this request.'}" # Give LLM context
    )
    logger.info("ArtBot: Generating text response...")
    text_response_raw = await binding.generate(
        prompt=text_prompt_for_llm,
        params=params,
        request_info=request_info,
        multimodal_data=[item for item in input_data if item.type != 'text'] # Pass non-text input again
    )

    # Standardize text output (same as before)
    if isinstance(text_response_raw, dict) and "text" in text_response_raw:
        text_response = text_response_raw["text"]
    elif isinstance(text_response_raw, str):
        text_response = text_response_raw
    # --- Handle list output for text ---
    elif isinstance(text_response_raw, list):
        text_items = [item['data'] for item in text_response_raw if isinstance(item,dict) and item.get('type')=='text']
        text_response = "\n".join(text_items)
    # ---------------------------------
    else:
         logger.warning(f"ArtBot: Unexpected text response format: {type(text_response_raw)}. Using empty string.")
         text_response = ""

    if text_response:
        output_list.append({"type": "text", "data": text_response.strip()})
    elif not generate_image: # Only add placeholder if no image is being made either
        output_list.append({"type": "text", "data": "(Could not generate text response)"})


    # --- Step 3: Generate Image (if intent confirmed) ---
    if generate_image:
        logger.info("ArtBot: Proceeding with image generation.")
        # --- a) Get TTI Binding (same as before) ---
        tti_binding_name = config.defaults.tti_binding
        tti_model_name = config.defaults.tti_model
        if not tti_binding_name:
            logger.warning("ArtBot: No default TTI binding configured.")
            output_list.append({"type": "error", "data": "Image generation failed: No TTI binding configured."})
            return output_list

        tti_binding = binding_manager.get_binding(tti_binding_name)
        if not tti_binding:
            logger.error(f"ArtBot: Could not find configured TTI binding '{tti_binding_name}'.")
            output_list.append({"type": "error", "data": f"Image generation failed: Binding '{tti_binding_name}' not found."})
            return output_list

        # --- b) Generate Image Prompt (Ask LLM to refine) ---
        # Ask the primary binding to create a good prompt for the TTI model
        prompt_refinement_request = (
            f"Based on the user's original request below, generate a concise and descriptive prompt suitable for an image generation model (like DALL-E or Stable Diffusion). "
            f"Focus on visual details implied by the request.\n\n"
            f"Original Request: '{prompt}'\n\n"
            f"Refined Image Prompt:"
        )
        logger.info("ArtBot: Asking LLM to refine image prompt...")
        # Use lower temp for more predictable prompt generation
        prompt_params = params.copy()
        prompt_params['temperature'] = 0.3
        prompt_params['max_tokens'] = 150 # Limit prompt length

        refined_prompt_raw = await binding.generate(
            prompt=prompt_refinement_request,
            params=prompt_params,
            request_info=request_info # Pass original request context
        )

        # Extract refined prompt text
        if isinstance(refined_prompt_raw, dict) and "text" in refined_prompt_raw:
            image_prompt = refined_prompt_raw["text"].strip()
        elif isinstance(refined_prompt_raw, str):
            image_prompt = refined_prompt_raw.strip()
        # --- Handle list output ---
        elif isinstance(refined_prompt_raw, list):
            text_items = [item['data'] for item in refined_prompt_raw if isinstance(item,dict) and item.get('type')=='text']
            image_prompt = "\n".join(text_items).strip()
        # -------------------------
        else:
             logger.warning("ArtBot: Could not refine image prompt, using original.")
             image_prompt = prompt # Fallback to original if refinement fails

        # Clean up potential artifacts from LLM prompt generation
        image_prompt = image_prompt.replace("Refined Image Prompt:", "").strip()
        if not image_prompt: image_prompt = prompt # Ensure not empty

        logger.info(f"ArtBot: Using refined image prompt: '{image_prompt}'")

        # --- c) Call TTI Binding (same as before) ---
        try:
            tti_params = { "prompt": image_prompt, }
            tti_request_info = request_info.copy(); tti_request_info["generation_type"] = "tti"
            logger.info(f"ArtBot: Calling TTI binding '{tti_binding_name}'...")
            effective_tti_model = tti_model_name or tti_binding.model_name or "tti_default"

            # Import manage_model_loading if not already imported at top level
            try: from lollms_server.core.generation import manage_model_loading, ModelLoadingError
            except ImportError: manage_model_loading = None; ModelLoadingError = Exception # type: ignore

            if manage_model_loading:
                async with manage_model_loading(tti_binding, effective_tti_model):
                     image_result_raw = await tti_binding.generate( prompt=image_prompt, params=tti_params, request_info=tti_request_info )
            else: # Fallback if import fails
                 logger.warning("manage_model_loading context manager not found, calling generate directly.")
                 image_result_raw = await tti_binding.generate( prompt=image_prompt, params=tti_params, request_info=tti_request_info )

            # --- d) Process Image Result (same as before) ---
            if isinstance(image_result_raw, dict) and "image_base64" in image_result_raw:
                 logger.info("ArtBot: Image generated successfully by TTI binding.")
                 output_list.append({ "type": "image", "data": image_result_raw["image_base64"], "mime_type": image_result_raw.get("mime_type", "image/png"), "metadata": { "prompt_used": image_prompt, "model": tti_binding.model_name or effective_tti_model, **(image_result_raw.get("metadata", {})) } })
            # --- Handle list output from TTI ---
            elif isinstance(image_result_raw, list):
                 found_img_in_list = False
                 for item in image_result_raw:
                     if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                         logger.info("ArtBot: Image found in list from TTI binding.")
                         output_list.append({ "type": "image", "data": item["data"], "mime_type": item.get("mime_type", "image/png"), "metadata": { "prompt_used": image_prompt, "model": tti_binding.model_name or effective_tti_model, **(item.get("metadata", {})) } })
                         found_img_in_list = True
                         break # Take first image from list
                 if not found_img_in_list:
                     logger.error(f"ArtBot: TTI binding returned list, but no image found: {image_result_raw}")
                     output_list.append({"type": "error", "data": "Image generation failed: TTI binding returned list without image."})
            # ---------------------------------
            else:
                  logger.error(f"ArtBot: TTI binding returned unexpected result: {image_result_raw}")
                  output_list.append({"type": "error", "data": "Image generation failed: Unexpected result from TTI binding."})

        except Exception as e:
            logger.error(f"ArtBot: Error during image generation: {e}", exc_info=True)
            output_list.append({"type": "error", "data": f"Image generation failed: {e}"})

    # --- Step 4: Return Combined Results ---
    logger.info(f"ArtBot workflow finished. Returning {len(output_list)} output items.")
    # Filter out potential empty text responses if an image was successfully generated
    if any(item.get("type") == "image" for item in output_list):
        output_list = [item for item in output_list if item.get("type") != "text" or (item.get("type") == "text" and item.get("data"))]

    return output_list