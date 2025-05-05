# zoos/personalities/artbot/scripts/workflow.py
import ascii_colors as logging
import asyncio
from typing import Any, Dict, Optional, List, Union, AsyncGenerator
from pathlib import Path

# Need Binding, BindingManager, AppConfig for type hints and access
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from lollms_server.core.bindings import Binding, BindingManager
    except ImportError: Binding = Any; BindingManager = Any # type: ignore
    try: from lollms_server.core.config import ConfigGuard # Updated type hint
    except ImportError: ConfigGuard = Any # type: ignore
    # OutputData is needed for the return type hint
    try: from lollms_server.api.models import OutputData, InputData # Add InputData
    except ImportError: OutputData = Dict; InputData = Dict # type: ignore

# Import utility and potentially Binding type hint
# Already imported above, adding parse_thought_tags if it wasn't explicit
from lollms_server.utils.helpers import extract_code_blocks, parse_thought_tags

logger = logging.getLogger(__name__)


async def run_workflow(prompt: str, params: Dict, context: Dict) -> List[Dict[str, Any]]: # Return List[OutputData]-like
    """
    ArtBot workflow: Generates text and potentially an image if requested.
    Uses LLM yes/no check to determine image generation intent.
    Separates thoughts into distinct output items.
    Excludes thoughts from the image generation prompt.
    Returns a list containing text, info (for thoughts), and/or image OutputData dictionaries.
    """
    logger.info(f"ArtBot workflow started for prompt: '{prompt[:50]}...'")

    # Extract necessary components from context
    binding: Optional['Binding'] = context.get('binding') # Main TTT binding
    binding_manager: Optional['BindingManager'] = context.get('binding_manager')
    config: Optional['ConfigGuard'] = context.get('config') # Updated type hint
    request_info = context.get("request_info", {}) # Original request details
    input_data: List['InputData'] = context.get("input_data", []) # Get original input_data

    if not binding or not binding_manager or not config:
        logger.error("ArtBot workflow missing required context (binding, manager, or config).")
        return [{"type": "error", "data": "Workflow context incomplete."}]

    output_list: List[Dict[str, Any]] = [] # Store results here

    # --- Step 1: Determine Image Generation Intent ---
    # (Intent check logic remains the same as previous version)
    generate_image = False
    yes_no_question = f"Does the following user request *explicitly* ask to generate, create, draw, or show an image/picture?\n\nUser Request: '{prompt}'"
    logger.info("ArtBot: Asking LLM if image generation is explicitly requested...")
    try:
        image_intent_result = await binding.ask_yes_no(
            question=yes_no_question,
            params=params,
            request_info=request_info,
            multimodal_data=[item for item in input_data if item.type != 'text']
        )
        if image_intent_result is True: generate_image = True; logger.info("ArtBot: LLM confirmed image generation intent.")
        elif image_intent_result is False: generate_image = False; logger.info("ArtBot: LLM indicated no explicit image generation intent.")
        else: logger.warning("ArtBot: Could not determine image generation intent from LLM yes/no response. Assuming NO image generation."); generate_image = False
    except NotImplementedError:
        logger.warning(f"ArtBot: Binding '{binding.binding_instance_name}' does not support ask_yes_no. Falling back to keyword check.");
        IMAGE_KEYWORDS_FALLBACK = ["generate image", "draw", "create a picture", "make an image", "show me a picture"]
        generate_image = any(keyword in prompt.lower() for keyword in IMAGE_KEYWORDS_FALLBACK)
        if generate_image: logger.info("ArtBot: Keyword check indicates image generation.")
    except Exception as e: logger.error(f"ArtBot: Error during ask_yes_no check: {e}. Assuming NO image generation.", exc_info=True); generate_image = False
    # --- End Intent Check ---


    # --- Step 2: Generate Initial Text Response ---
    text_prompt_for_llm = (
        f"User asked: '{prompt}'.\n"
        f"Respond conversationally. "
        f"{'You should also be generating an image based on this request.' if generate_image else 'You are NOT generating an image for this request.'}"
    )
    logger.info("ArtBot: Generating text response...")
    text_response_raw = await binding.generate(
        prompt=text_prompt_for_llm,
        params=params,
        request_info=request_info,
        multimodal_data=[item for item in input_data if item.type != 'text']
    )

    # --- Extract text, thoughts, metadata ---
    text_response = ""
    text_thoughts = None
    text_metadata = {}
    # Standardize to always get list, even for single output
    standardized_text_output = []
    if isinstance(text_response_raw, list): standardized_text_output = text_response_raw
    elif isinstance(text_response_raw, dict): standardized_text_output = [text_response_raw]
    elif isinstance(text_response_raw, str): standardized_text_output = [{"type": "text", "data": text_response_raw}] # Wrap string

    first_text_item = next((item for item in standardized_text_output if isinstance(item, dict) and item.get('type') == 'text'), None)
    if first_text_item:
        # Use parse_thought_tags on the raw data from the first text item
        # Bindings should ideally return the raw text including tags if standardization didn't parse them.
        # If the binding already parsed thoughts, text_thoughts will be populated.
        raw_data = first_text_item.get("data", "")
        cleaned_data, parsed_thoughts_from_data = parse_thought_tags(raw_data)
        text_response = cleaned_data.strip()
        # Prioritize thoughts already parsed by binding, fallback to parsing here
        text_thoughts = first_text_item.get("thoughts") or parsed_thoughts_from_data
        text_metadata = first_text_item.get("metadata", {})
    else:
        logger.warning("ArtBot: Text generation did not produce a valid text output item.")

    # --- Add Text Output (if exists) ---
    if text_response:
        output_list.append({
            "type": "text",
            "data": text_response,
            "thoughts": None, # Thoughts are handled separately
            "metadata": text_metadata
        })

    # --- Add Thoughts Output (if exists) ---
    if text_thoughts:
        logger.info("ArtBot: Adding separate 'info' output for thoughts.")
        output_list.append({
            "type": "thoughts", # Use 'info' type for thoughts
            "data": text_thoughts,
            "thoughts": None, # Thoughts field is not for the content itself
            "metadata": {"content_type": "thoughts", **text_metadata} # Add marker in metadata
        })

    # Add placeholder only if no text, no thoughts, and no image planned
    if not text_response and not text_thoughts and not generate_image:
        output_list.append({"type": "text", "data": "(Could not generate text response)"})


    # --- Step 3: Generate Image (if intent confirmed) ---
    if generate_image:
        logger.info("ArtBot: Proceeding with image generation.")
        # --- a) Get TTI Binding ---
        tti_binding_name = None
        try: tti_binding_name = config.defaults.tti_binding
        except AttributeError: logger.error("ArtBot: Failed to access default TTI binding name from config.defaults.")
        if not tti_binding_name:
            logger.warning("ArtBot: No default TTI binding configured."); output_list.append({"type": "error", "data": "Image generation failed: No TTI binding configured."}); return output_list

        tti_binding = binding_manager.get_binding(tti_binding_name)
        if not tti_binding:
            logger.error(f"ArtBot: Could not find configured TTI binding '{tti_binding_name}'."); output_list.append({"type": "error", "data": f"Image generation failed: Binding '{tti_binding_name}' not found."}); return output_list

        # --- b) Generate & Clean Image Prompt ---
        prompt_refinement_request = (
            f"Based on the user's original request below, generate ONLY a concise and descriptive prompt suitable for an image generation model (like DALL-E or Stable Diffusion). " # Added ONLY
            f"Focus on visual details implied by the request. Do NOT add any commentary or formatting other than the prompt itself.\n\n" # Added instruction
            f"Original Request: '{prompt}'\n\n"
            f"Refined Image Prompt:"
        )
        logger.info("ArtBot: Asking LLM to refine image prompt...")
        prompt_params = params.copy(); prompt_params['temperature'] = 0.3; prompt_params['max_tokens'] = 150

        refined_prompt_raw = await binding.generate( prompt=prompt_refinement_request, params=prompt_params, request_info=request_info)

        # Standardize and extract CLEANED text for the image prompt
        image_prompt_raw_text = ""
        if isinstance(refined_prompt_raw, list) and refined_prompt_raw:
            first_text = next((item.get("data") for item in refined_prompt_raw if isinstance(item,dict) and item.get("type")=="text"), None)
            if first_text: image_prompt_raw_text = first_text
        elif isinstance(refined_prompt_raw, dict):
            if refined_prompt_raw.get("type") == "text": image_prompt_raw_text = refined_prompt_raw.get("data", "")
            elif "text" in refined_prompt_raw: image_prompt_raw_text = refined_prompt_raw["text"]
        elif isinstance(refined_prompt_raw, str): image_prompt_raw_text = refined_prompt_raw

        # --- IMPORTANT: Remove thoughts from the image prompt ---
        image_prompt_cleaned, refinement_thoughts = parse_thought_tags(image_prompt_raw_text)
        if refinement_thoughts:
            logger.info("ArtBot: Thoughts detected during image prompt refinement (discarded from final prompt).")
            logger.debug(f"Refinement thoughts: {refinement_thoughts[:100]}...")
        # -----------------------------------------------------

        image_prompt = image_prompt_cleaned.strip()
        if not image_prompt:
             logger.warning("ArtBot: Could not refine image prompt, using original user prompt.")
             image_prompt = prompt # Fallback to original USER prompt if refinement fails

        # Clean up potential artifacts
        image_prompt = image_prompt.replace("Refined Image Prompt:", "").strip()
        if not image_prompt: image_prompt = prompt # Ensure not empty

        logger.info(f"ArtBot: Using cleaned image prompt: '{image_prompt}'")

        # --- c) Call TTI Binding ---
        try:
            tti_params = { "prompt": image_prompt, **params.get("tti_params", {}) }
            tti_request_info = request_info.copy(); tti_request_info["generation_type"] = "tti"
            logger.info(f"ArtBot: Calling TTI binding '{tti_binding_name}'...")
            effective_tti_model = tti_binding.model_name or tti_binding.default_model_name
            if not effective_tti_model: raise ValueError(f"TTI Binding '{tti_binding_name}' has no model configured.")

            try:
                from lollms_server.core.generation import manage_model_loading, ModelLoadingError
                context_manager = manage_model_loading(tti_binding, effective_tti_model)
            except ImportError: logger.warning("manage_model_loading context manager not found."); from contextlib import nullcontext; context_manager = nullcontext(); ModelLoadingError = Exception

            async with context_manager:
                 image_result_raw = await tti_binding.generate( prompt=image_prompt, params=tti_params, request_info=tti_request_info )

            # --- d) Process Image Result ---
            # (Processing logic remains the same)
            if isinstance(image_result_raw, list) and image_result_raw:
                 found_img_in_list = False
                 for item in image_result_raw:
                     if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                         logger.info("ArtBot: Image found in list from TTI binding.")
                         output_list.append({ "type": "image", "data": item["data"], "mime_type": item.get("mime_type", "image/png"), "metadata": { "prompt_used": image_prompt, "model": tti_binding.model_name or effective_tti_model, **(item.get("metadata", {})) } })
                         found_img_in_list = True; break
                 if not found_img_in_list: logger.error(f"ArtBot: TTI binding returned list, but no image found: {image_result_raw}"); output_list.append({"type": "error", "data": "Image generation failed: TTI binding returned list without image."})
            elif isinstance(image_result_raw, dict) and image_result_raw.get("type") == "image" and "data" in image_result_raw:
                 logger.info("ArtBot: Image generated successfully by TTI binding (dict format).")
                 output_list.append({ "type": "image", "data": image_result_raw["data"], "mime_type": image_result_raw.get("mime_type", "image/png"), "metadata": { "prompt_used": image_prompt, "model": tti_binding.model_name or effective_tti_model, **(image_result_raw.get("metadata", {})) } })
            else: logger.error(f"ArtBot: TTI binding returned unexpected result format: {type(image_result_raw)}"); output_list.append({"type": "error", "data": "Image generation failed: Unexpected result format from TTI binding."})

        except ModelLoadingError as mle: logger.error(f"ArtBot: Failed to load TTI model '{effective_tti_model}' via binding '{tti_binding_name}': {mle}", exc_info=True); output_list.append({"type": "error", "data": f"Image generation failed: Could not load model '{effective_tti_model}'."})
        except Exception as e: logger.error(f"ArtBot: Error during image generation: {e}", exc_info=True); output_list.append({"type": "error", "data": f"Image generation failed: {e}"})

    # --- Step 4: Return Combined Results ---
    logger.info(f"ArtBot workflow finished. Returning {len(output_list)} output items.")
    # (Final filtering logic remains the same)
    if any(item.get("type") == "image" for item in output_list):
        output_list = [item for item in output_list if item.get("type") != "text" or (item.get("type") == "text" and item.get("data"))]

    return output_list