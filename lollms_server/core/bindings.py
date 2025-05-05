# lollms_server/core/bindings.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Manages AI model bindings, discovery, loading, and interaction, using ConfigGuard.
# Modification Date: 2025-05-04
# Refactored get_current_model_info -> get_model_info and adjusted default model handling.

import importlib
import inspect
import yaml
import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Type, Optional, AsyncGenerator, Tuple, Union, cast
import re
try:
    import ascii_colors as logging # Use logging alias
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

try:
    from configguard import ConfigGuard, ValidationError
    from configguard.exceptions import ConfigGuardError
    # Import handlers to check dependencies if needed, though ConfigGuard does this
    from configguard.handlers import JsonHandler, YamlHandler, TomlHandler, SqliteHandler
except ImportError:
    logging.critical("FATAL: ConfigGuard library not found.")
    raise

# Core components & Utils
from lollms_server.core.config import get_config, get_binding_instances_config_path, get_encryption_key, get_server_root # Use updated config getters
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.utils.file_utils import find_classes_in_module, add_path_to_sys_path, safe_load_module # Added safe_load_module
from lollms_server.utils.helpers import extract_code_blocks, parse_thought_tags # Added parse_thought_tags

# API Models for type hints (use TYPE_CHECKING)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lollms_server.api.models import InputData, OutputData, StreamChunk, GetModelInfoResponse


logger = logging.getLogger(__name__)

# Maps file extensions to their corresponding ConfigGuard handler classes.
_binding_handler_map: Dict[str, Type] = {
    ".yaml": YamlHandler, ".yml": YamlHandler,
    ".json": JsonHandler, ".toml": TomlHandler,
    ".db": SqliteHandler, ".sqlite": SqliteHandler, ".sqlite3": SqliteHandler
}
# --- Binding Base Class ---

class Binding(ABC):
    """Abstract Base Class for all generation bindings."""

    binding_type_name: str = "base_binding" # Unique type name for the binding class

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the binding instance using the loaded instance configuration.

        Args:
            config: The configuration dictionary loaded from the instance's config file.
                    This dictionary is validated against the schema defined in the
                    binding's binding_card.yaml.
            resource_manager: The shared resource manager instance.
        """
        # Store the validated instance configuration dictionary
        self.config: Dict[str, Any] = config
        self.resource_manager = resource_manager
        # Extract the instance name which is injected by BindingManager during loading
        self.binding_instance_name: str = config.get("binding_instance_name", "unknown_instance")
        # Get the instance-specific default model if configured
        self.default_model_name: Optional[str] = self.config.get("default_model")

        self.model_name: Optional[str] = None # Name of the model file/API identifier currently loaded
        self._model_loaded = False
        self._load_lock = asyncio.Lock() # Protects model loading/unloading state

        logger.debug(f"Base Binding initialized for instance '{self.binding_instance_name}' (Type: {self.binding_type_name})")

    @classmethod
    def get_binding_package_path(cls) -> Optional[Path]:
        """Finds the directory containing the binding's definition (__init__.py)."""
        try:
            module_file = inspect.getfile(cls)
            package_dir = Path(module_file).parent.resolve()
            if package_dir.is_dir():
                 return package_dir
            else:
                logger.error(f"Determined package directory is not valid: {package_dir}")
        except TypeError: logger.error(f"Could not determine file path for class {cls.__name__}")
        except Exception as e: logger.error(f"Error finding package path for {cls.__name__}: {e}")
        return None

    @classmethod
    def get_binding_card(cls) -> Dict[str, Any]:
        """
        Loads and returns the metadata and schema from the binding's binding_card.yaml.

        Returns:
            A dictionary containing the binding card data, including the 'instance_schema'.
            Returns an empty dict if the card cannot be found or loaded correctly.
        """
        package_dir = cls.get_binding_package_path()
        if not package_dir:
            logger.error(f"Cannot locate package directory for binding class '{cls.__name__}'. Cannot load card.")
            return {}

        card_path = package_dir / "binding_card.yaml"
        if not card_path.is_file():
             logger.error(f"Binding card not found for class '{cls.__name__}' at expected path: {card_path}")
             return {}

        try:
             with open(card_path, 'r', encoding='utf-8') as f:
                 card_data = yaml.safe_load(f)
             if not isinstance(card_data, dict):
                 logger.error(f"Invalid format in binding card: {card_path}. Expected a dictionary.")
                 return {}
             if "type_name" not in card_data or "instance_schema" not in card_data:
                 logger.error(f"Binding card {card_path} is missing 'type_name' or 'instance_schema'.")
                 return {}
             if not isinstance(card_data["instance_schema"], dict):
                  logger.error(f"Binding card {card_path} has invalid 'instance_schema' (not a dictionary).")
                  return {}
             card_data["package_path"] = package_dir
             # Extract supports_streaming from card
             cls.supports_streaming = card_data.get("supports_streaming", False)
             return card_data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in binding card {card_path}: {e}")
            return {}
        except Exception as e:
             logger.error(f"Error loading binding card {card_path}: {e}", exc_info=True)
             return {}

    # --- Abstract Methods (Must be implemented by subclasses) ---
    @abstractmethod
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available to *this specific binding instance*."""
        pass

    @abstractmethod
    def get_supported_input_modalities(self) -> List[str]:
        """Returns a list of supported input data types (e.g., ['text', 'image'])."""
        pass

    @abstractmethod
    def get_supported_output_modalities(self) -> List[str]:
        """Returns a list of supported output data types (e.g., ['text', 'image', 'audio'])."""
        pass

    @abstractmethod
    async def load_model(self, model_name: str) -> bool:
        """Loads the specified model into the binding."""
        pass

    @abstractmethod
    async def unload_model(self) -> bool:
        """Unloads the currently loaded model, releasing resources."""
        pass

    @abstractmethod
    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Generates output based on the prompt and optional multimodal data."""
        pass

    @abstractmethod
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns information about a specific model or the default/currently loaded one.
        If model_name is None, returns info for the current/default model.
        If model_name is provided, attempts to get info for that specific model without loading it.
        The returned dictionary should contain keys matching fields in `api.models.GetModelInfoResponse`.
        """
        pass

    # --- Common Properties and Methods ---
    @property
    def supports_streaming(self) -> bool:
        """Indicates if the binding natively supports streaming generation (based on card)."""
        # Class attribute set by get_binding_card
        return getattr(self.__class__, "supports_streaming", False)

    async def count_tokens(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> int:
        """Counts the number of tokens in the provided text using the binding's method."""
        # Pass model_name down to tokenize
        tokens = await self.tokenize(text, add_bos=add_bos, add_eos=add_eos, model_name=model_name)
        return len(tokens)

    @property
    def is_model_loaded(self)->bool:
        """Returns True if a model is considered loaded by the binding."""
        return self._model_loaded

    # --- Optional Methods (Subclasses can override) ---
    def supports_input_role(self, data_type: str, role: str) -> bool:
        """Checks if a specific type/role combination is supported. Default implementation just checks type."""
        return data_type in self.get_supported_input_modalities()

    def get_instance_config(self) -> Dict[str, Any]:
        """Returns the specific configuration dictionary for this binding instance."""
        # Return a copy to prevent external modification
        return self.config.copy()

    async def health_check(self) -> Tuple[bool, str]:
        """Performs a basic health check on the binding. Subclasses should override for specific checks (e.g., API connection)."""
        # Default implementation assumes healthy if initialized
        return True, f"Binding '{self.binding_instance_name}' initialized. No specific health check implemented."

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generates output as a stream. Default implementation simulates via non-streaming."""
        if not self.supports_streaming:
             logger.warning(f"Binding '{self.binding_instance_name}' (Type: {self.binding_type_name}) does not support native streaming. Simulating.")
        else:
             # This might happen if a binding says it supports streaming but doesn't override this method
             logger.error(f"Binding '{self.binding_instance_name}' claims to support streaming but hasn't implemented generate_stream(). Simulating.")

        full_response: Union[str, Dict[str, Any], List[Dict[str, Any]]]
        try:
            # Ensure generate() is called with the effective model name
            # This relies on the model being loaded beforehand, either via explicit request or instance default
            # We assume generate() internally uses self.model_name which is set by load_model
            full_response = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
        except Exception as e:
             logger.error(f"Error during non-stream call for stream simulation in '{self.binding_instance_name}': {e}", exc_info=True)
             yield {"type": "error", "content": f"Generation failed during simulation: {e}"}
             return

        final_output_list: List[Dict[str, Any]] = []
        # Use the standardized output helper (which now handles thoughts)
        from lollms_server.core.generation import standardize_output
        final_output_list = standardize_output(full_response)

        # Yield individual chunks if possible (only for simple text simulation)
        if len(final_output_list) == 1 and final_output_list[0].get("type") == "text":
            text_content = final_output_list[0].get("data", "")
            text_thoughts = final_output_list[0].get("thoughts") # Get thoughts from standardized output

            if text_content:
                # Simple simulation: Split by space, yield with delay
                words = text_content.split()
                for i, word in enumerate(words):
                    yield {"type": "chunk", "content": word + (" " if i < len(words) - 1 else ""), "thoughts": None} # Don't yield thoughts per chunk
                    await asyncio.sleep(0.01)
            elif text_thoughts:
                 # If only thoughts were generated, yield them in an info chunk or similar
                 yield {"type":"info", "content":"Generation resulted only in thoughts.", "thoughts": text_thoughts}
            else:
                 yield {"type":"info", "content":"Empty text result from non-stream generate."}

        elif final_output_list:
             # For non-text or complex list outputs, yield an info chunk
             yield {"type":"info", "content":"Non-streamable output generated."}
        else:
             yield {"type":"error", "content":"Non-stream generate returned empty/invalid result."}


        # Yield the standardized final chunk (including thoughts)
        final_metadata = {"status": "complete", "simulated_stream": True}
        if final_output_list:
            # Merge metadata from all output items if needed, or take from first
            final_metadata.update(final_output_list[0].get("metadata", {}))

        yield {"type": "final", "content": final_output_list, "metadata": final_metadata}
        return

    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        """Tokenizes text (Optional). Requires model_name parameter for potential model-specific tokenization."""
        raise NotImplementedError(f"Binding type '{self.binding_type_name}' does not support tokenization.")

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        """Detokenizes tokens (Optional). Requires model_name parameter."""
        raise NotImplementedError(f"Binding type '{self.binding_type_name}' does not support detokenization.")

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Estimates resource requirements (Optional). Default assumes GPU needed."""
        return {"gpu_required": True, "estimated_vram_mb": 1024}


    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Retrieves the instance configuration schema defined in the binding's binding_card.yaml.

        This method is used by tools like the configuration wizard to dynamically understand
        the settings required for creating or editing instances of this binding type.

        Returns:
            A dictionary representing the ConfigGuard schema for instance configuration,
            or an empty dictionary if the card or schema cannot be loaded.
        """
        logger.debug(f"Getting config schema for binding type: {cls.__name__}")
        try:
            # Reuse the existing method to load the card data
            card_data = cls.get_binding_card()

            if not card_data or not isinstance(card_data, dict):
                logger.error(f"Binding card data not found or invalid for {cls.__name__}. Cannot retrieve schema.")
                return {}

            # Extract the instance_schema part
            instance_schema = card_data.get("instance_schema")

            if not instance_schema or not isinstance(instance_schema, dict):
                logger.error(f"Key 'instance_schema' missing or not a dictionary in binding card for {cls.__name__}.")
                return {}

            # Add standard internal fields expected by ConfigGuard/manager if not present,
            # although they should ideally be in the card schema itself.
            # This helps ensure consistency for the wizard/editor.
            schema_copy = instance_schema.copy() # Work on a copy
            schema_copy.setdefault("__version__", {"type": "str", "default": "0.1.0", "help":"Schema version for this instance."})
            schema_copy.setdefault("type", {"type": "str", "default": cls.binding_type_name, "help": "Internal binding type identifier."})
            schema_copy.setdefault("binding_instance_name", {"type": "str", "default": "default_instance_name", "help":"Internal name assigned to this binding instance."})

            logger.debug(f"Successfully retrieved instance schema for {cls.__name__}.")
            return schema_copy

        except Exception as e:
            logger.error(f"Unexpected error retrieving config schema for {cls.__name__}: {e}", exc_info=True)
            trace_exception(e)
            return {}
        
    # --- Helpers for Structured Output (Keep these, they use self.generate) ---
    # These helpers might need slight adjustments if the binding's `generate` signature changes significantly
    # They now rely on `generate` using `self.model_name` correctly internally.
    async def generate_structured_output( self, prompt: str, structure_definition: str, output_language_tag: str = "json", params: Optional[Dict[str, Any]] = None, request_info: Optional[Dict[str, Any]] = None, multimodal_data: Optional[List['InputData']] = None, system_message_prefix: str = "Follow instructions precisely:\n" ) -> Optional[Any]:
        """Generates text and attempts to extract structured output (e.g., JSON)."""
        if params is None: params = {}
        if request_info is None: request_info = {}
        structured_system_message = ( f"{system_message_prefix}" f"Generate a response answering: '{prompt}'.\n" f"Your response MUST contain EXACTLY ONE markdown code block tagged '{output_language_tag}'.\n" f"The content inside MUST conform to:\n{structure_definition}\n" f"Output ONLY the markdown block." )
        structured_params = params.copy()
        existing_sys_msg = structured_params.get("system_message", "")
        separator = "\n\n" if existing_sys_msg else ""
        structured_params["system_message"] = f"{existing_sys_msg}{separator}{structured_system_message}"
        logger.debug(f"Generating structured output (tag: '{output_language_tag}') for instance '{self.binding_instance_name}'")
        try:
            # Generate call now passes the multimodal data down
            llm_response_list = await self.generate( prompt=prompt, params=structured_params, request_info=request_info, multimodal_data=multimodal_data )
            # Process the potentially list-based response
            response_text = ""
            if isinstance(llm_response_list, list):
                text_items = [item['data'] for item in llm_response_list if isinstance(item,dict) and item.get('type')=='text' and item.get('data')]
                response_text = "\n".join(text_items)
            elif isinstance(llm_response_list, str): # Handle direct string return (less ideal but possible)
                 response_text = llm_response_list
            else:
                 logger.warning(f"generate_structured_output unexpected type: {type(llm_response_list)}. Trying str conversion."); response_text = str(llm_response_list)

            # Extract code block using the helper
            all_blocks = extract_code_blocks(response_text)
            first_matching_block = None
            for block in all_blocks:
                lang_match = ( not output_language_tag or (block.get('type') and block['type'].lower() == output_language_tag.lower()) )
                if lang_match:
                    first_matching_block = block
                    break

            if first_matching_block is None:
                 logger.warning(f"Could not extract block '{output_language_tag}' from response. LLM Text:\n{response_text}")
                 return None

            if not first_matching_block.get('is_complete', True):
                 logger.warning(f"Extracted block '{output_language_tag}' seems incomplete.")

            extracted_content = first_matching_block.get('content', '')

            if output_language_tag.lower() == "json":
                try:
                    parsed_json = json.loads(extracted_content)
                    logger.debug("Successfully parsed JSON.")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from extracted block. Content:\n{extracted_content}\nError: {e}")
                    return None
            else:
                 logger.debug(f"Successfully extracted text content '{output_language_tag}'.")
                 return extracted_content
        except Exception as e:
             logger.error(f"Error during generate_structured_output for '{self.binding_instance_name}': {e}", exc_info=True)
             return None

    async def generate_and_extract_all_codes( self, prompt: str, params: Optional[Dict[str, Any]] = None, request_info: Optional[Dict[str, Any]] = None, multimodal_data: Optional[List['InputData']] = None ) -> List[Dict[str, Any]]:
        """Generates text and extracts ALL markdown code blocks."""
        if params is None: params = {}
        if request_info is None: request_info = {}
        logger.debug(f"Generating to extract all code blocks for instance '{self.binding_instance_name}'...")
        try:
            # Generate call now passes the multimodal data down
            llm_response_list = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
            response_text = ""
            if isinstance(llm_response_list, list):
                text_items = [item['data'] for item in llm_response_list if isinstance(item,dict) and item.get('type')=='text' and item.get('data')]
                response_text = "\n".join(text_items)
            elif isinstance(llm_response_list, str):
                 response_text = llm_response_list
            else:
                 logger.warning(f"generate_and_extract_all_codes unexpected type: {type(llm_response_list)}. Trying str conversion."); response_text = str(llm_response_list)

            return extract_code_blocks(response_text)
        except Exception as e:
             logger.error(f"Error during generate_and_extract_all_codes for '{self.binding_instance_name}': {e}", exc_info=True)
             return []

    async def ask_yes_no( self, question: str, params: Optional[Dict[str, Any]] = None, request_info: Optional[Dict[str, Any]] = None, multimodal_data: Optional[List['InputData']] = None ) -> Optional[bool]:
        """
        Asks the LLM a yes/no question, attempting robust JSON parsing first,
        then falling back to checking the start of the raw text response.

        Args:
            question: The yes/no question to ask the LLM.
            params: Generation parameters.
            request_info: Original request details.
            multimodal_data: Optional multimodal context.

        Returns:
            True if the answer is yes, False if the answer is no, None if undetermined.
        """
        if params is None: params = {}
        if request_info is None: request_info = {}

        # --- System Message Instructing Enhanced JSON Output ---
        # Still try to get JSON first, as it's the most reliable if it works.
        yes_no_system_message = (
            f"Analyze the following question regarding the user's request. Your task is to respond with ONLY a single JSON code block containing the answer and optional explanation. "
            f"The JSON object MUST have an 'answer' key with a value of either the lowercase string 'yes' or 'no'. "
            f"It MAY optionally include an 'explanation' key with a brief string value explaining your reasoning. "
            f"Do NOT include any other text, comments, or formatting outside the JSON code block.\n\n"
            f"Example Response (with explanation):\n```json\n{{\n  \"answer\": \"no\",\n  \"explanation\": \"The user asked about past events, not image generation.\"\n}}\n```\n"
            f"Example Response (without explanation):\n```json\n{{\n  \"answer\": \"yes\"\n}}\n```\n\n"
            f"Now, answer this question based on the user's request:\nQuestion: '{question}'"
        )
        # --- End System Message ---

        # Prepare parameters
        yes_no_params = params.copy()
        yes_no_params["system_message"] = yes_no_system_message
        yes_no_params.setdefault("temperature", 0.01) # Very low temp
        # Use a reasonable max token count, allowing for JSON + explanation + potential <think> tags
        global_config = get_config(); fallback_max_tokens = getattr(global_config.defaults, "default_max_output_tokens", 512)
        binding_max_tokens = None
        try: model_info = await self.get_model_info(); binding_max_tokens = model_info.get("max_output_tokens")
        except Exception: pass
        yes_no_max_tokens = params.get("max_tokens") or binding_max_tokens or fallback_max_tokens
        yes_no_params["max_tokens"] = max(100, int(yes_no_max_tokens)) # Ensure at least 100 tokens
        yes_no_params.pop("top_p", None); yes_no_params.pop("top_k", None); yes_no_params.pop("grammar", None)

        logger.debug(f"Asking yes/no (JSON->Text Fallback): '{question}' for instance '{self.binding_instance_name}'")

        try:
            # --- Step 1: Generate the response using the JSON prompt ---
            llm_response_list_or_str = await self.generate(
                prompt=question,
                params=yes_no_params,
                request_info=request_info,
                multimodal_data=multimodal_data
            )

            # --- Step 2: Extract Raw Text ---
            raw_response_text = ""
            if isinstance(llm_response_list_or_str, list):
                # Standardize: get thoughts first, then concatenate text
                all_thoughts = [item.get('thoughts') for item in llm_response_list_or_str if isinstance(item, dict) and item.get('thoughts')]
                text_items = [item['data'] for item in llm_response_list_or_str if isinstance(item, dict) and item.get('type') == 'text' and item.get('data')]
                raw_response_text = "\n".join(text_items)
                # Log thoughts if extracted
                if all_thoughts: logger.debug(f"Thoughts from yes/no generate call:\n" + "\n---\n".join(filter(None, all_thoughts)))
            elif isinstance(llm_response_list_or_str, str):
                raw_response_text = llm_response_list_or_str
            else:
                logger.warning(f"ask_yes_no unexpected raw response type: {type(llm_response_list_or_str)}. Trying str().")
                raw_response_text = str(llm_response_list_or_str)

            logger.debug(f"Raw yes/no response text from generate:\n{raw_response_text}")

            # --- Step 3: Try parsing JSON from extracted block ---
            logger.debug("Attempt 1: Parsing JSON from extracted ```json block...")
            code_blocks = extract_code_blocks(raw_response_text)
            json_block_content = None
            for block in code_blocks:
                if block.get('type', '').lower() == 'json' and block.get('content'):
                    json_block_content = block.get('content').strip()
                    logger.debug(f"Found JSON block content:\n{json_block_content}")
                    break # Use the first valid JSON block

            if json_block_content:
                try:
                    parsed_json = json.loads(json_block_content)
                    if isinstance(parsed_json, dict):
                        answer = parsed_json.get("answer")
                        if isinstance(answer, str):
                            answer_lower = answer.strip().lower()
                            if answer_lower == "yes": logger.info("Parsed yes/no from JSON block: YES"); return True
                            if answer_lower == "no": logger.info("Parsed yes/no from JSON block: NO"); return False
                            logger.warning(f"JSON block 'answer' is not 'yes' or 'no': '{answer}'.")
                        else: logger.warning(f"JSON block 'answer' is not a string: {answer}.")
                    else: logger.warning(f"Parsed JSON block content is not a dictionary: {parsed_json}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON from extracted block content: {e}. Content:\n{json_block_content}")
            else:
                logger.debug("No valid JSON block found in the response.")

            # --- Step 4: If block parsing failed, try parsing the *entire* raw response ---
            logger.debug("Attempt 2: Parsing entire raw response as JSON...")
            # Clean the raw text slightly before attempting full parse
            cleaned_for_json_parse = raw_response_text.strip()
            if cleaned_for_json_parse.startswith("```json"): cleaned_for_json_parse = cleaned_for_json_parse[len("```json"):].strip()
            if cleaned_for_json_parse.endswith("```"): cleaned_for_json_parse = cleaned_for_json_parse[:-len("```")].strip()

            try:
                parsed_json = json.loads(cleaned_for_json_parse)
                if isinstance(parsed_json, dict):
                    answer = parsed_json.get("answer")
                    if isinstance(answer, str):
                        answer_lower = answer.strip().lower()
                        if answer_lower == "yes": logger.info("Parsed yes/no from raw response JSON: YES"); return True
                        if answer_lower == "no": logger.info("Parsed yes/no from raw response JSON: NO"); return False
                        logger.warning(f"Raw response JSON 'answer' is not 'yes'/'no': '{answer}'.")
                    else: logger.warning(f"Raw response JSON 'answer' is not string: {answer}.")
                else: logger.warning(f"Raw response parsed as JSON but is not dict: {parsed_json}")
            except json.JSONDecodeError:
                logger.debug("Raw response is not valid JSON.")
            except Exception as e: # Catch other potential errors during raw parse
                logger.warning(f"Error attempting to parse raw response as JSON: {e}")

            # --- Step 5: If JSON parsing failed, fallback to text start check ---
            logger.debug("Attempt 3: Falling back to checking start of cleaned text...")
            # Use the text *after* removing any <think> tags for this check
            cleaned_text_no_thoughts, _ = parse_thought_tags(raw_response_text)
            cleaned_response = cleaned_text_no_thoughts.strip().lower()
            # Remove common introductory phrases the model might add before 'yes'/'no'
            cleaned_response = re.sub(r"^(answer:|response:|certainly[,]?|absolutely[,]?|yes[,]?|no[,]?)\s*", "", cleaned_response).strip()
            # Remove punctuation
            cleaned_response = ''.join(c for c in cleaned_response if c.isalnum() or c.isspace())
            words = cleaned_response.split()

            logger.debug(f"Cleaned words for text fallback check: {words}")

            if words:
                first_word = words[0]
                # Be slightly more lenient, check startswith as well
                if first_word == "yes": logger.info("Parsed yes/no from text fallback: YES"); return True
                if first_word == "no": logger.info("Parsed yes/no from text fallback: NO"); return False

            # --- Step 6: If all methods failed ---
            logger.error(f"Could not determine yes/no answer for '{self.binding_instance_name}'. Raw response:\n{raw_response_text}")
            return None

        except Exception as e:
             logger.error(f"Error during JSON/Text-based ask_yes_no for '{self.binding_instance_name}': {e}", exc_info=True)
             return None
    # --- End Structured Output Helpers ---


# --- Binding Manager ---

class BindingManager:
    """Discovers binding types, loads instance configurations, and manages binding instances."""

    def __init__(self, main_config: ConfigGuard, resource_manager: ResourceManager):
        """
        Initializes the BindingManager.

        Args:
            main_config: The loaded main ConfigGuard object.
            resource_manager: The shared ResourceManager instance.
        """
        self.main_config = main_config
        self.resource_manager = resource_manager
        self.encryption_key = get_encryption_key() # Get key from main config/env

        self._discovered_binding_types: Dict[str, Dict[str, Any]] = {}
        self._binding_instances: Dict[str, Binding] = {}
        self._load_errors: Dict[str, str] = {} # {identifier: error_message}

    def _discover_binding_types(self):
        """Scans configured folders for binding directories with valid binding_card.yaml."""
        # (Implementation remains the same)
        logger.info("Discovering binding types...")
        self._discovered_binding_types = {}
        self._load_errors = {}
        binding_folders = []
        server_root = get_server_root()

        example_folder = Path(self.main_config.paths.example_bindings_folder) if self.main_config.paths.example_bindings_folder else None
        personal_folder = Path(self.main_config.paths.bindings_folder) if self.main_config.paths.bindings_folder else None

        if example_folder and example_folder.is_dir():
             binding_folders.append(example_folder)
             add_path_to_sys_path(example_folder.parent)
             logger.debug(f"Added example binding parent path: {example_folder.parent}")
        else: logger.debug(f"Example bindings folder not found or not configured: {example_folder}")

        if personal_folder and personal_folder.is_dir():
             if not example_folder or personal_folder != example_folder:
                  binding_folders.append(personal_folder)
                  add_path_to_sys_path(personal_folder.parent)
                  logger.debug(f"Added personal binding parent path: {personal_folder.parent}")
             else: logger.debug("Personal binding folder is same as example folder.")
        else: logger.debug(f"Personal bindings folder not found or not configured: {personal_folder}")

        if not binding_folders:
            logger.warning("No binding type folders configured or found.")
            return

        for folder in binding_folders:
            logger.info(f"Scanning for binding types in: {folder}")
            for potential_path in folder.iterdir():
                if potential_path.is_dir() and not potential_path.name.startswith(('.', '_')):
                    init_file = potential_path / "__init__.py"
                    card_file = potential_path / "binding_card.yaml"
                    if init_file.exists() and card_file.exists():
                        try:
                            with open(card_file, 'r', encoding='utf-8') as f:
                                card_data = yaml.safe_load(f)
                            if not isinstance(card_data, dict): raise ValueError("Invalid format (not a dictionary)")
                            if "type_name" not in card_data or not isinstance(card_data["type_name"], str): raise ValueError("Missing or invalid 'type_name'")
                            if "instance_schema" not in card_data or not isinstance(card_data["instance_schema"], dict): raise ValueError("Missing or invalid 'instance_schema'")

                            type_name = card_data["type_name"]
                            card_data["package_path"] = potential_path.resolve()
                            if type_name not in self._discovered_binding_types or folder == personal_folder:
                                self._discovered_binding_types[type_name] = card_data
                                logger.info(f"Discovered binding type: '{type_name}' from {potential_path.name}")
                            else: logger.debug(f"Ignoring duplicate binding type '{type_name}' from {potential_path.name}")
                        except yaml.YAMLError as e: self._load_errors[str(potential_path)] = f"YAML Error: {e}"
                        except ValueError as e: self._load_errors[str(potential_path)] = f"Card Error: {e}"
                        except Exception as e: self._load_errors[str(potential_path)] = f"Error loading card: {e}"
                    else: logger.debug(f"Skipping directory {potential_path.name}: Missing __init__.py or binding_card.yaml")

        logger.info(f"Discovery finished. Found {len(self._discovered_binding_types)} valid binding types.")
        if self._load_errors: logger.warning(f"Binding Type Discovery Errors: {self._load_errors}")

    async def load_bindings(self):
        """Loads and instantiates binding instances based on the main config's bindings_map."""
        self._discover_binding_types()
        self._binding_instances = {}
        instance_configs_dir = get_binding_instances_config_path()

        logger.info(f"Loading binding instances based on 'bindings_map'...")
        logger.info(f"Expecting instance config files in: {instance_configs_dir}")

        bindings_map_section = getattr(self.main_config, "bindings_map", None)
        if not bindings_map_section:
             logger.warning("No 'bindings_map' section found. No instances to load.")
             return

        try: bindings_to_load = bindings_map_section.get_dict()
        except AttributeError: logger.error("Could not get bindings_map dictionary."); return
        except Exception as e: logger.error(f"Error retrieving bindings_map: {e}", exc_info=True); return

        if not bindings_to_load: logger.info("Bindings map is empty."); return

        for instance_name, type_name in bindings_to_load.items():
            if not isinstance(type_name, str) or not type_name:
                 err_msg = f"Invalid type name '{type_name}' for instance '{instance_name}'"; logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            logger.info(f"--- Loading instance: '{instance_name}' (Type: '{type_name}') ---")

            # 1. Find Type Info
            if type_name not in self._discovered_binding_types:
                 err_msg = f"Type '{type_name}' for instance '{instance_name}' not discovered."; logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            type_card_data = self._discovered_binding_types[type_name]
            instance_schema = type_card_data.get("instance_schema"); package_path = type_card_data.get("package_path")
            if not instance_schema or not isinstance(instance_schema, dict): err_msg = f"Schema missing/invalid for type '{type_name}'."; logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            if not package_path or not isinstance(package_path, Path): err_msg = f"Package path missing for type '{type_name}'."; logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue

            # 2. Load Instance Config
            instance_config_guard: Optional[ConfigGuard] = None; instance_config_dict: Optional[Dict[str, Any]] = None
            instance_config_path = None; found_handler_class = None
            for ext, handler_cls in _binding_handler_map.items():
                 potential_path = instance_configs_dir / f"{instance_name}{ext}"
                 if potential_path.is_file(): instance_config_path = potential_path; found_handler_class = handler_cls; break
            if not instance_config_path: err_msg = f"Config file for '{instance_name}' not found in {instance_configs_dir}"; logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue

            try:
                logger.info(f"Loading instance config from: {instance_config_path}")
                schema_copy = instance_schema.copy()
                schema_copy.setdefault("type", {"type": "str", "default": type_name}); schema_copy.setdefault("binding_instance_name", {"type": "str", "default": instance_name})
                # Get schema version from card, default if missing
                instance_schema_version = schema_copy.get("__version__", "0.1.0")

                instance_config_guard = ConfigGuard(
                    schema=schema_copy, instance_version=instance_schema_version, # Pass version from card schema
                    config_path=instance_config_path, encryption_key=self.encryption_key, handler=found_handler_class()
                )
                instance_config_guard.load(update_file=True); instance_config_dict = instance_config_guard.get_config_dict()
                loaded_type = instance_config_dict.get("type")
                if loaded_type != type_name: raise ConfigGuardError(f"Type mismatch: Map expects '{type_name}', file has '{loaded_type}'.")
                instance_config_dict["binding_instance_name"] = instance_name # Ensure it's set
            except (ConfigGuardError, ImportError, Exception) as e:
                 err_msg = f"Error loading config for '{instance_name}' from {instance_config_path}: {e}"; logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg; continue

            # 3. Import Binding Package/Module
            BindingClass: Optional[Type[Binding]] = None
            try:
                parent_dir_name = package_path.parent.name; package_module_name = package_path.name
                package_import_name = f"{parent_dir_name}.{package_module_name}"
                logger.info(f"Importing binding package: {package_import_name}")
                module = importlib.import_module(package_import_name)
                if not module: raise ImportError("Module loaded as None")
                found_classes = find_classes_in_module(module, Binding)
                logger.debug(f"Found potential binding classes in {package_import_name}: {[c.__name__ for c in found_classes]}")
                for cls in found_classes:
                    cls_type_name = getattr(cls, 'binding_type_name', None)
                    if cls_type_name == type_name: BindingClass = cls; logger.info(f"Found matching Binding class: {BindingClass.__name__}"); break
                if not BindingClass: raise TypeError(f"No class with binding_type_name='{type_name}' found.")
            except (ImportError, TypeError, Exception) as e:
                 err_msg = f"Error importing/finding class for type '{type_name}': {e}"; logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg; continue

            # 4. Instantiate Binding
            try:
                logger.info(f"Instantiating binding '{instance_name}' (type: {type_name})")
                instance = BindingClass(config=instance_config_dict, resource_manager=self.resource_manager)
                healthy, message = await instance.health_check()
                if healthy:
                    logger.info(f"Successfully instantiated '{instance_name}'. Health OK: {message}")
                    self._binding_instances[instance_name] = instance
                else:
                    err_msg = f"Health check failed for instance '{instance_name}': {message}"; logger.error(err_msg); self._load_errors[instance_name] = err_msg
            except Exception as e:
                 err_msg = f"Failed instantiate binding '{instance_name}': {e}"; logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg

        logger.info(f"Finished loading instances. Instantiated: {len(self._binding_instances)}. Errors: {len(self._load_errors)}")
        if self._load_errors: logger.warning(f"Instance Loading Errors: {self._load_errors}")


    def list_binding_types(self) -> Dict[str, Dict[str, Any]]:
        """Returns metadata for discovered binding types from their cards."""
        # (Implementation remains the same)
        types_info = {}
        for type_name, card_data in self._discovered_binding_types.items():
            info = {k: v for k, v in card_data.items() if k not in ["instance_schema", "package_path"]}
            types_info[type_name] = info
        return types_info

    def list_binding_instances(self) -> Dict[str, Dict[str, Any]]:
        """Returns the configuration dictionary for successfully instantiated bindings."""
        # (Implementation remains the same)
        instances_info = {}
        for name, instance in self._binding_instances.items():
            try:
                config_copy = instance.get_instance_config()
                instances_info[name] = config_copy
            except Exception as e:
                logger.warning(f"Could not get config dict for instance '{name}': {e}")
                instances_info[name] = {"error": "Failed to retrieve config dict", "type": instance.binding_type_name}
        return instances_info

    def get_binding(self, instance_name: str) -> Optional[Binding]:
        """Gets a specific binding instance by its configured name."""
        # (Implementation remains the same)
        instance = self._binding_instances.get(instance_name)
        if not instance:
             logger.error(f"Binding instance '{instance_name}' not found or failed load. Check config/logs.")
             if instance_name in self._load_errors: logger.error(f" -> Instance Load Error: {self._load_errors[instance_name]}")
             elif instance_name in (getattr(self.main_config, "bindings_map", {}) or {}): logger.error(" -> Instance was defined in bindings_map but failed to load.")
             else: logger.error(" -> Instance was not found in bindings_map.")
        return instance

    async def cleanup(self):
        """Cleans up resources used by binding instances (e.g., unloads models)."""
        # (Implementation remains the same)
        logger.info("Cleaning up binding instances...")
        unload_tasks = []
        for name, instance in self._binding_instances.items():
            if hasattr(instance, 'unload_model') and callable(instance.unload_model):
                logger.debug(f"Scheduling unload for instance '{name}'...")
                unload_tasks.append(instance.unload_model())
            else: logger.debug(f"Instance '{name}' does not have an unload_model method.")
        results = await asyncio.gather(*unload_tasks, return_exceptions=True)
        instance_names = list(self._binding_instances.keys())
        for i, result in enumerate(results):
            instance_name = instance_names[i]
            if isinstance(result, Exception): logger.error(f"Error unloading model for '{instance_name}': {result}", exc_info=result)
            else: logger.info(f"Unload successful for '{instance_name}'.")
        logger.info("Binding cleanup finished.")