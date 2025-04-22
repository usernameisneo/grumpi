# lollms_server/core/bindings.py
import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import List, Dict, Any, Type, Optional, AsyncGenerator, Tuple, Union
import asyncio
import json

# Core components
from .config import AppConfig
from lollms_server.utils.file_utils import safe_load_module, find_classes_in_module, add_path_to_sys_path
from .resource_manager import ResourceManager
from lollms_server.utils.helpers import extract_code_blocks

# --- IMPORT: InputData from api.models ---
try:
    # Use TYPE_CHECKING to avoid circular import errors at runtime if InputData needs Binding later
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from lollms_server.api.models import InputData
    else:
        # If not type checking, import it but handle potential import error
        try:
            from lollms_server.api.models import InputData
        except ImportError:
            class InputData: pass # type: ignore # Define placeholder if import fails
except ImportError:
    # Fallback if TYPE_CHECKING itself fails? Unlikely but safe.
    class InputData: pass # type: ignore


logger = logging.getLogger(__name__)

# --- Binding Base Class ---

class Binding(ABC):
    """Abstract Base Class for all generation bindings."""

    # --- binding_type_name: Added for clarity, subclasses should override ---
    binding_type_name: str = "base_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        self.config = config
        self.resource_manager = resource_manager
        self.binding_name = config.get("binding_name", "unknown_binding")
        self.model_name: Optional[str] = None
        self._model_loaded = False
        self._load_lock = asyncio.Lock()

    @abstractmethod
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lists models available to this specific binding instance.
        Contract: Must return List[Dict], each Dict MUST have 'name'.
        SHOULD populate standardized keys (size, modified_at, format, family,
        context_size, max_output_tokens, supports_vision, supports_audio).
        Other info goes into 'details'.
        """
        pass

    @classmethod
    @abstractmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the binding (type_name, requirements, etc.)."""
        pass

    # --- NEW: Abstract Methods for Capability Reporting ---
    @abstractmethod
    def get_supported_input_modalities(self) -> List[str]:
        """Returns a list of supported input data types (e.g., ['text', 'image'])."""
        # Example implementation in subclass: return ['text', 'image']
        pass

    @abstractmethod
    def get_supported_output_modalities(self) -> List[str]:
        """Returns a list of supported output data types (e.g., ['text', 'image', 'audio'])."""
        # Example implementation in subclass: return ['text']
        pass

    # Optional, more granular check - can be implemented by subclasses if needed
    def supports_input_role(self, data_type: str, role: str) -> bool:
        """
        Checks if a specific type/role combination is supported.
        Default implementation relies on the broader modality check.
        Subclasses (like a diffusion binding) should override for roles like 'controlnet_image'.
        """
        return data_type in self.get_supported_input_modalities()
    # --- End New Abstract Methods ---


    def get_instance_config(self) -> Dict[str, Any]:
        return self.config

    async def health_check(self) -> Tuple[bool, str]:
        return True, "Binding initialized. No specific health check implemented."

    @abstractmethod
    async def load_model(self, model_name: str) -> bool:
        """Loads the specified model. Uses self.resource_manager if needed."""
        pass

    @abstractmethod
    async def unload_model(self) -> bool:
        """Unloads the currently loaded model."""
        pass

    # --- MODIFIED: generate signature ---
    @abstractmethod
    async def generate(
        self,
        prompt: str, # The primary text prompt extracted from input_data
        params: Dict[str, Any],
        request_info: Dict[str, Any], # Contains original request type, etc.
        # NEW: Pass relevant non-text data
        multimodal_data: Optional[List['InputData']] = None # Use forward reference for type hint
    ) -> Union[str, Dict[str, Any]]:
        """
        Generates output based on the prompt and optional multimodal data.
        Assumes the model is loaded.

        Args:
            prompt: The primary text input.
            params: Generation parameters.
            request_info: Dictionary containing original request context.
            multimodal_data: List of validated non-text InputData objects relevant
                             to this binding, prepared by process_generation_request.

        Returns:
            The generated output. String for TTT, dictionary for others.
            Bindings should structure dicts consistently (e.g., {"text": "..."},
            {"image_base64": "...", "mime_type": "..."}, etc.).
        """
        pass


    # --- MODIFIED: generate_stream signature ---
    async def generate_stream(
        self,
        prompt: str, # The primary text prompt
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        # NEW: Pass relevant non-text data
        multimodal_data: Optional[List['InputData']] = None # Use forward reference for type hint
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates output as an asynchronous stream of chunks (standardized format).
        Handles TTT streaming by default using the non-streaming generate method.
        Bindings supporting native streaming (text, audio) should override this.

        Args:
            prompt: The primary text input.
            params: Generation parameters.
            request_info: Dictionary containing original request context.
            multimodal_data: List of validated non-text InputData objects.

        Yields:
            Dictionaries matching the StreamChunk format (type, content, metadata).
        """
        logger.warning(f"Binding {self.binding_name} does not support native streaming or multimodal streaming simulation. Simulating using non-stream 'generate'.")

        # --- UPDATED: Call generate with new signature ---
        full_response = await self.generate(
            prompt=prompt,
            params=params,
            request_info=request_info,
            multimodal_data=multimodal_data # Pass multimodal data along
        )
        # --- End Update ---

        # Format the full response into appropriate stream chunks
        if isinstance(full_response, str):
                # Simulate TTT stream
                yield {"type": "chunk", "content": full_response, "metadata": {}}
                yield {"type": "final", "content": full_response, "metadata": {}}
        elif isinstance(full_response, dict):
             # For non-TTT results, send the whole dict as the final chunk content
             # Use the actual dictionary content directly
             yield {"type": "final", "content": full_response, "metadata": {}}
        else:
             logger.error(f"generate_stream simulation failed: generate returned unexpected type {type(full_response)}")
             yield {"type": "error", "content": "Streaming simulation failed"}
        return

    # --- get_resource_requirements (unchanged) ---
    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        return {"gpu_required": True, "estimated_vram_mb": 1024}

    # --- is_model_loaded (unchanged) ---
    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded

    # --- Structured output helpers (generate_structured_output, generate_and_extract_all_codes, ask_yes_no) ---
    # TODO (Phase 6+): Update structured output helpers to handle multimodal_data if necessary for their tasks.
    async def generate_structured_output(
        self,
        prompt: str,
        structure_definition: str,
        output_language_tag: str = "json",
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None,
        # NEW: Add multimodal_data placeholder
        multimodal_data: Optional[List['InputData']] = None, # Use forward reference for type hint
        system_message_prefix: str = "Follow these instructions precisely:\n"
    ) -> Optional[Any]:
        """
        Generates text (potentially considering multimodal input) and attempts
        to extract structured output.
        *Note: Multimodal data handling within this helper needs review.*
        """
        if params is None: params = {}
        if request_info is None: request_info = {}
        # ... (Construct structured_system_message - remains the same) ...
        structured_system_message = (
             f"{system_message_prefix}"
             f"Generate a response that directly answers the user's query: '{prompt}'.\n"
             f"Your response MUST contain EXACTLY ONE markdown code block.\n"
             f"The markdown code block MUST be tagged with the language '{output_language_tag}'.\n"
             f"The content inside the markdown code block MUST conform to the following structure definition:\n"
             f"{structure_definition}\n"
             f"Do NOT include any explanation, conversation, or text outside the markdown code block."
        )

        structured_params = params.copy()
        existing_sys_msg = structured_params.get("system_message", "")
        separator = "\n\n" if existing_sys_msg else ""
        structured_params["system_message"] = f"{existing_sys_msg}{separator}{structured_system_message}"

        logger.debug(f"Generating structured output (expecting first block). Tag: '{output_language_tag}'")

        try:
            # --- CALL with new signature ---
            llm_response = await self.generate(
                prompt=prompt,
                params=structured_params,
                request_info=request_info,
                multimodal_data=multimodal_data # Pass it along
            )
            # --- END CALL ---

            if not isinstance(llm_response, str):
                logger.warning(f"generate_structured_output expected string response, got {type(llm_response)}.")
                return None

            all_blocks = extract_code_blocks(llm_response)
            first_matching_block = None
            for block in all_blocks:
                lang_match = (
                    not output_language_tag or
                    (block.get('type') and block['type'].lower() == output_language_tag.lower())
                )
                if lang_match:
                    first_matching_block = block
                    break

            if first_matching_block is None:
                logger.warning(f"Could not extract any markdown block with tag '{output_language_tag}'. LLM Response:\n{llm_response}")
                return None

            if not first_matching_block.get('is_complete', True):
                 logger.warning(f"Extracted block for tag '{output_language_tag}' seems incomplete.")

            extracted_content = first_matching_block.get('content', '')

            if output_language_tag.lower() == "json":
                try:
                    parsed_json = json.loads(extracted_content)
                    logger.debug("Successfully extracted and parsed JSON from first matching block.")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted content as JSON. Content:\n{extracted_content}\nError: {e}")
                    return None
            else:
                logger.debug(f"Successfully extracted text content for tag '{output_language_tag}' from first matching block.")
                return extracted_content

        except Exception as e:
            logger.error(f"Error during generate_structured_output: {e}", exc_info=True)
            return None

    async def generate_and_extract_all_codes(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None,
        # NEW: Add multimodal_data placeholder
        multimodal_data: Optional[List['InputData']] = None # Use forward reference for type hint
    ) -> List[Dict[str, Any]]:
        """
        Generates text (potentially considering multimodal input) and extracts ALL
        markdown code blocks.
        *Note: Multimodal data handling within this helper needs review.*
        """
        if params is None: params = {}
        if request_info is None: request_info = {}
        logger.debug("Generating response to extract all code blocks...")
        try:
            # --- CALL with new signature ---
            llm_response = await self.generate(
                prompt=prompt,
                params=params,
                request_info=request_info,
                multimodal_data=multimodal_data # Pass it along
            )
            # --- END CALL ---

            if not isinstance(llm_response, str):
                 logger.warning(f"generate_and_extract_all_codes expected string response, got {type(llm_response)}.")
                 return []

            return extract_code_blocks(llm_response)

        except Exception as e:
            logger.error(f"Error during generate_and_extract_all_codes: {e}", exc_info=True)
            return []

    async def ask_yes_no(
        self,
        question: str,
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None,
        # NEW: Add multimodal_data placeholder
        multimodal_data: Optional[List['InputData']] = None # Use forward reference for type hint
    ) -> Optional[bool]:
        """
        Asks the LLM a yes/no question (potentially considering multimodal input).
        *Note: Multimodal data handling within this helper needs review.*
        """
        if params is None: params = {}
        if request_info is None: request_info = {}

        # Construct the system message for the yes/no task
        # ... (system message construction remains the same) ...
        base_system_message = params.get("system_message", "")
        separator = "\n\n" if base_system_message else ""
        yes_no_system_message = (
            f"{base_system_message}{separator}"
            f"Critically evaluate the user's question based on any prior context.\n"
            f"Your final response MUST be ONLY the single word 'yes' or the single word 'no'.\n"
            f"You MUST enclose this single word answer within a markdown code block like this: ```text\nyes``` or ```text\nno```.\n"
            f"Do NOT provide any explanation or any other text outside the markdown code block."
        )

        yes_no_params = params.copy()
        yes_no_params["system_message"] = yes_no_system_message
        yes_no_params.setdefault("temperature", 0.1)
        yes_no_params.setdefault("max_tokens", 10)

        logger.debug(f"Asking yes/no question: '{question}'")

        try:
            # --- CALL with new signature ---
            llm_response = await self.generate(
                prompt=question,
                params=yes_no_params,
                request_info=request_info,
                multimodal_data=multimodal_data # Pass it along
            )
            # --- END CALL ---

            if not isinstance(llm_response, str):
                logger.warning(f"ask_yes_no expected string response, got {type(llm_response)}.")
                return None

            code_blocks = extract_code_blocks(llm_response)
            answer_block = None
            for block in code_blocks:
                 # Look for 'text' or unknown type specifically
                 block_type = block.get('type', 'unknown').lower()
                 if block_type in ['text', 'unknown']:
                      answer_block = block
                      break

            if answer_block is None:
                 logger.warning(f"Could not find ```text block in yes/no response. Response:\n{llm_response}")
                 return None

            if not answer_block.get('is_complete', True):
                logger.warning("Extracted yes/no answer block might be incomplete.")

            answer = answer_block.get('content', '').strip().lower()
            if answer == "yes":
                logger.debug("Parsed yes/no answer: YES")
                return True
            elif answer == "no":
                logger.debug("Parsed yes/no answer: NO")
                return False
            else:
                logger.warning(f"LLM answer in text block was not 'yes' or 'no': '{answer}'. Raw block content: '{answer_block.get('content')}'")
                return None

        except Exception as e:
            logger.error(f"Error during ask_yes_no for question '{question}': {e}", exc_info=True)
            return None
    # --- END Structured Output Helpers ---


# --- Binding Manager (no changes needed in Phase 1) ---
class BindingManager:
    """Discovers, loads, and manages available binding types and instances."""

    def __init__(self, config: AppConfig, resource_manager: ResourceManager):
        self.config = config
        self.resource_manager = resource_manager
        self._binding_classes: Dict[str, Type[Binding]] = {}
        self._binding_instances: Dict[str, Binding] = {}
        self._discovery_errors: Dict[str, str] = {}
        self._instantiation_errors: Dict[str, str] = {}

    async def load_bindings(self):
        """Discovers binding classes and instantiates configured bindings."""
        logger.info("Discovering binding classes...")
        self._binding_classes = {} # Reset
        self._discovery_errors = {} # Reset

        binding_folders = []
        if self.config.paths.example_bindings_folder and self.config.paths.example_bindings_folder.exists():
                binding_folders.append(self.config.paths.example_bindings_folder)
                add_path_to_sys_path(self.config.paths.example_bindings_folder.parent)
        if self.config.paths.bindings_folder and self.config.paths.bindings_folder.exists():
                binding_folders.append(self.config.paths.bindings_folder)
                add_path_to_sys_path(self.config.paths.bindings_folder.parent)

        if not binding_folders:
            logger.warning("No binding folders configured or found.")
            return

        for folder in binding_folders:
            logger.info(f"Scanning for bindings in: {folder}")
            for file_path in folder.glob("*.py"):
                if file_path.name == "__init__.py":
                    continue
                module, error = safe_load_module(file_path, package_path=folder)
                if module:
                    found_classes = find_classes_in_module(module, Binding)
                    for binding_class in found_classes:
                        try:
                            # --- Use binding_type_name attribute if defined ---
                            type_name = getattr(binding_class, 'binding_type_name', None)
                            if not type_name or type_name == "base_binding":
                                 class_config = binding_class.get_binding_config()
                                 type_name = class_config.get("type_name") # Fallback to config
                                 if not type_name:
                                     type_name = file_path.stem # Fallback to filename
                                     logger.warning(f"Binding class {binding_class.__name__} in {file_path} does not define 'binding_type_name' attribute or 'type_name' in get_binding_config(). Falling back to filename '{type_name}'.")

                            if type_name in self._binding_classes:
                                logger.warning(f"Duplicate binding type name '{type_name}' found. Overwriting with class from {file_path}.")
                            self._binding_classes[type_name] = binding_class
                            logger.info(f"Discovered binding type: '{type_name}' from class {binding_class.__name__}")
                        except Exception as e:
                            err_msg = f"Error processing class {binding_class.__name__} in {file_path}: {e}"
                            logger.error(err_msg)
                            self._discovery_errors[str(file_path)] = err_msg
                elif error:
                    self._discovery_errors[str(file_path)] = error

        logger.info(f"Found {len(self._binding_classes)} binding types.")
        if self._discovery_errors:
            logger.warning(f"Encountered errors during binding discovery: {self._discovery_errors}")

        await self._instantiate_bindings()

    async def _instantiate_bindings(self):
        logger.info("Instantiating configured bindings...")
        self._binding_instances = {} # Reset
        self._instantiation_errors = {} # Reset

        for logical_name, instance_config in self.config.bindings.items():
            binding_type = instance_config.get("type")
            if not binding_type:
                err_msg = f"Binding '{logical_name}' in config.toml is missing the required 'type' field."
                logger.error(err_msg)
                self._instantiation_errors[logical_name] = err_msg
                continue

            BindingClass = self._binding_classes.get(binding_type)
            if not BindingClass:
                err_msg = f"Binding type '{binding_type}' specified for '{logical_name}' not found or failed to load."
                logger.error(err_msg)
                self._instantiation_errors[logical_name] = err_msg
                continue

            try:
                full_config = instance_config.copy()
                full_config["binding_name"] = logical_name
                instance = BindingClass(config=full_config, resource_manager=self.resource_manager)

                healthy, message = await instance.health_check()
                if healthy:
                    logger.info(f"Successfully instantiated binding '{logical_name}' (type: {binding_type}). Health check OK: {message}")
                    self._binding_instances[logical_name] = instance
                else:
                    err_msg = f"Health check failed for binding '{logical_name}' (type: {binding_type}): {message}"
                    logger.error(err_msg)
                    self._instantiation_errors[logical_name] = err_msg

            except Exception as e:
                err_msg = f"Failed to instantiate binding '{logical_name}' (type: {binding_type}): {e}"
                logger.error(err_msg, exc_info=True)
                self._instantiation_errors[logical_name] = err_msg

        logger.info(f"Instantiated {len(self._binding_instances)} bindings.")
        if self._instantiation_errors:
            logger.warning(f"Encountered errors during binding instantiation: {self._instantiation_errors}")

    # --- list_binding_types (no changes) ---
    def list_binding_types(self) -> Dict[str, Dict[str, Any]]:
        types_info = {}
        for name, cls in self._binding_classes.items():
            try:
                types_info[name] = cls.get_binding_config()
            except Exception as e:
                logger.warning(f"Could not get config for binding type '{name}': {e}")
                types_info[name] = {"error": "Failed to retrieve config"}
        return types_info

    # --- list_binding_instances (no changes) ---
    def list_binding_instances(self) -> Dict[str, Dict[str, Any]]:
            instances_info = {}
            for name, instance in self._binding_instances.items():
                try:
                    instances_info[name] = instance.get_instance_config()
                except Exception as e:
                    logger.warning(f"Could not get config for binding instance '{name}': {e}")
                    instances_info[name] = {"error": "Failed to retrieve config"}
            return instances_info

    # --- get_binding (no changes) ---
    def get_binding(self, logical_name: str) -> Optional[Binding]:
        instance = self._binding_instances.get(logical_name)
        if not instance:
                logger.error(f"Binding instance '{logical_name}' not found or failed to instantiate.")
                return None
        return instance

    # --- cleanup (no changes) ---
    async def cleanup(self):
        logger.info("Cleaning up binding instances...")
        for name, instance in self._binding_instances.items():
            try:
                logger.info(f"Unloading model for binding instance '{name}'...")
                await instance.unload_model()
            except Exception as e:
                logger.error(f"Error unloading model for binding '{name}': {e}", exc_info=True)
        logger.info("Binding cleanup finished.")