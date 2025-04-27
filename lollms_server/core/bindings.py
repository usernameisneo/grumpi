# lollms_server/core/bindings.py
import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
import ascii_colors as logging
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

    binding_type_name: str = "base_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the binding.

        Args:
            config: The configuration dictionary for this binding instance.
            resource_manager: The shared resource manager instance.
        """
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

        Returns:
            A list of dictionaries, each representing an available model.
        """
        pass

    @classmethod
    @abstractmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """
        Returns metadata about the binding class.

        Returns:
            A dictionary containing binding metadata like type_name, requirements, etc.
        """
        pass

    @abstractmethod
    def get_supported_input_modalities(self) -> List[str]:
        """
        Returns a list of supported input data types (e.g., ['text', 'image']).

        Returns:
            List of supported input modality types.
        """
        pass

    @abstractmethod
    def get_supported_output_modalities(self) -> List[str]:
        """
        Returns a list of supported output data types (e.g., ['text', 'image', 'audio']).

        Returns:
            List of supported output modality types.
        """
        pass

    def supports_input_role(self, data_type: str, role: str) -> bool:
        """
        Checks if a specific type/role combination is supported.

        Default implementation relies on the broader modality check.
        Subclasses (like a diffusion binding) should override for roles like 'controlnet_image'.

        Args:
            data_type: The type of input data (e.g., 'image').
            role: The role of the input data (e.g., 'controlnet_image').

        Returns:
            True if supported, False otherwise.
        """
        return data_type in self.get_supported_input_modalities()

    def get_instance_config(self) -> Dict[str, Any]:
        """
        Returns the specific configuration for this binding instance.

        Returns:
            The configuration dictionary.
        """
        return self.config

    async def health_check(self) -> Tuple[bool, str]:
        """
        Performs a health check on the binding (e.g., API connection).

        Returns:
            A tuple (is_healthy: bool, message: str).
        """
        return True, "Binding initialized. No specific health check implemented."

    @abstractmethod
    async def load_model(self, model_name: str) -> bool:
        """
        Loads the specified model into the binding.

        Uses self.resource_manager if needed for resource acquisition (e.g., GPU).

        Args:
            model_name: The name of the model to load.

        Returns:
            True if loading was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def unload_model(self) -> bool:
        """
        Unloads the currently loaded model, releasing resources.

        Returns:
            True if unloading was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates output based on the prompt and optional multimodal data.
        Assumes the model is loaded.

        Args:
            prompt: The primary text input.
            params: Generation parameters.
            request_info: Dictionary containing original request context (e.g., generation_type).
            multimodal_data: List of validated non-text InputData objects relevant
                             to this binding, prepared by process_generation_request.

        Returns:
            - For simple text output: A string.
            - For single binary output (legacy/simple bindings): A dict (e.g., {"image_base64": ..., "mime_type": ...}).
            - For potentially multiple outputs (preferred): A list of dicts, where each dict
              should resemble the OutputData model structure (e.g.,
              [{"type": "text", "data": "..."}, {"type": "image", "data": "b64", "mime_type": "...", "metadata": {...}}]).
              The `process_generation_request` function will standardize single string/dict returns into the list format.
        """
        pass

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
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
            - Individual chunks ('chunk', 'info', 'error') have content specific to that chunk.
            - The **final** chunk ('final') MUST contain a list of dicts resembling
              the OutputData model structure in its 'content' field, representing
              all generated outputs for the request. E.g.,
              `{"type": "final", "content": [{"type":"text", "data":"..."}, {"type":"image", "data":"b64..."}], "metadata": ...}`
        """
        logger.warning(f"Binding {self.binding_name} does not support native streaming or multimodal streaming simulation. Simulating using non-stream 'generate'.")
        full_response = await self.generate(
            prompt=prompt,
            params=params,
            request_info=request_info,
            multimodal_data=multimodal_data
        )

        # Standardize the simulated final output
        final_output_list = []
        if isinstance(full_response, str):
            final_output_list.append({"type": "text", "data": full_response})
            # Simulate TTT stream chunk
            yield {"type": "chunk", "content": full_response, "metadata": {}}
        elif isinstance(full_response, dict):
            # Attempt to map single dict to OutputData structure
            if "text" in full_response:
                final_output_list.append({"type": "text", "data": full_response["text"], "metadata": full_response.get("metadata", {})})
            elif "image_base64" in full_response:
                final_output_list.append({"type": "image", "data": full_response["image_base64"], "mime_type": full_response.get("mime_type"), "metadata": full_response.get("metadata", {})})
            elif "audio_base64" in full_response:
                 final_output_list.append({"type": "audio", "data": full_response["audio_base64"], "mime_type": full_response.get("mime_type"), "metadata": full_response.get("metadata", {})})
            # Add more mappings as needed...
            else: # Fallback for unknown dict structure
                logger.warning(f"Simulated stream: Unknown dict structure from generate: {full_response.keys()}. Wrapping as json.")
                final_output_list.append({"type": "json", "data": full_response})
        elif isinstance(full_response, list):
            # Assume it's already the correct list format
            final_output_list = full_response
        else:
             logger.error(f"generate_stream simulation failed: generate returned unexpected type {type(full_response)}")
             yield {"type": "error", "content": "Streaming simulation failed"}
             return # Exit the generator

        # Yield the standardized final chunk
        yield {"type": "final", "content": final_output_list, "metadata": {}}
        return

    # --- NEW: Tokenization / Detokenization / Info Methods ---
    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Tokenizes the given text using the currently loaded model's tokenizer.
        (Optional: Bindings should implement this if they support tokenization).
        """
        logger.warning(f"Tokenization requested but not implemented for binding type '{self.binding_type_name}'.")
        raise NotImplementedError(f"Binding type '{self.binding_type_name}' does not support tokenization.")

    # @abstractmethod # REMOVED
    async def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenizes a list of token IDs back into text using the currently loaded model.
        (Optional: Bindings should implement this if they support detokenization).
        """
        logger.warning(f"Detokenization requested but not implemented for binding type '{self.binding_type_name}'.")
        raise NotImplementedError(f"Binding type '{self.binding_type_name}' does not support detokenization.")

    @abstractmethod
    async def get_current_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the currently loaded model.

        Returns:
            A dictionary containing details like name, context size, max output tokens, etc.
            Keys should align with the ModelInfo Pydantic model where possible.
            Returns an empty dict or dict with None values if no model is loaded.
        """
        pass
    # --- END NEW METHODS ---

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimates resource requirements (e.g., GPU needed).

        Args:
            model_name: Optional model name to estimate for.

        Returns:
            A dictionary indicating resource needs (e.g., {"gpu_required": True}).
        """
        return {"gpu_required": True, "estimated_vram_mb": 1024} # Default estimate

    @property
    def is_model_loaded(self) -> bool:
        """Returns True if a model is currently loaded."""
        return self._model_loaded

    async def generate_structured_output(
        self,
        prompt: str,
        structure_definition: str,
        output_language_tag: str = "json",
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None,
        multimodal_data: Optional[List['InputData']] = None,
        system_message_prefix: str = "Follow these instructions precisely:\n"
    ) -> Optional[Any]:
        """
        Generates text (potentially considering multimodal input) and attempts
        to extract structured output (e.g., JSON).
        """
        if params is None: params = {}
        if request_info is None: request_info = {}

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
            llm_response = await self.generate(
                prompt=prompt,
                params=structured_params,
                request_info=request_info,
                multimodal_data=multimodal_data
            )

            # --- Extract text from potential dict response ---
            response_text = ""
            if isinstance(llm_response, dict) and "text" in llm_response:
                response_text = llm_response["text"]
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                 logger.warning(f"generate_structured_output received unexpected response type: {type(llm_response)}. Trying string conversion.")
                 response_text = str(llm_response)
            # --- End Extraction ---


            all_blocks = extract_code_blocks(response_text)
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
                logger.warning(f"Could not extract any markdown block with tag '{output_language_tag}'. LLM Response Text:\n{response_text}")
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
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates text (potentially considering multimodal input) and extracts ALL
        markdown code blocks.
        """
        if params is None: params = {}
        if request_info is None: request_info = {}
        logger.debug("Generating response to extract all code blocks...")
        try:
            llm_response = await self.generate(
                prompt=prompt,
                params=params,
                request_info=request_info,
                multimodal_data=multimodal_data
            )
            # --- Extract text from potential dict response ---
            response_text = ""
            if isinstance(llm_response, dict) and "text" in llm_response:
                response_text = llm_response["text"]
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                 logger.warning(f"generate_and_extract_all_codes received unexpected response type: {type(llm_response)}. Trying string conversion.")
                 response_text = str(llm_response)
            # --- End Extraction ---
            return extract_code_blocks(response_text)

        except Exception as e:
            logger.error(f"Error during generate_and_extract_all_codes: {e}", exc_info=True)
            return []

    async def ask_yes_no(
        self,
        question: str,
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None,
        multimodal_data: Optional[List['InputData']] = None
    ) -> Optional[bool]:
        """
        Asks the LLM a yes/no question (potentially considering multimodal input).
        """
        if params is None: params = {}
        if request_info is None: request_info = {}

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
            llm_response = await self.generate(
                prompt=question,
                params=yes_no_params,
                request_info=request_info,
                multimodal_data=multimodal_data
            )
            # --- Extract text from potential dict response ---
            response_text = ""
            if isinstance(llm_response, dict) and "text" in llm_response:
                response_text = llm_response["text"]
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                 logger.warning(f"ask_yes_no received unexpected response type: {type(llm_response)}. Trying string conversion.")
                 response_text = str(llm_response)
            # --- End Extraction ---

            code_blocks = extract_code_blocks(response_text)
            answer_block = None
            for block in code_blocks:
                 block_type = block.get('type', 'unknown').lower()
                 if block_type in ['text', 'unknown']:
                      answer_block = block
                      break

            if answer_block is None:
                 logger.warning(f"Could not find ```text block in yes/no response. Response:\n{response_text}")
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


# --- Binding Manager (unchanged) ---
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

    def list_binding_types(self) -> Dict[str, Dict[str, Any]]:
        """Returns metadata for all discovered binding types."""
        types_info = {}
        for name, cls in self._binding_classes.items():
            try:
                types_info[name] = cls.get_binding_config()
            except Exception as e:
                logger.warning(f"Could not get config for binding type '{name}': {e}")
                types_info[name] = {"error": "Failed to retrieve config"}
        return types_info

    def list_binding_instances(self) -> Dict[str, Dict[str, Any]]:
        """Returns the configuration for all successfully instantiated bindings."""
        instances_info = {}
        for name, instance in self._binding_instances.items():
            try:
                instances_info[name] = instance.get_instance_config()
            except Exception as e:
                logger.warning(f"Could not get config for binding instance '{name}': {e}")
                instances_info[name] = {"error": "Failed to retrieve config"}
        return instances_info

    def get_binding(self, logical_name: str) -> Optional[Binding]:
        """
        Gets a specific binding instance by its logical name from the config.

        Args:
            logical_name: The name assigned to the binding instance in config.toml.

        Returns:
            The Binding instance, or None if not found or failed to load.
        """
        instance = self._binding_instances.get(logical_name)
        if not instance:
                logger.error(f"Binding instance '{logical_name}' not found or failed to instantiate.")
                return None
        return instance

    async def cleanup(self):
        """Cleans up resources used by bindings (e.g., unloads models)."""
        logger.info("Cleaning up binding instances...")
        for name, instance in self._binding_instances.items():
            try:
                logger.info(f"Unloading model for binding instance '{name}'...")
                await instance.unload_model()
            except Exception as e:
                logger.error(f"Error unloading model for binding '{name}': {e}", exc_info=True)
        logger.info("Binding cleanup finished.")