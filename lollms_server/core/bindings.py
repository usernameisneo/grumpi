# lollms_server/core/bindings.py
import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import List, Dict, Any, Type, Optional, AsyncGenerator, Tuple, Union
import asyncio
import json

from .config import AppConfig
from lollms_server.utils.file_utils import safe_load_module, find_classes_in_module, add_path_to_sys_path
from .resource_manager import ResourceManager # Import after definition
from lollms_server.utils.helpers import extract_code_blocks

logger = logging.getLogger(__name__)

# --- Binding Base Class ---

class Binding(ABC):
    """Abstract Base Class for all generation bindings."""

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the binding instance.

        Args:
            config: The specific configuration dictionary for this binding instance
                    (from the main config.toml under [bindings.<binding_name>]).
            resource_manager: The global resource manager instance.
        """
        self.config = config
        self.resource_manager = resource_manager
        self.binding_name = config.get("binding_name", "unknown_binding") # Injected by manager
        self.model_name: Optional[str] = None
        self._model_loaded = False
        self._load_lock = asyncio.Lock() # Prevents concurrent load/unload attempts on the same instance

    @abstractmethod
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lists models available to this specific binding instance.

        Implementation Contract:
        - Must return a list of dictionaries.
        - Each dictionary MUST contain a 'name' key (string identifier).
        - Implementations SHOULD attempt to populate other standardized keys if
          the information is readily available from the source API/listing method:
            - 'size' (int): Bytes
            - 'modified_at' (datetime)
            - 'quantization_level' (str)
            - 'format' (str)
            - 'family' (str)
            - 'families' (List[str])
            - 'parameter_size' (str)
            - 'context_size' (int): Total context window
            - 'max_output_tokens' (int): Max generation length
            - 'template' (str)
            - 'license' (str)
            - 'homepage' (str/HttpUrl)
        - Any other relevant information specific to the binding or model
          should be included as additional keys in the dictionary. These
          will be captured in the 'details' field of the API response.

        Returns:
            A list of dictionaries adhering to the contract.
        """
        pass
    @classmethod
    @abstractmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the binding."""
        # Example: return {"name": "openai_binding", "version": "1.0", "type": "api", "requirements": ["openai>=1.0.0"]}
        pass

    def get_instance_config(self) -> Dict[str, Any]:
        """Returns the specific configuration of this binding instance."""
        return self.config

    async def health_check(self) -> Tuple[bool, str]:
        """Performs a basic health check (e.g., connection to API). Returns (status, message)."""
        # Default implementation assumes health if it loads
        return True, "Binding initialized. No specific health check implemented."

    @abstractmethod
    async def load_model(self, model_name: str) -> bool:
        """
        Loads the specified model. Implementations SHOULD use the
        `self.resource_manager.acquire_gpu_resource()` context manager
        if the loading process requires significant shared resources like a GPU.
        This method must be idempotent if the model is already loaded.
        It should set self.model_name and self._model_loaded upon success.

        Args:
            model_name: The identifier of the model to load.

        Returns:
            True if the model was loaded successfully (or was already loaded), False otherwise.
        """
        # Example structure (implementations should fill this):
        # async with self._load_lock: # Instance specific lock
        #     if self._model_loaded and self.model_name == model_name:
        #         return True
        #
        #     # --- Resource Acquisition ---
        #     requirements = self.get_resource_requirements(model_name)
        #     needs_gpu = requirements.get("gpu_required", False) # Check if GPU needed
        #     acquired_context = nullcontext() # Default: no resource needed
        #     if needs_gpu and self.resource_manager:
        #          logger.info(f"Binding '{self.binding_name}': Acquiring GPU resource for loading model '{model_name}'...")
        #          acquired_context = self.resource_manager.acquire_gpu_resource(
        #              task_name=f"load_{self.binding_name}_{model_name}"
        #          )
        #
        #     try:
        #         async with acquired_context:
        #             # --- Actual Loading Logic Here ---
        #             # if needs_gpu: logger.info(f"Binding '{self.binding_name}': GPU resource acquired.")
        #             # ... perform model loading using self.config, model_name ...
        #             # If successful:
        #             # self.model_name = model_name
        #             # self._model_loaded = True
        #             # logger.info(f"Binding '{self.binding_name}': Model '{model_name}' loaded.")
        #             # return True
        #             # If failed:
        #             # logger.error(f"Binding '{self.binding_name}': Failed to load model '{model_name}'.")
        #             # return False
        #             pass # Placeholder
        #
        #     except asyncio.TimeoutError:
        #         logger.error(f"Binding '{self.binding_name}': Timeout waiting for GPU resource to load model '{model_name}'.")
        #         return False
        #     except Exception as e:
        #         logger.error(f"Binding '{self.binding_name}': Error during model loading '{model_name}': {e}", exc_info=True)
        #         return False
        #
        # # Fallback if logic not implemented
        # raise NotImplementedError("load_model must be implemented by binding subclasses.")
        pass # Keep abstract


    @abstractmethod
    async def unload_model(self) -> bool:
        """
        Unloads the currently loaded model and releases associated resources.
        Implementations should release any resources acquired via the resource manager
        if they were held persistently (though the context manager usually handles this).
        Must be idempotent. Should reset self.model_name and self._model_loaded.

        Returns:
            True if the model was unloaded successfully (or wasn't loaded), False otherwise.
        """
        pass # Keep abstract

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> Union[str, Dict[str, Any]]:
        """
        Generates output based on the prompt and parameters.
        Assumes the model is already loaded. If the generation process itself
        requires exclusive access to a shared resource (like GPU), implementations
        might need to acquire it using `self.resource_manager.acquire_gpu_resource()`.
        However, it's often preferable if bindings can handle concurrent generation
        on a loaded model without exclusive locking for the entire generation duration.

        Args:
            prompt: The input prompt.
            params: Generation parameters (e.g., max_tokens, temperature).
            request_info: Dictionary containing additional request context.

        Returns:
            The generated output. String for TTT, dictionary for others.
        """
        pass # Keep abstract
    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates text output as an asynchronous stream of chunks.
        Similar to `generate`, might need resource acquisition if the underlying
        streaming generation process requires exclusive resource access.

        Args:
            prompt: The input prompt.
            params: Generation parameters.
            request_info: Dictionary containing additional request context.


        Yields:
            Dictionaries representing chunks (e.g., StreamChunk format).
        """
        # Default implementation using non-streaming generate (remains the same)
        logger.warning(f"Binding {self.binding_name} does not support native streaming. Simulating.")
        full_response = await self.generate(prompt, params, request_info)
        # ... (rest of default implementation) ...
        if isinstance(full_response, str):
                yield {"type": "chunk", "content": full_response}
                yield {"type": "final", "content": full_response, "metadata": {}}
        else:
                logger.error("generate_stream called on non-TTT binding or generate returned non-string.")
                yield {"type": "error", "content": "Streaming not supported for this output type"}
        return

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimates the resource requirements (e.g., VRAM, CPU) for this binding,
        potentially specific to a model. This is indicative.

        Args:
            model_name: Optional model name to get specific requirements.

        Returns:
            A dictionary describing resource needs, e.g., {"gpu_required": True, "estimated_vram_mb": 4096}.
        """
        # Default: Assume it might need GPU if not overridden
        return {"gpu_required": True, "estimated_vram_mb": 1024} # Default placeholder

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
        system_message_prefix: str = "Follow these instructions precisely:\n"
    ) -> Optional[Any]:
        """
        Generates text and attempts to extract structured output from the FIRST
        markdown code block matching the specified language tag.
        (Uses the new multi-block extractor but processes only the first hit for compatibility).
        """
        # ... (Construct structured_system_message - remains the same) ...
        structured_system_message = (
             # ... (system message construction) ...
        )
        structured_params = params.copy() if params else {}
        structured_params["system_message"] = structured_system_message
        if request_info is None: request_info = {}


        logger.debug(f"Generating structured output (expecting first block). Tag: '{output_language_tag}'")

        try:
            llm_response = await self.generate(
                prompt=prompt,
                params=structured_params,
                request_info=request_info
            )

            if not isinstance(llm_response, str):
                logger.warning(f"generate_structured_output expected string response, got {type(llm_response)}.")
                return None

            # --- Use the new extractor ---
            all_blocks = extract_code_blocks(llm_response)
            # --- Find the first block matching the language tag (if specified) ---
            first_matching_block = None
            for block in all_blocks:
                # Match language tag case-insensitively if provided
                lang_match = (
                    not output_language_tag or # Match any if tag is empty
                    (block['language'] and block['language'].lower() == output_language_tag.lower())
                )
                if lang_match:
                    first_matching_block = block
                    break # Found the first one

            if first_matching_block is None:
                logger.warning(f"Could not extract any markdown block with tag '{output_language_tag}'. LLM Response:\n{llm_response}")
                return None

            # Check completeness (optional, based on flag from extractor)
            if not first_matching_block['complete']:
                 logger.warning(f"Extracted block for tag '{output_language_tag}' seems incomplete.")
                 # Decide whether to return incomplete data or None. Let's return it for now.

            extracted_content = first_matching_block['content']

            # Try parsing if it's JSON
            if output_language_tag.lower() == "json":
                try:
                    parsed_json = json.loads(extracted_content)
                    logger.debug("Successfully extracted and parsed JSON from first matching block.")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted content as JSON. Content:\n{extracted_content}\nError: {e}")
                    return None # Failed parsing
            else:
                logger.debug(f"Successfully extracted text content for tag '{output_language_tag}' from first matching block.")
                return extracted_content # Return as string

        except Exception as e:
            logger.error(f"Error during generate_structured_output: {e}", exc_info=True)
            return None
    async def generate_and_extract_all_codes(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates text and extracts ALL markdown code blocks found in the response.

        Args:
            prompt: The user prompt.
            params: Optional generation parameters.
            request_info: Optional request context.

        Returns:
            A list of dictionaries, each representing a code block found,
            using the format from `extract_code_blocks`. Returns
            an empty list if generation fails or no blocks are found.
        """
        if params is None: params = {}
        if request_info is None: request_info = {}
        logger.debug("Generating response to extract all code blocks...")
        try:
            # Use the binding's own generate method
            llm_response = await self.generate(
                prompt=prompt,
                params=params, # Use original params, maybe add instruction to use markdown?
                request_info=request_info
            )
            if not isinstance(llm_response, str):
                 logger.warning(f"generate_and_extract_all_codes expected string response, got {type(llm_response)}.")
                 return []

            return extract_code_blocks(llm_response)

        except Exception as e:
            logger.error(f"Error during generate_and_extract_all_codes: {e}", exc_info=True)
            return []
    # --- NEW ask_yes_no HELPER ---
    async def ask_yes_no(
        self,
        question: str,
        params: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None
    ) -> Optional[bool]:
        """
        Asks the LLM a yes/no question using a structured prompt and extracts the answer.

        Args:
            question: The question to ask the LLM.
            params: Optional generation parameters (will be modified).
            request_info: Optional request context.

        Returns:
            True if the answer is 'yes', False if 'no', None otherwise.
        """
        if params is None: params = {}
        if request_info is None: request_info = {}

        # Construct the system message for the yes/no task
        base_system_message = params.get("system_message", "")
        separator = "\n\n" if base_system_message else ""
        yes_no_system_message = (
            f"{base_system_message}{separator}"
            f"Critically evaluate the user's question based on any prior context.\n"
            f"Your final response MUST be ONLY the single word 'yes' or the single word 'no'.\n"
            f"You MUST enclose this single word answer within a markdown code block like this: ```text\nyes``` or ```text\nno```.\n"
            f"Do NOT provide any explanation or any other text outside the markdown code block."
        )

        # Prepare parameters for this specific call
        yes_no_params = params.copy()
        yes_no_params["system_message"] = yes_no_system_message
        # Optional: Use lower temperature for factual answers
        yes_no_params.setdefault("temperature", 0.1)
        # Optional: Limit max tokens to prevent long erroneous answers
        yes_no_params.setdefault("max_tokens", 10) # Should be enough for 'yes'/'no' + block

        logger.debug(f"Asking yes/no question: '{question}'")

        try:
            # Generate the response using the core generate method
            llm_response = await self.generate(
                prompt=question,
                params=yes_no_params,
                request_info=request_info
            )

            if not isinstance(llm_response, str):
                logger.warning(f"ask_yes_no expected string response, got {type(llm_response)}.")
                return None

            # Extract the text block using the helper
            code_blocks = extract_code_blocks(llm_response)

            # Find the first block with 'text' type or unknown type
            answer_block = None
            for block in code_blocks:
                 # Allow 'text' or no language tag (unknown)
                 if block.get('type', 'unknown').lower() in ['text', 'unknown']:
                      answer_block = block
                      break

            if answer_block is None:
                 logger.warning(f"Could not find ```text block in yes/no response. Response:\n{llm_response}")
                 return None

            # Check completeness? Optional, 'yes'/'no' is short.
            if not answer_block.get('is_complete', False):
                logger.warning("Extracted yes/no answer block might be incomplete.")

            # Parse the answer
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
    # --- END ask_yes_no HELPER ---
# --- Binding Manager ---

class BindingManager:
    """Discovers, loads, and manages available binding types and instances."""

    def __init__(self, config: AppConfig, resource_manager: ResourceManager):
        self.config = config
        self.resource_manager = resource_manager
        self._binding_classes: Dict[str, Type[Binding]] = {} # Stores discovered binding classes {name: class}
        self._binding_instances: Dict[str, Binding] = {} # Stores instantiated bindings {logical_name: instance}
        self._discovery_errors: Dict[str, str] = {} # Stores errors during discovery {path: error}
        self._instantiation_errors: Dict[str, str] = {} # Stores errors during instantiation {logical_name: error}


    async def load_bindings(self):
        """Discovers binding classes and instantiates configured bindings."""
        logger.info("Discovering binding classes...")
        self._binding_classes = {} # Reset
        self._discovery_errors = {} # Reset

        binding_folders = []
        if self.config.paths.example_bindings_folder and self.config.paths.example_bindings_folder.exists():
                binding_folders.append(self.config.paths.example_bindings_folder)
                add_path_to_sys_path(self.config.paths.example_bindings_folder.parent) # Ensure parent is in path
        if self.config.paths.bindings_folder and self.config.paths.bindings_folder.exists():
                binding_folders.append(self.config.paths.bindings_folder)
                add_path_to_sys_path(self.config.paths.bindings_folder.parent) # Ensure parent is in path


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
                            # Use class name or a defined property as the key? Let's use module stem for now.
                            # Or better, use a name defined within the class itself.
                            class_config = binding_class.get_binding_config()
                            type_name = class_config.get("type_name") # Expecting binding class to define its 'type' name
                            if not type_name:
                                    type_name = file_path.stem # Fallback to filename
                                    logger.warning(f"Binding class {binding_class.__name__} in {file_path} does not define 'type_name' in get_binding_config(). Falling back to filename '{type_name}'.")

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

        # Now instantiate bindings defined in the config
        await self._instantiate_bindings()


    async def _instantiate_bindings(self):
        """Instantiates binding instances based on the [bindings] section of config.toml."""
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
                # Inject the logical name into the config passed to the instance
                full_config = instance_config.copy()
                full_config["binding_name"] = logical_name

                instance = BindingClass(config=full_config, resource_manager=self.resource_manager)

                # Perform optional health check on instantiation
                healthy, message = await instance.health_check()
                if healthy:
                        logger.info(f"Successfully instantiated binding '{logical_name}' (type: {binding_type}). Health check OK: {message}")
                        self._binding_instances[logical_name] = instance
                else:
                        err_msg = f"Health check failed for binding '{logical_name}' (type: {binding_type}): {message}"
                        logger.error(err_msg)
                        self._instantiation_errors[logical_name] = err_msg
                        # Decide if you want to keep unhealthy instances or discard them
                        # self._binding_instances[logical_name] = instance # Keep it but mark as unhealthy?
                        # Or just don't add it:
                        pass


            except Exception as e:
                err_msg = f"Failed to instantiate binding '{logical_name}' (type: {binding_type}): {e}"
                logger.error(err_msg, exc_info=True)
                self._instantiation_errors[logical_name] = err_msg

        logger.info(f"Instantiated {len(self._binding_instances)} bindings.")
        if self._instantiation_errors:
            logger.warning(f"Encountered errors during binding instantiation: {self._instantiation_errors}")

    def list_binding_types(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of discovered binding types and their metadata."""
        types_info = {}
        for name, cls in self._binding_classes.items():
            try:
                types_info[name] = cls.get_binding_config()
            except Exception as e:
                logger.warning(f"Could not get config for binding type '{name}': {e}")
                types_info[name] = {"error": "Failed to retrieve config"}
        return types_info

    def list_binding_instances(self) -> Dict[str, Dict[str, Any]]:
            """Returns a dictionary of instantiated binding logical names and their config."""
            instances_info = {}
            for name, instance in self._binding_instances.items():
                try:
                    # Add health status maybe?
                    # healthy, msg = await instance.health_check() # Avoid await here, maybe store health status?
                    instances_info[name] = instance.get_instance_config()
                except Exception as e:
                    logger.warning(f"Could not get config for binding instance '{name}': {e}")
                    instances_info[name] = {"error": "Failed to retrieve config"}
            return instances_info


    def get_binding(self, logical_name: str) -> Optional[Binding]:
        """
        Gets an instantiated binding by its logical name (defined in config.toml).
        """
        instance = self._binding_instances.get(logical_name)
        if not instance:
                logger.error(f"Binding instance '{logical_name}' not found or failed to instantiate.")
                return None
        # Potentially check health again here if needed
        return instance

    async def cleanup(self):
        """Perform cleanup tasks, like unloading models from all instances."""
        logger.info("Cleaning up binding instances...")
        for name, instance in self._binding_instances.items():
            try:
                logger.info(f"Unloading model for binding instance '{name}'...")
                await instance.unload_model()
            except Exception as e:
                logger.error(f"Error unloading model for binding '{name}': {e}", exc_info=True)
        logger.info("Binding cleanup finished.")