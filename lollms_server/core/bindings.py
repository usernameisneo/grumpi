# encoding:utf-8
# Project: lollms_server
# File: lollms_server/core/bindings.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Manages AI model bindings, discovery, loading, and interaction, using ConfigGuard.

import importlib
import inspect
import yaml
import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Type, Optional, AsyncGenerator, Tuple, Union, cast

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
from lollms_server.utils.helpers import extract_code_blocks

# API Models for type hints (use TYPE_CHECKING)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lollms_server.api.models import InputData, OutputData, StreamChunk


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

        self.model_name: Optional[str] = None # Name of the model file/API identifier currently loaded
        self._model_loaded = False
        self._load_lock = asyncio.Lock() # Protects model loading/unloading state

        logger.debug(f"Base Binding initialized for instance '{self.binding_instance_name}' (Type: {self.binding_type_name})")

    @classmethod
    def get_binding_package_path(cls) -> Optional[Path]:
        """Finds the directory containing the binding's definition (__init__.py)."""
        try:
            # Find the file where the class `cls` is defined
            module_file = inspect.getfile(cls)
            # The directory containing __init__.py is the package path
            package_dir = Path(module_file).parent.resolve() # Resolve to absolute path
            if package_dir.is_dir():
                 return package_dir
            else:
                logger.error(f"Determined package directory is not valid: {package_dir}")
        except TypeError: # Built-in types etc.
            logger.error(f"Could not determine file path for class {cls.__name__}")
        except Exception as e:
             logger.error(f"Error finding package path for {cls.__name__}: {e}")
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
             # Ensure mandatory keys are present
             if "type_name" not in card_data or "instance_schema" not in card_data:
                 logger.error(f"Binding card {card_path} is missing 'type_name' or 'instance_schema'.")
                 return {}
             if not isinstance(card_data["instance_schema"], dict):
                  logger.error(f"Binding card {card_path} has invalid 'instance_schema' (not a dictionary).")
                  return {}
             # Add the package path to the returned data for convenience (resolved absolute path)
             card_data["package_path"] = package_dir
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
    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded model."""
        pass

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
        logger.warning(f"Binding '{self.binding_instance_name}' (Type: {self.binding_type_name}) does not implement native streaming. Simulating.")
        full_response: Union[str, Dict[str, Any], List[Dict[str, Any]]]
        try:
            full_response = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
        except Exception as e:
             logger.error(f"Error during non-stream call for stream simulation in '{self.binding_instance_name}': {e}", exc_info=True)
             yield {"type": "error", "content": f"Generation failed during simulation: {e}"}
             return

        final_output_list: List[Dict[str, Any]] = []
        # --- Standardize the simulated final output ---
        # Use a helper if available, otherwise implement basic logic here
        from lollms_server.core.generation import standardize_output # Assuming helper exists
        final_output_list = standardize_output(full_response)

        # Yield individual chunks if possible (only for simple text simulation)
        if len(final_output_list) == 1 and final_output_list[0].get("type") == "text":
            text_content = final_output_list[0].get("data", "")
            if text_content:
                # Simulate word-by-word streaming for text
                words = text_content.split()
                for i, word in enumerate(words):
                    yield {"type": "chunk", "content": word + (" " if i < len(words) - 1 else "")}
                    await asyncio.sleep(0.01) # Small delay
            else:
                 yield {"type":"info", "content":"Empty text result from non-stream generate."}
        elif final_output_list:
            # For non-text or complex list outputs, just yield an info chunk
             yield {"type":"info", "content":"Non-streamable output generated."}
        else:
             yield {"type":"error", "content":"Non-stream generate returned empty result."}


        # Yield the standardized final chunk
        final_metadata = {"status": "complete", "simulated_stream": True}
        # Add metadata from the first standardized output item if it exists
        if final_output_list:
            final_metadata.update(final_output_list[0].get("metadata", {}))

        yield {"type": "final", "content": final_output_list, "metadata": final_metadata}
        return

    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenizes text (Optional)."""
        raise NotImplementedError(f"Binding type '{self.binding_type_name}' does not support tokenization.")

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenizes tokens (Optional)."""
        raise NotImplementedError(f"Binding type '{self.binding_type_name}' does not support detokenization.")

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Estimates resource requirements (Optional). Default assumes GPU needed."""
        return {"gpu_required": True, "estimated_vram_mb": 1024}

    # --- Helpers for Structured Output (Keep these, they use self.generate) ---
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
            llm_response = await self.generate( prompt=prompt, params=structured_params, request_info=request_info, multimodal_data=multimodal_data )
            response_text = ""
            if isinstance(llm_response, list): # Prefer list format
                text_items = [item['data'] for item in llm_response if isinstance(item,dict) and item.get('type')=='text' and item.get('data')]
                response_text = "\n".join(text_items)
            elif isinstance(llm_response, dict) and "text" in llm_response: # Handle older dict format
                response_text = llm_response["text"]
            elif isinstance(llm_response, str): # Handle plain string
                response_text = llm_response
            else:
                logger.warning(f"generate_structured_output unexpected type: {type(llm_response)}. Trying str conversion."); response_text = str(llm_response)

            all_blocks = extract_code_blocks(response_text)
            first_matching_block = None
            for block in all_blocks:
                lang_match = ( not output_language_tag or (block.get('type') and block['type'].lower() == output_language_tag.lower()) )
                if lang_match:
                    first_matching_block = block
                    break # Found the first matching block

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
            llm_response = await self.generate( prompt=prompt, params=params, request_info=request_info, multimodal_data=multimodal_data )
            response_text = ""
            # Handle different return types from generate
            if isinstance(llm_response, list): # Prefer list format
                text_items = [item['data'] for item in llm_response if isinstance(item,dict) and item.get('type')=='text' and item.get('data')]
                response_text = "\n".join(text_items)
            elif isinstance(llm_response, dict) and "text" in llm_response: # Handle older dict format
                response_text = llm_response["text"]
            elif isinstance(llm_response, str): # Handle plain string
                response_text = llm_response
            else:
                logger.warning(f"generate_and_extract_all_codes unexpected type: {type(llm_response)}. Trying str conversion."); response_text = str(llm_response)

            return extract_code_blocks(response_text)
        except Exception as e:
             logger.error(f"Error during generate_and_extract_all_codes for '{self.binding_instance_name}': {e}", exc_info=True)
             return []

    async def ask_yes_no( self, question: str, params: Optional[Dict[str, Any]] = None, request_info: Optional[Dict[str, Any]] = None, multimodal_data: Optional[List['InputData']] = None ) -> Optional[bool]:
        """Asks the LLM a yes/no question."""
        if params is None: params = {}
        if request_info is None: request_info = {}
        base_system_message = params.get("system_message", "")
        separator = "\n\n" if base_system_message else ""
        yes_no_system_message = ( f"{base_system_message}{separator}" f"Evaluate the question: '{question}'.\nYour response MUST be ONLY 'yes' or 'no', enclosed in ```text ... ```." )
        yes_no_params = params.copy()
        yes_no_params["system_message"] = yes_no_system_message
        yes_no_params.setdefault("temperature", 0.1) # Low temp for predictable answer
        yes_no_params.setdefault("max_tokens", 10) # Short response needed

        logger.debug(f"Asking yes/no: '{question}' for instance '{self.binding_instance_name}'")
        try:
            llm_response = await self.generate( prompt=question, params=yes_no_params, request_info=request_info, multimodal_data=multimodal_data )
            response_text = ""
            # Handle different return types from generate
            if isinstance(llm_response, list): # Prefer list format
                text_items = [item['data'] for item in llm_response if isinstance(item,dict) and item.get('type')=='text' and item.get('data')]
                response_text = "\n".join(text_items)
            elif isinstance(llm_response, dict) and "text" in llm_response: # Handle older dict format
                response_text = llm_response["text"]
            elif isinstance(llm_response, str): # Handle plain string
                response_text = llm_response
            else:
                logger.warning(f"ask_yes_no unexpected type: {type(llm_response)}. Trying str conversion."); response_text = str(llm_response)

            code_blocks = extract_code_blocks(response_text)
            answer_block = None
            for block in code_blocks:
                 block_type = block.get('type', 'unknown').lower()
                 # Accept text or unknown block type for yes/no
                 if block_type in ['text', 'unknown']:
                     answer_block = block
                     break # Found the first potential answer block

            if answer_block is None:
                 logger.warning(f"Could not find ``` block in yes/no response for '{self.binding_instance_name}'. Response:\n{response_text}")
                 # Fallback: check direct response text
                 direct_answer = response_text.strip().lower()
                 if direct_answer == "yes": logger.debug("Parsed yes/no directly: YES"); return True
                 elif direct_answer == "no": logger.debug("Parsed yes/no directly: NO"); return False
                 return None # Could not determine

            if not answer_block.get('is_complete', True):
                 logger.warning("Yes/no block seems incomplete.")

            answer = answer_block.get('content', '').strip().lower()
            if answer == "yes":
                logger.debug("Parsed yes/no from block: YES"); return True
            elif answer == "no":
                logger.debug("Parsed yes/no from block: NO"); return False
            else:
                logger.warning(f"Answer in block not 'yes' or 'no': '{answer}'. Raw Content: '{answer_block.get('content')}'"); return None
        except Exception as e:
             logger.error(f"Error during ask_yes_no for '{self.binding_instance_name}' question '{question}': {e}", exc_info=True)
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

        # Stores metadata for *discovered* binding types {type_name: card_data_dict}
        self._discovered_binding_types: Dict[str, Dict[str, Any]] = {}
        # Stores successfully instantiated binding instances {instance_name: BindingInstance}
        self._binding_instances: Dict[str, Binding] = {}
        # Stores errors during type discovery or instance loading
        self._load_errors: Dict[str, str] = {} # {identifier: error_message}

    def _discover_binding_types(self):
        """Scans configured folders for binding directories with valid binding_card.yaml."""
        logger.info("Discovering binding types...")
        self._discovered_binding_types = {}
        self._load_errors = {} # Reset discovery errors
        binding_folders = []
        server_root = get_server_root() # Should be absolute

        # Use paths from the main ConfigGuard object
        example_folder = Path(self.main_config.paths.example_bindings_folder) if self.main_config.paths.example_bindings_folder else None
        personal_folder = Path(self.main_config.paths.bindings_folder) if self.main_config.paths.bindings_folder else None

        # Paths should already be absolute after config initialization
        if example_folder and example_folder.is_dir():
             binding_folders.append(example_folder)
             add_path_to_sys_path(example_folder.parent) # Add parent (e.g., zoos)
             logger.debug(f"Added example binding parent path: {example_folder.parent}")
        else: logger.debug(f"Example bindings folder not found or not configured: {example_folder}")

        if personal_folder and personal_folder.is_dir():
             # Avoid adding same path twice if they resolve identically
             if not example_folder or personal_folder != example_folder:
                  binding_folders.append(personal_folder)
                  add_path_to_sys_path(personal_folder.parent) # Add parent (e.g., project root)
                  logger.debug(f"Added personal binding parent path: {personal_folder.parent}")
             else: logger.debug("Personal binding folder is same as example folder.")
        else: logger.debug(f"Personal bindings folder not found or not configured: {personal_folder}")


        if not binding_folders:
            logger.warning("No binding type folders configured or found.")
            return

        for folder in binding_folders:
            logger.info(f"Scanning for binding types in: {folder}")
            for potential_path in folder.iterdir():
                # Ensure it's a directory and has potential to be a binding package
                if potential_path.is_dir() and not potential_path.name.startswith(('.', '_')):
                    init_file = potential_path / "__init__.py"
                    card_file = potential_path / "binding_card.yaml"
                    # Check for required files
                    if init_file.exists() and card_file.exists():
                        try:
                            # Load the card first to get the type_name and schema
                            with open(card_file, 'r', encoding='utf-8') as f:
                                card_data = yaml.safe_load(f)

                            # Validate basic card structure
                            if not isinstance(card_data, dict):
                                 self._load_errors[str(potential_path)] = f"Invalid binding_card.yaml format (not a dictionary) in {potential_path.name}"; continue
                            if "type_name" not in card_data or not isinstance(card_data["type_name"], str):
                                 self._load_errors[str(potential_path)] = f"Missing or invalid 'type_name' in binding_card.yaml in {potential_path.name}"; continue
                            if "instance_schema" not in card_data or not isinstance(card_data["instance_schema"], dict):
                                 self._load_errors[str(potential_path)] = f"Missing or invalid 'instance_schema' in binding_card.yaml in {potential_path.name}"; continue

                            type_name = card_data["type_name"]
                            card_data["package_path"] = potential_path.resolve() # Store absolute path
                            # Prioritize personal bindings over examples
                            if type_name not in self._discovered_binding_types or folder == personal_folder:
                                self._discovered_binding_types[type_name] = card_data
                                logger.info(f"Discovered binding type: '{type_name}' from {potential_path.name}")
                            else:
                                 logger.debug(f"Ignoring duplicate binding type '{type_name}' from {potential_path.name} (already found in prioritized folder).")
                        except yaml.YAMLError as e:
                            self._load_errors[str(potential_path)] = f"YAML Error in binding_card.yaml: {e}"
                        except Exception as e:
                             self._load_errors[str(potential_path)] = f"Error loading binding card: {e}"
                    else:
                        logger.debug(f"Skipping directory {potential_path.name}: Missing __init__.py or binding_card.yaml")

        logger.info(f"Discovery finished. Found {len(self._discovered_binding_types)} valid binding types.")
        if self._load_errors: logger.warning(f"Binding Type Discovery Errors: {self._load_errors}")

    async def load_bindings(self):
        """Loads and instantiates binding instances based on the main config's bindings_map."""
        self._discover_binding_types() # Find all available binding types first
        self._binding_instances = {} # Reset instances (keep discovery errors)
        instance_configs_dir = get_binding_instances_config_path() # Gets absolute path

        logger.info(f"Loading binding instances based on 'bindings_map' in main config...")
        logger.info(f"Expecting instance config files in: {instance_configs_dir}")

        # Iterate through the bindings_map section from the main config
        bindings_map_section = getattr(self.main_config, "bindings_map", None)
        if not bindings_map_section:
             logger.warning("No 'bindings_map' section found in main configuration. No instances to load.")
             return

        try:
             # Use get_dict() for dynamic sections in ConfigGuard >= 0.4.0
             bindings_to_load = bindings_map_section.get_dict()
        except AttributeError:
             logger.error("Could not retrieve bindings_map dictionary. Check ConfigGuard version or config structure.")
             return
        except Exception as e:
             logger.error(f"Error retrieving bindings_map: {e}", exc_info=True)
             return

        if not bindings_to_load:
            logger.info("Bindings map is empty. No binding instances configured.")
            return

        for instance_name, type_name in bindings_to_load.items():
            if not isinstance(type_name, str) or not type_name:
                 err_msg = f"Invalid type name '{type_name}' for instance '{instance_name}' in bindings_map."
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue

            logger.info(f"--- Loading instance: '{instance_name}' (Type: '{type_name}') ---")

            # 1. Find Discovered Type Info
            if type_name not in self._discovered_binding_types:
                 err_msg = f"Binding type '{type_name}' required by instance '{instance_name}' not discovered or failed discovery."
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            type_card_data = self._discovered_binding_types[type_name]
            instance_schema = type_card_data.get("instance_schema")
            package_path = type_card_data.get("package_path") # Should be absolute Path obj
            if not instance_schema or not isinstance(instance_schema, dict):
                 err_msg = f"Instance schema missing or invalid in binding card for type '{type_name}'."
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            if not package_path or not isinstance(package_path, Path):
                 err_msg = f"Package path missing for type '{type_name}'."
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue

            # 2. Load Instance Configuration using ConfigGuard
            instance_config_guard: Optional[ConfigGuard] = None
            instance_config_dict: Optional[Dict[str, Any]] = None
            # Try finding file with common extensions
            instance_config_path = None
            found_handler_class = None
            for ext, handler_cls in _binding_handler_map.items():
                 potential_path = instance_configs_dir / f"{instance_name}{ext}"
                 if potential_path.is_file():
                     instance_config_path = potential_path
                     found_handler_class = handler_cls
                     break # Found the file

            if not instance_config_path:
                 err_msg = f"Config file for instance '{instance_name}' not found in {instance_configs_dir} (tried common extensions)."
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue

            try:
                logger.info(f"Loading instance config from: {instance_config_path}")
                # Add the expected type and instance name to the schema for validation/clarity
                # This ensures the loaded config dict contains these essential fields.
                schema_copy = instance_schema.copy() # Don't modify original
                schema_copy.setdefault("type", {"type": "str", "default": type_name, "help": "Internal type name."})
                schema_copy.setdefault("binding_instance_name", {"type": "str", "default": instance_name, "help":"Internal instance name."})

                instance_config_guard = ConfigGuard(
                    schema=schema_copy,
                    config_path=instance_config_path,
                    encryption_key=self.encryption_key, # Use key from main config/env
                    handler=found_handler_class() # Instantiate the found handler
                )
                instance_config_guard.load() # Load the specific settings for this instance
                instance_config_dict = instance_config_guard.get_config_dict()

                # Verify loaded type matches expected type from map
                loaded_type = instance_config_dict.get("type")
                if loaded_type != type_name:
                     raise ConfigGuardError(f"Instance config file '{instance_config_path.name}' has type '{loaded_type}', but map expects '{type_name}'.")

                # Ensure binding_instance_name is correctly set (should be by default)
                instance_config_dict["binding_instance_name"] = instance_name

            except ConfigGuardError as e:
                 err_msg = f"Error loading/validating config for instance '{instance_name}' from {instance_config_path}: {e}"
                 logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg; continue
            except ImportError as e: # Missing handler dependency
                 err_msg = f"Missing dependency for instance config '{instance_name}' ({instance_config_path.suffix}): {e}"
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            except Exception as e:
                 err_msg = f"Unexpected error loading instance config '{instance_name}': {e}"
                 logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg; continue

            # 3. Import Binding Package/Module
            BindingClass: Optional[Type[Binding]] = None
            try:
                # Use the absolute package path found during discovery
                # Example: package_path = /path/to/server/zoos/bindings/ollama_binding
                # parent_dir_name = zoos/bindings ; package_path.name = ollama_binding
                # package_import_name = zoos.bindings.ollama_binding (assuming zoos is in sys.path)
                parent_dir_name = package_path.parent.name
                package_module_name = package_path.name
                # Construct relative import path from the parent dir added to sys.path
                package_import_name = f"{parent_dir_name}.{package_module_name}"

                logger.info(f"Importing binding package: {package_import_name}")
                # Use safe_load_module which handles adding path temporarily if needed
                # module, import_error = safe_load_module(package_path / "__init__.py", package_path=package_path)
                # Or simpler, rely on sys.path being set correctly earlier:
                module = importlib.import_module(package_import_name)

                if not module: raise ImportError("Module loaded as None")

                # Find the class inheriting from Binding with the correct type_name
                found_classes = find_classes_in_module(module, Binding)
                logger.debug(f"Found potential binding classes in {package_import_name}: {[c.__name__ for c in found_classes]}")
                for cls in found_classes:
                    # Check type_name defined in class or loaded from its card
                    cls_type_name = getattr(cls, 'binding_type_name', None)
                    # We don't need to load the card again here, type_name should be on the class
                    # if not cls_type_name:
                    #     try: cls_type_name = cls.get_binding_card().get("type_name")
                    #     except Exception: pass

                    if cls_type_name == type_name:
                        BindingClass = cls
                        logger.info(f"Found matching Binding class: {BindingClass.__name__}")
                        break
                if not BindingClass:
                    raise TypeError(f"No class with binding_type_name='{type_name}' found in package '{package_import_name}'. Found: {[c.__name__ for c in found_classes]}")

            except ImportError as e:
                 err_msg = f"Failed import package '{package_import_name}' for type '{type_name}': {e}"
                 logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg; continue
            except TypeError as e:
                 err_msg = f"Error finding class for type '{type_name}': {e}"
                 logger.error(err_msg); self._load_errors[instance_name] = err_msg; continue
            except Exception as e:
                 err_msg = f"Error loading/inspecting package for type '{type_name}': {e}"
                 logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg; continue

            # 4. Instantiate Binding
            try:
                logger.info(f"Instantiating binding '{instance_name}' (type: {type_name})")
                # Pass the instance config *dictionary* to the constructor
                instance = BindingClass(config=instance_config_dict, resource_manager=self.resource_manager)

                # 5. Health Check (Optional but recommended)
                healthy, message = await instance.health_check()
                if healthy:
                    logger.info(f"Successfully instantiated '{instance_name}'. Health OK: {message}")
                    self._binding_instances[instance_name] = instance
                else:
                    err_msg = f"Health check failed for instance '{instance_name}': {message}"
                    logger.error(err_msg)
                    # Store error, but don't add instance if health check fails initially
                    self._load_errors[instance_name] = err_msg

            except Exception as e:
                 err_msg = f"Failed instantiate binding '{instance_name}': {e}"
                 logger.error(err_msg, exc_info=True); self._load_errors[instance_name] = err_msg


        logger.info(f"Finished loading instances. Instantiated: {len(self._binding_instances)}. Errors: {len(self._load_errors)}")
        if self._load_errors: logger.warning(f"Instance Loading Errors: {self._load_errors}")


    def list_binding_types(self) -> Dict[str, Dict[str, Any]]:
        """Returns metadata for discovered binding types from their cards."""
        types_info = {}
        for type_name, card_data in self._discovered_binding_types.items():
            # Return a subset of card data, exclude schema and path for listing
            info = {k: v for k, v in card_data.items() if k not in ["instance_schema", "package_path"]}
            types_info[type_name] = info
        return types_info

    def list_binding_instances(self) -> Dict[str, Dict[str, Any]]:
        """Returns the configuration dictionary for successfully instantiated bindings."""
        instances_info = {}
        for name, instance in self._binding_instances.items():
            try:
                # Return the dictionary that was passed during init
                # Make a copy to avoid external modification
                config_copy = instance.get_instance_config()
                # Maybe remove sensitive keys like api_key before returning?
                # config_copy.pop('api_key', None) # Example removal
                instances_info[name] = config_copy
            except Exception as e:
                logger.warning(f"Could not get config dict for instance '{name}': {e}")
                instances_info[name] = {"error": "Failed to retrieve config dict", "type": instance.binding_type_name}
        return instances_info

    def get_binding(self, instance_name: str) -> Optional[Binding]:
        """Gets a specific binding instance by its configured name."""
        instance = self._binding_instances.get(instance_name)
        if not instance:
             logger.error(f"Binding instance '{instance_name}' not found or failed load. Check config/logs.")
             # Report specific load error if available
             if instance_name in self._load_errors:
                 logger.error(f" -> Instance Load Error: {self._load_errors[instance_name]}")
             # Check if it was defined in map but failed load vs not defined at all
             elif instance_name in (getattr(self.main_config, "bindings_map", {}) or {}):
                 logger.error(" -> Instance was defined in bindings_map but failed to load.")
             else:
                 logger.error(" -> Instance was not found in bindings_map.")

        return instance

    async def cleanup(self):
        """Cleans up resources used by binding instances (e.g., unloads models)."""
        logger.info("Cleaning up binding instances...")
        unload_tasks = []
        for name, instance in self._binding_instances.items():
            if hasattr(instance, 'unload_model') and callable(instance.unload_model):
                logger.debug(f"Scheduling unload for instance '{name}'...")
                unload_tasks.append(instance.unload_model())
            else:
                 logger.debug(f"Instance '{name}' does not have an unload_model method.")

        results = await asyncio.gather(*unload_tasks, return_exceptions=True)
        instance_names = list(self._binding_instances.keys()) # Get names before iterating results

        for i, result in enumerate(results):                                                
            instance_name = instance_names[i] # Correlate result with instance name
            if isinstance(result, Exception):
                 logger.error(f"Error unloading model for '{instance_name}': {result}", exc_info=result)
            else:
                 logger.info(f"Unload successful for '{instance_name}'.")
        logger.info("Binding cleanup finished.") 