# encoding:utf-8
# Project: lollms_server
# File: lollms_server/core/personalities.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Manages LOLLMS personalities discovery and loading.

import yaml
from pathlib import Path
import asyncio
import inspect # Needed for Personality class method potentially
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

# Use ascii_colors for logging if available
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    # Define a fallback logger and ASCIIColors class if ascii_colors is not installed
    class MockASCIIColors:
        @staticmethod
        def error(msg): logging.error(msg)
        @staticmethod
        def warning(msg): logging.warning(msg)
        @staticmethod
        def info(msg): logging.info(msg)
        @staticmethod
        def success(msg): logging.info(msg) # No color fallback
        @staticmethod
        def config(msg): logging.info(msg) # No color fallback
    ASCIIColors = MockASCIIColors # Assign the mock class

    def trace_exception(e):
        logging.exception(e)

# Use TYPE_CHECKING for ConfigGuard import
from typing import TYPE_CHECKING
from configguard import ConfigGuard, ConfigSection
if TYPE_CHECKING:
    # Other type hints needed for context/return values
    try: from lollms_server.core.bindings import Binding # For context hint in Personality
    except ImportError: Binding = Any # type: ignore
    try: from lollms_server.api.models import OutputData # For type hint in Personality
    except ImportError: OutputData = Dict # type: ignore


# Assuming these utils are correctly defined elsewhere
from lollms_server.utils.file_utils import safe_load_module, add_path_to_sys_path
# Import Pydantic for PersonalityConfig (yaml structure) - keep this
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# --- Personality Configuration Model (from config.yaml) ---
# This Pydantic model defines the expected structure WITHIN individual personality config.yaml files
class PersonalityConfig(BaseModel):
    # --- Core Lollms Fields ---
    name: str = Field(..., description="The unique name of the personality.")
    version: Union[str, float, int] = Field(..., description="Version identifier for the personality.")
    author: str = Field(..., description="The author or creator of the personality.")
    personality_description: str = Field(..., description="A brief description of what the personality does or represents.")
    personality_conditioning: str = Field(..., description="The main system prompt or instructions given to the language model.")

    # Optional fields with defaults
    lollms_version: Optional[str] = Field(None, description="The lollms version this personality was designed for (optional).")
    user_name: Optional[str] = Field("user", description="Default name for the user interacting with the personality.")
    language: Optional[str] = Field("english", description="Primary language of the personality.")
    category: Optional[str] = Field("generic", description="Category for organizing personalities (e.g., 'writing', 'coding', 'roleplay').")
    welcome_message: Optional[str] = Field("Welcome! How can I help you today?", description="Initial message sent by the personality.")
    user_message_prefix: Optional[str] = Field("User", description="Prefix added before user messages in the context.")
    link_text: Optional[str] = Field("\n", description="Separator text between messages in the context.")
    ai_message_prefix: Optional[str] = Field("Assistant", description="Prefix added before AI messages in the context.")
    dependencies: List[str] = Field(default_factory=list, description="List of Python packages required by the personality's script.")
    anti_prompts: List[str] = Field(default_factory=list, description="List of strings that, if generated, should cause the generation to stop.")
    disclaimer: Optional[str] = Field("", description="A disclaimer message shown to the user (e.g., for legal or safety warnings).")

    # Default Model Parameters (can be overridden)
    model_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Default sampling temperature (creativity vs. focus).")
    model_n_predicts: Optional[int] = Field(None, gt=0, description="Default maximum number of tokens to generate (also known as max_tokens).")
    model_top_k: Optional[int] = Field(None, gt=0, description="Default top-K sampling parameter.")
    model_top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Default top-P (nucleus) sampling parameter.")
    model_repeat_penalty: Optional[float] = Field(None, gt=0.0, description="Default penalty for repeating tokens.")
    model_repeat_last_n: Optional[int] = Field(None, ge=0, description="Default window size for repeat penalty.")

    # Lollms WebUI specific prompt list (metadata for potential UIs)
    prompts_list: List[str] = Field(default_factory=list, description="List of example prompts suitable for this personality.")

    # --- Fields needed for lollms_server logic ---
    tags: List[str] = Field(default_factory=list, description="Keywords for searching or filtering personalities.")
    icon: Optional[str] = Field("default.png", description="Filename of the personality icon within its assets folder.")
    script_path: Optional[str] = Field(None, description="Relative path to the workflow script (e.g., 'scripts/workflow.py') within the personality folder.")

    # Allow extra fields not explicitly defined in the model
    model_config = {
        "extra": "allow"
    }


# --- Personality Representation ---
class Personality:
    """Represents a loaded lollms personality, combining its config.yaml data and override settings."""
    def __init__(
            self,
            config: PersonalityConfig, # Parsed from config.yaml using Pydantic
            path: Path, # Absolute path to the personality's main folder
            assets_path: Path, # Absolute path to the assets subfolder
            script_module: Optional[Any] = None, # Loaded Python module if scripted
            # Store the raw override dictionary from main config's [personalities_config] section
            instance_override_config: Optional[Dict[str, Any]] = None
        ):
        """
        Initializes a Personality instance.

        Args:
            config: The validated PersonalityConfig object from config.yaml.
            path: The absolute path to the personality's directory.
            assets_path: The absolute path to the personality's assets directory.
            script_module: The loaded Python module if the personality is scripted, otherwise None.
            instance_override_config: A dictionary containing overrides from the main server configuration
                                     (e.g., {'enabled': False}). Defaults to an empty dict.
        """
        self.name: str = config.name
        self.config: PersonalityConfig = config # The Pydantic model from config.yaml
        self.path: Path = path
        self.assets_path: Path = assets_path
        self.script_module: Optional[Any] = script_module
        # Store instance_override_config, ensuring it's a dictionary
        self.instance_override_config: Dict[str, Any] = instance_override_config if isinstance(instance_override_config, dict) else {}

    @property
    def is_enabled(self) -> bool:
        """Checks if the personality is explicitly disabled in the main server config."""
        # Defaults to True if 'enabled' key is missing in the override config
        enabled_override = self.instance_override_config.get('enabled')
        if isinstance(enabled_override, bool):
            return enabled_override
        elif enabled_override is not None:
             # Log if the 'enabled' value is not a boolean as expected
             logger.warning(f"Personality '{self.name}' has non-boolean 'enabled' override value ({enabled_override}). Treating as enabled.")
             return True
        else:
             # Key 'enabled' not present, default to enabled
             return True

    @property
    def is_scripted(self) -> bool:
        """Returns True if this personality has a loaded script module with a 'run_workflow' function."""
        return self.script_module is not None and hasattr(self.script_module, 'run_workflow')

    async def run_workflow(
        self,
        prompt: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
        # Define a more precise return type if possible, encompassing all potential returns
    ) -> Union[str, Dict, List['OutputData'], AsyncGenerator[Dict[str, Any], None]]:
        """
        Executes the 'run_workflow' function from the personality's script module.

        Args:
            prompt: The primary text prompt from the user.
            params: Generation parameters (merged from defaults, personality, request).
            context: A dictionary containing server components (binding, managers, config etc.)
                     and request details ('input_data', 'request_info').

        Returns:
            The result from the workflow script, which could be:
            - A string (simple text response).
            - A dictionary (e.g., for single image/audio output - DEPRECATED, prefer list).
            - A list of OutputData-like dictionaries (standardized multi-output format).
            - An async generator yielding StreamChunk-like dictionaries for streaming output.

        Raises:
            NotImplementedError: If the personality is not scripted.
            AttributeError: If the script module lacks a 'run_workflow' function.
            Exception: Propagates exceptions raised within the workflow script.
        """
        if not self.is_scripted or not self.script_module:
            logger.error(f"Attempted to run workflow for non-scripted personality '{self.name}'.")
            raise NotImplementedError(f"Personality '{self.name}' is not scripted or its script failed to load.")

        workflow_func = getattr(self.script_module, 'run_workflow', None)
        if not callable(workflow_func):
            logger.error(f"Script for personality '{self.name}' is missing a callable 'run_workflow' function.")
            raise AttributeError(f"Script for personality '{self.name}' does not have a callable 'run_workflow' function.")

        logger.info(f"Executing run_workflow for scripted personality '{self.name}'...")
        try:
            if asyncio.iscoroutinefunction(workflow_func):
                # Directly await async workflow functions
                return await workflow_func(prompt, params, context)
            else:
                # Run synchronous workflow functions using asyncio.to_thread to avoid blocking
                logger.warning(f"'run_workflow' in {self.name} is synchronous. Running in thread executor.")
                loop = asyncio.get_running_loop()
                # Ensure workflow_func receives expected arguments correctly
                return await loop.run_in_executor(None, lambda: workflow_func(prompt, params, context))
        except Exception as e:
            logger.error(f"Error executing run_workflow for personality '{self.name}': {e}", exc_info=True)
            trace_exception(e)
            # Return a standardized error format
            return [{"type":"error", "data": f"Workflow execution failed: {e}", "metadata":{"personality": self.name}}]


# --- Personality Manager ---
class PersonalityManager:
    """Discovers personalities, loads their configurations, and manages access."""

    def __init__(self, main_config: 'ConfigGuard'):
        """
        Initializes the PersonalityManager.

        Args:
            main_config: The loaded ConfigGuard object for the main server configuration.
        """
        self.main_config: 'ConfigGuard' = main_config
        self._personalities: Dict[str, Personality] = {} # Stores loaded Personality objects {name: Personality}
        self._load_errors: Dict[str, str] = {} # Stores errors during loading {path_str: error_message}
        self._dependency_errors: Dict[str, str] = {} # Stores dependency installation errors {path_str: error_message}

    def load_personalities(self):
        """
        Scans configured folders (example and personal) for personality directories,
        parses their config.yaml, checks enablement status from the main config,
        and loads valid and enabled personalities.
        """
        logger.info("Loading personalities...")
        self._personalities = {} # Reset loaded personalities
        self._load_errors = {}   # Reset load errors
        self._dependency_errors = {} # Reset dependency errors

        personality_folders_to_scan: List[Path] = []

        # Access paths configuration section from the main ConfigGuard object
        # Paths should already be resolved to absolute by initialize_config
        paths_config = self.main_config.paths

        # Get example personalities path
        example_folder_str = getattr(paths_config, "example_personalities_folder", None)
        if example_folder_str:
            example_folder = Path(example_folder_str)
            if example_folder.is_dir():
                personality_folders_to_scan.append(example_folder)
                add_path_to_sys_path(example_folder.parent) # Add parent (e.g., zoos)
                logger.debug(f"Added example personality parent path: {example_folder.parent}")
            else:
                 logger.warning(f"Example personalities folder path specified but not found: {example_folder}")
        else:
             logger.debug("Example personalities folder not configured.")


        # Get personal personalities path
        personal_folder_str = getattr(paths_config, "personalities_folder", None)
        if personal_folder_str:
            personal_folder = Path(personal_folder_str)
            if personal_folder.is_dir():
                # Avoid adding the same path twice if example/personal point to the same place
                if not example_folder or personal_folder != example_folder:
                     personality_folders_to_scan.append(personal_folder)
                     add_path_to_sys_path(personal_folder.parent) # Add parent (e.g., project root)
                     logger.debug(f"Added personal personality parent path: {personal_folder.parent}")
                else: logger.debug("Personal personalities folder is same as example folder.")
            else:
                 logger.warning(f"Personal personalities folder path specified but not found: {personal_folder}")
        else:
            logger.debug("Personal personalities folder not configured.")


        if not personality_folders_to_scan:
            logger.warning("No valid personality folders configured or found. No personalities will be loaded.")
            return

        # Get the personality override configuration map from the main config
        # Use getattr for safe access as the section might be null or not present
        instance_config_section: Optional[ConfigSection] = getattr(self.main_config, "personalities_config", None)
        # Convert ConfigSection to dict; defaults to empty dict if section is missing/null
        instance_override_map: Dict[str, Any] = {}
        if isinstance(instance_config_section, ConfigSection):
            try:
                instance_override_map = instance_config_section.get_dict()
            except Exception as e:
                 logger.error(f"Failed to get dictionary from personalities_config section: {e}")
        elif instance_config_section is not None:
            logger.warning(f"personalities_config section is not a ConfigSection object: {type(instance_config_section)}. Ignoring overrides.")


        logger.debug(f"Personality override map loaded from main config: {instance_override_map}")

        # --- Load Personalities Synchronously During Startup ---
        for folder in personality_folders_to_scan:
            logger.info(f"Scanning for personalities in: {folder}")
            try:
                for potential_path in folder.iterdir():
                    # Check if it's a directory and not hidden
                    if potential_path.is_dir() and not potential_path.name.startswith('.'):
                        personality_folder_name = potential_path.name

                        # --- Get the override dictionary for this specific personality ---
                        # The override map keys are the personality folder names
                        instance_override_config: Optional[Dict[str, Any]] = instance_override_map.get(personality_folder_name)

                        # --- Check enablement status from override ---
                        is_enabled = True # Default to enabled
                        if isinstance(instance_override_config, dict):
                            enabled_value = instance_override_config.get('enabled', True) # Default to True if key missing
                            if isinstance(enabled_value, bool):
                                is_enabled = enabled_value
                            else:
                                logger.warning(f"Non-boolean 'enabled' value ({enabled_value}) found for '{personality_folder_name}' in main config's personalities_config. Assuming enabled.")
                        elif instance_override_config is not None:
                            # Log if the override exists but isn't a dictionary as expected
                            logger.warning(f"Invalid format for override config of '{personality_folder_name}' in main config. Expected dict, got {type(instance_override_config)}. Assuming enabled.")

                        if not is_enabled:
                            logger.info(f"Skipping disabled personality: '{personality_folder_name}' (disabled in main config).")
                            continue # Skip to the next potential personality

                        # --- Load the personality (if enabled) ---
                        # Pass the raw override dictionary (or None) to the loading function
                        self._load_personality(potential_path, instance_override_config)
            except OSError as e:
                 logger.error(f"OS Error scanning directory {folder}: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error scanning directory {folder}: {e}", exc_info=True)


        logger.info(f"Finished loading. Loaded {len(self._personalities)} enabled personalities.")
        if self._load_errors:
            logger.warning(f"Encountered config errors during personality loading: {self._load_errors}")
        if self._dependency_errors:
            logger.warning(f"Encountered dependency errors during personality loading: {self._dependency_errors}")


    def _load_personality(self, path: Path, instance_override_config: Optional[Dict[str, Any]]):
        """
        Loads a single personality from its directory, including parsing config.yaml,
        checking dependencies, and loading scripts.

        Args:
            path: The absolute path to the personality directory.
            instance_override_config: The override dictionary from the main server config, or None.
        """
        config_file = path / "config.yaml"
        assets_path = path / "assets" # Define expected assets path
        script_module = None
        personality_folder_name = path.name # For logging

        if not config_file.is_file():
            # Don't log error, just skip folders without config.yaml
            logger.debug(f"Skipping folder {path.name}: No config.yaml found.")
            return

        try:
            # 1. Load and Parse config.yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            if not config_data or not isinstance(config_data, dict):
                 raise ValueError("config.yaml is empty or not a dictionary")

            # 2. Validate config.yaml structure using Pydantic
            try:
                 p_config = PersonalityConfig(**config_data)
            except ValidationError as e:
                 # Provide clearer error including the path
                 error_detail = f"Invalid config.yaml structure in '{config_file}': {e}"
                 raise ValueError(error_detail) from e

            # 3. Check Dependencies (Synchronously)
            if p_config.dependencies:
                logger.info(f"Checking dependencies for personality '{p_config.name}': {p_config.dependencies}")
                try:
                    # Assuming dependency_manager is refactored to work without AppConfig if needed
                    from lollms_server.utils.dependency_manager import ensure_dependencies_sync
                    deps_ok = ensure_dependencies_sync(p_config.dependencies, source_name=f"Personality '{p_config.name}'")
                    if not deps_ok:
                        err_msg = f"Dependency installation failed for personality '{p_config.name}'. It may not function correctly."
                        logger.error(err_msg)
                        # Store error but continue loading the personality
                        self._dependency_errors[str(path)] = err_msg
                except ImportError:
                     logger.error("Could not import dependency_manager to check personality dependencies.")
                except Exception as dep_err:
                     logger.error(f"Error checking dependencies for '{p_config.name}': {dep_err}", exc_info=True)
                     self._dependency_errors[str(path)] = f"Error checking dependencies: {dep_err}"


            # 4. Load Script Module (if specified)
            if p_config.script_path:
                script_file = (path / p_config.script_path).resolve()
                if script_file.is_file() and script_file.suffix == ".py":
                    logger.info(f"Attempting to load script for personality '{p_config.name}': {script_file}")
                    add_path_to_sys_path(script_file.parent) # Add script's directory to path
                    # Use safe_load_module which handles imports within the script's package context
                    module, error = safe_load_module(script_file, package_path=path) # Pass personality path as package path

                    if module:
                        script_module = module
                        logger.info(f"Successfully loaded script module for '{p_config.name}'.")
                        if not hasattr(script_module, 'run_workflow'):
                            logger.warning(f"Script for personality '{p_config.name}' loaded but does not contain a 'run_workflow' function.")
                    else:
                        err_msg = f"Failed to load script '{script_file}' for personality '{p_config.name}': {error}"
                        logger.error(err_msg)
                        # Append script load error to any existing errors for this path
                        self._load_errors[str(path)] = self._load_errors.get(str(path), "") + f"; Script load error: {error}"
                        script_module = None # Ensure module is None on failure
                else:
                    logger.warning(f"Script path '{p_config.script_path}' defined for personality '{p_config.name}' but file not found or invalid: {script_file}")

            # 5. Create Personality Object
            # Ensure assets path exists for consistency, even if empty
            if not assets_path.exists():
                try:
                    assets_path.mkdir()
                    logger.debug(f"Created missing assets directory: {assets_path}")
                except OSError as e:
                    logger.warning(f"Could not create assets directory for {personality_folder_name}: {e}")


            # Pass the validated Pydantic config and the raw override dict
            personality = Personality(
                config=p_config,
                path=path.resolve(), # Ensure path is absolute
                assets_path=assets_path.resolve(), # Ensure path is absolute
                script_module=script_module,
                instance_override_config=instance_override_config # Pass overrides
            )

            # Store the loaded personality, potentially overwriting duplicates (personal > example)
            # Use personality name from validated config
            if personality.name in self._personalities:
                logger.warning(f"Duplicate personality name '{personality.name}' found. Overwriting previous entry with personality from {path}.")
            self._personalities[personality.name] = personality
            logger.info(f"Successfully loaded personality: '{personality.name}' from {path}")

        except yaml.YAMLError as e:
            err_msg = f"Error parsing YAML in {config_file}: {e}"
            logger.error(err_msg); self._load_errors[str(path)] = err_msg
        except ValueError as e: # Catches Pydantic validation errors and empty file errors
            err_msg = f"Error processing personality config at {path}: {e}"
            logger.error(err_msg); self._load_errors[str(path)] = err_msg
        except Exception as e:
            err_msg = f"Unexpected error loading personality from {path}: {e}"
            logger.error(err_msg, exc_info=True); self._load_errors[str(path)] = err_msg


    def list_personalities(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary of currently loaded and enabled personalities
        and their basic metadata extracted from their PersonalityConfig.
        """
        info: Dict[str, Dict[str, Any]] = {}
        for name, p in self._personalities.items():
            # Check if the personality is enabled according to its override config
            if not p.is_enabled:
                continue # Skip disabled personalities

            # Extract metadata primarily from the p.config (PersonalityConfig model)
            try:
                # Use model_dump for Pydantic v2 compatibility if needed,
                # but accessing attributes directly is fine here.
                personality_info = {
                    "name": p.name,
                    "author": p.config.author,
                    "version": str(p.config.version), # Ensure version is string
                    "description": p.config.personality_description,
                    "category": p.config.category,
                    "language": p.config.language,
                    "tags": p.config.tags or [],
                    "icon": p.config.icon, # Filename relative to assets
                    "is_scripted": p.is_scripted,
                    "path": str(p.path), # Include path for reference
                }
                info[name] = personality_info
            except Exception as e:
                 logger.warning(f"Error retrieving metadata for personality '{name}': {e}")
                 # Optionally add an error entry or skip
                 # info[name] = {"name": name, "error": "Failed to retrieve metadata"}

        return info

    def get_personality(self, name: str) -> Optional[Personality]:
        """
        Gets a loaded and enabled personality instance by its unique name.

        Args:
            name: The name of the personality to retrieve.

        Returns:
            The Personality object if found and enabled, otherwise None.
        """
        personality = self._personalities.get(name)
        # Return only if found AND enabled property returns True
        if personality and personality.is_enabled:
             return personality
        elif personality: # Found but disabled
             logger.debug(f"Attempted to get disabled personality: '{name}'")
             return None
        else: # Not found
             logger.debug(f"Personality '{name}' not found.")
             return None