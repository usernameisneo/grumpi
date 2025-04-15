# lollms_server/core/personalities.py
import yaml
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError
import asyncio # Keep asyncio if used elsewhere in the file

# Assuming these are correctly imported or defined elsewhere
from .config import AppConfig
from lollms_server.utils.file_utils import safe_load_module, add_path_to_sys_path

logger = logging.getLogger(__name__)
# --- Updated Personality Configuration Model ---

class PersonalityConfig(BaseModel):
    # --- Core Lollms Fields ---
    # Required fields based on common usage and your example
    name: str
    version: Union[str, float, int]
    author: str # Even if not top-level in example, it's crucial metadata
    personality_description: str # Description of the personality
    personality_conditioning: str # The core instructions/system prompt

    # Optional fields from your example with defaults where appropriate
    lollms_version: Optional[str] = None
    user_name: Optional[str] = "user"
    language: Optional[str] = "english"
    category: Optional[str] = "generic"
    welcome_message: Optional[str] = "Welcome! How can I help you today?" # Default welcome
    user_message_prefix: Optional[str] = "User"
    link_text: Optional[str] = "\n"
    ai_message_prefix: Optional[str] = "Assistant" # Generic default, specific personalities override
    dependencies: List[str] = Field(default_factory=list)
    anti_prompts: List[str] = Field(default_factory=list) # Often used to stop generation
    disclaimer: Optional[str] = ""

    # Default Model Parameters (optional in the model, allows overrides)
    model_temperature: Optional[float] = None
    model_n_predicts: Optional[int] = None # Max tokens equivalent
    model_top_k: Optional[int] = None
    model_top_p: Optional[float] = None
    model_repeat_penalty: Optional[float] = None
    model_repeat_last_n: Optional[int] = None

    # Lollms WebUI specific prompt list (metadata for potential UIs)
    prompts_list: List[str] = Field(default_factory=list)

    # --- Fields needed for lollms_server logic ---
    # These might not be in every lollms config but are useful for server features
    tags: List[str] = Field(default_factory=list) # Useful for categorization/searching
    icon: Optional[str] = "default.png" # Relative path within personality assets/
    script_path: Optional[str] = None # For scripted personalities

    # Allow extra fields not explicitly defined in the model,
    # useful if lollms adds new fields later.
    model_config = {
        "extra": "allow"
    }


# --- Personality Representation ---
# (The Personality class remains largely the same, but accesses updated config fields)
class Personality:
    """Represents a loaded lollms personality."""
    def __init__(self, config: PersonalityConfig, path: Path, assets_path: Path, script_module: Optional[Any] = None):
        self.config = config
        self.path = path
        self.assets_path = assets_path
        self.script_module = script_module
        # Use the required 'name' field from the config
        self.name = config.name

    @property
    def is_scripted(self) -> bool:
        """Returns True if this personality has an associated script."""
        return self.script_module is not None and hasattr(self.script_module, 'run_workflow')

    async def run_workflow(self, prompt: str, params: Dict[str, Any], context: Dict[str, Any]) -> Any:
         """ Executes the run_workflow function from the personality's script. """
         if not self.is_scripted or not self.script_module:
             raise NotImplementedError(f"Personality '{self.name}' is not scripted or script failed to load.")

         workflow_func = getattr(self.script_module, 'run_workflow', None)
         if not callable(workflow_func):
             raise AttributeError(f"Script for personality '{self.name}' does not have a callable 'run_workflow' function.")

         logger.info(f"Executing run_workflow for scripted personality '{self.name}'")
         try:
             if asyncio.iscoroutinefunction(workflow_func):
                 return await workflow_func(prompt, params, context)
             else:
                 logger.warning(f"'run_workflow' in {self.name} is not async. Consider making it async to avoid blocking.")
                 # Execute sync function directly (potential blocking risk)
                 return workflow_func(prompt, params, context)
         except Exception as e:
             logger.error(f"Error executing run_workflow for personality '{self.name}': {e}", exc_info=True)
             raise


# --- Personality Manager ---
# (PersonalityManager class and its methods remain the same,
#  _load_personality will now use the updated PersonalityConfig model for validation)
class PersonalityManager:
    """Discovers and manages lollms personalities."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._personalities: Dict[str, Personality] = {}
        self._load_errors: Dict[str, str] = {}
        self._dependency_errors: Dict[str, str] = {} # Store dependency errors

    def load_personalities(self):
        """Scans configured folders and loads valid personalities."""
        logger.info("Loading personalities...")
        self._personalities = {}
        self._load_errors = {}
        self._dependency_errors = {}

        personality_folders = []
        # ... (folder scanning logic remains the same) ...
        if self.config.paths.example_personalities_folder and self.config.paths.example_personalities_folder.exists():
            personality_folders.append(self.config.paths.example_personalities_folder)
            add_path_to_sys_path(self.config.paths.example_personalities_folder.parent) # Add parent for potential relative imports in scripts

        if self.config.paths.personalities_folder and self.config.paths.personalities_folder.exists():
            personality_folders.append(self.config.paths.personalities_folder)
            add_path_to_sys_path(self.config.paths.personalities_folder.parent)

        if not personality_folders:
            logger.warning("No personality folders configured or found.")
            return
        # Get the configuration map for personalities
        enabled_map = self.config.personalities_config or {}
        
        # Use asyncio.gather to run async loading tasks concurrently
        # Need to make _load_personality async first
        # For now, keep it synchronous loading during startup
        for folder in personality_folders:
            logger.info(f"Scanning for personalities in: {folder}")
            for potential_path in folder.iterdir():
                if potential_path.is_dir():
                    personality_folder_name = potential_path.name
                    # --- Check if personality is explicitly disabled in config ---
                    instance_config = enabled_map.get(personality_folder_name)
                    is_enabled = True # Default to enabled
                    if instance_config is not None: # Check if an entry exists
                        is_enabled = instance_config.enabled # Use the configured value

                    if not is_enabled:
                        logger.info(f"Skipping disabled personality: '{personality_folder_name}' (disabled in config.toml)")
                        continue # Skip loading this personality
                    # --- End Check ---
                    self._load_personality(potential_path) # Keep sync for now

        logger.info(f"Loaded {len(self._personalities)} personalities.")
        if self._load_errors:
            logger.warning(f"Encountered config errors during personality loading: {self._load_errors}")
        if self._dependency_errors:
            logger.warning(f"Encountered dependency errors during personality loading: {self._dependency_errors}")


    # Make _load_personality async to handle async dependency checks
    async def _load_personality_async(self, path: Path):
        # ... (async version of loading including await ensure_dependencies) ...
        # This might be needed if dependency checks become slow, but adds complexity
        # to the initial synchronous load_personalities call in main.py lifespan.
        # Let's stick to sync loading for now and handle deps there.
        pass


    # Keep _load_personality synchronous for now to simplify startup lifespan
    def _load_personality(self, path: Path):
        """Loads a single personality from its directory."""
        config_file = path / "config.yaml"
        assets_path = path / "assets"
        script_module = None

        if not config_file.exists():
            return

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data or not isinstance(config_data, dict):
                 raise ValueError("config.yaml is empty or not a dictionary")

            # Validate config data using the updated Pydantic model
            try:
                 p_config = PersonalityConfig(**config_data)
            except ValidationError as e:
                 # Provide a clearer error message including the path
                 error_detail = f"Invalid config.yaml structure in '{config_file}': {e}"
                 raise ValueError(error_detail) from e

            # --- Dependency Check ---
            # Run dependency check synchronously during load
            if p_config.dependencies:
                logger.info(f"Checking dependencies for personality '{p_config.name}': {p_config.dependencies}")
                # Note: This runs pipmaster synchronously, which might block startup
                # If this becomes an issue, loading needs to be fully async.
                from lollms_server.utils.dependency_manager import ensure_dependencies_sync # Assuming a sync version exists or is created
                deps_ok = ensure_dependencies_sync(p_config.dependencies, source_name=f"Personality '{p_config.name}'")
                if not deps_ok:
                    err_msg = f"Dependency installation failed for personality '{p_config.name}'. It may not function correctly."
                    logger.error(err_msg)
                    self._dependency_errors[str(path)] = err_msg
                    # Decide: prevent loading or just warn? Let's warn and load.

            # --- Script Loading ---
            if p_config.script_path:
                script_file = (path / p_config.script_path).resolve()
                if script_file.exists() and script_file.is_file() and script_file.suffix == ".py":
                    logger.info(f"Found script for personality '{p_config.name}': {script_file}")
                    add_path_to_sys_path(script_file.parent)
                    module, error = safe_load_module(script_file) # safe_load_module is sync
                    if module:
                        script_module = module
                        logger.info(f"Successfully loaded script module for '{p_config.name}'.")
                        if not hasattr(script_module, 'run_workflow'):
                            logger.warning(f"Script for personality '{p_config.name}' loaded but does not contain a 'run_workflow' function.")
                    else:
                        err_msg = f"Failed to load script '{script_file}' for personality '{p_config.name}': {error}"
                        logger.error(err_msg)
                        self._load_errors[str(path)] = self._load_errors.get(str(path), "") + f"; Script load error: {error}"
                        script_module = None
                else:
                    logger.warning(f"Script path '{p_config.script_path}' defined for personality '{p_config.name}' but file not found or invalid: {script_file}")

            # --- Create Personality Object ---
            personality = Personality(
                config=p_config,
                path=path,
                assets_path=assets_path,
                script_module=script_module
            )

            if personality.name in self._personalities:
                logger.warning(f"Duplicate personality name '{personality.name}' found. Overwriting with personality from {path}.")
            self._personalities[personality.name] = personality
            logger.info(f"Successfully loaded personality: '{personality.name}' from {path}")

        except yaml.YAMLError as e:
            err_msg = f"Error parsing YAML in {config_file}: {e}"
            logger.error(err_msg)
            self._load_errors[str(path)] = err_msg
        except ValueError as e: # Catches Pydantic validation errors and empty file errors
            err_msg = f"Error processing personality config at {path}: {e}"
            logger.error(err_msg)
            self._load_errors[str(path)] = err_msg
        except Exception as e:
            err_msg = f"Unexpected error loading personality from {path}: {e}"
            logger.error(err_msg, exc_info=True)
            self._load_errors[str(path)] = err_msg


    # --- list_personalities remains the same, but might expose more fields now ---
    def list_personalities(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of loaded personalities and their basic info."""
        info = {}
        for name, p in self._personalities.items():
            # Expose fields based on the new config model
            info[name] = {
                "name": p.name,
                "author": p.config.author,
                "version": p.config.version,
                # Use the specific description field from the config
                "description": p.config.personality_description,
                "category": p.config.category,
                "language": p.config.language,
                "tags": p.config.tags,
                "icon": p.config.icon,
                "is_scripted": p.is_scripted,
                "path": str(p.path),
            }
        return info

    def get_personality(self, name: str) -> Optional[Personality]:
        """Gets a loaded personality by its name."""
        return self._personalities.get(name)