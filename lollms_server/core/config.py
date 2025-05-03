# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# File: lollms_server/core/config.py
# Creation Date: 2025-05-01
# Description: Manages loading and accessing the main server configuration using ConfigGuard.

import sys
import typing # Required for type hinting the schema dict
from pathlib import Path
import os
import secrets
import subprocess # Added for wizard trigger

try:
    from configguard import ConfigGuard, ConfigSection
    from configguard.exceptions import ConfigGuardError, SettingNotFoundError # Import specific error
    from configguard.handlers import JsonHandler, YamlHandler, TomlHandler, SqliteHandler
except ImportError:
    print("FATAL ERROR: ConfigGuard library not found.")
    print("Please ensure core dependencies were installed correctly (run install script).")
    sys.exit(1)

# Use ascii_colors for logging/output if available
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)

logger = logging.getLogger(__name__)

# --- Main Configuration Schema Definition ---
MAIN_CONFIG_VERSION = "0.3.0" # Current schema version

# Schema definition dictionary - WITHOUT the top-level __version__ key
MAIN_SCHEMA: typing.Dict[str, typing.Any] = {
    "server": {
        "type": "section", "help": "Core server settings.",
        "schema": {
            "host": { "type": "str", "default": "0.0.0.0", "help": "Host address" },
            "port": { "type": "int", "default": 9601, "min_val": 1, "max_val": 65535, "help": "Port number" },
            "allowed_origins": {
                "type": "list", "default": [
                    "http://localhost", "http://localhost:8000", "http://localhost:5173",
                    "http://127.0.0.1", "http://127.0.0.1:8000", "http://127.0.0.1:5173", "null"
                ],
                "help": "List of allowed CORS origins."
            }
        }
    },
    "paths": {
        "type": "section", "help": "Directory paths for server components.",
        "schema": {
            "config_base_dir": { "type": "str", "default": "lollms_configs", "help": "Base directory for configuration files (main, bindings etc.). Relative to server root if not absolute." },
            "instance_bindings_folder": { "type": "str", "default": "bindings", "help": "Subfolder within config_base_dir for binding instance configs." },
            "personalities_folder": { "type": "str", "default": "personal/personalities", "help": "Folder for your custom personalities (relative to server root)." },
            "bindings_folder": { "type": "str", "default": "personal/bindings", "help": "Folder for your custom binding types (relative to server root)." },
            "functions_folder": { "type": "str", "default": "personal/functions", "help": "Folder for your custom functions (relative to server root)." },
            "models_folder": { "type": "str", "default": "models", "help": "Base folder for model files (relative to server root)." },
            "example_personalities_folder": { "type": "str", "default": "zoos/personalities", "nullable": True, "help": "Path to built-in example personalities (relative to server root)." },
            "example_bindings_folder": { "type": "str", "default": "zoos/bindings", "nullable": True, "help": "Path to built-in example binding types (relative to server root)." },
            "example_functions_folder": { "type": "str", "default": "zoos/functions", "nullable": True, "help": "Path to built-in example functions (relative to server root)." }
        }
    },
    "security": {
        "type": "section", "help": "Security settings.",
        "schema": {
            "allowed_api_keys": { "type": "list", "default": [], "help": "List of allowed API keys." },
            "encryption_key": { "type": "str", "nullable": True, "default": None, "secret": True, "help": "Fernet key for encrypting binding configs (optional)." }
        }
    },
    "defaults": {
        "type": "section", "help": "Default binding instance names and models.",
        "schema": {
            "ttt_binding": { "type": "str", "nullable": True, "default": None, "help": "Default Text-to-Text binding instance name." },
            "tti_binding": { "type": "str", "nullable": True, "default": None, "help": "Default Text-to-Image binding instance name." },
            "tts_binding": { "type": "str", "nullable": True, "default": None, "help": "Default Text-to-Speech binding instance name." },
            "stt_binding": { "type": "str", "nullable": True, "default": None, "help": "Default Speech-to-Text binding instance name." },
            "ttv_binding": { "type": "str", "nullable": True, "default": None, "help": "Default Text-to-Video binding instance name." },
            "ttm_binding": { "type": "str", "nullable": True, "default": None, "help": "Default Text-to-Music binding instance name." },
            "ttt_model": { "type": "str", "nullable": True, "default": None, "help": "Default model for the TTT binding." },
            "tti_model": { "type": "str", "nullable": True, "default": None, "help": "Default model for the TTI binding." },
            "tts_model": { "type": "str", "nullable": True, "default": None, "help": "Default model for the TTS binding." },
            "stt_model": { "type": "str", "nullable": True, "default": None, "help": "Default model for the STT binding." },
            "ttv_model": { "type": "str", "nullable": True, "default": None, "help": "Default model for the TTV binding." },
            "ttm_model": { "type": "str", "nullable": True, "default": None, "help": "Default model for the TTM binding." },
            "default_context_size": { "type": "int", "default": 4096, "min_val": 64, "help": "Default context window size." },
            "default_max_output_tokens": { "type": "int", "default": 1024, "min_val": 1, "help": "Default max generation tokens." }
        }
    },
    "bindings_map": {
        "type": "section",
        "help": "Maps binding instance names to their type names (e.g., my_ollama = ollama_binding).",
        "schema": {} # Dynamically populated by user/wizard/tools
    },
     "resource_manager": {
        "type": "section", "help": "Resource management settings.",
        "schema": {
            "gpu_strategy": { "type": "str", "default": "semaphore", "options": ["semaphore", "simple_lock", "none"], "help": "GPU resource locking strategy." },
            "gpu_limit": { "type": "int", "default": 1, "min_val": 1, "help": "Max concurrent GPU tasks for 'semaphore'." },
            "queue_timeout": { "type": "int", "default": 120, "min_val": 1, "help": "Queue timeout in seconds." }
        }
    },
    "webui": {
        "type": "section", "help": "Built-in Web UI settings.",
        "schema": {
            "enable_ui": { "type": "bool", "default": False, "help": "Enable serving the Web UI." }
        }
    },
    "logging": {
        "type": "section", "help": "Logging configuration.",
        "schema": {
            "log_level": { "type": "str", "default": "INFO", "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "help": "Logging verbosity level." },
            "level": { "type": "int", "default": 20, "help": "Logging level numeric value (e.g., DEBUG=10, INFO=20)." }
        }
    },
    "personalities_config": {
        "type": "section",
        "help": "Overrides for personalities (e.g., { 'python_builder_executor': { 'enabled': False } }).",
        "nullable": True,
        "default": {},
        "schema": {} # Schema allows any key (personality name), value will be dict
    }
}

# --- Global Config Instance ---
global _config, _config_path, _server_root
_config: typing.Optional[ConfigGuard] = None
_config_path: typing.Optional[Path] = None
_server_root: typing.Optional[Path] = None # Store server root

# --- Handler Map ---
_handler_map = {
    ".yaml": YamlHandler, ".yml": YamlHandler,
    ".json": JsonHandler, ".toml": TomlHandler,
    ".db": SqliteHandler, ".sqlite": SqliteHandler, ".sqlite3": SqliteHandler
}

def _find_main_config_file(config_base_dir: Path) -> typing.Optional[Path]:
    """Finds the first supported config file in the base directory."""
    supported_extensions = list(_handler_map.keys())
    # Prioritize common formats
    preferred_order = [".yaml", ".yml", ".toml", ".json", ".db", ".sqlite", ".sqlite3"]
    for ext in preferred_order:
        if ext in supported_extensions:
            potential_path = config_base_dir / f"main_config{ext}"
            if potential_path.is_file():
                logger.debug(f"Found main config file: {potential_path}")
                return potential_path
    # Fallback: check for any supported extension
    for ext in supported_extensions:
         potential_path = config_base_dir / f"main_config{ext}"
         if potential_path.is_file():
             logger.debug(f"Found main config file (fallback): {potential_path}")
             return potential_path
    logger.debug(f"No main_config.* file found in {config_base_dir}")
    return None

def initialize_config(server_root_dir: Path, trigger_wizard_if_missing: bool = True) -> ConfigGuard:
    """
    Initializes and loads the main configuration using ConfigGuard.
    Determines config path, handles missing files (optional wizard trigger),
    and resolves relative paths within the config.

    Args:
        server_root_dir: The absolute path to the lollms_server project root.
        trigger_wizard_if_missing: If True, run configuration_wizard.py if no config is found.

    Returns:
        The loaded ConfigGuard instance.

    Raises:
        SystemExit: If configuration fails critically (e.g., wizard aborted, handler deps missing).
    """
    global _config, _config_path, _server_root
    if _config:
        return _config # Already initialized

    _server_root = server_root_dir.resolve() # Ensure it's absolute
    logger.info(f"Server Root Directory set to: {_server_root}")

    # 1. Determine Base Config Directory
    # Use environment variable first, then default relative to server root
    env_config_dir = os.environ.get("LOLLMS_CONFIG_DIR")
    if env_config_dir:
        config_base_dir = Path(env_config_dir).resolve()
        logger.info(f"Using configuration directory from LOLLMS_CONFIG_DIR: {config_base_dir}")
    else:
        # Default relative to server root, using schema default for the name
        default_relative_path_name = MAIN_SCHEMA.get("paths", {}).get("schema", {}).get("config_base_dir", {}).get("default", "lollms_configs")
        config_base_dir = (_server_root / default_relative_path_name).resolve()
        logger.info(f"Using default configuration directory: {config_base_dir}")

    # Ensure base directory exists
    try:
        config_base_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         logger.critical(f"CRITICAL: Could not create configuration directory {config_base_dir}: {e}")
         sys.exit(1) # Exit directly on critical fs error

    # 2. Find the Main Config File
    _config_path = _find_main_config_file(config_base_dir)

    # 3. Handle Missing Config File
    if not _config_path:
        ASCIIColors.warning(f"No main configuration file (main_config.*) found in {config_base_dir}.")
        if trigger_wizard_if_missing:
            ASCIIColors.info("Attempting to launch configuration wizard...")
            wizard_script = _server_root / "configuration_wizard.py"
            if not wizard_script.exists():
                ASCIIColors.error(f"Configuration wizard script not found at {wizard_script}")
                ASCIIColors.error("Cannot proceed without configuration. Please ensure the wizard script exists or create a config file manually.")
                sys.exit(1) # Exit if wizard is missing and needed

            try:
                # Run the wizard as a separate process
                python_exe = sys.executable or "python" # Use current executable or guess
                logger.info(f"Running wizard: {python_exe} {wizard_script}")
                process = subprocess.run([python_exe, str(wizard_script)], check=False, cwd=_server_root) # Run from server root
                if process.returncode != 0:
                     ASCIIColors.error(f"Configuration wizard exited with error code {process.returncode}.")
                     sys.exit(1) # Exit if wizard failed

                # Wizard finished, try finding the config file again
                _config_path = _find_main_config_file(config_base_dir)
                if not _config_path:
                     ASCIIColors.error("Configuration wizard completed, but still couldn't find a main config file (main_config.*).")
                     ASCIIColors.error(f"Please check the directory: {config_base_dir}")
                     sys.exit(1) # Exit if still not found after wizard
                ASCIIColors.success(f"Configuration wizard completed. Found config file: {_config_path}")

            except ImportError as ie:
                 ASCIIColors.error(f"Missing dependency required by configuration wizard: {ie}")
                 ASCIIColors.error("Please run the installation script (install.sh/install.bat) again.")
                 sys.exit(1)
            except FileNotFoundError:
                 ASCIIColors.error(f"Could not find Python executable '{python_exe}' to run the wizard.")
                 sys.exit(1)
            except Exception as e:
                 logger.error(f"Error running configuration wizard: {e}", exc_info=True)
                 trace_exception(e)
                 ASCIIColors.error(f"An unexpected error occurred while running the configuration wizard: {e}")
                 sys.exit(1)
        else:
            ASCIIColors.error(f"No main configuration file found in {config_base_dir} and wizard trigger is disabled.")
            ASCIIColors.error("Please create a configuration file (e.g., main_config.yaml) or enable the wizard trigger.")
            sys.exit(1) # Exit as configuration is mandatory

    # 4. Initialize ConfigGuard
    encryption_key_str = os.environ.get("LOLLMS_ENCRYPTION_KEY")
    encryption_key = encryption_key_str.encode('utf-8') if encryption_key_str else None

    try:
        logger.info(f"Loading main configuration from: {_config_path}")
        _config = ConfigGuard(
            schema=MAIN_SCHEMA,             # Pass schema dict WITHOUT __version__
            config_path=_config_path,       # Path to the config file
            encryption_key=encryption_key,  # Optional encryption key
            autosave=False,                 # Disable autosave for server
            instance_version=MAIN_CONFIG_VERSION # Pass expected version here
        )
        _config.load() # Load data, handles migration based on version comparison

        # Check for encryption key in loaded config if not from env
        # Needs careful access because section might not exist yet during initial load/migration
        if not encryption_key and hasattr(_config, 'security') and getattr(_config.security, "encryption_key", None):
             config_key_str = getattr(_config.security, "encryption_key")
             if config_key_str and isinstance(config_key_str, str):
                 try:
                     encryption_key = config_key_str.encode('utf-8')
                     # Re-apply the key to the ConfigGuard instance if found in file
                     # Needed because the initial load might not have used it if env var wasn't set
                     _config.set_encryption_key(encryption_key)
                     logger.info("Loaded encryption key from main config file and applied to instance.")
                 except Exception as key_err:
                     logger.error(f"Failed to encode encryption key from config: {key_err}")
             else:
                  logger.debug("Encryption key found in config but was null or not a string.")
        elif encryption_key:
            logger.info("Using encryption key from environment variable.")


    except ConfigGuardError as cge:
         logger.critical(f"ConfigGuard Error loading main config: {cge}", exc_info=True)
         trace_exception(cge)
         ASCIIColors.error(f"CRITICAL CONFIGURATION ERROR: {cge}")
         ASCIIColors.error(f"Please check your configuration file: {_config_path}")
         ASCIIColors.error("You might need to backup the file and run the wizard again or fix it manually.")
         sys.exit(1)
    except ImportError as e:
        logger.critical(f"Missing dependency for config handler ({_config_path.suffix}): {e}", exc_info=True)
        ASCIIColors.error(f"Missing dependency for {_config_path.suffix} files: {e}")
        ASCIIColors.error("Please install required extras (e.g., 'pip install lollms_server[yaml,toml]').")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error loading main configuration: {e}", exc_info=True)
        trace_exception(e)
        ASCIIColors.error(f"CRITICAL: Failed to load configuration from {_config_path}")
        sys.exit(1)

    # 5. Resolve Paths
    if hasattr(_config, 'paths'):
         _resolve_paths(_config.paths, _server_root)
    else:
         logger.critical("Configuration loaded, but 'paths' section is missing. Cannot resolve paths.")
         sys.exit(1) # Paths section is critical

    # 6. Update derived paths based on config_base_dir
    if hasattr(_config, 'paths'):
        # Ensure config_base_dir itself is absolute before using it
        base_dir_str = getattr(_config.paths, "config_base_dir", None)
        if not base_dir_str or not Path(base_dir_str).is_absolute():
            logger.critical(f"Internal error: config_base_dir '{base_dir_str}' was not resolved to an absolute path.")
            sys.exit(1)
        resolved_config_base_dir = Path(base_dir_str)
        _update_derived_paths(_config.paths, resolved_config_base_dir)
    else:
        # This case should have been caught by check in step 5
        logger.critical("Cannot update derived paths: 'paths' section missing.")
        sys.exit(1)

    # 7. Ensure Directories Exist
    if hasattr(_config, 'paths'):
         _ensure_directories(_config.paths)
    else:
        # This case should have been caught by check in step 5
        logger.critical("Cannot ensure directories: 'paths' section missing.")
        sys.exit(1)

    logger.info(f"Main configuration loaded successfully from {_config_path}")
    return _config

def get_config() -> ConfigGuard:
    """Returns the loaded main ConfigGuard instance."""
    global _config, _config_path, _server_root
    if _config is None:
        logger.critical("Configuration accessed before initialization!")
        raise RuntimeError("Configuration has not been initialized. Call initialize_config first.")
    return _config

def get_config_path() -> typing.Optional[Path]:
    """Returns the absolute path of the loaded main config file, if any."""
    return _config_path

def get_server_root() -> Path:
    """Returns the detected server root directory."""
    global _config, _config_path, _server_root
    if _server_root is None:
        try:
             # Try to determine from __file__ of this module
             this_file = Path(__file__).resolve()
             # Assume structure is lollms_server/lollms_server/core/config.py
             root = this_file.parent.parent.parent
             if not (root / "lollms_server").is_dir(): # Basic sanity check
                  raise RuntimeError("Could not verify server root structure.")
             _server_root = root
        except Exception:
             # Fallback to current working directory if __file__ fails
             _server_root = Path(".").resolve()
             logger.warning(f"Could not reliably determine server root from __file__. Using CWD: {_server_root}")
    return _server_root

def get_config_base_dir() -> Path:
    """Returns the resolved base directory for configurations."""
    config = get_config() # Ensures config is initialized
    # Should be absolute after initialization
    base_dir_str = getattr(config.paths, "config_base_dir", None)
    if not base_dir_str or not Path(base_dir_str).is_absolute():
         # This should ideally not happen if initialization worked
         logger.error("Config base dir attribute missing or not absolute after init. Re-resolving.")
         default_relative_path = MAIN_SCHEMA.get("paths", {}).get("schema", {}).get("config_base_dir", {}).get("default", "lollms_configs")
         root = get_server_root()
         return (root / default_relative_path).resolve()
    return Path(base_dir_str)

def get_binding_instances_config_path() -> Path:
    """Returns the resolved path to the binding instances configuration directory."""
    config = get_config() # Ensures config is initialized
    # Both paths should be absolute after initialization
    config_base = Path(getattr(config.paths, "config_base_dir"))
    instance_folder_abs = Path(getattr(config.paths, "instance_bindings_folder"))
    if not instance_folder_abs.is_absolute():
         # This indicates an error in _update_derived_paths
         logger.error(f"Instance bindings folder was not resolved correctly: {instance_folder_abs}")
         instance_folder_rel = MAIN_SCHEMA.get("paths",{}).get("schema",{}).get("instance_bindings_folder",{}).get("default","bindings")
         return (config_base / instance_folder_rel).resolve() # Fallback resolution
    return instance_folder_abs

def _resolve_paths(paths_section: ConfigSection, server_root_dir: Path):
    """Resolves relative paths in the paths section relative to the server root."""
    if not paths_section:
        logger.error("Cannot resolve paths: 'paths' section object is missing.")
        return

    # Iterate through the schema definition of the PathsConfig section
    try:
        # Access schema definition via ConfigSection attribute
        paths_schema = paths_section._schema_definition
    except AttributeError:
        logger.error("Could not get schema definition for paths section.")
        return

    logger.debug(f"Resolving paths relative to server root: {server_root_dir}")
    for key in paths_schema.keys():
        # Skip keys that aren't path-like or internal
        if key.startswith('__') or key == "instance_bindings_folder": # instance_bindings_folder resolved later
            continue

        try:
            path_str = getattr(paths_section, key, None)
            # Resolve only if it's a non-empty string
            if path_str and isinstance(path_str, str):
                path_obj = Path(path_str)
                if not path_obj.is_absolute():
                    resolved_path = (server_root_dir / path_obj).resolve()
                    # Update the value back into the ConfigSection object
                    setattr(paths_section, key, str(resolved_path))
                    logger.debug(f"Resolved path '{key}': {resolved_path}")
                else:
                    logger.debug(f"Path '{key}' is already absolute: {path_str}")
            elif path_str is None and paths_schema[key].get("nullable") is False:
                 logger.warning(f"Path key '{key}' is null but schema expects non-nullable value.")
            elif path_str is not None:
                 logger.warning(f"Path key '{key}' has non-string value '{path_str}' ({type(path_str)}). Skipping resolution.")

        except AttributeError:
             # This shouldn't happen if iterating schema keys, but check anyway
             logger.debug(f"Path key '{key}' not found in paths section (should be present based on schema).")
        except Exception as e:
             logger.error(f"Error resolving path '{key}': {e}", exc_info=True)


def _update_derived_paths(paths_section: ConfigSection, config_base_dir: Path):
    """Updates paths that depend on the config_base_dir (e.g., instance_bindings_folder)."""
    if not paths_section:
        logger.error("Cannot update derived paths: 'paths' section object is missing.")
        return
    if not config_base_dir.is_absolute() or not config_base_dir.exists():
        logger.error(f"Cannot update derived paths: Resolved config_base_dir '{config_base_dir}' is invalid.")
        return

    logger.debug(f"Updating derived paths relative to config base dir: {config_base_dir}")
    key = "instance_bindings_folder"
    try:
         instance_folder_name_str = getattr(paths_section, key, None)
         if instance_folder_name_str and isinstance(instance_folder_name_str, str):
             instance_folder_path = Path(instance_folder_name_str)
             # Only resolve if it's NOT already absolute
             if not instance_folder_path.is_absolute():
                  full_instance_path = (config_base_dir / instance_folder_path).resolve()
                  setattr(paths_section, key, str(full_instance_path))
                  logger.debug(f"Resolved instance bindings folder: {full_instance_path}")
             else:
                  logger.debug(f"Instance bindings folder is already absolute: {instance_folder_path}")
         elif instance_folder_name_str is None:
              logger.warning(f"Path key '{key}' is missing or null.")
         else:
              logger.warning(f"Path key '{key}' has non-string value '{instance_folder_name_str}' ({type(instance_folder_name_str)}). Skipping update.")
    except AttributeError:
         logger.warning(f"Path key '{key}' not found during derived path update.")
    except Exception as e:
         logger.error(f"Error updating derived path '{key}': {e}", exc_info=True)


def _ensure_directories(paths_section: ConfigSection):
    """Creates essential directories defined in the config if they don't exist."""
    if not paths_section:
        logger.error("Cannot ensure directories: 'paths' section object is missing.")
        return

    try:
        # Access schema definition via ConfigSection attribute
        paths_schema = paths_section._schema_definition
        if not paths_schema:
             logger.warning("Could not retrieve schema keys for paths section.")
             return
    except AttributeError:
        logger.error("Could not retrieve schema definition for paths section.")
        return

    paths_to_check: typing.List[Path] = []
    for key in paths_schema.keys():
        if key.startswith('__'): continue
        try:
            path_str = getattr(paths_section, key, None)
            if path_str and isinstance(path_str, str):
                path_obj = Path(path_str)
                # Assume it's a directory path if key ends with '_folder' or '_dir'
                # or if it's one of the specific known folder keys
                if key.endswith(('_folder', '_dir')) or \
                   key in ["config_base_dir", "instance_bindings_folder", "personalities_folder",
                            "bindings_folder", "functions_folder", "models_folder",
                            "example_personalities_folder", "example_bindings_folder",
                            "example_functions_folder"]:
                     # Make sure it's absolute before adding
                     if path_obj.is_absolute():
                          paths_to_check.append(path_obj)
                     else:
                          logger.warning(f"Skipping directory check for non-absolute path '{key}': {path_obj}")
        except Exception as e:
             logger.warning(f"Could not access or process path '{key}' for directory check: {e}")

    logger.info("Ensuring essential configuration directories exist...")
    created_count = 0
    checked_count = 0
    for folder_path in paths_to_check:
        checked_count += 1
        if folder_path: # Ensure Path object is not None/empty and absolute
            try:
                if not folder_path.exists():
                     folder_path.mkdir(parents=True, exist_ok=True)
                     logger.info(f"Created directory: {folder_path}")
                     created_count += 1
                elif not folder_path.is_dir():
                     logger.warning(f"Path exists but is not a directory: {folder_path}")
                else:
                     logger.debug(f"Directory already exists: {folder_path}")
            except Exception as e:
                logger.error(f"Failed to create or verify directory {folder_path}: {e}")
        else:
            logger.debug("Skipping None path in directory check.")
    logger.info(f"Directory check complete. Checked: {checked_count}, Created: {created_count}.")

    # Ensure model subdirectories explicitly
    try:
        model_base_str = getattr(paths_section, "models_folder", None)
        if model_base_str and isinstance(model_base_str, str):
            model_base = Path(model_base_str)
            if model_base.is_absolute() and model_base.is_dir(): # Check if it's valid absolute dir
                logger.info(f"Ensuring model type subdirectories exist in: {model_base}")
                # Define required subfolders
                model_subfolders = ["ttt", "tti", "ttv", "ttm", "tts", "stt", "audio2audio", "i2i", "gguf", "diffusers_models"]
                for subfolder in model_subfolders:
                    try:
                        subdir_path = model_base / subfolder
                        if not subdir_path.exists():
                             subdir_path.mkdir(parents=True, exist_ok=True)
                             logger.info(f" -> Created model subdir: {subdir_path}")
                        elif not subdir_path.is_dir():
                             logger.warning(f" -> Path exists but is not a directory: {subdir_path}")
                        else:
                             logger.debug(f" -> Model subdir exists: {subdir_path}")
                    except Exception as e:
                        logger.error(f"Failed to create model sub-directory {model_base / subfolder}: {e}")
            elif not model_base.is_absolute():
                logger.warning(f"Configured models_folder path is not absolute: {model_base}. Cannot create subdirectories reliably.")
            elif not model_base.is_dir():
                 logger.warning(f"Configured models_folder path exists but is not a directory: {model_base}. Cannot create subdirectories.")
        elif model_base_str is None:
             logger.info("No models_folder configured, skipping sub-directory check.")
        else:
             logger.warning(f"Configured models_folder path is not a string ({type(model_base_str)}). Skipping sub-directory check.")
    except Exception as e:
        logger.error(f"Error ensuring model subdirectories: {e}", exc_info=True)


def get_encryption_key() -> typing.Optional[bytes]:
    """Gets the encryption key from the loaded config or environment."""
    # Try environment first
    env_key = os.environ.get("LOLLMS_ENCRYPTION_KEY")
    if env_key:
        try:
            # Basic length check for sanity
            if len(env_key) > 10:
                 return env_key.encode('utf-8')
            else:
                 logger.warning("LOLLMS_ENCRYPTION_KEY from environment seems too short. Ignoring.")
                 return None
        except Exception as e:
             logger.error(f"Error encoding LOLLMS_ENCRYPTION_KEY from environment: {e}")
             return None

    # Try loaded config (safe access)
    if _config and hasattr(_config, 'security'):
        try:
            # Use getattr for safe access in case 'security' section or 'encryption_key' setting is missing
            key_str = getattr(_config.security, "encryption_key", None)
            if key_str and isinstance(key_str, str):
                 if len(key_str) > 10: # Basic length check
                      return key_str.encode('utf-8')
                 else:
                      logger.warning("Encryption key found in config but seems too short. Ignoring.")
                      return None
            elif key_str is not None:
                 logger.warning(f"Encryption key in config is not a string (type: {type(key_str)}). Ignoring.")
        except AttributeError:
             logger.debug("Security section or encryption_key not found in config.")
        except Exception as e:
             logger.error(f"Error encoding encryption key from config: {e}")

    return None