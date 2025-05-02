# encoding:utf-8
# Project: lollms_server
# File: install_core.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Core installation script: bootstraps pipmaster, installs deps,
#              offers optional installs & CLI binding management.

import sys
import os
import subprocess
import venv
import shutil
import json
import yaml # Needed for reading binding cards
import toml # Needed for _cli_edit_instance helper
from pathlib import Path
import platform
from typing import List, Optional, Tuple, Dict, Any, Union

# --- Minimal Imports & Constants (Bootstrap Phase) ---
import logging as pylogging

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MIN_PYTHON_VERSION = (3, 9)
# Default name for the user data/config directory, relative to project root
DEFAULT_CONFIG_BASE_DIR_NAME = "lollms_configs"

# --- Bootstrap Logging Setup ---
# Use standard logging ONLY until ascii_colors is confirmed installed
pylogging.basicConfig(level=pylogging.INFO, format='BOOTSTRAP %(levelname)s: %(message)s')
bootstrap_logger = pylogging.getLogger("lollms_installer_bootstrap")

# --- Bootstrap Helper Functions ---

def print_step_bootstrap(message: str):
    """Prints a formatted step message during bootstrap."""
    print(f"\n--- {message} ---")

def exit_with_error_bootstrap(message: str, exit_code: int = 1):
    """Logs an error and exits during bootstrap."""
    bootstrap_logger.error(message)
    if platform.system() == "Windows":
        input("Press Enter to exit...")
    sys.exit(exit_code)

def check_python_version():
    """Checks if the current Python version meets the minimum requirement."""
    print_step_bootstrap(f"Checking Python Version (>= {'.'.join(map(str, MIN_PYTHON_VERSION))})")
    current_version = sys.version_info
    if current_version < MIN_PYTHON_VERSION:
        exit_with_error_bootstrap(
            f"Python {'.'.join(map(str, MIN_PYTHON_VERSION))}+ required. "
            f"Found {platform.python_version()}."
        )
    bootstrap_logger.info(f"Success: Python version {platform.python_version()} compatible.")

def create_virtual_environment():
    """Creates or verifies the virtual environment."""
    print_step_bootstrap(f"Creating/Verifying Virtual Environment in '{VENV_DIR}'")
    if VENV_DIR.exists() and (VENV_DIR / "pyvenv.cfg").exists():
        bootstrap_logger.info(f"Venv '{VENV_DIR}' already exists. Using existing.")
        return True
    try:
        bootstrap_logger.info("Creating venv...")
        venv.create(VENV_DIR, with_pip=True)
        bootstrap_logger.info("Success: Virtual environment created.")
        return True
    except Exception as e:
        bootstrap_logger.exception(f"Failed create venv: {e}")
        exit_with_error_bootstrap(f"Venv creation failed: {e}")
        # return False # Unreachable due to exit

def get_executable_path(env_dir: Path, executable_name: str) -> str:
    """Gets the platform-specific path to an executable in the venv."""
    if platform.system() == "Windows":
        script_dir = env_dir / "Scripts"
        exe_path = script_dir / f"{executable_name}.exe"
        # Check for python.exe specifically
        if executable_name == "python":
            py_path = script_dir / "python.exe"
            if py_path.exists():
                return str(py_path.resolve())
    else: # Linux, macOS
        script_dir = env_dir / "bin"
        exe_path = script_dir / executable_name
        if executable_name == "python":
            py_path = script_dir / "python"
            if py_path.exists():
                return str(py_path.resolve())

    if exe_path.exists():
        return str(exe_path.resolve())
    else:
        # Fallback for pip/pip3 variations on non-Windows
        if executable_name == "pip" and platform.system() != "Windows":
            pip3_path = script_dir / "pip3"
            if pip3_path.exists():
                return str(pip3_path.resolve())
        elif executable_name == "pip3" and platform.system() != "Windows":
             pip_path = script_dir / "pip"
             if pip_path.exists():
                 return str(pip_path.resolve())

        # If specific path not found, warn and return base name, relying on PATH
        bootstrap_logger.warning(
            f"Could not find specific path for {executable_name} in venv "
            f"({script_dir}). Relying on PATH after activation or default name."
        )
        return executable_name

def _get_venv_pip_executable(venv_dir: Path) -> Optional[str]:
    """Finds the pip executable within the venv, returning its absolute path."""
    pip_exe = get_executable_path(venv_dir, "pip")

    # If get_executable_path returned just the name (fallback), try harder
    if not Path(pip_exe).is_absolute() and pip_exe in ["pip", "pip3"]:
        venv_python_exe = get_executable_path(venv_dir, "python")
        if Path(venv_python_exe).is_absolute():
            # Try common names relative to the Python executable
            script_dir = Path(venv_python_exe).parent
            pip_names_to_try = ["pip", "pip3"] if platform.system() != "Windows" else ["pip"]
            for pip_name in pip_names_to_try:
                potential_pip_path = script_dir / pip_name
                if potential_pip_path.exists():
                    bootstrap_logger.info(f"Found venv pip via relative path: {potential_pip_path}")
                    return str(potential_pip_path.resolve())
        # If python path wasn't absolute or relative search failed
        bootstrap_logger.error("Could not reliably determine venv pip executable path.")
        return None

    elif Path(pip_exe).exists():
        return pip_exe # Already found absolute path
    else: # Path was likely absolute but didn't exist?
        bootstrap_logger.error(f"Calculated pip executable path '{pip_exe}' does not exist.")
        return None

def _run_subprocess_command(command: List[str], description: str, timeout: int = 300) -> bool:
    """Runs a subprocess command, logs output, and returns success status."""
    # --- Conditional import of ASCIIColors ---
    try:
         from ascii_colors import ASCIIColors
         colors_available = True
    except ImportError:
         colors_available = False
         print(f"Running: {' '.join(command)}") # Use standard print if colors unavailable
    # -----------------------------------------

    try:
        if colors_available: # Use ASCIIColors only if available
             ASCIIColors.magenta(f"Running: {' '.join(command)}")

        process = subprocess.run(
            command,
            check=True, # Raises CalledProcessError on non-zero exit
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace', # Added to handle potential decoding errors
            timeout=timeout
        )
        bootstrap_logger.info(f"{description} Output:\n{process.stdout}")
        if process.stderr:
            bootstrap_logger.warning(f"{description} Stderr:\n{process.stderr}")
        bootstrap_logger.info(f"Success: {description} completed.")
        return True
    except subprocess.CalledProcessError as e:
        bootstrap_logger.error(f"Failed {description}. Command: '{' '.join(command)}'. Error code: {e.returncode}")
        bootstrap_logger.error(f"--- {description} Output ---")
        bootstrap_logger.error(e.stdout or "(No stdout captured)")
        bootstrap_logger.error(f"--- {description} Stderr ---")
        bootstrap_logger.error(e.stderr or "(No stderr captured)")
        bootstrap_logger.error(f"--- End {description} Logs ---")
        return False
    except subprocess.TimeoutExpired:
         bootstrap_logger.error(f"Timeout ({timeout}s) occurred during {description}. Command: {' '.join(command)}")
         return False
    except FileNotFoundError:
         exit_with_error_bootstrap(f"Command '{command[0]}' not found during {description}.", 1)
         # return False # Unreachable
    except Exception as e:
         bootstrap_logger.exception(f"An unexpected error occurred during {description}: {e}")
         return False

def _bootstrap_pipmaster(venv_pip_exe: str) -> bool:
    """Uses subprocess to install pipmaster into the venv."""
    print_step_bootstrap("Ensuring pipmaster is installed")
    if not venv_pip_exe:
        exit_with_error_bootstrap("Cannot bootstrap pipmaster: venv pip executable path not found.", 1)
        # return False # Unreachable

    bootstrap_logger.info("Checking if pipmaster is installed in venv...")
    venv_python_exe = get_executable_path(VENV_DIR, "python")
    check_cmd = [venv_python_exe, "-m", "pip", "show", "pipmaster"]

    try:
        result = subprocess.run(check_cmd, check=False, capture_output=True, text=True, encoding='utf-8', timeout=30)
        if result.returncode == 0 and "Name: pipmaster" in result.stdout:
            bootstrap_logger.info("pipmaster already installed.")
            return True
    except Exception as e:
         bootstrap_logger.warning(f"Could not run pipmaster check command: {e}. Assuming not installed.")

    # If check failed or package not found, attempt installation
    bootstrap_logger.info("pipmaster not found or check failed. Attempting installation...")
    install_cmd = [venv_pip_exe, "install", "--upgrade", "pipmaster>=0.7.2"]
    return _run_subprocess_command(install_cmd, "pipmaster installation", timeout=300)

# --- Main Execution Flow ---

if __name__ == "__main__":
    # ============================
    # ===== BOOTSTRAP PHASE ======
    # ============================
    check_python_version()
    if not create_virtual_environment():
        sys.exit(1) # Error printed by function

    # Get critical Venv Executables
    VENV_PYTHON_EXE = get_executable_path(VENV_DIR, "python")
    if not Path(VENV_PYTHON_EXE).is_absolute():
        exit_with_error_bootstrap(
            f"Could not locate absolute python executable in venv '{VENV_DIR}'. "
            "Activation might be needed or venv creation failed partially.", 1
        )

    VENV_PIP_EXE = _get_venv_pip_executable(VENV_DIR)
    if not VENV_PIP_EXE:
         exit_with_error_bootstrap("Failed to locate pip executable in venv.", 1)

    # Bootstrap pipmaster
    if not _bootstrap_pipmaster(VENV_PIP_EXE):
        exit_with_error_bootstrap("Failed to install pipmaster. Cannot proceed.", 1)

    # ======================================
    # ===== POST-BOOTSTRAP SETUP PHASE =====
    # ======================================
    # Add Venv Site-Packages to Path & Import Core Libs
    venv_lib_path: Optional[Path] = None
    if platform.system() == "Windows":
        venv_lib_path = VENV_DIR / "Lib" / "site-packages"
    else: # Linux/macOS
        try:
             py_version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
             venv_lib_path = VENV_DIR / "lib" / py_version_dir / "site-packages"
        except Exception as path_e:
             bootstrap_logger.warning(f"Could not construct standard venv site-packages path: {path_e}")

    if venv_lib_path and venv_lib_path.is_dir() and str(venv_lib_path) not in sys.path:
        bootstrap_logger.info(f"Adding venv site-packages to sys.path: {venv_lib_path}")
        sys.path.insert(0, str(venv_lib_path))
    else:
         bootstrap_logger.warning(f"Could not find or add venv site-packages path: {venv_lib_path}")

    try:
        # --- Core Library Imports ---
        import pipmaster as pm
        import ascii_colors as logging # Use logging alias NOW
        from ascii_colors import ASCIIColors, trace_exception, Menu, MenuItem # Import Menu for CLI
        from configguard import ConfigGuard, ValidationError, generate_encryption_key, ConfigSection
        from configguard.exceptions import ConfigGuardError, SchemaError, SettingNotFoundError # Import specific error
        from configguard.handlers import JsonHandler, YamlHandler, TomlHandler, SqliteHandler
    except ImportError as e:
        # If this happens, bootstrap or requirements likely failed
        exit_with_error_bootstrap(
            f"Critical: Failed import core libraries after install ({e}). "
            f"Check venv ({VENV_DIR}), ensure it's activated, or check logs.", 1
        )
    except Exception as e:
         bootstrap_logger.exception(f"Critical error during post-bootstrap imports: {e}")
         exit_with_error_bootstrap(f"Critical error during library imports: {e}", 1)


    # --- Configure ascii_colors Logging ---
    try:
        # Configure logging using the imported ascii_colors library
        logging.basicConfig(
            level=logging.INFO,
            format='{asctime} | {levelname:<8} | {name} | {message}',
            style='{'
        )
        logger = logging.getLogger("lollms_installer") # Main installer logger
        logger.info("Installer logging configured using ascii_colors.")
    except Exception as e:
        # Fallback to standard logging if ascii_colors config fails unexpectedly
        pylogging.basicConfig(level=pylogging.INFO, format='FALLBACK %(levelname)s: %(message)s')
        logger = pylogging.getLogger("lollms_installer_fallback")
        logger.error(f"Failed to configure ascii_colors logging: {e}. Using basic fallback logging.")
        # Define dummy ASCIIColors for print functions if config failed
        class DummyASCIIColors:
            @staticmethod
            def print(text, **kwargs): print(text)
            @staticmethod
            def error(text): logger.error(text)
            @staticmethod
            def warning(text): logger.warning(text)
            @staticmethod
            def info(text): logger.info(text)
            @staticmethod
            def success(text): logger.info(f"SUCCESS: {text}")
            @staticmethod
            def prompt(text, **kwargs): return input(text)
            @staticmethod
            def confirm(text, default_yes=True): suffix = " (Y/n)" if default_yes else " (y/N)"; resp = input(f"{text}{suffix}: ").strip().lower(); return default_yes if not resp else resp in ['y', 'yes']
            @staticmethod
            def bold(text, **kwargs): print(f"\n--- {text} ---")
            @staticmethod
            def execute_with_animation(pending_text, func, *args, **kwargs): print(f"Executing: {pending_text}..."); return func(*args, **kwargs) # No animation
        ASCIIColors = DummyASCIIColors # type: ignore
        # Use standard logger exception for trace_exception fallback
        def trace_exception(e): logger.exception(e)

    # --- Global Variables (Post-Import) ---
    CONFIG_BASE_DIR: Optional[Path] = None # Set by user prompt later
    # Handlers map for ConfigGuard instance creation
    HANDLER_MAP = { ".yaml": YamlHandler, ".yml": YamlHandler, ".json": JsonHandler, ".toml": TomlHandler, ".db": SqliteHandler, ".sqlite": SqliteHandler, ".sqlite3": SqliteHandler }
    # Cached main config
    _main_config_guard: Optional[ConfigGuard] = None
    # Initialized PackageManager
    pm_manager: Optional[pm.PackageManager] = None

    # --- Helper Functions using ASCIIColors ---
    def print_step(message: str):
        """Prints a formatted step message using ASCIIColors."""
        ASCIIColors.bold(message, color=ASCIIColors.color_bright_cyan)

    def exit_with_error(message: str, exit_code: int = 1):
        """Logs an error and exits using ASCIIColors."""
        logger.error(message)
        trace_exception(Exception(message)) # Log stack trace for context
        if platform.system() == "Windows":
            ASCIIColors.prompt("Press Enter to exit...")
        sys.exit(exit_code)

    # --- Install Core Requirements ---
    print_step(f"Installing Core Dependencies from '{REQUIREMENTS_FILE.name}'")
    if not REQUIREMENTS_FILE.exists():
        exit_with_error(f"'{REQUIREMENTS_FILE.name}' not found.")

    logger.info("Using pipmaster to install/verify core packages...")
    core_install_ok = False
    try:
        # Use the venv python executable directly with pip module
        pip_cmd = [VENV_PYTHON_EXE, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE.resolve()), "--upgrade", "--no-cache-dir"]
        core_install_ok = _run_subprocess_command(
            pip_cmd,
            f"core dependency installation from {REQUIREMENTS_FILE.name}",
            timeout=900 # Increase timeout significantly for installs
        )

        if core_install_ok:
            logger.info("Core dependencies installed/verified successfully.")
        else:
            logger.error("Core dependency installation failed (check logs above).")

    except Exception as e:
        logger.error(f"An unexpected error occurred during core dependency installation: {e}")
        trace_exception(e)
        core_install_ok = False

    if not core_install_ok:
        exit_with_error("Core dependency installation failed. Cannot continue.", 1)

    # --- Initialize PackageManager ---
    try:
        pm_manager = pm.PackageManager(python_executable=VENV_PYTHON_EXE)
        logger.info("Pipmaster PackageManager initialized for optional installs.")
    except Exception as e:
        exit_with_error(f"Failed to initialize PackageManager: {e}", 1)

    # --- Select User Data Directory ---
    def select_user_data_directory() -> Path:
        """Prompts the user to select or confirm the directory for user data/configs."""
        global CONFIG_BASE_DIR # Allow modifying global
        print_step("Select User Data/Configuration Directory")
        logger.info(
            "This directory will store main_config.*, binding configs, "
            "personal components (bindings, personalities, functions), logs, etc."
        )
        default_dir = PROJECT_ROOT / DEFAULT_CONFIG_BASE_DIR_NAME
        while True:
            ASCIIColors.print(f"\nEnter the full path for your personal data directory.")
            ASCIIColors.print(f"Default:", color=ASCIIColors.color_yellow)
            ASCIIColors.print(f"'{default_dir}'")
            user_input = ASCIIColors.prompt("Directory path [Enter for default]: ").strip()
            chosen_path_str = user_input or str(default_dir)
            try:
                chosen_path = Path(chosen_path_str).resolve()
                if not chosen_path.exists():
                    ASCIIColors.warning(f"Directory '{chosen_path}' does not exist.")
                    if ASCIIColors.confirm("Create it?", default_yes=True):
                        try:
                            chosen_path.mkdir(parents=True, exist_ok=True)
                            logger.info(f"Created directory: {chosen_path}")
                        except Exception as mkdir_e:
                            logger.error(f"Failed create directory {chosen_path}: {mkdir_e}")
                            trace_exception(mkdir_e)
                            continue # Ask again
                    else:
                        print("Please choose existing directory or allow creation.")
                        continue # Ask again
                elif not chosen_path.is_dir():
                    logger.error(f"Path exists but is not a directory: {chosen_path}")
                    continue # Ask again

                # Set the global variable
                CONFIG_BASE_DIR = chosen_path
                logger.info(f"User data directory set to: {CONFIG_BASE_DIR}")
                return chosen_path
            except Exception as e:
                logger.error(f"Invalid path or error during selection: {e}")
                trace_exception(e)
                ASCIIColors.warning("An error occurred. Please try entering the path again.")

    # Prompt user immediately after core installs
    CONFIG_BASE_DIR = select_user_data_directory()

    # --- Define CLI Menu Functions ---

    def run_configuration_wizard():
        """Runs the separate configuration wizard script."""
        print_step("Running Configuration Wizard")
        if not CONFIG_BASE_DIR:
            ASCIIColors.error("Cannot run wizard: Configuration base directory not set.")
            return

        wizard_script = PROJECT_ROOT / "configuration_wizard.py"
        if not wizard_script.exists():
            ASCIIColors.error(f"Wizard script not found at {wizard_script}. Cannot configure automatically.")
            return

        ASCIIColors.info(
            "The wizard will guide you through setting up 'main_config.*' "
            "and optionally binding instances."
        )

        try:
            # Run from project root, using venv python, inherit environment
            process = subprocess.run(
                [VENV_PYTHON_EXE, str(wizard_script)],
                env=None, # Inherit environment from activated venv
                check=False, # Keep manual check below
                cwd=PROJECT_ROOT
            )

            # --- IMPROVED LOGGING ---
            logger.debug(f"Wizard process completed. Return Code: {process.returncode}")
            # Wizard now handles its own output logging

            if process.returncode == 0:
                ASCIIColors.success("Configuration wizard completed successfully.")
            else:
                ASCIIColors.error(f"Configuration wizard exited with error code: {process.returncode}.")
                ASCIIColors.error("Please check the output above for details.")

        except FileNotFoundError:
            ASCIIColors.error(f"Failed to execute Python: '{VENV_PYTHON_EXE}'")
        except Exception as e:
            ASCIIColors.error(f"Error running wizard: {e}")
            trace_exception(e)

        # Invalidate cached main config after wizard runs, forcing reload
        global _main_config_guard
        _main_config_guard = None

    def install_optional_dependencies():
        """Installs optional dependencies based on pyproject.toml extras."""
        global pm_manager
        if not pm_manager:
            logger.error("Pipmaster manager not initialized!")
            return

        print_step("Install Optional Features / Bindings")
        ASCIIColors.info("Select features or binding support to install.")
        ASCIIColors.info("This may take some time depending on the feature.")

        # Load extras from pyproject.toml dynamically
        options_list: List[Tuple[str, str]] = []
        try:
            with open(PROJECT_ROOT / "pyproject.toml", "r") as f:
                pyproject_data = toml.load(f)
            optional_deps = pyproject_data.get("project", {}).get("optional-dependencies", {})
            for extra_name, deps_list in optional_deps.items():
                 # Construct the install specifier (e.g., lollms_server[openai])
                 install_spec = f"lollms_server[{extra_name}]"
                 # Create a user-friendly description
                 desc = extra_name.replace("_", " ").title()
                 # Customize descriptions for clarity
                 if extra_name == 'all': desc = "All Optional Features & Bindings"
                 elif extra_name == 'dev': desc = "Development Tools (pytest, linters, etc)"
                 elif 'handler' in extra_name: desc += " (Config Format)"
                 elif 'binding' in extra_name: desc += " (Backend)"
                 elif 'encryption' in extra_name: desc += " (Security)"
                 else: desc += " Support" # Generic fallback
                 options_list.append((desc, install_spec))
            options_list.sort() # Sort alphabetically by description
        except Exception as e:
             logger.error(f"Could not read optional dependencies from pyproject.toml: {e}. Using hardcoded list.")
             # Fallback hardcoded list
             options_list = [
                ("YAML Handler (ConfigGuard)", "lollms_server[yaml]"),
                ("TOML Handler (ConfigGuard)", "lollms_server[toml]"),
                ("Encryption Support (ConfigGuard)", "lollms_server[encryption]"),
                ("OpenAI Binding", "lollms_server[openai]"),
                ("Ollama Binding", "lollms_server[ollama]"),
                ("Gemini Binding", "lollms_server[gemini]"),
                ("DALL-E Binding", "lollms_server[dalle]"),
                ("Llama.cpp Binding (CPU/GPU)", "lollms_server[llamacpp]"),
                ("Diffusers Binding (GPU)", "lollms_server[diffusers]"),
                ("All Optional Features", "lollms_server[all]"),
                ("Development Tools (pytest, etc)", "lollms_server[dev]"),
            ]


        while True:
            menu = Menu("Optional Installations Menu", enable_filtering=True)
            for desc, spec in options_list:
                # Pass spec and desc via lambda for correct scope
                menu.add_action(desc, lambda s=spec, d=desc: (s, d))
            menu.add_action("Return to Main Menu", lambda: "done")

            selection = menu.run()

            if selection == "done" or selection is None:
                break
            elif isinstance(selection, tuple) and len(selection) == 2:
                package_spec, description = selection
                ASCIIColors.info(f"\nAttempting install: {description} ({package_spec})")
                try:
                    # Install using the venv python executable directly with pip module
                    # Use the format `.[extra]` for installing extras of the current editable project
                    install_cmd = []
                    if '[' in package_spec and ']' in package_spec:
                        extra_name = package_spec.split('[')[1].split(']')[0]
                        # Install the extra for the current project (assumed editable)
                        install_cmd = [VENV_PYTHON_EXE, "-m", "pip", "install", f".[{extra_name}]", "--upgrade", "--no-cache-dir"]
                    else:
                        # Handle cases where the spec might just be a package name (unlikely with extras)
                         install_cmd = [VENV_PYTHON_EXE, "-m", "pip", "install", package_spec, "--upgrade", "--no-cache-dir"]


                    if install_cmd:
                        success = _run_subprocess_command(install_cmd, f"installation for {description}", timeout=1800) # Even longer timeout

                        if success:
                            ASCIIColors.success(f"Finished installation/verification for {description}.")
                        else:
                            ASCIIColors.error(f"Installation failed for {description}. Check logs above.")
                    else:
                         ASCIIColors.error(f"Could not parse package spec: {package_spec}")

                except Exception as e:
                    ASCIIColors.error(f"Error setting up installation task: {e}")
                    trace_exception(e)
                ASCIIColors.prompt("Press Enter to continue...") # Pause after each attempt
            else:
                ASCIIColors.warning("Invalid selection.")

    # --- Binding Instance Management (CLI Helpers) ---

    def _find_main_config_file(config_base_dir: Path) -> Optional[Path]:
        """Finds the first supported config file in the base directory."""
        supported_extensions = list(HANDLER_MAP.keys())
        preferred_order = [".yaml", ".yml", ".toml", ".json", ".db", ".sqlite", ".sqlite3"]
        for ext in preferred_order:
            if ext in supported_extensions:
                potential_path = config_base_dir / f"main_config{ext}"
                if potential_path.is_file():
                    logger.debug(f"Found main config file: {potential_path}")
                    return potential_path
        logger.debug(f"No main_config.* file found in {config_base_dir} using preferred extensions.")
        return None # Return None if not found with preferred

    def _load_main_config_guard() -> Optional[ConfigGuard]:
        """Loads or retrieves the main config using ConfigGuard."""
        global _main_config_guard, CONFIG_BASE_DIR
        if _main_config_guard:
            return _main_config_guard
        if not CONFIG_BASE_DIR:
            ASCIIColors.error("Configuration base directory not set. Cannot load main config.")
            return None

        # Find the actual config file
        config_file = _find_main_config_file(CONFIG_BASE_DIR)

        if not config_file:
            ASCIIColors.error(f"No main_config.* file found in {CONFIG_BASE_DIR}. Run Wizard first.")
            return None

        try:
             # Import schema definition from core.config
             # Ensure lollms_server is importable (sys.path should be set)
             from lollms_server.core.config import MAIN_SCHEMA, MAIN_CONFIG_VERSION

             # Try getting encryption key from environment first
             enc_key_str = os.environ.get("LOLLMS_ENCRYPTION_KEY")
             enc_key = enc_key_str.encode() if enc_key_str else None

             _main_config_guard = ConfigGuard(
                 schema=MAIN_SCHEMA, # Pass schema without version key
                 instance_version=MAIN_CONFIG_VERSION, # Pass expected version
                 config_path=config_file,
                 encryption_key=enc_key
                 # Handler is automatically determined by ConfigGuard from path
             )
             _main_config_guard.load() # Load and perform version check/migration

             # Load key from file if not in env and if loaded successfully
             if not enc_key and hasattr(_main_config_guard, 'security'):
                 config_key_str = getattr(_main_config_guard.security, "encryption_key", None)
                 if config_key_str and isinstance(config_key_str, str):
                     try:
                          enc_key_from_file = config_key_str.encode('utf-8')
                          _main_config_guard.set_encryption_key(enc_key_from_file)
                          logger.info("Applied encryption key found in main config.")
                     except Exception as key_err:
                          logger.error(f"Failed to encode/apply encryption key from config: {key_err}")

             # --- Resolve Paths After Loading ---
             if hasattr(_main_config_guard, 'paths'):
                 # Import helpers here to avoid circular dependencies at module level
                 from lollms_server.core.config import _resolve_paths, _update_derived_paths, _ensure_directories
                 logger.debug("Resolving paths within loaded main config...")
                 paths_section = _main_config_guard.paths
                 server_root = PROJECT_ROOT # Assuming installer runs from root
                 # Resolve base paths relative to server root
                 _resolve_paths(paths_section, server_root)
                 # Update derived paths relative to the (now absolute) config_base_dir
                 _update_derived_paths(paths_section, Path(paths_section.config_base_dir))
                 # Optionally ensure directories exist here if needed by CLI tools
                 # _ensure_directories(paths_section) # Maybe skip for CLI tool
             else:
                 logger.error("Main config loaded but missing 'paths' section. Path-dependent operations may fail.")


             logger.info(f"Main configuration loaded successfully from {config_file}")
             return _main_config_guard
        except ImportError as ie:
            # This can happen if the handler for the found config file is missing
            ASCIIColors.error(f"Handler dependency missing for {config_file}: {ie}")
            ASCIIColors.error("Please install the required optional dependency (e.g., 'toml', 'pyyaml').")
            return None
        except SchemaError as se:
             ASCIIColors.error(f"Schema Error loading main config {config_file}: {se}")
             trace_exception(se)
             return None
        except ConfigGuardError as cge:
             ASCIIColors.error(f"ConfigGuard Error loading main config {config_file}: {cge}")
             trace_exception(cge)
             return None
        except Exception as e:
            ASCIIColors.error(f"Failed to load main config file {config_file}: {e}")
            trace_exception(e)
            return None

    def _get_binding_types_from_disk(main_config_obj: ConfigGuard) -> Dict[str, Dict[str, Any]]:
        """Scans binding folders to find available types and their cards."""
        binding_types = {}
        potential_dirs: List[Path] = []
        server_root = PROJECT_ROOT # Installer runs from project root

        # Use resolved paths from the loaded main config
        try:
             paths_section = getattr(main_config_obj, "paths")
             example_folder_str = getattr(paths_section, "example_bindings_folder", None)
             personal_folder_str = getattr(paths_section, "bindings_folder", None)
        except AttributeError:
             logger.error("Could not access 'paths' section in main config. Cannot scan for binding types.")
             return {}

        # Paths should be absolute after _load_main_config_guard resolves them
        example_folder = Path(example_folder_str) if example_folder_str and Path(example_folder_str).is_absolute() else None
        personal_folder = Path(personal_folder_str) if personal_folder_str and Path(personal_folder_str).is_absolute() else None

        if example_folder and example_folder.is_dir(): potential_dirs.append(example_folder)
        if personal_folder and personal_folder.is_dir():
            if not example_folder or personal_folder != example_folder: potential_dirs.append(personal_folder)

        if not potential_dirs: logger.warning("No binding type folders configured or found."); return {}

        logger.debug(f"Scanning for binding types in: {[str(d) for d in potential_dirs]}")
        for bdir in potential_dirs:
            try:
                for item in bdir.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        card_path = item / "binding_card.yaml"
                        init_path = item / "__init__.py"
                        if init_path.exists() and card_path.exists():
                            type_name_from_folder = item.name # Use folder name as fallback/key
                            display_name = type_name_from_folder
                            card_data = None
                            try:
                                with open(card_path, 'r', encoding='utf-8') as f:
                                    card_data = yaml.safe_load(f)
                                if not isinstance(card_data, dict):
                                    logger.warning(f"Invalid format (not dict) in binding card: {card_path}")
                                    card_data = None # Invalidate card data
                                else:
                                    # Validate essential keys from card
                                    type_name = card_data.get("type_name")
                                    instance_schema = card_data.get("instance_schema")
                                    if not type_name or not isinstance(instance_schema, dict):
                                        logger.warning(f"Card {card_path.name} missing 'type_name' or invalid 'instance_schema'.")
                                        card_data = None # Invalidate
                                    else:
                                        display_name = card_data.get('display_name', type_name)

                            except yaml.YAMLError as e: logger.error(f"YAML parse error in {card_path}: {e}")
                            except Exception as e: logger.error(f"Error loading card {card_path}: {e}")

                            # Store if valid and prioritize personal
                            if card_data: # Only store if card_data is still valid
                                 type_name = card_data["type_name"] # Use validated type_name
                                 card_data["package_path"] = item.resolve() # Store absolute path
                                 if type_name not in binding_types or bdir == personal_folder:
                                     binding_types[type_name] = {"display_name": display_name, "path": item, "card": card_data}

            except OSError as e: logger.warning(f"OS error scanning directory {bdir}: {e}")
            except Exception as e: logger.warning(f"Unexpected error scanning directory {bdir}: {e}"); trace_exception(e)

        logger.info(f"Discovered {len(binding_types)} valid binding types.")
        return binding_types

    def _get_binding_instances_info(main_config_obj: ConfigGuard) -> List[Tuple[str, Path, Dict[str, Any]]]:
        """Lists existing binding instance config files and tries to load basic info."""
        instances_info: List[Tuple[str, Path, Dict[str, Any]]] = []
        instance_folder: Optional[Path] = None

        # Safely get instance folder path from loaded main config (should be absolute)
        try:
            paths_section = getattr(main_config_obj, "paths")
            instance_folder_str = getattr(paths_section, "instance_bindings_folder", None)
            if instance_folder_str: instance_folder = Path(instance_folder_str)
        except AttributeError:
             logger.error("Could not access 'paths.instance_bindings_folder' in main config.")
        except Exception as e:
             logger.error(f"Error getting instance folder path from config: {e}")

        if not instance_folder or not instance_folder.is_dir():
            logger.debug(f"Binding instances directory does not exist or not configured: {instance_folder}")
            return []

        # Get bindings map from main config
        bindings_map: Dict[str, str] = {}
        try:
            map_section = getattr(main_config_obj, "bindings_map", None)
            if map_section: bindings_map = map_section.get_dict() or {}
        except Exception as e: logger.warning(f"Could not get bindings_map from config: {e}")

        # Get encryption key from main config
        enc_key: Optional[bytes] = None
        try:
            sec_section = getattr(main_config_obj, "security", None)
            enc_key_str = getattr(sec_section, "encryption_key", None) if sec_section else None
            if enc_key_str: enc_key = enc_key_str.encode()
        except Exception as e: logger.warning(f"Could not get encryption key from config: {e}")

        # Scan the instance directory
        try:
            for item in instance_folder.iterdir():
                if item.is_file() and item.suffix.lower() in HANDLER_MAP:
                    instance_name = item.stem
                    instance_type_from_map = bindings_map.get(instance_name, "Unknown (Not in map)")
                    basic_info = {"name": instance_name, "type_from_map": instance_type_from_map, "file": item.name}
                    # Try loading basic info (like type) from the file itself
                    try:
                         handler_class = HANDLER_MAP.get(item.suffix.lower())
                         if handler_class:
                             # Load without schema just to peek at values
                             temp_guard = ConfigGuard(schema={}, config_path=item, encryption_key=enc_key, handler=handler_class())
                             temp_guard.load(ignore_schema_errors=True)
                             loaded_dict = temp_guard.get_config_dict()
                             if isinstance(loaded_dict, dict):
                                  basic_info["type_from_file"] = loaded_dict.get("type", "N/A")
                             else: basic_info["type_from_file"] = "ErrorLoading(NotDict)"
                    except Exception as load_err:
                         logger.debug(f"Quick load failed for {item.name}: {load_err}")
                         basic_info["type_from_file"] = "ErrorLoading"

                    instances_info.append((instance_name, item, basic_info))
        except OSError as e: logger.warning(f"Error scanning instance directory {instance_folder}: {e}")
        except Exception as e: logger.warning(f"Unexpected error scanning {instance_folder}: {e}"); trace_exception(e)

        logger.debug(f"Found {len(instances_info)} instance config files.")
        return instances_info

    def _select_from_list(options: List[Tuple[str, Any]], prompt: str) -> Optional[Any]:
        """Generic helper to select an item from a list using ASCIIColors Menu."""
        if not options:
            ASCIIColors.warning("No options available to select from.")
            return None

        menu_items = [MenuItem(text=display, value=value) for display, value in options]
        menu_items.append(MenuItem(text="(Cancel / Go Back)", value=None)) # Add cancel option

        menu = Menu(prompt, mode='select_single', item_color=ASCIIColors.color_cyan, title_color=ASCIIColors.color_yellow)
        selected_value = menu.add_choices(menu_items).run() # Returns the 'value' directly

        return selected_value

    def _interactive_edit_config(config_guard: ConfigGuard, section_name: str) -> bool:
        """Interactively edits a ConfigGuard object based on its schema."""
        modified = False
        try:
            # Get schema definition from the ConfigGuard instance
            schema_definition = config_guard.get_instance_schema_definition()
        except Exception as e:
            ASCIIColors.error(f"Could not get schema definition for '{section_name}': {e}")
            return False

        if not schema_definition:
            ASCIIColors.error(f"Schema definition for '{section_name}' is empty.")
            return False

        sorted_keys = sorted(schema_definition.keys())

        for setting_key in sorted_keys:
            # Skip internal keys or keys managed automatically
            if setting_key.startswith('__') or setting_key in ["binding_type", "binding_instance_name", "type"]:
                continue

            setting_schema = schema_definition[setting_key]
            # Extract schema properties safely
            setting_type = setting_schema.get("type", "unknown")
            setting_help = setting_schema.get("help", "")
            setting_options = setting_schema.get("options")
            setting_default = setting_schema.get("default")
            is_nullable = setting_schema.get("nullable", False)
            is_secret = setting_schema.get("secret", False)

            current_value = getattr(config_guard, setting_key, setting_default)
            display_value = "(******)" if is_secret and current_value else repr(current_value)

            while True: # Loop for valid input for this setting
                # --- Display Prompt ---
                prompt_text = f"- {setting_key} ({setting_type})"
                if setting_help: prompt_text += f": {setting_help}"
                if is_nullable: prompt_text += " (Optional)"
                if is_secret: prompt_text += " [SECRET]"
                ASCIIColors.print(f"\n{prompt_text}", color=ASCIIColors.color_cyan)
                ASCIIColors.print(f"  (Current: {display_value})", color=ASCIIColors.color_yellow)
                if setting_default is not None:
                    ASCIIColors.print(f"  (Default: {setting_default!r})", color=ASCIIColors.color_yellow)

                # --- Get User Input ---
                user_input_val: Any = None
                input_type = setting_type.lower()

                if setting_options:
                    opts = [(str(opt), opt) for opt in setting_options]
                    if is_nullable: opts.append(("(Set to None)", None)) # Add None option if nullable
                    opts.append(("(Keep Current)", "##KEEP##"))
                    selected = _select_from_list(opts, f"Select value for {setting_key}")
                    if selected == "##KEEP##":
                        user_input_val = current_value # Keep current
                        break # Move to next setting
                    else:
                        user_input_val = selected # Could be None if chosen
                elif input_type == "bool":
                    current_bool = bool(current_value) if current_value is not None else bool(setting_default)
                    user_input_val = ASCIIColors.confirm("Enable this setting?", default_yes=current_bool)
                elif input_type == "list":
                    ASCIIColors.print("  (Enter items separated by commas)")
                    raw_input = ASCIIColors.prompt("New list value (or Enter to keep current): ").strip()
                    if not raw_input: # User pressed Enter
                        user_input_val = current_value
                        break # Move to next setting
                    else:
                        # Split and strip each item, filter out empty strings
                        user_input_val = [item.strip() for item in raw_input.split(',') if item.strip()]
                elif input_type == "dict": # Handle simple dictionary input
                    ASCIIColors.print("  (Enter key-value pairs like key1=value1, key2=value2)")
                    raw_input = ASCIIColors.prompt("New dictionary values (or Enter to keep current): ").strip()
                    if not raw_input:
                        user_input_val = current_value
                        break
                    else:
                        try:
                             new_dict = {}
                             pairs = raw_input.split(',')
                             for pair in pairs:
                                 if '=' in pair:
                                     key, value = pair.split('=', 1)
                                     new_dict[key.strip()] = value.strip() # Store as strings for now
                             user_input_val = new_dict
                        except Exception as dict_e:
                             ASCIIColors.error(f"Invalid dictionary format: {dict_e}")
                             continue # Re-prompt for dictionary
                else: # str, int, float
                    prompt_msg = "New value (or Enter to keep current): "
                    raw_input = ASCIIColors.prompt(prompt_msg, hide_input=is_secret).strip()
                    if not raw_input: # User pressed Enter
                        user_input_val = current_value
                        break # Move to next setting
                    else:
                        user_input_val = raw_input # ConfigGuard handles type conversion on set

                # --- Attempt to Set and Validate ---
                try:
                    value_before = getattr(config_guard, setting_key, None)
                    # ConfigGuard handles type casting and validation here
                    setattr(config_guard, setting_key, user_input_val)
                    value_after = getattr(config_guard, setting_key)
                    display_after = "(******)" if is_secret and value_after else repr(value_after)
                    ASCIIColors.success(f"Set '{setting_key}' = {display_after}")
                    if value_before != value_after:
                        modified = True
                    break # Valid input received, move to next setting
                except ValidationError as e:
                    ASCIIColors.error(f"Invalid value: {e}")
                    # Loop continues to re-prompt for this setting
                except Exception as e:
                    ASCIIColors.error(f"Error setting value: {e}")
                    trace_exception(e)
                    # Loop continues, maybe prompt again or offer skip? Let's re-prompt.
        return modified # Return whether any value was changed

    def _manage_binding_instances_cli():
        """CLI Submenu for managing binding instances."""
        print_step("Manage Binding Instances (CLI)")
        # Load main config each time to ensure we have latest map/paths/key
        main_config_obj = _load_main_config_guard()
        if not main_config_obj:
            ASCIIColors.error("Could not load main configuration. Cannot manage instances.")
            return

        # Get necessary info from main config
        enc_key_str = getattr(main_config_obj.security, "encryption_key", None)
        enc_key = enc_key_str.encode() if enc_key_str else None
        # Get instance folder path (should be absolute after loading)
        instance_folder_str = getattr(main_config_obj.paths, "instance_bindings_folder", "")
        instance_folder = Path(instance_folder_str) if instance_folder_str else None

        if not instance_folder or not instance_folder.is_dir():
            ASCIIColors.error(f"Instance configuration folder is invalid or missing: {instance_folder}")
            ASCIIColors.error("Please run the Configuration Wizard or correct the 'instance_bindings_folder' path in your main config.")
            return

        # Load available binding types only once per entry into this menu
        available_binding_types = _get_binding_types_from_disk(main_config_obj)

        while True:
            # Refresh instance list and map each time the menu loops
            instance_infos = _get_binding_instances_info(main_config_obj)
            try:
                 bindings_map = getattr(main_config_obj, "bindings_map", {}).get_dict() or {}
            except Exception as e:
                 logger.warning(f"Could not get bindings_map from config: {e}"); bindings_map={}

            ASCIIColors.print("\n--- Configured Binding Instances ---", color=ASCIIColors.color_cyan)
            if not instance_infos:
                ASCIIColors.print(f"(None found in {instance_folder})")
            else:
                 sorted_instances = sorted(instance_infos, key=lambda x: x[0]) # Sort by instance name
                 for name, path, info in sorted_instances:
                      map_type = bindings_map.get(name)
                      file_type = info.get('type_from_file')
                      type_info = f"Type (Map): {map_type or 'Not Mapped!'}"
                      if file_type and file_type != "ErrorLoading" and file_type != "ErrorLoading(NotDict)" and file_type != "N/A":
                           type_info += f" (File: {file_type})"
                           if map_type and map_type != file_type: type_info += ASCIIColors.color_bright_red + " - MISMATCH!" + ASCIIColors.color_reset
                      elif file_type == "ErrorLoading" or file_type == "ErrorLoading(NotDict)": type_info += " (File: Error Reading)"
                      elif file_type == "N/A": type_info += " (File: Type Missing)"
                      else: type_info += " (File: Type Unknown)"
                      ASCIIColors.print(f"- {name} ({path.name}) - {type_info}")

            # Build Menu
            menu = Menu("Binding Instance Management")
            menu.add_action("Add New Instance", lambda: "add")
            if instance_infos:
                menu.add_action("Edit Instance", lambda: "edit")
                menu.add_action("Remove Instance", lambda: "remove")
            menu.add_action("Return to Main Menu", lambda: "done")

            action = menu.run()

            if action == "done" or action is None: break # Exit this submenu

            # --- ADD INSTANCE ---
            elif action == "add":
                if not available_binding_types:
                    ASCIIColors.error("Cannot add instance: No binding types were discovered."); continue

                # 1. Select Binding Type
                type_options = [ (f"{info.get('display_name', name)} ({name})", name) for name, info in sorted(available_binding_types.items()) ]
                selected_type = _select_from_list(type_options, "Select Binding Type to Add:")
                if not selected_type: continue # User cancelled

                type_info = available_binding_types[selected_type]
                instance_schema = type_info.get("card", {}).get("instance_schema")
                if not instance_schema or not isinstance(instance_schema, dict):
                    ASCIIColors.error(f"Invalid or missing instance schema for type '{selected_type}'. Cannot add."); continue

                # 2. Get unique instance name
                instance_name = ""
                existing_names = [name for name, _, _ in instance_infos] + list(bindings_map.keys())
                while True:
                     instance_name = ASCIIColors.prompt(f"Enter unique name for this '{selected_type}' instance: ").strip()
                     if not instance_name: ASCIIColors.warning("Name cannot be empty."); continue
                     if instance_name in existing_names: ASCIIColors.warning("Instance name already exists or is mapped."); continue
                     if not instance_name.isidentifier(): ASCIIColors.warning("Invalid name (use letters, numbers, underscore, not starting with number)."); continue
                     break # Valid name received

                # 3. Choose file format (default to YAML)
                selected_format = ".yaml" # Default
                handler_class = HANDLER_MAP.get(selected_format)
                if not handler_class: ASCIIColors.error("Internal error: YAML handler missing."); continue
                # Check dependency explicitly for the chosen format
                if not handler_class.check_dependency(raise_exception=False):
                     ASCIIColors.error(f"Dependency missing for YAML handler. Please install PyYAML."); continue

                instance_file_path = instance_folder / f"{instance_name}{selected_format}"

                ASCIIColors.print(f"\n--- Configuring '{instance_name}' (Type: {selected_type}) ---")
                ASCIIColors.print(f"File will be: {instance_file_path}")
                config_instance = None
                try:
                     # 4. Prepare Schema and Initialize ConfigGuard for Instance
                     schema_copy = instance_schema.copy()
                     schema_copy.setdefault("__version__", "0.1.0") # Ensure version
                     schema_copy.setdefault("type", {"type": "str", "default": selected_type})
                     schema_copy.setdefault("binding_instance_name", {"type": "str", "default": instance_name})
                     instance_schema_version = schema_copy.get("__version__", "0.1.0")

                     config_instance = ConfigGuard(
                         schema=schema_copy,
                         instance_version=instance_schema_version,
                         config_path=instance_file_path,
                         handler=handler_class(),
                         autosave=False,
                         encryption_key=enc_key
                     )
                     # Set fixed fields explicitly (though defaults handle it)
                     setattr(config_instance, "type", selected_type)
                     setattr(config_instance, "binding_instance_name", instance_name)

                     # 5. Ask about Customization
                     customize = ASCIIColors.confirm("Customize settings now (or use defaults)?", default_yes=False)
                     modified_by_user = False
                     if customize:
                         modified_by_user = _interactive_edit_config(config_instance, f"Instance '{instance_name}'")

                     # 6. Confirm Save
                     save_action = "Create" if not instance_file_path.exists() else "Update"
                     if modified_by_user or ASCIIColors.confirm(f"{save_action} '{instance_name}' config {'with defaults' if not modified_by_user else 'with modifications'}?", default_yes=True):
                         config_instance.save(mode='values') # Save only values
                         ASCIIColors.success(f"Instance configuration SAVED: {instance_file_path}")

                         # 7. Update and Save Main Config Map
                         try:
                             bindings_map_section = getattr(main_config_obj, "bindings_map")
                             bindings_map_section[instance_name] = selected_type # Add/update map
                             main_config_obj.save(mode='values') # Save updated main config
                             ASCIIColors.success(f"Added/Updated '{instance_name}' -> '{selected_type}' mapping in main config.")
                             _main_config_guard = None # Invalidate cache so next load gets fresh map
                         except Exception as map_err:
                             ASCIIColors.error(f"Failed update main config bindings_map: {map_err}"); trace_exception(map_err)
                             ASCIIColors.warning("Instance file saved, but main config map was NOT updated.")
                     else:
                         ASCIIColors.info("Instance configuration not saved.")

                except Exception as add_err:
                     ASCIIColors.error(f"Error adding instance '{instance_name}': {add_err}"); trace_exception(add_err)

            # --- EDIT INSTANCE ---
            elif action == "edit":
                if not instance_infos: ASCIIColors.warning("No instances to edit."); continue
                edit_options = [(f"{name} ({p.name})", (p, name, info)) for name, p, info in sorted(instance_infos)]
                selection = _select_from_list(edit_options, "Select Instance Configuration to Edit:")
                if not selection: continue # User cancelled
                instance_path, instance_name_to_edit, instance_info = selection

                # Determine instance type (prefer map, fallback to file)
                instance_type = bindings_map.get(instance_name_to_edit)
                type_from_file = instance_info.get('type_from_file')
                if not instance_type:
                    if type_from_file and type_from_file not in ["N/A", "ErrorLoading", "ErrorLoading(NotDict)"]:
                        instance_type = type_from_file
                        ASCIIColors.warning(f"Instance '{instance_name_to_edit}' not found in bindings_map. Using type '{instance_type}' from file.")
                    else:
                         ASCIIColors.error(f"Cannot determine binding type for instance '{instance_name_to_edit}' from map or file. Cannot edit."); continue
                elif instance_type != type_from_file and type_from_file not in ["N/A", "ErrorLoading", "ErrorLoading(NotDict)"]:
                     ASCIIColors.warning(f"Type mismatch for '{instance_name_to_edit}': Map='{instance_type}', File='{type_from_file}'. Using type from map: '{instance_type}'.")

                # Get schema for the determined type
                type_info = available_binding_types.get(instance_type)
                if not type_info or not type_info.get("card"):
                    ASCIIColors.error(f"Cannot edit: Binding type definition '{instance_type}' not found or invalid."); continue
                instance_schema = type_info["card"].get("instance_schema")
                if not instance_schema or not isinstance(instance_schema, dict):
                    ASCIIColors.error(f"Invalid instance schema found for type '{instance_type}'. Cannot edit."); continue

                # Get handler and check dependencies
                handler_class = HANDLER_MAP.get(instance_path.suffix.lower());
                if not handler_class: ASCIIColors.error(f"Unsupported file type for editing: {instance_path.suffix}"); continue
                if not handler_class.check_dependency(raise_exception=False):
                     ASCIIColors.error(f"Dependency missing for {instance_path.suffix}. Cannot edit."); continue

                ASCIIColors.print(f"\n--- Editing '{instance_name_to_edit}' (Type: {instance_type}) ---")
                ASCIIColors.print(f"File: {instance_path}")
                try:
                     # Prepare schema and initialize ConfigGuard
                     schema_copy = instance_schema.copy()
                     schema_copy.setdefault("__version__", "0.1.0")
                     schema_copy.setdefault("type", {"type": "str", "default": instance_type})
                     schema_copy.setdefault("binding_instance_name", {"type": "str", "default": instance_name_to_edit})
                     instance_schema_version = schema_copy.get("__version__", "0.1.0")

                     config_instance = ConfigGuard(
                         schema=schema_copy,
                         instance_version=instance_schema_version,
                         config_path=instance_path,
                         handler=handler_class(),
                         autosave=False,
                         encryption_key=enc_key
                     )
                     config_instance.load() # Load existing values

                     # Run interactive edit
                     modified = _interactive_edit_config(config_instance, f"Instance '{instance_name_to_edit}'")

                     if modified:
                          if ASCIIColors.confirm(f"Save changes to {instance_path.name}?", default_yes=True):
                               config_instance.save(mode='values') # Save updated values
                               ASCIIColors.success("Changes saved.")
                          else: ASCIIColors.info("Changes not saved.")
                     else: ASCIIColors.info("No changes were made.")
                except Exception as edit_err:
                     ASCIIColors.error(f"Error editing instance '{instance_name_to_edit}': {edit_err}"); trace_exception(edit_err)

            # --- REMOVE INSTANCE ---
            elif action == "remove":
                if not instance_infos: ASCIIColors.warning("No instances to remove."); continue
                remove_options = [(f"{name} ({p.name})", (p, name)) for name, p, _ in sorted(instance_infos)]
                selection = _select_from_list(remove_options, "Select Instance Configuration to Remove:")
                if not selection: continue # User cancelled
                instance_path_to_remove, instance_name_to_remove = selection

                ASCIIColors.warning(f"This will permanently DELETE the file: {instance_path_to_remove.name}")
                if ASCIIColors.confirm(f"Proceed with deleting '{instance_name_to_remove}' config file?", default_yes=False):
                    try:
                         instance_path_to_remove.unlink()
                         ASCIIColors.success(f"Instance config file deleted: {instance_path_to_remove.name}")

                         # Remove from main config map if present
                         try:
                              bindings_map_section = getattr(main_config_obj, "bindings_map")
                              if instance_name_to_remove in bindings_map_section:
                                   # Use __delitem__ or del to remove from dynamic section
                                   del bindings_map_section[instance_name_to_remove]
                                   main_config_obj.save(mode='values') # Save updated map
                                   ASCIIColors.success(f"Removed '{instance_name_to_remove}' mapping from main config.")
                                   _main_config_guard = None # Invalidate cache
                              else:
                                   ASCIIColors.info("Instance was not found in main config's bindings_map anyway.")
                         except (KeyError, SettingNotFoundError): # Handle if key already gone
                              ASCIIColors.info("Instance mapping was already absent from main config.")
                         except Exception as map_err:
                              ASCIIColors.error(f"Failed to update main config bindings_map after deletion: {map_err}"); trace_exception(map_err)
                              ASCIIColors.warning("Instance file deleted, but main config map update FAILED.")

                    except OSError as rm_err: ASCIIColors.error(f"Failed to delete file: {rm_err}")
                    except Exception as rm_err: ASCIIColors.error(f"Error removing instance: {rm_err}"); trace_exception(rm_err)
                else: ASCIIColors.info("Removal cancelled.")

            else: ASCIIColors.warning("Unknown action selected.")

            # Pause after action before showing menu again
            if action != "done" and action is not None:
                ASCIIColors.prompt("\nPress Enter to return to the binding management menu...")


    # --- Main Installer Menu ---
    while True:
        main_menu = Menu("LoLLMs Server Installer Menu")
        main_menu.add_action("1. Run Configuration Wizard (Setup main_config.*)", run_configuration_wizard)
        main_menu.add_action("2. Install Optional Dependencies / Bindings", install_optional_dependencies)
        main_menu.add_action("3. Manage Binding Instances (Add/Edit/Remove)", _manage_binding_instances_cli)
        main_menu.add_action("4. Exit Installer", lambda: "exit")

        print_step("Main Menu")
        chosen_action = main_menu.run()

        if chosen_action == "exit" or chosen_action is None:
            logger.info("Exiting installer.")
            break
        elif callable(chosen_action):
            try:
                 chosen_action()
            except Exception as menu_action_err:
                 logger.error(f"Error executing menu action: {menu_action_err}")
                 trace_exception(menu_action_err)
            ASCIIColors.prompt("\nPress Enter to return to main menu...")
        else:
             ASCIIColors.warning(f"Invalid action '{chosen_action}' returned from menu.")


    # --- Final Instructions ---
    print_step("Setup Steps Complete!")
    ASCIIColors.print("\nNext Steps:", style=ASCIIColors.style_bold)
    ASCIIColors.print("------------")
    activate_cmd_win_cmd = f".\\{VENV_DIR.name}\\Scripts\\activate.bat"
    activate_cmd_win_ps = f".\\{VENV_DIR.name}\\Scripts\\Activate.ps1"
    activate_cmd_nix = f"source {VENV_DIR.name}/bin/activate"
    run_cmd_win = f".\\run.bat"
    run_cmd_nix = f"./run.sh"
    ASCIIColors.print("1. Activate Virtual Environment:", color=ASCIIColors.color_cyan)
    if platform.system() == "Windows":
        ASCIIColors.print(f"   - Cmd: ", end=""); ASCIIColors.print(activate_cmd_win_cmd, color=ASCIIColors.color_green)
        ASCIIColors.print(f"   - PowerShell: ", end=""); ASCIIColors.print(activate_cmd_win_ps, color=ASCIIColors.color_green)
        run_cmd = run_cmd_win
    else: # Linux, macOS
        ASCIIColors.print(f"   - Bash/Zsh: ", end=""); ASCIIColors.print(activate_cmd_nix, color=ASCIIColors.color_green)
        run_cmd = run_cmd_nix
    ASCIIColors.print("   (Activate in each new terminal session before running)")
    ASCIIColors.print("\n2. Verify Configuration:", color=ASCIIColors.color_cyan)
    if CONFIG_BASE_DIR:
        main_config_file = _find_main_config_file(CONFIG_BASE_DIR)
        if main_config_file:
            ASCIIColors.print(f"   - Main config found: '{main_config_file}'", color=ASCIIColors.color_yellow)
        else:
            ASCIIColors.warning(f"   - Main config (e.g., main_config.yaml) NOT found in: '{CONFIG_BASE_DIR}'")

        # Try loading config again to get instance path reliably
        mcg = _load_main_config_guard()
        if mcg and hasattr(mcg, 'paths'):
             instance_dir = Path(getattr(mcg.paths, "instance_bindings_folder", CONFIG_BASE_DIR / "bindings"))
             ASCIIColors.print(f"   - Binding instance configs are in: '{instance_dir}'", color=ASCIIColors.color_yellow)
        else:
             instance_dir_guess = CONFIG_BASE_DIR / "bindings" # Guess default location
             ASCIIColors.print(f"   - Binding instance configs *should be* in: '{instance_dir_guess}' (Could not verify from main config)", color=ASCIIColors.color_yellow)
    else:
        ASCIIColors.print("   - Configuration directory was not set during this script run.")
    ASCIIColors.print("\n3. Run the LoLLMs Server:", color=ASCIIColors.color_cyan)
    ASCIIColors.print(f"   - After activating venv (Step 1), run: ", end="")
    ASCIIColors.print(run_cmd, color=ASCIIColors.color_green)
    ASCIIColors.print("\n---------------------------------")
    ASCIIColors.success("Installation script finished.")

    if platform.system() == "Windows":
        ASCIIColors.prompt("Press Enter to exit the installer...")
    sys.exit(0) # Explicitly exit with success code