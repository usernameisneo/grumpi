# configure_wizard.py
import sys
import os
from pathlib import Path
import toml
import yaml
import importlib
import platform
from typing import Dict, Any, List, Optional, Type, Tuple

# --- Add project root to sys.path ---
# Assuming this script is in the root of the lollms_server project
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Import ascii_colors and project components ---
try:
    import ascii_colors as logging  # Use logging alias
    from ascii_colors import ASCIIColors, Menu, MenuItem, ProgressBar, trace_exception
except ImportError:
    print("ERROR: This wizard requires the 'ascii_colors' library.")
    print("Please install it: pip install ascii_colors")
    sys.exit(1)

try:
    # Make sure lollms_server package is discoverable (often needed if running script directly)
    # This might not be necessary if running as `python -m lollms_server.configure_wizard`
    # but adds robustness if run as `python configure_wizard.py` from the root.
    add_project_root_to_path = True
    if 'lollms_server' in sys.modules:
        try:
            # Check if the imported lollms_server is the one from our project root
            imported_loc = Path(sys.modules['lollms_server'].__file__).resolve().parent
            if imported_loc != project_root / 'lollms_server':
                 print(f"Warning: Imported 'lollms_server' seems to be from {imported_loc}, not the project root {project_root / 'lollms_server'}. Adding project root to path.")
            else:
                 add_project_root_to_path = False # Already correct
        except Exception:
            pass # Couldn't check, add path just in case

    if add_project_root_to_path and str(project_root) not in sys.path:
         sys.path.insert(0, str(project_root))
         print(f"Added {project_root} to sys.path for import discovery.")

    from lollms_server.core.bindings import Binding
    from lollms_server.core.personalities import PersonalityConfig
    from lollms_server.utils.file_utils import safe_load_module, find_classes_in_module, add_path_to_sys_path
except ImportError as e:
    logging.error(f"ERROR: Could not import lollms_server components: {e}")
    logging.error("Please ensure you are running this script from the main 'lollms_server' project directory and have installed dependencies (e.g., using `pip install -e .`)")
    sys.exit(1)

# --- Configuration ---
CONFIG_FILENAME = "config.toml"
CONFIG_EXAMPLE_FILENAME = "config.toml.example"
DEFAULT_ZOOS_DIR = project_root / "zoos"
DEFAULT_PERSONAL_DIR = project_root

# --- Logging Setup ---
# Use a logger obtained via the compatibility layer for consistency
logger = logging.getLogger("ConfigWizard")
# Configure logging using basicConfig if needed, or assume it's set elsewhere
# For standalone script use, configure it here:
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- Helper Functions ---

def print_wizard_title(text: str):
    """Prints a title formatted string using yellow."""
    # Use direct print for titles, independent of logging system
    ASCIIColors.bold(f"\n--- {text} ---", color=ASCIIColors.color_yellow)

def get_default_config() -> Dict[str, Any]:
    """Loads structure from example or provides a basic default."""
    example_path = project_root / CONFIG_EXAMPLE_FILENAME
    config = {}
    if example_path.exists():
        try:
            with open(example_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
            ASCIIColors.info(f"Loaded base structure from {CONFIG_EXAMPLE_FILENAME}")
        except Exception as e:
            logger.warning(f"Parse error in {CONFIG_EXAMPLE_FILENAME}: {e}. Using basic structure.")
            config = {} # Fallback to basic if example fails

    # Ensure essential sections exist, using example content or basic defaults
    config.setdefault('server', {"host": "0.0.0.0", "port": 9601})
    config.setdefault('logging', {"log_level": "INFO", "level": 20}) # Level is redundant? Keep standard log_level
    config.setdefault('paths', {})
    config.setdefault('security', {"allowed_api_keys": []})
    config.setdefault('defaults', {})
    config.setdefault('bindings', {})
    config.setdefault('personalities_config', {})
    config.setdefault('resource_manager', {"gpu_strategy": "semaphore", "gpu_limit": 1, "queue_timeout": 120})
    config.setdefault('webui', {"enable_ui": False})

    # Cleanup defaults that might contain sensitive or user-specific info from example
    if 'security' in config and 'allowed_api_keys' in config['security']:
        config['security']['allowed_api_keys'] = []
    if 'bindings' in config:
        keys_to_del = [
            k for k, v in config['bindings'].items()
            if isinstance(v, dict) and ('api_key' in v or 'google_api_key' in v or k.startswith(('local_', 'my_')))
        ]
        for k in keys_to_del:
            if k in config['bindings']: # Check again in case deleted by other rule
                del config['bindings'][k]
    if 'defaults' in config:
        for key in list(config['defaults'].keys()):
            if key.endswith(('_binding', '_model')):
                config['defaults'][key] = None # Reset specific defaults

    return config


def load_existing_config() -> Optional[Dict[str, Any]]:
    """Loads existing config.toml if it exists."""
    config_path = project_root / CONFIG_FILENAME
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except Exception as e:
            # Use logging system here
            logger.error(f"Error loading existing {CONFIG_FILENAME}: {e}")
            trace_exception(e) # Log the traceback
            return None
    return None

def save_config(config_data: Dict[str, Any]) -> bool:
    """Saves the configuration data to config.toml."""
    config_path = project_root / CONFIG_FILENAME
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            toml.dump(config_data, f)
        # Use direct print for immediate success feedback
        ASCIIColors.success(f"Configuration saved successfully to: {config_path}")
        return True
    except Exception as e:
        # Use logging system for errors
        logger.error(f"Error saving configuration to {config_path}: {e}")
        trace_exception(e)
        # Also provide direct feedback
        ASCIIColors.error(f"Error saving configuration: {e}")
        return False

def prompt_for_path(prompt_text: str, default_path: Path, must_exist: bool = False, create_if_not_exist: bool = False) -> Path:
    """Prompts user for a path, validates, and optionally creates it."""
    while True:
        # Use direct print methods for user interaction
        ASCIIColors.print(f"\n{prompt_text}", color=ASCIIColors.color_cyan)
        if default_path:
            ASCIIColors.print(f"(Default: '{default_path}')", color=ASCIIColors.color_yellow)
        else:
             ASCIIColors.print("(No default, path is required)", color=ASCIIColors.color_yellow)

        user_input = ASCIIColors.prompt("Enter path or press Enter for default (if available): ")
        chosen_path_str = user_input.strip()

        if not chosen_path_str and default_path:
            chosen_path_str = str(default_path)
        elif not chosen_path_str and not default_path:
             ASCIIColors.warning("Path cannot be empty.")
             continue

        # Resolve to absolute path
        try:
            chosen_path = Path(chosen_path_str).resolve()
        except Exception as e:
             ASCIIColors.error(f"Invalid path string '{chosen_path_str}': {e}")
             continue


        if must_exist and not chosen_path.exists():
            ASCIIColors.warning(f"Path does not exist: {chosen_path}")
            continue
        elif not chosen_path.exists() and create_if_not_exist:
            if ASCIIColors.confirm(f"Path does not exist: {chosen_path}. Create it?", default_yes=True):
                try:
                    chosen_path.mkdir(parents=True, exist_ok=True)
                    ASCIIColors.success(f"Created directory: {chosen_path}")
                    return chosen_path
                except Exception as e:
                    ASCIIColors.error(f"Could not create directory: {e}")
                    continue
            else:
                continue # Ask again if user doesn't want to create
        elif chosen_path.exists() and not chosen_path.is_dir() and (must_exist or create_if_not_exist):
             ASCIIColors.warning(f"Path exists but is not a directory: {chosen_path}")
             continue

        # If validation passed or wasn't required
        return chosen_path

def discover_component_files(component_type: str, personal_folder: Optional[Path], example_folder: Optional[Path]) -> Dict[str, Path]:
    """Discovers Python files (bindings) or subfolders (personalities)."""
    discovered = {}
    scan_folders = []
    # Add example folder first if it exists and is a directory
    if example_folder and example_folder.is_dir():
        scan_folders.append(example_folder)
        # Add parent to sys.path for potential relative imports within examples
        if example_folder.parent not in sys.path:
             add_path_to_sys_path(str(example_folder.parent))

    # Add personal folder if it exists, is a directory, and different from example
    if personal_folder and personal_folder.is_dir():
        if not example_folder or personal_folder.resolve() != example_folder.resolve():
             scan_folders.append(personal_folder)
             # Add parent to sys.path for potential relative imports within personal items
             if personal_folder.parent not in sys.path:
                 add_path_to_sys_path(str(personal_folder.parent))

    # Use logging for internal info messages
    logger.info(f"Scanning for {component_type} in: {[str(f.relative_to(project_root)) if f and f.is_relative_to(project_root) else str(f) for f in scan_folders]}")

    for folder in scan_folders:
        try:
            for item in folder.iterdir():
                item_resolved = item.resolve() # Resolve once
                if component_type == "bindings":
                    # Store by file stem initially for bindings
                    if item_resolved.is_file() and item_resolved.suffix == ".py" and not item_resolved.name.startswith("__"):
                        # Prioritize personal folder items if stem conflicts
                        if item_resolved.stem not in discovered or folder == personal_folder:
                            discovered[item_resolved.stem] = item_resolved
                elif component_type == "personalities":
                    # Store by folder name for personalities
                    if item_resolved.is_dir() and not item_resolved.name.startswith('.') and (item_resolved / "config.yaml").exists():
                         # Prioritize personal folder items if name conflicts
                        if item_resolved.name not in discovered or folder == personal_folder:
                            discovered[item_resolved.name] = item_resolved
        except Exception as e:
            logger.warning(f"Error scanning {folder}: {e}")

    logger.info(f"Discovered {len(discovered)} potential {component_type}.")
    return discovered

def get_binding_metadata(binding_file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely loads binding metadata."""
    logger.info(f"Loading metadata from: {binding_file_path.name}")
    # Ensure parent dir is in path *before* loading
    if binding_file_path.parent not in sys.path:
         add_path_to_sys_path(str(binding_file_path.parent))

    module, error = safe_load_module(binding_file_path, package_path=binding_file_path.parent)
    metadata = None # Initialize metadata

    if error or not module:
        logger.warning(f"Could not load module {binding_file_path.name}: {error}")
        return None
    try:
        binding_classes = find_classes_in_module(module, Binding)
        if not binding_classes:
            logger.warning(f"No class inheriting from 'Binding' found in {binding_file_path.name}.")
            return None
        binding_class = binding_classes[0] # Assume first is main
        if hasattr(binding_class, 'get_binding_config') and callable(binding_class.get_binding_config):
            # Call the static/class method to get config
            metadata = binding_class.get_binding_config()
            if metadata and 'type_name' in metadata:
                logger.info(f"Successfully loaded metadata for '{metadata.get('type_name')}' from {binding_file_path.name}")
            else:
                 logger.warning(f"Metadata loaded from {binding_file_path.name}, but 'type_name' is missing.")
                 metadata = None # Treat as failure if type_name missing
        else:
            logger.warning(f"Class {binding_class.__name__} in {binding_file_path.name} missing required static method get_binding_config(). Falling back.")
            # Fallback metadata if method missing
            type_name_fallback = getattr(binding_class, 'binding_type_name', binding_file_path.stem) # Check for class attribute
            metadata = {
                "type_name": type_name_fallback,
                "description": "(Metadata method missing or invalid)",
                "config_template": {"type": type_name_fallback} # Basic template
            }
            logger.info(f"Using fallback metadata for type '{type_name_fallback}' from {binding_file_path.name}")

    except Exception as e:
        logger.error(f"Error processing metadata in {binding_file_path.name}: {e}")
        trace_exception(e)
        metadata = None # Ensure failure on exception

    finally:
        # --- Module Unloading Logic ---
        # Try to determine the canonical module name used by importlib
        module_name_to_unload = None
        if module and hasattr(module, '__name__'):
            module_name_to_unload = module.__name__
        elif hasattr(module, '__file__'): # Fallback using file path
             # Construct a plausible module name (might not always be perfect)
             rel_path = binding_file_path.relative_to(project_root) if binding_file_path.is_relative_to(project_root) else binding_file_path
             module_name_to_unload = '.'.join(rel_path.parts[:-1] + (rel_path.stem,))


        # Unload if a name was determined and it's in sys.modules
        if module_name_to_unload and module_name_to_unload in sys.modules:
            try:
                del sys.modules[module_name_to_unload]
                logger.debug(f"Unloaded module: {module_name_to_unload}")
            except KeyError:
                pass # Already unloaded somehow
        elif module_name_to_unload:
             logger.debug(f"Module {module_name_to_unload} determined but not found in sys.modules for unloading.")
        else:
             logger.debug(f"Could not determine canonical module name for {binding_file_path.name} to unload.")
        # Clear reference to module object
        module = None
        binding_classes = []
        binding_class = None

    return metadata

# --- Main Wizard Class ---

class ConfigWizard:
    def __init__(self):
        self.current_config = get_default_config()
        self._saved = False
        # Store discovered paths, not metadata initially
        self.binding_files: Dict[str, Path] = {}  # {file_stem: path}
        self.available_personalities: Dict[str, Path] = {}  # {folder_name: path}
        # Cache for loaded binding metadata to avoid reloading
        self.loaded_binding_metadata: Dict[str, Optional[Dict[str, Any]]] = {}  # {type_name: metadata}
        # Map type_name to path once metadata is loaded successfully
        self.binding_type_to_path: Dict[str, Path] = {}

        # --- Monkey-patch Menu class if methods are missing ---
        # This is a workaround. Ideally, these methods should be in the library.
        if not hasattr(Menu, 'clear_actions'):
            def _menu_clear_actions(menu_self):
                menu_self.items = []
            Menu.clear_actions = _menu_clear_actions
            logger.debug("[Wizard Patch] Added clear_actions to Menu class.")

        if not hasattr(Menu, 'add_info'):
            def _menu_add_info(menu_self, text: str):
                # Add an item that is not selectable, for informational purposes
                menu_self.items.append(MenuItem(text, 'info', None)) # Use 'info' type
            Menu.add_info = _menu_add_info
            logger.debug("[Wizard Patch] Added add_info to Menu class.")


    def load_or_initialize(self):
        """Loads existing config or starts with defaults."""
        existing = load_existing_config()
        if existing:
            if ASCIIColors.confirm(f"Existing {CONFIG_FILENAME} found. Load it and modify?", default_yes=True):
                base = get_default_config() # Start with default structure/defaults
                # Deep merge logic - more robust than simple update
                for section, content in existing.items():
                    if section not in base:
                        base[section] = content # Add new sections from existing
                    elif isinstance(base[section], dict) and isinstance(content, dict):
                         # Recursively merge dictionaries (simple 1-level merge here)
                         # For deeper structures, a recursive merge function would be better
                         base[section].update(content)
                    else:
                         # Overwrite non-dict base values with existing values
                         base[section] = content

                self.current_config = base
                ASCIIColors.info("Loaded and merged existing configuration.")
                return
        ASCIIColors.info("Starting with a default configuration template.")

    # --- Menu Builder Functions ---
    def _build_security_menu_items(self, menu: Menu):
        """Builds items for the Security submenu dynamically."""
        menu.clear_actions() # Clear previous items
        keys = self.current_config.get('security', {}).get('allowed_api_keys', [])
        keys_exist = bool(keys)

        menu.add_action("List Current Keys", self._list_api_keys)
        menu.add_action("Add New Key", self._add_api_key)
        if keys_exist:
            menu.add_action("Edit Existing Key", self._edit_api_key)
            menu.add_action("Remove Key", self._remove_api_key, item_color=ASCIIColors.color_red)
        else:
            menu.add_info(" (No API keys configured)")
        # Back action is added automatically by Menu.run for submenus

    def _build_bindings_menu_items(self, menu: Menu):
         """Builds items for the Bindings submenu dynamically."""
         menu.clear_actions()
         # Discovery happens in the wrapper method _run_bindings_menu
         bindings_exist = bool(self.current_config.get('bindings'))
         bindings_available = bool(self.binding_files)

         if bindings_available:
             menu.add_action("Add New Binding Instance", self._add_binding_instance)
         else:
             menu.add_info("(No binding files found/configured in Paths)")

         if bindings_exist:
             menu.add_action("Modify Existing Instance", self._modify_binding_instance)
             menu.add_action("Remove Binding Instance", self._remove_binding_instance, item_color=ASCIIColors.color_red)
         else:
             if bindings_available: # Only show if add is possible
                 menu.add_info(" (No binding instances configured yet)")
         # Back action is added automatically

    def _build_personality_menu_items(self, menu: Menu):
        """Builds items for the Personality submenu dynamically."""
        menu.clear_actions()
        # Discovery happens in the wrapper method _run_personalities_menu
        pers_config = self.current_config.get('personalities_config', {})
        sorted_names = sorted(self.available_personalities.keys())

        if not sorted_names:
             menu.add_info("(No personalities found/configured in Paths)")
        else:
             menu.add_info("Select personality to toggle Enabled/Disabled:")
             for name in sorted_names:
                 # Default to enabled if not explicitly in config
                 is_enabled = pers_config.get(name, {}).get('enabled', True)
                 prefix = "[✓] " if is_enabled else "[✗] "
                 # Action toggles the state. Menu will be rebuilt on next entry.
                 menu.add_action(f"{prefix}{name}", lambda n=name: self._toggle_personality(n))
         # Back action is added automatically

    # --- New Wrapper Methods for Submenus ---
    def _run_security_menu(self):
        """Creates, populates, and runs the security menu."""
        security_menu = Menu(
            title="Security (API Keys)",
            parent=None, # Set parent=None; Menu.run handles back if called from another menu context if needed
            title_color=ASCIIColors.color_bright_magenta,
            item_color=ASCIIColors.color_cyan,
            selected_background=ASCIIColors.color_bg_cyan,
            selected_prefix="» "
        )
        self._build_security_menu_items(security_menu) # Populate BEFORE running
        security_menu.run() # Run this specific menu level

    def _run_bindings_menu(self):
        """Creates, populates, and runs the bindings menu."""
        # Perform discovery *before* building the menu
        if not self._discover_all():
            ASCIIColors.warning("Cannot configure bindings: Paths discovery failed (likely not configured).")
            ASCIIColors.prompt("Press Enter to return...")
            return # Abort if discovery failed

        bindings_menu = Menu(
            title="Bindings (Models/APIs)", parent=None,
            title_color=ASCIIColors.color_bright_magenta,
            selected_background=ASCIIColors.color_bg_magenta,
            selected_prefix="» "
        )
        self._build_bindings_menu_items(bindings_menu) # Populate AFTER discovery
        bindings_menu.run()

    def _run_personalities_menu(self):
        """Creates, populates, and runs the personalities menu."""
        # Perform discovery *before* building the menu
        if not self._discover_all():
            ASCIIColors.warning("Cannot configure personalities: Paths discovery failed (likely not configured).")
            ASCIIColors.prompt("Press Enter to return...")
            return

        personalities_menu = Menu(
            title="Personalities (Enable/Disable)", parent=None,
            title_color=ASCIIColors.color_bright_cyan,
            selected_background=ASCIIColors.color_bg_cyan,
            selected_prefix="» "
        )
        self._build_personality_menu_items(personalities_menu) # Populate AFTER discovery
        personalities_menu.run()

    # --- Main Run Method ---
    def run(self):
        """Runs the main wizard flow using interactive menus."""
        ASCIIColors.print("\n--- lollms_server Configuration Wizard ---", style=ASCIIColors.style_bold, color=ASCIIColors.color_bright_cyan)
        ASCIIColors.print("This wizard helps set up basic configuration using arrow keys and Enter.")

        self.load_or_initialize()

        # --- Define Root Menu ---
        root_menu = Menu(
            title="Main Configuration Steps",
            title_color=ASCIIColors.color_bright_yellow,
            selected_background=ASCIIColors.color_bg_blue,
            selected_prefix="➔ ",
            unselected_prefix="  "
            # Note: Parent is None for the root menu
        )

        # --- Add Actions/Submenu Triggers to the Root Menu ---
        # Actions now point to the actual configuration logic or submenu runner methods
        root_menu.add_action("Paths Configuration", self.configure_paths)
        root_menu.add_action("Security Settings", self._run_security_menu) # Use wrapper
        root_menu.add_action("Binding Configuration", self._run_bindings_menu) # Use wrapper
        root_menu.add_action("Personality Configuration", self._run_personalities_menu) # Use wrapper
        root_menu.add_action("Default Bindings", self.configure_defaults)
        root_menu.add_action("Web UI Settings", self.configure_webui)
        root_menu.add_action("Review & Save Configuration", self.review_and_save)
        # No explicit "Exit" needed; Menu adds "Quit" automatically

        # --- Run the Root Menu ONCE ---
        # Let Menu.run handle the interaction loop until the user selects "Quit"
        # or presses Ctrl+C in this root menu.
        root_menu.run()

        # --- Check Save Status AFTER Menu Exits ---
        # Execution continues here only after root_menu.run() finishes.
        if self._saved:
            ASCIIColors.success("\nConfiguration Wizard finished and configuration was saved.")
        else:
            ASCIIColors.warning("\nConfiguration Wizard finished or was aborted. Changes were NOT saved.")
            ASCIIColors.info("Run the wizard again to make and save changes.")
            # Consider exiting with an error code if not saved: sys.exit(1) ?

    # --- Configuration Action Functions ---

    def configure_paths(self):
        """Guides user through setting essential paths."""
        print_wizard_title("Path Configuration")
        paths = self.current_config.setdefault('paths', {})

        # Define default paths relative to project root for clarity
        default_models = paths.get('models_folder') or (DEFAULT_PERSONAL_DIR / "models")
        default_pers_bindings = paths.get('personal_bindings_folder') or (DEFAULT_PERSONAL_DIR / "personal_bindings")
        default_pers_pers = paths.get('personal_personalities_folder') or (DEFAULT_PERSONAL_DIR / "personal_personalities")
        default_pers_funcs = paths.get('personal_functions_folder') or (DEFAULT_PERSONAL_DIR / "personal_functions")


        paths['models_folder'] = str(prompt_for_path(
            "Enter folder for AI models (e.g., GGUF, Diffusers):",
            Path(default_models),
            create_if_not_exist=True
        ))
        paths['personal_bindings_folder'] = str(prompt_for_path(
            "Enter folder for your custom bindings (.py files):",
            Path(default_pers_bindings),
            create_if_not_exist=True
        ))
        paths['personal_personalities_folder'] = str(prompt_for_path(
            "Enter folder for your custom personalities (subfolders):",
            Path(default_pers_pers),
            create_if_not_exist=True
        ))
        paths['personal_functions_folder'] = str(prompt_for_path(
            "Enter folder for your custom functions (.py files):",
            Path(default_pers_funcs),
            create_if_not_exist=True
        ))

        if ASCIIColors.confirm("Use the built-in examples ('zoos' folder) for bindings/personalities/functions?", default_yes=True):
            paths['example_bindings_folder'] = str(DEFAULT_ZOOS_DIR / "bindings")
            paths['example_personalities_folder'] = str(DEFAULT_ZOOS_DIR / "personalities")
            paths['example_functions_folder'] = str(DEFAULT_ZOOS_DIR / "functions")
        else:
            # Explicitly set to None if not used
            paths['example_bindings_folder'] = None
            paths['example_personalities_folder'] = None
            paths['example_functions_folder'] = None

        ASCIIColors.success("Paths configuration updated.")
        # Invalidate discovery caches as paths might have changed
        self.binding_files = {}
        self.available_personalities = {}
        self.loaded_binding_metadata = {}
        self.binding_type_to_path = {}
        ASCIIColors.prompt("Press Enter to return to the main menu...") # User feedback
        return None # Indicate success to main loop (though loop removed)

    # --- Security Actions ---
    def _list_api_keys(self):
        """Lists configured API keys."""
        keys = self.current_config.get('security', {}).get('allowed_api_keys', [])
        if not keys:
            ASCIIColors.info("No API keys currently configured.")
            ASCIIColors.prompt("Press Enter to return...")
            return None # Return to security menu

        ASCIIColors.print("\nCurrent API Keys:", color=ASCIIColors.color_yellow)
        for i, key in enumerate(keys):
            # Obscure the key for display
            obscured = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else key[:4] + "..."
            ASCIIColors.print(f" {i+1}: {obscured}")

        ASCIIColors.prompt("\nPress Enter to return to the security menu...")
        return None # Stay in the security menu (handled by its run loop)

    def _add_api_key(self):
        """Prompts to add a new API key."""
        # Ensure the security structure exists
        sec_config = self.current_config.setdefault('security', {})
        keys = sec_config.setdefault('allowed_api_keys', [])

        while True:
            key = ASCIIColors.prompt("Enter new API key (leave empty to cancel): ").strip()
            if not key:
                ASCIIColors.info("Cancelled adding key.")
                return None # Return to security menu

            if len(key) < 10:
                ASCIIColors.warning("Warning: Key seems short. Continue anyway?")
                if not ASCIIColors.confirm("", default_yes=True):
                    continue # Ask for key again

            if key in keys:
                ASCIIColors.warning("Key already exists. Please enter a unique key.")
            else:
                keys.append(key)
                ASCIIColors.success("API Key added successfully.")
                return None # Return to security menu

    def _select_key_to_modify(self, action_name: str) -> Optional[int]:
        """Helper menu to select an API key by index for modification/removal."""
        keys = self.current_config.get('security', {}).get('allowed_api_keys', [])
        if not keys:
            ASCIIColors.info(f"No API keys available to {action_name}.")
            ASCIIColors.prompt("Press Enter to return...")
            return None

        # Create a temporary menu for this specific selection task
        menu = Menu(f"Select Key to {action_name}", title_color=ASCIIColors.color_yellow, parent=None) # No parent needed here
        for i, key in enumerate(keys):
            obscured = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else key[:4] + "..."
            # Action returns the index when selected
            menu.add_action(f"{i+1}: {obscured}", lambda idx=i: idx)
        # Cancel action returns None
        # menu.add_action("Cancel", lambda: None) # Back/Quit handles cancellation

        # Run the temporary selection menu
        selected_index = menu.run() # Returns the index or None if user Quits/Backs out

        # Check if a valid index was returned (not None)
        if selected_index is None or not isinstance(selected_index, int):
             ASCIIColors.info(f"{action_name} cancelled.")
             return None

        return selected_index

    def _edit_api_key(self):
        """Action to edit an existing API key."""
        selected_index = self._select_key_to_modify("Edit")
        if selected_index is None:
            # Message already printed by _select_key_to_modify if cancelled
            return None # Return to security menu

        keys = self.current_config['security']['allowed_api_keys']
        old_key = keys[selected_index]
        old_key_obscured = f"{old_key[:4]}...{old_key[-4:]}" if len(old_key) > 8 else old_key[:4] + "..."

        while True:
            new_key = ASCIIColors.prompt(f"Enter new value for key {selected_index + 1} ('{old_key_obscured}'): ").strip()
            if not new_key:
                ASCIIColors.warning("Key cannot be empty. Enter the new value or Ctrl+C to cancel edit.")
                continue # Ask again

            if len(new_key) < 10:
                ASCIIColors.warning("Warning: New key seems short. Use it anyway?")
                if not ASCIIColors.confirm("", default_yes=True):
                     continue # Ask for key again

            # Check if the new key conflicts with *other* existing keys
            if new_key in keys and new_key != old_key:
                ASCIIColors.warning("This key value already exists elsewhere. Please enter a unique key.")
                continue

            # Update the key
            keys[selected_index] = new_key
            ASCIIColors.success(f"Key {selected_index + 1} updated successfully.")
            return None # Return to security menu

    def _remove_api_key(self):
        """Action to remove an existing API key."""
        selected_index = self._select_key_to_modify("Remove")
        if selected_index is None:
            # Message already printed by _select_key_to_modify if cancelled
            return None # Return to security menu

        keys = self.current_config['security']['allowed_api_keys']
        key_to_remove = keys[selected_index]
        key_obscured = f"{key_to_remove[:4]}...{key_to_remove[-4:]}" if len(key_to_remove) > 8 else key_to_remove[:4] + "..."

        if ASCIIColors.confirm(f"Are you sure you want to remove key {selected_index + 1} ('{key_obscured}')?", default_yes=False):
            del keys[selected_index]
            ASCIIColors.success(f"Key {selected_index + 1} removed successfully.")
        else:
            ASCIIColors.info("Removal cancelled.")
        return None # Return to security menu

    # --- Discovery and Metadata Helpers ---
    def _discover_all(self) -> bool:
        """Discovers components if not already cached. Returns True on success."""
        # Only rediscover if caches are empty
        if not self.binding_files or not self.available_personalities:
            logger.info("Performing component discovery...")
            paths = self.current_config.get('paths')
            if not paths:
                logger.error("Cannot discover components: Paths section is missing in configuration.")
                return False # Indicate failure

            # Ensure paths are Path objects, handle None or empty strings
            pb_path = Path(paths.get('personal_bindings_folder')) if paths.get('personal_bindings_folder') else None
            eb_path = Path(paths.get('example_bindings_folder')) if paths.get('example_bindings_folder') else None
            pp_path = Path(paths.get('personal_personalities_folder')) if paths.get('personal_personalities_folder') else None
            ep_path = Path(paths.get('example_personalities_folder')) if paths.get('example_personalities_folder') else None

            # Use execute_with_animation for feedback during potentially slow discovery
            def scan_task():
                self.binding_files = discover_component_files("bindings", pb_path, eb_path)
                self.available_personalities = discover_component_files("personalities", pp_path, ep_path)
            try:
                 ASCIIColors.execute_with_animation(
                     "Scanning bindings/personalities folders...",
                     scan_task,
                     color=ASCIIColors.color_cyan
                 )
                 logger.info(f"Discovery finished: Found {len(self.binding_files)} bindings, {len(self.available_personalities)} personalities.")
            except Exception as e:
                 logger.error(f"Error during component discovery: {e}")
                 trace_exception(e)
                 ASCIIColors.error("Component discovery failed.")
                 return False # Indicate failure

        # Return True if discovery ran or caches were already populated
        return True

    def _get_metadata_for_type(self, type_name: str) -> Optional[Dict[str, Any]]:
        """Gets (loads if needed) metadata for a specific binding type_name."""
        # Check cache first
        if type_name in self.loaded_binding_metadata:
            return self.loaded_binding_metadata[type_name]

        # If not cached by type_name, try to find its path and load
        binding_path = self.binding_type_to_path.get(type_name)
        if binding_path and binding_path.exists():
            meta = get_binding_metadata(binding_path)
            self.loaded_binding_metadata[type_name] = meta # Cache result (even if None)
            return meta
        else:
            # If path not mapped, iterate through all discovered files to find it
            logger.warning(f"Metadata for '{type_name}' was not pre-mapped. Searching all binding files...")
            found_meta = None
            for file_stem, path in self.binding_files.items():
                 # Avoid reloading if already cached under a different key (shouldn't happen often)
                 already_loaded = False
                 for loaded_type, loaded_path in self.binding_type_to_path.items():
                     if loaded_path == path:
                         already_loaded = True
                         if loaded_type == type_name:
                             found_meta = self.loaded_binding_metadata.get(loaded_type)
                         break
                 if already_loaded and found_meta: break # Found it via existing mapping

                 if not already_loaded:
                     meta = get_binding_metadata(path)
                     if meta:
                         loaded_type = meta.get('type_name')
                         if loaded_type:
                             # Cache loaded metadata and path mapping
                             self.loaded_binding_metadata[loaded_type] = meta
                             self.binding_type_to_path[loaded_type] = path
                             if loaded_type == type_name:
                                 found_meta = meta # Found the one we were looking for
                                 break # Stop searching once found
            # Cache the result for type_name, even if not found (as None)
            self.loaded_binding_metadata[type_name] = found_meta
            if not found_meta:
                 logger.error(f"Could not find or load metadata for binding type: {type_name}")
            return found_meta


    def _get_metadata_for_stem(self, stem: str) -> Optional[Dict[str, Any]]:
        """Gets (loads if needed) metadata for a specific binding file stem."""
        binding_path = self.binding_files.get(stem)
        if not binding_path:
            logger.error(f"No path found for binding file stem '{stem}'. Ensure discovery ran.")
            return None

        # Check cache: see if this path is already mapped to a type_name
        cached_meta = None
        for type_name, path_in_map in self.binding_type_to_path.items():
            if path_in_map == binding_path:
                # Found the type name associated with this path
                cached_meta = self.loaded_binding_metadata.get(type_name)
                if cached_meta:
                    logger.debug(f"Using cached metadata for stem '{stem}' (type: {type_name})")
                    return cached_meta
                else:
                    # Path was mapped but metadata is None/missing in cache? Should reload.
                    logger.warning(f"Path for stem '{stem}' mapped but metadata missing in cache. Reloading.")
                    break # Exit loop to force reload below

        # Not cached or cache was invalid, load it now
        logger.debug(f"Loading metadata directly for stem '{stem}' from path: {binding_path}")
        meta = get_binding_metadata(binding_path)
        if meta:
            type_name = meta.get('type_name')
            if type_name:
                # Update cache with the loaded metadata and path mapping
                self.loaded_binding_metadata[type_name] = meta
                self.binding_type_to_path[type_name] = binding_path
            else:
                 logger.warning(f"Metadata loaded for {stem}.py, but it's missing 'type_name'.")
        return meta

    # --- Binding Actions ---
    def _add_binding_instance(self):
        """Action handler to add a new binding instance."""
        if not self.binding_files:
            # This check is slightly redundant due to the build logic, but safe
            ASCIIColors.warning("No binding files available to configure.")
            ASCIIColors.prompt("Press Enter to return...")
            return None # Return to bindings menu

        # --- Menu to Choose Binding File Stem ---
        type_menu = Menu("Select Binding File to Add Instance For", item_color=ASCIIColors.color_cyan, selected_background=ASCIIColors.color_bg_blue, parent=None)
        sorted_stems = sorted(self.binding_files.keys())
        for stem in sorted_stems:
            # Action returns the selected stem
            type_menu.add_action(f"{stem}.py", lambda s=stem: s)
        # Cancel action returns None (handled by menu Quit/Back)

        selected_stem = type_menu.run() # Run selection menu
        if selected_stem is None or not isinstance(selected_stem, str):
            ASCIIColors.info("Cancelled adding binding instance.")
            return None # Return to bindings menu if cancelled

        # --- Load Metadata ---
        ASCIIColors.info(f"Loading metadata for {selected_stem}.py...")
        binding_meta = self._get_metadata_for_stem(selected_stem)
        if not binding_meta:
            ASCIIColors.error(f"Could not load valid metadata for {selected_stem}.py.")
            ASCIIColors.prompt("Press Enter to return...")
            return None # Return to bindings menu

        selected_type = binding_meta.get('type_name')
        if not selected_type:
            # Should have been caught by _get_metadata_for_stem warning, but double-check
            ASCIIColors.error(f"Binding file {selected_stem}.py metadata is missing 'type_name'. Cannot configure.")
            ASCIIColors.prompt("Press Enter to return...")
            return None

        template = binding_meta.get('config_template', {})
        ASCIIColors.info(f"\nConfiguring a new '{selected_type}' binding instance (from {selected_stem}.py)")
        ASCIIColors.print(f"Description: {binding_meta.get('description', 'N/A')}")

        # --- Get Instance Name (No Menu Needed) ---
        bindings_section = self.current_config.setdefault('bindings', {})
        while True:
            instance_name = ASCIIColors.prompt(f"Enter a unique name for this '{selected_type}' instance: ").strip()
            if not instance_name:
                ASCIIColors.warning("Instance name cannot be empty.")
                continue
            if instance_name in bindings_section:
                ASCIIColors.warning(f"An instance named '{instance_name}' already exists. Please choose a unique name.")
                continue
            break

        # --- Configure Parameters ---
        instance_config = {"type": selected_type} # Start with the type
        ASCIIColors.info(f"\nConfiguring parameters for instance '{instance_name}':")

        # Iterate through parameters defined in the template
        param_cancelled = False
        sorted_template_keys = sorted([k for k in template.keys() if k != 'type']) # Sort keys for consistent order

        for key in sorted_template_keys:
            param_info = template[key]
            # Parse info needed for prompting
            prompt_text, param_type, required, default, options = self._parse_param_info(key, param_info)

            # Prompt for the value (might involve a sub-menu for options)
            user_value = self._prompt_for_param_value(key, prompt_text, param_type, required, default, options)

            # Handle cancellation from within _prompt_for_param_value
            if user_value == "##CANCEL_PARAM##":
                 if required:
                     ASCIIColors.warning(f"Parameter '{key}' is required. Cancelling configuration for instance '{instance_name}'.")
                     param_cancelled = True
                     break # Stop configuring this instance
                 else:
                     ASCIIColors.info(f"Skipping optional parameter '{key}'.")
                     continue # Skip to next parameter

            # Store the value if it's required, or if it differs from the default,
            # or if it was explicitly provided when the default was None.
            # This avoids storing default values unnecessarily unless they were required.
            if required or user_value != default or (user_value is not None and default is None):
                instance_config[key] = user_value
            elif user_value is None and required: # Double-check for required empty input
                 ASCIIColors.warning(f"Required parameter '{key}' was left empty. Cancelling instance configuration.")
                 param_cancelled = True
                 break

        # Only add the instance if configuration wasn't cancelled
        if not param_cancelled:
            bindings_section[instance_name] = instance_config
            ASCIIColors.success(f"Binding instance '{instance_name}' added successfully.")
        else:
            ASCIIColors.warning(f"Configuration for '{instance_name}' cancelled.")

        ASCIIColors.prompt("Press Enter to return to the bindings menu...")
        return None # Always return to bindings menu after attempt

    def _parse_param_info(self, key: str, param_info: Any) -> Tuple[str, str, bool, Any, Optional[List[Any]]]:
        """Helper to parse parameter info from binding's config_template."""
        prompt_text = f"Parameter: '{key}'"
        default, required, param_type, options, description = None, False, 'string', None, None

        if isinstance(param_info, dict):
            description = param_info.get('description')
            prompt_text += f" - {description}" if description else ""
            required = param_info.get('required', False)
            default = param_info.get('default', param_info.get('value')) # Allow 'default' or 'value'
            param_type = param_info.get('type', 'string').lower() # Normalize type
            options = param_info.get('options')
        else:
            # Handle simpler key: value case (assume string, optional, value is default)
            default = param_info
            param_type = 'string'
            required = False

        # Append type and requirement/default info to prompt
        prompt_text += f" [{param_type}]"
        if required:
             prompt_text += " (Required)"
        elif default is not None:
             # Represent default nicely
             default_str = repr(default)
             if len(default_str) > 30: default_str = default_str[:27] + "..."
             prompt_text += f" (Optional, Default: {default_str})"
        else:
             prompt_text += " (Optional)"

        return prompt_text, param_type, required, default, options

    def _prompt_for_param_value(self, key, prompt_text, param_type, required, default, options):
        """Handles prompting for a single parameter value. Uses a temporary menu for options."""
        user_value = default # Start with default assumption

        if options and isinstance(options, list):
            # --- Use Temporary Menu for Options ---
            opt_menu = Menu(f"Select value for '{key}'", selected_background=ASCIIColors.color_bg_green, parent=None)
            opt_menu.add_info(prompt_text) # Display the prompt text as info

            # Add options as actions that return the option value
            for opt in options:
                opt_menu.add_action(str(opt), lambda o=opt: o)

            # Add default/cancel options if applicable
            if not required:
                 default_display = repr(default)
                 if len(default_display) > 20: default_display = default_display[:17]+"..."
                 opt_menu.add_action(f"(Use Default: {default_display})", lambda: "##USE_DEFAULT##", item_color=ASCIIColors.color_yellow)

            # Cancel returns a specific marker
            opt_menu.add_action("(Cancel Parameter Entry)", lambda: "##CANCEL_PARAM##", item_color=ASCIIColors.color_red)

            selected = opt_menu.run() # Run the temporary menu

            if selected == "##CANCEL_PARAM##":
                return "##CANCEL_PARAM##" # Propagate cancellation signal
            elif selected == "##USE_DEFAULT##":
                user_value = default # Explicitly choose default
            elif selected is not None: # User selected an actual option
                # Coerce type if needed based on options (e.g., if options were numbers)
                try:
                    if param_type == 'int': user_value = int(selected)
                    elif param_type == 'float': user_value = float(selected)
                    elif param_type == 'bool': user_value = str(selected).lower() in ('true', '1', 'yes', 'y')
                    else: user_value = selected # Assume string or correct type already
                except (ValueError, TypeError):
                     ASCIIColors.warning(f"Could not convert selected option '{selected}' to {param_type}. Using raw value.")
                     user_value = selected # Fallback to raw selected value
            else: # Menu exited via Quit/Back without selection
                 return "##CANCEL_PARAM##" # Treat Quit/Back as cancel here

        # --- Non-menu input logic ---
        elif param_type == "bool":
            # Confirm takes default_yes (bool or None)
            confirm_default = None
            if isinstance(default, bool):
                 confirm_default = default
            user_value = ASCIIColors.confirm(f"{prompt_text}", default_yes=confirm_default)
            # Handle required boolean needing explicit answer
            if user_value is None and required: # Should not happen with current confirm logic
                 ASCIIColors.warning("Response required for boolean parameter.")
                 # Loop or treat as cancel? For now, proceed with None->False default
                 user_value = False

        elif param_type == "int":
            while True:
                default_prompt = f" (Default: {default})" if default is not None else ""
                user_input = ASCIIColors.prompt(f"{prompt_text}{default_prompt}: ").strip()
                if not user_input and default is not None: user_value = default; break
                if not user_input and required: ASCIIColors.warning("Input is required."); continue
                if not user_input and not required: user_value = None; break # Allow empty optional int
                try:
                    user_value = int(user_input); break
                except ValueError:
                    ASCIIColors.warning("Invalid integer. Please try again.")

        elif param_type == "float":
            while True:
                default_prompt = f" (Default: {default})" if default is not None else ""
                user_input = ASCIIColors.prompt(f"{prompt_text}{default_prompt}: ").strip()
                if not user_input and default is not None: user_value = default; break
                if not user_input and required: ASCIIColors.warning("Input is required."); continue
                if not user_input and not required: user_value = None; break # Allow empty optional float
                try:
                    user_value = float(user_input); break
                except ValueError:
                    ASCIIColors.warning("Invalid float (number). Please try again.")

        elif param_type == "list":
            ASCIIColors.print(f"{prompt_text}. Enter items one per line. Press Enter on an empty line to finish.", color=ASCIIColors.color_yellow)
            user_list = []
            item_count = 0
            while True:
                item_count += 1
                item = ASCIIColors.prompt(f"  - Item {item_count} for '{key}': ").strip()
                if not item: break
                user_list.append(item)

            if user_list:
                user_value = user_list
            elif required:
                ASCIIColors.warning(f"At least one item is required for list '{key}'.")
                user_value = [] # Set to empty list, but maybe should signal cancel?
                return "##CANCEL_PARAM##" # Signal cancellation for required list
            else: # Optional and empty
                 user_value = default if default is not None else [] # Use default or empty list


        elif param_type == "dict":
            ASCIIColors.print(f"{prompt_text}. Enter 'key=value' pairs one per line. Press Enter on an empty line to finish.", color=ASCIIColors.color_yellow)
            user_dict = {}
            item_count = 0
            while True:
                item_count += 1
                item = ASCIIColors.prompt(f"  - Entry {item_count} for '{key}' (key=value): ").strip()
                if not item: break
                if '=' not in item:
                    ASCIIColors.warning("Invalid format. Please use 'key=value'. Entry skipped.")
                    continue
                k, v = item.split('=', 1)
                user_dict[k.strip()] = v.strip() # Basic parsing

            if user_dict:
                user_value = user_dict
            elif required:
                 ASCIIColors.warning(f"At least one entry is required for dictionary '{key}'.")
                 user_value = {}
                 return "##CANCEL_PARAM##" # Signal cancellation for required dict
            else: # Optional and empty
                 user_value = default if default is not None else {}

        else: # Default to string input
            default_prompt = f" (Default: '{default}')" if default is not None else ""
            user_input = ASCIIColors.prompt(f"{prompt_text}{default_prompt}: ").strip()
            if user_input:
                user_value = user_input
            elif required:
                # If required and user entered nothing, keep prompting or cancel?
                # For simplicity here, use default if available, else force non-empty
                if default is not None:
                     user_value = default
                     ASCIIColors.info(f"Using default value for required parameter '{key}': {default}")
                else:
                     ASCIIColors.warning(f"Input is required for '{key}'.")
                     # Loop until input given or treat as cancel? Treat as cancel.
                     return "##CANCEL_PARAM##"
            elif default is not None:
                 user_value = default # Entered nothing, use default
            else:
                 user_value = None # Optional, entered nothing, no default -> None

        return user_value

    def _modify_binding_instance(self):
        """Action handler to modify an existing binding instance."""
        bindings_section = self.current_config.get('bindings', {})
        if not bindings_section:
            ASCIIColors.warning("No binding instances configured yet to modify.")
            ASCIIColors.prompt("Press Enter to return...")
            return None # Return to bindings menu

        # --- Menu to Select Instance ---
        menu = Menu("Select Binding Instance to Modify", title_color=ASCIIColors.color_yellow, parent=None)
        sorted_names = sorted(bindings_section.keys())
        for name in sorted_names:
            # Action returns the instance name
            menu.add_action(name, lambda n=name: n)
        # Cancel handled by menu Quit/Back

        instance_name_to_modify = menu.run()
        if instance_name_to_modify is None or not isinstance(instance_name_to_modify, str):
            ASCIIColors.info("Cancelled modifying binding instance.")
            return None # Return to bindings menu

        # --- Parameter Modification Logic ---
        instance_config = bindings_section[instance_name_to_modify]
        selected_type = instance_config.get("type")
        if not selected_type:
            ASCIIColors.error(f"Instance '{instance_name_to_modify}' is missing 'type'. Cannot modify parameters accurately.")
            ASCIIColors.prompt("Press Enter to return...")
            return None

        ASCIIColors.info(f"Loading metadata for type '{selected_type}' to guide modification...")
        binding_meta = self._get_metadata_for_type(selected_type)
        template = binding_meta.get('config_template', {}) if binding_meta else {}
        if not binding_meta:
            ASCIIColors.warning(f"Could not load metadata for type '{selected_type}'. Parameter descriptions/types might be inaccurate.")

        ASCIIColors.info(f"\nModifying instance '{instance_name_to_modify}' (Type: {selected_type}):")
        ASCIIColors.print("(Leave input empty to keep current value, unless required)")

        # Combine keys from template and existing config for completeness
        all_keys = set(template.keys()) | set(instance_config.keys())
        all_keys.discard("type") # Cannot change the type
        new_config = {"type": selected_type} # Start building the new config
        param_cancelled = False

        for key in sorted(list(all_keys)):
            param_info = template.get(key, {}) # Get info from template if available
            current_value = instance_config.get(key) # Get current value

            # Parse info, primarily for type and options
            prompt_text, param_type, required_in_template, _, options = self._parse_param_info(key, param_info)
            # Modify prompt for editing context
            prompt_text = prompt_text.split('(')[0].strip() # Get base prompt
            current_display = repr(current_value)
            if len(current_display) > 30: current_display = current_display[:27]+"..."
            prompt_text += f" [{param_type}] (Current: {current_display})"

            # Use the prompt helper. Treat as optional (required=False) because we want
            # the user to be able to skip/keep the current value easily.
            # Pass current_value as the default for the prompt.
            user_value = self._prompt_for_param_value(key, prompt_text, param_type, False, current_value, options)

            if user_value == "##CANCEL_PARAM##":
                 ASCIIColors.warning(f"Modification cancelled for parameter '{key}'. Keeping current value: {current_value}")
                 # Keep the existing value if modification cancelled
                 if current_value is not None:
                     new_config[key] = current_value
                 continue # Move to next parameter

            # Store the value if it was explicitly changed or if it's different from None when current was None
            if user_value != current_value:
                new_config[key] = user_value
                ASCIIColors.print(f"  ✓ Set '{key}' to: {user_value}", color=ASCIIColors.color_green)
            elif current_value is not None: # Value didn't change, keep existing non-None value
                new_config[key] = current_value
            # If user_value == current_value and current_value was None, it remains omitted


        # Update the configuration only if not cancelled midway (though cancellation now keeps current value)
        self.current_config['bindings'][instance_name_to_modify] = new_config
        ASCIIColors.success(f"Instance '{instance_name_to_modify}' updated.")

        ASCIIColors.prompt("Press Enter to return to the bindings menu...")
        return None # Always return to bindings menu

    def _remove_binding_instance(self):
        """Action handler to remove a binding instance."""
        bindings_section = self.current_config.get('bindings', {})
        if not bindings_section:
            ASCIIColors.warning("No binding instances configured yet to remove.")
            ASCIIColors.prompt("Press Enter to return...")
            return None # Return to bindings menu

        # --- Menu to Select Instance ---
        menu = Menu("Select Binding Instance to Remove", item_color=ASCIIColors.color_red, selected_background=ASCIIColors.color_bg_red, parent=None)
        sorted_names = sorted(bindings_section.keys())
        for name in sorted_names:
             menu.add_action(name, lambda n=name: n)
        # Cancel handled by menu Quit/Back

        instance_name_to_remove = menu.run()
        if instance_name_to_remove is None or not isinstance(instance_name_to_remove, str):
            ASCIIColors.info("Cancelled removing binding instance.")
            return None # Return to bindings menu

        # --- Confirmation and Deletion ---
        if ASCIIColors.confirm(f"Are you sure you want to remove instance '{instance_name_to_remove}'?", default_yes=False):
            del bindings_section[instance_name_to_remove]
            ASCIIColors.success(f"Instance '{instance_name_to_remove}' removed.")

            # Check and update defaults if the removed instance was used
            defaults = self.current_config.get('defaults', {})
            updated_defaults = False
            for key, value in list(defaults.items()):
                if value == instance_name_to_remove:
                    defaults[key] = None
                    ASCIIColors.warning(f"Removed '{instance_name_to_remove}' as default for '{key}'. Please set a new default if needed.")
                    updated_defaults = True
            if updated_defaults:
                 ASCIIColors.info("Check 'Default Bindings' menu to set new defaults.")
        else:
             ASCIIColors.info("Removal cancelled.")

        ASCIIColors.prompt("Press Enter to return to the bindings menu...")
        return None # Always return to bindings menu


    # --- Personality Actions ---
    def _toggle_personality(self, name: str):
        """Action handler to toggle the enabled state of a personality."""
        pers_config = self.current_config.setdefault('personalities_config', {})
        instance_cfg = pers_config.setdefault(name, {})
        # Default to enabled=True if the key doesn't exist
        current_state = instance_cfg.get('enabled', True)
        instance_cfg['enabled'] = not current_state
        ASCIIColors.info(f"Personality '{name}' set to {'ENABLED' if instance_cfg['enabled'] else 'DISABLED'}.")
        # The menu will be rebuilt automatically next time it's entered via the wrapper
        return None # Return to personality menu


    # --- configure_defaults (Action, uses menus internally) ---
    def configure_defaults(self):
        """Action to configure default bindings for different modalities."""
        print_wizard_title("Default Binding Configuration")
        defaults = self.current_config.setdefault('defaults', {})
        bindings_section = self.current_config.get('bindings', {})

        if not bindings_section:
            ASCIIColors.warning("No binding instances have been configured yet.")
            ASCIIColors.warning("Please configure instances under 'Binding Configuration' first.")
            ASCIIColors.prompt("Press Enter to return to the main menu...")
            return None # Back to main menu

        configured_instance_names = sorted(list(bindings_section.keys()))

        # Define the default keys to configure
        default_keys_info = {
            'ttt_binding': "Text-to-Text (Main LLM)",
            'tti_binding': "Text-to-Image",
            'tts_binding': "Text-to-Speech",
            'stt_binding': "Speech-to-Text",
            'ttv_binding': "Text-to-Video",
            'ttm_binding': "Text-to-Music",
            # Add other modalities as needed
        }

        for default_key, description in default_keys_info.items():
            current_default = defaults.get(default_key)

            # --- Temporary Menu to Select Default ---
            menu = Menu(
                f"Select default for {description} (Current: {current_default or 'None'})",
                selected_background=ASCIIColors.color_bg_green, parent=None
            )
            menu.add_info(f"Choose the default binding instance for {description}:")

            # Option to explicitly set to None
            menu.add_action("(Set to None / No Default)", lambda: "##NONE##", item_color=ASCIIColors.color_yellow)

            # Add configured binding instances as options
            for name in configured_instance_names:
                # Action returns the instance name
                menu.add_action(name, lambda n=name: n)

            # Add a skip/cancel option
            menu.add_action("(Keep Current / Cancel)", lambda: "##CANCEL##", item_color=ASCIIColors.color_red)

            selected_instance = menu.run() # Run temporary menu

            # Process selection
            if selected_instance == "##CANCEL##":
                ASCIIColors.info(f"Keeping current default for {default_key} as: {current_default or 'None'}")
            elif selected_instance == "##NONE##":
                if current_default is not None:
                    defaults[default_key] = None
                    ASCIIColors.info(f"Set default for {default_key} to: None")
                else:
                    ASCIIColors.info(f"Default for {default_key} remains: None")
            elif selected_instance is not None and isinstance(selected_instance, str): # User selected an instance name
                if defaults.get(default_key) != selected_instance:
                    defaults[default_key] = selected_instance
                    ASCIIColors.info(f"Set default for {default_key} to: {selected_instance}")
                else:
                     ASCIIColors.info(f"Default for {default_key} remains: {selected_instance}")
            # else: Menu Quit/Back treated same as Cancel

        ASCIIColors.success("\nDefault bindings configuration finished.")
        ASCIIColors.prompt("Press Enter to return to the main menu...")
        return None # Back to main menu

    # --- configure_webui (Action, no menu needed) ---
    def configure_webui(self):
        """Action to enable/disable the Web UI."""
        print_wizard_title("Web UI Configuration")
        webui_config = self.current_config.setdefault('webui', {"enable_ui": False})
        # Ensure key exists with default if section was empty
        current_state = webui_config.setdefault('enable_ui', False)

        ASCIIColors.print("The Web UI allows interaction with lollms_server via a web browser.")
        ASCIIColors.print("Note: Requires building the UI first (cd web; npm install; npm run build)")

        enable = ASCIIColors.confirm("Enable the Web UI?", default_yes=current_state)
        webui_config['enable_ui'] = enable

        ASCIIColors.success(f"Web UI configuration set to: {'ENABLED' if enable else 'DISABLED'}")
        ASCIIColors.prompt("Press Enter to return to the main menu...")
        return None # Back to main menu

    # --- review_and_save (Action, no menu needed) ---
    def review_and_save(self):
        """Action to review the current configuration and prompt for saving."""
        print_wizard_title("Review Configuration")
        try:
            # Ensure essential sections exist before dumping
            self.current_config.setdefault('paths', {})
            self.current_config.setdefault('security', {}).setdefault('allowed_api_keys', [])
            self.current_config.setdefault('bindings', {})
            self.current_config.setdefault('personalities_config', {})
            self.current_config.setdefault('defaults', {})
            self.current_config.setdefault('webui', {}).setdefault('enable_ui', False)

            config_str = toml.dumps(self.current_config)
            ASCIIColors.print("\n--- Current Configuration ---", color=ASCIIColors.color_bright_yellow)
            # Use standard print for potentially long config dump
            print(config_str)
            ASCIIColors.print("--- End Configuration ---\n", color=ASCIIColors.color_bright_yellow)

            if ASCIIColors.confirm("Save this configuration to config.toml?", default_yes=True):
                 if save_config(self.current_config):
                      self._saved = True # Set flag for check AFTER wizard.run() finishes
                      ASCIIColors.info("\nConfiguration saved. You can now run the server.")
                      ASCIIColors.info("Next steps: Run './run.sh' or '.\\run.bat'.")
                      # Optionally force exit from the menu? No, let user Quit normally.
                 else:
                      ASCIIColors.error("Failed to save configuration. Please check permissions or errors above.")
            else:
                ASCIIColors.warning("Configuration review complete. Changes NOT saved.")

        except Exception as e:
            logger.error(f"Error displaying or processing configuration for review: {e}")
            trace_exception(e)
            ASCIIColors.error(f"An error occurred during configuration review: {e}")

        ASCIIColors.prompt("Press Enter to return to the main menu...")
        return None # Always return to main menu after review/save attempt

# --- Run the Wizard ---
if __name__ == "__main__":
    # Basic check to ensure script is run from a plausible project root
    if not (project_root / "lollms_server").is_dir() or not (project_root / "zoos").is_dir():
        # Use direct print as logging might not be fully set up if imports failed
        ASCIIColors.error("ERROR: Please run this script from the main 'lollms_server' project directory.")
        ASCIIColors.error(f"(Detected project root: {project_root})")
        sys.exit(1)

    wizard = ConfigWizard()
    try:
        wizard.run() # Call run just once - it handles the main interaction loop
    except KeyboardInterrupt:
        ASCIIColors.warning("\nWizard aborted by user (Ctrl+C).")
        sys.exit(1) # Exit with error code on abort
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during wizard execution: {e}")
        trace_exception(e)
        ASCIIColors.error(f"Wizard failed unexpectedly: {e}")
        sys.exit(1) # Exit with error code on unexpected failure

    # Exit with success code (0) if saved, or error code (1) if not saved
    sys.exit(0)