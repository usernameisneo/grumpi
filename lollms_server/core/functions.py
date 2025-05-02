# encoding:utf-8
# Project: lollms_server
# File: lollms_server/core/functions.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Manages discovery and execution of custom Python functions.

import importlib
import inspect
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

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
if TYPE_CHECKING:
    try:
        from configguard import ConfigGuard # For type hinting
    except ImportError:
        ConfigGuard = Any # type: ignore

# Assuming these utils are correctly defined elsewhere
from lollms_server.utils.file_utils import safe_load_module, add_path_to_sys_path

logger = logging.getLogger(__name__)

# --- Function Manager ---

class FunctionManager:
    """Discovers, manages, and executes available custom Python functions."""

    def __init__(self, main_config: 'ConfigGuard'):
        """
        Initializes the FunctionManager.

        Args:
            main_config: The loaded ConfigGuard object for the main server configuration.
        """
        self.main_config: 'ConfigGuard' = main_config
        # Stores discovered functions {module_stem.function_name: callable}
        self._functions: Dict[str, Callable[..., Any]] = {}
        # Stores errors during loading {path_str: error_message}
        self._load_errors: Dict[str, str] = {}

    def load_functions(self):
        """
        Scans configured function folders (example and personal) and loads valid
        asynchronous Python functions found in .py files (excluding __init__.py).
        Functions are namespaced by their module filename.
        """
        logger.info("Loading custom functions...")
        self._functions = {} # Reset discovered functions
        self._load_errors = {} # Reset load errors

        function_folders_to_scan: List[Path] = []

        # Access paths configuration section from the main ConfigGuard object
        # Paths should already be resolved to absolute by initialize_config
        paths_config = self.main_config.paths

        # Get example functions path
        example_folder_str = getattr(paths_config, "example_functions_folder", None)
        if example_folder_str:
            example_folder = Path(example_folder_str)
            if example_folder.is_dir():
                function_folders_to_scan.append(example_folder)
                # Add parent directory to sys.path for potential relative imports
                add_path_to_sys_path(example_folder.parent)
                logger.debug(f"Added example function parent path: {example_folder.parent}")
            else:
                 logger.warning(f"Example functions folder path specified but not found: {example_folder}")
        else:
            logger.debug("Example functions folder not configured.")


        # Get personal functions path
        personal_folder_str = getattr(paths_config, "functions_folder", None)
        if personal_folder_str:
            personal_folder = Path(personal_folder_str)
            if personal_folder.is_dir():
                # Avoid adding the same path twice if example/personal point to the same place
                if not example_folder or personal_folder != example_folder:
                     function_folders_to_scan.append(personal_folder)
                     # Add parent directory to sys.path
                     add_path_to_sys_path(personal_folder.parent)
                     logger.debug(f"Added personal function parent path: {personal_folder.parent}")
                else: logger.debug("Personal functions folder is same as example folder.")
            else:
                 logger.warning(f"Personal functions folder path specified but not found: {personal_folder}")
        else:
             logger.debug("Personal functions folder not configured.")


        if not function_folders_to_scan:
            logger.warning("No valid function folders configured or found. No custom functions will be loaded.")
            return

        # --- Scan Folders and Load Functions ---
        for folder in function_folders_to_scan:
            logger.info(f"Scanning for functions in: {folder}")
            try:
                for file_path in folder.glob("*.py"):
                    if file_path.name == "__init__.py":
                        continue # Skip __init__.py files

                    logger.debug(f"Attempting to load module from: {file_path}")
                    # Pass the folder as package_path hint for potential relative imports
                    module, error = safe_load_module(file_path, package_path=folder)

                    if module:
                        # Discover async functions within the successfully loaded module
                        function_found_in_module = False
                        for name, obj in inspect.getmembers(module):
                            # Check if it's a function and specifically an async function
                            if inspect.isfunction(obj) and asyncio.iscoroutinefunction(obj):
                                # Skip functions starting with underscore (private convention)
                                if name.startswith("_"):
                                    continue

                                # Namespace the function name with the module filename (without .py)
                                func_name = f"{file_path.stem}.{name}"

                                if func_name in self._functions:
                                    # Prioritize personal folder if names clash
                                    if folder == personal_folder:
                                         logger.warning(f"Duplicate function name '{func_name}' found. Overwriting with function from personal folder: {file_path}")
                                         self._functions[func_name] = obj
                                    else:
                                         logger.warning(f"Duplicate function name '{func_name}' found in example folder '{file_path}'. Skipping (personal folder takes precedence).")
                                else:
                                    self._functions[func_name] = obj
                                    logger.info(f"Discovered function: '{func_name}'")
                                    function_found_in_module = True

                        if not function_found_in_module:
                             logger.debug(f"No suitable async functions found in module: {file_path.name}")

                    elif error:
                        # Store loading error associated with the file path
                        self._load_errors[str(file_path)] = error
                        logger.warning(f"Failed to load function module {file_path.name}: {error}")

            except OSError as e:
                 logger.error(f"OS Error scanning directory {folder}: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error scanning directory {folder}: {e}", exc_info=True)
                 trace_exception(e) # Use trace_exception if available


        logger.info(f"Function loading finished. Loaded {len(self._functions)} functions.")
        if self._load_errors:
            logger.warning(f"Encountered errors during function loading: {self._load_errors}")


    def list_functions(self) -> List[str]:
        """Returns a list of names of the successfully discovered functions."""
        return sorted(list(self._functions.keys()))

    def get_function(self, name: str) -> Optional[Callable[..., Any]]:
        """
        Gets a loaded function by its namespaced name (module_stem.function_name).

        Args:
            name: The namespaced name of the function.

        Returns:
            The callable function object if found, otherwise None.
        """
        func = self._functions.get(name)
        if not func:
             logger.warning(f"Attempted to get unknown function: '{name}'")
        return func

    async def execute_function(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Executes a discovered asynchronous function by its namespaced name.

        Args:
            name: The namespaced name of the function (e.g., 'my_module.do_something').
            args: A dictionary of keyword arguments to pass to the function.

        Returns:
            A tuple: (success: bool, result: Any).
            'result' is the function's return value on success,
            or an error message string on failure (function not found, argument mismatch, execution error).
        """
        func = self.get_function(name)
        if not func:
            error_msg = f"Function '{name}' not found or failed to load."
            logger.error(error_msg)
            return False, error_msg

        # Ensure it's still an async function (should be due to discovery filter)
        if not asyncio.iscoroutinefunction(func):
                error_msg = f"Function '{name}' is not an async function. Execution skipped."
                logger.error(error_msg)
                return False, error_msg

        logger.info(f"Executing function '{name}' with args: {args}")
        try:
            # Inspect function signature to pass only relevant args and check required ones
            sig = inspect.signature(func)
            valid_args = {}
            missing_required_args = []

            for param_name, param in sig.parameters.items():
                if param_name in args:
                    valid_args[param_name] = args[param_name]
                elif param.default is inspect.Parameter.empty:
                    # Argument is required but not provided in args dict
                    missing_required_args.append(param_name)

            if missing_required_args:
                msg = f"Missing required arguments for function '{name}': {', '.join(missing_required_args)}"
                logger.error(msg)
                return False, msg

            # Execute the async function with validated arguments
            result = await func(**valid_args)
            logger.info(f"Function '{name}' executed successfully.")
            return True, result # Return success and the function's result

        except Exception as e:
            # Catch execution errors within the called function
            error_msg = f"Error executing function '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True) # Log full traceback
            trace_exception(e) # Use trace_exception if available
            return False, error_msg # Return failure and error message