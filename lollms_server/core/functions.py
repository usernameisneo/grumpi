# lollms_server/core/functions.py
import importlib
import inspect
from pathlib import Path
import ascii_colors as logging
from typing import List, Dict, Any, Callable, Optional, Tuple
import asyncio

from .config import AppConfig
from lollms_server.utils.file_utils import safe_load_module, add_path_to_sys_path

logger = logging.getLogger(__name__)

# --- Function Representation (using Callable for now) ---
# We could define a class later if more metadata is needed per function

# --- Function Manager ---

class FunctionManager:
    """Discovers and manages available custom functions."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._functions: Dict[str, Callable] = {} # Stores discovered functions {name: callable}
        self._load_errors: Dict[str, str] = {} # Stores errors during loading {path/name: error}

    def load_functions(self):
        """Scans configured folders and loads valid functions."""
        logger.info("Loading custom functions...")
        self._functions = {} # Reset
        self._load_errors = {} # Reset

        function_folders = []
        if self.config.paths.example_functions_folder and self.config.paths.example_functions_folder.exists():
            function_folders.append(self.config.paths.example_functions_folder)
            add_path_to_sys_path(self.config.paths.example_functions_folder.parent)
        if self.config.paths.functions_folder and self.config.paths.functions_folder.exists():
            function_folders.append(self.config.paths.functions_folder)
            add_path_to_sys_path(self.config.paths.functions_folder.parent)

        if not function_folders:
            logger.warning("No function folders configured or found.")
            return

        for folder in function_folders:
            logger.info(f"Scanning for functions in: {folder}")
            for file_path in folder.glob("*.py"):
                if file_path.name == "__init__.py":
                    continue
                module, error = safe_load_module(file_path, package_path=folder)
                if module:
                    # Discover functions within the module (e.g., marked with a decorator or by convention)
                    # Convention: Look for top-level async functions?
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and asyncio.iscoroutinefunction(obj):
                            # Let's use function name directly
                            if name.startswith("_"): # Skip private functions
                                    continue

                            func_name = f"{file_path.stem}.{name}" # Namespace by module name

                            if func_name in self._functions:
                                logger.warning(f"Duplicate function name '{func_name}' found. Overwriting with function from {file_path}.")
                            self._functions[func_name] = obj
                            logger.info(f"Discovered function: '{func_name}'")
                        # Add logic here if using decorators or specific naming conventions
                elif error:
                    self._load_errors[str(file_path)] = error

        logger.info(f"Loaded {len(self._functions)} functions.")
        if self._load_errors:
            logger.warning(f"Encountered errors during function loading: {self._load_errors}")


    def list_functions(self) -> List[str]:
        """Returns a list of available function names."""
        return list(self._functions.keys())

    def get_function(self, name: str) -> Optional[Callable]:
        """Gets a loaded function by its name."""
        return self._functions.get(name)

    async def execute_function(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Executes a function by name with given arguments.

        Args:
            name: The name of the function to execute.
            args: A dictionary of arguments to pass to the function.

        Returns:
            A tuple: (success: bool, result: Any). Result is the function's
            return value on success, or an error message/object on failure.
        """
        func = self.get_function(name)
        if not func:
            logger.error(f"Function '{name}' not found.")
            return False, f"Function '{name}' not found."

        if not asyncio.iscoroutinefunction(func):
                logger.error(f"Function '{name}' is not an async function. Only async functions are supported.")
                return False, f"Function '{name}' is not async."

        logger.info(f"Executing function '{name}' with args: {args}")
        try:
            # Inspect function signature to pass only relevant args? (More robust)
            sig = inspect.signature(func)
            valid_args = {k: v for k, v in args.items() if k in sig.parameters}
            missing_args = [p for p in sig.parameters if p not in args and sig.parameters[p].default is inspect.Parameter.empty]
            if missing_args:
                msg = f"Missing required arguments for function '{name}': {', '.join(missing_args)}"
                logger.error(msg)
                return False, msg

            result = await func(**valid_args)
            logger.info(f"Function '{name}' executed successfully.")
            return True, result
        except Exception as e:
            logger.error(f"Error executing function '{name}': {e}", exc_info=True)
            return False, f"Error executing function '{name}': {str(e)}"
