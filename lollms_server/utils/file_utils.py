# lollms_server/utils/file_utils.py
import importlib
import sys
from pathlib import Path
import ascii_colors as logging
from typing import List, Tuple, Type, Any, Optional

logger = logging.getLogger(__name__)

def add_path_to_sys_path(path: Path|str) -> None:
    """Adds a given path to sys.path if not already present."""
    abs_path = str(Path(path).resolve())
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
        logger.debug(f"Added {abs_path} to sys.path")

def safe_load_module(module_path: Path, package_path: Optional[Path] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Safely loads a Python module from a given file path.

    Args:
        module_path: The Path object pointing to the Python file (.py).
        package_path: The Path object pointing to the parent package directory (optional).

    Returns:
        A tuple containing the loaded module object or None if loading failed,
        and an error message string or None if loading succeeded.
    """
    module_name = module_path.stem
    spec = None
    module = None
    error_message = None

    try:
        # If it's part of a package, ensure the package path is in sys.path
        if package_path:
            add_path_to_sys_path(package_path.parent) # Add parent of package folder
            module_name = f"{package_path.name}.{module_name}"

        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Add module to sys.modules BEFORE execution to handle circular imports etc.
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.debug(f"Successfully loaded module: {module_name} from {module_path}")
        else:
            error_message = f"Could not create module spec for {module_path}"
            logger.warning(error_message)

    except ImportError as e:
        error_message = f"Import error loading module {module_name} from {module_path}: {e}"
        logger.error(error_message)
        if module_name in sys.modules: # Clean up failed load
            del sys.modules[module_name]
        module = None
    except SyntaxError as e:
        error_message = f"Syntax error in module {module_name} at {module_path}: {e}"
        logger.error(error_message)
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = None
    except Exception as e:
        error_message = f"Unexpected error loading module {module_name} from {module_path}: {e}"
        logger.error(error_message, exc_info=True)
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = None

    return module, error_message

def find_classes_in_module(module: Any, base_class: Type) -> List[Type]:
    """Finds classes in a module that inherit from a specific base class."""
    classes = []
    for name, obj in module.__dict__.items():
        try:
            if isinstance(obj, type) and issubclass(obj, base_class) and obj is not base_class:
                classes.append(obj)
                logger.debug(f"Found class {name} inheriting from {base_class.__name__} in module {module.__name__}")
        except TypeError: # issubclass might fail on non-type objects
            continue
        except Exception as e:
            logger.warning(f"Error inspecting object {name} in module {module.__name__}: {e}")
    return classes