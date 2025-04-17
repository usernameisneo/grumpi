# tests/utils/test_file_utils.py
import pytest
import sys
from pathlib import Path
import importlib

from lollms_server.utils.file_utils import add_path_to_sys_path, safe_load_module, find_classes_in_module

# --- Test add_path_to_sys_path ---

def test_add_path_to_sys_path_adds_new_path(tmp_path):
    """Test that a new path is added correctly."""
    new_path = tmp_path / "my_new_lib"
    new_path.mkdir()
    abs_path_str = str(new_path.resolve())

    # Ensure it's not already there
    if abs_path_str in sys.path:
        sys.path.remove(abs_path_str)
    assert abs_path_str not in sys.path

    add_path_to_sys_path(new_path)
    assert abs_path_str in sys.path
    # Clean up sys.path
    sys.path.remove(abs_path_str)

def test_add_path_to_sys_path_does_not_duplicate(tmp_path):
    """Test that adding an existing path doesn't duplicate it."""
    existing_path = tmp_path / "existing_lib"
    existing_path.mkdir()
    abs_path_str = str(existing_path.resolve())

    # Add it once
    if abs_path_str not in sys.path:
        sys.path.insert(0, abs_path_str)
    initial_count = sys.path.count(abs_path_str)

    # Add it again using the function
    add_path_to_sys_path(existing_path)
    final_count = sys.path.count(abs_path_str)

    assert final_count == initial_count # Should not increase
    # Clean up sys.path
    if abs_path_str in sys.path:
         # Remove all occurrences just in case test environment is weird
         sys.path = [p for p in sys.path if p != abs_path_str]


# --- Test safe_load_module ---

@pytest.fixture
def dummy_module_file(tmp_path: Path) -> Path:
    """Creates a simple valid Python module file."""
    module_content = """
# Dummy module
MY_VAR = 123

def my_func():
    return MY_VAR

class MyClass:
    pass

class AnotherClass(MyClass):
    pass
"""
    module_path = tmp_path / "dummy_module.py"
    module_path.write_text(module_content)
    return module_path

@pytest.fixture
def syntax_error_module_file(tmp_path: Path) -> Path:
    """Creates a Python module file with a syntax error."""
    module_content = "this is not valid python ="
    module_path = tmp_path / "syntax_error_module.py"
    module_path.write_text(module_content)
    return module_path

@pytest.fixture
def import_error_module_file(tmp_path: Path) -> Path:
    """Creates a Python module file with an import error."""
    module_content = "import non_existent_library_xyz"
    module_path = tmp_path / "import_error_module.py"
    module_path.write_text(module_content)
    return module_path

def test_safe_load_module_success(dummy_module_file: Path):
    """Test loading a valid module."""
    module, error = safe_load_module(dummy_module_file)
    assert error is None
    assert module is not None
    assert hasattr(module, "MY_VAR")
    assert module.MY_VAR == 123
    assert hasattr(module, "my_func")
    assert module.my_func() == 123
    # Clean up sys.modules
    del sys.modules[dummy_module_file.stem]

def test_safe_load_module_syntax_error(syntax_error_module_file: Path):
    """Test loading a module with a syntax error."""
    module, error = safe_load_module(syntax_error_module_file)
    assert module is None
    assert error is not None
    assert "Syntax error" in error

def test_safe_load_module_import_error(import_error_module_file: Path):
    """Test loading a module with an import error."""
    module, error = safe_load_module(import_error_module_file)
    assert module is None
    assert error is not None
    assert "Import error" in error
    assert "non_existent_library_xyz" in error

def test_safe_load_module_non_existent():
    """Test loading a non-existent file."""
    module, error = safe_load_module(Path("non_existent_module.py"))
    assert module is None
    assert error is not None # Should error during spec_from_file_location or exec
    # The exact error might vary, check it contains relevant info
    assert "non_existent_module.py" in error or "Could not create module spec" in error


# --- Test find_classes_in_module ---

@pytest.fixture
def loaded_dummy_module(dummy_module_file: Path):
    """Provides the loaded dummy module for class finding tests."""
    module, _ = safe_load_module(dummy_module_file)
    yield module
    # Cleanup
    del sys.modules[dummy_module_file.stem]


def test_find_classes_in_module(loaded_dummy_module):
    """Test finding classes inheriting from a base class."""
    base_class = loaded_dummy_module.MyClass
    derived_class = loaded_dummy_module.AnotherClass

    found_classes = find_classes_in_module(loaded_dummy_module, base_class)

    assert len(found_classes) == 1
    assert found_classes[0] is derived_class

def test_find_classes_in_module_no_inheritance(loaded_dummy_module):
    """Test finding classes when none inherit from the specified base."""
    class SomeOtherBase: pass

    found_classes = find_classes_in_module(loaded_dummy_module, SomeOtherBase)
    assert len(found_classes) == 0

def test_find_classes_in_module_base_is_object(loaded_dummy_module):
    """Test finding classes inheriting directly from object."""
    # Should find MyClass and AnotherClass
    found_classes = find_classes_in_module(loaded_dummy_module, object)
    # Note: Might also find built-in types if not careful, the implementation
    # seems okay as it checks `obj is not base_class`.
    assert len(found_classes) >= 2 # At least MyClass and AnotherClass
    assert loaded_dummy_module.MyClass in found_classes
    assert loaded_dummy_module.AnotherClass in found_classes