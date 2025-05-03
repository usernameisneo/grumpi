# tests/core/test_config.py
import pytest
from pathlib import Path
import toml
from pydantic import ValidationError
from unittest.mock import patch # Use patch directly

from lollms_server.core.config import load_config, get_config, AppConfig, _resolve_paths, _ensure_directories, PathsConfig

# Use the fixtures defined in conftest.py

@patch('lollms_server.core.config.logger.warning') # Use patch decorator
def test_load_config_non_existent(mock_warning): # Inject mock automatically
    """Test loading config when the file doesn't exist."""
    config_path = Path("non_existent_config.toml")
    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9601
    assert config.security.allowed_api_keys == []
    assert config.paths.personalities_folder.name == "personal_personalities"
    mock_warning.assert_any_call(f"Configuration file not found at {config_path}. Using default settings.")

@patch('lollms_server.core.config.logger.error')
@patch('lollms_server.core.config.logger.warning')
def test_load_config_invalid_toml(mock_warning, mock_error, temp_config_dir): # Order matters for decorators
    """Test loading config with invalid TOML format."""
    config_path = temp_config_dir / "invalid.toml"
    config_path.write_text("this is not valid toml = ")

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.server.host == "0.0.0.0"
    mock_error.assert_called_once()
    mock_warning.assert_any_call("Using default configuration due to error.")


def test_load_config_valid(dummy_config_path: Path):
    """Test loading a valid config file."""
    config = load_config(dummy_config_path)

    assert isinstance(config, AppConfig)
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 9999
    assert config.logging.log_level == "DEBUG"
    assert config.security.allowed_api_keys == ["test_key_1", "test_key_2"]
    assert "test_ollama" in config.bindings
    assert config.bindings["test_ollama"]["type"] == "ollama_binding"
    assert config.paths.personalities_folder.name == "test_pers"
    assert config.paths.personalities_folder.is_absolute()

def test_get_config_singleton(dummy_config_path: Path):
    """Test that get_config returns the loaded singleton."""
    config1 = load_config(dummy_config_path)
    config2 = get_config()
    assert config1 is config2

def test_resolve_paths(temp_config_dir: Path):
    """Test the _resolve_paths helper function."""
    base_dir = temp_config_dir
    # Instantiate directly, not via AppConfig
    relative_paths = PathsConfig(
        personalities_folder=Path("my_pers"),
        models_folder=Path("my_models/"),
        # Leave others as defaults
    )
    _resolve_paths(relative_paths, base_dir)

    assert relative_paths.personalities_folder == (base_dir / "my_pers").resolve()
    assert relative_paths.models_folder == (base_dir / "my_models").resolve()
    # Check a default path (assuming default is relative)
    default_binding_path = PathsConfig().bindings_folder # Get default value
    assert relative_paths.bindings_folder == (base_dir / default_binding_path).resolve()


def test_ensure_directories(temp_config_dir: Path):
    """Test the _ensure_directories helper function."""
    # Create paths config manually for this test
    paths = PathsConfig(
        personalities_folder=temp_config_dir / "p",
        bindings_folder=temp_config_dir / "b",
        functions_folder=temp_config_dir / "f",
        models_folder=temp_config_dir / "m",
        # Example folders can be None or paths
        example_personalities_folder = None,
        example_bindings_folder = None,
        example_functions_folder = None,
    )

    # Ensure they don't exist initially
    assert not paths.personalities_folder.exists()
    assert not paths.bindings_folder.exists()
    assert not paths.functions_folder.exists()
    assert not paths.models_folder.exists()

    _ensure_directories(paths)

    # Check they exist now
    assert paths.personalities_folder.is_dir()
    assert paths.bindings_folder.is_dir()
    assert paths.functions_folder.is_dir()
    assert paths.models_folder.is_dir()
    assert (paths.models_folder / "ttt").is_dir()
    assert (paths.models_folder / "tti").is_dir()
    assert (paths.models_folder / "ttv").is_dir()
    assert (paths.models_folder / "ttm").is_dir()

    # Test idempotency
    _ensure_directories(paths)
    assert paths.models_folder.is_dir()