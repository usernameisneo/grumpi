# tests/conftest.py
import pytest
import asyncio
from pathlib import Path
import tempfile
import os
import shutil
import toml

# --- Asyncio Setup ---
# Use pytest-asyncio's default event_loop fixture

# --- Temporary Directory Fixtures ---

@pytest.fixture(scope="function") # Create a new temp dir for each test function
def temp_config_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory structure for config tests."""
    personalities = tmp_path / "test_pers"
    bindings = tmp_path / "test_bind"
    functions = tmp_path / "test_func"
    models = tmp_path / "test_models"
    zoos = tmp_path / "test_zoos"
    personalities.mkdir()
    bindings.mkdir()
    functions.mkdir()
    models.mkdir()
    (models / "ttt").mkdir()
    (models / "tti").mkdir()
    zoos.mkdir()
    (zoos / "personalities").mkdir()
    (zoos / "bindings").mkdir()
    return tmp_path

@pytest.fixture
def dummy_config_path(temp_config_dir: Path) -> Path:
    """Creates a dummy config.toml file in the temp directory."""
    config_content = {
        "server": {"host": "127.0.0.1", "port": 9999},
        "logging": {"log_level": "DEBUG"},
        "paths": {
            "personalities_folder": str(temp_config_dir / "test_pers"),
            "bindings_folder": str(temp_config_dir / "test_bind"),
            "functions_folder": str(temp_config_dir / "test_func"),
            "models_folder": str(temp_config_dir / "test_models"),
            "example_personalities_folder": str(temp_config_dir / "test_zoos" / "personalities"),
            "example_bindings_folder": str(temp_config_dir / "test_zoos" / "bindings"),
        },
        "security": {"allowed_api_keys": ["test_key_1", "test_key_2"]},
        "defaults": {
            "ttt_binding": "test_ollama",
            "ttt_model": "test_llama3"
        },
        "bindings": {
            "test_ollama": {"type": "ollama_binding", "host": "http://localhost:11434"},
            "test_openai": {"type": "openai_binding", "api_key": "test_sk"}
        },
        "resource_manager": {"gpu_strategy": "simple_lock", "gpu_limit": 1}
    }
    config_path = temp_config_dir / "config.toml"
    with open(config_path, "w") as f:
        toml.dump(config_content, f)
    return config_path

# --- Fixture to Reset Config ---
@pytest.fixture(autouse=True) # Automatically used by tests needing config
def reset_config_singleton():
    """Resets the config module's singleton before and after tests."""
    from lollms_server.core import config
    original_config = config._config
    original_path = config._config_path
    config._config = None # Reset before test
    config._config_path = None
    yield # Test runs here
    config._config = original_config # Restore after test
    config._config_path = original_path


# Add more shared fixtures here (e.g., mock managers, sample data)