# lollms_server/core/config.py
import toml
from pydantic import BaseModel, Field, DirectoryPath, FilePath, HttpUrl
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import os
import ascii_colors as logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Configuration Structure ---

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9600
    # workers: int = 1 # Worker count is usually managed by Uvicorn/Gunicorn
    # Default allows common local development origins + file:// (use with caution)
    allowed_origins: List[str] = Field(default=[
        "http://localhost",
        "http://localhost:8000", # Common dev port
        "http://localhost:5173", # Default Vite port
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
        "null" # For file:// access - REMOVE FOR PRODUCTION if not needed
    ])

# --- Add WebUI Config Model ---
class WebUIConfig(BaseModel):
    # Enable/disable serving the web UI static files
    enable_ui: bool = False # Default to disabled if section exists but key missing
    # Optional: Specify a custom build directory relative to project root
    # build_directory: str = "webui/dist" # Default handled in main.py logic    
class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    level:int = 0

class PathsConfig(BaseModel):
    personalities_folder: Path = Field(default_factory=lambda: Path("personal_personalities/"))
    bindings_folder: Path = Field(default_factory=lambda: Path("personal_bindings/"))
    functions_folder: Path = Field(default_factory=lambda: Path("personal_functions/"))
    models_folder: Path = Field(default_factory=lambda: Path("models/"))
    example_personalities_folder: Optional[Path] = Field(default_factory=lambda: Path("zoos/personalities/"))
    example_bindings_folder: Optional[Path] = Field(default_factory=lambda: Path("zoos/bindings/"))
    example_functions_folder: Optional[Path] = Field(default_factory=lambda: Path("zoos/functions/"))

class SecurityConfig(BaseModel):
    allowed_api_keys: List[str] = Field(default_factory=list)

class DefaultsConfig(BaseModel):
    ttt_binding: Optional[str] = None
    ttt_model: Optional[str] = None
    tti_binding: Optional[str] = None
    tti_model: Optional[str] = None
    ttv_binding: Optional[str] = None
    ttv_model: Optional[str] = None
    ttm_binding: Optional[str] = None
    ttm_model: Optional[str] = None
    # --- NEW Default Generation Parameters ---
    default_context_size: int = 4096     # Default value if not in config
    default_max_output_tokens: int = 1024 # Default value if not in config

class ResourceManagerConfig(BaseModel):
    gpu_strategy: str = "semaphore" # or "simple_lock"
    gpu_limit: int = 1
    queue_timeout: int = 120 # seconds


# --- NEW: Model for individual personality settings in config ---
class PersonalityInstanceConfig(BaseModel):
    enabled: bool = True # Default to enabled if entry exists
    # Add other per-personality overrides here later if needed
    # e.g., specific_model: Optional[str] = None
    #    

class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    bindings: Dict[str, Dict[str, Any]] = Field(default_factory=dict) # Specific binding configs
    resource_manager: ResourceManagerConfig = Field(default_factory=ResourceManagerConfig)
    webui: Optional[WebUIConfig] = None
    # --- NEW: Optional dictionary for personality enable/disable states ---
    # Key: personality name (folder name), Value: PersonalityInstanceConfig
    personalities_config: Optional[Dict[str, PersonalityInstanceConfig]] = Field(default_factory=dict)

# --- Configuration Loading Function ---

_config: Optional[AppConfig] = None
_config_path: Optional[Path] = None

def load_config(config_path: Union[str, Path] = "config.toml") -> AppConfig:
    """Loads the configuration from the specified TOML file."""
    global _config, _config_path
    path = Path(config_path)
    _config_path = path.resolve() # Store absolute path

    if not path.exists():
        logger.warning(f"Configuration file not found at {path}. Using default settings.")
        _config = AppConfig()
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = toml.load(f)
            _config = AppConfig(**config_data)
            logger.info(f"Configuration loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e}")
            logger.warning("Using default configuration due to error.")
            _config = AppConfig() # Fallback to defaults on error

    # Resolve relative paths relative to the config file location
    config_dir = _config_path.parent
    _resolve_paths(_config.paths, config_dir)

    # Create essential directories if they don't exist
    _ensure_directories(_config.paths)

    return _config

def get_config() -> AppConfig:
    """Returns the loaded configuration."""
    if _config is None:
        # Load default config if not already loaded (e.g., during tests or if load failed)
        return load_config()
    return _config

def get_config_path() -> Optional[Path]:
    """Returns the absolute path of the loaded config file, if any."""
    return _config_path

def _resolve_paths(paths_config: PathsConfig, base_dir: Path):
    """Resolves relative paths in the PathsConfig object."""
    for field_name in PathsConfig.model_fields:
        path_value = getattr(paths_config, field_name)
        if path_value and isinstance(path_value, Path) and not path_value.is_absolute():
            resolved_path = (base_dir / path_value).resolve()
            setattr(paths_config, field_name, resolved_path)
            # logger.debug(f"Resolved path {field_name}: {resolved_path}") # Optional: for debugging

def _ensure_directories(paths_config: PathsConfig):
    """Creates directories specified in the config if they don't exist."""
    required_folders = [
        paths_config.personalities_folder,
        paths_config.bindings_folder,
        paths_config.functions_folder,
        paths_config.models_folder,
    ]
    for folder_path in required_folders:
        if folder_path:
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {folder_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {folder_path}: {e}")

    # Ensure model subdirectories exist
    model_base = paths_config.models_folder
    if model_base:
        for subfolder in ["ttt", "tti", "ttv", "ttm"]:
            try:
                (model_base / subfolder).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create model sub-directory {model_base / subfolder}: {e}")

# Load configuration immediately when module is imported
# This makes 'get_config()' readily available
# You might prefer lazy loading depending on application structure
# load_config() # Or load it explicitly in main.py startup
