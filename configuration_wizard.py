# File: configuration_wizard.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: Interactive wizard to create the initial main configuration for lollms_server using ConfigGuard.

import sys
import platform
import codecs
import secrets
import typing
from pathlib import Path
import os
import asyncio
import yaml  # For reading binding cards
import json # For displaying config review
from typing import Dict, Any, Optional, List, Tuple, Union

# --- Force UTF-8 output on Windows consoles ---
if platform.system() == "Windows":
    try:
        # Check if stdout and stderr encodings need changing
        stdout_encoding_needs_change = hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != 'utf-8'
        stderr_encoding_needs_change = hasattr(sys.stderr, 'encoding') and sys.stderr.encoding.lower() != 'utf-8'

        if stdout_encoding_needs_change:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')  # type: ignore
            print("INFO: Reconfigured sys.stdout to UTF-8")
        if stderr_encoding_needs_change:
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')  # type: ignore
            print("INFO: Reconfigured sys.stderr to UTF-8")

    except Exception as e:
        print(f"WARNING: Failed to force UTF-8 output encoding on Windows: {e}")
        print("         Special characters in the console might not display correctly.")
# --- END UTF-8 FIX BLOCK ---


# --- Ensure pipmaster and core dependencies are available ---
# Assume installer_core.py ran first, but check critical libs
try:
    # Bootstrap check: Is pipmaster available?
    import pipmaster
except ImportError:
    print("FATAL ERROR: pipmaster library not found. Cannot ensure dependencies.")
    print("Please run the main installation script (install.sh/install.bat) first.")
    sys.exit(1)

try:
    # Use pipmaster to ensure runtime dependencies for the wizard itself
    pm = pipmaster.PackageManager()
    pm.install_if_missing("ascii_colors>=0.10.0")
    pm.install_if_missing("configguard>=0.4.2")
    pm.install_if_missing("pyyaml>=6.0")  # Needed for binding cards
    pm.install_if_missing("toml>=0.10.0")  # Optional handler, good to have
    pm.install_if_missing("cryptography>=3.4")  # For encryption key gen
    # Optional API clients needed for model listing - install if missing
    pm.install_if_missing("ollama")
    pm.install_if_missing("openai")
except Exception as e:
    print(f"FATAL ERROR during dependency check/install: {e}")
    # Print traceback manually if ascii_colors isn't loaded yet
    import traceback
    traceback.print_exc()
    sys.exit(1)
# --- END DEPENDENCY CHECK ---

# --- Core Imports (Assume available after checks) ---
try:
    import ascii_colors as logging # Use logging alias for consistency
    # Import Menu with epilog_text support (assuming ascii_colors>=0.10.0)
    from ascii_colors import ASCIIColors, Menu, trace_exception, MenuItem
    from configguard import ConfigGuard, ConfigSection, generate_encryption_key, ValidationError
    from configguard.exceptions import ConfigGuardError, SchemaError, SettingNotFoundError
    from configguard.handlers import JsonHandler, YamlHandler, TomlHandler, SqliteHandler

    # Import sync clients for model listing (handle potential missing install)
    try:
        import ollama as ollama_client
        OLLAMA_AVAILABLE = True
    except ImportError:
        ollama_client = None  # type: ignore
        OLLAMA_AVAILABLE = False
        logging.warning("Ollama library not installed. Model listing for Ollama disabled.")
    try:
        from openai import OpenAI as OpenAIClient  # Use sync client
        OPENAI_AVAILABLE = True
    except ImportError:
        OpenAIClient = None  # type: ignore
        OPENAI_AVAILABLE = False
        logging.warning("OpenAI library not installed. Model listing for OpenAI disabled.")

except ImportError as e:
    print(f"FATAL ERROR: Missing core library after checks ({e}). Installation might have failed.")
    sys.exit(1)
# --- END CORE IMPORTS ---

# --- Logging Setup ---
# Use ascii_colors for logging
logging.basicConfig(level=logging.INFO, format='{asctime} | {levelname:<8} | {name} | {message}', style='{')
logger = logging.getLogger(__name__) # Use __name__ for logger


# --- Main Server Configuration Schema (Mirrors core/config.py) ---
MAIN_CONFIG_VERSION = "0.3.0"
# Schema definition dictionary - WITHOUT the top-level __version__ key
MAIN_SCHEMA: typing.Dict[str, typing.Any] = {
    "server": {
        "type": "section", "help": "Core server settings.",
        "schema": {
            "host": {"type": "str", "default": "0.0.0.0", "help": "Host address"},
            "port": {"type": "int", "default": 9601, "min_val": 1, "max_val": 65535, "help": "Port number"},
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
            "config_base_dir": {
                "type": "str",
                "default": "lollms_configs",
                "help": "Base directory for configuration files (main, bindings etc.). Relative to server root if not absolute."
            },
            "instance_bindings_folder": {
                "type": "str",
                "default": "bindings",
                "help": "Subfolder within config_base_dir for binding instance configs."
            },
            "personalities_folder": {
                "type": "str",
                "default": "personal/personalities",
                "help": "Folder for your custom personalities (relative to server root)."
            },
            "bindings_folder": {
                "type": "str",
                "default": "personal/bindings",
                "help": "Folder for your custom binding types (relative to server root)."
            },
            "functions_folder": {
                "type": "str",
                "default": "personal/functions",
                "help": "Folder for your custom functions (relative to server root)."
            },
            "models_folder": {
                "type": "str",
                "default": "models",
                "help": "Base folder for model files (relative to server root)."
            },
            "example_personalities_folder": {
                "type": "str",
                "default": "zoos/personalities",
                "nullable": True,
                "help": "Path to built-in example personalities (relative to server root)."
            },
            "example_bindings_folder": {
                "type": "str",
                "default": "zoos/bindings",
                "nullable": True,
                "help": "Path to built-in example binding types (relative to server root)."
            },
            "example_functions_folder": {
                "type": "str",
                "default": "zoos/functions",
                "nullable": True,
                "help": "Path to built-in example functions (relative to server root)."
            }
        }
    },
    "security": {
        "type": "section", "help": "Security settings.",
        "schema": {
            "allowed_api_keys": {"type": "list", "default": [], "help": "List of allowed API keys."},
            "encryption_key": {
                "type": "str",
                "nullable": True,
                "default": None,
                "secret": True,
                "help": "Fernet key for encrypting binding configs (optional)."
            }
        }
    },
    "defaults": {
        "type": "section", "help": "Default binding instance names and models.",
        "schema": {
            "ttt_binding": {"type": "str", "nullable": True, "default": None, "help": "Default Text-to-Text binding instance name."},
            "tti_binding": {"type": "str", "nullable": True, "default": None, "help": "Default Text-to-Image binding instance name."},
            "tts_binding": {"type": "str", "nullable": True, "default": None, "help": "Default Text-to-Speech binding instance name."},
            "stt_binding": {"type": "str", "nullable": True, "default": None, "help": "Default Speech-to-Text binding instance name."},
            "ttv_binding": {"type": "str", "nullable": True, "default": None, "help": "Default Text-to-Video binding instance name."},
            "ttm_binding": {"type": "str", "nullable": True, "default": None, "help": "Default Text-to-Music binding instance name."},
            "ttt_model": {"type": "str", "nullable": True, "default": None, "help": "Default model for the TTT binding."},
            "tti_model": {"type": "str", "nullable": True, "default": None, "help": "Default model for the TTI binding."},
            "tts_model": {"type": "str", "nullable": True, "default": None, "help": "Default model for the TTS binding."},
            "stt_model": {"type": "str", "nullable": True, "default": None, "help": "Default model for the STT binding."},
            "ttv_model": {"type": "str", "nullable": True, "default": None, "help": "Default model for the TTV binding."},
            "ttm_model": {"type": "str", "nullable": True, "default": None, "help": "Default model for the TTM binding."},
            "default_context_size": {"type": "int", "default": 4096, "min_val": 64, "help": "Default context window size."},
            "default_max_output_tokens": {"type": "int", "default": 1024, "min_val": 1, "help": "Default max generation tokens."}
        }
    },
    "bindings_map": {
        "type": "section",
        "help": "Maps binding instance names to their type names (e.g., my_ollama = ollama_binding).",
        "schema": {} # Dynamic section
    },
    "resource_manager": {
        "type": "section", "help": "Resource management settings.",
        "schema": {
            "gpu_strategy": {
                "type": "str",
                "default": "semaphore",
                "options": ["semaphore", "simple_lock", "none"],
                "help": "GPU resource locking strategy."
            },
            "gpu_limit": {"type": "int", "default": 1, "min_val": 1, "help": "Max concurrent GPU tasks for 'semaphore'."},
            "queue_timeout": {"type": "int", "default": 120, "min_val": 1, "help": "Queue timeout in seconds."}
        }
    },
    "webui": {
        "type": "section", "help": "Built-in Web UI settings.",
        "schema": {
            "enable_ui": {"type": "bool", "default": False, "help": "Enable serving the Web UI."}
        }
    },
    "logging": {
        "type": "section", "help": "Logging configuration.",
        "schema": {
            "log_level": {
                "type": "str",
                "default": "INFO",
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "help": "Logging verbosity level."
            },
            "level": {"type": "int", "default": 20, "help": "Logging level numeric value (e.g., DEBUG=10, INFO=20)."}
        }
    },
    "personalities_config": {
        "type": "section",
        "help": "Overrides for personalities (e.g., { 'python_builder_executor': { 'enabled': False } }).",
        "nullable": True,
        "default": {},
        "schema": {} # Dynamic section
    }
}
# --- END SCHEMA ---


# --- Configuration Presets ---
PRESETS = {
    "Ollama CPU (Default Text)": {
        "defaults": {
            "ttt_binding": "my_ollama_cpu", "ttt_model": "llama3:latest",
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_ollama_cpu": "ollama_binding", },
        "resource_manager": { "gpu_strategy": "none", },
        "suggested_instance_configs": { "my_ollama_cpu": {"type": "ollama_binding", "host": "http://localhost:11434"} }
    },
    "Ollama GPU (Default Text)": { # Added GPU variant
        "defaults": {
            "ttt_binding": "my_ollama_gpu", "ttt_model": "mixtral:latest", # Suggest Mixtral for GPU
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_ollama_gpu": "ollama_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # Assume GPU use
        "suggested_instance_configs": { "my_ollama_gpu": {"type": "ollama_binding", "host": "http://localhost:11434"} }
    },
    "Ollama GPU + Diffusers GPU": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_ollama_gpu", "ttt_model": "mixtral:latest",
            "tti_binding": "my_diffusers_gpu", "tti_model": None, # User selects Diffusers model later
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_ollama_gpu": "ollama_binding", "my_diffusers_gpu": "diffusers_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, },
        "suggested_instance_configs": {
            "my_ollama_gpu": {"type": "ollama_binding", "host": "http://localhost:11434"},
            "my_diffusers_gpu": { "type": "diffusers_binding", "models_folder": "models/diffusers_models", "device": "cuda", "use_fp16": True }
        }
    },
    "Ollama GPU + DALL-E 3": { # NEW COMBINATION
        "defaults": {
            "ttt_binding": "my_ollama_gpu", "ttt_model": "mixtral:latest",
            "tti_binding": "my_dalle3_api", "tti_model": "dall-e-3",
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_ollama_gpu": "ollama_binding", "my_dalle3_api": "dalle_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # Ollama uses GPU
        "suggested_instance_configs": {
            "my_ollama_gpu": {"type": "ollama_binding", "host": "http://localhost:11434"},
            "my_dalle3_api": {"type": "dalle_binding", "api_key": None, "model": "dall-e-3"} # Prompt for key
        }
    },
    "OpenAI API (Text & Image)": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_openai_api", "ttt_model": "gpt-4o",
            "tti_binding": "my_dalle3_api", "tti_model": "dall-e-3",
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_openai_api": "openai_binding", "my_dalle3_api": "dalle_binding", },
        "resource_manager": { "gpu_strategy": "none", },
        "suggested_instance_configs": {
            "my_openai_api": {"type": "openai_binding", "api_key": None}, # Prompt for key
            "my_dalle3_api": {"type": "dalle_binding", "api_key": None, "model": "dall-e-3"} # Prompt for key
        }
    },
    "OpenAI API + Diffusers GPU": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_openai_api", "ttt_model": "gpt-4o",
            "tti_binding": "my_diffusers_gpu", "tti_model": None,
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_openai_api": "openai_binding", "my_diffusers_gpu": "diffusers_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # Diffusers uses GPU
        "suggested_instance_configs": {
            "my_openai_api": {"type": "openai_binding", "api_key": None}, # Prompt for key
            "my_diffusers_gpu": { "type": "diffusers_binding", "models_folder": "models/diffusers_models", "device": "cuda", "use_fp16": True }
        }
    },
    "LlamaCpp CPU (Local Text)": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_llamacpp_cpu", "ttt_model": None, # User selects model later
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_llamacpp_cpu": "llamacpp_binding", },
        "resource_manager": { "gpu_strategy": "none", },
        "suggested_instance_configs": { "my_llamacpp_cpu": { "type": "llamacpp_binding", "models_folder": "models/gguf", "n_gpu_layers": 0, "n_ctx": 4096, } }
    },
    "LlamaCpp GPU (Local Text)": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_llamacpp_gpu", "ttt_model": None, # User selects model later
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_llamacpp_gpu": "llamacpp_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, },
        "suggested_instance_configs": { "my_llamacpp_gpu": { "type": "llamacpp_binding", "models_folder": "models/gguf", "n_gpu_layers": -1, "n_ctx": 4096, } }
    },
    "LlamaCpp GPU + Diffusers GPU": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_llamacpp_gpu", "ttt_model": None, # User selects GGUF model later
            "tti_binding": "my_diffusers_gpu", "tti_model": None, # User selects Diffusers model later
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_llamacpp_gpu": "llamacpp_binding", "my_diffusers_gpu": "diffusers_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, },
        "suggested_instance_configs": {
            "my_llamacpp_gpu": { "type": "llamacpp_binding", "models_folder": "models/gguf", "n_gpu_layers": -1, "n_ctx": 4096 },
            "my_diffusers_gpu": { "type": "diffusers_binding", "models_folder": "models/diffusers_models", "device": "cuda", "use_fp16": True }
        }
    },
    "LlamaCpp GPU + DALL-E 3": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_llamacpp_gpu", "ttt_model": None,
            "tti_binding": "my_dalle3_api", "tti_model": "dall-e-3",
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_llamacpp_gpu": "llamacpp_binding", "my_dalle3_api": "dalle_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # LlamaCpp uses GPU
        "suggested_instance_configs": {
            "my_llamacpp_gpu": { "type": "llamacpp_binding", "models_folder": "models/gguf", "n_gpu_layers": -1, "n_ctx": 4096 },
            "my_dalle3_api": {"type": "dalle_binding", "api_key": None, "model": "dall-e-3"} # Prompt for key
        }
    },
    "Hugging Face CPU (Local Text)": { # NEW HF PRESET
        "defaults": {
            "ttt_binding": "my_hf_cpu", "ttt_model": None, # User selects model later
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_hf_cpu": "hf_binding", },
        "resource_manager": { "gpu_strategy": "none", },
        "suggested_instance_configs": {
            "my_hf_cpu": {
                "type": "hf_binding", "model_name_or_path": "microsoft/phi-2", # Example default
                "device": "cpu", "use_fp16": False, "use_bf16": False, "quantization": "none",
            }
        }
    },
    "Hugging Face GPU (Local Text)": { # NEW HF PRESET
        "defaults": {
            "ttt_binding": "my_hf_gpu", "ttt_model": None, # User selects model later
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_hf_gpu": "hf_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, },
        "suggested_instance_configs": {
            "my_hf_gpu": {
                "type": "hf_binding", "model_name_or_path": "google/gemma-7b-it", # Example default
                "device": "auto", "use_fp16": True, "use_bf16": False, "quantization": "none",
                "use_flash_attention_2": True, # Suggest enabling if possible
            }
        }
    },
    "Hugging Face GPU + Diffusers GPU": { # NEW COMBINATION
        "defaults": {
            "ttt_binding": "my_hf_gpu", "ttt_model": None,
            "tti_binding": "my_diffusers_gpu", "tti_model": None,
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_hf_gpu": "hf_binding", "my_diffusers_gpu": "diffusers_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # Both use GPU potentially
        "suggested_instance_configs": {
            "my_hf_gpu": { "type": "hf_binding", "model_name_or_path": "google/gemma-7b-it", "device": "auto", "use_fp16": True, "use_flash_attention_2": True, },
            "my_diffusers_gpu": { "type": "diffusers_binding", "models_folder": "models/diffusers_models", "device": "cuda", "use_fp16": True }
        }
    },
     "Hugging Face GPU + DALL-E 3": { # NEW COMBINATION
        "defaults": {
            "ttt_binding": "my_hf_gpu", "ttt_model": None,
            "tti_binding": "my_dalle3_api", "tti_model": "dall-e-3",
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_hf_gpu": "hf_binding", "my_dalle3_api": "dalle_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # HF uses GPU
        "suggested_instance_configs": {
            "my_hf_gpu": { "type": "hf_binding", "model_name_or_path": "google/gemma-7b-it", "device": "auto", "use_fp16": True, "use_flash_attention_2": True, },
            "my_dalle3_api": {"type": "dalle_binding", "api_key": None, "model": "dall-e-3"} # Prompt for key
        }
    },
    "Gemini API (Text & Vision)": { # Existing Preset
        "defaults": {
            "ttt_binding": "my_gemini_api", "ttt_model": "gemini-1.5-pro-latest", # Example vision model
            "tti_binding": None, "tti_model": None, "tts_binding": None, "tts_model": None,
            "stt_binding": None, "stt_model": None, "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_gemini_api": "gemini_binding", },
        "resource_manager": { "gpu_strategy": "none", },
        "suggested_instance_configs": { "my_gemini_api": { "type": "gemini_binding", "google_api_key": None, "auto_detect_limits": True } }
    },
    "Gemini API + Diffusers GPU": { # NEW COMBINATION
        "defaults": {
            "ttt_binding": "my_gemini_api", "ttt_model": "gemini-1.5-pro-latest",
            "tti_binding": "my_diffusers_gpu", "tti_model": None,
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_gemini_api": "gemini_binding", "my_diffusers_gpu": "diffusers_binding", },
        "resource_manager": { "gpu_strategy": "semaphore", "gpu_limit": 1, }, # Diffusers uses GPU
        "suggested_instance_configs": {
            "my_gemini_api": { "type": "gemini_binding", "google_api_key": None, "auto_detect_limits": True },
            "my_diffusers_gpu": { "type": "diffusers_binding", "models_folder": "models/diffusers_models", "device": "cuda", "use_fp16": True }
        }
    },
    "Gemini API + DALL-E 3": { # NEW COMBINATION
        "defaults": {
            "ttt_binding": "my_gemini_api", "ttt_model": "gemini-1.5-pro-latest",
            "tti_binding": "my_dalle3_api", "tti_model": "dall-e-3",
            "tts_binding": None, "tts_model": None, "stt_binding": None, "stt_model": None,
            "ttv_binding": None, "ttv_model": None, "ttm_binding": None, "ttm_model": None,
        },
        "bindings_map": { "my_gemini_api": "gemini_binding", "my_dalle3_api": "dalle_binding", },
        "resource_manager": { "gpu_strategy": "none", }, # Neither uses local GPU
        "suggested_instance_configs": {
            "my_gemini_api": { "type": "gemini_binding", "google_api_key": None, "auto_detect_limits": True },
            "my_dalle3_api": {"type": "dalle_binding", "api_key": None, "model": "dall-e-3"} # Prompt for key
        }
    },
}
# --- END PRESETS ---

# --- Diffusers Model Suggestions ---
# List of (Display Name, Hugging Face Hub ID)
DIFFUSERS_SUGGESTIONS = [
    ("Stable Diffusion XL Base 1.0 (General Purpose, High Quality)", "stabilityai/stable-diffusion-xl-base-1.0"),
    ("Stable Diffusion XL Refiner 1.0 (Use with Base)", "stabilityai/stable-diffusion-xl-refiner-1.0"),
    ("Stable Diffusion 1.5 (Lower VRAM, Faster)", "runwayml/stable-diffusion-v1-5"),
    ("Stable Diffusion 2.1 (Alternative)", "stabilityai/stable-diffusion-2-1"),
    ("SDXL Turbo (Fast Generation)", "stabilityai/sdxl-turbo"),
    ("LCM LoRA SDXL (Latent Consistency Model - Needs Base SDXL)", "latent-consistency/lcm-lora-sdxl"),
    # Add more suggestions here
    ("Animagine XL 3.1 (Anime Style)", "cagliostrolab/animagine-xl-3.1"),
    ("DreamShaper XL (Semi-Realistic/Fantasy)", "Lykon/dreamshaper-xl-1024-v2-baked-vae"),
]
# --- END SUGGESTIONS ---

# --- DALL-E Model Suggestions ---
DALLE_MODEL_SUGGESTIONS = ["dall-e-3", "dall-e-2"]

# --- Binding Type Modality Mapping ---
# Used to filter binding instances when selecting defaults
BINDING_MODALITY_MAP = {
    "ollama_binding": ["ttt", "tti_vision"], # Ollama can serve vision models
    "openai_binding": ["ttt", "tti_vision"], # OpenAI models like GPT-4o handle vision
    "llamacpp_binding": ["ttt", "tti_vision"], # Llama.cpp supports LLaVA models
    "hf_binding": ["ttt", "tti_vision"], # HF can load vision models
    "diffusers_binding": ["tti", "i2i"],
    "dalle_binding": ["tti"],
    "gemini_binding": ["ttt", "tti_vision"], # Gemini supports vision input
    "dummy_binding": ["ttt", "tti", "tts", "stt", "ttv", "ttm", "i2i", "audio2audio"], # Can simulate anything
    # Add other binding types as they are created
}
# --- END MODALITY MAPPING ---


# --- Utility Functions ---

def print_wizard_title(text: str):
    """Prints a formatted title using ASCIIColors."""
    ASCIIColors.bold(f"\n--- {text} ---", color=ASCIIColors.color_bright_cyan)

def suggest_api_key() -> str:
    """Generates a secure random API key."""
    return secrets.token_urlsafe(32)

def check_handler_dependency(handler_class: type) -> bool:
    """Checks if the dependency for a ConfigGuard handler is installed."""
    if hasattr(handler_class, 'check_dependency') and callable(handler_class.check_dependency):
        try:
            handler_class.check_dependency()
            return True
        except ImportError:
            return False
        except Exception as e:
             logger.warning(f"Unexpected error checking dependency for {handler_class.__name__}: {e}")
             return False # Assume problematic if check fails unexpectedly
    else:
         logger.debug(f"Handler {handler_class.__name__} doesn't have check_dependency. Assuming OK.")
         return True

def _get_binding_types_from_disk(main_config: ConfigGuard, server_root: Path) -> Dict[str, Dict[str, Any]]:
    """Scans configured binding folders for binding_card.yaml files."""
    binding_types: Dict[str, Dict[str, Any]] = {}
    potential_dirs: List[Path] = []
    paths_section = getattr(main_config, "paths", None)
    if not paths_section:
         logger.error("Paths section missing in main config. Cannot scan for binding types.")
         return {}

    # Use paths directly from the ConfigGuard object (should be absolute after init)
    example_folder_str = getattr(paths_section, "example_bindings_folder", None)
    personal_folder_str = getattr(paths_section, "bindings_folder", None)
    example_folder = Path(example_folder_str) if example_folder_str and Path(example_folder_str).is_absolute() else None
    personal_folder = Path(personal_folder_str) if personal_folder_str and Path(personal_folder_str).is_absolute() else None

    # Fallback resolution if paths weren't absolute (should not happen after fix)
    if example_folder_str and (not example_folder or not example_folder.is_absolute()):
        example_folder = (server_root / example_folder_str).resolve()
        logger.warning(f"Resolved non-absolute example_bindings_folder to: {example_folder}")
    if personal_folder_str and (not personal_folder or not personal_folder.is_absolute()):
        personal_folder = (server_root / personal_folder_str).resolve()
        logger.warning(f"Resolved non-absolute bindings_folder to: {personal_folder}")


    # Add directories to scan list if they exist
    if example_folder and example_folder.is_dir():
        potential_dirs.append(example_folder)
    if personal_folder and personal_folder.is_dir():
        # Avoid adding the same path twice
        if not example_folder or personal_folder != example_folder:
            potential_dirs.append(personal_folder)

    if not potential_dirs:
        logger.warning("No binding type folders configured or found.")
        return {}

    logger.debug(f"Wizard scanning for binding types in: {[str(d) for d in potential_dirs]}")
    for bdir in potential_dirs:
        try:
            for item in bdir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    card_path = item / "binding_card.yaml"
                    init_path = item / "__init__.py"
                    if init_path.exists() and card_path.exists():
                        type_name_from_folder = item.name
                        display_name = type_name_from_folder
                        card_data = None
                        try:
                            with open(card_path, 'r', encoding='utf-8') as f:
                                card_data = yaml.safe_load(f)

                            # Validate loaded card data
                            if not isinstance(card_data, dict):
                                logger.warning(f"Invalid card format (not dict) in {card_path}")
                                card_data = None # Mark as invalid
                                continue

                            type_name = card_data.get('type_name')
                            instance_schema = card_data.get('instance_schema')

                            if not type_name or not isinstance(instance_schema, dict):
                                logger.warning(f"Card {card_path.name} missing 'type_name' or invalid 'instance_schema'.")
                                card_data = None # Mark as invalid
                                continue

                            display_name = card_data.get('display_name', type_name)

                        except yaml.YAMLError as e:
                            logger.error(f"YAML Error loading card {card_path}: {e}")
                            continue # Skip this card
                        except Exception as e:
                            logger.error(f"Error loading card {card_path}: {e}")
                            trace_exception(e)
                            continue # Skip this card

                        # Store valid card data, prioritize personal folder
                        if card_data:
                             card_data["package_path"] = item.resolve() # Store absolute path
                             # Overwrite if already exists and current is from personal folder
                             if type_name not in binding_types or bdir == personal_folder:
                                 binding_types[type_name] = {
                                     "display_name": display_name,
                                     "path": item, # Keep original path object if needed
                                     "card": card_data
                                 }
        except OSError as e:
            logger.warning(f"OS Error scanning directory {bdir}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error scanning directory {bdir}: {e}")
            trace_exception(e)

    logger.info(f"Wizard discovered {len(binding_types)} valid binding types.")
    return binding_types
# --- END UTILITY FUNCTIONS ---

# --- Helper for path resolution ---
# Simplified path resolution logic (moved from core/config.py)
def _wizard_resolve_paths(paths_section: ConfigSection, server_root_dir: Path):
    """Resolves relative paths in the paths section relative to the server root."""
    if not paths_section: return
    try:
        paths_schema = paths_section._schema_definition
    except AttributeError:
        return

    logger.debug(f"Resolving paths relative to server root: {server_root_dir}")
    for key in paths_schema.keys():
        # Skip internal keys and derived paths handled later
        if key.startswith('__') or key == "instance_bindings_folder": continue
        try:
            path_str = getattr(paths_section, key, None)
            if path_str and isinstance(path_str, str):
                path_obj = Path(path_str)
                if not path_obj.is_absolute():
                    resolved_path = (server_root_dir / path_obj).resolve()
                    setattr(paths_section, key, str(resolved_path))
        except Exception as e:
            logger.error(f"Error resolving path '{key}': {e}")

def _wizard_update_derived_paths(paths_section: ConfigSection, config_base_dir: Path):
    """Updates paths that depend on the config_base_dir."""
    if not paths_section or not config_base_dir.is_absolute(): return
    logger.debug(f"Updating derived paths relative to config base dir: {config_base_dir}")
    key = "instance_bindings_folder"
    try:
         instance_folder_name_str = getattr(paths_section, key, None)
         if instance_folder_name_str and isinstance(instance_folder_name_str, str):
             instance_folder_path = Path(instance_folder_name_str)
             if not instance_folder_path.is_absolute():
                  full_instance_path = (config_base_dir / instance_folder_path).resolve()
                  setattr(paths_section, key, str(full_instance_path))
    except Exception as e:
        logger.error(f"Error updating derived path '{key}': {e}")


# --- Main Wizard Class ---
class ConfigurationWizard:
    """Interactive wizard for configuring lollms_server."""

    def __init__(self, server_root: Path):
        self.server_root = server_root
        self.config_base_dir: Optional[Path] = None
        self.main_config_path: Optional[Path] = None
        self.config: Optional[ConfigGuard] = None # Holds the main config instance
        self.selected_handler_class: type = YamlHandler # Default handler class
        self.preset_suggested_configs: Dict[str, Dict[str, Any]] = {} # Store configs from chosen preset
        self.saved_successfully = False
        self._model_list_cache: Dict[str, List[Tuple[str, str]]] = {} # Cache for model lists
        # Map file extensions to ConfigGuard handlers
        self.handler_map: Dict[str, type] = {
            ".yaml": YamlHandler, ".yml": YamlHandler,
            ".json": JsonHandler, ".toml": TomlHandler,
            ".db": SqliteHandler, ".sqlite": SqliteHandler, ".sqlite3": SqliteHandler
        }


    def _get_binding_instance_config_path(self, instance_name: str) -> Optional[Path]:
        """Finds the config file path for a specific binding instance (best guess)."""
        if not self.config or not hasattr(self.config, "paths") or not self.config_base_dir:
            logger.error("Cannot find instance config path: Main config/paths not loaded.")
            return None
        try:
            # Get the absolute instance folder path from the main config
            instance_folder_str = getattr(self.config.paths, "instance_bindings_folder", None)
            if not instance_folder_str:
                logger.error("Instance bindings folder path not set in main config.")
                return None

            instance_dir = Path(instance_folder_str)
            # Ensure it's absolute (should be after init + _wizard_update_derived_paths)
            if not instance_dir.is_absolute():
                logger.error(f"Instance folder path '{instance_dir}' was not absolute. Trying manual resolve relative to '{self.config_base_dir}'.")
                instance_dir = (self.config_base_dir / instance_folder_str).resolve()

            # Check for existing files with common extensions
            for ext in self.handler_map.keys():
                potential_path = instance_dir / f"{instance_name}{ext}"
                if potential_path.is_file():
                    logger.debug(f"Found existing instance config: {potential_path}")
                    return potential_path
            logger.debug(f"No existing config file found for instance '{instance_name}' in {instance_dir}")
            return None # Not found
        except AttributeError:
             logger.error("Error accessing paths section in main config.")
        except Exception as e:
             logger.error(f"Error determining instance config path for '{instance_name}': {e}")
        return None

    def _list_models_for_binding(self, binding_instance_name: str) -> List[Tuple[str, str]]:
        """Attempts to list models for a given binding instance by loading its config."""
        if binding_instance_name in self._model_list_cache:
            logger.debug(f"Using cached model list for '{binding_instance_name}'.")
            return self._model_list_cache[binding_instance_name]
        if not self.config:
            logger.error("Main config not loaded. Cannot list models.")
            return []

        model_options: List[Tuple[str, str]] = []
        logger.info(f"Attempting to list models for binding instance '{binding_instance_name}'...")

        # --- Determine Binding Type and Instance Config ---
        binding_type: Optional[str] = None
        instance_config_dict: Optional[Dict[str, Any]] = None
        try:
            # Get binding type from the main config's map
            bindings_map_section = getattr(self.config, "bindings_map", None)
            if bindings_map_section:
                # Use get() on ConfigSection if available, otherwise iterate
                try:
                    binding_type = bindings_map_section.get(binding_instance_name) # Use get() if ConfigGuard >= 0.4
                except AttributeError: # Fallback for older ConfigGuard or if get fails
                    binding_type = getattr(bindings_map_section, binding_instance_name, None)
            if not binding_type:
                logger.error(f"Binding type for instance '{binding_instance_name}' not found in main config's bindings_map.")
                return []

            # Find and load the instance's configuration file
            instance_config_path = self._get_binding_instance_config_path(binding_instance_name)
            if not instance_config_path:
                logger.warning(f"Config file for instance '{binding_instance_name}' not found. Cannot list API models for it.")
                return []

            try:
                handler_class = self.handler_map.get(instance_config_path.suffix.lower())
                if not handler_class:
                    raise ValueError(f"Unsupported config file extension: {instance_config_path.suffix}")

                # Load instance config safely, ignoring schema errors as schema might not be available/perfect
                enc_key_str = getattr(self.config.security, "encryption_key", None)
                enc_key = enc_key_str.encode() if enc_key_str else None
                instance_cg = ConfigGuard(
                    schema={}, # Load without schema validation here
                    config_path=instance_config_path,
                    encryption_key=enc_key,
                    handler=handler_class()
                )
                instance_cg.load()
                # Use get_config_dict() which should be available
                instance_config_dict = instance_cg.get_config_dict()
                if not isinstance(instance_config_dict, dict):
                     raise ValueError("Loaded instance config is not a dictionary.")

            except Exception as load_err:
                logger.error(f"Error loading instance config file '{instance_config_path}' for model listing: {load_err}")
                trace_exception(load_err)
                return []

        except Exception as e:
            logger.error(f"Error preparing to list models for '{binding_instance_name}': {e}")
            trace_exception(e)
            return []

        # --- List Models Based on Type and Instance Config ---
        try:
            models_base_path_str = getattr(self.config.paths, "models_folder", "models")
            models_base_path = Path(models_base_path_str) # Should be absolute after init
            if not models_base_path.is_absolute(): # Resolve relative to server root if needed
                 logger.warning(f"Models base path '{models_base_path}' was not absolute. Resolving relative to server root '{self.server_root}'.")
                 models_base_path = (self.server_root / models_base_path).resolve()

            # --- Llama.cpp ---
            if binding_type == "llamacpp_binding":
                # Get models folder from instance config, fallback to standard path relative to BASE path
                instance_models_folder_str = instance_config_dict.get("models_folder", "models/gguf")
                gguf_dir = Path(instance_models_folder_str)
                if not gguf_dir.is_absolute():
                     # Assume relative to server root if not absolute
                     gguf_dir = (self.server_root / gguf_dir).resolve()
                     logger.debug(f"Resolved relative llamacpp models folder to: {gguf_dir}")

                if gguf_dir.is_dir():
                    try:
                        model_options = sorted([(f.name, f.name) for f in gguf_dir.glob("*.gguf") if f.is_file()])
                    except Exception as scan_e:
                        logger.warning(f"Error scanning {gguf_dir}: {scan_e}")
                else:
                    logger.warning(f"LLaMA CPP models folder not found or not a directory: {gguf_dir}")

            # --- Diffusers ---
            elif binding_type == "diffusers_binding":
                instance_models_folder_str = instance_config_dict.get("models_folder", "models/diffusers_models")
                diffusers_dir = Path(instance_models_folder_str)
                if not diffusers_dir.is_absolute():
                    diffusers_dir = (self.server_root / diffusers_dir).resolve()
                    logger.debug(f"Resolved relative diffusers models folder to: {diffusers_dir}")

                if diffusers_dir.is_dir():
                    try:
                         # Only list subdirectories
                         model_options = sorted([(d.name, d.name) for d in diffusers_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
                         # ADD suggestions if folder is empty
                         if not model_options:
                             ASCIIColors.warning(f"The diffusers models folder for '{binding_instance_name}' ({diffusers_dir}) is empty.")
                             ASCIIColors.print("Consider downloading a model from Hugging Face Hub.")
                             model_options.extend([(f"{name} (Hub Suggestion)", hub_id) for name, hub_id in DIFFUSERS_SUGGESTIONS])

                    except Exception as scan_e:
                        logger.warning(f"Error scanning {diffusers_dir}: {scan_e}")
                else:
                    logger.warning(f"Diffusers models folder not found or not a directory: {diffusers_dir}")

            # --- Ollama ---
            elif binding_type == "ollama_binding":
                if OLLAMA_AVAILABLE and ollama_client:
                    host = instance_config_dict.get("host", "http://localhost:11434")
                    try:
                        # Use asyncio.to_thread for synchronous client call
                        def _list_ollama_sync(h):
                             sync_client = ollama_client.Client(host=h)
                             return sync_client.list()
                        response = asyncio.run(asyncio.to_thread(_list_ollama_sync, host)) # Run sync in thread
                        models = response.get("models", [])
                        model_options = sorted([(m['model'], m['model']) for m in models if 'model' in m])
                    except Exception as e:
                        logger.error(f"Error listing Ollama models from {host}: {e}")
                else:
                    logger.warning("Ollama library not available.")

            # --- OpenAI ---
            elif binding_type == "openai_binding":
                if OPENAI_AVAILABLE and OpenAIClient:
                    api_key = instance_config_dict.get("api_key") or os.environ.get("OPENAI_API_KEY")
                    base_url = instance_config_dict.get("base_url")
                    if api_key:
                        try:
                            # Use asyncio.to_thread for synchronous client call
                            def _list_openai_sync(key, url):
                                 sync_client = OpenAIClient(api_key=key, base_url=url)
                                 return sync_client.models.list()
                            response = asyncio.run(asyncio.to_thread(_list_openai_sync, api_key, base_url)) # Run sync in thread
                            # Filter for common chat/instruct models
                            model_options = sorted([
                                (m.id, m.id) for m in response.data
                                if any(tag in m.id.lower() for tag in ['gpt', 'instruct', 'claude', 'mistral', 'command', 'ft:', '/'])
                            ])
                        except Exception as e:
                            logger.error(f"Error listing OpenAI models from {base_url or 'default endpoint'}: {e}")
                    else:
                        logger.warning(f"API key missing for OpenAI instance '{binding_instance_name}'. Cannot list models.")
                else:
                    logger.warning("OpenAI library not available.")

            # --- DALL-E ---
            elif binding_type == "dalle_binding":
                 # List predefined DALL-E models
                 model_options = [(model_name, model_name) for model_name in DALLE_MODEL_SUGGESTIONS]

            # --- Hugging Face ---
            elif binding_type == "hf_binding":
                logger.warning(f"Automatic model listing for 'hf_binding' type '{binding_instance_name}' is complex. Please specify the model name/path manually or via preset.")
                # Optionally, list Hub models (could be slow and noisy)
                # model_options.append(("(Enter Hub ID or Local Path Manually)", "##MANUAL##"))

            # Add logic for other binding types if they support model listing

        except Exception as e:
             logger.error(f"Error occurred during model listing logic for '{binding_instance_name}': {e}")
             trace_exception(e)

        # Cache and return results
        if model_options:
            logger.info(f"Found {len(model_options)} potential models for '{binding_instance_name}'.")
            self._model_list_cache[binding_instance_name] = model_options
            return model_options
        else:
            logger.warning(f"Could not list models automatically for '{binding_instance_name}'.")
            return []

    def run(self):
        """Main execution flow of the configuration wizard."""
        print_wizard_title("LOLLMS Server Initial Configuration Wizard")
        ASCIIColors.print("This wizard will help you create the main server configuration (main_config.*).")

        # --- Step 1: Select Base Directory ---
        self.prompt_config_directory()
        if not self.config_base_dir:
            ASCIIColors.error("Configuration directory setup failed. Aborting.")
            sys.exit(1)

        # --- Step 2: Select Main Config File Format and Path ---
        self.prompt_main_config_file()
        if not self.main_config_path:
             ASCIIColors.error("Main configuration file setup failed. Aborting.")
             sys.exit(1)

        # --- Step 3: Initialize Main ConfigGuard Instance ---
        try:
            # Use the determined path and handler class
            handler_instance = self.selected_handler_class()
            self.config = ConfigGuard(
                schema=MAIN_SCHEMA, # Pass schema without version here
                instance_version=MAIN_CONFIG_VERSION, # Pass expected version separately
                config_path=self.main_config_path,
                handler=handler_instance,
                autosave=False, # Important: Manual save at the end
            )

            # Critical Fix: Resolve paths immediately after initialization
            if hasattr(self.config, 'paths'):
                try:
                    # Use the already determined absolute config_base_dir
                    # Set it explicitly in the config object for consistency
                    setattr(self.config.paths, 'config_base_dir', str(self.config_base_dir))
                    # Use the wizard's path resolution helpers
                    _wizard_resolve_paths(self.config.paths, self.server_root)
                    _wizard_update_derived_paths(self.config.paths, Path(self.config.paths.config_base_dir))
                    logger.info("Paths resolved after ConfigGuard initialization.")
                except Exception as path_e:
                     logger.error(f"Error resolving/updating paths after init: {path_e}")
                     trace_exception(path_e)
                     ASCIIColors.error("Failed initial path setup. Aborting.")
                     sys.exit(1)
            else:
                logger.critical("Cannot resolve paths: 'paths' section missing in main config object after init.")
                sys.exit(1)


            ASCIIColors.print(f"Using {type(handler_instance).__name__} for {self.main_config_path.name}.")

        except ImportError as e:
            ASCIIColors.error(f"Failed to initialize config handler dependency: {e}")
            ASCIIColors.error("Please install required extras (e.g., 'pip install lollms_server[toml]') and restart.")
            sys.exit(1)
        except Exception as e:
            ASCIIColors.error(f"Failed to initialize ConfigGuard for main config: {e}")
            trace_exception(e)
            sys.exit(1)

        # --- Step 4: Run Main Menu ---
        self.main_menu()

        # --- Step 5: Final message ---
        if self.saved_successfully:
            ASCIIColors.success("\nWizard finished. Main configuration saved.")
            ASCIIColors.print(
                "You may need to use the installer menu (install_core.py) again "
                "to add/edit specific binding instances if you didn't configure them via a preset."
            )
        else:
             ASCIIColors.warning("\nWizard finished, but main configuration was NOT saved.")

    def prompt_config_directory(self):
        """Prompts user for the main configuration base directory."""
        print_wizard_title("1. Select Configuration Base Directory")
        ASCIIColors.print("This directory will store main_config.*, binding configs, etc.")
        # Get default name from schema
        default_relative_path = MAIN_SCHEMA.get("paths", {}).get("schema", {}).get("config_base_dir", {}).get("default", "lollms_configs")
        default_dir = (self.server_root / default_relative_path).resolve()

        while True:
            ASCIIColors.print("\nEnter the full path for your configurations directory.")
            ASCIIColors.print(f"Default:", color=ASCIIColors.color_yellow)
            ASCIIColors.print(f"'{default_dir}'")
            user_input = ASCIIColors.prompt("Directory path [Enter for default]: ").strip()
            chosen_path_str = user_input or str(default_dir)
            try:
                chosen_path = Path(chosen_path_str).resolve()
                # Create if doesn't exist
                if not chosen_path.exists():
                    ASCIIColors.warning(f"Directory '{chosen_path}' does not exist.")
                    if ASCIIColors.confirm("Create it?", default_yes=True):
                        try:
                            chosen_path.mkdir(parents=True, exist_ok=True)
                            ASCIIColors.green(f"Created directory: {chosen_path}")
                        except Exception as mkdir_e:
                            ASCIIColors.error(f"Failed create directory {chosen_path}: {mkdir_e}")
                            trace_exception(mkdir_e)
                            continue # Ask again
                    else:
                        ASCIIColors.print("Please choose existing directory or allow creation.")
                        continue # Ask again
                elif not chosen_path.is_dir():
                     ASCIIColors.error(f"Path exists but is not a directory: {chosen_path}")
                     continue # Ask again

                # Valid directory selected
                self.config_base_dir = chosen_path
                ASCIIColors.green(f"Configuration base directory set to: {self.config_base_dir}")
                return # Exit loop
            except Exception as e:
                ASCIIColors.error(f"Invalid path or error during selection: {e}")
                trace_exception(e)
                ASCIIColors.warning("Please try entering the path again.")


    def prompt_main_config_file(self):
        """Prompts user for the main config file format and sets path/handler."""
        if not self.config_base_dir:
            ASCIIColors.error("Cannot set main config file: Base directory not set.")
            self.main_config_path = None # Ensure path is None
            return

        print_wizard_title("2. Select Main Configuration File Format")
        ASCIIColors.print("Choose the format for your main server configuration file (main_config.*).")
        # Prepare options for the menu
        format_options: List[Tuple[str, Tuple[str, type]]] = []
        # Prefer YAML as default
        preferred_handler = YamlHandler
        preferred_ext = ".yaml"
        other_handlers = [
            ("JSON (.json)", (".json", JsonHandler)),
            ("TOML (.toml)", (".toml", TomlHandler)),
            ("SQLite (.db)", (".db", SqliteHandler)), # Added SQLite
        ]

        # Add preferred first if dependency met
        if check_handler_dependency(preferred_handler):
             format_options.append(("YAML (.yaml)", (preferred_ext, preferred_handler)))
        else:
             format_options.append(("YAML (.yaml) (Dependency Missing!)", (preferred_ext, preferred_handler)))

        # Add others
        for display, (ext, handler_cls) in other_handlers:
            dep_ok = check_handler_dependency(handler_cls)
            label = f"{display}" + ("" if dep_ok else " (Dependency Missing!)")
            format_options.append((label, (ext, handler_cls)))


        while True:
            # Create and run the menu
            menu = Menu(
                "Select Format",
                mode='select_single',
                item_color=ASCIIColors.color_cyan,
                title_color=ASCIIColors.color_yellow
            )
            selected_data = menu.add_choices(format_options).run()

            if selected_data is None:
                 self.main_config_path = None # User cancelled
                 ASCIIColors.warning("Selection cancelled.")
                 return

            selected_extension, handler_class = selected_data

            # Now check dependency for the chosen handler
            if not check_handler_dependency(handler_class):
                 handler_name = handler_class.__name__.replace('Handler', '')
                 dep_name = ""
                 if handler_name == "Yaml": dep_name = "PyYAML (`pip install pyyaml` or `lollms_server[yaml]`)"
                 elif handler_name == "Toml": dep_name = "toml (`pip install toml` or `lollms_server[toml]`)"
                 elif handler_name == "Sqlite": dep_name = "(built-in, should be available)"
                 else: dep_name = "(Unknown dependency)"
                 ASCIIColors.error(f"Handler '{handler_name}' requires dependency: {dep_name}")
                 ASCIIColors.warning("Please install the dependency (e.g., via installer menu option 2) and restart the wizard, or choose a different format.")
                 continue # Re-prompt format selection

            # Valid handler selected
            self.selected_handler_class = handler_class
            self.main_config_path = self.config_base_dir / f"main_config{selected_extension}"
            ASCIIColors.success(f"Main configuration file set to: {self.main_config_path}")
            ASCIIColors.success(f"Using handler: {handler_class.__name__}")
            return # Exit loop


    def main_menu(self):
        """Displays the main configuration menu and handles choices."""
        if not self.config:
            raise RuntimeError("Main ConfigGuard object not initialized before calling main_menu.")

        menu = Menu(
            "Main Configuration Setup",
            title_color=ASCIIColors.color_bright_yellow,
            item_color=ASCIIColors.color_green,
            selected_background=ASCIIColors.color_bg_blue
        )
        # Define menu actions
        menu.add_action("A. Apply Configuration Preset", self.apply_preset)
        menu.add_action("1. Server Settings (Host, Port, CORS)", self.configure_server)
        menu.add_action("2. Path Settings", self.configure_paths)
        menu.add_action("3. Security (API Keys, Encryption)", self.configure_security)
        menu.add_action("4. Default Bindings & Models", self.configure_defaults)
        # --- Update Binding Map Menu Text ---
        menu.add_action("5. Manage Binding Instances", self.manage_binding_instances_cli)
        # ----------------------------------
        menu.add_action("6. Web UI Settings", self.configure_webui)
        menu.add_action("7. Resource Manager", self.configure_resource_manager)
        menu.add_action("8. Logging", self.configure_logging)
        menu.add_action("9. Review & Save Configuration", self.review_and_save)

        while True:
            choice = menu.run() # Display menu and get user choice

            if choice == self.review_and_save:
                if self.review_and_save(): # If saved successfully
                    break # Exit main menu loop
                else:
                    continue # Stay in menu if save cancelled
            elif choice is None: # User pressed Ctrl+C or Esc
                if not self.saved_successfully:
                    if ASCIIColors.confirm("Exit wizard without saving main configuration changes?", default_yes=False):
                        ASCIIColors.warning("Configuration not saved.")
                        sys.exit(1) # Exit script if user confirms discard
                    else:
                        continue # Stay in menu
                else:
                    # Already saved, confirm exit
                    ASCIIColors.info("Main configuration was previously saved.")
                    if ASCIIColors.confirm("Exit wizard?", default_yes=True):
                         break
                    else:
                         continue # Stay in menu
            elif callable(choice):
                try:
                    choice() # Call the selected method (e.g., self.configure_server)
                except Exception as action_err:
                     logger.error(f"Error executing menu action {getattr(choice, '__name__', 'N/A')}: {action_err}")
                     trace_exception(action_err)
                     ASCIIColors.error("An unexpected error occurred. Returning to menu.")
                     ASCIIColors.prompt("Press Enter to continue...")
            else:
                 logger.warning(f"Invalid choice returned from menu: {choice}")

    def _interactive_edit_section(
        self,
        section_config: Union[ConfigSection, ConfigGuard],
        section_name: str,
        skip_keys: Optional[List[str]] = None,
        binding_type: Optional[str] = None # Added for binding-specific suggestions
    ) -> bool:
        """
        Generic interactive editing helper based on ConfigGuard schema.

        Args:
            section_config: The ConfigGuard or ConfigSection object to edit.
            section_name: A user-friendly name for the section being edited.
            skip_keys: A list of keys within the schema to skip during editing.
            binding_type: The binding type name if editing a binding instance.

        Returns:
            True if any values were changed, False otherwise.
        """
        if skip_keys is None:
            skip_keys = []
        modified_in_section = False

        try:
            # Get the schema definition from the object
            if isinstance(section_config, ConfigGuard):
                 schema_definition = section_config.get_instance_schema_definition()
            elif isinstance(section_config, ConfigSection):
                 schema_definition = section_config._schema_definition # Access internal schema
            else:
                 raise TypeError("Input must be a ConfigGuard or ConfigSection object.")
            if not schema_definition:
                raise ValueError("Schema definition is empty or could not be retrieved.")
        except Exception as e:
             ASCIIColors.error(f"Could not get schema definition for '{section_name}': {e}")
             trace_exception(e)
             return False # Cannot proceed without schema

        sorted_keys = sorted(schema_definition.keys())

        for setting_key in sorted_keys:
            # Skip internal keys and explicitly skipped keys
            if setting_key.startswith('__') or setting_key in skip_keys:
                 continue

            setting_schema_dict = schema_definition[setting_key]
            # Extract schema properties safely using .get()
            setting_type = setting_schema_dict.get("type", "unknown")
            setting_help = setting_schema_dict.get("help", "")
            setting_options = setting_schema_dict.get("options")
            setting_default = setting_schema_dict.get("default")
            is_nullable = setting_schema_dict.get("nullable", False)
            is_secret = setting_schema_dict.get("secret", False)

            # Get current value, using default from schema as fallback
            current_value = getattr(section_config, setting_key, setting_default)
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
                    ASCIIColors.print(f"  (Default: {repr(setting_default)})", color=ASCIIColors.color_yellow)

                # --- Binding-Specific Suggestions (DALL-E model) ---
                if binding_type == "dalle_binding" and setting_key == "model":
                    ASCIIColors.print("  (Suggestions: dall-e-3, dall-e-2)", color=ASCIIColors.color_magenta)
                # --- End Suggestions ---

                if setting_options:
                     ASCIIColors.print(f"  (Options: {setting_options})", color=ASCIIColors.color_yellow)

                # --- Get User Input ---
                user_input_val: typing.Any = None
                input_type = setting_type.lower()

                # --- Special Handling for DALL-E Model Selection ---
                if binding_type == "dalle_binding" and setting_key == "model":
                    opts = [(name, name) for name in DALLE_MODEL_SUGGESTIONS]
                    opts.append(("(Enter Manually)", "##MANUAL##"))
                    opts.append(("(Keep Current)", "##KEEP##"))
                    menu = Menu(
                        f"Select value for {setting_key}",
                        mode='select_single',
                        item_color=ASCIIColors.color_cyan,
                        epilog_text=f"Current: {display_value}" # Use epilog
                    )
                    selected_action = menu.add_choices(opts).run()

                    if selected_action == "##KEEP##":
                        user_input_val = current_value
                        break
                    elif selected_action == "##MANUAL##":
                         user_input_val = ASCIIColors.prompt("Enter DALL-E model name manually: ").strip() or current_value
                    else: # A suggested model was selected
                         user_input_val = selected_action
                # --- End DALL-E Handling ---

                elif setting_options:
                    # Use Menu for options
                    opts = [(str(opt), opt) for opt in setting_options]
                    # Add nullable option if allowed
                    if is_nullable:
                        opts.append(("(Set to None)", None))
                    opts.append(("(Keep Current)", "##KEEP##"))

                    menu = Menu(
                        f"Select value for {setting_key}",
                        mode='select_single',
                        item_color=ASCIIColors.color_cyan,
                        epilog_text=f"Current: {display_value}" # Use epilog
                    )
                    selected_option_value = menu.add_choices(opts).run()

                    if selected_option_value == "##KEEP##":
                        user_input_val = current_value # Keep current
                        break # Move to next setting
                    else:
                        user_input_val = selected_option_value # Could be None if chosen
                elif input_type == "bool":
                    current_bool_val = bool(current_value) if current_value is not None else bool(setting_default)
                    user_input_val = ASCIIColors.confirm("Enable this setting?", default_yes=current_bool_val)
                elif input_type == "list":
                    ASCIIColors.print("  (Enter items separated by commas, e.g., item1, item2)")
                    raw_input = ASCIIColors.prompt("New list value (or Enter to keep current): ").strip()
                    if not raw_input: # User pressed Enter
                        user_input_val = current_value
                        break # Move to next setting
                    else:
                        user_input_val = [item.strip() for item in raw_input.split(',') if item.strip()] # Store non-empty strings
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
                        user_input_val = raw_input # Let ConfigGuard handle type conversion on set

                # --- Attempt to Set and Validate ---
                try:
                    value_before_set = getattr(section_config, setting_key, None)
                    # ConfigGuard handles type casting and validation here
                    setattr(section_config, setting_key, user_input_val)
                    value_after_set = getattr(section_config, setting_key)
                    display_after = "(******)" if is_secret and value_after_set else repr(value_after_set)
                    ASCIIColors.success(f"Set '{setting_key}' to: {display_after}")
                    # Check if value actually changed after potential type casting
                    if value_before_set != value_after_set:
                        modified_in_section = True
                    break # Valid input received, move to next setting
                except ValidationError as e:
                    ASCIIColors.error(f"Invalid value: {e}")
                    # Loop continues to re-prompt for this setting
                except Exception as e:
                    ASCIIColors.error(f"Error setting value: {e}")
                    trace_exception(e)
                    # Loop continues, maybe prompt again or offer skip? Let's re-prompt.

        return modified_in_section # Return True if any value was changed

    # --- Configuration Methods ---

    def apply_preset(self):
        """Applies a configuration preset to the main config and suggests instance setup."""
        if not self.config:
            ASCIIColors.error("Main config not initialized.")
            return

        print_wizard_title("Apply Configuration Preset")
        ASCIIColors.print("Presets provide starting configurations. You can still customize settings afterwards.")

        preset_options = [(name, name) for name in sorted(PRESETS.keys())] # Sort presets alphabetically
        menu = Menu(
            "Select Preset",
            mode='select_single',
            enable_filtering=True,
            item_color=ASCIIColors.color_cyan
            # No epilog needed here
        )
        selected_preset_name = menu.add_choices(preset_options).run()

        if not selected_preset_name:
             ASCIIColors.warning("Preset selection cancelled.")
             return

        preset_data = PRESETS.get(selected_preset_name)
        if not preset_data:
             ASCIIColors.error(f"Internal error: Preset data not found for '{selected_preset_name}'.")
             return

        ASCIIColors.info(f"Applying preset: '{selected_preset_name}'...")
        self.preset_suggested_configs = {} # Reset previous suggestions
        warnings: List[str] = []
        applied_bindings_map: Dict[str, str] = {}
        dynamic_section_updates: Dict[str, Dict[str, Any]] = {}

        # --- Apply preset values to main config sections ---
        for section_key, section_values in preset_data.items():
             if section_key == "suggested_instance_configs":
                  self.preset_suggested_configs = section_values # Store for later
                  continue

             if hasattr(self.config, section_key):
                  section_obj = getattr(self.config, section_key)
                  is_dynamic_section = section_key in ["bindings_map", "personalities_config"]

                  if is_dynamic_section:
                      # Handle dynamic sections separately using import_config
                      if isinstance(section_values, dict):
                          ASCIIColors.print(f"  - Preparing updates for dynamic section '{section_key}'")
                          # Clear existing keys in the section object if possible
                          if hasattr(section_obj, 'clear_dynamic_keys'): # Check for ConfigGuard method
                              section_obj.clear_dynamic_keys()
                              logger.debug(f"Cleared existing keys in dynamic section '{section_key}'.")
                          elif isinstance(section_obj, dict): # Fallback if it acts like a dict
                              section_obj.clear()
                              logger.debug(f"Cleared existing keys in dynamic section '{section_key}' (dict fallback).")
                          else:
                              logger.warning(f"Cannot clear dynamic section '{section_key}' (no clear method). Preset will overwrite/add keys.")
                          # Store updates to be applied via import_config later
                          dynamic_section_updates[section_key] = section_values
                          if section_key == "bindings_map":
                              # Keep track of the map intended by the preset
                              applied_bindings_map = section_values.copy()
                      else:
                          warnings.append(f"Preset data for dynamic section '{section_key}' is not a dictionary. Ignoring.")
                  elif isinstance(section_obj, ConfigSection):
                      # Apply values to standard sections directly
                      for key, value in section_values.items():
                          if hasattr(section_obj, key):
                              try:
                                  setattr(section_obj, key, value)
                                  ASCIIColors.success(f"  - Set {section_key}.{key} = {value!r}")
                              except ValidationError as e:
                                  warnings.append(f"Preset value invalid for '{section_key}.{key}': {e}. Ignoring.")
                              except Exception as e:
                                  warnings.append(f"Error applying '{section_key}.{key}': {e}. Ignoring.")
                                  trace_exception(e)
                          else:
                              warnings.append(f"Preset key '{section_key}.{key}' not found in schema section '{section_key}'. Ignoring.")
                  else:
                      warnings.append(f"Preset targets '{section_key}', which is not a ConfigSection object. Ignoring.")
             else:
                 warnings.append(f"Preset targets unknown main config section '{section_key}'. Ignoring.")

        # --- Apply dynamic section updates using import_config ---
        if dynamic_section_updates:
             try:
                 logger.debug(f"Importing dynamic section updates via import_config: {dynamic_section_updates}")
                 # ignore_unknown=True allows adding keys to dynamic sections
                 self.config.import_config(dynamic_section_updates, ignore_unknown=True)
                 ASCIIColors.success("  - Successfully applied dynamic section updates (e.g., bindings_map).")
             except Exception as e:
                 warnings.append(f"Error applying dynamic section updates via import_config: {e}")
                 trace_exception(e)

        ASCIIColors.success(f"Preset '{selected_preset_name}' applied to main configuration.")
        if warnings:
             ASCIIColors.warning("\nPreset Application Warnings:")
             for warn in warnings:
                 ASCIIColors.yellow(f" - {warn}")

        self.saved_successfully = False # Mark main config as modified
        self._model_list_cache = {} # Clear model cache after potentially changing defaults

        # --- Re-resolve Paths After Preset Application ---
        # This is crucial before configuring instances which relies on absolute instance path
        if hasattr(self.config, 'paths'):
            try:
                ASCIIColors.info("Re-resolving path relationships after applying preset...")
                _wizard_resolve_paths(self.config.paths, self.server_root)
                _wizard_update_derived_paths(self.config.paths, Path(self.config.paths.config_base_dir))
                ASCIIColors.success("Path relationships updated.")
            except Exception as path_e:
                 logger.error(f"Error re-resolving paths after preset: {path_e}")
                 ASCIIColors.warning("Path resolution after preset failed. Instance configuration might fail.")


        # --- Offer to configure suggested instances ---
        if self.preset_suggested_configs:
             # Verify the bindings map was actually updated as expected
             final_bindings_map = getattr(self.config, "bindings_map", {}).get_dict() or {}
             successful_bindings = {
                 k: v for k, v in applied_bindings_map.items()
                 if final_bindings_map.get(k) == v
             }

             if not successful_bindings:
                  logger.error("Bindings map was not populated correctly by preset. Cannot configure suggested instances.")
             elif ASCIIColors.confirm("\nConfigure suggested binding instances now?", default_yes=True):
                  self._configure_suggested_instances(successful_bindings)
             else:
                  ASCIIColors.info("Skipping instance configuration. You can add/edit instances later using the installer menu (Option 3).")

        ASCIIColors.prompt("\nPress Enter to return to the main menu...")


    def _configure_suggested_instances(self, bindings_to_configure: Dict[str, str]):
        """Guides user through configuring binding instances suggested by a preset."""
        if not self.preset_suggested_configs or not self.config or not self.config_base_dir:
            logger.error("Cannot configure suggested instances: Missing data or main config.")
            return
        if not bindings_to_configure:
             logger.warning("No bindings map provided. Skipping suggested instance configuration.")
             return

        print_wizard_title("Configure Suggested Binding Instances")
        ASCIIColors.print("Based on the preset, we will now configure the required binding instances.")
        ASCIIColors.print("Review and modify the suggested settings.")

        # Get instance folder path from main config (should be absolute after preset path fix)
        instance_folder_str = getattr(self.config.paths, "instance_bindings_folder", None)
        if not instance_folder_str:
             ASCIIColors.error("Instance bindings folder path not set in main config. Cannot save instances.")
             return
        instance_dir = Path(instance_folder_str)
        # Check if it's absolute NOW
        if not instance_dir.is_absolute():
             ASCIIColors.error(f"Instance folder path '{instance_dir}' is still not absolute after preset path resolution. Configuration error.")
             ASCIIColors.error("Please check the 'paths' section and preset logic.")
             return
        try:
            instance_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        except OSError as e:
             ASCIIColors.error(f"Could not create binding instances directory: {instance_dir}. Error: {e}")
             return

        # Load available binding type definitions (schemas)
        available_binding_types = _get_binding_types_from_disk(self.config, self.server_root)
        if not available_binding_types:
            ASCIIColors.error("Failed discover binding type definitions. Cannot configure instances.")
            return

        # Get encryption key from main config for instance saving
        enc_key_str = getattr(self.config.security, "encryption_key", None)
        enc_key = enc_key_str.encode() if enc_key_str else None

        configured_any_instance = False
        # Iterate through the bindings map provided by the preset application logic
        for instance_name, binding_type in bindings_to_configure.items():
            if instance_name not in self.preset_suggested_configs:
                logger.debug(f"No suggested config found for instance '{instance_name}' in preset. Skipping.")
                continue

            suggested_config_data = self.preset_suggested_configs[instance_name]
            # Preset data should ideally include 'type', but double-check
            if suggested_config_data.get("type") != binding_type:
                 logger.warning(f"Preset suggestion type mismatch for '{instance_name}'. Expected '{binding_type}', got '{suggested_config_data.get('type')}'. Proceeding with '{binding_type}'.")

            # Get the schema for this binding type
            if binding_type not in available_binding_types:
                logger.error(f"Binding type definition '{binding_type}' for instance '{instance_name}' not found. Skipping.")
                continue
            type_info = available_binding_types[binding_type]
            type_card_data = type_info.get("card")
            original_instance_schema = type_card_data.get("instance_schema") if isinstance(type_card_data, dict) else None
            if not original_instance_schema or not isinstance(original_instance_schema, dict):
                 logger.error(f"Binding card for '{binding_type}' missing or has invalid 'instance_schema'. Cannot configure '{instance_name}'.")
                 continue

            # === Diffusers Model Check/Suggestion (Moved inside the loop) ===
            if binding_type == "diffusers_binding":
                # Check the models folder specified for *this instance*
                instance_models_folder_str = suggested_config_data.get("models_folder", "models/diffusers_models")
                diffusers_model_dir = Path(instance_models_folder_str)
                if not diffusers_model_dir.is_absolute():
                    diffusers_model_dir = (self.server_root / diffusers_model_dir).resolve()

                try:
                    # Check if exists and has no subdirs (excluding hidden)
                    has_model_subdirs = diffusers_model_dir.is_dir() and \
                                        any(d.is_dir() for d in diffusers_model_dir.iterdir() if not d.name.startswith('.'))

                    if diffusers_model_dir.is_dir() and not has_model_subdirs:
                        ASCIIColors.warning(f"\nWARNING: The diffusers models folder for instance '{instance_name}' ({diffusers_model_dir}) is empty or contains no model subdirectories!")
                        ASCIIColors.print("Consider downloading a model from Hugging Face Hub into this folder.")
                        ASCIIColors.print("Suggested models (download the *entire folder* for each):")
                        for name, hub_id in DIFFUSERS_SUGGESTIONS:
                             ASCIIColors.print(f"  - {name}: `{hub_id}`")
                        ASCIIColors.prompt("Press Enter after reviewing suggestions...")
                    elif not diffusers_model_dir.is_dir():
                         ASCIIColors.warning(f"\nWARNING: The diffusers models folder for instance '{instance_name}' ({diffusers_model_dir}) does not exist.")
                except Exception as e:
                     logger.warning(f"Error checking diffusers dir {diffusers_model_dir}: {e}")
            # === End Diffusers Check ===

            # Prepare instance config path (defaulting to YAML)
            instance_file_path = instance_dir / f"{instance_name}.yaml"
            handler = YamlHandler()
            if not check_handler_dependency(handler.__class__): # Check dependency of the class
                logger.error("PyYAML required for saving instance config. Install 'pip install pyyaml'. Skipping instance.")
                continue

            ASCIIColors.print(f"\n--- Configuring Instance: '{instance_name}' (Type: {binding_type}) ---", color=ASCIIColors.color_bright_magenta)
            ASCIIColors.print(f"File will be created/updated at: {instance_file_path}")
            ASCIIColors.print("Review and confirm/edit the settings below (defaults from preset applied).")

            config_instance = None
            try:
                # Add fixed fields to schema for validation and clarity
                # Ensure __version__ key exists in the instance schema from card
                schema_copy = original_instance_schema.copy()
                schema_copy.setdefault("__version__", "0.1.0") # Add default if missing
                schema_copy.setdefault("type", {"type": "str", "default": binding_type})
                schema_copy.setdefault("binding_instance_name", {"type": "str", "default": instance_name})

                # Determine instance version from card schema, fallback if missing
                instance_schema_version = schema_copy.get("__version__", "0.1.0")

                # Initialize ConfigGuard for the instance
                config_instance = ConfigGuard(
                    schema=schema_copy, # Pass schema without version key here
                    instance_version=instance_schema_version, # Pass version separately
                    config_path=instance_file_path,
                    handler=handler,
                    autosave=False,
                    encryption_key=enc_key,
                )

                # Apply preset suggestions to the ConfigGuard object
                modified_by_preset = False
                logger.debug(f"Applying preset suggestions for '{instance_name}': {suggested_config_data}")
                preset_applied_info = []
                preset_failed_info = []
                for key, preset_value in suggested_config_data.items():
                     # Skip internal/fixed keys and keys not in the actual schema
                     if key in ["binding_type", "binding_instance_name", "__version__", "type"] or key not in original_instance_schema:
                         continue
                     # Set attribute if it exists in the schema-based object
                     if hasattr(config_instance, key):
                        try:
                            # Set value, ConfigGuard handles validation
                            setattr(config_instance, key, preset_value)
                            preset_applied_info.append(f"'{key}' = {repr(preset_value)}")
                            modified_by_preset = True
                        except ValidationError as e:
                            preset_failed_info.append(f"'{key}' ({repr(preset_value)}): {e}")
                        except Exception as set_e:
                            logger.error(f"Error setting preset value '{key}' for '{instance_name}': {set_e}")
                            trace_exception(set_e)

                if preset_applied_info:
                    ASCIIColors.info(f"Applied preset values: {'; '.join(preset_applied_info)}.")
                if preset_failed_info:
                    ASCIIColors.warning(f"Invalid preset values ignored: {'; '.join(preset_failed_info)}.")
                if not modified_by_preset:
                    ASCIIColors.info("Using schema defaults (no valid preset values applied).")

                # Allow interactive editing, passing binding type for context
                instance_modified = self._interactive_edit_section(
                    section_config=config_instance,
                    section_name=f"Instance '{instance_name}'",
                    skip_keys=["__version__", "type", "binding_instance_name"], # Don't edit fixed keys
                    binding_type=binding_type # Pass binding type for suggestions
                )

                # Confirm save if modified by preset or user
                if instance_modified or modified_by_preset:
                    if ASCIIColors.confirm(f"Create/Update config file for instance '{instance_name}'?", default_yes=True):
                        try:
                            instance_file_path.parent.mkdir(parents=True, exist_ok=True)
                            # Save values, including secrets if key exists
                            config_instance.save(mode='values') # Only save values
                            ASCIIColors.success(f"Instance configuration SAVED: {instance_file_path}")
                            configured_any_instance = True
                        except Exception as e:
                            ASCIIColors.error(f"Failed save instance '{instance_name}': {e}")
                            trace_exception(e)
                    else:
                        ASCIIColors.warning(f"Configuration for instance '{instance_name}' not saved.")
                else:
                    ASCIIColors.info(f"No changes made to defaults for instance '{instance_name}'. Not saved.")

            except SchemaError as schema_e:
                 ASCIIColors.error(f"Schema Error configuring instance '{instance_name}': {schema_e}")
                 trace_exception(schema_e)
            except Exception as e:
                ASCIIColors.error(f"Error during config setup for instance '{instance_name}': {e}")
                trace_exception(e)

        # --- End Loop ---
        if configured_any_instance:
            ASCIIColors.success("\nFinished configuring suggested binding instances.")
        else:
            ASCIIColors.info("\nNo suggested binding instances were configured or saved.")


    def _configure_generic_section(self, section_name: str):
        """Generic helper to configure a standard section of the main config."""
        # Implementation remains the same
        if not self.config:
            ASCIIColors.error("Main config not initialized.")
            return
        print_wizard_title(f"{section_name.replace('_',' ').title()} Configuration")

        section_obj = getattr(self.config, section_name, None)
        if not section_obj or not isinstance(section_obj, ConfigSection):
            ASCIIColors.error(f"Section '{section_name}' not found or invalid in config object!")
            return

        ASCIIColors.info(f"Current settings for [{section_name}]:")
        # Use the interactive editing helper
        modified = self._interactive_edit_section(
            section_config=section_obj,
            section_name=section_name,
            skip_keys=["__version__"] # Skip internal keys
        )

        if modified:
             self.saved_successfully = False # Mark main config as needing save

        ASCIIColors.prompt("\nPress Enter to return to the main menu...")

    # --- Section Configuration Methods ---
    def configure_server(self): self._configure_generic_section("server")
    def configure_logging(self): self._configure_generic_section("logging")
    def configure_resource_manager(self): self._configure_generic_section("resource_manager")
    def configure_webui(self): self._configure_generic_section("webui")

    def configure_paths(self):
        """Configure paths, handling relative/absolute concept."""
        # Implementation remains the same
        if not self.config: return
        print_wizard_title("Path Configuration")
        ASCIIColors.print("Define paths for models, personalities, bindings, etc.")
        ASCIIColors.print("Paths can be absolute or relative to the server root directory:")
        ASCIIColors.print(f"Server Root: {self.server_root}")
        ASCIIColors.print("NOTE: Relative paths are stored as entered; resolution happens at server start.")

        paths_section = getattr(self.config, "paths", None)
        if not paths_section or not isinstance(paths_section, ConfigSection):
            ASCIIColors.error("Paths section not found or invalid."); return

        try:
            schema_definition = paths_section._schema_definition
        except Exception as e:
            ASCIIColors.error(f"Could not get paths schema: {e}"); return

        sorted_keys = sorted(schema_definition.keys())
        modified_in_section = False

        for setting_key in sorted_keys:
            if setting_key.startswith('__'): continue

            # Display read-only derived paths differently
            # config_base_dir is set earlier, instance_bindings_folder derived from it
            if setting_key in ['config_base_dir', 'instance_bindings_folder']:
                current_value_str = getattr(paths_section, setting_key, "N/A")
                help_text = schema_definition.get(setting_key,{}).get('help',"Path setting.")
                ASCIIColors.print(f"\n- {setting_key}: ({help_text})", color=ASCIIColors.color_cyan)
                ASCIIColors.print(f"  (Current Resolved: '{current_value_str}')", color=ASCIIColors.color_yellow)
                if setting_key == 'config_base_dir':
                    ASCIIColors.print(f"  (Set in Step 1)", color=ASCIIColors.color_magenta)
                else:
                    ASCIIColors.print(f"  (Derived from config_base_dir)", color=ASCIIColors.color_magenta)
                continue

            # Allow editing other paths
            setting_schema_dict = schema_definition[setting_key]
            setting_help = setting_schema_dict.get("help", "")
            setting_default = setting_schema_dict.get("default")
            current_value_str = getattr(paths_section, setting_key, setting_default) # Get current raw value

            prompt_text = f"- {setting_key}: {setting_help}"
            ASCIIColors.print(f"\n{prompt_text}", color=ASCIIColors.color_cyan)
            ASCIIColors.print(f"  (Current: '{current_value_str}')", color=ASCIIColors.color_yellow)
            ASCIIColors.print(f"  (Default: '{setting_default}')", color=ASCIIColors.color_yellow)

            while True:
                user_input = ASCIIColors.prompt("New path (or Enter to keep current): ").strip()
                if not user_input: break # Keep current value

                try:
                    value_before_set = getattr(paths_section, setting_key, None)
                    # Validate type is string if schema requires it
                    if not isinstance(user_input, str) and setting_schema_dict.get("type") == "str":
                        raise ValidationError("Input must be a string path.")
                    # Set the value (ConfigGuard handles basic type)
                    setattr(paths_section, setting_key, user_input)
                    value_after_set = getattr(paths_section, setting_key)
                    ASCIIColors.success(f"Set '{setting_key}' path to: '{user_input}'")
                    if value_before_set != value_after_set:
                        modified_in_section = True
                    break # Exit input loop for this setting
                except ValidationError as e:
                    ASCIIColors.error(f"Invalid value: {e}") # Re-prompt
                except Exception as e:
                    ASCIIColors.error(f"Error processing path: {e}")
                    trace_exception(e)
                    break # Stop on unexpected error for this setting

        if modified_in_section:
            self.saved_successfully = False # Mark main config as modified
            # Re-resolve paths immediately after changes for feedback (optional)
            try:
                 ASCIIColors.info("Re-resolving path relationships...")
                 _wizard_resolve_paths(paths_section, self.server_root)
                 _wizard_update_derived_paths(paths_section, Path(getattr(paths_section,"config_base_dir")))
                 ASCIIColors.success("Path relationships updated based on changes.")
            except Exception as path_e:
                logger.error(f"Error re-resolving paths: {path_e}")

        ASCIIColors.print("\nNote: Final path resolution and directory creation happen at server start.")
        ASCIIColors.prompt("\nPress Enter to return to the main menu...")


    def configure_defaults(self):
        """Configure default binding instance names and model names."""
        # Implementation mostly the same, just add epilog to menus
        if not self.config: return
        print_wizard_title("Default Bindings & Models Configuration")
        ASCIIColors.print("Select the default binding *instance* and model for each generation type.")

        defaults_section = getattr(self.config, "defaults", None)
        bindings_map_section = getattr(self.config, "bindings_map", None)
        if not defaults_section or not bindings_map_section:
            ASCIIColors.error("Defaults or Bindings Map section missing or invalid in main config!"); return

        try:
            available_instance_map = bindings_map_section.get_dict() or {}
        except Exception as e:
            logger.error(f"Could not get bindings_map dictionary: {e}"); available_instance_map = {}

        modified = False
        default_schema = MAIN_SCHEMA.get("defaults", {}).get("schema", {})
        modality_types = ["ttt", "tti", "tts", "stt", "ttv", "ttm"]

        for mod_type in modality_types:
            binding_key = f"{mod_type}_binding"
            model_key = f"{mod_type}_model"

            # Configure Default Binding Instance for Modality
            if binding_key in default_schema:
                current_binding_instance = getattr(defaults_section, binding_key, None)
                ASCIIColors.print(f"\n--- Configure Default {mod_type.upper()} Binding Instance ---", color=ASCIIColors.color_bright_cyan)
                binding_help = default_schema.get(binding_key, {}).get("help", "")
                if binding_help: ASCIIColors.print(binding_help)
                current_info = f"Current Default Instance: {current_binding_instance or 'None'}"
                ASCIIColors.print(f"  ({current_info})", color=ASCIIColors.color_yellow)

                # Filter instances
                compatible_instances = []
                for instance_name, binding_type in available_instance_map.items():
                     supported_mods = BINDING_MODALITY_MAP.get(binding_type, [])
                     modality_check = 'tti_vision' if mod_type == 'tti' and 'tti_vision' in supported_mods else mod_type
                     if modality_check in supported_mods: compatible_instances.append(instance_name)

                # Prepare menu options
                binding_options: List[Tuple[str, Optional[str]]] = [(name, name) for name in sorted(compatible_instances)]
                if not binding_options: ASCIIColors.warning(f"  No compatible binding instances found for modality '{mod_type}'.")
                binding_options.extend([("(None - Use Request/Personality)", None), ("(Keep Current)", "##KEEP##")])

                # Show menu with epilog
                menu = Menu(
                    f"Select Default Binding Instance for {mod_type.upper()}",
                    mode='select_single',
                    item_color=ASCIIColors.color_cyan,
                    epilog_text=current_info # Use epilog
                )
                selected_binding = menu.add_choices(binding_options).run()

                # Determine new value
                new_binding_val = selected_binding if selected_binding != "##KEEP##" else current_binding_instance

                # Update config if changed
                if new_binding_val != current_binding_instance:
                    try:
                        setattr(defaults_section, binding_key, new_binding_val)
                        ASCIIColors.success(f"Set default {mod_type.upper()} binding instance to: {new_binding_val or 'None'}")
                        modified = True
                        if getattr(defaults_section, model_key, None) is not None:
                            setattr(defaults_section, model_key, None); ASCIIColors.warning(f"Cleared default {mod_type.upper()} model.")
                    except Exception as e: ASCIIColors.error(f"Error setting {binding_key}: {e}"); trace_exception(e)

                current_binding_instance = getattr(defaults_section, binding_key, None)

            # Configure Default Model for the Selected Binding Instance
            if model_key in default_schema:
                current_model = getattr(defaults_section, model_key, None)
                ASCIIColors.print(f"\n--- Configure Default {mod_type.upper()} Model ---", color=ASCIIColors.color_bright_cyan)
                model_help = default_schema.get(model_key, {}).get("help", "");
                if model_help: ASCIIColors.print(model_help)

                if not current_binding_instance:
                    ASCIIColors.warning(f"  Cannot set default model: Default {mod_type.upper()} binding not selected.")
                    if current_model is not None: setattr(defaults_section, model_key, None); modified = True
                    continue

                current_model_info = f"Current Default Model: {current_model or 'None'}"
                ASCIIColors.info(f"  (For Binding Instance: '{current_binding_instance}')", color=ASCIIColors.color_cyan)
                ASCIIColors.info(f"  ({current_model_info})", color=ASCIIColors.color_yellow)

                model_options = self._list_models_for_binding(current_binding_instance)
                selected_model_name = None
                if model_options:
                    menu_options = model_options[:]; menu_options.extend([ ("(Enter Manually)", "##MANUAL##"), ("(Clear Default Model)", None), ("(Keep Current)", "##KEEP##") ])
                    menu = Menu(
                        f"Select Default Model for {current_binding_instance}",
                        mode='select_single',
                        enable_filtering=True,
                        item_color=ASCIIColors.color_cyan,
                        epilog_text=current_model_info # Use epilog
                    )
                    selected_model_or_action = menu.add_choices(menu_options).run()
                    if selected_model_or_action == "##MANUAL##": selected_model_name = ASCIIColors.prompt(f"Enter model name/ID for '{current_binding_instance}': ").strip() or None
                    elif selected_model_or_action != "##KEEP##": selected_model_name = selected_model_or_action
                    else: selected_model_name = current_model
                else:
                    ASCIIColors.warning(f"  Could not list models for '{current_binding_instance}'. Please enter manually.")
                    selected_model_name = ASCIIColors.prompt(f"Enter model name/ID for '{current_binding_instance}' (or Enter): ").strip() or current_model

                if selected_model_name != current_model:
                     try: setattr(defaults_section, model_key, selected_model_name); ASCIIColors.success(f"Set default {mod_type.upper()} model to: {selected_model_name or 'None'}"); modified = True
                     except Exception as e: ASCIIColors.error(f"Error setting {model_key}: {e}"); trace_exception(e)

        if modified: self.saved_successfully = False; self._model_list_cache = {}
        ASCIIColors.prompt("\nPress Enter to return...")

    def configure_security(self):
        """Configure API keys and encryption key."""
        # Implementation remains the same, menus here less likely to need epilog
        if not self.config: return
        print_wizard_title("Security Configuration")
        sec_section = getattr(self.config, "security", None)
        if not sec_section:
            ASCIIColors.error("Security section not found or invalid in config object!")
            return

        modified = False
        original_keys = set(getattr(sec_section, "allowed_api_keys", []))
        original_enc_key = getattr(sec_section, "encryption_key", None)

        # --- API Keys ---
        ASCIIColors.bold("\n--- API Keys ---", color=ASCIIColors.color_bright_yellow)
        ASCIIColors.print("Define keys clients must send in 'X-API-Key' header.")
        current_keys = list(getattr(sec_section, "allowed_api_keys", []))
        while True:
            ASCIIColors.print("\nCurrent Allowed API Keys:", color=ASCIIColors.color_yellow)
            if not current_keys:
                ASCIIColors.warning("  (None configured - WARNING: Server endpoints may be unprotected!)")
            else:
                for i, key in enumerate(current_keys):
                    obscured = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else key[:4]+"..."
                    ASCIIColors.print(f"  {i+1}: {obscured}")

            # API Key Actions Menu
            menu = Menu("API Key Actions", item_color=ASCIIColors.color_cyan)
            menu.add_action("Add New Key", lambda: "add")
            if current_keys:
                 menu.add_action("Remove Key", lambda: "remove")
            menu.add_action("Suggest Key", lambda: "suggest")
            menu.add_action("Done with API Keys", lambda: "done_keys")
            action = menu.run()

            if action == "add":
                new_key = ASCIIColors.prompt("Enter new API key (leave blank to cancel): ").strip()
                if new_key:
                    if new_key not in current_keys:
                        current_keys.append(new_key)
                        ASCIIColors.success("Key added to list.")
                    else: ASCIIColors.warning("Key already exists in list.")
            elif action == "remove" and current_keys:
                 # Use _select_from_list helper
                 remove_options = [(f"{i+1}: {k[:4]}...", i) for i, k in enumerate(current_keys)]
                 idx_to_remove = _select_from_list(remove_options, "Select Key to Remove", epilog="Select key number to remove.")
                 if idx_to_remove is not None :
                     removed_key = current_keys.pop(idx_to_remove)
                     ASCIIColors.success(f"Key removed from list.")
                 else: ASCIIColors.warning("Removal cancelled.")
            elif action == "suggest":
                 suggested = suggest_api_key()
                 ASCIIColors.print(f"Suggested key: {suggested}")
                 if ASCIIColors.confirm("Add this suggested key to the list?", default_yes=True):
                     if suggested not in current_keys: current_keys.append(suggested); ASCIIColors.success("Suggested key added.")
                     else: ASCIIColors.warning("Suggested key already exists.")
            elif action == "done_keys" or action is None: break

        # Update config object if keys changed
        if set(current_keys) != original_keys:
             try:
                 sec_section.allowed_api_keys = current_keys
                 ASCIIColors.success("API Key list updated in configuration object.")
                 modified = True
             except Exception as e: ASCIIColors.error(f"Failed to update API keys: {e}"); trace_exception(e)

        # --- Encryption Key ---
        ASCIIColors.bold("\n--- Binding Config Encryption Key ---", color=ASCIIColors.color_bright_yellow)
        ASCIIColors.print("Optional: Provide key to encrypt sensitive data in binding instance configs."); ASCIIColors.print("Requires 'cryptography' library (`pip install cryptography`)."); ASCIIColors.print("If set/changed, re-save relevant binding configs!")
        current_enc_key = getattr(sec_section, "encryption_key", None); current_display = "(Set - value hidden)" if current_enc_key else "(Not set)"; ASCIIColors.print(f"\nCurrent Encryption Key: {current_display}", color=ASCIIColors.color_yellow)
        crypto_installed = False;
        try:
            import cryptography
            crypto_installed = True
        except ImportError:
            crypto_installed = False
        enc_menu = Menu("Encryption Key Actions");
        if crypto_installed: enc_menu.add_action("Generate New Key", lambda: "generate")
        else: enc_menu.add_action("Generate New Key (Requires 'cryptography')", lambda: "generate_disabled")
        enc_menu.add_action("Enter Key Manually", lambda: "manual");
        if current_enc_key: enc_menu.add_action("Remove Key (Disable Encryption)", lambda: "remove_enc")
        enc_menu.add_action("Keep Current / Skip", lambda: "skip"); enc_action = enc_menu.run(); new_enc_key = current_enc_key
        if enc_action == "generate":
            try: new_enc_key = generate_encryption_key(as_string=True); ASCIIColors.success(f"Generated key: {new_enc_key}"); ASCIIColors.warning("Store securely!")
            except Exception as gen_e: ASCIIColors.error(f"Error generating key: {gen_e}")
        elif enc_action == "generate_disabled": ASCIIColors.warning("Cannot generate key. Install 'cryptography'.")
        elif enc_action == "manual":
            manual_key = ASCIIColors.prompt("Paste Fernet key (leave blank to cancel): ", hide_input=True).strip();
            if manual_key:
                if len(manual_key) < 40: ASCIIColors.warning("Key seems short.");
                new_enc_key = manual_key # Set anyway
            else: ASCIIColors.warning("Manual entry cancelled.")
        elif enc_action == "remove_enc":
            if ASCIIColors.confirm("Remove key? Secrets won't be encrypted.", default_yes=False): new_enc_key = None; ASCIIColors.info("Key removed.")
            else: ASCIIColors.warning("Cancelled.")
        elif enc_action == "skip" or enc_action is None: ASCIIColors.info("Encryption key setting kept.")

        # Update config object if key changed
        if new_enc_key != original_enc_key:
            try:
                setattr(sec_section, "encryption_key", new_enc_key)
                if new_enc_key: ASCIIColors.success("Encryption key setting updated.")
                else: ASCIIColors.success("Encryption key removed (encryption disabled).")
                modified = True
            except Exception as e: ASCIIColors.error(f"Failed update encryption key: {e}"); trace_exception(e)

        if modified: self.saved_successfully = False # Mark main config changed
        ASCIIColors.prompt("\nPress Enter to return to the main menu...")


    def manage_binding_instances_cli(self):
        """CLI Submenu for managing binding instances using ASCIIColors Menu."""
        print_wizard_title("Manage Binding Instances (CLI)")

        main_config_obj = self.config
        if not main_config_obj:
            ASCIIColors.error("Could not load main configuration object (self.config). Cannot manage instances.")
            return

        # Get necessary info from main config
        enc_key_str = getattr(main_config_obj.security, "encryption_key", None)
        enc_key = enc_key_str.encode() if enc_key_str else None
        instance_folder_str = getattr(main_config_obj.paths, "instance_bindings_folder", "")
        instance_folder = Path(instance_folder_str) if instance_folder_str else None

        if not instance_folder or not instance_folder.is_dir():
            ASCIIColors.error(f"Instance configuration folder is invalid or missing: {instance_folder}")
            ASCIIColors.error("Verify the 'instance_bindings_folder' path in your main config.")
            return

        # Load available binding types only once per entry into this menu
        available_binding_types = _get_binding_types_from_disk(main_config_obj, self.server_root)

        # --- Generate Epilog Text ---
        epilog_parts = ["Current Binding Instances:"]
        instance_infos = self._get_binding_instances_info(main_config_obj)
        bindings_map = getattr(main_config_obj, "bindings_map", {}).get_dict() or {}

        if not instance_infos:
            epilog_parts.append("  (None)")
        else:
            sorted_instances = sorted(instance_infos, key=lambda x: x[0])
            for name, path, info in sorted_instances:
                map_type = bindings_map.get(name)
                file_type = info.get('type_from_file')
                type_info = f"Type: {map_type or 'Not Mapped!'}"
                if file_type and file_type != "ErrorLoading" and file_type != "ErrorLoading(NotDict)" and file_type != "N/A":
                    if map_type and map_type != file_type:
                        type_info += f" (File: {file_type} - MISMATCH!)"
                    else:
                            type_info += f" (File: {file_type})"

                epilog_parts.append(f"  - {name} ({path.name}) - {type_info}")
        epilog_text = "\n".join(epilog_parts)
        # --- End Epilog Generation ---

        # --- Build Main Management Menu ---
        menu = Menu(
            "Binding Instance Management",
            item_color=ASCIIColors.color_cyan,
            title_color=ASCIIColors.color_bright_yellow,
            epilog_text=epilog_text # Display current instances
        )
        def add_action():
            # --- RELOAD available types EACH time add is selected ---
            # This ensures we pick up changes if the user modifies paths
            # or adds bindings without restarting the wizard menu loop.
            available_binding_types = _get_binding_types_from_disk(main_config_obj, self.server_root)
            # ------------------------------------------------------

            if not available_binding_types:
                ASCIIColors.error("Cannot add instance: No binding types were discovered.")
                ASCIIColors.warning("Please check:")
                ASCIIColors.warning(f"  - 'example_bindings_folder' path in main config: {getattr(main_config_obj.paths, 'example_bindings_folder', 'N/A')}")
                ASCIIColors.warning(f"  - 'bindings_folder' path in main config: {getattr(main_config_obj.paths, 'bindings_folder', 'N/A')}")
                ASCIIColors.warning("  - Ensure binding subfolders contain __init__.py and a valid binding_card.yaml.")
                ASCIIColors.prompt("Press Enter to continue...") # Pause so user can read
                return # Go back to the management menu

            # 1. Select Binding Type using Menu
            type_options = [ (f"{info.get('display_name', name)} ({name})", name) for name, info in sorted(available_binding_types.items()) ]
            selected_type = _select_from_list(type_options, "Select Binding Type for New Instance:")
            if not selected_type:
                ASCIIColors.warning("Cancelled type selection.")
                return # Go back to the management menu

            type_info = available_binding_types[selected_type]
            instance_schema = type_info.get("card", {}).get("instance_schema")
            if not instance_schema or not isinstance(instance_schema, dict):
                ASCIIColors.error(f"Invalid or missing instance schema for type '{selected_type}'. Cannot add."); return

            # 2. Get Unique Instance Name
            instance_name = ""
            current_instance_infos = self._get_binding_instances_info(main_config_obj)
            current_bindings_map = getattr(main_config_obj, "bindings_map", {}).get_dict() or {}
            existing_names = [name for name, _, _ in current_instance_infos] + list(current_bindings_map.keys())
            while True:
                    instance_name = ASCIIColors.prompt(f"Enter unique name for this '{selected_type}' instance: ").strip()
                    if not instance_name: ASCIIColors.warning("Name cannot be empty."); return
                    if instance_name in existing_names: ASCIIColors.warning("Instance name already exists or is mapped."); return
                    if not instance_name.isidentifier(): ASCIIColors.warning("Invalid name (use letters, numbers, underscore, not starting with number)."); return
                    break # Valid name received

            # 3. Choose File Format (Default to YAML)
            selected_format = ".yaml"
            handler_class = self.handler_map.get(selected_format)
            if not handler_class or not check_handler_dependency(handler_class):
                    ASCIIColors.error("YAML handler/dependency missing. Cannot save instance."); return

            instance_file_path = instance_folder / f"{instance_name}{selected_format}"

            ASCIIColors.print(f"\n--- Configuring '{instance_name}' (Type: {selected_type}) ---")
            ASCIIColors.print(f"File will be: {instance_file_path}")
            config_instance = None
            try:
                    # 4. Prepare Schema and Initialize ConfigGuard
                    schema_copy = instance_schema.copy()
                    schema_copy.setdefault("__version__", "0.1.0")
                    schema_copy.setdefault("type", {"type": "str", "default": selected_type})
                    schema_copy.setdefault("binding_instance_name", {"type": "str", "default": instance_name})
                    instance_schema_version = schema_copy.get("__version__", "0.1.0")

                    config_instance = ConfigGuard(
                        schema=schema_copy, instance_version=instance_schema_version,
                        config_path=instance_file_path, handler=handler_class(),
                        autosave=False, encryption_key=enc_key
                    )
                    setattr(config_instance, "type", selected_type)
                    setattr(config_instance, "binding_instance_name", instance_name)

                    # 5. Ask about Customization using Menu
                    customize_options = [
                        ("Customize Settings Now", True),
                        ("Use Defaults (save directly)", False)
                    ]
                    customize = _select_from_list(customize_options, "Customize or use defaults?")
                    if customize is None: # User cancelled customization choice
                        ASCIIColors.warning("Cancelled adding instance."); return

                    modified_by_user = False
                    if customize:
                        modified_by_user = self._interactive_edit_section(
                            section_config=config_instance,
                            section_name=f"Instance '{instance_name}' Settings",
                            skip_keys=["__version__", "type", "binding_instance_name"],
                            binding_type=selected_type
                        )

                    # 6. Confirm Save using confirm
                    save_action = "Create" if not instance_file_path.exists() else "Update"
                    prompt_msg = f"{save_action} '{instance_name}' config " + \
                                ("with defaults" if not customize else "with modifications") + "?"

                    # Save if customized, or if using defaults, or if user confirms modifications
                    should_save = (customize and modified_by_user) or \
                                (not customize) or \
                                (customize and not modified_by_user and ASCIIColors.confirm(prompt_msg, default_yes=True))

                    if should_save:
                        try:
                            instance_file_path.parent.mkdir(parents=True, exist_ok=True)
                            config_instance.save(mode='values')
                            ASCIIColors.success(f"Instance configuration SAVED: {instance_file_path}")

                            # 7. Update and Save Main Config Map
                            try:
                                bindings_map_section = getattr(main_config_obj, "bindings_map")
                                bindings_map_section[instance_name] = selected_type
                                main_config_obj.save(mode='values')
                                ASCIIColors.success(f"Updated '{instance_name}' -> '{selected_type}' mapping in main config.")
                                self.saved_successfully = False # Mark main config changed
                            except Exception as map_err:
                                ASCIIColors.error(f"Failed update main config bindings_map: {map_err}"); trace_exception(map_err)
                                ASCIIColors.warning("Instance file saved, but main config map was NOT updated.")
                        except Exception as save_err:
                            ASCIIColors.error(f"Failed to save instance config file '{instance_file_path}': {save_err}")
                    else:
                        ASCIIColors.info("Instance configuration not saved.")

            except Exception as add_err:
                    ASCIIColors.error(f"Error adding instance '{instance_name}': {add_err}"); trace_exception(add_err)
        def edit_action():
            # Keep the existing edit logic for now
            if not instance_infos: ASCIIColors.warning("No instances to edit."); return
            edit_options = [(f"{name} ({p.name})", (p, name, info)) for name, p, info in sorted(instance_infos)]
            selection = _select_from_list(edit_options, "Select Instance Configuration to Edit:")
            if not selection: return # User cancelled
            instance_path, instance_name_to_edit, instance_info = selection

            # Determine instance type (prefer map, fallback to file)
            instance_type = bindings_map.get(instance_name_to_edit)
            type_from_file = instance_info.get('type_from_file')
            if not instance_type:
                if type_from_file and type_from_file not in ["N/A", "ErrorLoading", "ErrorLoading(NotDict)"]:
                    instance_type = type_from_file
                    ASCIIColors.warning(f"Instance '{instance_name_to_edit}' not found in bindings_map. Using type '{instance_type}' from file.")
                else:
                        ASCIIColors.error(f"Cannot determine binding type for instance '{instance_name_to_edit}' from map or file. Cannot edit."); return
            elif instance_type != type_from_file and type_from_file not in ["N/A", "ErrorLoading", "ErrorLoading(NotDict)"]:
                    ASCIIColors.warning(f"Type mismatch for '{instance_name_to_edit}': Map='{instance_type}', File='{type_from_file}'. Using type from map: '{instance_type}'.")

            # Get schema for the determined type
            type_info = available_binding_types.get(instance_type)
            if not type_info or not type_info.get("card"):
                ASCIIColors.error(f"Cannot edit: Binding type definition '{instance_type}' not found or invalid."); return
            instance_schema = type_info["card"].get("instance_schema")
            if not instance_schema or not isinstance(instance_schema, dict):
                ASCIIColors.error(f"Invalid instance schema found for type '{instance_type}'. Cannot edit."); return

            # Get handler and check dependencies
            handler_class = self.handler_map.get(instance_path.suffix.lower()) # Use self.handler_map
            if not handler_class: ASCIIColors.error(f"Unsupported file type for editing: {instance_path.suffix}"); return
            if not check_handler_dependency(handler_class):
                    ASCIIColors.error(f"Dependency missing for {instance_path.suffix}. Cannot edit."); return

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
                    modified = self._interactive_edit_section(
                        section_config=config_instance,
                        section_name=f"Instance '{instance_name_to_edit}' Settings",
                        skip_keys=["__version__", "type", "binding_instance_name"],
                        binding_type=instance_type # Pass type for context
                    )

                    if modified:
                        if ASCIIColors.confirm(f"Save changes to {instance_path.name}?", default_yes=True):
                            config_instance.save(mode='values') # Save updated values
                            ASCIIColors.success("Changes saved.")
                            self.saved_successfully = False # Mark main config changed
                        else: ASCIIColors.info("Changes not saved.")
                    else: ASCIIColors.info("No changes were made.")
            except Exception as edit_err:
                    ASCIIColors.error(f"Error editing instance '{instance_name_to_edit}': {edit_err}"); trace_exception(edit_err)
        def remove_action():
            if not instance_infos:
                ASCIIColors.warning("No instances to remove."); return

            # 1. Select Instance to Remove using Menu
            remove_options = [(f"{name} ({p.name})", (p, name)) for name, p, _ in sorted(instance_infos)]
            selection = _select_from_list(remove_options, "Select Instance Configuration to Remove:")
            if not selection:
                ASCIIColors.warning("Cancelled removal."); return # User cancelled

            instance_path_to_remove, instance_name_to_remove = selection

            # 2. Confirm Deletion using confirm
            ASCIIColors.warning(f"This will permanently DELETE the file: {instance_path_to_remove.name}")
            if ASCIIColors.confirm(f"Proceed with deleting '{instance_name_to_remove}' config file?", default_yes=False):
                try:
                        # 3. Delete File
                        instance_path_to_remove.unlink()
                        ASCIIColors.success(f"Instance config file deleted: {instance_path_to_remove.name}")

                        # 4. Remove from Main Config Map
                        try:
                            bindings_map_section = getattr(main_config_obj, "bindings_map")
                            if instance_name_to_remove in bindings_map_section:
                                del bindings_map_section[instance_name_to_remove]
                                main_config_obj.save(mode='values') # Save updated map
                                ASCIIColors.success(f"Removed '{instance_name_to_remove}' mapping from main config.")
                                self.saved_successfully = False # Mark main config changed
                            else:
                                ASCIIColors.info("Instance was not found in main config's bindings_map anyway.")
                        except (KeyError, SettingNotFoundError):
                            ASCIIColors.info("Instance mapping was already absent from main config.")
                        except Exception as map_err:
                            ASCIIColors.error(f"Failed to update main config bindings_map after deletion: {map_err}"); trace_exception(map_err)
                            ASCIIColors.warning("Instance file deleted, but main config map update FAILED.")

                except OSError as rm_err: ASCIIColors.error(f"Failed to delete file: {rm_err}")
                except Exception as rm_err: ASCIIColors.error(f"Error removing instance: {rm_err}"); trace_exception(rm_err)
            else:
                ASCIIColors.info("Removal cancelled.")
        menu.add_action("Add New Instance", add_action)
        if instance_infos:
            menu.add_action("Edit Instance", edit_action)
            menu.add_action("Remove Instance", remove_action)

        menu.run()


    # --- Helper to get instance info ---
    def _get_binding_instances_info(self, main_config_obj: ConfigGuard) -> List[Tuple[str, Path, Dict[str, Any]]]:
        """Lists existing binding instance config files and tries to load basic info."""
        # This helper function remains the same as before, just ensure it's called correctly.
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
                if item.is_file() and item.suffix.lower() in self.handler_map: # Use self.handler_map
                    instance_name = item.stem
                    instance_type_from_map = bindings_map.get(instance_name, "Unknown (Not in map)")
                    basic_info = {"name": instance_name, "type_from_map": instance_type_from_map, "file": item.name}
                    # Try loading basic info (like type) from the file itself
                    try:
                         handler_class = self.handler_map.get(item.suffix.lower())
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
    
    def review_and_save(self) -> bool:
        """Reviews and saves the main configuration file."""
        # Implementation remains the same
        if not self.config or not self.main_config_path:
            ASCIIColors.error("Configuration not initialized properly."); return False

        print_wizard_title("Review Main Configuration")
        try:
            # Get config as dict, excluding secrets for display safety
            config_dict = self.config.get_config_dict()
            # Use JSON for display clarity
            config_str = json.dumps(config_dict, indent=2)
            ASCIIColors.print("\n--- Current Main Configuration (Secrets Hidden) ---", color=ASCIIColors.color_bright_yellow)
            print(config_str) # Direct print to avoid potential color issues with JSON
            ASCIIColors.print("--- End Configuration ---", color=ASCIIColors.color_bright_yellow)

            ASCIIColors.info("\nNote: Binding instance configurations are managed separately.")
            instance_folder_path = Path(getattr(self.config.paths, "instance_bindings_folder", "N/A"))
            ASCIIColors.info(f"Instance configs should be created/managed in: '{instance_folder_path}'")
            ASCIIColors.info("(Use menu option 5 or configure presets here).") # Updated info text

            # Confirm save
            if ASCIIColors.confirm(f"Save this main configuration to {self.main_config_path}?", default_yes=True):
                # Ensure parent directory exists (should already, but double-check)
                self.main_config_path.parent.mkdir(parents=True, exist_ok=True)
                # Save with secrets included
                self.config.save(mode='values')
                ASCIIColors.success(f"Configuration saved successfully to: {self.main_config_path}")
                self.saved_successfully = True
                return True # Indicate save occurred
            else:
                ASCIIColors.warning("Configuration NOT saved.")
                self.saved_successfully = False
                return False # Indicate save cancelled

        except Exception as e:
            ASCIIColors.error(f"Error displaying or saving configuration: {e}")
            trace_exception(e)
            self.saved_successfully = False
            return False # Indicate error
# --- END WIZARD CLASS ---

# --- Helper for list selection ---
def _select_from_list(
    options: List[Tuple[str, Any]],
    prompt: str,
    epilog_text: Optional[str] = None # Added epilog parameter
) -> Optional[Any]:
    """Generic helper to select an item from a list using ASCIIColors Menu."""
    if not options:
        ASCIIColors.warning("No options available to select from.")
        return None

    menu_items = [(display, value) for display, value in options]

    menu = Menu(
        prompt,
        mode='select_single',
        item_color=ASCIIColors.color_cyan,
        title_color=ASCIIColors.color_yellow,
        epilog_text=epilog_text # Pass epilog to Menu constructor
    )
    menu.add_choices(menu_items)
    selected_value = menu.run() # Returns the 'value' directly

    return selected_value


# --- Main Execution Block ---
def main():
    """Determines server root and runs the wizard."""
    # Determine server root relative to this script file
    server_root_dir = Path(__file__).resolve().parent # Assumes wizard is in project root
    if not (server_root_dir / "lollms_server").is_dir(): # Basic check
        # Fallback to Current Working Directory if structure isn't as expected
        server_root_dir = Path(".").resolve()
        logger.warning(f"Could not verify server structure relative to wizard script. Using CWD as server root: {server_root_dir}")
        if not (server_root_dir / "lollms_server").is_dir():
            ASCIIColors.error(f"Could not determine server root. Please run wizard from project root directory containing 'lollms_server'.")
            sys.exit(1)

    logger.info(f"Wizard using Server Root: {server_root_dir}")
    wizard = ConfigurationWizard(server_root=server_root_dir)
    try:
        wizard.run()
        ASCIIColors.print("\n---------------------------------")
        ASCIIColors.print("Wizard finished.")
        ASCIIColors.print("You can now start the server using 'run.sh' or 'run.bat'.")
        ASCIIColors.print("Use installer menu (install_core.py) again to manage optional deps or binding instances if needed.")
        ASCIIColors.print("---------------------------------")
    except KeyboardInterrupt:
        ASCIIColors.warning("\nWizard aborted by user (Ctrl+C). Configuration may not be saved.")
        sys.exit(1)
    except Exception as e:
        ASCIIColors.error(f"\nAn unexpected error occurred during the wizard: {e}")
        trace_exception(e)
        sys.exit(1)

# --- Run the Wizard ---
if __name__ == "__main__":
    main()
    sys.exit(0) # Explicit exit with success code