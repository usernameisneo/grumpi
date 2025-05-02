# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: FastAPI application entry point for lollms_server, using ConfigGuard and ascii_colors.

import sys
import os
import uvicorn
import importlib.metadata # To get version at runtime
import importlib.util # To check if package is installed
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Any # For type hinting fallback

# --- Ensure project root is in Python path ---
# Determine SERVER_ROOT early and robustly
try:
    SERVER_ROOT = Path(__file__).resolve().parent.parent
    if str(SERVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVER_ROOT))
        print(f"INFO: Added project root to sys.path: {SERVER_ROOT}")
except NameError:
    # Fallback if __file__ is not defined (e.g., in some execution environments)
    SERVER_ROOT = Path(".").resolve()
    print(f"WARNING: __file__ not defined. Assuming project root is CWD: {SERVER_ROOT}")
    if str(SERVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVER_ROOT))

# --- Imports ---
# Logging (Assume ascii_colors is installed via requirements)
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors, trace_exception
except ImportError:
    # Basic logging fallback if ascii_colors somehow fails after install
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logging.exception(e)
    print("WARNING: ascii_colors library not found or failed import. Using basic logging.")

# FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ConfigGuard
try:
    from configguard import ConfigGuard
except ImportError:
    logging.critical("FATAL ERROR: ConfigGuard library not found. Please run install script.")
    sys.exit(1)

# Core Components (Using ConfigGuard for initialization)
from lollms_server.core.config import initialize_config, get_config, get_config_path, get_server_root
from lollms_server.core.bindings import BindingManager
from lollms_server.core.personalities import PersonalityManager
from lollms_server.core.functions import FunctionManager
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.api.endpoints import router as api_router # Import API router

# --- Initial Logging Setup (before config load) ---
# Set a basic default level, will be overridden after config load
logging.basicConfig(level=logging.INFO, format='{asctime} | {levelname:<8} | {name} | {message}', style='{', force=True)
logger = logging.getLogger("lollms_server") # Main logger for the application

# Silence overly verbose libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# --- Load Configuration BEFORE Creating App ---
# This is crucial as middleware and lifespan need the config
main_config: Optional[ConfigGuard] = None
try:
    logger.info("Initializing configuration...")
    # initialize_config now determines root, finds config file, or triggers wizard
    main_config = initialize_config(SERVER_ROOT, trigger_wizard_if_missing=True)
    config_file_path = get_config_path() # Get the path that was actually loaded
    if not main_config or not config_file_path:
        raise RuntimeError("Configuration object or path is None after initialization.")
    logger.info(f"Main configuration loaded successfully from: {config_file_path}")
except SystemExit as e:
    # Initialization might exit if wizard fails or config is critically broken
    logger.critical(f"Configuration initialization failed with exit code {e.code}. Server cannot start.")
    sys.exit(e.code or 1)
except Exception as e:
    logger.critical(f"Unexpected error during configuration initialization: {e}", exc_info=True)
    trace_exception(e)
    sys.exit(1)


# --- Reconfigure Logging Level based on loaded config (immediately after load) ---
try:
    log_level_name = main_config.logging.log_level.upper()
    log_level_int = getattr(logging, log_level_name, logging.INFO)
    # Use ASCIIColors method to set level globally if available
    if hasattr(logging, 'ASCIIColors') and hasattr(logging.ASCIIColors, 'set_log_level'):
        logging.ASCIIColors.set_log_level(log_level_int)
        # Re-apply basicConfig to ensure handlers use the new level
        logging.basicConfig(level=log_level_int, force=True, format='{asctime} | {levelname:<8} | {name} | {message}', style='{')
    else: # Fallback to standard logging setLevel
        logging.getLogger().setLevel(log_level_int)
        # Update handlers if possible (simple example)
        for handler in logging.getLogger().handlers:
             handler.setLevel(log_level_int)

    logger.info(f"Logging level set to {log_level_name} ({log_level_int}) from configuration.")
except AttributeError as e:
     logger.error(f"Failed to find logging settings in config: {e}. Using default level.")
except Exception as e:
    logger.error(f"Failed to apply logging configuration from file: {e}", exc_info=True)


# --- FastAPI Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup (initializes managers) and shutdown (cleans up resources).
    Relies on the 'main_config' variable loaded above.
    """
    global main_config # Access the globally loaded config
    logger.info("--- Starting LOLLMS Server Lifespan ---")
    if main_config is None:
         logger.critical("Lifespan started but main_config is None. Initialization failed earlier.")
         # Optionally raise an error or allow degraded startup
         raise RuntimeError("Server cannot start without a valid configuration.")

    # 1. Store loaded config in app state for dependencies
    app.state.config = main_config
    logger.debug("Main config stored in app state.")

    # 2. Determine Web UI Status and Path (uses main_config)
    app.state.webui_enabled = False
    webui_build_dir_relative = Path("webui") / "dist" # Default relative path
    webui_build_dir = (SERVER_ROOT / webui_build_dir_relative).resolve() # Default absolute path
    serve_ui = False
    try:
        webui_config_section = getattr(main_config, "webui", None)
        if webui_config_section and getattr(webui_config_section, "enable_ui", False):
            serve_ui = True
            logger.info("Web UI serving is enabled via config.")
            # Add logic here if config allows specifying a custom build dir path
            # custom_dir_str = getattr(webui_config_section, "build_directory", None)
            # if custom_dir_str: webui_build_dir = Path(custom_dir_str); if not webui_build_dir.is_absolute(): webui_build_dir = (SERVER_ROOT / webui_build_dir).resolve()

        else:
            logger.info("Web UI serving is disabled via config.")

        if serve_ui:
            # Use the determined (default or custom) absolute path
            index_html_path = webui_build_dir / "index.html"
            if not webui_build_dir.is_dir() or not index_html_path.is_file():
                 logger.warning(f"Web UI enabled, but build directory or index.html not found at {webui_build_dir}. Disabling UI serving.")
                 serve_ui = False
            else:
                 logger.info(f"Web UI build directory found: {webui_build_dir}")
                 app.state.webui_enabled = True
                 app.state.webui_build_dir = webui_build_dir
                 app.state.webui_index_path = index_html_path
                 logger.info("Web UI serving is active.")
    except AttributeError as e:
        logger.error(f"Error accessing Web UI configuration settings: {e}. Web UI disabled.")
        serve_ui = False
    except Exception as e:
         logger.error(f"Error setting up Web UI configuration: {e}", exc_info=True)
         serve_ui = False

    # 3. Initialize Managers (pass loaded main_config)
    logger.info("Initializing core managers...")
    initialization_successful = True
    try:
        # ResourceManager needs its specific config section or a dictionary
        resource_config_section = getattr(main_config, "resource_manager", None)
        if not resource_config_section: raise ValueError("Resource manager configuration section missing.")
        app.state.resource_manager = ResourceManager(config=resource_config_section)
        logger.info("ResourceManager initialized.")

        app.state.binding_manager = BindingManager(main_config=main_config, resource_manager=app.state.resource_manager)
        await app.state.binding_manager.load_bindings() # Load instances based on config
        logger.info("BindingManager initialized and bindings loaded.")

        app.state.personality_manager = PersonalityManager(main_config=main_config)
        app.state.personality_manager.load_personalities() # Load personalities based on config paths/overrides
        logger.info("PersonalityManager initialized and personalities loaded.")

        app.state.function_manager = FunctionManager(main_config=main_config)
        app.state.function_manager.load_functions() # Load functions based on config paths
        logger.info("FunctionManager initialized and functions loaded.")

        logger.info("All managers initialized successfully.")
    except Exception as e:
         logger.critical(f"Failed to initialize one or more managers: {e}", exc_info=True)
         trace_exception(e)
         initialization_successful = False
         # Decide how to handle partial failure - exit or degraded mode?
         # For now, log critical error but let FastAPI start (API might fail)
         logger.critical("SERVER STARTING IN DEGRADED STATE due to manager initialization failure.")


    # --- Yield control to the running application ---
    yield
    # --- End of application runtime ---


    # --- Shutdown Sequence ---
    logger.info("--- Starting LOLLMS Server Shutdown ---")
    # Cleanup Binding Manager (which should handle stopping servers etc.)
    if hasattr(app.state, 'binding_manager') and app.state.binding_manager:
        try:
            logger.info("Cleaning up binding manager...")
            await app.state.binding_manager.cleanup()
            logger.info("Binding manager cleanup finished.")
        except Exception as e:
             logger.error(f"Error during binding manager cleanup: {e}", exc_info=True)
             trace_exception(e)
    else:
         logger.info("Binding manager not found in state or already cleaned up.")

    # Add cleanup for other managers if needed (e.g., closing resources)

    logger.info("--- Server shutdown complete. ---")


# --- Create FastAPI App Instance ---
# Determine package version for API docs
try:
     app_version = importlib.metadata.version("lollms_server")
except importlib.metadata.PackageNotFoundError:
     app_version = "0.3.0" # Fallback version
     logger.warning("Could not get version from package metadata. Using fallback.")


app = FastAPI(
    title="LOLLMS Server",
    description="A multi-modal generation server compatible with LOLLMS personalities, using ConfigGuard.",
    version=app_version,
    lifespan=lifespan, # Use the lifespan manager defined above
)

# --- Configure CORS Middleware (using pre-loaded main_config) ---
try:
    # Access nested server settings safely
    server_section = getattr(main_config, 'server', None)
    cors_origins = getattr(server_section, "allowed_origins", []) if server_section else []

    if isinstance(cors_origins, list) and cors_origins:
         logger.info(f"Configuring CORS for origins: {cors_origins}")
         app.add_middleware(
             CORSMiddleware,
             allow_origins=cors_origins,
             allow_credentials=True, # Allow cookies if needed by UI
             allow_methods=["*"], # Allow all standard methods
             allow_headers=["*"], # Allow all headers, including X-API-Key
         )
    else:
         logger.warning("CORS origins not configured or invalid format in configuration. Web UI might not connect properly from different origins.")
except AttributeError as e:
     logger.error(f"Error accessing server/CORS settings in config: {e}. CORS may not be configured correctly.")
except Exception as e:
     logger.error(f"Failed to configure CORS middleware: {e}", exc_info=True)


# --- API Router ---
app.include_router(api_router, prefix="/api/v1")
logger.info("API router /api/v1 included.")


# --- Static Files & Web UI Catch-all Route ---
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_vue_app_or_docs(request: Request, full_path: str):
    """
    Serves the index.html for Single Page Application routing (if UI enabled),
    static files from the Web UI build directory, or a simple info page.
    """
    # Access paths from app.state (set during lifespan)
    webui_enabled: bool = getattr(request.app.state, 'webui_enabled', False)
    webui_build_dir: Optional[Path] = getattr(request.app.state, 'webui_build_dir', None)
    webui_index_path: Optional[Path] = getattr(request.app.state, 'webui_index_path', None)

    # Serve index.html or info page if root is requested
    if full_path == "" or full_path == "/":
        if webui_enabled and webui_index_path and webui_index_path.is_file():
            logger.debug("Serving index.html for SPA root.")
            return FileResponse(webui_index_path)
        else:
            # Provide basic info and links to docs if UI is disabled or not found
            api_docs_url = request.url_for("swagger_ui_html")
            redoc_url = request.url_for("redoc_html")
            status_msg = "Web UI is disabled or build files not found." if not webui_enabled else "Web UI build files not found."
            return PlainTextResponse(f"LOLLMS Server is running.\nAPI Docs: {api_docs_url}\nReDoc: {redoc_url}\n{status_msg}", status_code=200)

    # If UI is enabled, try serving static files or fall back to index.html for SPA routing
    if webui_enabled and webui_build_dir and webui_index_path:
        # Sanitize path to prevent directory traversal
        # Normalize, remove leading slashes, check for '..'
        clean_path_str = os.path.normpath(full_path).lstrip(os.path.sep + os.path.altsep if os.path.altsep else os.path.sep)
        if '..' in clean_path_str.split(os.path.sep):
            logger.warning(f"Directory traversal attempt blocked: {full_path}")
            return PlainTextResponse("Forbidden", status_code=403)

        potential_file_path = webui_build_dir / clean_path_str
        try:
            # Resolve symbolic links and check if path is still within build dir
            resolved_file_path = potential_file_path.resolve()
            resolved_build_dir = webui_build_dir.resolve()
            if not str(resolved_file_path).startswith(str(resolved_build_dir)):
                 logger.warning(f"Attempted access outside web root: {full_path} -> {resolved_file_path}")
                 # Fallback to index.html for potential SPA routes that look like file paths
                 logger.debug(f"Serving index.html for SPA route (outside web root): /{full_path}")
                 return FileResponse(webui_index_path)
        except Exception as path_e:
             # If path resolution fails (e.g., invalid chars), treat as SPA route
             logger.warning(f"Path resolution error for '{full_path}': {path_e}")
             logger.debug(f"Serving index.html for SPA route (path resolution error): /{full_path}")
             return FileResponse(webui_index_path)

        # Serve the file if it exists
        if resolved_file_path.is_file():
             logger.debug(f"Serving static file: {resolved_file_path}")
             return FileResponse(resolved_file_path)
        else:
             # File not found, assume it's an SPA route, serve index.html
             logger.debug(f"Serving index.html for SPA route (file not found): /{full_path}")
             return FileResponse(webui_index_path)
    else:
        # UI is disabled, return 404 for any non-root path
        return PlainTextResponse("Not Found", status_code=404)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Uvicorn configuration using loaded main_config as fallback for env vars
    uvicorn_host = "0.0.0.0"
    uvicorn_port = 9600
    uvicorn_workers = 1
    try:
        if main_config and hasattr(main_config, 'server'):
             uvicorn_host = os.environ.get("LOLLMS_HOST", main_config.server.host)
             try: uvicorn_port = int(os.environ.get("LOLLMS_PORT", main_config.server.port))
             except (ValueError, TypeError):
                 logger.warning(f"Invalid LOLLMS_PORT env var or config value. Using default {uvicorn_port}.")
                 uvicorn_port = 9600 # Hardcoded default if config access fails badly

             try: uvicorn_workers = int(os.environ.get("LOLLMS_WORKERS", 1)); uvicorn_workers = max(1, uvicorn_workers)
             except ValueError: logger.warning("Invalid LOLLMS_WORKERS env var. Using default 1."); uvicorn_workers = 1
        else:
            logger.warning("Main config or server section not fully loaded. Using hardcoded Uvicorn defaults (0.0.0.0:9600, 1 worker).")
            uvicorn_host = os.environ.get("LOLLMS_HOST", "0.0.0.0")
            try: uvicorn_port = int(os.environ.get("LOLLMS_PORT", 9600))
            except ValueError: uvicorn_port = 9600
            try: uvicorn_workers = int(os.environ.get("LOLLMS_WORKERS", 1)); uvicorn_workers = max(1, uvicorn_workers)
            except ValueError: uvicorn_workers = 1

    except AttributeError as e:
        logger.error(f"Error accessing server configuration for Uvicorn: {e}. Using hardcoded defaults.")
        uvicorn_host = os.environ.get("LOLLMS_HOST", "0.0.0.0")
        try: uvicorn_port = int(os.environ.get("LOLLMS_PORT", 9600))
        except ValueError: uvicorn_port = 9600
        try: uvicorn_workers = int(os.environ.get("LOLLMS_WORKERS", 1)); uvicorn_workers = max(1, uvicorn_workers)
        except ValueError: uvicorn_workers = 1


    reload_flag = os.environ.get("LOLLMS_RELOAD", "false").lower() in ["true", "1", "yes"]
    if reload_flag:
        logger.warning("Development reload mode is ON. DO NOT use in production.")

    logger.info(f"--- Starting Uvicorn ---")
    logger.info(f" Address: http://{uvicorn_host}:{uvicorn_port}")
    logger.info(f" Workers: {uvicorn_workers}")
    logger.info(f" Reload:  {'Enabled' if reload_flag else 'Disabled'}")
    logger.info(f" Log Level: {logging.getLevelName(0)}")
    logger.info(f"------------------------")

    # Standard Uvicorn setup
    # Pass the app object string correctly
    uvicorn.run(
        "lollms_server.main:app", # String reference to the app object
        host=uvicorn_host,
        port=uvicorn_port,
        workers=uvicorn_workers,
        log_config=None, # Disable default Uvicorn logging config, use ours
        reload=reload_flag,
        # reload_dirs=[str(SERVER_ROOT / "lollms_server")] if reload_flag else None, # Watch only source dir
    )

    logger.info("Uvicorn server process finished.")