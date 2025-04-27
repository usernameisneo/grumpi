# lollms_server/main.py
import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request # Request needed for dependencies
from contextlib import asynccontextmanager
import ascii_colors as logging
import uvicorn
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from fastapi.responses import FileResponse # Import FileResponse
from fastapi.middleware.cors import CORSMiddleware


# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Core imports
from lollms_server.core.config import load_config, get_config as get_config_core, AppConfig
from lollms_server.core.security import verify_api_key
from lollms_server.core.bindings import BindingManager
from lollms_server.core.personalities import PersonalityManager
from lollms_server.core.functions import FunctionManager
from lollms_server.core.resource_manager import ResourceManager
# Import router *after* other core components are defined/imported
from lollms_server.api.endpoints import router as api_router


# Setup logging
# --- Initial Basic Logging Config ---
# Configure root logger initially - messages before config load will use this level.
# Set to DEBUG initially to see all startup messages, will be adjusted by config later.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True # force=True might be needed if basicConfig was called elsewhere implicitly
)
# Get the logger for the main module
logger = logging.getLogger(__name__)
# Optionally silence verbose libraries like httpx if needed at DEBUG level
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration - Calculate path relative to *this* file's location
# This makes it more robust if the script is run from different CWDs
SERVER_ROOT = Path(__file__).resolve().parent.parent
# Default WEBUI path relative to SERVER_ROOT
DEFAULT_WEBUI_BUILD_DIR = SERVER_ROOT / "webui" / "dist"

# No longer need global manager variables here

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # Use app.state to store shared instances
    logger.info("Starting lollms_server...")

    # Load configuration and store it
    app.state.config = load_config("config.toml")
    logger.info(f"Configuration loaded from: {app.state.config.paths}") # Example log
    loaded_config: AppConfig = app.state.config # Alias for easier access

    # --- Reconfigure Logging Level ---
    try:
        log_level_name = loaded_config.logging.log_level
        log_level_int = loaded_config.logging.level # Use the property from LoggingConfig
        logger.info(f"Attempting to set log level to: {log_level_name} ({log_level_int})")

        # Get the root logger and set its effective level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_name)

        # Optional: Update level for existing handlers if needed (basicConfig adds one)
        # for handler in root_logger.handlers:
        #     handler.setLevel(log_level_int) # Affects which messages the handler outputs

        logger.info(f"Successfully set root logger level to {log_level_name}.") # This message confirms the change

    except Exception as e:
        logger.error(f"Failed to apply logging configuration from file: {e}", exc_info=True)
        logger.warning("Continuing with initial logging level.")

    # --- Determine Web UI Status and Path ---
    app.state.webui_enabled = False # Default to disabled
    webui_build_dir = DEFAULT_WEBUI_BUILD_DIR # Start with default path
    serve_ui = False

    # Check if webui section exists and is enabled in config
    if loaded_config.webui and loaded_config.webui.enable_ui:
        serve_ui = True
        # Optional: Check for custom build directory path in config later if needed
        # if hasattr(loaded_config.webui, 'build_directory') and loaded_config.webui.build_directory:
        #    webui_build_dir = SERVER_ROOT / loaded_config.webui.build_directory
        #    logger.info(f"Using custom Web UI build directory from config: {webui_build_dir}")
        # else:
        #    logger.info(f"Using default Web UI build directory: {webui_build_dir}")
        logger.info(f"Web UI serving is enabled via config.")
    else:
        logger.info("Web UI serving is disabled via config or config section missing.")


    # Check if directory and index.html exist ONLY if serving is enabled
    if serve_ui:
        index_html_path = webui_build_dir / "index.html"
        if not webui_build_dir.is_dir() or not index_html_path.is_file():
             logger.warning(f"Web UI build directory or index.html missing at: {webui_build_dir}")
             logger.warning("Web UI serving disabled because build files were not found.")
             serve_ui = False # Disable serving if files are missing
        else:
             logger.info(f"Web UI build directory found at: {webui_build_dir}")
             app.state.webui_enabled = True # Mark as enabled in state
             app.state.webui_build_dir = webui_build_dir # Store path in state
             app.state.webui_index_path = index_html_path # Store index path


    # --- Continue with Initialization ---
    logger.info(f"Configuration loaded.") # Visibility depends on configured level
    logger.debug(f"Config paths: {loaded_config.paths}") # Example debug log
    # Initialize managers and store in app.state
    logger.info("Initializing Resource Manager...")
    app.state.resource_manager = ResourceManager(config=app.state.config.resource_manager)

    logger.info("Initializing Binding Manager...")
    # Pass resource_manager correctly during instantiation
    app.state.binding_manager = BindingManager(config=app.state.config, resource_manager=app.state.resource_manager)
    await app.state.binding_manager.load_bindings()

    logger.info("Initializing Personality Manager...")
    app.state.personality_manager = PersonalityManager(config=app.state.config)
    app.state.personality_manager.load_personalities() # Synchronous

    logger.info("Initializing Function Manager...")
    app.state.function_manager = FunctionManager(config=app.state.config)
    app.state.function_manager.load_functions() # Synchronous

    logger.info("Server startup complete.")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Shutting down lollms_server...")
    # Cleanup logic using app.state
    if hasattr(app.state, 'binding_manager') and app.state.binding_manager:
        logger.info("Cleaning up binding manager...")
        await app.state.binding_manager.cleanup()
    # Add cleanup for other managers if needed
    logger.info("Server shutdown complete.")


app = FastAPI(
    title="LOLLMS Server",
    description="A multi-modal generation server compatible with LOLLMS personalities.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Add CORS Middleware ---
# Read origins from the loaded config
# Need to access config AFTER lifespan has run, so access via request or app.state later?
# Let's configure it using the initially loaded config object. This assumes config
# doesn't change dynamically while server is running.

# Load config once for setup purposes (outside lifespan)
# This might duplicate loading slightly but ensures config is available for middleware setup
setup_config = get_config_core()
cors_origins = setup_config.server.allowed_origins
logger.info(f"Configuring CORS for origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins, # Use list from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Middleware ---


# --- API Router ---
app.include_router(api_router, prefix="/api/v1")


# --- Static Files & Web UI Serving ---
# We need to access app.state here, which is only available *after* lifespan starts.
# This means we need to mount static files *conditionally* within endpoint logic
# or potentially restructure how StaticFiles is mounted (e.g., inside lifespan?).
# Mounting inside lifespan is tricky. Let's handle it conditionally in the catch-all route.

# We CAN mount the /assets path if we know the relative path, but need full path later.
# Let's NOT mount /assets here, rely on the catch-all to serve index.html, which
# then loads assets using relative paths from index.html's location.

# --- Catch-all route for SPA ---
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_vue_app(request: Request, full_path: str):
    """ Serves the index.html for SPA routing. """
    # Check if Web UI is enabled AND files found during startup
    if not getattr(request.app.state, 'webui_enabled', False):
         raise HTTPException(status_code=404, detail="Web UI is not enabled or not found.")

    index_path = getattr(request.app.state, 'webui_index_path', None)
    build_dir = getattr(request.app.state, 'webui_build_dir', None)

    if not index_path or not build_dir:
        logger.error("Web UI paths not found in app state despite UI being enabled.")
        raise HTTPException(status_code=500, detail="Web UI configuration error.")

    # Basic security check
    if full_path.endswith((".py", ".pyc", ".toml", ".yaml", ".log")):
        raise HTTPException(status_code=404, detail="Not found")

    # Construct the potential path to a static file within the build directory
    potential_file_path = build_dir / full_path

    # If the requested path corresponds to an existing file in the build dir (e.g., CSS, JS), serve it
    if potential_file_path.is_file():
         logger.debug(f"Serving static file: {potential_file_path}")
         # FastAPI automatically handles appropriate headers for common file types
         return FileResponse(potential_file_path)
    else:
         # Otherwise, serve the index.html for SPA routing
         logger.debug(f"Serving index.html for SPA route: {full_path}")
         return FileResponse(index_path)
    
    
# --- Main Execution ---
if __name__ == "__main__":
    # Load config temporarily just for host/port
    # Using the core config loader directly
    temp_config = get_config_core("config.toml") # Load it explicitly here if needed
    if not temp_config:
         # Fallback if config file doesn't exist or fails loading early
         print("Warning: config.toml not found or failed to load. Using default host/port.")
         host = "0.0.0.0"
         port = 9600
    else:
         host = temp_config.server.host
         port = temp_config.server.port


    print(f"Attempting to start server on {host}:{port}")
    uvicorn.run(
        "lollms_server.main:app", # Uvicorn needs the string path
        host=host,
        port=port,
        reload=False, # Use --reload flag in terminal for development
        workers=1, # Uvicorn default or set via --workers flag
        log_config=None # Prevent uvicorn from overriding our basicConfig
    )