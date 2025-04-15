# lollms_server/main.py
import sys
import os
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Request # Request needed for dependencies
from contextlib import asynccontextmanager
import logging
import uvicorn

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
        root_logger.setLevel(log_level_int)

        # Optional: Update level for existing handlers if needed (basicConfig adds one)
        # for handler in root_logger.handlers:
        #     handler.setLevel(log_level_int) # Affects which messages the handler outputs

        logger.info(f"Successfully set root logger level to {log_level_name}.") # This message confirms the change

    except Exception as e:
        logger.error(f"Failed to apply logging configuration from file: {e}", exc_info=True)
        logger.warning("Continuing with initial logging level.")

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

# --- Middleware (Optional) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request: {request.method} {request.url}") # Use debug for less noise
    # Example: Add request ID or timing here later
    response = await call_next(request)
    logger.debug(f"Response: {response.status_code}")
    return response

# --- Include API Router ---
# Ensure this comes AFTER the app definition and lifespan
app.include_router(api_router, prefix="/api/v1")

# --- Root Endpoint ---
@app.get("/", summary="Root endpoint", description="Provides basic server information.")
async def read_root(request: Request): # Inject Request
    # Basic check to ensure config loaded during lifespan
    if not hasattr(request.app.state, 'config'):
         raise HTTPException(status_code=503, detail="Server is starting up, configuration not yet loaded.")

    # Access config from app.state via request
    cfg_dump = request.app.state.config.model_dump(exclude={'security': {'allowed_api_keys'}})
    return {
        "message": "Welcome to LOLLMS Server!",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "loaded_config": cfg_dump # Be careful not to expose secrets if any remain
    }

# --- REMOVE Dependency Injection Utility functions that cause circular import ---
# def get_binding_manager(): ...
# def get_personality_manager(): ...
# def get_function_manager(): ...
# def get_resource_manager(): ...


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