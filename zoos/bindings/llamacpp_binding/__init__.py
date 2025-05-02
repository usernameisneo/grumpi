# encoding:utf-8
# Project: lollms_server
# File: zoos/bindings/llamacpp_binding/__init__.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Binding implementation for llama.cpp GGUF models via its HTTP server.

import asyncio
import subprocess
import json
import time
import socket
import threading
import re
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
import httpx # For async streaming client
import requests # For sync health check during startup wait
import sys
# Use pipmaster if needed
try:
    import pipmaster as pm
    pm.ensure_packages(["requests", "httpx"])
    if not pm.is_installed("llama_cpp_binaries"):
        if sys.platform.startswith("win"):
            print("Running on Windows")
            pm.install("https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.8.0/llama_cpp_binaries-0.8.0+cu124-py3-none-win_amd64.whl")

        elif sys.platform.startswith("linux"):
            print("Running on Linux")
        elif sys.platform == "darwin":
            print("Running on macOS")
        elif sys.platform.startswith("cygwin"):
            print("Running on Cygwin (Unix-like environment on Windows)")
        elif sys.platform.startswith("aix"):
            print("Running on AIX (IBM Unix)")
        elif sys.platform.startswith("freebsd"):
            print("Running on FreeBSD")
        elif sys.platform.startswith("openbsd"):
            print("Running on OpenBSD")
        elif sys.platform.startswith("sunos") or sys.platform.startswith("solaris"):
            print("Running on Solaris")
        else:
            print(f"Unknown platform: {sys.platform}")
except ImportError:
    pass # Assume installed or handle import error below

# Prepare logger early (before importing llama_cpp_python)
import ascii_colors as logging
from ascii_colors import ASCIIColors, trace_exception

# Attempt to import llama_cpp_python and find server binary path
DEFAULT_SERVER_PATH: Optional[str] = None
try:
    import llama_cpp_python.server as llama_server
    DEFAULT_SERVER_PATH = llama_server.get_binary_path("server")
    llamacpp_installed = True
    logging.info(f"Found default llama.cpp server binary via llama-cpp-python: {DEFAULT_SERVER_PATH}")
except ImportError:
    llama_server = None
    llamacpp_installed = False
    logging.warning("llama-cpp-python not found. Server path must be specified manually in config.")
except AttributeError: # Might happen if get_binary_path changes or is missing
    llamacpp_installed = True # Assume installed, just couldn't find path automatically
    logging.warning("Couldn't automatically find llama.cpp server binary path via llama-cpp-python.")
except Exception as e:
    llamacpp_installed = True # Assume installed, but path finding failed
    logging.warning(f"Error trying to find llama.cpp server binary path: {e}")


from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.core.config import get_server_root
from lollms_server.utils.helpers import parse_thought_tags # --- ADDED HELPER IMPORT ---

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from lollms_server.api.models import StreamChunk, InputData, ModelInfo
    except ImportError:
        class StreamChunk: pass
        class InputData: pass
        class ModelInfo: pass # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_LLAMACPP_TIMEOUT = 3600 * 8 # Default timeout for API calls (8 hours)

class LlamaCppBinding(Binding):
    """Binding for llama.cpp server inference via its HTTP API."""
    binding_type_name = "llamacpp_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """Initializes the LlamaCppBinding."""
        super().__init__(config, resource_manager)

        if not llamacpp_installed:
            raise ImportError("LlamaCpp binding requires 'llama-cpp-python'.")
        if not requests or not httpx:
            raise ImportError("LlamaCpp binding requires 'requests' and 'httpx'.")

        models_folder_str = self.config.get("models_folder")
        if not models_folder_str:
            raise ValueError(f"Missing 'models_folder' in llama.cpp config for '{self.binding_instance_name}'.")

        self.models_folder = Path(models_folder_str)
        if not self.models_folder.is_absolute():
            self.models_folder = (get_server_root() / self.models_folder).resolve()
            logger.info(f"LlamaCpp '{self.binding_instance_name}': Resolved models_folder to: {self.models_folder}")

        if not self.models_folder.is_dir():
            logger.warning(f"LlamaCpp models_folder does not exist: {self.models_folder}")

        self.server_path = self.config.get("server_path") or DEFAULT_SERVER_PATH
        self.n_gpu_layers = int(self.config.get("n_gpu_layers", -1))
        self.n_ctx = int(self.config.get("n_ctx", 4096))
        self.batch_size = int(self.config.get("batch_size", 512))
        self.threads = int(self.config.get("threads", 0)) # 0 means auto
        self.threads_batch = int(self.config.get("threads_batch", 0)) # 0 means auto
        self.tensor_split = self.config.get("tensor_split") # e.g., "20,rest" or None
        self.cache_type = self.config.get("cache_type", "fp16") # "fp16", "q8_0", "q4_0"
        self.compress_pos_emb = float(self.config.get("compress_pos_emb", 1.0)) # Compression factor
        self.rope_freq_base = float(self.config.get("rope_freq_base", 0.0)) # 0 means auto/default

        raw_additional_args = self.config.get("additional_args", {})
        if isinstance(raw_additional_args, dict):
            self.additional_args = raw_additional_args
        else:
            logger.warning(f"Invalid 'additional_args' in instance '{self.binding_instance_name}'. Ignoring.")
            self.additional_args = {}

        self.server_process: Optional[asyncio.subprocess.Process] = None
        self.server_port: Optional[int] = None
        self.server_host = "127.0.0.1" # Bind to localhost only
        self.session = requests.Session() # For sync health checks during startup
        self.current_model_full_path: Optional[Path] = None

        logger.info(
            f"Initialized LlamaCppBinding instance '{self.binding_instance_name}': "
            f"Models='{self.models_folder}', Ctx={self.n_ctx}, GPU Layers={self.n_gpu_layers}"
        )

    # --- Capabilities and Listing ---

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on loaded model name."""
        # Basic check for vision models (LLaVA variants)
        if self._model_loaded and self.model_name and \
           ("llava" in self.model_name.lower() or "bakllava" in self.model_name.lower()):
            return ["text", "image"]
        return ["text"]

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ["text"]

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Scans the models_folder for GGUF files."""
        logger.info(f"LlamaCpp '{self.binding_instance_name}': Scanning for models in {self.models_folder}")
        if not self.models_folder.is_dir():
            logger.error(f"Models folder not found for instance '{self.binding_instance_name}': {self.models_folder}")
            return []

        available_models = []
        try:
            for gguf_file in self.models_folder.glob("*.gguf"):
                if gguf_file.is_file():
                    try:
                        stat_info = gguf_file.stat()
                        model_name = gguf_file.name
                        is_vision = "llava" in model_name.lower() or "bakllava" in model_name.lower()
                        context_size = None # Try to infer from filename

                        # Try inferring context size (e.g., model-8k.gguf, model-32768.gguf)
                        try:
                            match_k = re.search(r'[_-](\d+)k[_-]', model_name, re.IGNORECASE)
                            if match_k:
                                context_size = int(match_k.group(1)) * 1024
                            else:
                                # Look for raw numbers >= 1000
                                match_num = re.search(r'[_-]([1-9]\d{3,})[_-]', model_name)
                                if match_num:
                                    context_size = int(match_num.group(1))
                        except Exception:
                            pass # Ignore errors during context size inference

                        model_data = {
                            "name": model_name,
                            "size": stat_info.st_size,
                            "modified_at": datetime.fromtimestamp(stat_info.st_mtime),
                            "supports_vision": is_vision,
                            "supports_audio": False,
                            "format": "gguf",
                            "family": None, # Difficult to determine reliably from filename
                            "context_size": context_size, # May be None if not inferred
                            "details": {"full_path": str(gguf_file.resolve())}
                        }
                        available_models.append(model_data)
                    except OSError as stat_err:
                        logger.warning(f"Could not get file info for {gguf_file.name}: {stat_err}")
                    except Exception as e:
                        logger.warning(f"Could not process file {gguf_file.name}: {e}")
        except Exception as scan_err:
            logger.error(f"Error scanning models folder {self.models_folder}: {scan_err}", exc_info=True)

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Found {len(available_models)} GGUF models.")
        return available_models

    async def health_check(self) -> Tuple[bool, str]:
        """Checks if the server process is running and responding to /health."""
        if not self.server_process or self.server_process.returncode is not None:
            return False, "Server process not running."
        if not self.server_port:
            return False, "Server port not set."

        health_url = f"http://{self.server_host}:{self.server_port}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx
                data = response.json()
                if data.get("status") == "ok":
                    slots = data.get('slots_idle', '?')
                    return True, f"Server is healthy (Idle Slots: {slots})."
                else:
                    return False, f"Server status not 'ok': {data.get('status', 'Unknown')}"
        except httpx.RequestError as e:
            logger.warning(f"LlamaCpp Health check connection failed for instance '{self.binding_instance_name}': {e}")
            return False, f"Health check connection failed: {e}"
        except httpx.HTTPStatusError as e:
            logger.warning(f"LlamaCpp Health check failed for instance '{self.binding_instance_name}': Status {e.response.status_code}")
            return False, f"Server responded status {e.response.status_code}"
        except Exception as e:
            logger.warning(f"LlamaCpp Health check failed for instance '{self.binding_instance_name}' with unexpected error: {e}", exc_info=True)
            return False, f"Health check unexpected error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Indicates GPU requirement based on n_gpu_layers setting."""
        # Return True if n_gpu_layers is not 0 (-1 means all, >0 means specific count)
        return {"gpu_required": self.n_gpu_layers != 0, "estimated_vram_mb": 0} # VRAM estimation is complex

    # --- Model Loading / Unloading ---

    async def load_model(self, model_name: str) -> bool:
        """Starts the llama.cpp server process with the specified model."""
        target_model_path = self.models_folder / model_name
        if not target_model_path.is_file():
            logger.error(f"LlamaCpp '{self.binding_instance_name}': Model file not found: {target_model_path}")
            return False

        async with self._load_lock:
            # Check if already loaded and running
            if self._model_loaded and self.current_model_full_path == target_model_path:
                if self.server_process and self.server_process.returncode is None:
                    logger.info(f"LlamaCpp '{self.binding_instance_name}': Model '{model_name}' server already running.")
                    return True
                else:
                    # Loaded in state, but process is down. Needs restart.
                    logger.warning(f"LlamaCpp '{self.binding_instance_name}': Model '{model_name}' loaded but server process down. Restarting.")
                    self._model_loaded = False
                    self.server_process = None
                    self.server_port = None
                    # Fall through to start sequence

            # If a different model is loaded, unload it first
            if self._model_loaded and self.current_model_full_path != target_model_path:
                logger.info(f"LlamaCpp '{self.binding_instance_name}': Switching model. Stopping server for '{self.model_name}'...")
                await self.unload_model() # This will reset state

            logger.info(f"LlamaCpp '{self.binding_instance_name}': Starting server for model: {model_name}")
            self.server_port = self._find_available_port()
            if not self.server_port:
                logger.critical("CRITICAL: Failed to find an available port for llama.cpp server!")
                return False
            logger.info(f" -> Using port {self.server_port}")

            server_executable = self.server_path
            if not server_executable or not Path(server_executable).exists():
                logger.error(f"llama.cpp server binary not found at '{server_executable}'. Cannot start server for '{self.binding_instance_name}'.")
                return False

            # Build command line arguments
            cmd = [
                server_executable,
                "--model", str(target_model_path.resolve()),
                "--ctx-size", str(self.n_ctx),
                "--n-gpu-layers", str(self.n_gpu_layers),
                "--batch-size", str(self.batch_size),
                "--port", str(self.server_port),
                "--host", self.server_host,
                "--log-disable", # Disable server's internal logging to avoid duplicate output
            ]
            if self.threads > 0:
                cmd += ["--threads", str(self.threads)]
            if self.threads_batch > 0:
                cmd += ["--threads-batch", str(self.threads_batch)]
            if self.tensor_split:
                cmd += ["--tensor-split", self.tensor_split]

            # Cache type arguments
            if self.cache_type == "q8_0":
                cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
            elif self.cache_type == "q4_0":
                cmd += ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]
            # fp16 is default, no args needed

            # RoPE scaling / Compression (Positional Embedding)
            if self.compress_pos_emb != 1.0:
                # Server uses rope-freq-scale, which is 1 / compress_pos_emb
                scale_factor = 1.0 / self.compress_pos_emb if self.compress_pos_emb != 0 else 0
                if scale_factor != 1.0: # Only add if not default
                    cmd += ["--rope-freq-scale", str(scale_factor)]
            if self.rope_freq_base > 0.0: # Only add if explicitly set
                cmd += ["--rope-freq-base", str(self.rope_freq_base)]

            # Add additional user-defined arguments
            if isinstance(self.additional_args, dict):
                for flag, value in self.additional_args.items():
                    if isinstance(flag, str) and flag.startswith('-'):
                        if value is True: # Boolean flag present
                            cmd.append(flag)
                        elif value is not False and value is not None: # Flag with value
                             cmd.extend([flag, str(value)])
                    else:
                        logger.warning(f"Ignoring invalid additional_arg flag: '{flag}'")

            needs_gpu = self.n_gpu_layers != 0
            resource_context = self.resource_manager.acquire_gpu_resource(f"load_{self.binding_instance_name}_{model_name}") \
                               if needs_gpu else nullcontext()

            try:
                async with resource_context:
                    if needs_gpu:
                        logger.info(f"LlamaCpp '{self.binding_instance_name}': GPU resource acquired.")

                    logger.info(f"Starting server process: {' '.join(cmd)}")
                    self.server_process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL, # Ignore stdout
                        stderr=asyncio.subprocess.PIPE     # Capture stderr
                    )

                    # Start a separate thread to monitor stderr non-blockingly
                    stderr_thread = threading.Thread(
                        target=self._monitor_stderr,
                        args=(self.server_process.stderr,),
                        daemon=True
                    )
                    stderr_thread.start()

                    # Wait for the server to become healthy
                    if not await self._wait_for_server_health():
                        rc = self.server_process.returncode if self.server_process else 'N/A'
                        raise RuntimeError(f"Server failed health check (RC: {rc}).")

                    # Success
                    self.model_name = model_name
                    self.current_model_full_path = target_model_path.resolve()
                    self._model_loaded = True
                    logger.info(f"LlamaCpp '{self.binding_instance_name}': Server started successfully for {model_name}.")
                    return True

            except asyncio.TimeoutError:
                logger.error(f"LlamaCpp '{self.binding_instance_name}': Timeout waiting for GPU to load model {model_name}")
                await self._terminate_server() # Clean up attempt
                return False
            except Exception as e:
                logger.error(f"LlamaCpp '{self.binding_instance_name}': Failed server start for {model_name}: {e}", exc_info=True)
                await self._terminate_server() # Clean up attempt
                return False
            # `finally` block not needed here as resource_context handles release

    def _monitor_stderr(self, stream: asyncio.StreamReader):
        """Reads and logs stderr lines from the server process in a separate thread."""
        if not stream:
            logger.error(f"Stderr stream is None for instance '{self.binding_instance_name}'. Cannot monitor.")
            return

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def read_stderr():
            lines_processed = 0
            while True:
                try:
                    # Read until newline
                    line = await stream.readuntil(b'\n')
                    if not line: # End of stream
                        break
                    lines_processed += 1
                    line_str = line.decode('utf-8', errors='replace').strip()

                    # Filter or log the line
                    if line_str:
                        # Reduce noise from frequent server messages
                        if 'slot' in line_str or '/health' in line_str or 'empty' in line_str or 'processing' in line_str:
                            logger.debug(f"[{self.binding_instance_name} Server DEBUG]: {line_str}")
                        else:
                            logger.info(f"[{self.binding_instance_name} Server]: {line_str}")

                except asyncio.IncompleteReadError:
                    # Stream ended before finding newline (expected at process exit)
                    logger.debug(f"Stderr stream ended (IncompleteReadError) for {self.binding_instance_name}.")
                    break
                except Exception as e:
                    # Catch other potential errors during read
                    logger.debug(f"Stderr monitoring error for {self.binding_instance_name}: {e}")
                    break
            logger.info(f"Stderr monitoring thread finished for {self.binding_instance_name} (Lines: {lines_processed}).")

        try:
            # Run the async read function within this thread's event loop
            loop.run_until_complete(read_stderr())
        finally:
            loop.close()

    async def _wait_for_server_health(self, timeout=120):
        """Waits for the server /health endpoint to respond with status 'ok'."""
        if not self.server_port:
            logger.error(f"Cannot wait for server health: Port not assigned for '{self.binding_instance_name}'.")
            return False

        start_time = time.time()
        health_url = f"http://{self.server_host}:{self.server_port}/health"
        last_error_log_time = 0
        log_interval = 10 # Log connection errors every 10s

        while time.time() - start_time < timeout:
            # Check if process died prematurely
            if self.server_process and self.server_process.returncode is not None:
                logger.error(f"LlamaCpp server process for '{self.binding_instance_name}' terminated prematurely (RC: {self.server_process.returncode}).")
                return False

            try:
                # Use a sync request in a thread for the wait loop to avoid blocking main async loop entirely
                # Use the shared requests.Session for potential connection reuse
                response = await asyncio.to_thread(self.session.get, health_url, timeout=2)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        logger.info(f"LlamaCpp '{self.binding_instance_name}': Server is healthy (Ready).")
                        return True
                    else:
                        # Server responded but status is not ok yet
                        logger.debug(f"Health check status not ok: {data.get('status')}")
                else:
                    # Server responded with non-200 status
                    logger.debug(f"Health check received status {response.status_code}")

            except requests.exceptions.ConnectionError:
                # Most common case during startup
                current_time = time.time()
                if current_time - last_error_log_time > log_interval:
                    logger.info(f"Health check: Server for '{self.binding_instance_name}' not ready yet (connection error)... Retrying.")
                    last_error_log_time = current_time
                else:
                    logger.debug("Health check: Server not ready yet (connection error)...")
            except Exception as e:
                # Catch other errors like timeouts, JSON decode errors, etc.
                logger.debug(f"Health check failed with error: {e}")

            # Wait before retrying
            await asyncio.sleep(1)

        # Loop timed out
        logger.error(f"LlamaCpp '{self.binding_instance_name}': Server health check timed out after {timeout} seconds.")
        return False

    async def _terminate_server(self):
        """Attempts to gracefully terminate and then kill the server process."""
        if self.server_process and self.server_process.returncode is None:
            pid = self.server_process.pid
            logger.info(f"LlamaCpp '{self.binding_instance_name}': Terminating server process (PID: {pid})...")
            try:
                # Try graceful termination first
                self.server_process.terminate()
                # Wait for process to exit with a timeout
                await asyncio.wait_for(self.server_process.wait(), timeout=10.0)
                logger.info(f"LlamaCpp '{self.binding_instance_name}': Server process (PID: {pid}) terminated.")
            except asyncio.TimeoutExpired:
                # Termination timed out, force kill
                logger.warning(f"LlamaCpp '{self.binding_instance_name}': Server process (PID: {pid}) did not terminate gracefully. Killing.")
                try:
                    self.server_process.kill()
                    # Wait briefly for kill to register
                    await self.server_process.wait()
                    logger.info(f"LlamaCpp '{self.binding_instance_name}': Server process (PID: {pid}) killed.")
                except ProcessLookupError:
                    # Process already exited between terminate and kill
                    logger.warning(f"LlamaCpp '{self.binding_instance_name}': Process (PID: {pid}) already dead after kill attempt.")
                except Exception as kill_e:
                    logger.error(f"LlamaCpp '{self.binding_instance_name}': Error killing server process (PID: {pid}): {kill_e}")
            except Exception as e:
                logger.error(f"LlamaCpp '{self.binding_instance_name}': Error during server termination (PID: {pid}): {e}")
        elif self.server_process:
             # Process exists but already has a return code
            logger.info(f"LlamaCpp '{self.binding_instance_name}': Server process already terminated (RC: {self.server_process.returncode}).")
        else:
            # No process object exists
             logger.info(f"LlamaCpp '{self.binding_instance_name}': No server process to terminate.")

        # Reset state regardless of termination outcome
        self.server_process = None
        self.server_port = None

    async def unload_model(self) -> bool:
        """Stops the llama.cpp server process and resets state."""
        async with self._load_lock:
            if not self._model_loaded and not self.server_process:
                logger.info(f"LlamaCpp '{self.binding_instance_name}': Already unloaded.")
                return True

            logger.info(f"LlamaCpp '{self.binding_instance_name}': Unloading model by stopping server...")
            await self._terminate_server() # Handle process shutdown

            # Reset binding state
            self.model_name = None
            self._model_loaded = False
            self.current_model_full_path = None
            self.session.close() # Close the sync requests session
            self.session = requests.Session() # Recreate for next potential load

            logger.info(f"LlamaCpp '{self.binding_instance_name}': Model unloaded, server stopped.")
            return True

    def _find_available_port(self) -> Optional[int]:
        """Finds an available TCP port on the host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Binding to port 0 asks the OS for an available ephemeral port
                s.bind(('', 0))
                # Enable address reuse to avoid issues in rapid restart scenarios
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Get the port number assigned by the OS
                port = s.getsockname()[1]
                logger.debug(f"Found available port: {port}")
                return port
        except Exception as e:
            logger.error(f"Error finding available port: {e}", exc_info=True)
            return None

    # --- Generation Methods ---

    async def _api_call(self, endpoint: str, payload: Dict, timeout: Optional[int] = None) -> Dict:
        """Internal helper to make asynchronous API calls to the running server."""
        if not self.server_port:
            raise RuntimeError(f"Llama.cpp server port not set for instance '{self.binding_instance_name}'.")
        if not self.server_process or self.server_process.returncode is not None:
            raise RuntimeError(f"Llama.cpp server process not running for instance '{self.binding_instance_name}'.")

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        request_timeout = timeout or DEFAULT_LLAMACPP_TIMEOUT

        try:
            # Use httpx for async POST request
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status() # Raise exception for 4xx/5xx status codes
                return response.json() # Parse JSON response

        except httpx.TimeoutException:
            logger.error(f"LlamaCpp API call timed out ({request_timeout}s) for '{endpoint}' on '{self.binding_instance_name}'.")
            raise RuntimeError(f"Server request timed out after {request_timeout}s.") from None # Avoid chaining TimeoutException
        except httpx.RequestError as e:
            # Connection errors, invalid URL, etc.
            logger.error(f"LlamaCpp API Request Error ({endpoint}) for instance '{self.binding_instance_name}': {e}")
            # Check if the server died during the request
            if self.server_process and self.server_process.returncode is not None:
                logger.error(f" -> Server process terminated unexpectedly (RC: {self.server_process.returncode}).")
                self._model_loaded = False # Mark as unloaded if server died
            raise RuntimeError(f"Server request error: {e}") from e
        except httpx.HTTPStatusError as e:
            # Specific HTTP errors (4xx, 5xx)
            logger.error(f"LlamaCpp API HTTP Error ({endpoint}) for instance '{self.binding_instance_name}': Status {e.response.status_code}")
            raise RuntimeError(f"Server HTTP Error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            # Other unexpected errors (e.g., JSON decoding errors if response is malformed)
            logger.error(f"LlamaCpp Error during API call ({endpoint}) for '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected API call error: {e}") from e

    async def _prepare_llamacpp_payload(
        self,
        prompt: str,
        params: Dict[str, Any],
        stream: bool,
        multimodal_data: Optional[List['InputData']] = None
    ) -> Optional[Dict[str, Any]]:
        """Maps lollms parameters to llama.cpp /completion payload format."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "stream": stream,
            "temperature": float(params.get("temperature", 0.7)),
            "top_k": int(params.get("top_k", 40)),
            "top_p": float(params.get("top_p", 0.95)),
            "repeat_penalty": float(params.get("repeat_penalty", 1.1)), # llama.cpp uses 'repeat_penalty'
            "n_keep": int(params.get("n_keep", 0)), # Keep initial tokens
            "seed": int(params.get("seed", -1)), # -1 for random
            "n_predict": int(params.get("max_tokens", -1)), # -1 for unlimited (or until ctx limit)
            "cache_prompt": bool(params.get("cache_prompt", True)), # Whether to cache prompt tokens
            "presence_penalty": float(params.get("presence_penalty", 0.0)), # llama.cpp uses these names
            "frequency_penalty": float(params.get("frequency_penalty", 0.0)),
        }

        # Optional parameters
        if "repeat_last_n" in params: # Controls tokens considered for repeat penalty
            payload["repeat_last_n"] = int(params["repeat_last_n"])

        # Stop sequences
        stop = params.get("stop_sequences") or params.get("stop")
        if stop:
            payload["stop"] = stop if isinstance(stop, list) else [stop]

        # Grammar constraint
        if "grammar" in params:
            payload["grammar"] = params["grammar"]

        # Logit bias
        if "logit_bias" in params and isinstance(params["logit_bias"], list):
            try:
                # Validate format: list of [token_id (int), bias (float)]
                valid_bias = []
                for item in params["logit_bias"]:
                    if isinstance(item, list) and len(item) == 2:
                         valid_bias.append([int(item[0]), float(item[1])])
                    else: raise ValueError("Invalid item format") # Short-circuit if any item is bad

                if len(valid_bias) == len(params["logit_bias"]): # Check if all were valid
                     payload["logit_bias"] = valid_bias
                else: # Should not happen if exception is raised correctly
                    logger.warning("Invalid format in logit_bias parameter. Skipping.")
            except (ValueError, TypeError, IndexError) as bias_err:
                logger.warning(f"Could not parse logit_bias: {bias_err}. Skipping.")

        # Mirostat sampling
        mirostat_mode = params.get("mirostat_mode", 0) # 0=disabled, 1=Mirostat, 2=Mirostat 2.0
        if mirostat_mode in [1, 2]:
            payload["mirostat"] = mirostat_mode
            payload["mirostat_tau"] = float(params.get("mirostat_tau", 5.0))
            payload["mirostat_eta"] = float(params.get("mirostat_eta", 0.1))

        # --- Multimodal (LLaVA) Handling ---
        # Check if vision is supported by *currently loaded* model
        has_vision_support = self._model_loaded and self.model_name and \
                             ("llava" in self.model_name.lower() or "bakllava" in self.model_name.lower())

        if multimodal_data and has_vision_support:
            image_parts = [
                item for item in multimodal_data
                if item.type == 'image' and item.data and isinstance(item.data, str)
            ]
            if image_parts:
                logger.info(f"LlamaCpp '{self.binding_instance_name}': Processing {len(image_parts)} image(s) for LLaVA request.")
                # llama.cpp expects 'image_data' list with base64 strings and IDs
                payload["image_data"] = []
                prompt_with_placeholders = payload["prompt"]
                placeholders_present_in_original = False

                # Image placeholders expected by llama.cpp server are like [img-ID] where ID matches image_data
                # User prompt might contain placeholders like <image> or similar. We need to replace them.
                # Standard LoLLMs convention uses [img-N] where N is 1-based index.
                # We map [img-N] from user prompt to [img-ID] for the server (using ID >= 10)
                for idx, img_item in enumerate(image_parts):
                    server_img_id = idx + 10 # Use IDs >= 10 to avoid conflicts with potential future server features
                    user_placeholder = f"[img-{idx+1}]" # Placeholder expected from UI/user

                    # Replace user placeholder with server placeholder if found
                    if user_placeholder in prompt_with_placeholders:
                         placeholders_present_in_original = True
                         prompt_with_placeholders = prompt_with_placeholders.replace(user_placeholder, f"[img-{server_img_id}]")

                    # Add image data in server format
                    payload["image_data"].append({"data": img_item.data, "id": server_img_id})

                if not placeholders_present_in_original:
                    # If no [img-N] found, maybe user used different format or no placeholders.
                    # llama.cpp server might prepend images if no placeholders found. Log a warning.
                     logger.warning(
                        f"LLaVA prompt for instance '{self.binding_instance_name}' might be missing "
                        f"expected image placeholders like [img-1], [img-2], etc. "
                        f"Server might prepend images."
                    )
                    # Keep original prompt in this case
                else:
                    logger.debug(f"Prompt with LLaVA placeholders processed: {prompt_with_placeholders}")
                    payload["prompt"] = prompt_with_placeholders # Update prompt with server placeholders

            elif multimodal_data: # Data present, but not images
                 logger.warning(f"Instance '{self.binding_instance_name}' received non-image multimodal data, ignored by LlamaCpp.")

        elif multimodal_data: # Data present, but model doesn't support vision
            logger.warning(
                f"Instance '{self.binding_instance_name}' received multimodal data, "
                f"but model '{self.model_name}' doesn't appear to support vision. Ignoring images."
            )

        # Auto-adjust max tokens if requested and n_predict is not set
        if params.get("auto_max_new_tokens", False) and payload.get("n_predict", -1) <= 0:
            buffer = params.get("_auto_max_buffer", 20) # Safety buffer tokens
            try:
                # Tokenize the prompt to get its length
                # Use add_bos=True as server usually adds it implicitly for context calculation
                prompt_tokens_list = await self.tokenize(payload["prompt"], add_bos=True, add_eos=False)
                prompt_tokens = len(prompt_tokens_list)
                max_new = self.n_ctx - prompt_tokens - buffer

                if max_new < 1:
                    logger.warning(
                        f"Auto-adjust tokens: Prompt ({prompt_tokens}) + buffer ({buffer}) "
                        f"exceeds context ({self.n_ctx}). Setting n_predict=1."
                    )
                    payload["n_predict"] = 1
                else:
                    payload["n_predict"] = max_new

                logger.info(
                    f"LlamaCpp '{self.binding_instance_name}': Auto-setting n_predict to {payload['n_predict']} "
                    f"(context={self.n_ctx}, prompt_tokens={prompt_tokens}, buffer={buffer})"
                )
            except Exception as token_err:
                # Fallback if tokenization fails
                logger.warning(f"Could not tokenize prompt for auto_max_new_tokens: {token_err}. Using estimate.")
                # Rough estimate: 3 chars per token (very approximate)
                est_prompt_tokens = len(payload["prompt"]) // 3
                max_new = self.n_ctx - est_prompt_tokens - buffer
                payload["n_predict"] = max(1, max_new) # Ensure at least 1 token
                logger.info(
                    f"LlamaCpp '{self.binding_instance_name}': Auto-setting n_predict to {payload['n_predict']} "
                    f"(using estimate)."
                )

        return payload

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> List[Dict[str, Any]]: # Return List[OutputData]-like structure
        """Generates text using the llama.cpp server (non-streaming)."""
        if not self._model_loaded or not self.server_process or self.server_process.returncode is not None:
             raise RuntimeError(f"Llama.cpp server not loaded/running for instance '{self.binding_instance_name}'.")

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Generating non-stream with {self.model_name}...")
        payload = await self._prepare_llamacpp_payload(prompt, params, stream=False, multimodal_data=multimodal_data)
        if not payload:
            raise ValueError("Failed to prepare llama.cpp payload.")

        # Make the non-streaming API call
        result = await self._api_call("completion", payload, timeout=DEFAULT_LLAMACPP_TIMEOUT)

        # Extract the raw completion text
        raw_completion = result.get("content", "")

        # --- ADDED: Parse thoughts ---
        cleaned_completion, thoughts = parse_thought_tags(raw_completion)
        # --------------------------

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Generation successful.")

        # Prepare metadata from the result
        output_metadata = {
            "model_used": self.model_name,
            "binding_instance": self.binding_instance_name,
            "finish_reason": result.get("stop_reason"), # Reason generation stopped
            "timings": result.get("timings"), # Performance timings if available
            "usage": None
        }
        # Add usage info if tokens are reported
        prompt_tok = result.get("tokens_evaluated")
        compl_tok = result.get("tokens_predicted")
        if isinstance(prompt_tok, int) and isinstance(compl_tok, int):
            output_metadata["usage"] = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "total_tokens": prompt_tok + compl_tok
            }

        # Return standardized list format
        return [{
            "type": "text",
            "data": cleaned_completion.strip(),
            "thoughts": thoughts,
            "metadata": output_metadata
        }]

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
         """Generates text using the llama.cpp server (streaming)."""
         if not self._model_loaded or not self.server_process or self.server_process.returncode is not None:
             yield {"type": "error", "content": f"Llama.cpp server not loaded/running for instance '{self.binding_instance_name}'."}
             return

         logger.info(f"LlamaCpp '{self.binding_instance_name}': Generating stream with {self.model_name}...")
         payload = await self._prepare_llamacpp_payload(prompt, params, stream=True, multimodal_data=multimodal_data)
         if not payload:
             yield {"type": "error", "content": "Failed to prepare llama.cpp payload."}
             return

         url = f"http://{self.server_host}:{self.server_port}/completion"

         # State variables for streaming and thought parsing
         full_raw_response_text = ""
         accumulated_thoughts = ""
         is_thinking = False
         final_metadata = {"model_used": self.model_name, "binding_instance": self.binding_instance_name}
         finish_reason = None
         usage_info = None
         last_chunk_stats = {} # Store final stats from the 'stop' chunk

         try:
             async with httpx.AsyncClient(timeout=DEFAULT_LLAMACPP_TIMEOUT) as client:
                # Use client.stream for server-sent events (SSE)
                async with client.stream("POST", url, json=payload) as response:
                    # Check for initial HTTP errors
                    if response.status_code != 200:
                        error_content = await response.aread() # Read the error body
                        logger.error(f"LlamaCpp stream request failed: Status {response.status_code}, Body: {error_content.decode()}")
                        yield {"type": "error", "content": f"Server error {response.status_code}: {error_content.decode()}"}
                        return

                    # Process the SSE stream line by line
                    async for line in response.aiter_lines():
                        # SSE lines start with 'data: '
                        if not line or not line.startswith('data: '):
                            continue

                        line_data_str = line[len('data: '):]
                        if not line_data_str: # Handle empty data lines if they occur
                            continue

                        try:
                            data = json.loads(line_data_str)

                            # Check if this chunk indicates the end of the stream
                            is_done = data.get('stop', False)
                            if is_done:
                                # Store stats from the final chunk (stop_reason, timings, tokens)
                                last_chunk_stats = {
                                    k: v for k, v in data.items()
                                    if k in ['stop_reason', 'timings', 'tokens_evaluated', 'tokens_predicted']
                                }
                                break # Exit the loop, generation is complete

                            # Extract content from the current chunk
                            chunk_raw_content = data.get('content', '')

                            if chunk_raw_content:
                                full_raw_response_text += chunk_raw_content

                                # --- ADDED: Stream parsing logic for thoughts ---
                                current_text_to_process = chunk_raw_content
                                processed_text_chunk = ""
                                processed_thoughts_chunk = None

                                while current_text_to_process:
                                    if is_thinking:
                                        end_tag_pos = current_text_to_process.find("</think>")
                                        if end_tag_pos != -1:
                                            # Found end tag: complete the thought
                                            thought_part = current_text_to_process[:end_tag_pos]
                                            accumulated_thoughts += thought_part
                                            processed_thoughts_chunk = accumulated_thoughts # Yield complete thought
                                            accumulated_thoughts = "" # Reset accumulator
                                            is_thinking = False
                                            current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                                        else:
                                            # End tag not in this chunk part: accumulate and break inner loop
                                            accumulated_thoughts += current_text_to_process
                                            current_text_to_process = ""
                                    else: # Not currently thinking
                                        start_tag_pos = current_text_to_process.find("<think>")
                                        if start_tag_pos != -1:
                                            # Found start tag: yield text before it, start accumulating thought
                                            text_part = current_text_to_process[:start_tag_pos]
                                            processed_text_chunk += text_part
                                            is_thinking = True
                                            current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                                        else:
                                            # No start tag in this chunk part: yield as text and break inner loop
                                            processed_text_chunk += current_text_to_process
                                            current_text_to_process = ""

                                # Yield the processed parts for this chunk
                                if processed_text_chunk or processed_thoughts_chunk:
                                    yield {
                                        "type": "chunk",
                                        "content": processed_text_chunk if processed_text_chunk else None,
                                        "thoughts": processed_thoughts_chunk # Yields None if no thought completed in this chunk
                                    }
                                # --- End Stream parsing logic ---

                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON data line from llama.cpp stream: {line_data_str}")
                        except Exception as chunk_err:
                            logger.error(f"Error processing llama.cpp stream chunk: {chunk_err}", exc_info=True)
                            yield {"type": "error", "content": f"Chunk processing error: {chunk_err}"}
                            finish_reason = "error" # Mark stream as failed
                            break # Exit the loop on chunk processing error

             # --- After the loop finishes or breaks ---
             if finish_reason != "error": # Stream completed normally or was stopped by server
                # Handle any incomplete thought tag at the end
                if is_thinking and accumulated_thoughts:
                    logger.warning(
                        f"LlamaCpp stream ended mid-thought for '{self.binding_instance_name}'. "
                        f"Thought content:\n{accumulated_thoughts}"
                    )
                    final_metadata["incomplete_thoughts"] = accumulated_thoughts

                # Populate final metadata from the last chunk stats
                finish_reason = last_chunk_stats.get("stop_reason", "completed") # Assume completed if not specified
                final_metadata["finish_reason"] = finish_reason
                final_metadata["timings"] = last_chunk_stats.get("timings")

                prompt_tok = last_chunk_stats.get("tokens_evaluated")
                compl_tok = last_chunk_stats.get("tokens_predicted")
                if isinstance(prompt_tok, int) and isinstance(compl_tok, int):
                    final_metadata["usage"] = {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": compl_tok,
                        "total_tokens": prompt_tok + compl_tok
                    }
                    usage_info = final_metadata["usage"] # For logging

                # Re-parse the full accumulated text to get final cleaned output and thoughts
                final_cleaned_text, final_thoughts_str = parse_thought_tags(full_raw_response_text)

                # Append incomplete thoughts to the final thoughts string if they exist
                if final_metadata.get("incomplete_thoughts"):
                    incomplete = final_metadata["incomplete_thoughts"]
                    if final_thoughts_str:
                        final_thoughts_str = (
                            final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + incomplete
                        ).strip()
                    else:
                         final_thoughts_str = incomplete

                # Prepare the final output list
                final_output_list = [{
                    "type": "text",
                    "data": final_cleaned_text.strip(),
                    "thoughts": final_thoughts_str,
                    "metadata": final_metadata # Contains usage, finish reason, timings etc.
                }]
                yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}
                logger.info(f"LlamaCpp stream complete for '{self.binding_instance_name}'. Reason: {finish_reason}, Usage: {usage_info}")

             else: # Stream broke due to an error during chunk processing
                 # Ensure a final error message is yielded
                 yield {
                     "type": "final",
                     "content": [{"type": "error", "data": "Stream processing failed."}],
                     "metadata": {"status": "failed"}
                 }

         except httpx.RequestError as e:
             # Handle errors connecting or during the stream request itself
             logger.error(f"LlamaCpp HTTP Request Error during stream for '{self.binding_instance_name}': {e}")
             yield {"type": "error", "content": f"Server request error: {e}"}
             # Check if the server died
             if self.server_process and self.server_process.returncode is not None:
                 logger.error(f" -> Server process terminated unexpectedly (RC: {self.server_process.returncode}).")
                 self._model_loaded = False
         except Exception as e:
             # Catch any other unexpected errors
             logger.error(f"LlamaCpp Unexpected Stream Error for '{self.binding_instance_name}': {e}", exc_info=True)
             yield {"type": "error", "content": f"Unexpected stream error: {e}"}
         # No finally block needed here as httpx client/response handles resource cleanup


    # --- Tokenizer / Info Methods ---

    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenizes text using the llama.cpp server's /tokenize endpoint."""
        if not self.server_process or self.server_process.returncode is not None:
            raise RuntimeError(f"Llama.cpp server not running for '{self.binding_instance_name}'.")

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Tokenizing text...")
        payload = {"content": text}

        # Note: llama.cpp server /tokenize doesn't support add_bos/add_eos flags directly.
        # The tokenization reflects the model's internal vocabulary handling.
        # BOS/EOS are typically added implicitly by the /completion endpoint based on context.
        if add_bos or add_eos:
            logger.warning("LlamaCpp binding tokenize cannot guarantee add_bos/add_eos via server endpoint.")

        result = await self._api_call("tokenize", payload)
        tokens = result.get("tokens")

        if tokens is None or not isinstance(tokens, list):
            logger.error(f"Tokenization response invalid from '{self.binding_instance_name}': {result}")
            raise RuntimeError("Tokenization failed or returned invalid format.")

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Tokenized into {len(tokens)} tokens.")
        return tokens

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenizes tokens using the llama.cpp server's /detokenize endpoint."""
        if not self.server_process or self.server_process.returncode is not None:
            raise RuntimeError(f"Llama.cpp server not running for '{self.binding_instance_name}'.")

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Detokenizing {len(tokens)} tokens...")
        payload = {"tokens": tokens}
        result = await self._api_call("detokenize", payload)
        text = result.get("content")

        if text is None: # Check for None explicitly, empty string is valid
            logger.error(f"Detokenization response invalid from '{self.binding_instance_name}': {result}")
            raise RuntimeError("Detokenization failed or returned invalid format.")

        logger.info(f"LlamaCpp '{self.binding_instance_name}': Detokenization successful.")
        return text

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded model and server configuration."""
        if not self._model_loaded or not self.model_name:
            # Return default/configured info if not loaded
            return {
                "name": None,
                "context_size": self.n_ctx, # Return configured context size
                "max_output_tokens": -1, # Max output not fixed, depends on context
                "supports_vision": False,
                "supports_audio": False,
                "details": {}
            }

        # Model is loaded, provide details
        is_vision = "llava" in self.model_name.lower() or "bakllava" in self.model_name.lower()
        details = {
            "full_path": str(self.current_model_full_path) if self.current_model_full_path else None,
            "instance_name": self.binding_instance_name,
            "server_port": self.server_port,
            "n_gpu_layers": self.n_gpu_layers,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "threads_batch": self.threads_batch,
            "tensor_split": self.tensor_split,
            "cache_type": self.cache_type,
            "compress_pos_emb": self.compress_pos_emb,
            "rope_freq_base": self.rope_freq_base,
            "additional_args": self.additional_args,
            "server_model_props": None # Placeholder for server properties
        }

        # Try fetching live properties from the server if running
        if self.server_process and self.server_process.returncode is None:
            try:
                # Fetch properties like actual context size from the server
                props_response = await self._api_call("props", {})
                details["server_model_props"] = props_response
                # Update n_ctx based on server report if available
                server_ctx = props_response.get("ctx_size")
                if isinstance(server_ctx, int) and server_ctx > 0:
                    self.n_ctx = server_ctx # Update internal state with actual server context
                else:
                    logger.warning(f"Server props returned invalid ctx_size: {server_ctx}")
            except Exception as e:
                details["server_model_props"] = "error fetching props"
                logger.warning(f"Couldn't fetch server props for instance '{self.binding_instance_name}': {e}")

        return {
            "name": self.model_name,
            "context_size": self.n_ctx, # Return potentially updated context size
            "max_output_tokens": -1,
            "supports_vision": is_vision,
            "supports_audio": False,
            "details": details
        }

    # --- Context Manager and Cleanup ---

    def __enter__(self):
        """Sync context entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context exit - performs best-effort server termination."""
        if self.server_process and self.server_process.returncode is None:
            pid = getattr(self.server_process, 'pid', 'N/A')
            logger.warning(f"LlamaCppBinding exiting context while server process {pid} might be running. Attempting sync termination.")
            try:
                # Use platform-specific forceful termination for sync exit
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/PID", str(self.server_process.pid), "/F"], check=False, capture_output=True)
                else: # Linux/macOS
                    self.server_process.terminate() # Send SIGTERM
                    try:
                        # Wait briefly for termination
                        self.server_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # Force kill if termination fails
                        self.server_process.kill() # Send SIGKILL
            except Exception as e:
                logger.error(f"Error during synchronous __exit__ cleanup for '{self.binding_instance_name}': {e}")
        # Reset state variables immediately in sync exit
        self.server_process = None
        self.server_port = None
        self._model_loaded = False


    def __del__(self):
        """Destructor - unreliable, logs warning if server might be running."""
        # __del__ is unreliable for resource cleanup, especially with async processes.
        # The async unload_model or sync __exit__ should be used explicitly.
        if self.server_process and self.server_process.returncode is None:
            pid = getattr(self.server_process, 'pid', 'N/A')
            logger.warning(
                f"LlamaCppBinding deleted while server process {pid} may still be running. "
                f"Ensure unload_model() was called or use context manager."
             )