# lollms_server/bindings/llamacpp_server_binding.py
import json
import os
import pprint
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any, Set, AsyncGenerator, Tuple
import base64
import asyncio
import tempfile

# Check/Install dependencies
try:
    import pipmaster as pm
    pm.ensure_packages(["requests", "pillow", "aiohttp"]) # pillow for dummy image, aiohttp for async client
except ImportError:
    # If pipmaster is not available, we assume dependencies are met or user will handle them.
    pass

if not pm.is_installed("llama-cpp-binaries"):
    def install_llama_cpp():
        system = platform.system()
        python_version_simple = f"py{sys.version_info.major}{sys.version_info.minor}"
        cuda_suffix = "+cu124" 

        if system == "Windows":
            url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.12.0/llama_cpp_binaries-0.12.0{cuda_suffix}-{python_version_simple}-none-win_amd64.whl"
            fallback_url = "https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.12.0/llama_cpp_binaries-0.12.0+cu124-py3-none-win_amd64.whl"
        elif system == "Linux":
            url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.12.0/llama_cpp_binaries-0.12.0{cuda_suffix}-{python_version_simple}-none-linux_x86_64.whl"
            fallback_url = "https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.12.0/llama_cpp_binaries-0.12.0+cu124-py3-none-linux_x86_64.whl"
        else:
            ASCIIColors.warning(f"Unsupported OS for prebuilt llama-cpp-binaries: {system}. Please install manually.")
            return

        ASCIIColors.info(f"Attempting to install llama-cpp-binaries from: {url}")
        try:
            pm.install(url)
        except Exception as e:
            ASCIIColors.warning(f"Failed to install specific version from {url}: {e}")
            ASCIIColors.info(f"Attempting fallback URL: {fallback_url}")
            try:
                pm.install(fallback_url)
            except Exception as e_fallback:
                ASCIIColors.error(f"Failed to install from fallback URL {fallback_url}: {e_fallback}")
                ASCIIColors.error("Please try installing llama-cpp-binaries manually, e.g., 'pip install llama-cpp-python[server]' or from a wheel.")
    
    # Conditional import of ASCIIColors for installer, actual logging handled by lollms_server later
    try: from ascii_colors import ASCIIColors
    except ImportError: 
        class ASCIIColors: # Dummy for installer if not available globally yet
            @staticmethod
            def warning(x): print(f"WARN: {x}")
            @staticmethod
            def info(x): print(f"INFO: {x}")
            @staticmethod
            def error(x): print(f"ERR: {x}")
    install_llama_cpp()


# Core libraries
try:
    import llama_cpp_binaries
except ImportError:
    # This will be caught by the binding constructor if still not found
    llama_cpp_binaries = None 
try:
    import aiohttp
except ImportError:
    # This will be caught by the binding constructor
    aiohttp = None
try:
    from PIL import Image # For image size validation if needed, not strictly for base64
    pillow_installed = True
except ImportError:
    Image = None
    pillow_installed = False

# lollms_server components
from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from lollms_server.utils.helpers import parse_thought_tags
from lollms_server.utils.paths import Paths # For managing global paths if needed
import platform # Already imported but good to ensure its scope

try:
    from ascii_colors import ASCIIColors, trace_exception # For logging
    import logging # Standard logging
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback if ascii_colors is not in the Python path for some reason
    # (should be handled by lollms_server environment)
    import logging
    logger = logging.getLogger(__name__)
    class ASCIIColors: pass # type: ignore
    def trace_exception(e): logger.exception(e)


# Use TYPE_CHECKING for API model imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lollms_server.api.models import StreamChunk, InputData, OutputData


# --- Predefined patterns (from original) ---
_QUANT_COMPONENTS_SET: Set[str] = {
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q2_K_S", "Q3_K_S", "Q4_K_S", "Q5_K_S",
    "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q3_K_L", "Q2_K_XS", "Q3_K_XS", "Q4_K_XS", "Q5_K_XS", "Q6_K_XS",
    "Q2_K_XXS", "Q3_K_XXS", "Q4_K_XXS", "Q5_K_XXS", "Q6_K_XXS", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "F16", "FP16", "F32", "FP32", "BF16", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS", "IQ3_M_K", "IQ3_S_K", "IQ4_XS_K", "IQ4_NL_K",
    "I8", "I16", "I32", "ALL_F32", "MOSTLY_F16", "MOSTLY_Q4_0", "MOSTLY_Q4_1", "MOSTLY_Q5_0", "MOSTLY_Q5_1",
    "MOSTLY_Q8_0", "MOSTLY_Q2_K", "MOSTLY_Q3_K_S", "MOSTLY_Q3_K_M", "MOSTLY_Q3_K_L",
    "MOSTLY_Q4_K_S", "MOSTLY_Q4_K_M", "MOSTLY_Q5_K_S", "MOSTLY_Q5_K_M", "MOSTLY_Q6_K",
    "MOSTLY_IQ1_S", "MOSTLY_IQ1_M", "MOSTLY_IQ2_XXS", "MOSTLY_IQ2_XS", "MOSTLY_IQ2_S", "MOSTLY_IQ2_M",
    "MOSTLY_IQ3_XXS", "MOSTLY_IQ3_S", "MOSTLY_IQ3_M", "MOSTLY_IQ4_NL", "MOSTLY_IQ4_XS"
}
_MODEL_NAME_SUFFIX_COMPONENTS_SET: Set[str] = {
    "instruct", "chat", "GGUF", "HF", "ggml", "pytorch", "AWQ", "GPTQ", "EXL2",
    "base", "cont", "continue", "ft", "v0.1", "v0.2", "v1.0", "v1.1", "v1.5", "v1.6", "v2.0"
}
_ALL_REMOVABLE_COMPONENTS: List[str] = sorted(
    list(_QUANT_COMPONENTS_SET.union(_MODEL_NAME_SUFFIX_COMPONENTS_SET)), key=len, reverse=True
)

def get_gguf_model_base_name(file_path_or_name: Union[str, Path]) -> str:
    if isinstance(file_path_or_name, str): p = Path(file_path_or_name)
    elif isinstance(file_path_or_name, Path): p = file_path_or_name
    else: raise TypeError(f"Input must be a string or Path object. Got: {type(file_path_or_name)}")
    name_part = p.stem if p.suffix.lower() == ".gguf" else p.name
    if name_part.lower().endswith(".gguf"): name_part = name_part[:-5]
    while True:
        original_name_part_len = len(name_part)
        stripped_in_this_iteration = False
        for component in _ALL_REMOVABLE_COMPONENTS:
            component_lower = component.lower()
            for separator in [".", "-", "_"]:
                pattern_to_check = f"{separator}{component_lower}"
                if name_part.lower().endswith(pattern_to_check):
                    name_part = name_part[:-(len(pattern_to_check))]
                    stripped_in_this_iteration = True; break
            if stripped_in_this_iteration: break
        if not stripped_in_this_iteration or not name_part: break
    while name_part and name_part[-1] in ['.', '-', '_']: name_part = name_part[:-1]
    return name_part

# --- Global Server Registry ---
# These are module-level to be shared across all LlamaCppServerBindingImpl instances
_active_servers: Dict[tuple, 'LlamaCppServerProcess'] = {}
_server_ref_counts: Dict[tuple, int] = {}
_server_registry_lock = threading.Lock() # Use threading.Lock for thread safety

DEFAULT_LLAMACPP_SERVER_HOST = "127.0.0.1"
DEFAULT_LLAMACPP_BASE_PORT_SEARCH = 9600 # Default base for port searching
DEFAULT_MAX_PORT_SEARCH_ATTEMPTS = 100

class LlamaCppServerProcess:
    """
    Manages a single llama.cpp server subprocess.
    This class remains largely synchronous for process management.
    HTTP interactions for health checks during startup are synchronous (using requests).
    """
    def __init__(self, model_path: Union[str, Path], clip_model_path: Optional[Union[str, Path]] = None, 
                 server_binary_path: Optional[Union[str, Path]]=None, server_args: Dict[str, Any]={}):
        self.model_path = Path(model_path)
        self.clip_model_path = Path(clip_model_path) if clip_model_path else None
        
        if server_binary_path:
            self.server_binary_path = Path(server_binary_path)
        elif llama_cpp_binaries:
            self.server_binary_path = Path(llama_cpp_binaries.get_binary_path())
        else:
            # This case should ideally be caught before LlamaCppServerProcess instantiation
            raise FileNotFoundError("llama_cpp_binaries not found and no server_binary_path provided.")

        self.port: Optional[int] = None
        self.server_args = server_args
        self.process: Optional[subprocess.Popen] = None
        self.host = self.server_args.get("host", DEFAULT_LLAMACPP_SERVER_HOST)
        self.base_url: Optional[str] = None
        self.is_healthy = False
        self._stderr_lines: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self.clip_model_path and not self.clip_model_path.exists():
            logger.warning(f"Clip model file '{self.clip_model_path}' not found. Vision features may not work.")
        if not self.server_binary_path.exists():
            raise FileNotFoundError(f"Llama.cpp server binary not found: {self.server_binary_path}")

    def _filter_stderr(self, stderr_pipe):
        try:
            for line_bytes in iter(stderr_pipe.readline, b''): # Read bytes
                line = line_bytes.decode('utf-8', errors='replace').strip() # Decode
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 100: self._stderr_lines.pop(0) # Keep last 100 lines
                    # Log important lines or errors
                    if "llama_model_loaded" in line or "error" in line.lower() or "failed" in line.lower() or "running on port" in line:
                        logger.debug(f"[LLAMA_SERVER_STDERR:{self.port or 'Pre-Port'}] {line}")
        except ValueError: pass # Pipe closed
        except Exception as e: logger.warning(f"Exception in stderr filter thread for port {self.port}: {e}")

    def start(self, port_to_use: int):
        """Starts the llama.cpp server process and waits for it to become healthy."""
        self.port = port_to_use
        self.base_url = f"http://{self.host}:{self.port}"
        
        cmd = [
            str(self.server_binary_path),
            "--model", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
        ]

        arg_map = {
            "n_ctx": "--ctx-size", "n_gpu_layers": "--gpu-layers", "main_gpu": "--main-gpu",
            "tensor_split": "--tensor-split", "use_mmap": (lambda v: ["--no-mmap"] if not v else []),
            "use_mlock": (lambda v: ["--mlock"] if v else []), "seed": "--seed",
            "n_batch": "--batch-size", "n_threads": "--threads", "n_threads_batch": "--threads-batch",
            "rope_scaling_type": "--rope-scaling", "rope_freq_base": "--rope-freq-base",
            "rope_freq_scale": "--rope-freq-scale",
            "embedding": (lambda v: ["--embedding"] if v else []),
            "verbose": (lambda v: ["--verbose"] if v else []),
            "chat_template": "--chat-template",
            "parallel_slots": "--parallel",
        }
        
        if self.clip_model_path:
            cmd.extend(["--mmproj", str(self.clip_model_path)])

        for key, cli_arg in arg_map.items():
            val = self.server_args.get(key)
            if val is not None:
                if callable(cli_arg): cmd.extend(cli_arg(val))
                else: cmd.extend([cli_arg, str(val)])
        
        extra_cli_flags = self.server_args.get("extra_cli_flags", [])
        if isinstance(extra_cli_flags, str): extra_cli_flags = extra_cli_flags.split()
        cmd.extend(extra_cli_flags)

        logger.info(f"Starting Llama.cpp server with command: {' '.join(cmd)}")
        
        env = os.environ.copy()
        if os.name == 'posix' and self.server_binary_path.parent != Path('.'): # Check if not executing from binary dir
            lib_path_str = str(self.server_binary_path.parent.resolve())
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{lib_path_str}:{current_ld_path}" if current_ld_path else lib_path_str
            logger.debug(f"Updated LD_LIBRARY_PATH for server: {env['LD_LIBRARY_PATH']}")

        try:
            # Use Popen with byte streams for stderr/stdout
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0) # bufsize=0 for unbuffered bytes
        except Exception as e:
            logger.error(f"Failed to start llama.cpp server process on port {self.port}: {e}"); trace_exception(e); raise

        self._stderr_thread = threading.Thread(target=self._filter_stderr, args=(self.process.stderr,), daemon=True)
        self._stderr_thread.start()

        health_url = f"{self.base_url}/health"
        max_wait_time = self.server_args.get("server_startup_timeout", 60) # Timeout from server_args
        start_time = time.time()
        
        # Synchronous health check loop using requests
        startup_session = requests.Session()
        while time.time() - start_time < max_wait_time:
            if self.process.poll() is not None:
                stderr_output = "\n".join(self._stderr_lines[-20:]) # Get last 20 lines of stderr
                raise RuntimeError(f"Llama.cpp server (port {self.port}) terminated unexpectedly (exit code {self.process.poll()}) during startup. Stderr:\n{stderr_output}")
            try:
                response = startup_session.get(health_url, timeout=1) # Short timeout for each check
                if response.status_code == 200 and response.json().get("status") == "ok":
                    self.is_healthy = True
                    logger.info(f"Llama.cpp server started successfully on port {self.port}.")
                    startup_session.close()
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(0.5) # Wait and retry
            except Exception as e:
                logger.warning(f"Health check for port {self.port} failed: {e}")
                time.sleep(0.5)
        
        startup_session.close()
        self.is_healthy = False
        # Capture stderr before shutdown if timeout occurs
        stderr_output_timeout = "\n".join(self._stderr_lines[-20:])
        self.shutdown() # Ensure process is cleaned up if health check fails
        raise TimeoutError(f"Llama.cpp server failed to become healthy on port {self.port} within {max_wait_time}s. Stderr:\n{stderr_output_timeout}")

    def shutdown(self):
        self.is_healthy = False
        if self.process:
            logger.info(f"Shutting down Llama.cpp server (PID: {self.process.pid} on port {self.port})...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10) # Wait for graceful termination
            except subprocess.TimeoutExpired:
                logger.warning(f"Llama.cpp server (port {self.port}) did not terminate gracefully, killing...")
                self.process.kill()
                try: self.process.wait(timeout=5) # Wait for kill
                except subprocess.TimeoutExpired: logger.error(f"Failed to kill llama.cpp server process (port {self.port}).")
            except Exception as e: logger.error(f"Error during server shutdown (port {self.port}): {e}")
            finally:
                self.process = None
                # Join stderr thread
                if self._stderr_thread and self._stderr_thread.is_alive():
                    # Help thread exit if stuck on readline
                    if hasattr(self, '_stderr_pipe_obj') and self._stderr_pipe_obj: # Assuming _stderr_pipe_obj is the pipe
                        try: self._stderr_pipe_obj.close() 
                        except: pass
                    self._stderr_thread.join(timeout=2)
                logger.info(f"Llama.cpp server on port {self.port} shut down.")


class LlamaCppServerBindingImpl(Binding):
    binding_type_name = "llamacpp_server_binding"

    DEFAULT_SERVER_ARGS = {
        "n_gpu_layers": 0, "n_ctx": 4096, "n_batch": 512, # Conservative n_ctx
        "embedding": False, "verbose": False, "server_startup_timeout": 120,
        "parallel_slots": 4, 
        "base_port_search": DEFAULT_LLAMACPP_BASE_PORT_SEARCH,
        "max_port_search_attempts": DEFAULT_MAX_PORT_SEARCH_ATTEMPTS,
        "host": DEFAULT_LLAMACPP_SERVER_HOST,
        # Default generation params (can be overridden by request)
        "temperature": 0.7, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.1, "repeat_last_n": 64,
        "seed": -1, # Random seed by default
    }

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager) # Sets self.config, self.binding_instance_name, self.default_model_name
        
        if llama_cpp_binaries is None:
             logger.error("llama-cpp-binaries package is required but not found or failed to import.")
             raise ImportError("llama-cpp-binaries package is required for LlamaCppServerBindingImpl.")
        if aiohttp is None:
            logger.error("aiohttp package is required but not found.")
            raise ImportError("aiohttp package is required for LlamaCppServerBindingImpl.")

        # Configurable paths and settings
        self.models_path = Path(self.config.get("models_path", Paths.personal_models_path() / self.binding_type_name)) # Default to a subfolder in personal_models_path
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.default_completion_format = self.config.get("default_completion_format", "chat") # "chat" or "completion"
        
        # Merge default server args with instance config
        self.server_args = {**self.DEFAULT_SERVER_ARGS, **self.config.get("server_args", {})}
        # Ensure certain critical args from top-level config are preferred if they exist
        for key in ["n_gpu_layers", "n_ctx", "embedding", "host", "base_port_search", "max_port_search_attempts", "server_startup_timeout", "parallel_slots", "temperature", "top_k", "top_p", "repeat_penalty", "repeat_last_n", "seed"]:
            if key in self.config: self.server_args[key] = self.config[key]

        self.server_binary_path = self._get_server_binary_path()
        
        self.current_model_path: Optional[Path] = None
        self.loaded_clip_model_path: Optional[Path] = None # Actual clip model path used by current server
        self.server_process: Optional[LlamaCppServerProcess] = None
        self.port: Optional[int] = None
        self.server_key: Optional[tuple] = None # (model_path_str, clip_model_path_str)
        
        self.model_supports_vision: bool = False # Updated by load_model
        self.current_model_props: Dict[str, Any] = {} # Store props from /props endpoint

        self.aiohttp_session: Optional[aiohttp.ClientSession] = None # Created in load_model, closed in unload

        logger.info(f"LlamaCppServerBindingImpl instance '{self.binding_instance_name}' initialized. Models path: {self.models_path}")

        # Auto-load default model if specified
        if self.default_model_name:
            logger.info(f"Attempting to auto-load default model: {self.default_model_name}")
            # asyncio.create_task(self.load_model(self.default_model_name)) # Careful with unawaited tasks in init
            # For __init__, it's better to schedule this if the server framework handles it, or do it synchronously if required before use.
            # Since load_model is async, direct call is not feasible. For now, user must explicitly load or it loads on first generate.
            pass


    def _get_server_binary_path(self) -> Path:
        custom_path_str = self.config.get("llama_server_binary_path")
        if custom_path_str:
            custom_path = Path(custom_path_str)
            if custom_path.exists() and custom_path.is_file():
                logger.info(f"Using custom llama.cpp server binary: {custom_path}"); return custom_path
            else: logger.warning(f"Custom binary '{custom_path_str}' not found. Falling back.")
        
        if llama_cpp_binaries:
            bin_path_str = llama_cpp_binaries.get_binary_path()
            if bin_path_str:
                bin_path = Path(bin_path_str)
                if bin_path.exists() and bin_path.is_file():
                    logger.info(f"Using binary from llama-cpp-binaries: {bin_path}"); return bin_path
        
        raise FileNotFoundError("Llama.cpp server binary not found. Ensure 'llama-cpp-binaries' or 'llama-cpp-python[server]' is installed, or provide 'llama_server_binary_path' in config.")

    def _resolve_model_path(self, model_name_or_path: str) -> Path:
        model_p = Path(model_name_or_path)
        if model_p.is_absolute():
            if model_p.exists(): return model_p
            else: raise FileNotFoundError(f"Absolute model path specified but not found: {model_p}")
        
        # Check relative to configured models_path
        path_in_models_dir = self.models_path / model_name_or_path
        if path_in_models_dir.exists() and path_in_models_dir.is_file():
            logger.info(f"Found model '{model_name_or_path}' at: {path_in_models_dir}"); return path_in_models_dir
        
        # Check relative to other known model paths if resource_manager provides them (future enhancement)
        # For now, only check self.models_path

        raise FileNotFoundError(f"Model '{model_name_or_path}' not found as absolute path or within '{self.models_path}'.")

    def _find_os_assigned_port(self, host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    def _release_server_instance(self):
        """Synchronously releases a server instance from the registry and shuts it down if ref count is zero."""
        if self.server_process and self.server_key:
            with _server_registry_lock: # Synchronous lock
                if self.server_key in _server_ref_counts:
                    _server_ref_counts[self.server_key] -= 1
                    logger.info(f"Decremented ref count for server {self.server_key}. New count: {_server_ref_counts[self.server_key]}")
                    if _server_ref_counts[self.server_key] <= 0:
                        logger.info(f"Ref count for server {self.server_key} is zero. Shutting it down.")
                        server_to_stop = _active_servers.pop(self.server_key, None)
                        _server_ref_counts.pop(self.server_key, None)
                        if server_to_stop:
                            try: server_to_stop.shutdown() # Synchronous shutdown
                            except Exception as e: logger.error(f"Error shutting down server {self.server_key}: {e}")
                else:
                     logger.warning(f"Server key {self.server_key} not in ref counts during release.")
                     _active_servers.pop(self.server_key, None)

        self.server_process = None
        self.port = None
        self.server_key = None
        self._model_loaded = False
        self.model_name = None
        self.model_supports_vision = False
        self.current_model_props = {}
        # self.aiohttp_session is closed in unload_model

    async def _create_aiohttp_session(self):
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            # You might want to configure connector limits if making many concurrent requests
            # For now, default session.
            self.aiohttp_session = aiohttp.ClientSession()
            logger.debug(f"Created new aiohttp.ClientSession for {self.binding_instance_name}")

    async def _close_aiohttp_session(self):
        if self.aiohttp_session and not self.aiohttp_session.closed:
            await self.aiohttp_session.close()
            logger.debug(f"Closed aiohttp.ClientSession for {self.binding_instance_name}")
            self.aiohttp_session = None

    async def load_model(self, model_name: str) -> bool:
        resolved_model_path = self._resolve_model_path(model_name)
        
        # Determine clip model path
        final_clip_model_path: Optional[Path] = None
        requested_clip_model_name = self.config.get("clip_model_name") # From binding instance config
        if requested_clip_model_name:
            p_clip_req = Path(requested_clip_model_name)
            if p_clip_req.is_absolute() and p_clip_req.exists(): final_clip_model_path = p_clip_req
            elif (self.models_path / requested_clip_model_name).exists(): final_clip_model_path = self.models_path / requested_clip_model_name
            else: logger.warning(f"Specified clip_model_name '{requested_clip_model_name}' not found. Attempting auto-detection.")

        if not final_clip_model_path: # Auto-detection
            base_name = get_gguf_model_base_name(resolved_model_path.stem)
            potential_paths = [
                resolved_model_path.parent / f"{base_name}.mmproj",
                resolved_model_path.parent / f"mmproj-{base_name}.gguf",
                resolved_model_path.with_suffix(".mmproj"),
                self.models_path / f"{base_name}.mmproj",
                self.models_path / f"mmproj-{base_name}.gguf",
            ]
            for p_clip in potential_paths:
                if p_clip.exists():
                    final_clip_model_path = p_clip
                    logger.info(f"Auto-detected LLaVA clip model: {final_clip_model_path}")
                    break
        
        final_clip_model_path_str = str(final_clip_model_path) if final_clip_model_path else None
        new_server_key = (str(resolved_model_path), final_clip_model_path_str)

        # Ensure aiohttp session is ready
        await self._create_aiohttp_session()

        with _server_registry_lock: # Synchronous lock for registry manipulation
            if self.server_process and self.server_key == new_server_key and self.server_process.is_healthy:
                logger.info(f"Model '{model_name}' with clip '{final_clip_model_path_str}' is already loaded and server is healthy on port {self.port}.")
                self._model_loaded = True; self.model_name = model_name
                self.model_supports_vision = final_clip_model_path is not None
                return True

            if self.server_process and self.server_key != new_server_key:
                logger.info(f"Switching models. Releasing previous server: {self.server_key}")
                self._release_server_instance() # This clears self.server_process, port, key

            if new_server_key in _active_servers:
                existing_server = _active_servers[new_server_key]
                if existing_server.is_healthy:
                    logger.info(f"Reusing existing healthy server for {new_server_key} on port {existing_server.port}.")
                    self.server_process = existing_server
                    self.port = existing_server.port
                    _server_ref_counts[new_server_key] += 1
                    self.current_model_path = resolved_model_path
                    self.loaded_clip_model_path = final_clip_model_path
                    self.model_supports_vision = final_clip_model_path is not None
                    self.server_key = new_server_key
                    self._model_loaded = True; self.model_name = model_name
                    await self._fetch_and_store_model_props() # Fetch props for the reused server
                    return True
                else:
                    logger.warning(f"Found unhealthy server for {new_server_key}. Attempting to remove and restart.")
                    try: existing_server.shutdown()
                    except Exception as e: logger.error(f"Error shutting down unhealthy server {new_server_key}: {e}")
                    _active_servers.pop(new_server_key, None)
                    _server_ref_counts.pop(new_server_key, None)
            
            logger.info(f"Starting new server for {new_server_key}.")
            self.current_model_path = resolved_model_path
            self.loaded_clip_model_path = final_clip_model_path
            self.model_supports_vision = final_clip_model_path is not None
            self.server_key = new_server_key

            host_to_bind = self.server_args.get("host", DEFAULT_LLAMACPP_SERVER_HOST)
            base_port_search = self.server_args.get("base_port_search", DEFAULT_LLAMACPP_BASE_PORT_SEARCH)
            max_port_search_attempts = self.server_args.get("max_port_search_attempts", DEFAULT_MAX_PORT_SEARCH_ATTEMPTS)
            new_port_for_server = -1; port_found_sequentially = False
            
            ports_in_use_by_our_servers = {srv.port for srv in _active_servers.values() if srv.port is not None}

            for i in range(max_port_search_attempts):
                candidate_port = base_port_search + i
                if candidate_port > 65535: break 
                if candidate_port in ports_in_use_by_our_servers: continue
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind((host_to_bind, candidate_port))
                    new_port_for_server = candidate_port; port_found_sequentially = True; break
                except socket.error: continue
            
            if not port_found_sequentially:
                new_port_for_server = self._find_os_assigned_port(host_to_bind)
                logger.info(f"Using OS-assigned port {new_port_for_server} for host '{host_to_bind}'.")
            
            current_server_args_for_new_process = self.server_args.copy()
            if "parallel_slots" not in current_server_args_for_new_process or \
               not isinstance(current_server_args_for_new_process["parallel_slots"], int) or \
               current_server_args_for_new_process["parallel_slots"] <=0:
                current_server_args_for_new_process["parallel_slots"] = self.DEFAULT_SERVER_ARGS["parallel_slots"]
            
            logger.info(f"New Llama.cpp server: model={self.current_model_path}, clip={self.loaded_clip_model_path}, port={new_port_for_server}, slots={current_server_args_for_new_process['parallel_slots']}")

            try:
                new_server_process = LlamaCppServerProcess(
                    model_path=str(self.current_model_path),
                    clip_model_path=str(self.loaded_clip_model_path) if self.loaded_clip_model_path else None,
                    server_binary_path=str(self.server_binary_path),
                    server_args=current_server_args_for_new_process,
                )
                # Run synchronous start in a thread to avoid blocking event loop
                await asyncio.to_thread(new_server_process.start, port_to_use=new_port_for_server)

                if new_server_process.is_healthy:
                    self.server_process = new_server_process
                    self.port = new_port_for_server
                    _active_servers[self.server_key] = new_server_process
                    _server_ref_counts[self.server_key] = 1
                    logger.info(f"New server {self.server_key} started on port {self.port}.")
                    self._model_loaded = True; self.model_name = model_name
                    await self._fetch_and_store_model_props()
                    return True
                else: # Should have been caught by new_server_process.start() raising an error
                    logger.error(f"New server {self.server_key} failed to become healthy (this state should be rare).")
                    self._release_server_instance() # Cleanup attempt
                    return False
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}' and start server: {e}")
                trace_exception(e)
                self._release_server_instance()
                return False

    async def unload_model(self) -> bool:
        if self.server_process:
            logger.info(f"Unloading model for binding '{self.binding_instance_name}'. Current server: {self.server_key}, port: {self.port}")
            self._release_server_instance() # Synchronous
        await self._close_aiohttp_session() # Close session associated with this binding instance
        self.current_model_path = None 
        self.loaded_clip_model_path = None
        logger.info(f"Model unloaded for binding '{self.binding_instance_name}'.")
        return True

    async def _get_request_url(self, endpoint: str) -> str:
        if not self.server_process or not self.server_process.is_healthy or not self.port:
            # Attempt to load default model if none is loaded
            effective_model_name = await self._get_effective_model_name()
            if not effective_model_name:
                raise ConnectionError(f"Llama.cpp server for binding '{self.binding_instance_name}' is not running, not healthy, or no model loaded.")
            # If _get_effective_model_name loaded a model, server_process should now be set
            if not self.server_process or not self.server_process.is_healthy or not self.port:
                 raise ConnectionError(f"Llama.cpp server for binding '{self.binding_instance_name}' could not be started/reconnected for model '{effective_model_name}'.")
        
        return f"http://{self.server_process.host}:{self.port}{endpoint}"

    def _prepare_generation_payload(self, prompt: str, system_message: str = "", n_predict: Optional[int] = None,
                                   temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                                   repeat_penalty: float = 1.1, repeat_last_n: Optional[int] = 64,
                                   seed: Optional[int] = None, stream: bool = False, 
                                   use_chat_format_flag: bool = True, # Explicit flag for format
                                   image_paths: Optional[List[str]] = None, **extra_params) -> Dict:
        
        # Start with binding defaults
        payload_params = {
            "temperature": self.server_args.get("temperature", 0.7), 
            "top_k": self.server_args.get("top_k", 40),
            "top_p": self.server_args.get("top_p", 0.9), 
            "repeat_penalty": self.server_args.get("repeat_penalty", 1.1),
            "repeat_last_n": self.server_args.get("repeat_last_n", 64),
            "mirostat": self.server_args.get("mirostat_mode", 0),
            "mirostat_tau": self.server_args.get("mirostat_tau", 5.0),
            "mirostat_eta": self.server_args.get("mirostat_eta", 0.1),
            "seed": self.server_args.get("seed", -1)
        }
        if "grammar" in self.config: # Check for grammar in binding config
             payload_params["grammar"] = self.config["grammar"]
        
        # Override with call-specific parameters
        payload_params.update({
            "temperature": temperature, "top_k": top_k, "top_p": top_p, 
            "repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n, "seed": seed
        })

        if n_predict is not None: payload_params['n_predict'] = n_predict
        payload_params = {k: v for k, v in payload_params.items() if v is not None} # Remove None values
        payload_params.update(extra_params) # Add any other explicitly passed params

        if use_chat_format_flag: # OpenAI-compatible /v1/chat/completions endpoint
            messages = []
            if system_message and system_message.strip(): messages.append({"role": "system", "content": system_message})
            
            user_content: Union[str, List[Dict[str, Any]]] = prompt
            if image_paths and self.model_supports_vision:
                image_parts = []
                for img_path in image_paths:
                    try:
                        with open(img_path, "rb") as image_file: 
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        image_type = Path(img_path).suffix[1:].lower() or "png"
                        image_type = "jpeg" if image_type == "jpg" else image_type
                        image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{encoded_string}"}})
                    except Exception as ex: logger.error(f"Error encoding image {img_path}: {ex}"); trace_exception(ex)
                user_content = [{"type": "text", "text": prompt}] + image_parts
            
            messages.append({"role": "user", "content": user_content})
            final_payload = {"messages": messages, "stream": stream, **payload_params}
            if 'n_predict' in final_payload and 'max_tokens' not in final_payload : # Server uses max_tokens for this endpoint
                final_payload['max_tokens'] = final_payload.pop('n_predict')
            return final_payload
        else: # Legacy /completion endpoint
            full_prompt = f"{system_message}\n\nUSER: {prompt}\nASSISTANT:" if system_message and system_message.strip() else prompt
            final_payload = {"prompt": full_prompt, "stream": stream, **payload_params}
            if image_paths and self.model_supports_vision:
                image_data_list = []
                for i, img_path in enumerate(image_paths):
                    try:
                        with open(img_path, "rb") as image_file: 
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        image_data_list.append({"data": encoded_string, "id": i + 10}) # ID for server to reference
                    except Exception as e_img: logger.error(f"Could not encode image {img_path} for /completion: {e_img}")
                if image_data_list: final_payload["image_data"] = image_data_list
            return final_payload
    
    async def _get_effective_model_name(self) -> Optional[str]:
        """Gets the model name to use, prioritizing loaded, then instance default."""
        if self._model_loaded and self.model_name:
             return self.model_name
        elif self.default_model_name:
             logger.warning(f"No model explicitly loaded for instance '{self.binding_instance_name}'. Attempting to load default '{self.default_model_name}'.")
             if await self.load_model(self.default_model_name): # This sets self.model_name
                 return self.model_name
             else:
                 logger.error(f"Failed to load instance default model '{self.default_model_name}'. Cannot proceed.")
                 return None
        else:
             logger.error(f"No model loaded or configured as default for LlamaCppServer instance '{self.binding_instance_name}'.")
             return None

    async def _fetch_and_store_model_props(self):
        """Fetches model properties from /props and stores them."""
        if not self.aiohttp_session or self.aiohttp_session.closed: await self._create_aiohttp_session()
        try:
            request_url = await self._get_request_url("/props")
            async with self.aiohttp_session.get(request_url, timeout=10) as response:
                response.raise_for_status()
                self.current_model_props = await response.json()
                logger.info(f"Fetched server props for {self.model_name}: {self.current_model_props.get('model_path')}")
                # Update vision support based on props if mmproj is listed
                if not self.model_supports_vision and self.current_model_props.get("mmproj"):
                    logger.info(f"Vision support confirmed by server props (mmproj: {self.current_model_props['mmproj']})")
                    self.model_supports_vision = True

        except Exception as e:
            logger.warning(f"Could not fetch /props from server for {self.model_name}: {e}")
            self.current_model_props = {} # Reset if fetch fails

    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None) -> List[Dict[str, Any]]: # Returns List[OutputData]-like
        effective_model_name = await self._get_effective_model_name()
        if not effective_model_name:
            return [{"type": "error", "content": f"No model available for LlamaCppServer instance '{self.binding_instance_name}'."}]
        if not self.aiohttp_session or self.aiohttp_session.closed: await self._create_aiohttp_session()

        image_paths_on_disk = []
        temp_files_to_clean = []
        try:
            if multimodal_data and self.model_supports_vision:
                for item in multimodal_data:
                    if item.type == 'image' and item.data and isinstance(item.data, str):
                        try:
                            image_bytes = base64.b64decode(item.data)
                            # Use a common extension, server should auto-detect or rely on --mmproj capabilities
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb') as tmp_file:
                                tmp_file.write(image_bytes)
                                image_paths_on_disk.append(tmp_file.name)
                                temp_files_to_clean.append(tmp_file.name)
                            logger.debug(f"Saved temporary image for generation: {tmp_file.name}")
                        except Exception as e: logger.error(f"Failed to save image data for generation: {e}")
            
            use_chat_override = params.get("use_chat_format_override")
            use_chat_flag = (self.default_completion_format == "chat") if use_chat_override is None else use_chat_override
            
            payload = self._prepare_generation_payload(
                prompt=prompt, system_message=params.get("system_message", ""),
                n_predict=params.get("max_tokens"), 
                temperature=params.get("temperature", self.server_args.get("temperature")),
                top_k=params.get("top_k", self.server_args.get("top_k")),
                top_p=params.get("top_p", self.server_args.get("top_p")),
                repeat_penalty=params.get("repeat_penalty", self.server_args.get("repeat_penalty")),
                repeat_last_n=params.get("repeat_last_n", self.server_args.get("repeat_last_n")),
                seed=params.get("seed", self.server_args.get("seed")),
                stream=False, use_chat_format_flag=use_chat_flag,
                image_paths=image_paths_on_disk,
                stop=params.get("stop_sequences") # Pass stop sequences if any
            )
            endpoint = "/v1/chat/completions" if use_chat_flag else "/completion"
            request_url = await self._get_request_url(endpoint)

            logger.debug(f"LlamaCppServer '{self.binding_instance_name}' non-stream request to {request_url} with payload (omitting image data): { {k:v for k,v in payload.items() if k not in ['image_data','messages'] or (k=='messages' and not any('image_url' in part.get('image_url',{}).get('url','').startswith('data:') for m in v for part in (m.get('content') if isinstance(m.get('content'),list) else [])) ) } }")
            
            async with self.aiohttp_session.post(request_url, json=payload, timeout=self.config.get("generation_timeout", 300)) as response:
                response.raise_for_status()
                response_data = await response.json()

            raw_completion = response_data.get('choices', [{}])[0].get('message', {}).get('content', '') if use_chat_flag else response_data.get('content','')
            cleaned_completion, thoughts = parse_thought_tags(raw_completion)
            
            output_metadata = {"model_used": effective_model_name, "binding_instance": self.binding_instance_name, "request_id": request_info.get("request_id")}
            if "timings" in response_data: output_metadata["timings"] = response_data["timings"]
            # TODO: Map timings to lollms_server standard usage structure if possible
            
            return [{"type": "text", "data": cleaned_completion.strip(), "thoughts": thoughts, "metadata": output_metadata}]

        except aiohttp.ClientResponseError as e:
            logger.error(f"Llama.cpp server API error (non-stream) for '{self.binding_instance_name}': {e.status} - {e.message} - {await e.response.text() if e.response else ''}")
            return [{"type": "error", "content": f"API Error ({e.status}): {e.message}", "details": await e.response.text() if e.response else ''}]
        except Exception as ex:
            logger.error(f"Llama.cpp server generation error (non-stream) for '{self.binding_instance_name}': {ex}"); trace_exception(ex)
            return [{"type": "error", "content": str(ex)}]
        finally:
            for p in temp_files_to_clean:
                try: os.unlink(p)
                except OSError: pass

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None) -> AsyncGenerator[Dict[str, Any], None]:
        effective_model_name = await self._get_effective_model_name()
        if not effective_model_name:
            yield {"type": "error", "content": f"No model available for LlamaCppServer instance '{self.binding_instance_name}'."}; return
        if not self.aiohttp_session or self.aiohttp_session.closed: await self._create_aiohttp_session()

        image_paths_on_disk = []
        temp_files_to_clean = []
        try:
            if multimodal_data and self.model_supports_vision:
                for item in multimodal_data:
                    if item.type == 'image' and item.data and isinstance(item.data, str):
                        try:
                            image_bytes = base64.b64decode(item.data)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb') as tmp_file:
                                tmp_file.write(image_bytes)
                                image_paths_on_disk.append(tmp_file.name)
                                temp_files_to_clean.append(tmp_file.name)
                            logger.debug(f"Saved temporary image for stream: {tmp_file.name}")
                        except Exception as e: logger.error(f"Failed to save image data for stream: {e}")
            
            use_chat_override = params.get("use_chat_format_override")
            use_chat_flag = (self.default_completion_format == "chat") if use_chat_override is None else use_chat_override

            payload = self._prepare_generation_payload(
                prompt=prompt, system_message=params.get("system_message", ""),
                n_predict=params.get("max_tokens"), 
                temperature=params.get("temperature", self.server_args.get("temperature")),
                top_k=params.get("top_k", self.server_args.get("top_k")),
                top_p=params.get("top_p", self.server_args.get("top_p")),
                repeat_penalty=params.get("repeat_penalty", self.server_args.get("repeat_penalty")),
                repeat_last_n=params.get("repeat_last_n", self.server_args.get("repeat_last_n")),
                seed=params.get("seed", self.server_args.get("seed")),
                stream=True, use_chat_format_flag=use_chat_flag,
                image_paths=image_paths_on_disk,
                stop=params.get("stop_sequences")
            )
            endpoint = "/v1/chat/completions" if use_chat_flag else "/completion"
            request_url = await self._get_request_url(endpoint)
            
            logger.debug(f"LlamaCppServer '{self.binding_instance_name}' stream request to {request_url} (payload structure similar to non-stream)")

            full_raw_response_text = ""; accumulated_thoughts = ""; is_thinking = False
            final_metadata = {"model_used": effective_model_name, "binding_instance": self.binding_instance_name, "request_id": request_info.get("request_id")}
            last_chunk_timings = {}

            async with self.aiohttp_session.post(request_url, json=payload, timeout=self.config.get("generation_timeout", 300)) as response:
                response.raise_for_status() # Check for HTTP errors before streaming
                async for line_bytes in response.content:
                    if not line_bytes: continue
                    line_str = line_bytes.decode('utf-8').strip()
                    if line_str.startswith('data: '): line_str = line_str[len('data: '):]
                    if line_str == '[DONE]': break
                    try:
                        chunk_data = json.loads(line_str)
                        chunk_raw_content = (chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '') if use_chat_flag else chunk_data.get('content', ''))
                        
                        if chunk_raw_content:
                            full_raw_response_text += chunk_raw_content
                            # Stream parsing for thoughts (same as Ollama example)
                            current_text_to_process = chunk_raw_content; processed_text_chunk = ""; processed_thoughts_chunk = None
                            while current_text_to_process:
                                if is_thinking:
                                    end_tag_pos = current_text_to_process.find("</think>")
                                    if end_tag_pos != -1:
                                        thought_part = current_text_to_process[:end_tag_pos]; accumulated_thoughts += thought_part
                                        processed_thoughts_chunk = accumulated_thoughts; accumulated_thoughts = ""; is_thinking = False
                                        current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                                    else: accumulated_thoughts += current_text_to_process; current_text_to_process = ""
                                else:
                                    start_tag_pos = current_text_to_process.find("<think>")
                                    if start_tag_pos != -1:
                                        text_part = current_text_to_process[:start_tag_pos]; processed_text_chunk += text_part
                                        is_thinking = True; current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                                    else: processed_text_chunk += current_text_to_process; current_text_to_process = ""
                            if processed_text_chunk or processed_thoughts_chunk:
                                yield {"type": "chunk", "content": processed_text_chunk if processed_text_chunk else None, "thoughts": processed_thoughts_chunk}
                        
                        if chunk_data.get('stop', False) or chunk_data.get('stopped_eos',False) or chunk_data.get('stopped_limit',False) or chunk_data.get("generation_stopped"):
                            if "timings" in chunk_data: last_chunk_timings = chunk_data["timings"]
                            break 
                    except json.JSONDecodeError: logger.warning(f"Failed to decode JSON stream chunk: {line_str}"); continue
            
            if is_thinking and accumulated_thoughts:
                final_metadata["incomplete_thoughts"] = accumulated_thoughts
            if last_chunk_timings: final_metadata["timings"] = last_chunk_timings
            
            final_cleaned_text, final_thoughts_str = parse_thought_tags(full_raw_response_text)
            if final_metadata.get("incomplete_thoughts"):
                 final_thoughts_str = (final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + final_metadata["incomplete_thoughts"]).strip() if final_thoughts_str else final_metadata["incomplete_thoughts"]

            final_output_list = [{"type": "text", "data": final_cleaned_text.strip(), "thoughts": final_thoughts_str, "metadata": final_metadata.copy()}] # Copy metadata
            yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}

        except aiohttp.ClientResponseError as e:
            logger.error(f"Llama.cpp server API error (stream) for '{self.binding_instance_name}': {e.status} - {e.message} - {await e.response.text() if e.response else ''}")
            yield {"type": "error", "content": f"API Error ({e.status}): {e.message}", "details": await e.response.text() if e.response else ''}
        except Exception as ex:
            logger.error(f"Llama.cpp server stream error for '{self.binding_instance_name}': {ex}"); trace_exception(ex)
            yield {"type": "error", "content": str(ex)}
        finally:
            for p in temp_files_to_clean:
                try: os.unlink(p)
                except OSError: pass

    async def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False, model_name: Optional[str] = None) -> List[int]:
        if not await self._get_effective_model_name(): raise RuntimeError("No model loaded for tokenization.")
        if not self.aiohttp_session or self.aiohttp_session.closed: await self._create_aiohttp_session()
        try:
            request_url = await self._get_request_url("/tokenize")
            async with self.aiohttp_session.post(request_url, json={"content": text}) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("tokens", [])
        except Exception as e: logger.error(f"Tokenization error for '{self.binding_instance_name}': {e}"); trace_exception(e); return []

    async def detokenize(self, tokens: List[int], model_name: Optional[str] = None) -> str:
        if not await self._get_effective_model_name(): raise RuntimeError("No model loaded for detokenization.")
        if not self.aiohttp_session or self.aiohttp_session.closed: await self._create_aiohttp_session()
        try:
            request_url = await self._get_request_url("/detokenize")
            async with self.aiohttp_session.post(request_url, json={"tokens": tokens}) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("content", "")
        except Exception as e: logger.error(f"Detokenization error for '{self.binding_instance_name}': {e}"); trace_exception(e); return ""

    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        target_model_name_req = model_name
        
        # Determine which model's info to get
        if target_model_name_req is None: # No specific model requested
            if self._model_loaded and self.model_name:
                target_model_name_req = self.model_name # Use current model
                if self.current_model_props and self.current_model_props.get("model_path","").endswith(Path(self.model_name).name): # Check if props match current model name
                    # Return cached props formatted for GetModelInfoResponse
                    return self._format_props_for_model_info(self.current_model_props, self.model_name)
            elif self.default_model_name:
                target_model_name_req = self.default_model_name
            else: # No model context
                return {"binding_instance_name": self.binding_instance_name, "model_name": None, "error": "No model specified or loaded."}
        
        # If info is for a model that isn't the currently loaded one, we can't query its server.
        # This binding can only give detailed info about the *currently running* server's model.
        # For other models, we can only give file-based info.
        if not self._model_loaded or self.model_name != target_model_name_req:
            try:
                resolved_path = self._resolve_model_path(target_model_name_req)
                return {
                    "binding_instance_name": self.binding_instance_name,
                    "model_name": target_model_name_req,
                    "path": str(resolved_path),
                    "size_bytes": resolved_path.stat().st_size,
                    "status": "not_active_in_this_binding_instance",
                    "details": {"comment": "This model is not currently loaded by this binding instance. Info is file-based."}
                }
            except FileNotFoundError:
                 return {"binding_instance_name": self.binding_instance_name, "model_name": target_model_name_req, "error": "Model not found on disk."}

        # If here, we want info for the currently loaded model, fetch fresh if no cache
        if not self.current_model_props or not self.current_model_props.get("model_path","").endswith(Path(self.model_name).name):
            await self._fetch_and_store_model_props()

        return self._format_props_for_model_info(self.current_model_props, self.model_name)

    def _format_props_for_model_info(self, props: Dict[str, Any], model_name_being_served: str) -> Dict[str, Any]:
        """Helper to format server /props into GetModelInfoResponse-like structure."""
        if not props: return {"binding_instance_name": self.binding_instance_name, "model_name": model_name_being_served, "error": "Server properties not available."}

        default_settings = props.get("default_generation_settings", {})
        info = {
            "binding_instance_name": self.binding_instance_name,
            "model_name": model_name_being_served, # Name this binding instance knows it by
            "model_type": "vlm" if self.model_supports_vision else "ttt",
            "context_size": default_settings.get("n_ctx"),
            "max_output_tokens": None, # Not directly available, server uses n_predict per req
            "supports_vision": self.model_supports_vision,
            "supports_audio": False,
            "supports_streaming": True,
            "details": {
                "server_reported_model_path": props.get("model_path"),
                "server_chat_format": props.get("chat_format"),
                "server_clip_model_path": props.get("mmproj"),
                "server_default_n_ctx": default_settings.get("n_ctx"),
                "server_default_n_gpu_layers": default_settings.get("n_gpu_layers"),
                # Add other relevant props if needed
            }
        }
        return info

    async def list_available_models(self) -> List[Dict[str, Any]]:
        models_found = []
        if self.models_path.exists() and self.models_path.is_dir():
            for model_file in self.models_path.rglob("*.gguf"):
                if model_file.is_file():
                    # Basic file info. For more details, model would need to be loaded.
                    models_found.append({
                        "name": model_file.name, # Consistent with OllamaBinding output
                        "model_name": model_file.name, 
                        "path_hint": str(model_file.relative_to(self.models_path.parent) if model_file.is_relative_to(self.models_path.parent) else model_file),
                        "size_bytes": model_file.stat().st_size,
                        "details": {"source": "local_scan", "format": "gguf"}
                    })
        return models_found
    
    def get_supported_input_modalities(self) -> List[str]:
        modalities = ['text']
        if self.model_supports_vision: # Based on loaded_clip_model_path or props
            modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        return ['text']

    async def health_check(self) -> Tuple[bool, str]:
        if not self.server_process or not self.server_process.is_healthy:
            # Check if binary is present as a basic health indicator if no server is running
            if not self.server_binary_path.exists():
                return False, f"Llama.cpp server binary not found at {self.server_binary_path}."
            return False, f"No Llama.cpp server process active for instance '{self.binding_instance_name}'."
        
        if not self.aiohttp_session or self.aiohttp_session.closed: await self._create_aiohttp_session()
        try:
            health_url = await self._get_request_url("/health") # Uses active server's details
            async with self.aiohttp_session.get(health_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "ok":
                        return True, f"Llama.cpp server for '{self.model_name}' on port {self.port} is healthy."
                return False, f"Llama.cpp server on port {self.port} unhealthy (status {response.status})."
        except Exception as e:
            logger.warning(f"Health check failed for Llama.cpp server on port {self.port}: {e}")
            return False, f"Health check connection error for server on port {self.port}: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        # n_gpu_layers from server_args (which includes config overrides)
        gpu_required = self.server_args.get("n_gpu_layers", 0) > 0
        # VRAM estimation is complex; for now, just indicate if GPU is used.
        return {"gpu_required": gpu_required, "estimated_vram_mb": 0} 

    async def shutdown(self):
        """Called by lollms_server framework when the binding is being shut down."""
        logger.info(f"Shutting down LlamaCppServerBindingImpl instance '{self.binding_instance_name}'.")
        await self.unload_model() # This handles server release and aiohttp session closure.

    def __del__(self):
        # This is synchronous. `unload_model` is async.
        # Rely on explicit `shutdown` or `unload_model` calls.
        # If a server process is still tied to this instance specifically (not just ref counted by others),
        # it might be an issue. However, _release_server_instance is sync.
        if self.server_key and self.server_process: # If this instance specifically holds a server ref
            logger.warning(f"LlamaCppServerBindingImpl instance '{self.binding_instance_name}' being deleted. Attempting synchronous release of server {self.server_key}.")
            # This is a best-effort. Async cleanup should be handled by framework calling shutdown().
            self._release_server_instance() 
        
        # Note: Closing aiohttp_session here is tricky because __del__ is sync.
        # It should be closed by an explicit async shutdown method.
        if self.aiohttp_session and not self.aiohttp_session.closed :
            logger.warning(f"aiohttp session for {self.binding_instance_name} still open during __del__. Ensure async shutdown is called.")
            # Attempt to close if loop is running, but this is not robust.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.aiohttp_session.close()) # Fire and forget
                # Else, can't do much here from sync context
            except RuntimeError: # No event loop
                pass