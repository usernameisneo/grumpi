# zoos/bindings/llamacpp_binding.py
import asyncio
import ascii_colors as logging
import sys
import subprocess
import json
import time
import socket
import threading
import os # Added for path.exists and stat
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime # For model info
import httpx
# Use pipmaster to ensure llama-cpp-python is installed for binaries path finding
import pipmaster as pm
import requests # For sync health check during startup wait
try:
    import llama_cpp_binaries # To find default server path
    llamacpp_installed = True
except ImportError:
    pm.install('https://github.com/oobabooga/llama-cpp-binaries/releases/download/textgen-webui/llama_cpp_binaries-0.2.0+cu124-cp311-cp311-win_amd64.whl')
    try:
        import llama_cpp_binaries # To find default server path
        llamacpp_installed = True
    except:
        llama_cpp_binaries = None
        llamacpp_installed = False

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
from ascii_colors import trace_exception
try:
    from lollms_server.api.models import StreamChunk, InputData, ModelInfo
except ImportError:
     class StreamChunk: pass # type: ignore
     class InputData: pass # type: ignore
     class ModelInfo: pass # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_LLAMACPP_TIMEOUT = 3600 * 8

class LlamaCppBinding(Binding):
    """Binding for llama.cpp server inference via its HTTP API."""
    binding_type_name = "llamacpp_binding"

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        super().__init__(config, resource_manager)
        if not llamacpp_installed:
            raise ImportError("LlamaCpp binding requires 'llama-cpp-python' and 'requests'.")

        # Configuration
        # --- CHANGED: Use models_folder instead of model_path ---
        self.models_folder = Path(self.config.get("models_folder", ""))
        if not self.models_folder:
             raise ValueError("Missing 'models_folder' in llama.cpp binding configuration.")
        if not self.models_folder.is_dir():
             logger.warning(f"llama.cpp models_folder does not exist or is not a directory: {self.models_folder}")
             # Allow init but listing/loading will fail later

        self.server_path = self.config.get("server_path")
        self.n_gpu_layers = int(self.config.get("n_gpu_layers", -1))
        self.n_ctx = int(self.config.get("n_ctx", 2048))
        self.batch_size = int(self.config.get("batch_size", 512))
        self.threads = int(self.config.get("threads", 0))
        self.threads_batch = int(self.config.get("threads_batch", 0))
        self.tensor_split = self.config.get("tensor_split")
        self.cache_type = self.config.get("cache_type", "fp16")
        self.compress_pos_emb = float(self.config.get("compress_pos_emb", 1.0))
        self.rope_freq_base = float(self.config.get("rope_freq_base", 0.0))
        self.additional_args = self.config.get("additional_args", {})

        # Internal state
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port: Optional[int] = None
        self.server_host = "127.0.0.1"
        self.session = requests.Session()
        # --- NEW: Store full path of loaded model ---
        self.current_model_full_path: Optional[Path] = None

    @classmethod
    def get_binding_config(cls) -> Dict[str, Any]:
        """Returns metadata about the llama.cpp binding."""
        return {
            "type_name": cls.binding_type_name,
            "version": "1.2", # Incremented version
            "description": "Binding for llama.cpp GGUF models via its HTTP server. Scans a folder for models.",
            "supports_streaming": True,
            "requirements": ["llama-cpp-python>=0.2.60", "requests"],
            "config_template": {
                "type": {"type": "string", "value": cls.binding_type_name, "required":True},
                # --- CHANGED: models_folder instead of model_path ---
                "models_folder": {"type": "string", "value": "models/llama_cpp_models/", "description": "Path to the folder containing GGUF model files.", "required": True},
                "server_path": {"type": "string", "value": None, "description": "Optional path to 'server' executable.", "required": False},
                "n_gpu_layers": {"type": "int", "value": -1, "required": False},
                "n_ctx": {"type": "int", "value": 2048, "required": False},
                "batch_size": {"type": "int", "value": 512, "required": False},
                "threads": {"type": "int", "value": 0, "required": False},
                "threads_batch": {"type": "int", "value": 0, "required": False},
                "tensor_split": {"type": "string", "value": None, "required": False},
                "cache_type": {"type": "string", "value": "fp16", "options":["fp16", "q8_0", "q4_0"], "required": False},
                "compress_pos_emb": {"type":"float", "value":1.0, "required":False},
                "rope_freq_base": {"type":"float", "value":0.0, "required":False},
                "additional_args": {"type":"dict", "value": {}, "description":"Extra cmd flags (e.g. {'--mlock':True})", "required":False}
            }
        }

    # --- Capabilities remain the same ---
    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types."""
        if self._model_loaded and self.model_name and "llava" in self.model_name.lower():
            return ["text", "image"]
        return ["text"]

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ["text"]

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Scans the configured models_folder for GGUF files."""
        logger.info(f"{self.binding_name}: Scanning for models in {self.models_folder}")
        if not self.models_folder.is_dir():
             logger.error(f"Models folder not found: {self.models_folder}")
             return []

        available_models = []
        for gguf_file in self.models_folder.glob("*.gguf"):
            if gguf_file.is_file():
                try:
                    stat_info = gguf_file.stat()
                    model_name = gguf_file.name
                    is_vision = "llava" in model_name.lower()
                    # Cannot easily get context size etc. without loading
                    model_data = {
                        "name": model_name,
                        "size": stat_info.st_size,
                        "modified_at": datetime.fromtimestamp(stat_info.st_mtime),
                        "supports_vision": is_vision,
                        "supports_audio": False,
                        "format": "gguf",
                        "details": {"full_path": str(gguf_file.resolve())} # Store full path
                    }
                    available_models.append(model_data)
                except Exception as e:
                    logger.warning(f"Could not read metadata for {gguf_file}: {e}")

        logger.info(f"{self.binding_name}: Found {len(available_models)} GGUF models.")
        return available_models


    async def health_check(self) -> Tuple[bool, str]:
        """Checks if the server process is running and responding."""
        if not self.server_process or self.server_process.poll() is not None: return False, "Server process not running."
        if not self.server_port: return False, "Server port not set."
        try:
            health_url = f"http://{self.server_host}:{self.server_port}/health"
            response = await asyncio.to_thread(self.session.get, health_url, timeout=5)
            if response.status_code == 200 and response.json().get("status") == "ok": return True, "Server is healthy."
            else: return False, f"Server responded status {response.status_code}: {response.text[:100]}"
        except Exception as e: logger.warning(f"Health check failed: {e}"); return False, f"Health check connection failed: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """llama.cpp uses GPU if layers > 0."""
        return {"gpu_required": self.n_gpu_layers != 0, "estimated_vram_mb": 0}

    async def load_model(self, model_name: str) -> bool:
        """Starts the llama.cpp server process with the selected model file."""
        async with self._load_lock:
            target_model_path = self.models_folder / model_name
            if not target_model_path.is_file():
                logger.error(f"{self.binding_name}: Target model file not found: {target_model_path}")
                return False

            # --- Stop existing server if loading a different model ---
            if self._model_loaded and self.current_model_full_path != target_model_path:
                 logger.info(f"{self.binding_name}: Different model requested. Stopping existing server for {self.model_name}...")
                 await self.unload_model() # This resets state and terminates process
            elif self._model_loaded and self.server_process and self.server_process.poll() is None:
                 logger.info(f"{self.binding_name}: Model '{model_name}' server already running.")
                 return True # Correct model is already running

            logger.info(f"{self.binding_name}: Starting server for model: {model_name}")
            self.server_port = self._find_available_port()
            logger.info(f"{self.binding_name}: Using port {self.server_port}")

            server_executable = self.server_path or llama_cpp_binaries.get_binary_path("server")
            if not Path(server_executable).exists(): logger.error(f"llama.cpp server binary not found: '{server_executable}'."); return False

            # --- Use target_model_path in the command ---
            cmd = [
                server_executable, "--model", str(target_model_path), # Use selected model path
                "--ctx-size", str(self.n_ctx), "--n-gpu-layers", str(self.n_gpu_layers),
                "--batch-size", str(self.batch_size), "--port", str(self.server_port),
            ]
            if self.threads > 0: cmd += ["--threads", str(self.threads)]
            if self.threads_batch > 0: cmd += ["--threads-batch", str(self.threads_batch)]
            if self.tensor_split: cmd += ["--tensor-split", self.tensor_split]
            if self.cache_type != "fp16" and self.cache_type in ["q8_0", "q4_0"]: cmd += ["--cache-type-k", self.cache_type, "--cache-type-v", self.cache_type]
            if self.compress_pos_emb != 1.0: cmd += ["--rope-freq-scale", str(1.0 / self.compress_pos_emb)]
            if self.rope_freq_base > 0.0: cmd += ["--rope-freq-base", str(self.rope_freq_base)]
            if isinstance(self.additional_args, dict):
                for flag, value in self.additional_args.items():
                    if value is True: cmd.append(flag)
                    elif value is not False and value is not None: cmd.extend([flag, str(value)])

            needs_gpu = self.n_gpu_layers != 0
            resource_context = self.resource_manager.acquire_gpu_resource(f"load_{self.binding_name}_{model_name}") if needs_gpu else nullcontext()

            try:
                async with resource_context:
                    logger.info(f"Starting server process: {' '.join(cmd)}")
                    self.server_process = await asyncio.create_subprocess_exec( *cmd, stderr=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL )
                    threading.Thread(target=self._monitor_stderr, args=(self.server_process.stderr,), daemon=True).start()
                    if not await self._wait_for_server_health(): raise RuntimeError("Server failed health check.")
                    self.model_name = model_name # Store filename
                    self.current_model_full_path = target_model_path # Store full path
                    self._model_loaded = True
                    logger.info(f"{self.binding_name}: Server started for {model_name}.")
                    return True
            except asyncio.TimeoutError: logger.error(f"{self.binding_name}: Timeout waiting for GPU."); await self._terminate_server(); return False
            except Exception as e: logger.error(f"{self.binding_name}: Failed server start: {e}", exc_info=True); await self._terminate_server(); return False

    def _monitor_stderr(self, stream):
        """Reads and logs stderr lines."""
        try:
            while True:
                line = stream.readline()
                if not line: break
                line_str = line.decode('utf-8', errors='replace').strip()
                if line_str and not line_str.startswith(('srv ', 'slot ')) and '/health' not in line_str: logger.info(f"[{self.binding_name} Server]: {line_str}")
        except Exception as e: logger.debug(f"Stderr monitoring stopped for {self.binding_name}: {e}")
        finally:
             if hasattr(stream, 'close'): stream.close()

    async def _wait_for_server_health(self, timeout=DEFAULT_LLAMACPP_TIMEOUT):
        """Waits for the server /health endpoint."""
        start_time = time.time(); health_url = f"http://{self.server_host}:{self.server_port}/health"
        while time.time() - start_time < timeout:
            if self.server_process and self.server_process.poll() is not None: logger.error(f"{self.binding_name}: Server terminated prematurely."); return False
            try:
                response = await asyncio.to_thread(self.session.get, health_url, timeout=1)
                if response.status_code == 200 and response.json().get("status") == "ok": logger.info(f"{self.binding_name}: Server healthy."); return True
            except Exception: pass
            await asyncio.sleep(1)
        logger.error(f"{self.binding_name}: Health check timed out."); return False

    async def _terminate_server(self):
        """Stops the server process."""
        if self.server_process and self.server_process.poll() is None:
            logger.info(f"{self.binding_name}: Terminating server process {self.server_process.pid}...")
            try:
                self.server_process.terminate(); await asyncio.wait_for(self.server_process.wait(), timeout=5)
                logger.info(f"{self.binding_name}: Server process terminated.")
            except asyncio.TimeoutExpired: logger.warning(f"{self.binding_name}: Server kill required."); self.server_process.kill(); await self.server_process.wait()
            except Exception as e: logger.error(f"{self.binding_name}: Error terminating server: {e}")
        self.server_process = None; self.server_port = None

    async def unload_model(self) -> bool:
        """Stops the llama.cpp server process."""
        async with self._load_lock:
            logger.info(f"{self.binding_name}: Unloading model by stopping server...")
            await self._terminate_server()
            self.model_name = None; self._model_loaded = False; self.current_model_full_path = None # Reset full path
            self.session.close(); self.session = requests.Session() # Recreate session
            logger.info(f"{self.binding_name}: Model unloaded, server stopped.")
            return True

    def _find_available_port(self):
        """Finds an available TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0)); return s.getsockname()[1]

    async def _encode(self, text: str, add_bos: bool) -> Optional[List[int]]:
        """Internal helper to call /tokenize."""
        if not self.server_port: return None
        url = f"http://{self.server_host}:{self.server_port}/tokenize"; payload = {"content": text, "add_special": add_bos}
        try:
            response = await asyncio.to_thread(self.session.post, url, json=payload, timeout=10); response.raise_for_status()
            return response.json().get("tokens")
        except Exception as e: logger.error(f"{self.binding_name}: Error tokenizing: {e}"); return None

    def _prepare_llamacpp_payload(self, prompt: str, params: Dict[str, Any], stream: bool, multimodal_data: Optional[List['InputData']] = None) -> Optional[Dict[str, Any]]:
        """Maps lollms parameters to llama.cpp /completion payload."""
        payload = { "prompt": prompt, "stream": stream, "temperature": params.get("temperature", 0.7), "top_k": params.get("top_k", 40), "top_p": params.get("top_p", 0.95), "repeat_penalty": params.get("repeat_penalty", 1.1), "repeat_last_n": params.get("repeat_last_n", 64), "seed": params.get("seed", -1), "n_predict": params.get("max_tokens", -1), "ignore_eos": params.get("ignore_eos", False), }
        stop = params.get("stop_sequences") or params.get("stop")
        if stop: payload["stop"] = stop if isinstance(stop, list) else [stop]
        if "grammar" in params: payload["grammar"] = params["grammar"]
        if "logit_bias" in params: payload["logit_bias"] = params["logit_bias"]
        options = params.get("options", {}); payload.update(options)
        if "min_p" in params: payload["min_p"] = params["min_p"]
        if "presence_penalty" in params: payload["presence_penalty"] = params["presence_penalty"]
        if "frequency_penalty" in params: payload["frequency_penalty"] = params["frequency_penalty"]
        if "mirostat_mode" in params: payload["mirostat"] = params["mirostat_mode"]
        if "mirostat_tau" in params: payload["mirostat_tau"] = params["mirostat_tau"]
        if "mirostat_eta" in params: payload["mirostat_eta"] = params["mirostat_eta"]

        # --- Multimodal Data Handling (LLaVA example) ---
        if multimodal_data and self.get_supported_input_modalities() == ["text", "image"]:
            image_parts = [item for item in multimodal_data if item.type=='image']
            if image_parts:
                logger.info(f"Processing {len(image_parts)} image(s) for LLaVA request.")
                # llama.cpp server expects 'image_data' field with list of {"data": base64, "id": int}
                payload["image_data"] = []
                for idx, img_item in enumerate(image_parts):
                    if img_item.data:
                         payload["image_data"].append({"data": img_item.data, "id": idx + 10}) # Use arbitrary ID > 0
                    else: logger.warning(f"Skipping image item {idx} with missing data.")
                # Prompt needs placeholders like '[img-ID]' e.g., "[img-10] Describe this"
                # This assumes the prompt is already formatted correctly by the user/personality
                # Simple check:
                if not any(f"[img-{i+10}]" in payload["prompt"] for i in range(len(image_parts))):
                     logger.warning("LLaVA prompt may be missing image placeholders like '[img-ID]'.")
        # --- End Multimodal ---

        return payload

    async def generate( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> Union[str, Dict[str, Any]]:
        """Generates text using the llama.cpp server (non-streaming)."""
        if not self._model_loaded or not self.server_instance: raise RuntimeError("Server not loaded/running.")
        logger.info(f"{self.binding_name}: Generating non-stream with {self.model_name}...")

        payload = self._prepare_llamacpp_payload(prompt, params, stream=False, multimodal_data=multimodal_data)
        if not payload: raise ValueError("Failed payload prep.")

        # Tokenize multimodal prompt if needed (currently server handles text prompt tokenization)
        # If LLaVA, prompt might already be formatted. If not, tokenization needs to happen here or be adjusted.
        # For now, assume server handles text tokenization internally. If sending tokens:
        # token_ids = await self._encode(payload["prompt"], add_bos=params.get("add_bos_token", True));
        # if token_ids is None: raise RuntimeError("Tokenization failed."); payload["prompt"] = token_ids

        if params.get("auto_max_new_tokens", False) and payload.get("n_predict", -1) < 0:
            # Estimate prompt tokens - very rough!
            est_prompt_tokens = len(payload["prompt"].split()) # Very basic estimate
            max_new = self.n_ctx - est_prompt_tokens - 4; payload["n_predict"] = max(1, max_new)
            logger.info(f"Auto-setting n_predict to {payload['n_predict']}")

        url = f"http://{self.server_host}:{self.server_port}/completion"
        try:
            response = await asyncio.to_thread(self.session.post, url, json=payload, timeout=DEFAULT_LLAMACPP_TIMEOUT)
            response.raise_for_status(); result = response.json(); completion = result.get("content", "")
            logger.info(f"{self.binding_name}: Generation successful.")
            return {"text": completion.strip()}
        except requests.exceptions.RequestException as e: logger.error(f"{self.binding_name} Request Error: {e}"); raise RuntimeError(f"Server request error: {e}") from e
        except Exception as e: logger.error(f"{self.binding_name} Generate Error: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}") from e

    async def generate_stream( self, prompt: str, params: Dict[str, Any], request_info: Dict[str, Any], multimodal_data: Optional[List['InputData']] = None ) -> AsyncGenerator[Dict[str, Any], None]:
         """Generates text using the llama.cpp server (streaming)."""
         if not self._model_loaded or not self.server_instance: yield {"type":"error", "content":"Server not loaded."}; return
         logger.info(f"{self.binding_name}: Generating stream with {self.model_name}...")

         payload = self._prepare_llamacpp_payload(prompt, params, stream=True, multimodal_data=multimodal_data)
         if not payload: yield {"type":"error", "content":"Payload prep failed."}; return

         # Tokenization handled by server in current setup for text
         # If sending tokens:
         # token_ids = await self._encode(payload["prompt"], add_bos=params.get("add_bos_token", True));
         # if token_ids is None: yield {"type":"error", "content":"Tokenization failed."}; return; payload["prompt"] = token_ids

         if params.get("auto_max_new_tokens", False) and payload.get("n_predict", -1) < 0:
             est_prompt_tokens = len(payload["prompt"].split()); max_new = self.n_ctx - est_prompt_tokens - 4
             payload["n_predict"] = max(1, max_new); logger.info(f"Auto-setting n_predict to {payload['n_predict']}")

         url = f"http://{self.server_host}:{self.server_port}/completion"
         full_response_text = ""
         try:
             async with httpx.AsyncClient(timeout=DEFAULT_LLAMACPP_TIMEOUT) as client: # Use httpx for async streaming
                  async with client.stream("POST", url, json=payload) as response:
                       response.raise_for_status()
                       async for line in response.aiter_lines():
                            if not line or not line.startswith('data: '): continue
                            line = line[6:]
                            try:
                                 data = json.loads(line); chunk_content = data.get('content', '')
                                 if chunk_content: full_response_text += chunk_content; yield {"type": "chunk", "content": chunk_content}
                                 if data.get('stop', False):
                                     logger.info(f"{self.binding_name}: Stream finished."); metadata = {"timings": data.get("timings")} if "timings" in data else {}
                                     yield {"type": "final", "content": {"text": full_response_text}, "metadata": metadata}; break
                            except json.JSONDecodeError: logger.warning(f"Non-JSON stream line: {line}")
                            except Exception as chunk_err: logger.error(f"Chunk processing error: {chunk_err}"); yield {"type": "error", "content": f"Chunk error: {chunk_err}"}; break
         except httpx.RequestError as e: logger.error(f"{self.binding_name} Request Error stream: {e}"); yield {"type": "error", "content": f"Server request error: {e}"}
         except Exception as e: logger.error(f"{self.binding_name} Stream Error: {e}", exc_info=True); yield {"type": "error", "content": f"Unexpected stream error: {e}"}


    # Keep __del__ and context manager methods
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): asyncio.run(self.unload_model())
    def __del__(self):
         if self.server_process and self.server_process.poll() is None:
            logger.warning(f"LlamaCppBinding deleted without explicit stop. Terminating server {self.server_process.pid}.")
            pass # Avoid risky sync unload

    @property
    def server_instance(self):
        """Checks if server process is running."""
        return self.server_process and self.server_process.poll() is None