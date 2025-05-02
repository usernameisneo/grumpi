# encoding:utf-8
# Project: lollms_server
# File: zoos/bindings/openai_binding/__init__.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Binding implementation for OpenAI and compatible APIs.

import asyncio
import os
import base64 # For image processing
from io import BytesIO # For image processing
from datetime import datetime
from typing import Dict, Any, Optional, Union, AsyncGenerator, Tuple, List

# Use pipmaster if needed, or rely on main installation
import pipmaster as pm
pm.ensure_packages(["openai","pillow"])

from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, NotFoundError
# Import specific types for clarity and type checking
from openai.types.model import Model as OpenaiModelType
from openai.types.chat import ChatCompletionChunk # For stream types
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam # For message structure
from PIL import Image # Optional: for potentially validating base64 data
openai_installed = True

import ascii_colors as logging # Use logging alias
from ascii_colors import ASCIIColors, trace_exception

from lollms_server.core.bindings import Binding
from lollms_server.core.resource_manager import ResourceManager
# Use TYPE_CHECKING for API model imports
from typing import TYPE_CHECKING
from lollms_server.api.models import StreamChunk, InputData
from lollms_server.utils.helpers import parse_thought_tags

logger = logging.getLogger(__name__)

class OpenAIBinding(Binding):
    """Binding for OpenAI and compatible APIs."""
    binding_type_name = "openai_binding" # MUST match type_name in binding_card.yaml

    def __init__(self, config: Dict[str, Any], resource_manager: ResourceManager):
        """
        Initializes the OpenAIBinding using the instance configuration.

        Args:
            config: The configuration dictionary loaded from the instance's config file,
                    validated against the schema in binding_card.yaml.
            resource_manager: The shared resource manager instance.
        """
        super().__init__(config, resource_manager)
        if not openai_installed or not AsyncOpenAI: # Check if class was imported
            raise ImportError("OpenAI binding requires the 'openai' library (>=1.0.0). Please install it.")

        # Load config directly from the validated instance config dictionary (self.config)
        self.api_key = self.config.get("api_key") # Get from instance config first
        self.base_url = self.config.get("base_url")
        # User override for context size from instance config, can be None
        self.instance_context_size_override = self.config.get("context_size")

        # Fallback to environment variable if not in instance config
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                logger.info(f"OpenAI Binding '{self.binding_instance_name}': Loaded API key from OPENAI_API_KEY environment variable.")

        # --- Client Initialization ---
        if not self.api_key:
            logger.warning(f"OpenAI Binding '{self.binding_instance_name}': API key not found in config or environment variables. API calls will likely fail.")
            # Initialize client even without key for potential health check failure message
            self.client = AsyncOpenAI(api_key=None, base_url=self.base_url)
        else:
            try:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"OpenAI Binding '{self.binding_instance_name}': Initialized client (Base URL: {self.base_url or 'Default OpenAI'}).")
            except Exception as e:
                 logger.error(f"Failed to initialize OpenAI client for instance '{self.binding_instance_name}': {e}")
                 trace_exception(e)
                 self.client = None # Mark client as failed

        # --- Internal State ---
        self.model_name: Optional[str] = None # The specific model identifier (e.g., "gpt-4o")
        self.model_supports_vision: bool = False
        # This will hold the effective context size after model load/detection
        self.current_context_size: Optional[int] = self.instance_context_size_override
        self.current_max_output_tokens: Optional[int] = None # Typically set per-request for OpenAI


    # --- Helper to parse OpenAI model details ---
    def _parse_openai_details(self, model: OpenaiModelType) -> Dict[str, Any]:
        """Parses the raw OpenAI model object attributes."""
        parsed = {}
        name = model.id
        parsed['name'] = name
        try:
            # Use datetime.utcfromtimestamp if created is a Unix timestamp
            parsed['modified_at'] = datetime.utcfromtimestamp(model.created) if model.created else None
        except TypeError: # Handle if model.created is already datetime or None
            parsed['modified_at'] = model.created if isinstance(model.created, datetime) else None
        except Exception as time_e:
             logger.warning(f"Could not parse created timestamp {model.created}: {time_e}")
             parsed['modified_at'] = None

        # Initialize standard fields
        parsed['size'] = None
        parsed['quantization_level'] = None
        parsed['format'] = "api"
        parsed['family'] = None # Attempt to guess below
        parsed['families'] = None # Attempt to guess below
        parsed['parameter_size'] = None # Not provided by OpenAI API
        parsed['template'] = None # Not provided
        parsed['license'] = None # Not provided

        # Context size guessing based on common model prefixes
        context_size = None
        if name.startswith("gpt-4-turbo") or name.startswith("gpt-4o"): context_size = 128000
        elif name.startswith("gpt-4-32k"): context_size = 32768
        elif name.startswith("gpt-4"): context_size = 8192
        elif name.startswith("gpt-3.5-turbo-16k") or name.startswith("gpt-3.5-turbo-0125"): context_size = 16385 # Newer 3.5 also has 16k
        elif name.startswith("gpt-3.5-turbo") or name.startswith("gpt-3.5-turbo-instruct"): context_size = 4096
        # Add other known prefixes if needed
        parsed['context_size'] = context_size

        # Max output tokens (often 4096 for newer chat models, but not fixed)
        parsed['max_output_tokens'] = None # Not reliably available from API

        # Vision support guessing
        supports_vision = any(tag in name.lower() for tag in ['vision', 'gpt-4v', 'gpt-4o'])
        parsed['supports_vision'] = supports_vision
        parsed['supports_audio'] = False # Assume false unless specifically known

        # Family guessing
        if name.startswith("gpt-4"): parsed['family'] = "gpt-4"; parsed['families'] = ["gpt-4"]
        elif name.startswith("gpt-3.5"): parsed['family'] = "gpt-3.5"; parsed['families'] = ["gpt-3.5"]

        # Add remaining raw details
        parsed['details'] = {
            "original_id": model.id,
            "object": model.object,
            "owned_by": model.owned_by,
        }
        return parsed


    # --- Required Binding Methods ---

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists models available via the configured API endpoint."""
        if not self.client:
             logger.error(f"OpenAI client not initialized for instance '{self.binding_instance_name}'. Cannot list models.")
             return []
        if not self.api_key:
            logger.warning(f"OpenAI '{self.binding_instance_name}': No API key configured. Cannot list models.")
            return []

        logger.info(f"OpenAI '{self.binding_instance_name}': Listing models from endpoint {self.base_url or 'Default OpenAI'}...")
        available_models = []
        try:
            models_response = await self.client.models.list()
            # Filter models - be more inclusive to support compatible APIs
            # Look for common patterns or just list most text/chat models
            for model in models_response.data:
                # Include models with 'gpt', 'instruct', 'text-', 'claude', 'mistral', 'llama', 'gemini', 'command', 'ft:' (fine-tuned) etc.
                model_id_lower = model.id.lower()
                if model.id and any(tag in model_id_lower for tag in ['gpt', 'instruct', 'text-', 'claude', 'mistral', 'llama', 'gemini', 'command', 'ft:', '/']):
                     parsed_data = self._parse_openai_details(model)
                     if parsed_data.get('name'):
                         available_models.append(parsed_data)
                else:
                    logger.debug(f"Skipping model '{model.id}' during list (doesn't match common patterns).")

            logger.info(f"OpenAI '{self.binding_instance_name}': Found {len(available_models)} potential models.")
            return available_models
        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error listing models for '{self.binding_instance_name}': {e}")
            raise RuntimeError(f"API Connection Error: {e}") from e
        except OpenAIError as e:
            logger.error(f"OpenAI API error listing models for '{self.binding_instance_name}': {e}")
            status = getattr(e, 'status_code', 'N/A')
            if status == 401:
                 raise RuntimeError(f"OpenAI Authentication Error (Invalid Key?): {e}") from e
            raise RuntimeError(f"OpenAI API Error ({status}): {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error listing OpenAI models for '{self.binding_instance_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error contacting OpenAI endpoint: {e}") from e

    # get_binding_config is REMOVED

    def get_supported_input_modalities(self) -> List[str]:
        """Returns supported input types based on loaded model state."""
        modalities = ['text']
        if self._model_loaded and self.model_supports_vision:
             modalities.append('image')
        return modalities

    def get_supported_output_modalities(self) -> List[str]:
        """Returns supported output types."""
        return ['text'] # OpenAI chat models output text

    async def health_check(self) -> Tuple[bool, str]:
        """Checks API key validity and connection by listing models."""
        if not self.client:
             return False, f"OpenAI client initialization failed for instance '{self.binding_instance_name}'."
        if not self.api_key:
            return False, "API key is not configured for this instance."
        try:
            # Use a timeout to prevent hanging indefinitely
            models = await asyncio.wait_for(self.client.models.list(), timeout=15.0) # Increased timeout slightly
            return True, f"Connection OK ({len(models.data)} models found)."
        except APIConnectionError as e:
            logger.error(f"OpenAI Health check fail for '{self.binding_instance_name}' (Connection Error): {e}")
            return False, f"API Connection Error: {e}"
        except RateLimitError as e:
            logger.warning(f"OpenAI Health check hit rate limit for '{self.binding_instance_name}': {e}")
            # Consider rate limit as 'healthy' connection but potentially unusable
            return True, f"Rate limit exceeded during check: {e}"
        except OpenAIError as e:
            logger.error(f"OpenAI Health check fail for '{self.binding_instance_name}' (OpenAI Error): {e}")
            status = getattr(e, 'status_code', 'N/A')
            if status == 401:
                return False, f"Authentication Error (Invalid API Key?): {e}"
            return False, f"OpenAI API Error ({status}): {e}"
        except asyncio.TimeoutError:
            logger.error(f"OpenAI Health check timed out for '{self.binding_instance_name}'.")
            return False, "Connection timed out."
        except Exception as e:
            logger.error(f"OpenAI Health check fail for '{self.binding_instance_name}' (Unexpected Error): {e}", exc_info=True)
            return False, f"Unexpected Error: {e}"

    def get_resource_requirements(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """OpenAI is remote, no local GPU needed."""
        return {"gpu_required": False}

    async def load_model(self, model_name: str) -> bool:
        """Sets the model name and determines capabilities (vision, context size)."""
        async with self._load_lock:
            if self._model_loaded and self.model_name == model_name:
                 logger.info(f"OpenAI '{self.binding_instance_name}': Model '{model_name}' already active.")
                 return True

            logger.info(f"OpenAI '{self.binding_instance_name}': Setting active model to '{model_name}'.")
            # Determine vision support based on name conventions
            self.model_supports_vision = any(tag in model_name.lower() for tag in ['vision', 'gpt-4v', 'gpt-4o'])
            logger.info(f" -> Detected Vision Support: {self.model_supports_vision}")

            # Determine context size: Priority -> Instance Override > Detected > Default (4096)
            detected_size = None
            if model_name.startswith("gpt-4-turbo") or model_name.startswith("gpt-4o"): detected_size = 128000
            elif model_name.startswith("gpt-4-32k"): detected_size = 32768
            elif model_name.startswith("gpt-4"): detected_size = 8192
            elif model_name.startswith("gpt-3.5-turbo-16k") or model_name.startswith("gpt-3.5-turbo-0125"): detected_size = 16385
            elif model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-3.5-turbo-instruct"): detected_size = 4096
            # Add checks for other model families if using compatible endpoints (Mistral, Claude, etc.)
            elif "mistral" in model_name.lower(): detected_size = 32768 # Common Mistral size
            elif "claude-3-opus" in model_name.lower(): detected_size = 200000
            elif "claude-3-sonnet" in model_name.lower(): detected_size = 200000
            elif "claude-3-haiku" in model_name.lower(): detected_size = 200000
            elif "claude-2.1" in model_name.lower(): detected_size = 200000
            elif "claude-2" in model_name.lower(): detected_size = 100000


            # Use instance config override if provided, otherwise use detected, else fallback to schema default (4096)
            schema_default_context = 4096 # Define schema default here or import from config schema if needed
            self.current_context_size = self.instance_context_size_override or detected_size or schema_default_context
            logger.info(f" -> Effective Context Size: {self.current_context_size} (Instance Override: {self.instance_context_size_override}, Detected: {detected_size}, Default: {schema_default_context})")

            self.model_name = model_name
            self._model_loaded = True
            return True # No actual loading needed for API, just setting state

    async def unload_model(self) -> bool:
        """Unsets the active model name and resets state."""
        async with self._load_lock:
            if not self._model_loaded: return True # Already unloaded
            logger.info(f"OpenAI '{self.binding_instance_name}': Unsetting active model '{self.model_name}'.")
            self.model_name = None
            self._model_loaded = False
            self.model_supports_vision = False
            # Reset context size to instance override or default
            schema_default_context = 4096
            self.current_context_size = self.instance_context_size_override or schema_default_context
            return True

    def _prepare_openai_messages(self, prompt: str, system_message: Optional[str], multimodal_data: Optional[List['InputData']]) -> List[ChatCompletionMessageParam]:
        """Constructs the message list for the OpenAI API, handling multimodal content."""
        messages: List[ChatCompletionMessageParam] = []

        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Prepare user message content (text + images if applicable)
        user_content_parts: List[Union[Dict[str, str], Dict[str, Dict[str, str]]]] = []

        # Always add the main text prompt if it exists
        if prompt:
            user_content_parts.append({"type": "text", "text": prompt})

        # Add images if model supports vision and images are provided
        if self.model_supports_vision and multimodal_data:
            image_items = [item for item in multimodal_data if item.type == 'image']
            if image_items:
                logger.info(f"Found {len(image_items)} image(s) for OpenAI vision request (instance '{self.binding_instance_name}').")
                for item in image_items:
                    if item.data and isinstance(item.data, str) and item.mime_type:
                         # Basic check if it looks like base64
                         # OpenAI API expects only the base64 part, not the data URI prefix
                         # Assuming the input `item.data` is *just* the base64 string
                         if len(item.data) > 10 and '=' not in item.data[-4:]:
                              # Construct the correct format for OpenAI API
                              image_url_content = {"url": f"data:{item.mime_type};base64,{item.data}"}
                              user_content_parts.append({"type": "image_url", "image_url": image_url_content})
                         else:
                             logger.warning(f"Skipping image item: data doesn't look like base64 (role={item.role})")
                    else:
                        logger.warning(f"Skipping image item: missing data/mime_type or data not string (role={item.role})")
            elif multimodal_data:
                logger.debug(f"Instance '{self.binding_instance_name}': Multimodal data provided but no valid images found.")
        elif multimodal_data and not self.model_supports_vision:
            logger.warning(f"OpenAI instance '{self.binding_instance_name}' received image data, but model '{self.model_name}' does not support vision. Ignoring images.")

        # Add the user message only if there's content (text or image)
        if user_content_parts:
            messages.append({"role": "user", "content": user_content_parts}) # type: ignore # Let Pydantic handle union type validation
        else:
            # This should not happen if validation is done upstream, but log just in case
            logger.error(f"No user prompt or valid image data found for OpenAI request instance '{self.binding_instance_name}'. Cannot create user message.")

        # Log the structure (without image data) for debugging
        log_messages = []
        for msg in messages:
             log_entry = {"role": msg["role"], "content": []}
             content = msg["content"]
             if isinstance(content, str):
                 log_entry["content"].append({"type": "text", "text": content[:100] + "..."})
             elif isinstance(content, list):
                 for part in content:
                     if part["type"] == "text": log_entry["content"].append({"type": "text", "text": part["text"][:100] + "..."})
                     elif part["type"] == "image_url": log_entry["content"].append({"type": "image_url", "url": part["image_url"]["url"][:50] + "..."})
             log_messages.append(log_entry)
        logger.debug(f"Prepared OpenAI messages structure: {log_messages}")

        return messages # type: ignore # Return the list adhering to ChatCompletionMessageParam structure

    async def generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]: # Return List[OutputData]-like
        """Generates text using the OpenAI API (non-streaming)."""
        if not self.client: raise RuntimeError(f"OpenAI client not initialized for instance '{self.binding_instance_name}'.")
        if not self._model_loaded or not self.model_name: raise RuntimeError(f"Model not set for OpenAI instance '{self.binding_instance_name}'. Call load_model first.")
        if not self.api_key: raise RuntimeError(f"API key not configured for OpenAI instance '{self.binding_instance_name}'.")

        logger.info(f"OpenAI '{self.binding_instance_name}': Generating non-stream with '{self.model_name}'...")

        # Extract parameters
        max_tokens = params.get("max_tokens") # Can be None, API will use model default
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)
        presence_penalty = params.get("presence_penalty", 0.0)
        frequency_penalty = params.get("frequency_penalty", 0.0)
        system_message = params.get("system_message", None)
        seed = params.get("seed") # Optional seed

        messages = self._prepare_openai_messages(prompt, system_message, multimodal_data)
        if not messages or not messages[-1].get("content"):
            raise ValueError(f"Invalid message structure for OpenAI request instance '{self.binding_instance_name}' (no user content).")

        try:
            # Construct API parameters, including optional ones
            api_params: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "presence_penalty": float(presence_penalty),
                "frequency_penalty": float(frequency_penalty),
                "stream": False
            }
            if max_tokens is not None:
                 try: api_params["max_tokens"] = int(max_tokens)
                 except ValueError: logger.warning(f"Invalid max_tokens value '{max_tokens}'. Ignoring.")
            if seed is not None:
                 try: api_params["seed"] = int(seed)
                 except ValueError: logger.warning(f"Invalid seed value '{seed}'. Ignoring.")

            stop = params.get("stop_sequences") or params.get("stop")
            if stop:
                 api_params["stop"] = stop if isinstance(stop, list) else [stop]

            logger.debug(f"OpenAI API call params (instance '{self.binding_instance_name}'): { {k:v for k,v in api_params.items() if k!='messages'} }") # Log params without messages
            response = await self.client.chat.completions.create(**api_params)

            if not response.choices:
                 raise RuntimeError("OpenAI API returned no choices.")

            raw_completion = response.choices[0].message.content or "" # Ensure string
            finish_reason = response.choices[0].finish_reason
            usage = response.usage

            # --- ADDED: Parse thoughts ---
            cleaned_completion, thoughts = parse_thought_tags(raw_completion)
            # --------------------------

            logger.info(f"OpenAI generation successful for '{self.binding_instance_name}'. Finish reason: {finish_reason}")

            output_metadata = {
                "model_used": self.model_name,
                "binding_instance": self.binding_instance_name,
                "finish_reason": finish_reason,
                "usage": usage.model_dump() if usage else None
            }
            # Return standardized list format, now including thoughts
            return [{
                "type": "text",
                "data": cleaned_completion.strip(), # Use cleaned text
                "thoughts": thoughts, # Add extracted thoughts
                "metadata": output_metadata
            }]
        except OpenAIError as e:
             logger.error(f"OpenAI API Error for '{self.binding_instance_name}': {e}")
             status = getattr(e, 'status_code', 'N/A'); err_body = getattr(e, 'body', {}); err_type = err_body.get('error',{}).get('type') if err_body else getattr(e,'type', 'unknown'); err_msg = err_body.get('error',{}).get('message') if err_body else str(e)
             logger.error(f" -> Status: {status}, Type: {err_type}, Message: {err_msg}")
             raise RuntimeError(f"OpenAI API Error ({status} {err_type}): {err_msg}") from e
        except Exception as e:
             logger.error(f"OpenAI unexpected error for '{self.binding_instance_name}': {e}", exc_info=True)
             raise RuntimeError(f"Unexpected OpenAI error: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        params: Dict[str, Any],
        request_info: Dict[str, Any],
        multimodal_data: Optional[List['InputData']] = None
    ) -> AsyncGenerator[Dict[str, Any], None]: # Yields StreamChunk-like dicts
        """Generates text using the OpenAI API (streaming)."""
        if not self.client: yield {"type": "error", "content": f"OpenAI client not initialized for instance '{self.binding_instance_name}'."}; return
        if not self._model_loaded or not self.model_name: yield {"type": "error", "content": f"Model not set for OpenAI instance '{self.binding_instance_name}'."}; return
        if not self.api_key: yield {"type": "error", "content": f"API key not configured for OpenAI instance '{self.binding_instance_name}'."}; return

        logger.info(f"OpenAI '{self.binding_instance_name}': Generating stream with '{self.model_name}'...")

        # Extract parameters
        max_tokens = params.get("max_tokens"); temperature = params.get("temperature", 0.7); top_p = params.get("top_p", 1.0)
        presence_penalty = params.get("presence_penalty", 0.0); frequency_penalty = params.get("frequency_penalty", 0.0)
        system_message = params.get("system_message", None); seed = params.get("seed")

        messages = self._prepare_openai_messages(prompt, system_message, multimodal_data)
        if not messages or not messages[-1].get("content"):
             yield {"type": "error", "content": f"Invalid message structure for OpenAI request instance '{self.binding_instance_name}'."}
             return

        full_response_text = ""; accumulated_thoughts = ""; finish_reason = None; usage_info = None
        final_metadata = {"model_used": self.model_name, "binding_instance": self.binding_instance_name}
        is_thinking = False # State variable
        try:
            # Construct API parameters
            api_params: Dict[str, Any] = { "model": self.model_name, "messages": messages, "temperature": float(temperature), "top_p": float(top_p), "presence_penalty": float(presence_penalty), "frequency_penalty": float(frequency_penalty), "stream": True }
            if max_tokens is not None:
                 try: api_params["max_tokens"] = int(max_tokens)
                 except ValueError: logger.warning(f"Invalid max_tokens value '{max_tokens}' for stream. Ignoring.")
            if seed is not None:
                 try: api_params["seed"] = int(seed)
                 except ValueError: logger.warning(f"Invalid seed value '{seed}' for stream. Ignoring.")
            stop = params.get("stop_sequences") or params.get("stop");
            if stop: api_params["stop"] = stop if isinstance(stop, list) else [stop]

            logger.debug(f"OpenAI stream API call params (instance '{self.binding_instance_name}'): { {k:v for k,v in api_params.items() if k!='messages'} }") # Log params without messages
            stream = await self.client.chat.completions.create(**api_params)

            async for chunk in stream:
                chunk_delta_content = None
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    chunk_content = delta.content # This is the text chunk

                # --- ADDED: Stream parsing logic ---
                if chunk_delta_content:
                    current_text_to_process = chunk_delta_content
                    processed_text_chunk = "" # Text to yield in this chunk
                    processed_thoughts_chunk = None # Thoughts ending in this chunk

                    while current_text_to_process:
                        if is_thinking:
                            end_tag_pos = current_text_to_process.find("</think>")
                            if end_tag_pos != -1:
                                # End of thought block found
                                thought_part = current_text_to_process[:end_tag_pos]
                                accumulated_thoughts += thought_part
                                processed_thoughts_chunk = accumulated_thoughts # Capture full thought
                                accumulated_thoughts = "" # Reset accumulator
                                is_thinking = False
                                # Process text *after* the tag in this chunk
                                current_text_to_process = current_text_to_process[end_tag_pos + len("</think>"):]
                                # logger.debug(f"Stream: Found </think>. Thoughts: '{processed_thoughts_chunk[:50]}...'. Remaining: '{current_text_to_process[:50]}...'")
                            else:
                                # Still thinking, accumulate entire chunk
                                accumulated_thoughts += current_text_to_process
                                # logger.debug(f"Stream: Accumulating thought: '{current_text_to_process[:50]}...'")
                                current_text_to_process = "" # Consumed whole chunk as thought
                        else: # Not currently thinking
                            start_tag_pos = current_text_to_process.find("<think>")
                            if start_tag_pos != -1:
                                # Start of thought block found
                                text_part = current_text_to_process[:start_tag_pos]
                                processed_text_chunk += text_part
                                is_thinking = True
                                # Process text *after* the tag in this chunk
                                current_text_to_process = current_text_to_process[start_tag_pos + len("<think>"):]
                                # logger.debug(f"Stream: Found <think>. Yielding text: '{text_part[:50]}...'. Remaining: '{current_text_to_process[:50]}...'")
                            else:
                                # No tags found, it's all regular text
                                processed_text_chunk += current_text_to_process
                                current_text_to_process = ""
                                # logger.debug(f"Stream: Yielding text chunk: '{processed_text_chunk[:50]}...'")

                    # Yield the processed text chunk if non-empty
                    if processed_text_chunk:
                        full_response_text += processed_text_chunk
                        yield {
                            "type": "chunk",
                            "content": processed_text_chunk,
                            "thoughts": processed_thoughts_chunk # Send thoughts ending in this chunk
                        }
                    # If only thoughts were processed (ending tag found), yield them if not already yielded
                    elif processed_thoughts_chunk:
                         yield {
                            "type": "chunk", # Yield as part of the chunk sequence
                            "content": None, # No visible content
                            "thoughts": processed_thoughts_chunk
                        }

                # --- End Stream parsing logic ---

                # Check for finish reason in this chunk
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Check for usage info
                chunk_usage = getattr(chunk, 'usage', None)
                if chunk_usage:
                     usage_info = chunk_usage.model_dump() if hasattr(chunk_usage,'model_dump') else chunk_usage


            # Handle case where stream ends mid-thought (optional: yield remaining thoughts)
            if is_thinking and accumulated_thoughts:
                 logger.warning(f"Stream ended mid-thought for '{self.binding_instance_name}'. Thought content:\n{accumulated_thoughts}")
                 # Optionally yield the partial thought in the final chunk or an info chunk
                 # For now, we'll just add it to the final metadata's thoughts
                 final_metadata["thoughts"] = accumulated_thoughts # Add incomplete thoughts here

            # Stream finished, prepare and yield final chunk
            final_metadata["finish_reason"] = finish_reason or ("incomplete_thought" if is_thinking else "completed")
            final_metadata["usage"] = usage_info

            # Prepare final output list with cleaned text and concatenated thoughts
            # Re-parse the full accumulated text to get all thoughts cleanly (in case of streaming issues)
            final_cleaned_text, final_thoughts_str = parse_thought_tags(full_response_text)
            # Add any incomplete thoughts captured at the end
            if is_thinking and accumulated_thoughts:
                final_thoughts_str = (final_thoughts_str + "\n\n--- Incomplete Thought Block ---\n" + accumulated_thoughts).strip() if final_thoughts_str else accumulated_thoughts


            final_output_list = [{
                "type": "text",
                "data": final_cleaned_text.strip(),
                "thoughts": final_thoughts_str, # Add concatenated thoughts
                "metadata": final_metadata
            }]
            yield {"type": "final", "content": final_output_list, "metadata": {"status": "complete"}}
            logger.info(f"OpenAI stream finished for '{self.binding_instance_name}' (Reason: {final_metadata['finish_reason']}). Usage: {usage_info}")

        except OpenAIError as e:
             logger.error(f"OpenAI API Error during stream for '{self.binding_instance_name}': {e}")
             status = getattr(e, 'status_code', 'N/A'); err_body = getattr(e, 'body', {}); err_type = err_body.get('error',{}).get('type') if err_body else getattr(e,'type', 'unknown'); err_msg = err_body.get('error',{}).get('message') if err_body else str(e)
             logger.error(f" -> Status: {status}, Type: {err_type}, Message: {err_msg}")
             yield {"type": "error", "content": f"OpenAI API Error ({status} {err_type}): {err_msg}"}
        except Exception as e:
             logger.error(f"OpenAI stream unexpected error for '{self.binding_instance_name}': {e}", exc_info=True)
             yield {"type": "error", "content": f"Unexpected OpenAI stream error: {e}"}


    async def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Tokenization is not directly supported via standard OpenAI API."""
        if not self._model_loaded: raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for tokenization")
        logger.warning(f"OpenAI binding '{self.binding_instance_name}': Tokenization not supported via standard API.")
        raise NotImplementedError(f"Binding '{self.binding_instance_name}' (OpenAI) does not support tokenization.")

    async def detokenize(self, tokens: List[int]) -> str:
        """Detokenization is not supported via standard OpenAI API."""
        if not self._model_loaded: raise RuntimeError(f"Model not loaded in instance '{self.binding_instance_name}' for detokenization")
        logger.warning(f"OpenAI binding '{self.binding_instance_name}': Detokenization not supported via standard API.")
        raise NotImplementedError(f"Binding '{self.binding_instance_name}' (OpenAI) does not support detokenization.")

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Returns information about the currently active OpenAI model."""
        if not self._model_loaded or not self.model_name:
             # Use instance override or schema default if no model is "loaded"
             schema_default_context = 4096
             ctx = self.instance_context_size_override or schema_default_context
             return { "name": None, "context_size": ctx, "max_output_tokens": None, "supports_vision": False, "supports_audio": False, "details": {} }

        # Use the context size determined during load_model
        return {
            "name": self.model_name,
            "context_size": self.current_context_size,
            "max_output_tokens": None, # Not fixed for OpenAI API models
            "supports_vision": self.model_supports_vision,
            "supports_audio": False, # Assume false
            "details": {"info": f"Active model for instance '{self.binding_instance_name}' is '{self.model_name}'. Endpoint: {self.base_url or 'Default OpenAI'}"}
        }