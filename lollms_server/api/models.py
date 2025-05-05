# lollms_server/api/models.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: Pydantic models defining the structure of API requests and responses for lollms_server.
# Modification Date: 2025-05-04
# Refactored based on plan to add GetModelInfoResponse and adjust defaults handling.

from pydantic import BaseModel, Field, HttpUrl, model_validator, ValidationError, ConfigDict
from typing import List, Dict, Optional, Any, Union, Literal
from datetime import datetime
import base64
import importlib.metadata # For version in HealthResponse

# Use TYPE_CHECKING for ConfigGuard import hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from configguard import ConfigGuard
    except ImportError: ConfigGuard = Any # type: ignore

# Use ascii_colors for logging if available
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors # For potential use
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore

logger = logging.getLogger(__name__)

# --- Basic Server Info ---
class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""
    status: str = Field("ok", description="Indicates the server is running.")
    version: Optional[str] = Field(None, description="The version of the lollms-server package.")
    api_key_required: bool = Field(..., description="Whether the server requires an API key for most endpoints.")


# --- Input/Output Data Structures ---
class InputData(BaseModel):
    """Represents a piece of multimodal input data."""
    type: Literal['text', 'image', 'audio', 'video', 'document'] = Field(
        ..., description="The type of the input data."
    )
    role: str = Field(
        ...,
        description="The role this input plays (e.g., 'user_prompt', 'input_image', 'system_context')."
    )
    data: str = Field(..., description="The data itself (e.g., text content, base64 string, URL).")
    mime_type: Optional[str] = Field(
        None, description="MIME type for binary data (e.g., 'image/png', 'audio/wav'). REQUIRED for likely binary types."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata like filename, description etc."
    )

    @model_validator(mode='before')
    @classmethod
    def check_mime_type_for_binary(cls, values):
        """Ensure mime_type is provided for likely binary types."""
        data_type = values.get('type')
        mime_type = values.get('mime_type')
        data = values.get('data')
        is_likely_binary = data_type in ['image', 'audio', 'video']
        is_likely_base64 = isinstance(data, str) and len(data) > 50 # Heuristic

        if is_likely_binary and is_likely_base64 and not mime_type:
             try: base64.b64decode(data[:24].encode('utf-8'), validate=True)
             except Exception: pass # Not base64, maybe ok without mime
             else: # Looks like base64, needs mime
                 raise ValueError(f"mime_type is REQUIRED for likely base64 data of type '{data_type}' (role: {values.get('role', 'N/A')})")
        return values

class OutputData(BaseModel):
    """Represents a piece of generated output data."""
    type: Literal['text', 'image', 'audio', 'video', 'json', 'error', 'info'] = Field(
        ..., description="The type of the output data."
    )
    data: Any = Field(..., description="The data itself (e.g., text content, base64 string, dict).")
    mime_type: Optional[str] = Field(
        None, description="MIME type for binary data (e.g., 'image/png', 'audio/wav')."
    )
    thoughts: Optional[str] = Field(
        None, description="Internal thoughts or reasoning from the model, parsed from <think> tags."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata (e.g., prompt_used, model_name, usage stats)."
    )


# --- Listing Endpoints ---

# - Bindings -
class BindingTypeInfo(BaseModel):
    """Information about a discovered binding type (from its card)."""
    type_name: str = Field(..., description="Unique identifier for the binding type.")
    display_name: Optional[str] = Field(None, description="User-friendly name for the binding.")
    version: Optional[str] = Field(None, description="Version of the binding's code.")
    author: Optional[str] = Field(None, description="Author(s) of the binding.")
    description: Optional[str] = Field(None, description="Description of the binding's purpose and capabilities.")
    requirements: Optional[List[str]] = Field(default_factory=list, description="Python packages required by this binding.")
    supports_streaming: Optional[bool] = Field(None, description="Indicates if the binding natively supports streaming output.")

    model_config = ConfigDict(extra="allow") # Allow extra fields from card


class BindingInstanceInfo(BaseModel):
    """Information about a configured binding instance (from its config file)."""
    type: str = Field(..., description="The type name of the binding this instance uses (e.g., 'ollama_binding').")
    binding_instance_name: str = Field(..., description="The unique name assigned to this instance in the configuration.")
    # Includes other non-sensitive config keys defined in the binding's instance_schema.

    model_config = ConfigDict(extra="allow") # Allow extra fields from instance config


class ListBindingsResponse(BaseModel):
    """Response model for listing binding types and configured instances."""
    binding_types: Dict[str, BindingTypeInfo] = Field(
        description="Dictionary of discovered binding types, keyed by type_name."
    )
    binding_instances: Dict[str, BindingInstanceInfo] = Field(
        description="Dictionary of configured and successfully loaded binding instances, keyed by instance_name."
    )

# - Personalities -
class PersonalityInfo(BaseModel):
    """Information about a loaded personality."""
    name: str = Field(..., description="Unique name of the personality.")
    author: Optional[str] = Field(None, description="Author of the personality.")
    version: Optional[str] = Field(None, description="Version identifier for the personality.")
    description: Optional[str] = Field(None, description="Brief description of the personality.")
    category: Optional[str] = Field(None, description="Category for organizing personalities.")
    language: Optional[str] = Field(None, description="Primary language of the personality.")
    tags: List[str] = Field(default_factory=list, description="Keywords for searching or filtering.")
    icon: Optional[str] = Field(None, description="Filename of the icon within the personality's assets folder.")
    is_scripted: bool = Field(..., description="Indicates if the personality uses a Python workflow script.")
    path: str = Field(..., description="Absolute path to the personality's directory on the server.")

    @model_validator(mode='before')
    @classmethod
    def ensure_version_string(cls, values):
        """Ensures the version field is always a string."""
        if 'version' in values and values['version'] is not None:
            values['version'] = str(values['version'])
        return values


class ListPersonalitiesResponse(BaseModel):
    """Response model for listing available personalities."""
    personalities: Dict[str, PersonalityInfo] = Field(description="Dictionary of loaded and enabled personalities, keyed by name.")

# - Functions -
class ListFunctionsResponse(BaseModel):
    """Response model for listing available custom functions."""
    functions: List[str] = Field(description="List of available function names (module_stem.function_name).")

# - Models (Available to a Binding) -
class ModelInfoDetails(BaseModel):
    """Flexible model for additional, binding-specific details provided by list_available_models or get_model_info."""
    model_config = ConfigDict(extra="allow")


class ModelInfo(BaseModel):
    """Standardized structure for reporting a single available model (used by list_available_models)."""
    name: str = Field(..., description="The name/tag/ID of the model usable in requests.")
    size: Optional[int] = Field(None, description="Model size in bytes (if applicable).")
    modified_at: Optional[datetime] = Field(None, description="Timestamp when the model file/resource was last modified.")
    quantization_level: Optional[str] = Field(None, description="Quantization level (e.g., Q4_K_M, Q8_0, F16, 8bit).")
    format: Optional[str] = Field(None, description="Model format (e.g., gguf, safetensors, api).")
    family: Optional[str] = Field(None, description="Primary model family (e.g., llama, gemma, phi3, stable-diffusion).")
    families: Optional[List[str]] = Field(None, description="List of model families it belongs to.")
    parameter_size: Optional[str] = Field(None, description="Reported parameter size (e.g., 7B, 8x7B).")
    context_size: Optional[int] = Field(None, description="Reported total context window size, if known.")
    max_output_tokens: Optional[int] = Field(None, description="Reported maximum output tokens, if known.")
    template: Optional[str] = Field(None, description="Default prompt template associated, if known.")
    license: Optional[str] = Field(None, description="Model license identifier, if known.")
    homepage: Optional[str] = Field(None, description="URL to the model's homepage or source, if known.")
    supports_vision: bool = Field(False, description="Indicates if the model likely supports image input.")
    supports_audio: bool = Field(False, description="Indicates if the model likely supports audio input/output.")
    # supports_streaming: bool = Field(False, description="Indicates if the binding *for this model* supports streaming generation.") # Handled by BindingTypeInfo
    details: ModelInfoDetails = Field(default_factory=dict, description="Additional binding-specific details.")


class ListAvailableModelsResponse(BaseModel):
    """Response model for the /list_available_models/{binding_instance_name} endpoint."""
    binding_instance_name: str = Field(..., description="The logical name of the binding instance queried.")
    models: List[ModelInfo] = Field(..., description="A list of models available to this binding instance.")

# - Models (Discovered in Folder) -
class ListModelsResponse(BaseModel):
    """Response model for listing models discovered by scanning the models folder (simple file scan)."""
    models: Dict[str, List[str]] = Field(description="Dictionary of discovered model files/folders, keyed by inferred type (ttt, tti, gguf, etc.).")


# --- Generation Request/Response ---
class GenerateRequest(BaseModel):
    """Request model for the /generate endpoint, supporting multimodal inputs."""
    input_data: List[InputData] = Field(
        ...,
        min_length=1, # Ensure the list is never empty
        description="List of input data items (text, images, audio). MUST include at least one item, typically the main prompt (type='text', role='user_prompt')."
    )
    text_prompt: Optional[str] = Field( None, description="DEPRECATED. Use input_data with type='text' and role='user_prompt'.", deprecated=True )
    personality: Optional[str] = Field(None, description="Optional: Name of the loaded personality to use.")
    binding_name: Optional[str] = Field(None, description="Optional: Specific binding *instance name* (from config) to use. Overrides defaults for the generation type.")
    model_name: Optional[str] = Field(None, description="Optional: Specific model name/ID to use. If omitted, the *binding instance's* default model is used.")
    generation_type: Literal['ttt', 'tti', 'ttv', 'ttm', 'tts', 'stt', 'i2i', 'audio2audio'] = Field( 'ttt', description="Primary type of generation task." )
    stream: bool = Field(False, description="Whether to stream the response (if supported by the binding).")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation parameters passed to the binding (e.g., max_tokens, temperature, image_size). Overrides personality/defaults.")
    functions: Optional[List[str]] = Field(None, description="Reserved for future structured function calling.")

    @model_validator(mode='before')
    @classmethod
    def handle_text_prompt_and_validate_input(cls, values):
        """Validator for GenerateRequest."""
        input_data = values.get('input_data', [])
        text_prompt = values.get('text_prompt')
        # Handle deprecated text_prompt
        if text_prompt:
            logger.warning("The 'text_prompt' field is deprecated. Please use 'input_data' instead.")
            text_input_dict = { 'type': 'text', 'role': 'user_prompt', 'data': text_prompt, 'mime_type': 'text/plain' }
            if not isinstance(input_data, list): input_data = []
            input_data.insert(0, text_input_dict)
            values['input_data'] = input_data
            if 'text_prompt' in values: values.pop('text_prompt')

        final_input_data = values.get('input_data')
        if not final_input_data or not isinstance(final_input_data, list):
            raise ValueError("Request must include a non-empty 'input_data' list.")
        return values

    model_config = ConfigDict(
        json_schema_extra = {
            "examples": [
                {
                    "input_data": [{"type": "text", "role": "user_prompt", "data": "Tell a story about a robot learning to paint."}],
                    "personality": "lollms",
                    "stream": False,
                    "parameters": {"max_tokens": 512}
                },
                {
                    "input_data": [
                        {"type": "text", "role": "user_prompt", "data": "A futuristic cityscape with flying cars"},
                        {"type": "image", "role": "style_reference", "data": "BASE64_STYLE_IMG_STRING", "mime_type": "image/jpeg"}
                    ],
                    "generation_type": "tti",
                    "binding_name": "my_stable_diffusion_instance",
                    "model_name": "sdxl-base-1.0",
                    "parameters": { "width": 1024, "height": 768, "guidance_scale": 7.5 }
                }
            ]
        }
    )


class GenerateResponse(BaseModel):
    """Standard non-streaming response wrapper."""
    personality: Optional[str] = Field(None, description="Name of the personality used, if any.")
    request_id: Optional[str] = Field(None, description="Unique ID generated for this request.")
    output: List[OutputData] = Field(..., description="List of generated outputs (text, images, audio, etc.).")
    execution_time: Optional[float] = Field(None, description="Total time taken for the generation in seconds.")


class StreamChunk(BaseModel):
    """Model for a chunk in a streaming response."""
    type: Literal["chunk", "final", "error", "info", "function_call"] = Field(..., description="Type of the stream message.")
    content: Optional[Any] = Field(None, description="The content of the chunk (e.g., text snippet, audio chunk, error message). For 'final' type, this should be List[OutputData].")
    thoughts: Optional[str] = Field(None, description="Internal thoughts or reasoning relevant to this chunk, parsed from <think> tags.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata associated with the chunk or final response.")


# --- Tokenizer Endpoint Models ---
class TokenizeRequest(BaseModel):
    """Request model for tokenizing text."""
    text: str = Field(..., description="The text to tokenize.")
    binding_name: Optional[str] = Field(None, description="Logical name of the binding *instance* to use. If omitted, uses default TTT binding.")
    model_name: Optional[str] = Field(None, description="Optional: Specific model name. If omitted, uses the binding instance's default model.")
    add_bos: bool = Field(False, description="Whether to add the beginning-of-sentence token (if supported by tokenizer).")
    add_eos: bool = Field(False, description="Whether to add the end-of-sentence token (if supported by tokenizer).")

class TokenizeResponse(BaseModel):
    """Response model for tokenizing text."""
    tokens: List[int] = Field(..., description="List of token IDs.")
    count: int = Field(..., description="Number of tokens generated.")

class DetokenizeRequest(BaseModel):
    """Request model for detokenizing tokens."""
    tokens: List[int] = Field(..., description="List of token IDs to detokenize.")
    binding_name: Optional[str] = Field(None, description="Logical name of the binding *instance* to use. If omitted, uses default TTT binding.")
    model_name: Optional[str] = Field(None, description="Optional: Specific model name. If omitted, uses the binding instance's default model.")

class DetokenizeResponse(BaseModel):
    """Response model for detokenizing tokens."""
    text: str = Field(..., description="The detokenized text.")

class CountTokensRequest(BaseModel):
    """Request model for counting tokens."""
    text: str = Field(..., description="The text to count tokens for.")
    binding_name: Optional[str] = Field(None, description="Logical name of the binding *instance* to use. If omitted, uses default TTT binding.")
    model_name: Optional[str] = Field(None, description="Optional: Specific model name. If omitted, uses the binding instance's default model.")
    add_bos: bool = Field(False, description="Whether to include BOS token in count (if applicable).")
    add_eos: bool = Field(False, description="Whether to include EOS token in count (if applicable).")

class CountTokensResponse(BaseModel):
    """Response model for counting tokens."""
    count: int = Field(..., description="The number of tokens in the provided text.")

# --- Default Bindings / Model Info Endpoint Models ---

class GetModelInfoResponse(BaseModel):
    """Response model for the /get_model_info endpoint."""
    binding_instance_name: str = Field(..., description="Logical name of the binding instance providing the info.")
    model_name: str = Field(..., description="The name/identifier of the model this information pertains to.")
    model_type: Optional[Literal['ttt', 'tti', 'tts', 'stt', 'ttv', 'ttm', 'vlm', 'generic']] = Field(None, description="The primary modality type of the model.")
    context_size: Optional[int] = Field(None, description="Context window size (tokens), if applicable.")
    max_output_tokens: Optional[int] = Field(None, description="Maximum number of output tokens supported, if fixed and known.")
    supports_vision: bool = Field(False, description="Does the model support image input?")
    supports_audio: bool = Field(False, description="Does the model support audio input/output?")
    supports_streaming: bool = Field(False, description="Does the binding support streaming generation for this model?")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional binding-specific details about the model.")

    model_config = ConfigDict(extra="allow") # Allow extra fields in 'details' if binding adds them


class ListActiveBindingsResponse(BaseModel):
    """Response model for listing successfully loaded binding instances."""
    bindings: Dict[str, BindingInstanceInfo] = Field(
        description="Dictionary of currently active binding instances, keyed by instance_name."
    )

class GetDefaultBindingsResponse(BaseModel):
    """Response model for getting the current default binding instance names and parameters."""
    defaults: Dict[str, Optional[Union[str, int]]] = Field(
        description="Dictionary mapping configuration keys (e.g., 'ttt_binding', 'tti_binding', 'default_context_size') to their currently configured values."
    )

# REMOVED GetContextLengthResponse as it's superseded by GetModelInfoResponse