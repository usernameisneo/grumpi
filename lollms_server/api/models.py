# lollms_server/api/models.py
from pydantic import BaseModel, Field, HttpUrl, model_validator, ValidationError, ConfigDict
from typing import List, Dict, Optional, Any, Union, Literal
from datetime import datetime
import base64 # For validator example
import importlib.metadata # For version in HealthResponse

# Use TYPE_CHECKING for ConfigGuard import hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from configguard import ConfigGuard
    except ImportError: ConfigGuard = Any

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
        None, description="MIME type for binary data (e.g., 'image/png', 'audio/wav'). REQUIRED for binary types."
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
             # Allow if not strictly base64 (e.g., short placeholders)
             try: base64.b64decode(data[:24].encode('utf-8'), validate=True) # Check start only
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
        None, description="Internal thoughts or reasoning from the model, excluded from the main output."
    )    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata (e.g., prompt_used, model_name)."
    )


# --- Listing Endpoints ---

# - Bindings -
class BindingTypeInfo(BaseModel):
    """Information about a discovered binding type (from its card)."""
    type_name: str
    display_name: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[List[str]] = Field(default_factory=list)
    supports_streaming: Optional[bool] = None
    # Exclude instance_schema and package_path

    # Allow extra fields from the card for flexibility
    model_config = ConfigDict(extra="allow")


class BindingInstanceInfo(BaseModel):
    """Information about a configured binding instance (from its config file)."""
    # Include type and instance name for clarity
    type: str = Field(..., description="The type name of the binding this instance uses.")
    binding_instance_name: str = Field(..., description="The unique name assigned to this instance.")
    # Include other non-sensitive config keys
    # Allow extra fields from the instance config
    model_config = ConfigDict(extra="allow")


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
    name: str
    author: Optional[str] = None
    version: Optional[str] = None # Ensure string
    description: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    icon: Optional[str] = None
    is_scripted: bool
    path: str # Include path for reference

    @model_validator(mode='before')
    @classmethod
    def ensure_version_string(cls, values):
        if 'version' in values and values['version'] is not None:
            values['version'] = str(values['version'])
        return values


class ListPersonalitiesResponse(BaseModel):
    personalities: Dict[str, PersonalityInfo] = Field(description="Dictionary of loaded and enabled personalities, keyed by name.")

# - Functions -
class ListFunctionsResponse(BaseModel):
    functions: List[str] = Field(description="List of available function names (module_stem.function_name).")

# - Models (Available to a Binding) -
class ModelInfoDetails(BaseModel):
    """Flexible model for additional details provided by list_available_models."""
    model_config = ConfigDict(extra="allow")


class ModelInfo(BaseModel):
    """Standardized structure for reporting a single available model."""
    name: str = Field(..., description="The name/tag/ID of the model usable in requests.")
    size: Optional[int] = Field(None, description="Model size in bytes (if available).")
    modified_at: Optional[datetime] = Field(None, description="Timestamp when the model was last modified (if available).")
    quantization_level: Optional[str] = Field(None, description="Quantization level (e.g., Q4_K_M, Q8_0, F16).")
    format: Optional[str] = Field(None, description="Model format (e.g., gguf, safetensors, api).")
    family: Optional[str] = Field(None, description="Primary model family (e.g., llama, gemma, phi3).")
    families: Optional[List[str]] = Field(None, description="List of model families it belongs to.")
    parameter_size: Optional[str] = Field(None, description="Reported parameter size (e.g., 7B, 8x7B).")
    context_size: Optional[int] = Field(None, description="Reported total context window size, if known.")
    max_output_tokens: Optional[int] = Field(None, description="Reported maximum output tokens, if known.")
    template: Optional[str] = Field(None, description="Default prompt template associated, if known.")
    license: Optional[str] = Field(None, description="Model license identifier, if known.")
    homepage: Optional[str] = Field(None, description="URL to the model's homepage or source, if known.") # Changed to str for flexibility
    supports_vision: bool = Field(False, description="Indicates if the model likely supports image input.")
    supports_audio: bool = Field(False, description="Indicates if the model likely supports audio input/output.")
    details: ModelInfoDetails = Field(default_factory=dict, description="Additional non-standard details provided by the binding.")

    # Add validator if needed, e.g., for homepage URL format if HttpUrl is desired


class ListAvailableModelsResponse(BaseModel):
    """Response model for the /list_available_models/{binding_instance_name} endpoint."""
    binding_instance_name: str = Field(..., description="The logical name of the binding instance queried.")
    models: List[ModelInfo] = Field(..., description="A list of models available to this binding instance.")

# - Models (Discovered in Folder) -
class ListModelsResponse(BaseModel):
    """Response model for listing models discovered by scanning the models folder."""
    models: Dict[str, List[str]] = Field(description="Dictionary of discovered model files/folders, keyed by type (ttt, tti, gguf, etc.).")


# --- Generation Request/Response ---
class GenerateRequest(BaseModel):
    """Request model for the /generate endpoint, supporting multimodal inputs."""
    input_data: List[InputData] = Field(
        ..., # Make input_data required
        description="List of input data items (text, images, audio). Include the main prompt here with type='text' and role='user_prompt'."
    )
    text_prompt: Optional[str] = Field( None, description="DEPRECATED. Use input_data with type='text' and role='user_prompt'.", deprecated=True )
    personality: Optional[str] = Field(None, description="Optional: Name of the personality to use.")
    # Changed binding_name help text
    binding_name: Optional[str] = Field(None, description="Optional: Specific binding *instance name* (from config) to use. Overrides defaults.")
    model_name: Optional[str] = Field(None, description="Optional: Specific model name/ID to use with the selected binding. Overrides defaults.")
    generation_type: Literal['ttt', 'tti', 'ttv', 'ttm', 'tts', 'stt', 'i2i', 'audio2audio'] = Field( 'ttt', description="Primary type of generation task." )
    stream: bool = Field(False, description="Whether to stream the response.")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation parameters passed to the binding.")
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
            # Ensure input_data is a list before appending/inserting
            if not isinstance(input_data, list): input_data = []
            input_data.insert(0, text_input_dict) # Prepend the text prompt
            values['input_data'] = input_data
            # Remove text_prompt after processing
            if 'text_prompt' in values: values.pop('text_prompt')

        # Validate final input_data
        final_input_data = values.get('input_data') # Get potentially modified list
        if not final_input_data or not isinstance(final_input_data, list):
            raise ValueError("Request must include a non-empty 'input_data' list.")
        # Further validation could check if at least one 'user_prompt' exists if needed

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
                    "binding_name": "my_stable_diffusion_instance", # Use instance name
                    "model_name": "sdxl-base-1.0", # Optional model for the instance
                    "parameters": { "width": 1024, "height": 768, "guidance_scale": 7.5 }
                }
            ]
        }
    )


class GenerateResponse(BaseModel):
    """Standard non-streaming response wrapper."""
    personality: Optional[str] = Field(None, description="Name of the personality used, if any.")
    request_id: Optional[str] = Field(None, description="Unique ID generated for this request.")
    output: List[OutputData] = Field(..., description="List of generated outputs (text, images, etc.).")
    execution_time: Optional[float] = Field(None, description="Total time taken for the generation in seconds.")


class StreamChunk(BaseModel):
    """Model for a chunk in a streaming response."""
    type: Literal["chunk", "final", "error", "info", "function_call"] = Field(..., description="Type of the stream message.")
    content: Optional[Any] = Field(None, description="The content of the chunk (text, audio chunk, error message, etc.). For 'final' type, this should be List[OutputData].")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata associated with the chunk or final response.")


# --- Tokenizer Endpoint Models ---
class TokenizeRequest(BaseModel):
    text: str = Field(..., description="The text to tokenize.")
    binding_name: str = Field(..., description="Logical name of the binding *instance* to use.") # Clarified instance
    add_bos: bool = Field(True, description="Whether to add the beginning-of-sentence token.")
    add_eos: bool = Field(False, description="Whether to add the end-of-sentence token.")

class TokenizeResponse(BaseModel):
    tokens: List[int] = Field(..., description="List of token IDs.")
    count: int = Field(..., description="Number of tokens generated.")

class DetokenizeRequest(BaseModel):
    tokens: List[int] = Field(..., description="List of token IDs to detokenize.")
    binding_name: str = Field(..., description="Logical name of the binding *instance* to use.") # Clarified instance

class DetokenizeResponse(BaseModel):
    text: str = Field(..., description="The detokenized text.")

class CountTokensRequest(BaseModel):
    text: str = Field(..., description="The text to count tokens for.")
    binding_name: str = Field(..., description="Logical name of the binding *instance* to use.") # Clarified instance

class CountTokensResponse(BaseModel):
    count: int = Field(..., description="The number of tokens in the provided text.")

# --- Model Info Endpoint Response Model ---
class GetModelInfoResponse(BaseModel):
    """Response model for the /get_model_info/{binding_instance_name} endpoint."""
    binding_instance_name: str = Field(..., description="Logical name of the binding instance queried.") # Renamed field
    model_name: Optional[str] = Field(None, description="Name of the currently loaded model, if any.")
    context_size: Optional[int] = Field(None, description="Context window size of the loaded model.")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens for the loaded model.")
    supports_vision: bool = Field(False, description="Does the loaded model support vision?")
    supports_audio: bool = Field(False, description="Does the loaded model support audio?")
    details: Dict[str, Any] = Field(default_factory=dict, description="Other details about the loaded model provided by the binding.")