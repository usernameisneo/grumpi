# lollms_server/api/models.py
from pydantic import BaseModel, Field, HttpUrl, model_validator, ValidationError
from typing import List, Dict, Optional, Any, Union, Literal
from datetime import datetime
import base64 # For validator example
import ascii_colors as logging # For validator warning
import importlib.metadata

logger = logging.getLogger(__name__)

class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""
    status: str = Field("ok", description="Indicates the server is running.")
    version: Optional[str] = Field(None, description="The version of the lollms-server.")
    api_key_required: bool = Field(..., description="Whether the server requires an API key for most endpoints.")


# --- Input Data Model ---
class InputData(BaseModel):
    """Represents a piece of multimodal input data."""
    type: Literal['text', 'image', 'audio', 'video', 'document'] = Field(
        ..., description="The type of the input data."
    )
    role: str = Field(
        ...,
        description="The role this input plays in the request (e.g., 'user_prompt', 'input_image', 'controlnet_image', 'mask_image', 'input_audio', 'system_context'). Allows for flexible workflows."
    )
    data: str = Field(..., description="The data itself (e.g., base64 string, text content, URL).")
    mime_type: Optional[str] = Field(
        None, description="MIME type of the data, important for interpretation (e.g., 'image/png', 'image/jpeg', 'audio/wav', 'text/plain')."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata like filename, description, coordinates etc."
    )

    @model_validator(mode='before')
    @classmethod
    def check_mime_type_for_binary(cls, values):
        """Ensure mime_type is provided for binary-like types if data looks like base64."""
        data_type = values.get('type')
        mime_type = values.get('mime_type')
        data = values.get('data')
        is_likely_binary = data_type in ['image', 'audio', 'video']
        is_likely_base64 = isinstance(data, str) and len(data) > 100

        if is_likely_binary and is_likely_base64 and not mime_type:
            try:
                 base64.b64decode(data[:12].encode('utf-8'), validate=True)
                 # logger.warning(f"mime_type is recommended for likely base64 data of type '{data_type}' (role: {values.get('role', 'N/A')})")
                 pass
            except (base64.binascii.Error, ValueError):
                 pass
        return values

# --- List Available Models ---

class ModelInfoDetails(BaseModel):
    """Flexible model for details provided by list_available_models."""
    model_config = {
        "extra": "allow"
    }

class ModelInfo(BaseModel):
    """Structure for reporting a single available model with standardized details."""
    name: str = Field(..., description="The name/tag of the model usable in requests.")
    size: Optional[int] = Field(None, description="Model size in bytes.")
    modified_at: Optional[datetime] = Field(None, description="Timestamp when the model was last modified.")
    quantization_level: Optional[str] = Field(None, description="Quantization level (e.g., Q4_K_M, Q8_0, F16).")
    format: Optional[str] = Field(None, description="Model format (e.g., gguf, safetensors).")
    family: Optional[str] = Field(None, description="Primary model family (e.g., llama, gemma, phi3).")
    families: Optional[List[str]] = Field(None, description="List of model families it belongs to.")
    parameter_size: Optional[str] = Field(None, description="Reported parameter size (e.g., 7B, 8x7B).")
    context_size: Optional[int] = Field(None, description="Reported total context window size (input + output tokens), if known.")
    max_output_tokens: Optional[int] = Field(None, description="Reported maximum number of tokens the model can generate in a single request, if known.")
    template: Optional[str] = Field(None, description="Default prompt template associated with the model, if known.")
    license: Optional[str] = Field(None, description="Model license identifier, if known.")
    homepage: Optional[HttpUrl] = Field(None, description="URL to the model's homepage or source, if known.")
    supports_vision: bool = Field(False, description="Indicates if the model likely supports image input.")
    supports_audio: bool = Field(False, description="Indicates if the model likely supports audio input/output.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional non-standard details provided by the binding.")


class ListAvailableModelsResponse(BaseModel):
    """Response model for the /list_available_models endpoint."""
    binding_name: str = Field(..., description="The logical name of the binding instance queried.")
    models: List[ModelInfo] = Field(..., description="A list of models available to this binding instance.")


# --- Generate Request/Response Models ---
class GenerateRequest(BaseModel):
    """Request model for the /generate endpoint, supporting multimodal inputs."""
    input_data: List[InputData] = Field(
        default_factory=list,
        description="List of input data items (text, images, audio). Include the main prompt here with type='text' and role='user_prompt'."
    )
    text_prompt: Optional[str] = Field( None, description="DEPRECATED. Use input_data with type='text' and role='user_prompt'.", deprecated=True )
    personality: Optional[str] = Field(None, description="Optional: Name of the personality to use.")
    model_name: Optional[str] = Field(None, description="Specific model name to use. Overrides defaults.")
    binding_name: Optional[str] = Field(None, description="Specific binding instance name to use. Overrides defaults.")
    generation_type: Literal['ttt', 'tti', 'ttv', 'ttm', 'tts', 'stt', 'i2i', 'audio2audio'] = Field( 'ttt', description="Primary type of generation task." )
    stream: bool = Field(False, description="Whether to stream the response.")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation parameters passed to the binding.")
    functions: Optional[List[str]] = Field(None, description="List of function names (future use).")

    @model_validator(mode='before')
    @classmethod
    def handle_text_prompt_and_validate_input(cls, values):
        """Validator for GenerateRequest."""
        input_data = values.get('input_data', [])
        text_prompt = values.get('text_prompt')
        if text_prompt:
            logger.warning("The 'text_prompt' field is deprecated. Use 'input_data' instead.")
            text_input_dict = { 'type': 'text', 'role': 'user_prompt', 'data': text_prompt, 'mime_type': 'text/plain' }
            if not isinstance(input_data, list): input_data = []
            input_data.insert(0, text_input_dict)
            values['input_data'] = input_data
            values.pop('text_prompt', None)
        final_input_data = values.get('input_data', [])
        if not final_input_data: raise ValueError("Request must include at least one item in 'input_data'.")
        if not isinstance(final_input_data, list): raise ValueError("'input_data' must be a list.")
        return values

    model_config = { "json_schema_extra": { "examples": [ {"input_data": [{"type": "text", "role": "user_prompt", "data": "Story about a robot"}], "personality": "lollms", "stream": False, "parameters": {"max_tokens": 512}}, {"input_data": [{"type": "text", "role": "user_prompt", "data": "Cat wizard hat"}, {"type": "image", "role": "controlnet_image", "data": "BASE64...", "mime_type": "image/png"}], "generation_type": "tti", "binding_name": "my_sd_binding", "parameters": { "controlnet_scale": 0.8 }} ] } }

class StreamChunk(BaseModel):
    """Model for a chunk in a streaming response."""
    type: Literal["chunk", "final", "error", "info", "function_call"] = Field(..., description="Type of the stream message.")
    content: Optional[Any] = Field(None, description="The content of the chunk (text, audio chunk, error message, etc.). For 'final' type, this will be List[OutputData].") # Modified description
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata associated with the chunk or final response.")

class OutputData(BaseModel):
    """Represents a piece of generated output data."""
    type: Literal['text', 'image', 'audio', 'video', 'json', 'error', 'info'] = Field(
        ..., description="The type of the output data."
    )
    data: Any = Field(..., description="The data itself (e.g., text content, base64 string, dict).")
    mime_type: Optional[str] = Field(
        None, description="MIME type for binary data (e.g., 'image/png', 'audio/wav')."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata (e.g., prompt_used, model_name)."
    )

class GenerateResponse(BaseModel):
    """Standard non-streaming response wrapper."""
    personality: Optional[str] = None
    request_id: Optional[str] = None
    output: List[OutputData] = Field(..., description="List of generated outputs (text, images, etc.).")
    execution_time: Optional[float] = None


# --- Listing Endpoint Models ---
class BindingInfo(BaseModel):
    """Information about a binding type."""
    type_name: str

class BindingInstanceInfo(BaseModel):
    """Information about a configured binding instance."""
    logical_name: str
    config: Dict[str, Any]

class ListBindingsResponse(BaseModel):
    binding_types: Dict[str, Dict[str, Any]] = Field(description="Discovered binding types and metadata.")
    binding_instances: Dict[str, Dict[str, Any]] = Field(description="Configured binding instances.")

class PersonalityInfo(BaseModel):
    """Basic information about a personality."""
    name: str
    author: str
    version: Union[str, float, int]
    description: str
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_scripted: bool
    path: str
    icon: Optional[str] = None
    language: Optional[str] = None

class ListPersonalitiesResponse(BaseModel):
    personalities: Dict[str, PersonalityInfo] = Field(description="Loaded personalities.")

class ListFunctionsResponse(BaseModel):
    functions: List[str] = Field(description="Available function names.")

class ListModelsResponse(BaseModel):
    models: Dict[str, List[str]] = Field(description="Discovered model files by type.")


# --- NEW: Tokenizer Endpoint Models ---
class TokenizeRequest(BaseModel):
    text: str = Field(..., description="The text to tokenize.")
    binding_name: str = Field(..., description="Logical name of the binding instance to use.")
    add_bos: bool = Field(True, description="Whether to add the beginning-of-sentence token.")
    add_eos: bool = Field(False, description="Whether to add the end-of-sentence token.")

class TokenizeResponse(BaseModel):
    tokens: List[int] = Field(..., description="List of token IDs.")
    count: int = Field(..., description="Number of tokens generated.")

class DetokenizeRequest(BaseModel):
    tokens: List[int] = Field(..., description="List of token IDs to detokenize.")
    binding_name: str = Field(..., description="Logical name of the binding instance to use.")

class DetokenizeResponse(BaseModel):
    text: str = Field(..., description="The detokenized text.")

class CountTokensRequest(BaseModel): # Reuses TokenizeRequest fields implicitly via body
    text: str = Field(..., description="The text to count tokens for.")
    binding_name: str = Field(..., description="Logical name of the binding instance to use.")

class CountTokensResponse(BaseModel):
    count: int = Field(..., description="The number of tokens in the provided text.")

# --- NEW: Model Info Endpoint Response Model ---
class GetModelInfoResponse(BaseModel):
    """Response model for the /get_model_info endpoint. Reuses ModelInfo fields."""
    binding_name: str
    model_name: Optional[str] = Field(None, description="Name of the currently loaded model, if any.")
    context_size: Optional[int] = Field(None, description="Context window size of the loaded model.")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens for the loaded model.")
    supports_vision: bool = Field(False, description="Does the loaded model support vision?")
    supports_audio: bool = Field(False, description="Does the loaded model support audio?")
    details: Dict[str, Any] = Field(default_factory=dict, description="Other details about the loaded model.")
