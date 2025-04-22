# lollms_server/api/models.py
from pydantic import BaseModel, Field, HttpUrl, model_validator, ValidationError
from typing import List, Dict, Optional, Any, Union, Literal
from datetime import datetime
import base64 # For validator example
import logging # For validator warning

logger = logging.getLogger(__name__)

# --- NEW: Input Data Model ---
class InputData(BaseModel):
    """Represents a piece of multimodal input data."""
    type: Literal['text', 'image', 'audio', 'video', 'document'] = Field(
        ..., description="The type of the input data."
    )
    role: str = Field(
        ...,
        description="The role this input plays in the request (e.g., 'user_prompt', 'input_image', 'controlnet_image', 'mask_image', 'input_audio', 'system_context'). Allows for flexible workflows."
    )
    # Data can be base64 encoded string, URL, or plain text depending on type/binding
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

        # Heuristic check for base64-like data for common binary types
        is_likely_binary = data_type in ['image', 'audio', 'video']
        # Simple check: string, reasonably long, maybe check padding?
        is_likely_base64 = isinstance(data, str) and len(data) > 100

        if is_likely_binary and is_likely_base64 and not mime_type:
            # Try to decode a small part to see if it's valid base64
            try:
                 base64.b64decode(data[:12].encode('utf-8'), validate=True)
                 # If it decodes, it's likely base64 and needs a mime_type
                 # Let's allow it for now and let the binding validate fully
                 # logger.warning(f"mime_type is recommended for likely base64 data of type '{data_type}' (role: {values.get('role', 'N/A')})")
                 pass
            except (base64.binascii.Error, ValueError):
                 pass # Not valid base64, might be URL or other string
        return values

# --- Models for List Available Models ---

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
    # --- NEW: Capability flags ---
    supports_vision: bool = Field(False, description="Indicates if the model likely supports image input.")
    supports_audio: bool = Field(False, description="Indicates if the model likely supports audio input/output.")
    # --- END Capability Flags ---
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional non-standard details provided by the binding.")


class ListAvailableModelsResponse(BaseModel):
    """Response model for the /list_available_models endpoint."""
    binding_name: str = Field(..., description="The logical name of the binding instance queried.")
    models: List[ModelInfo] = Field(..., description="A list of models available to this binding instance.")

# --- MODIFIED: Request Models ---
class GenerateRequest(BaseModel):
    """Request model for the /generate endpoint, supporting multimodal inputs."""
    # --- NEW: Multimodal Input Data ---
    input_data: List[InputData] = Field(
        default_factory=list,
        description="List of input data items (text, images, audio). Include the main prompt here with type='text' and role='user_prompt' or use text_prompt field."
    )
    # --- Optional convenience field for simple text prompts ---
    text_prompt: Optional[str] = Field(
        None,
        description="DEPRECATED (use input_data with type='text'). Convenience field for the primary text prompt. If provided, it PREPENDS to input_data as {'type':'text', 'role':'user_prompt', 'data': text_prompt}.",
        deprecated=True
    )
    # --- End Input Data Changes ---

    personality: Optional[str] = Field(None, description="Optional: Name of the personality to use. If None, uses default system prompt or none.")
    model_name: Optional[str] = Field(None, description="Specific model name to use. Overrides defaults.")
    binding_name: Optional[str] = Field(None, description="Specific binding instance name (from config.toml) to use. Overrides defaults.")
    # --- EXPANDED: Added 'tts', 'stt' etc. ---
    generation_type: Literal['ttt', 'tti', 'ttv', 'ttm', 'tts', 'stt', 'i2i', 'audio2audio'] = Field(
        'ttt', description="Primary type of generation task (text, image, video, music, speech synthesis, speech recognition, image-to-image, etc.). Determines default binding/model selection."
    )
    stream: bool = Field(False, description="Whether to stream the response (applies to TTT, potentially TTS).")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation parameters (e.g., max_tokens, temperature, system_message, image_size, controlnet_scale). Passed to the binding.")
    functions: Optional[List[str]] = Field(None, description="List of function names to make available (if supported).")
    # extra_data is deprecated
    # extra_data: Optional[Dict[str, Any]] = Field(None, description="DEPRECATED: Use input_data instead.")

    @model_validator(mode='before')
    @classmethod
    def handle_text_prompt_and_validate_input(cls, values):
        """
        1. If text_prompt is provided, convert it into an InputData item and prepend it.
        2. Ensure there is at least one input item after potential conversion.
        """
        input_data = values.get('input_data', [])
        text_prompt = values.get('text_prompt')

        if text_prompt:
            logger.warning("The 'text_prompt' field in GenerateRequest is deprecated. Please use 'input_data' with type='text' and role='user_prompt' instead.")
            # Prepend text_prompt as an InputData item
            text_input_dict = {
                'type': 'text',
                'role': 'user_prompt',
                'data': text_prompt,
                'mime_type': 'text/plain' # Assume plain text
            }
            # Ensure input_data is a list before prepending
            if not isinstance(input_data, list):
                 input_data = [] # Initialize as list if not present or wrong type
            # Insert as dict for Pydantic to process correctly later if needed
            input_data.insert(0, text_input_dict)
            values['input_data'] = input_data # Update values dict
            values['text_prompt'] = None # Clear the deprecated field internally
            del values['text_prompt'] # Remove from dict passed to model init

        # Validate that there's at least one input after processing text_prompt
        final_input_data = values.get('input_data', [])
        if not final_input_data:
            raise ValueError("Generation request must include at least one item in 'input_data' (e.g., a text prompt).")

        # Further validation: Ensure input_data is a list of dicts/InputData objects
        if not isinstance(final_input_data, list):
             raise ValueError("'input_data' must be a list.")
        # Pydantic will handle validation of individual items against InputData schema

        return values

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_data": [
                        {"type": "text", "role": "user_prompt", "data": "Tell me a short story about a robot learning to paint."}
                    ],
                    "personality": "lz_generic",
                    "stream": False,
                    "parameters": {"max_tokens": 512, "temperature": 0.7}
                },
                {
                    "input_data": [
                        {"type": "text", "role": "user_prompt", "data": "Create an image of a cat wearing a wizard hat."},
                        {"type": "image", "role": "controlnet_image", "data": "BASE64_STRING_OF_CAT_POSE...", "mime_type": "image/png"},
                    ],
                    "generation_type": "tti",
                    "binding_name": "my_stable_diffusion_binding", # Example
                    "model_name": "sdxl_controlnet_model",       # Example
                    "parameters": { "controlnet_scale": 0.8, "image_size": "1024x1024" }
                },
                {
                    "input_data": [
                        {"type": "text", "role": "user_prompt", "data": "Describe this image."},
                        {"type": "image", "role": "input_image", "data": "BASE64_STRING_OF_SCENE...", "mime_type": "image/jpeg"},
                    ],
                    "binding_name": "gemini_vision_binding", # Example
                    "model_name": "gemini-1.5-pro-latest",       # Example
                    "stream": True
                },
                 { # Example using deprecated text_prompt
                    "text_prompt": "Translate 'hello world' to French.",
                    "personality": None,
                    "binding_name": "default_ollama",
                    "model_name": "phi3:mini",
                    "parameters": { "system_message": "You are a helpful translation assistant." }
                }
            ]
        }
    }


# --- Response Models ---

class BindingInfo(BaseModel):
    """Information about a binding type."""
    type_name: str

class BindingInstanceInfo(BaseModel):
    """Information about a configured binding instance."""
    logical_name: str
    config: Dict[str, Any]

class ListBindingsResponse(BaseModel):
    binding_types: Dict[str, Dict[str, Any]] = Field(description="Dictionary of discovered binding types and their metadata.")
    binding_instances: Dict[str, Dict[str, Any]] = Field(description="Dictionary of configured binding instances and their configurations.")

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
    personalities: Dict[str, PersonalityInfo] = Field(description="Dictionary of loaded personalities.")

class ListFunctionsResponse(BaseModel):
    functions: List[str] = Field(description="List of available function names.")

class ListModelsResponse(BaseModel):
    models: Dict[str, List[str]] = Field(description="Dictionary mapping model types (ttt, tti, ttv, ttm, etc.) to lists of discovered model file names.")

class StreamChunk(BaseModel):
    """Model for a chunk in a streaming response."""
    type: Literal["chunk", "final", "error", "info", "function_call"] = Field(..., description="Type of the stream message.")
    content: Optional[Any] = Field(None, description="The content of the chunk (text, data, error message).")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata associated with the chunk or final response.")


class GenerateResponse(BaseModel):
    """Standard non-streaming response wrapper. Output structure depends on generation type."""
    personality: Optional[str] = None # Personality used
    request_id: Optional[str] = None # Add request ID tracking later
    # Output structure varies:
    # - TTT: {"text": "generated string"}
    # - TTI: {"image_base64": "...", "mime_type": "image/png", ...}
    # - TTS: {"audio_base64": "...", "mime_type": "audio/wav", ...}
    # - STT: {"text": "transcribed string"}
    # - Scripted: Can be any JSON-serializable type defined by the script.
    output: Dict[str, Any] = Field(..., description="The generated output, format depends on generation type.")
    execution_time: Optional[float] = None # Add timing later