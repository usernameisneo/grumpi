# lollms_server/api/models.py
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Union, Literal
from datetime import datetime
# --- Models for List Available Models ---

class ModelInfoDetails(BaseModel):
    """Flexible model for details provided by list_available_models."""
    # Let's allow any details for maximum flexibility between bindings
    model_config = {
        "extra": "allow"
    }
    # You could add common optional fields if known, e.g.:
    # size: Optional[int] = None
    # modified_at: Optional[str] = None
    # id: Optional[str] = None

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
    # --- NEW FIELDS ---
    context_size: Optional[int] = Field(None, description="Reported total context window size (input + output tokens), if known.")
    max_output_tokens: Optional[int] = Field(None, description="Reported maximum number of tokens the model can generate in a single request, if known.")
    # --- END NEW FIELDS ---
    template: Optional[str] = Field(None, description="Default prompt template associated with the model, if known.")
    license: Optional[str] = Field(None, description="Model license identifier, if known.")
    homepage: Optional[HttpUrl] = Field(None, description="URL to the model's homepage or source, if known.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional non-standard details provided by the binding.")

# ListAvailableModelsResponse remains the same, it just uses the enhanced ModelInfo
class ListAvailableModelsResponse(BaseModel):
    """Response model for the /list_available_models endpoint."""
    binding_name: str = Field(..., description="The logical name of the binding instance queried.")
    models: List[ModelInfo] = Field(..., description="A list of models available to this binding instance.")

# --- Request Models ---
class GenerateRequest(BaseModel):
    """Request model for the /generate endpoint."""
    # --- Make personality optional ---
    personality: Optional[str] = Field(None, description="Optional: Name of the personality to use. If None, uses default system prompt or none.")
    # --- End change ---
    prompt: str = Field(..., description="The input prompt for generation.")
    model_name: Optional[str] = Field(None, description="Specific model name to use. Overrides defaults.")
    binding_name: Optional[str] = Field(None, description="Specific binding instance name (from config.toml) to use. Overrides defaults.")
    generation_type: Literal['ttt', 'tti', 'ttv', 'ttm'] = Field('ttt', description="Type of generation (text, image, video, music).")
    stream: bool = Field(False, description="Whether to stream the response (applies to TTT only).")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation parameters (e.g., max_tokens, temperature, system_message). Passed to the binding.")
    functions: Optional[List[str]] = Field(None, description="List of function names to make available (if supported).")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="Optional: Arbitrary extra data to be included in the context/system prompt (e.g., RAG results). Structure determined by personality/script.")
    # ... (model_config with examples remains the same, maybe add an example with personality=None) ...
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "personality": "lz_generic",
                    "prompt": "Tell me a short story about a robot learning to paint.",
                    "stream": False,
                    "parameters": {"max_tokens": 512, "temperature": 0.7}
                },
                {
                    "prompt": "Translate 'hello world' to French.",
                    "personality": None, # Example without personality
                    "binding_name": "default_ollama", # Optional: specify if default isn't desired
                    "model_name": "phi3:mini",       # Optional: specify if default isn't desired
                    "parameters": {
                        "system_message": "You are a helpful translation assistant." # Optional: custom system message
                    }
                }
            ]
        }
    }

# --- Response Models ---

class BindingInfo(BaseModel):
    """Information about a binding type."""
    type_name: str
    # Add more metadata from Binding.get_binding_config() if needed
    # e.g. version: str, requirements: List[str]

class BindingInstanceInfo(BaseModel):
    """Information about a configured binding instance."""
    logical_name: str
    config: Dict[str, Any]
    # Add health status or other dynamic info if needed

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

class ListPersonalitiesResponse(BaseModel):
    personalities: Dict[str, PersonalityInfo] = Field(description="Dictionary of loaded personalities.")

class ListFunctionsResponse(BaseModel):
    functions: List[str] = Field(description="List of available function names.")

class ListModelsResponse(BaseModel):
    models: Dict[str, List[str]] = Field(description="Dictionary mapping model types (ttt, tti, ttv, ttm) to lists of discovered model file names.")
    # Note: This currently just lists files. More sophisticated discovery might be needed.

class StreamChunk(BaseModel):
    """Model for a chunk in a streaming response."""
    type: Literal["chunk", "final", "error", "info", "function_call"] = Field(..., description="Type of the stream message.")
    content: Optional[Any] = Field(None, description="The content of the chunk (text, data, error message).")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata associated with the chunk or final response.")


class GenerateResponse(BaseModel):
    """Standard non-streaming response wrapper (can be used for non-TTT or non-streaming TTT)."""
    personality: str
    request_id: Optional[str] = None # Add request ID tracking later
    output: Any = Field(..., description="The generated output (string for TTT, dict with base64 for others).")
    execution_time: Optional[float] = None # Add timing later

# Note: For successful non-streaming TTT, the API might return a raw string directly (content type text/plain).
# For successful non-TTT (image, video, music), the API will return a JSON dict like {"image_base64": "..."}.
# For streaming TTT, it uses Server-Sent Events (text/event-stream) yielding JSON encoded StreamChunk objects.
# Errors should ideally use standard HTTP status codes and FastAPI's HTTPException detail format.