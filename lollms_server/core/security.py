# lollms_server/core/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from .config import get_config

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verifies the provided API key against the list in the configuration.
    """
    config = get_config()
    if not config.security.allowed_api_keys:
        # If no keys are configured, maybe allow access or raise specific error?
        # For now, let's deny access if the list is empty but verification is requested.
        # Alternatively, remove the dependency from endpoints if auth is optional.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key security is not configured on the server.",
        )

    if api_key not in config.security.allowed_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key # Return the key itself or a user identifier if needed later