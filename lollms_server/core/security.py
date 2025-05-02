# lollms_server/core/security.py
# -*- coding: utf-8 -*-
# Project: lollms_server
# Author: ParisNeo
# Creation Date: 2025-05-01
# Description: Handles API key security verification.

from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from typing import Optional, Any

# Use TYPE_CHECKING for ConfigGuard import hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from configguard import ConfigGuard
    except ImportError: ConfigGuard = Any # type: ignore

# Use ascii_colors for logging if available
try:
    import ascii_colors as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
# Make auto_error=False so the dependency runs even if the header is missing.
# We handle the logic of requiring it based on config inside the function.
api_key_header_optional = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --- Dependency to get config within the request context ---
async def get_config_dependency(request: Request) -> 'ConfigGuard':
    """Dependency function to retrieve the ConfigGuard object from app state."""
    config = getattr(request.app.state, 'config', None)
    if config is None:
        # This should ideally not happen if lifespan management is correct
        logger.critical("Server configuration (ConfigGuard object) not found in app state during security check!")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server configuration not ready.",
        )
    # Type check for safety, although lifespan should guarantee it
    if TYPE_CHECKING:
        from configguard import ConfigGuard
        if not isinstance(config, ConfigGuard):
            logger.critical("Object in app.state.config is not a ConfigGuard instance!")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server configuration error.",
            )
    return config # type: ignore # Ignore type error outside TYPE_CHECKING

# --- API Key Verification Dependency ---
async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header_optional), # Key from header (optional at this stage)
    config: 'ConfigGuard' = Security(get_config_dependency)     # Config object via dependency
):
    """
    Verifies the provided API key *if* API key security is enabled in the server config.
    Allows requests if security is not configured (empty allowed_api_keys list).

    Raises:
        HTTPException 403: If keys are configured but none provided.
        HTTPException 401: If keys are configured and an invalid key is provided.
    """
    allowed_keys: Optional[List[str]] = None
    try:
        # Access nested security settings safely using getattr
        security_section = getattr(config, 'security', None)
        if security_section:
            allowed_keys = getattr(security_section, "allowed_api_keys", [])
        else:
            # Should not happen with validated config, but handle defensively
            logger.error("Security configuration section not found in loaded config.")
            allowed_keys = [] # Treat as insecure if section missing

    except AttributeError as e:
        logger.error(f"Error accessing security settings in config: {e}", exc_info=True)
        # If accessing config fails unexpectedly, default to secure behavior (require key if logic proceeds)
        allowed_keys = None # Indicate config access error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error checking security configuration."
        ) from e

    # If allowed_keys is explicitly empty list, security is OFF
    if isinstance(allowed_keys, list) and not allowed_keys:
        logger.debug("API key security is not configured (allowed_api_keys is empty). Allowing request.")
        return None # Indicate security is bypassed

    # If keys ARE configured, then the client MUST provide a valid one
    if api_key is None:
        logger.warning("API key required by server configuration, but none provided by client.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, # Use 403 Forbidden when key is required but missing
            detail="API Key required but not provided.",
        )

    if not isinstance(allowed_keys, list) or api_key not in allowed_keys:
        # Log invalid key attempt (obscure the key)
        obscured_key = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "(empty or short key)"
        logger.warning(f"Invalid API Key received: '{obscured_key}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # Use 401 Unauthorized for invalid credentials
            detail="Invalid API Key provided.",
        )

    # If keys are configured and the provided key is valid
    logger.debug("Valid API Key received.")
    # Return the valid key or potentially map it to a user/role object later if needed
    return api_key