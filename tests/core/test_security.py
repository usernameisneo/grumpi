# tests/core/test_security.py
import pytest
from fastapi import HTTPException, status
from unittest.mock import patch, MagicMock # Use patch directly

from lollms_server.core.security import verify_api_key, api_key_header
from lollms_server.core.config import AppConfig, SecurityConfig

@pytest.mark.asyncio
# Use patch as a context manager or decorator
async def test_verify_api_key_valid():
    """Test verification with a valid key."""
    valid_key = "my_secret_key"
    mock_config = AppConfig(security=SecurityConfig(allowed_api_keys=[valid_key]))

    # Patch the get_config function *within the security module*
    with patch("lollms_server.core.security.get_config", return_value=mock_config):
        # Simulate getting the key from the header (FastAPI does this)
        result = await verify_api_key(api_key=valid_key)
        assert result == valid_key

@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    """Test verification with an invalid key."""
    valid_key = "my_secret_key"
    mock_config = AppConfig(security=SecurityConfig(allowed_api_keys=[valid_key]))

    with patch("lollms_server.core.security.get_config", return_value=mock_config):
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key="wrong_key")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid or missing API Key" in exc_info.value.detail

@pytest.mark.asyncio
async def test_verify_api_key_no_keys_configured():
    """Test verification when no keys are configured on the server."""
    mock_config = AppConfig(security=SecurityConfig(allowed_api_keys=[]))

    with patch("lollms_server.core.security.get_config", return_value=mock_config):
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key="some_key")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "API key security is not configured" in exc_info.value.detail

@pytest.mark.asyncio
async def test_verify_api_key_missing_key_header():
    """Test the behavior when the API key header is missing (simulated)."""
    valid_key = "my_secret_key"
    mock_config = AppConfig(security=SecurityConfig(allowed_api_keys=[valid_key]))

    with patch("lollms_server.core.security.get_config", return_value=mock_config):
        # Simulate dependency injection passing None because header is missing
        # This tests the logic inside verify_api_key when api_key is None
        with pytest.raises(HTTPException) as exc_info:
             await verify_api_key(api_key=None) # type: ignore
        # The function raises 401 if key is None or invalid
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid or missing API Key" in exc_info.value.detail