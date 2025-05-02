# encoding:utf-8
# Project: lollms_server
# File: lollms_server/core/resource_manager.py
# Author: ParisNeo with Gemini 2.5
# Date: 2025-05-01
# Description: Manages access to limited resources like GPU locks/semaphores.

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Any, Union, Dict # Added Union for type hint

# Use ascii_colors for logging if available
try:
    import ascii_colors as logging
    from ascii_colors import ASCIIColors # For potential future colored logs here
except ImportError:
    import logging
    class ASCIIColors: pass # type: ignore

# Import ConfigGuard types for type hinting the config parameter
# Use TYPE_CHECKING to avoid runtime import dependency if ConfigGuard isn't strictly needed *inside* this module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from configguard import ConfigSection # Use ConfigSection for type hint

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages access to limited resources, primarily using asyncio locks/semaphores
    based on configuration passed during initialization.
    """
    def __init__(self, config: Union[Dict[str, Any], 'ConfigSection']):
        """
        Initializes the ResourceManager.

        Args:
            config: A dictionary or ConfigSection containing the resource manager
                    settings (gpu_strategy, gpu_limit, queue_timeout).
        """
        # Access config values using .get() for dicts or getattr() for ConfigSection
        # Providing defaults ensures stability if keys are missing.
        self.gpu_strategy = getattr(config, 'gpu_strategy', config.get('gpu_strategy', 'semaphore'))
        self.gpu_limit = int(getattr(config, 'gpu_limit', config.get('gpu_limit', 1)))
        self.queue_timeout = int(getattr(config, 'queue_timeout', config.get('queue_timeout', 120)))

        self._gpu_lock: Optional[Union[asyncio.Lock, asyncio.Semaphore]] = None # Use Union for type hint

        # Validate gpu_limit
        if self.gpu_limit < 1:
            logger.warning(f"GPU limit must be at least 1, but got {self.gpu_limit}. Setting to 1.")
            self.gpu_limit = 1
        # Validate queue_timeout
        if self.queue_timeout <= 0:
             logger.warning(f"Queue timeout must be positive, but got {self.queue_timeout}. Setting to 120.")
             self.queue_timeout = 120

        # Initialize the lock/semaphore based on the validated strategy
        if self.gpu_strategy == "simple_lock":
            self._gpu_lock = asyncio.Lock()
            logger.info("Initialized Resource Manager with simple GPU lock (limit 1)")
        elif self.gpu_strategy == "semaphore":
            self._gpu_lock = asyncio.Semaphore(self.gpu_limit)
            logger.info(f"Initialized Resource Manager with GPU semaphore (limit {self.gpu_limit})")
        elif self.gpu_strategy == "none":
             logger.info("Initialized Resource Manager with NO GPU resource limiting.")
             self._gpu_lock = None
        else:
            logger.warning(f"Unknown GPU strategy: '{self.gpu_strategy}'. Defaulting to 'semaphore' (limit {self.gpu_limit}).")
            self.gpu_strategy = "semaphore"
            self._gpu_lock = asyncio.Semaphore(self.gpu_limit)

        self._active_gpu_tasks = 0 # Counter for active tasks using the resource

    @asynccontextmanager
    async def acquire_gpu_resource(self, task_name: str = "unnamed_task"):
        """
        Asynchronous context manager to acquire and release the configured GPU resource lock/semaphore.

        Args:
            task_name (str): A descriptive name for the task acquiring the resource (for logging).

        Yields:
            None: The context manager yields control once the resource is acquired.

        Raises:
            asyncio.TimeoutError: If the resource cannot be acquired within the configured `queue_timeout`.
            RuntimeError: For other unexpected errors during acquisition/release.
        """
        if self._gpu_lock is None:
            # No locking configured, proceed immediately
            logger.debug(f"GPU resource acquisition skipped (no lock configured) for task: {task_name}")
            try:
                yield # Yield control to the caller
            finally:
                pass # No lock to release
            return

        # Attempt to acquire the lock/semaphore
        lock_acquired = False
        logger.debug(f"Task '{task_name}' attempting to acquire GPU resource (Timeout: {self.queue_timeout}s)...")
        try:
            await asyncio.wait_for(
                self._gpu_lock.acquire(),
                timeout=float(self.queue_timeout) # Ensure float for wait_for
            )
            lock_acquired = True
            self._active_gpu_tasks += 1
            logger.info(f"GPU resource acquired by task '{task_name}'. Active tasks: {self._active_gpu_tasks}")

            # Yield control to the code block within the 'async with' statement
            yield

        except asyncio.TimeoutError:
            logger.error(f"Timeout ({self.queue_timeout}s) waiting for GPU resource for task: {task_name}")
            # Re-raise the specific TimeoutError for calling code to handle
            raise asyncio.TimeoutError(f"Task '{task_name}' failed to acquire GPU resource within {self.queue_timeout}s")
        except Exception as e:
             logger.error(f"Unexpected error during GPU resource acquisition/yield for task '{task_name}': {e}", exc_info=True)
             # Re-raise as a generic RuntimeError to signal failure
             raise RuntimeError(f"Unexpected error in resource management for {task_name}: {e}") from e
        finally:
            # Release the lock/semaphore if it was acquired
            if lock_acquired and self._gpu_lock is not None:
                try:
                    self._gpu_lock.release()
                    self._active_gpu_tasks -= 1
                    logger.info(f"GPU resource released by task '{task_name}'. Active tasks: {self._active_gpu_tasks}")
                except Exception as release_err:
                     # Log error during release, but don't overshadow original errors
                     logger.error(f"Error releasing GPU resource for task '{task_name}': {release_err}", exc_info=True)

    # Placeholder for potential future more granular resource tracking (e.g., VRAM)
    def check_vram_availability(self, required_vram_mb: int) -> bool:
        """Checks if estimated VRAM is available (very basic placeholder)."""
        # This is highly platform and binding specific.
        # A real implementation would need libraries like pynvml or torch.cuda.
        logger.warning("VRAM availability check is not implemented yet. Always returning True.")
        return True

    # You could add methods to get current status if needed
    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the resource manager."""
        return {
            "gpu_strategy": self.gpu_strategy,
            "gpu_limit": self.gpu_limit if self.gpu_strategy == "semaphore" else (1 if self.gpu_strategy == "simple_lock" else "N/A"),
            "queue_timeout": self.queue_timeout,
            "active_gpu_tasks": self._active_gpu_tasks,
            "lock_type": type(self._gpu_lock).__name__ if self._gpu_lock else "None"
        }