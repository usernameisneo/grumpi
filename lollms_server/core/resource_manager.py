# lollms_server/core/resource_manager.py
import asyncio
import ascii_colors as logging
from .config import ResourceManagerConfig
from typing import Optional, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages access to limited resources, initially focusing on a simple GPU lock/semaphore.
    """
    def __init__(self, config: ResourceManagerConfig):
        self.config = config
        self._gpu_lock: Optional[asyncio.Lock | asyncio.Semaphore] = None

        if self.config.gpu_strategy == "simple_lock":
            self._gpu_lock = asyncio.Lock()
            logger.info("Initialized Resource Manager with simple GPU lock (limit 1)")
        elif self.config.gpu_strategy == "semaphore":
            self._gpu_lock = asyncio.Semaphore(self.config.gpu_limit)
            logger.info(f"Initialized Resource Manager with GPU semaphore (limit {self.config.gpu_limit})")
        else:
            logger.warning(f"Unknown GPU strategy: {self.config.gpu_strategy}. No GPU resource limiting enabled.")
            self._gpu_lock = None # No locking

        self._active_gpu_tasks = 0 # Counter for active tasks using the resource

    @asynccontextmanager
    async def acquire_gpu_resource(self, task_name: str = "unnamed_task"):
        """
        Asynchronous context manager to acquire and release the GPU resource.
        Raises asyncio.TimeoutError if the resource cannot be acquired within the timeout.
        """
        if self._gpu_lock is None:
            logger.debug(f"GPU resource acquisition skipped (no lock configured) for task: {task_name}")
            try:
                yield # No lock needed, just yield
            finally:
                pass # No lock to release
            return

        logger.debug(f"Task '{task_name}' attempting to acquire GPU resource...")
        try:
            # Wait for the lock/semaphore with a timeout
            await asyncio.wait_for(
                self._gpu_lock.acquire(),
                timeout=self.config.queue_timeout
            )
            self._active_gpu_tasks += 1
            logger.info(f"GPU resource acquired by task '{task_name}'. Active tasks: {self._active_gpu_tasks}")
            try:
                yield # Resource acquired, proceed with the task
            finally:
                self._gpu_lock.release()
                self._active_gpu_tasks -= 1
                logger.info(f"GPU resource released by task '{task_name}'. Active tasks: {self._active_gpu_tasks}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout ({self.config.queue_timeout}s) waiting for GPU resource for task: {task_name}")
            raise asyncio.TimeoutError(f"Task '{task_name}' failed to acquire GPU resource within {self.config.queue_timeout}s")
        except Exception as e:
            logger.error(f"Error during GPU resource management for task '{task_name}': {e}", exc_info=True)
            # Ensure lock is released if acquired before an unexpected error in the yield block
            if self._gpu_lock.locked() and hasattr(self._gpu_lock, '_get_value') and self._gpu_lock._get_value() < self.config.gpu_limit: # Check semaphore logic better if possible
                    self._gpu_lock.release() # Attempt release if needed
                    self._active_gpu_tasks -= 1
            raise # Re-raise the exception


    # Placeholder for potential future more granular resource tracking (e.g., VRAM)
    def check_vram_availability(self, required_vram_mb: int) -> bool:
        """Checks if estimated VRAM is available (very basic placeholder)."""
        # This is highly platform and binding specific.
        # A real implementation would need libraries like pynvml or torch.cuda.
        logger.warning("VRAM checking is not implemented yet. Returning True.")
        return True
