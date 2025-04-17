# tests/core/test_resource_manager.py
import pytest
import asyncio

from lollms_server.core.resource_manager import ResourceManager
from lollms_server.core.config import ResourceManagerConfig

# Defaults from the model definition
DEFAULT_GPU_STRATEGY = "semaphore"
DEFAULT_GPU_LIMIT = 1
DEFAULT_QUEUE_TIMEOUT = 120

@pytest.mark.asyncio
async def test_semaphore_acquire_release():
    """Test basic acquire/release with semaphore."""
    config = ResourceManagerConfig(gpu_strategy="semaphore", gpu_limit=2, queue_timeout=1)
    manager = ResourceManager(config)
    assert manager._active_gpu_tasks == 0
    async with manager.acquire_gpu_resource(task_name="task1"):
        assert manager._active_gpu_tasks == 1
        await asyncio.sleep(0.05)
        async with manager.acquire_gpu_resource(task_name="task2"):
            assert manager._active_gpu_tasks == 2
            await asyncio.sleep(0.05)
        assert manager._active_gpu_tasks == 1
    assert manager._active_gpu_tasks == 0

@pytest.mark.asyncio
async def test_semaphore_blocking_and_timeout():
    """Test that semaphore blocks when limit reached and times out."""
    limit = 1
    timeout_sec = 1 # --- FIX: Use integer timeout ---
    config = ResourceManagerConfig(
        gpu_strategy="semaphore",
        gpu_limit=limit,
        queue_timeout=timeout_sec # Use integer
    )
    manager = ResourceManager(config)
    async def hold_resource(duration):
        async with manager.acquire_gpu_resource(task_name="holder"):
            await asyncio.sleep(duration)
    holder_task = asyncio.create_task(hold_resource(timeout_sec * 3))
    await asyncio.sleep(0.01) # Allow holder task to acquire lock

    start_time = asyncio.get_event_loop().time()
    with pytest.raises(asyncio.TimeoutError) as exc_info:
        async with manager.acquire_gpu_resource(task_name="waiter"):
            pytest.fail("Should not have acquired the resource")
    end_time = asyncio.get_event_loop().time()

    # Check timeout duration - allow wider margin for scheduling variance
    assert (end_time - start_time) == pytest.approx(timeout_sec, abs=0.5)
    assert f"failed to acquire GPU resource within {timeout_sec}s" in str(exc_info.value)
    await holder_task


@pytest.mark.asyncio
async def test_simple_lock_acquire_release():
    """Test basic acquire/release with simple_lock."""
    config = ResourceManagerConfig(
        gpu_strategy="simple_lock",
        gpu_limit=DEFAULT_GPU_LIMIT,
        queue_timeout=1
    )
    manager = ResourceManager(config)
    assert manager._active_gpu_tasks == 0
    async with manager.acquire_gpu_resource(task_name="task1"):
        assert manager._active_gpu_tasks == 1
        await asyncio.sleep(0.05)
    assert manager._active_gpu_tasks == 0

@pytest.mark.asyncio
async def test_simple_lock_blocking_and_timeout():
    """Test blocking and timeout with simple_lock."""
    timeout_sec = 1 # --- FIX: Use integer timeout ---
    config = ResourceManagerConfig(
        gpu_strategy="simple_lock",
        gpu_limit=DEFAULT_GPU_LIMIT,
        queue_timeout=timeout_sec # Use integer
    )
    manager = ResourceManager(config)
    async def hold_resource(duration):
        async with manager.acquire_gpu_resource(task_name="holder"):
            await asyncio.sleep(duration)
    holder_task = asyncio.create_task(hold_resource(timeout_sec * 3))
    await asyncio.sleep(0.01) # Allow holder task to acquire lock

    start_time = asyncio.get_event_loop().time()
    with pytest.raises(asyncio.TimeoutError) as exc_info:
        async with manager.acquire_gpu_resource(task_name="waiter"):
            pytest.fail("Should not have acquired the resource")
    end_time = asyncio.get_event_loop().time()

    assert (end_time - start_time) == pytest.approx(timeout_sec, abs=0.5)
    assert f"failed to acquire GPU resource within {timeout_sec}s" in str(exc_info.value)
    await holder_task

@pytest.mark.asyncio
async def test_no_lock_strategy():
    """Test that acquire does nothing when strategy is unknown/none."""
    config = ResourceManagerConfig(
        gpu_strategy=DEFAULT_GPU_STRATEGY,
        gpu_limit=DEFAULT_GPU_LIMIT,
        queue_timeout=DEFAULT_QUEUE_TIMEOUT
    )
    manager = ResourceManager(config)
    manager._gpu_lock = None
    manager.config.gpu_strategy = "none"
    assert manager._gpu_lock is None
    acquired = False
    async with manager.acquire_gpu_resource(task_name="task_nolock"):
        acquired = True
        await asyncio.sleep(0.01)
    assert acquired
    assert manager._active_gpu_tasks == 0