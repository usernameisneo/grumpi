# lollms_server/utils/dependency_manager.py
import ascii_colors as logging
import asyncio
import subprocess # Needed for sync version
import sys
from typing import List

try:
    # Keep the async setup for potential future use or for bindings
    import pipmaster as pm_async
    pipmaster_installed = True

    # Create a sync instance for sync checks
    from pipmaster import PackageManager
    pm_sync = PackageManager()

except ImportError:
    # Mock pipmaster functions if not installed
    class MockPipMaster:
        def install_or_update(self, package, **kwargs):
            logger.warning(f"'pipmaster' not installed. Cannot manage dependency: {package}")
            return False, f"pipmaster not installed, cannot install {package}"

    # Assign mock to both sync and async versions
    pm_async = MockPipMaster()
    pm_sync = MockPipMaster()
    pipmaster_installed = False

logger = logging.getLogger(__name__)

# --- Async Version (for potential use by async loading) ---
async def ensure_dependencies(requirements: List[str], source_name: str = "component"):
    """
    Checks and installs dependencies using pipmaster (async version).
    Runs pipmaster operations in a thread pool executor.
    """
    if not requirements:
        logger.debug(f"No dependencies specified for {source_name}.")
        return True

    if not pipmaster_installed:
        logger.warning(f"'pipmaster' is not installed. Cannot automatically manage dependencies for {source_name}: {requirements}. Please install them manually or install pipmaster.")
        return False

    logger.info(f"Checking/installing dependencies for {source_name}: {requirements}")
    all_successful = True
    loop = asyncio.get_running_loop()

    for req in requirements:
        if not req or not isinstance(req, str):
            logger.warning(f"Skipping invalid requirement entry: {req} for {source_name}")
            continue
        try:
            logger.info(f"[Async Check] Ensuring dependency: '{req}' for {source_name}...")
            success, output = await loop.run_in_executor(
                None,
                lambda r=req: pm_async.install_or_update(r) # Use async instance here
            )
            if success:
                logger.info(f"Successfully ensured dependency '{req}' for {source_name}.")
            else:
                logger.error(f"Failed to install/update dependency '{req}' for {source_name}. Output:\n{output}")
                all_successful = False
        except Exception as e:
            logger.error(f"Unexpected error during pipmaster execution for '{req}' for {source_name}: {e}", exc_info=True)
            all_successful = False

    if not all_successful:
         logger.error(f"Failed to install one or more dependencies for {source_name}.")
    return all_successful


# --- Sync Version (for use during synchronous startup) ---
def ensure_dependencies_sync(requirements: List[str], source_name: str = "component") -> bool:
    """
    Checks and installs dependencies using pipmaster (synchronous version).
    Directly calls the synchronous PackageManager methods.
    """
    if not requirements:
        logger.debug(f"No dependencies specified for {source_name}.")
        return True

    if not pipmaster_installed:
        logger.warning(f"'pipmaster' is not installed. Cannot automatically manage dependencies for {source_name}: {requirements}. Please install them manually or install pipmaster.")
        return False # Treat as failure if dependencies are listed but pipmaster is missing

    logger.info(f"Checking/installing dependencies synchronously for {source_name}: {requirements}")
    all_successful = True

    for req in requirements:
        if not req or not isinstance(req, str):
            logger.warning(f"Skipping invalid requirement entry: {req} for {source_name}")
            continue
        try:
            logger.info(f"[Sync Check] Ensuring dependency: '{req}' for {source_name}...")
            # Use the synchronous instance
            success, output = pm_sync.install_or_update(req)

            if success:
                logger.info(f"Successfully ensured dependency '{req}' for {source_name}.")
            else:
                logger.error(f"Failed to install/update dependency '{req}' for {source_name}. Output:\n{output}")
                all_successful = False
        except Exception as e:
            logger.error(f"Unexpected error during sync pipmaster execution for '{req}' for {source_name}: {e}", exc_info=True)
            all_successful = False

    if not all_successful:
         logger.error(f"Failed to install one or more dependencies for {source_name}.")
    return all_successful