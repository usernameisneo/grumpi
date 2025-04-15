# examples/personalities/python_builder_executor/scripts/workflow.py

import logging
import asyncio
import subprocess # For running commands
import platform # To detect OS
import tempfile # To create temporary files securely
import os
import sys # To get current python executable
from pathlib import Path
from typing import Any, Dict, Optional, List

# Import utility and potentially Binding type hint
# Assuming utils/helpers.py contains the latest extract_code_blocks function
from lollms_server.utils.helpers import extract_code_blocks
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # These imports are only for type checkers, avoids runtime circular dependencies if Binding/Personality imports AppConfig etc.
    try: from lollms_server.core.bindings import Binding
    except ImportError: Binding = Any # type: ignore
    try: from lollms_server.core.personalities import Personality
    except ImportError: Personality = Any # type: ignore
    # No direct AppConfig import needed here anymore

logger = logging.getLogger(__name__)

# --- Constants ---
SAVE_SUBDIR = ".lollms_generated_scripts" # Subdirectory in user's home for scripts
EXECUTION_TIMEOUT = 60 # Seconds for subprocess execution timeout

# --- Execution Function ---
async def execute_script(cwd_path: Path, command: List[str]) -> Dict[str, Any]:
    """
    Executes a command using subprocess within a specified working directory.

    Args:
        cwd_path: The Path object for the current working directory.
        command: A list of strings representing the command and its arguments.

    Returns:
        A dictionary containing:
        - 'stdout' (str): Standard output, decoded.
        - 'stderr' (str): Standard error, decoded.
        - 'return_code' (int): Exit code of the process (-1 or other error codes on failure).
    """
    result = {"stdout": "", "stderr": "", "return_code": -1} # Default error state
    process = None
    try:
        logger.info(f"Executing command: {' '.join(command)} in CWD: {cwd_path}")

        # Ensure command elements are strings
        str_command = [str(c) for c in command]

        process = await asyncio.create_subprocess_exec(
            *str_command, # Pass command elements as separate arguments
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_path) # Set current working directory for the subprocess
        )

        # Wait for completion with timeout
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=EXECUTION_TIMEOUT)

        # Process finished, get return code
        result["return_code"] = process.returncode if process.returncode is not None else -1 # Handle case where process terminated early

        # Decode stdout/stderr
        if stdout_bytes: result["stdout"] = stdout_bytes.decode('utf-8', errors='replace')
        if stderr_bytes: result["stderr"] = stderr_bytes.decode('utf-8', errors='replace')

        logger.info(f"Execution finished. Return Code: {result['return_code']}")
        if result["stdout"]: logger.debug(f"Stdout:\n{result['stdout']}")
        if result["stderr"]: logger.warning(f"Stderr:\n{result['stderr']}")

    except asyncio.TimeoutError:
        logger.error(f"Command execution timed out after {EXECUTION_TIMEOUT} seconds.")
        result["stderr"] = f"Execution timed out after {EXECUTION_TIMEOUT} seconds."
        result["return_code"] = -1
        if process and process.returncode is None: # Check if process is still running
            try: process.kill(); logger.info("Killed timed-out process.")
            except ProcessLookupError: logger.debug("Process already finished when killing.")
            except Exception as kill_e: logger.warning(f"Could not kill process: {kill_e}")
    except FileNotFoundError:
        cmd_str = command[0] if command else "N/A"
        logger.error(f"Command not found: {cmd_str}")
        result["stderr"] = f"Error: Command '{cmd_str}' not found. Ensure it's installed and in the system PATH."
        result["return_code"] = 127
    except Exception as e:
        cmd_str_safe = ' '.join(command) if command else "N/A"
        logger.error(f"Error executing command {cmd_str_safe}: {e}", exc_info=True)
        result["stderr"] = f"An unexpected error occurred during execution: {str(e)}"
        result["return_code"] = -1 # Generic error code
        if process and process.returncode is None: # Check if process is still running
            try: process.kill(); logger.info("Killed process after unexpected error.")
            except ProcessLookupError: logger.debug("Process already finished when killing after error.")
            except Exception as kill_e: logger.warning(f"Could not kill process after error: {kill_e}")

    return result # Always return the result dictionary

# --- Workflow ---
async def run_workflow(prompt: str, params: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Main workflow: Generates dependencies, python code, saves, executes (with retries),
    and returns a detailed report.
    """
    logger.info(f"Python Builder workflow started for prompt: '{prompt[:60]}...'")
    binding: Optional['Binding'] = context.get('binding')
    personality: Optional['Personality'] = context.get('personality')
    request_info = context.get("request_info", {}) # Get request info passed down

    # Basic checks for required context
    if not binding: return "Error: Script could not access the language model binding."
    if not personality: return "Error: Script context is missing personality information."

    # --- Get Max Retries from Personality Config ---
    max_retries = 1 # Default value
    if personality.instance_config: # PersonalityManager attaches this from config.toml
        max_retries = getattr(personality.instance_config, 'max_execution_retries', 1)
    max_retries = max(0, max_retries) # Ensure non-negative
    logger.info(f"Using max_execution_retries={max_retries} configured for this personality.")
    # --- End Get Max Retries ---

    final_report = f"--- Python Builder Execution Report for Request: '{prompt}' ---\n\n"
    python_exe = sys.executable
    pip_command_base = [python_exe, "-m", "pip"] # Use 'python -m pip'

    # == Step 1: Generate Dependency Installation Commands ==
    dep_prompt = (
        f"Analyze the user request: '{prompt}'. "
        f"If specific Python libraries (not part of the standard library) need installation via pip for the Python code that would fulfill the request, output the required `pip install <library1> <library2>...` command inside a SINGLE ```bash ... ``` block. "
        f"List all needed libraries in one command. If ONLY standard Python libraries are needed, output EXACTLY the text 'No external libraries needed.' without any markdown block."
    )
    logger.info("Attempting to generate dependency installation commands...")
    dep_params = params.copy(); dep_params['temperature'] = 0.1; dep_params['max_tokens'] = 150
    dep_response = await binding.generate(dep_prompt, dep_params, request_info)
    logger.debug(f"Raw response for dependency check:\n{dep_response}")

    pip_commands_str = None
    dep_blocks = extract_code_blocks(dep_response) # Use the helper from utils
    # Check only the first block for the specific dependency prompt structure
    if dep_blocks and dep_blocks[0].get('type', '').lower() == 'bash':
        pip_commands_str = dep_blocks[0].get('content','').strip()
        if pip_commands_str:
            logger.info(f"Found pip command(s): {pip_commands_str}")
            final_report += f"1. Dependency Check: Identified required libraries.\n   Command Block:\n```bash\n{pip_commands_str}\n```\n"
        else:
             logger.info("LLM generated an empty bash block for dependencies.")
             final_report += "1. Dependency Check: No external libraries required (empty block generated).\n"
             pip_commands_str = None # Treat empty block as no commands
    else:
        logger.info("No pip install commands identified by LLM or response was not a bash block.")
        final_report += "1. Dependency Check: No external libraries identified as needed.\n"

    # == Step 2: Execute Dependency Installation ==
    dependencies_installed_successfully = True # Assume success if no commands needed
    if pip_commands_str:
        logger.info("Attempting to install dependencies...")
        dep_result = None # Initialize

        # Validate command format
        if not pip_commands_str.startswith("pip install") or len(pip_commands_str.split()) < 3:
            logger.warning(f"Generated dependency command invalid or missing packages: '{pip_commands_str}'. Skipping installation.")
            final_report += f"2. Dependency Installation: Skipped (invalid command format: '{pip_commands_str}').\n"
            dependencies_installed_successfully = False
        else:
            try:
                pip_args = pip_commands_str.split()[2:] # Get args after "pip install"
                full_command = pip_command_base + pip_args
                # Execute pip from the server's current working directory
                dep_result = await execute_script(Path.cwd(), full_command)
            except Exception as exec_err:
                 logger.error(f"Failed to prepare or initiate execution for pip command: {exec_err}", exc_info=True)
                 dep_result = {"stdout": "", "stderr": f"Failed to run command: {exec_err}", "return_code": -2} # Ensure dict on error

            # --- Report Dependency Results ---
            if dep_result is None: # Should not happen due to try/except now
                 logger.error("execute_script returned None unexpectedly for dependency installation.")
                 final_report += "2. Dependency Installation Results:\n   - Outcome: FAILED (Internal execution error)\n"
                 dependencies_installed_successfully = False
            else:
                final_report += f"2. Dependency Installation Results:\n"
                # Safe access using .get()
                final_report += f"   - Command: `{' '.join(full_command)}`\n"
                final_report += f"   - Return Code: {dep_result.get('return_code', 'N/A')}\n"
                stdout = dep_result.get('stdout', '')
                stderr = dep_result.get('stderr', '')
                if stdout: final_report += f"   - Stdout:\n```text\n{stdout}\n```\n" # Use text tag for clarity
                else: final_report += "   - Stdout: (None)\n"
                if stderr: final_report += f"   - Stderr:\n```text\n{stderr}\n```\n" # Use text tag for clarity
                else: final_report += "   - Stderr: (None)\n"

                if dep_result.get('return_code') != 0:
                    logger.error(f"Dependency installation failed! Return code: {dep_result.get('return_code')}")
                    dependencies_installed_successfully = False
                    final_report += "   - Outcome: FAILED\n"
                else:
                    logger.info("Dependency installation appears successful.")
                    final_report += "   - Outcome: SUCCESS\n"
                # --- End Reporting ---
    else:
         final_report += "2. Dependency Installation: Skipped (no commands needed).\n"


    # == Step 3-5: Code Generation, Saving, Execution (Retry Loop) ==
    python_code = None
    script_path = None
    exec_result = None
    last_error = None
    save_dir = Path.home() / SAVE_SUBDIR
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure save dir exists
    final_execution_success = False # Track overall success

    for attempt in range(max_retries + 1):
        logger.info(f"--- Code Generation Attempt {attempt + 1}/{max_retries + 1} ---")
        final_report += f"\n--- Code Attempt {attempt + 1} ---\n"
        current_python_code = None # Reset for this attempt
        script_path = None # Reset path for this attempt
        exec_result = None # Reset result for this attempt

        # --- a) Construct Code Generation Prompt ---
        code_gen_prompt = ""
        if attempt == 0:
            code_gen_prompt = (
                f"Generate the complete, runnable Python code to fulfill the user's request: '{prompt}'. "
                f"Assume libraries were installed if needed. Enclose the entire Python code within a SINGLE ```python ... ``` markdown block. "
                f"Do NOT include any other text outside the code block."
            )
        else:
            logger.info("Regenerating code based on previous error.")
            final_report += "Status: Regenerating code due to previous execution error.\n"
            code_gen_prompt = (
                f"The previous attempt to generate Python code for the request '{prompt}' resulted in an error when executed. "
                f"Below is the code that failed and the error message (stderr).\n\n"
                f"Failed Code:\n```python\n{python_code or 'Code not available'}\n```\n\n"
                f"Execution Error:\n```text\n{last_error or 'Error message not available'}\n```\n\n" # Use text tag
                f"Your task is to regenerate the **complete and corrected** Python code to fulfill the original request, addressing the error. "
                f"Enclose the entire corrected Python code within a SINGLE ```python ... ``` markdown block. "
                f"Do NOT include explanations outside the block."
            )

        # --- b) Generate Code ---
        logger.info("Generating Python code...")
        code_gen_params = params.copy() # Use original params unless specified for retry
        code_response = await binding.generate(code_gen_prompt, code_gen_params, request_info)
        logger.debug(f"Raw response for code generation (Attempt {attempt+1}):\n{code_response}")

        code_blocks = extract_code_blocks(code_response)
        found_block = False
        for block in code_blocks:
            if block.get('type', '').lower() == 'python' and block.get('content'):
                current_python_code = block.get('content').strip()
                # Always report the generated code
                report_status = "(WARNING: may be incomplete)" if not block.get('is_complete') else "Success."
                final_report += f"Code Generation {attempt+1}: {report_status}\n```python\n{current_python_code}\n```\n"
                if block.get('is_complete'):
                    logger.info(f"Attempt {attempt+1}: Extracted complete Python code block.")
                else:
                    logger.warning(f"Attempt {attempt+1}: Extracted Python code block seems incomplete.")
                found_block = True
                break

        if not found_block:
            logger.error(f"Attempt {attempt+1}: Failed to extract Python code from LLM response.")
            final_report += f"Code Generation {attempt+1}: Failed - Could not extract Python code.\nLLM Raw Response:\n{code_response}\n"
            last_error = "LLM failed to generate valid Python code block."
            python_code = None # Ensure code is None
            continue # Go to next retry attempt

        python_code = current_python_code # Store latest valid code

        # --- c) Save Code ---
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".py", dir=save_dir, delete=False) as tmp_f:
                script_path = Path(tmp_f.name)
            script_path.write_text(python_code, encoding='utf-8')
            logger.info(f"Attempt {attempt+1}: Python code saved to: {script_path}")
            final_report += f"Code Saving {attempt+1}: Saved to `{script_path}`.\n"
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Failed to save Python code: {e}", exc_info=True)
            final_report += f"Code Saving {attempt+1}: Failed - Error: {e}\n"
            last_error = f"Failed to save script: {e}"
            if script_path and script_path.exists():
                 try: script_path.unlink(); logger.info(f"Cleaned up script: {script_path}")
                 except OSError: pass
            script_path = None
            continue

        # --- d) Execute Code ---
        logger.info(f"Attempt {attempt+1}: Executing saved Python script: {script_path}...")
        exec_command = [python_exe, str(script_path)]
        # Use script's directory as CWD
        exec_result = await execute_script(script_path.parent, exec_command)

        final_report += f"Code Execution {attempt+1} Results:\n"
        final_report += f"   - Command: `{' '.join(exec_command)}`\n"
        final_report += f"   - Return Code: {exec_result.get('return_code', 'N/A')}\n"
        stdout = exec_result.get('stdout', '')
        stderr = exec_result.get('stderr', '')
        if stdout: final_report += f"   - Stdout:\n```text\n{stdout}\n```\n"
        else: final_report += "   - Stdout: (None)\n"
        if stderr: final_report += f"   - Stderr:\n```text\n{stderr}\n```\n"
        else: final_report += "   - Stderr: (None)\n"

        # --- e) Check for Success ---
        # Success = return code 0 AND empty stderr (adjust if stderr sometimes has non-error output)
        if exec_result.get('return_code') == 0 and not stderr:
            logger.info(f"Attempt {attempt+1}: Execution successful!")
            final_report += "   - Outcome: SUCCESS\n"
            final_execution_success = True
            # Optional: Clean up successful script?
            # if script_path and script_path.exists():
            #    try: script_path.unlink(); logger.info(f"Cleaned up successful script: {script_path}")
            #    except OSError: pass
            break # Exit the retry loop on success
        else:
            logger.warning(f"Attempt {attempt+1}: Execution failed or produced stderr.")
            final_report += f"   - Outcome: FAILED (Return Code: {exec_result.get('return_code')})\n"
            last_error = stderr or f"Execution failed with return code {exec_result.get('return_code')} and no stderr."
            # Clean up failed script immediately? Or leave for debugging? Leave for now.
            # if script_path and script_path.exists():
            #    try: script_path.unlink()
            #    except OSError: pass
            # Continue to next retry attempt if available

    # --- End of Retry Loop ---

    # == Final Reporting ==
    if final_execution_success:
         final_report += "\n--- Overall Result: SUCCESS ---"
    else:
         final_report += f"\n--- Overall Result: FAILED after {attempt + 1} attempts ---"
         if last_error:
              final_report += f"\nLast Error encountered:\n```text\n{last_error}\n```" # Format error

    logger.info("Python Builder workflow finished.")
    return final_report