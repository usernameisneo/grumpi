# examples/personalities/scripted_example/scripts/workflow.py
# or personal_personalities/python_builder_executor/scripts/workflow.py

import logging
import asyncio
import subprocess # For running commands
import platform # To detect OS
import tempfile # To create temporary files securely
import os, sys
from pathlib import Path
from typing import Any, Dict, Optional, List

# Import utility and potentially Binding type hint
from lollms_server.utils.helpers import extract_code_blocks
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from lollms_server.core.bindings import Binding
    except ImportError: Binding = Any # type: ignore
    try: from lollms_server.core.personalities import Personality
    except ImportError: Personality = Any # type: ignore

logger = logging.getLogger(__name__)

# --- Constants ---
# Define a safe subdirectory within the user's home for generated scripts
# Avoids cluttering the home directory directly.
SAVE_SUBDIR = ".lollms_generated_scripts"

# --- Execution Function ---
async def execute_script(script_path: Path, command: List[str]) -> Dict[str, Any]:
    """
    Executes a script file using subprocess.
    Returns a dictionary with stdout, stderr, and returncode.
    """
    result = {"stdout": "", "stderr": "", "return_code": -1}
    process = None
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        # Use asyncio's subprocess for better integration
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_path.parent # Run from the script's directory
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60) # 60 second timeout for execution

        result["return_code"] = process.returncode
        if stdout_bytes:
            result["stdout"] = stdout_bytes.decode('utf-8', errors='replace')
        if stderr_bytes:
            result["stderr"] = stderr_bytes.decode('utf-8', errors='replace')

        logger.info(f"Execution finished. Return Code: {result['return_code']}")
        if result["stdout"]: logger.debug(f"Stdout:\n{result['stdout']}")
        if result["stderr"]: logger.warning(f"Stderr:\n{result['stderr']}")

    except asyncio.TimeoutError:
        logger.error("Script execution timed out.")
        result["stderr"] = "Execution timed out after 60 seconds."
        result["return_code"] = -1
        if process: process.kill() # Attempt to kill runaway process
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        result["stderr"] = f"Error: Command '{command[0]}' not found. Is it in your PATH?"
        result["return_code"] = -1
    except Exception as e:
        logger.error(f"Error executing script {script_path}: {e}", exc_info=True)
        result["stderr"] = f"An unexpected error occurred during execution: {str(e)}"
        result["return_code"] = -1
        if process and process.returncode is None: process.kill()

    return result

# --- Workflow ---
async def run_workflow(prompt: str, params: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Generates dependencies, python code, saves, executes, and returns results.
    """
    logger.info(f"Python Builder workflow started for prompt: '{prompt[:60]}...'")

    binding: Optional['Binding'] = context.get('binding')
    personality: Optional['Personality'] = context.get('personality')
    # Get config to potentially access paths if needed later
    # app_config: Optional['AppConfig'] = context.get('config')
    request_info = context.get("request_info", {})

    if not binding: return "Error: Script could not access the language model binding."
    if not personality: return "Error: Script context is missing personality information."

    final_report = f"--- Python Builder Execution Report for Request: '{prompt}' ---\n\n"

    # == Step 1: Generate Dependency Installation Commands (if any) ==
    dep_prompt = (
        f"Analyze the following user request. Determine if any specific Python libraries need to be installed using pip for the Python code that would fulfill the request. "
        f"If libraries ARE needed, output ONLY the required `pip install ...` command(s) inside a single ```bash ... ``` block. "
        f"If NO external libraries are needed (only standard Python libraries), output the text 'No external libraries needed.' WITHOUT any markdown block.\n\n"
        f"User Request: {prompt}"
    )
    logger.info("Attempting to generate dependency installation commands...")
    dep_params = params.copy()
    dep_params['temperature'] = 0.1 # Be factual about dependencies
    dep_params['max_tokens'] = 150 # Limit token usage for this step

    dep_response = await binding.generate(dep_prompt, dep_params, request_info)
    logger.debug(f"Raw response for dependency check:\n{dep_response}")

    pip_commands_str = None
    if "```bash" in dep_response:
        dep_blocks = extract_code_blocks(dep_response)
        for block in dep_blocks:
            # Find first bash block
            if block.get('type', '').lower() == 'bash' and block.get('content'):
                pip_commands_str = block.get('content').strip()
                logger.info(f"Found pip command(s): {pip_commands_str}")
                final_report += f"1. Dependency Check: Identified required libraries.\n   Commands:\n```bash\n{pip_commands_str}\n```\n"
                break
    else:
         logger.info("No pip install commands identified by LLM.")
         final_report += "1. Dependency Check: No external libraries identified as needed.\n"

    # == Step 2: Execute Dependency Installation (if commands found) ==
    if pip_commands_str:
        logger.info("Attempting to install dependencies...")
        # Determine shell command prefix based on OS
        shell_prefix = []
        if platform.system() == "Windows":
            shell_prefix = ["cmd", "/c"] # or ["powershell", "-Command"]
        # For Linux/macOS, commands often run directly or via 'bash -c'

        # Split potentially multiple commands (handle '&&' or newlines)
        # Simple newline split for now
        commands_to_run = [cmd.strip() for cmd in pip_commands_str.split('\n') if cmd.strip()]

        dep_results_summary = ""
        success_count = 0
        for cmd_line in commands_to_run:
             full_command = shell_prefix + [cmd_line]
             # Using asyncio.create_subprocess_exec for execution
             dep_result = await execute_script(Path.cwd(), full_command) # Execute from server's CWD
             dep_results_summary += f"\n   - Command: `{' '.join(full_command)}`\n"
             dep_results_summary += f"     Return Code: {dep_result['return_code']}\n"
             if dep_result['stdout']: dep_results_summary += f"     Stdout: {dep_result['stdout'][:200]}...\n"
             if dep_result['stderr']: dep_results_summary += f"     Stderr: {dep_result['stderr'][:200]}...\n"
             if dep_result['return_code'] == 0:
                 success_count += 1

        final_report += f"2. Dependency Installation Results ({success_count}/{len(commands_to_run)} successful):\n{dep_results_summary}\n"
        if success_count != len(commands_to_run):
             logger.warning("Not all dependency installations succeeded.")
             # Optionally stop here or continue anyway? Let's continue for now.
    else:
         final_report += "2. Dependency Installation: Skipped (no commands generated).\n"


    # == Step 3: Generate Python Code ==
    code_gen_prompt = (
        f"Generate the complete, runnable Python code to fulfill the user's request. "
        f"Assume any necessary libraries identified previously have been installed. "
        f"Enclose the entire Python code within a SINGLE ```python ... ``` markdown block. "
        f"Do NOT include any other text, explanations, or comments outside the code block.\n\n"
        f"User Request: {prompt}"
    )
    logger.info("Generating Python code...")
    code_gen_params = params.copy() # Use original params for creativity
    python_code = None

    code_response = await binding.generate(code_gen_prompt, code_gen_params, request_info)
    logger.debug(f"Raw response for code generation:\n{code_response}")

    code_blocks = extract_code_blocks(code_response)
    for block in code_blocks:
        if block.get('type', '').lower() == 'python' and block.get('content'):
             python_code = block.get('content').strip()
             if block.get('is_complete'):
                 logger.info("Extracted complete Python code block.")
                 final_report += f"3. Code Generation: Successfully generated Python code.\n```python\n{python_code}\n```\n"
             else:
                  logger.warning("Extracted Python code block seems incomplete.")
                  final_report += f"3. Code Generation: Generated Python code (may be incomplete).\n```python\n{python_code}\n```\n"
             break # Take the first python block

    if not python_code:
        logger.error("Failed to extract Python code from LLM response.")
        final_report += "3. Code Generation: Failed - Could not extract Python code from the response.\n"
        # Include raw response for debugging
        final_report += f"\nLLM Raw Response for Code Generation:\n{code_response}\n"
        return final_report # Stop workflow if code generation failed

    # == Step 4: Save Python Code to File ==
    script_path = None
    try:
        # Create the save directory if it doesn't exist
        save_dir = Path.home() / SAVE_SUBDIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use tempfile to create a named temporary file securely within that dir
        # This avoids filename collisions and manages cleanup better if needed,
        # though we'll keep the file for inspection in this example.
        # We give it a .py suffix.
        with tempfile.NamedTemporaryFile(
            mode='w', suffix=".py", encoding='utf-8',
            dir=save_dir, delete=False # Keep the file after closing
        ) as tmp_file:
            tmp_file.write(python_code)
            script_path = Path(tmp_file.name) # Get the full path
        logger.info(f"Python code saved to: {script_path}")
        final_report += f"4. Code Saving: Saved Python code to `{script_path}`.\n"
    except Exception as e:
        logger.error(f"Failed to save Python code: {e}", exc_info=True)
        final_report += f"4. Code Saving: Failed - Could not save Python code.\nError: {e}\n"
        return final_report # Stop if saving failed

    # == Step 5: Execute Python Code ==
    if script_path:
        logger.info(f"Executing saved Python script: {script_path}...")
        # Determine Python command
        python_exe = sys.executable # Use the same python that runs the server
        exec_command = [python_exe, str(script_path)]

        exec_result = await execute_script(script_path, exec_command)

        final_report += f"5. Code Execution Results:\n"
        final_report += f"   - Command: `{' '.join(exec_command)}`\n"
        final_report += f"   - Return Code: {exec_result['return_code']}\n"
        if exec_result['stdout']: final_report += f"   - Stdout:\n{exec_result['stdout']}\n"
        if exec_result['stderr']: final_report += f"   - Stderr:\n{exec_result['stderr']}\n"
    else:
         final_report += "5. Code Execution: Skipped (code saving failed).\n"

    # == Step 6: Final Summary (Optional - LLM summarizes the report) ==
    # This might be overkill, the report itself is the result.
    # You could ask the LLM to summarize final_report if needed.

    logger.info("Python Builder workflow finished.")
    return final_report