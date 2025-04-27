# zoos/personalities/python_builder_executor/scripts/workflow.py

import ascii_colors as logging
import asyncio
import subprocess # For running commands
import platform # To detect OS
import tempfile # To create temporary files securely
import os
import sys # To get current python executable
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from ascii_colors import  ASCIIColors
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
SAVE_SUBDIR = ".lollms_generated_scripts" # Subdirectory in user's home for scripts
EXECUTION_TIMEOUT = 60 # Seconds for subprocess execution timeout

# --- Execution Function ---
async def execute_script(cwd_path: Path, command: List[str]) -> Dict[str, Any]:
    """Executes a command using subprocess."""
    result = {"stdout": "", "stderr": "", "return_code": -1}
    process = None
    try:
        logger.info(f"Executing command: {' '.join(command)} in CWD: {cwd_path}")
        str_command = [str(c) for c in command]
        process = await asyncio.create_subprocess_exec(
            *str_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(cwd_path)
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=EXECUTION_TIMEOUT)
        result["return_code"] = process.returncode if process.returncode is not None else -1
        if stdout_bytes: result["stdout"] = stdout_bytes.decode('utf-8', errors='replace')
        if stderr_bytes: result["stderr"] = stderr_bytes.decode('utf-8', errors='replace')
        logger.info(f"Execution finished. RC: {result['return_code']}")
        if result["stdout"]: logger.debug(f"Stdout:\n{result['stdout']}")
        if result["stderr"]: logger.warning(f"Stderr:\n{result['stderr']}")
    except asyncio.TimeoutError:
        logger.error(f"Command timed out after {EXECUTION_TIMEOUT}s."); result["stderr"] = f"Timeout ({EXECUTION_TIMEOUT}s)."; result["return_code"] = -1
        if process and process.returncode is None: 
            try:
                process.kill()
                logger.info("Killed timed-out process.")            
            except Exception as kill_e:
                logger.warning(f"Could not kill: {kill_e}")
    except FileNotFoundError:
        cmd_str = command[0] if command else "N/A"; logger.error(f"Command not found: {cmd_str}")
        result["stderr"] = f"Error: Command '{cmd_str}' not found."; result["return_code"] = 127
    except Exception as e:
        cmd_str_safe = ' '.join(command) if command else "N/A"; logger.error(f"Error executing {cmd_str_safe}: {e}", exc_info=True)
        result["stderr"] = f"Execution error: {str(e)}"; result["return_code"] = -1
        if process and process.returncode is None:
            try:
                process.kill()
                logger.info("Killed process after error.")
            except Exception as kill_e:
                logger.warning(f"Could not kill after error: {kill_e}")
    return result

# --- Workflow ---
async def run_workflow(prompt: str, params: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Main workflow: generates dependencies, python code, saves, executes."""
    logger.info(f"Python Builder workflow started for prompt: '{prompt[:60]}...'")
    binding: Optional['Binding'] = context.get('binding')
    personality: Optional['Personality'] = context.get('personality')
    request_info = context.get("request_info", {})

    if not binding: return "Error: Script could not access binding."
    if not personality: return "Error: Script context missing personality info."

    max_retries = getattr(personality.instance_config, 'max_execution_retries', 1) if personality.instance_config else 1
    max_retries = max(0, max_retries); logger.info(f"Using max_execution_retries={max_retries}.")

    final_report = f"--- Python Builder Execution Report ---\nRequest: '{prompt}'\n\n"
    python_exe = sys.executable
    pip_command_base = [python_exe, "-m", "pip"]

    # == Step 1: Generate Dependency Commands ==
    dep_prompt = ( f"Analyze user request: '{prompt}'. If specific Python libraries need pip install, output the `pip install ...` command in a SINGLE ```bash ... ``` block. List all in one command. If only standard libraries needed, output EXACTLY 'No external libraries needed.' without markdown." )
    logger.info("Generating dependency commands...")
    dep_params = params.copy(); dep_params['temperature'] = 0.1; dep_params['max_tokens'] = 150

    # --- FIX: Expect dict and extract 'text' ---
    dep_response_dict: Union[str, Dict[str, Any]] = await binding.generate(dep_prompt, dep_params, request_info)
    dep_response_text = ""
    if isinstance(dep_response_dict, dict) and "text" in dep_response_dict:
        dep_response_text = dep_response_dict["text"]
    elif isinstance(dep_response_dict, str): # Handle unexpected string return
        dep_response_text = dep_response_dict
        logger.warning("Dependency check binding returned string instead of dict.")
    else:
        logger.error(f"Dependency check failed: Unexpected response format: {dep_response_dict}")
        final_report += "1. Dependency Check: FAILED (Bad LLM response format)\n"
        dep_response_text = "" # Ensure it's a string for extract_code_blocks

    logger.debug(f"Raw text response for dependency check:\n{dep_response_text}")
    # --- END FIX ---

    pip_commands_str = None
    dep_blocks = extract_code_blocks(dep_response_text) # Pass the extracted text string
    if dep_blocks and dep_blocks[0].get('type', '').lower() == 'bash':
        pip_commands_str = dep_blocks[0].get('content','').strip()
        if pip_commands_str: logger.info(f"Found pip command(s): {pip_commands_str}"); final_report += f"1. Dependency Check: Required libraries identified.\n   Command:\n```bash\n{pip_commands_str}\n```\n"
        else: logger.info("LLM generated empty bash block for deps."); final_report += "1. Dependency Check: No external libraries required (empty block).\n"; pip_commands_str = None
    else: logger.info("No pip install commands identified."); final_report += "1. Dependency Check: No external libraries identified.\n"

    # == Step 2: Execute Dependency Installation ==
    dependencies_installed_successfully = True
    if pip_commands_str:
        logger.info("Attempting to install dependencies...")
        dep_result = None
        if not pip_commands_str.startswith("pip install") or len(pip_commands_str.split()) < 3:
            logger.warning(f"Invalid dependency command: '{pip_commands_str}'. Skipping."); final_report += f"2. Dependency Installation: Skipped (invalid format).\n"; dependencies_installed_successfully = False
        else:
            try:
                pip_args = pip_commands_str.split()[2:]; full_command = pip_command_base + pip_args
                dep_result = await execute_script(Path.cwd(), full_command)
            except Exception as exec_err: logger.error(f"Failed pip execution setup: {exec_err}", exc_info=True); dep_result = {"stderr": f"Failed command setup: {exec_err}", "return_code": -2}
            if dep_result is None: logger.error("Dependency installation execution returned None."); final_report += "2. Dependency Installation: FAILED (Exec error)\n"; dependencies_installed_successfully = False
            else:
                final_report += f"2. Dependency Installation Results:\n   - Command: `{' '.join(full_command)}`\n   - RC: {dep_result.get('return_code', 'N/A')}\n"; stdout=dep_result.get('stdout',''); stderr=dep_result.get('stderr','')
                final_report += f"   - Stdout:\n```text\n{stdout or '(None)'}\n```\n"; final_report += f"   - Stderr:\n```text\n{stderr or '(None)'}\n```\n"
                if dep_result.get('return_code') != 0: logger.error(f"Dependency installation failed! RC: {dep_result.get('return_code')}"); dependencies_installed_successfully = False; final_report += "   - Outcome: FAILED\n"
                else: logger.info("Dependency installation OK."); final_report += "   - Outcome: SUCCESS\n"
    else: final_report += "2. Dependency Installation: Skipped (none needed).\n"

    # == Step 3-5: Code Generation, Saving, Execution Loop ==
    python_code = None; script_path = None; exec_result = None; last_error = None
    save_dir = Path.home() / SAVE_SUBDIR; save_dir.mkdir(parents=True, exist_ok=True)
    final_execution_success = False

    for attempt in range(max_retries + 1):
        logger.info(f"--- Code Generation Attempt {attempt + 1}/{max_retries + 1} ---")
        final_report += f"\n--- Code Attempt {attempt + 1} ---\n"
        current_python_code = None; script_path = None; exec_result = None

        # --- a) Construct Code Generation Prompt ---
        code_gen_prompt = ""
        if attempt == 0: code_gen_prompt = f"Generate complete Python code for request: '{prompt}'. Assume libraries installed. Enclose code ONLY in ```python ... ``` block."
        else:
            logger.info("Regenerating code due to error."); final_report += "Status: Regenerating code...\n"
            code_gen_prompt = ( f"Previous attempt for request '{prompt}' failed.\n" f"Failed Code:\n```python\n{python_code or 'N/A'}\n```\n" f"Error:\n```text\n{last_error or 'N/A'}\n```\nRegenerate complete, corrected code in ONE ```python ... ``` block only." )

        # --- b) Generate Code ---
        logger.info("Generating Python code...")
        code_gen_params = params.copy()

        # --- FIX: Expect dict and extract 'text' ---
        code_response_dict: Union[str, Dict[str, Any]] = await binding.generate(code_gen_prompt, code_gen_params, request_info)
        code_response_text = ""
        if isinstance(code_response_dict, dict) and "text" in code_response_dict:
            code_response_text = code_response_dict["text"]
        elif isinstance(code_response_dict, str):
            code_response_text = code_response_dict
            logger.warning("Code generation binding returned string instead of dict.")
        else:
            logger.error(f"Code generation failed: Unexpected response format: {code_response_dict}")
            final_report += f"Code Generation {attempt+1}: FAILED (Bad LLM response format)\nLLM Raw Response:\n{code_response_dict}\n"
            last_error = "LLM failed to generate Python code block."
            python_code = None; continue # Retry

        logger.debug(f"Raw text response for code generation (Attempt {attempt+1}):\n{code_response_text}")
        # --- END FIX ---

        code_blocks = extract_code_blocks(code_response_text) # Pass extracted string
        found_block = False
        for block in code_blocks:
            if block.get('type', '').lower() == 'python' and block.get('content'):
                current_python_code = block.get('content').strip(); report_status = "(WARNING: may be incomplete)" if not block.get('is_complete') else "Success."
                final_report += f"Code Generation {attempt+1}: {report_status}\n```python\n{current_python_code}\n```\n"
                if block.get('is_complete'): logger.info(f"Attempt {attempt+1}: Extracted complete Python block.")
                else: logger.warning(f"Attempt {attempt+1}: Extracted Python block incomplete.")
                found_block = True; break
        if not found_block:
            logger.error(f"Attempt {attempt+1}: Failed to extract Python code."); final_report += f"Code Generation {attempt+1}: Failed - No Python block.\nLLM Raw Response:\n{code_response_text}\n"; last_error = "LLM failed valid Python block."; python_code = None; continue

        python_code = current_python_code

        # --- c) Save Code ---
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".py", dir=save_dir, delete=False) as tmp_f: script_path = Path(tmp_f.name)
            script_path.write_text(python_code, encoding='utf-8'); logger.info(f"Attempt {attempt+1}: Code saved to: {script_path}"); final_report += f"Code Saving {attempt+1}: Saved to `{script_path}`.\n"
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Failed save: {e}", exc_info=True); final_report += f"Code Saving {attempt+1}: Failed - Error: {e}\n"; last_error = f"Failed save: {e}"
            if script_path and script_path.exists(): 
                try: script_path.unlink(); logger.info(f"Cleaned up script: {script_path}"); 
                except OSError: pass
            script_path = None; continue

        # --- d) Execute Code ---
        logger.info(f"Attempt {attempt+1}: Executing script: {script_path}...")
        exec_command = [python_exe, str(script_path)]
        exec_result = await execute_script(script_path.parent, exec_command)
        final_report += f"Code Execution {attempt+1} Results:\n   - Command: `{' '.join(exec_command)}`\n   - RC: {exec_result.get('return_code', 'N/A')}\n"; stdout=exec_result.get('stdout',''); stderr=exec_result.get('stderr','')
        final_report += f"   - Stdout:\n```text\n{stdout or '(None)'}\n```\n"; final_report += f"   - Stderr:\n```text\n{stderr or '(None)'}\n```\n"

        # --- e) Check for Success ---
        if exec_result.get('return_code') == 0 and not stderr.strip(): # Check stderr is empty/whitespace only
            logger.info(f"Attempt {attempt+1}: Execution successful!"); final_report += "   - Outcome: SUCCESS\n"; final_execution_success = True; break
        else:
            logger.warning(f"Attempt {attempt+1}: Execution failed or produced stderr."); final_report += f"   - Outcome: FAILED (RC: {exec_result.get('return_code')})\n"; last_error = stderr or f"Failed with RC {exec_result.get('return_code')} (no stderr)."

    # --- End of Retry Loop ---

    # == Final Reporting ==
    final_report += f"\n--- Overall Result: {'SUCCESS' if final_execution_success else f'FAILED after {attempt + 1} attempts'} ---"
    if not final_execution_success and last_error: final_report += f"\nLast Error:\n```text\n{last_error}\n```"

    logger.info("Python Builder workflow finished.")
    return final_report