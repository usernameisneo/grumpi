# examples/personalities/scripted_example/scripts/code_exec_workflow.py
import logging
import asyncio
from typing import Any, Dict, Optional, List
from pathlib import Path # Import Path
from lollms_server.utils.helpers import extract_code_blocks
# TYPE_CHECKING block for Binding import (keep as is)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try: from lollms_server.core.bindings import Binding
    except ImportError: Binding = Any # type: ignore
    try: from lollms_server.core.personalities import Personality # For type hint
    except ImportError: Personality = Any # type: ignore

logger = logging.getLogger(__name__)

# --- SIMULATED EXECUTION ---
# In a real scenario, this would use subprocess.run or similar,
# BUT THAT IS EXTREMELY DANGEROUS AND REQUIRES SANDBOXING!

async def simulate_bash_execution(command: str) -> Dict[str, Any]:
    """Simulates running a bash command."""
    logger.warning(f"--- SIMULATING EXECUTION (NOT REAL): bash command ---")
    logger.warning(f"COMMAND: {command}")
    await asyncio.sleep(0.5) # Simulate execution time

    # Simulate some outputs based on common commands
    stdout = ""
    stderr = ""
    return_code = 0
    if command.strip() == "ls -l":
        stdout = """-rw-r--r-- 1 user group 1024 May 10 10:00 file1.txt
drwxr-xr-x 2 user group 4096 May 9 09:00 directory/"""
    elif command.startswith("echo "):
            stdout = command[5:] # Echo back the argument
    elif command.strip() == "pwd":
            stdout = "/simulated/current/directory"
    elif "invalid_command" in command:
            stderr = "bash: invalid_command: command not found"
            return_code = 127
    else:
            stdout = "Command executed successfully (simulated)."

    logger.warning(f"STDOUT (simulated):\n{stdout}")
    if stderr:
            logger.warning(f"STDERR (simulated):\n{stderr}")
    logger.warning(f"RETURN CODE (simulated): {return_code}")
    logger.warning(f"--- SIMULATION END ---")

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code
    }
# --- END SIMULATED EXECUTION ---
async def run_workflow(prompt: str, params: Dict[str, Any], context: Dict[str, Any]) -> str: # Return type hint changed to str
    """
    Workflow that generates bash commands, simulates execution, and returns a summary report.
    """

    # --- Extract request_info from context ---
    request_info = context.get("request_info", {}) # Get request_info, default to empty dict if missing
    # -----------------------------------------
    logger.info(f"Code Execution workflow started for prompt: '{prompt[:50]}...'")

    binding: Optional['Binding'] = context.get('binding')
    personality: Optional['Personality'] = context.get('personality') # Get the personality object

    if not binding: # No need to check Binding type definition availability at runtime here
        logger.error("Binding not found in context.")
        return "Error: Script could not access the language model binding."
    if not personality:
         logger.error("Personality object not found in context (should not happen for scripted).")
         return "Error: Script context is missing personality information."


    # --- Accessing Personality Data (Example) ---
    personality_root_path: Path = personality.path
    data_folder = personality_root_path / "data"
    db_file = data_folder / "db.sqlite"
    if db_file.exists():
         logger.info(f"Found data file: {db_file}")
         # Add code here to connect and query db_file if needed for RAG
    else:
         logger.info("No data/db.sqlite found for this personality.")
    # --- End Data Access Example ---


    generation_prompt = (
        f"Your task is to generate ONLY the necessary bash commands to achieve the user's goal, based on the following request. "
        f"You MUST enclose each distinct command or sequence of related commands in its own ```bash ... ``` markdown code block. "
        f"Do NOT include any explanations, comments, or other text outside of the ```bash ... ``` blocks.\n\n"
        f"User Request: {prompt}"
    )

    # Generate initial plan/commands
    # Use generate directly first to see the raw output
    raw_llm_response = await binding.generate(
        prompt=generation_prompt,
        params=params, # Includes personality conditioning as system message
        request_info=request_info
    )

    # --- ADD LOGGING FOR RAW RESPONSE ---
    logger.debug("--- RAW LLM Response for Command Generation ---")
    logger.debug(raw_llm_response)
    logger.debug("--- END RAW LLM Response ---")
    # --- END LOGGING ---

    # Now extract blocks from the raw response
    code_blocks_response = extract_code_blocks(raw_llm_response) # Assuming extract is sync now, or make binding.extract async
    if not code_blocks_response:
        logger.warning("LLM did not generate any code blocks.")
        fallback_response = await binding.generate(prompt, params, context)
        return f"I couldn't generate specific commands for that request. Here's a general response based on '{prompt}':\n{fallback_response}"

    execution_summary = f"--- Simulation Summary for Request: '{prompt}' ---\n"
    # --- Refine Prompt for Final Summary ---
    # Focus ONLY on the log, explicitly state the goal is summarization.
    final_result_prompt = "You are provided with a log of simulated bash command executions that were attempted to fulfill a user's request.\n"
    final_result_prompt += "Your task is to provide a concise summary of what happened during the execution based ONLY on the log below.\n"
    final_result_prompt += "Do NOT generate new commands or attempt to fulfill the original request again.\n\n"
    final_result_prompt += f"Original User Request (for context only): {prompt}\n\n"
    final_result_prompt += "Execution Log:\n"
    final_result_prompt += "------ LOG START ------\n"

    # Iterate through extracted blocks and simulate execution
    for i, block in enumerate(code_blocks_response):
        # ... (logic to extract lang, content, complete - same as before) ...
        lang = block.get('type')
        content = block.get('content')
        complete = block.get('is_complete')

        execution_summary += f"\nBlock {i+1} (Lang: {lang or 'N/A'}, Complete: {complete}):\n```bash\n{content}\n```\n"
        final_result_prompt += f"Block {i+1} Command(s):\n{content}\n"

        if lang == "bash" and content and complete:
            result = await simulate_bash_execution(content)
            execution_summary += "--- Simulated Result ---\n"
            execution_summary += f"Return Code: {result['return_code']}\n"
            if result['stdout']: execution_summary += f"Stdout:\n{result['stdout']}\n"
            if result['stderr']: execution_summary += f"Stderr:\n{result['stderr']}\n"
            # Append to prompt for final summary
            final_result_prompt += f"Result {i+1}: "
            if result['stdout']: final_result_prompt += f"Output: {result['stdout']} "
            if result['stderr']: final_result_prompt += f"Error: {result['stderr']} "
            final_result_prompt += "\n---\n"
        else:
             # ... (logging and appending skip messages - same as before) ...
             skip_reason = ""
             if not complete: skip_reason = "incomplete"
             elif lang != "bash": skip_reason = f"not bash (was {lang})"
             elif not content: skip_reason = "empty"
             msg = f"Skipping Block {i+1}: {skip_reason}."
             logger.warning(msg)
             execution_summary += f"{msg}\n"
             final_result_prompt += f"Block {i+1} was skipped ({skip_reason}).\n---\n"


    # Ask the LLM to summarize the results
    logger.info("Asking LLM to summarize execution results...")
    # Ensure we pass context which might be needed by generate internally
    final_summary = await binding.generate(
        prompt=final_result_prompt,
        params=params,
        request_info=context # Pass original context
    )


    # --- Return the combined result as a single string ---
    final_output = f"{execution_summary}\n--- Final Summary by LLM ---\n{final_summary}"
    logger.info("Code Execution workflow finished.")
    return final_output