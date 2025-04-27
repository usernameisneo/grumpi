import gradio as gr
import requests
import json
import sys
import os
import datetime
from pathlib import Path
import traceback
import tempfile

# --- Configuration Loading ---
CONFIG_FILE = Path("app_config.json")
DEFAULT_OUTPUT_DIR = Path("generated_data")

# --- Global Variables ---
# Values will be loaded/set by load_config or the wizard
BASE_URL: str = "http://localhost:9600"    # Default placeholder
API_KEY: str | None = None                # Default placeholder
OUTPUT_DIR: Path = DEFAULT_OUTPUT_DIR     # Default
HEADERS_NO_STREAM: dict[str, str] = {}
config_needed: bool = False               # Flag to control UI state

def load_config() -> bool:
    """
    Loads configuration from app_config.json if it exists.
    Sets up global variables.
    Returns True if config loaded successfully, False otherwise (e.g., file missing).
    """
    global BASE_URL, API_KEY, HEADERS_NO_STREAM, OUTPUT_DIR, config_needed

    if not CONFIG_FILE.exists():
        print(f"⚠️ '{CONFIG_FILE}' not found. Configuration wizard needed.")
        config_needed = True
        # Ensure default output dir exists even before config
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR # Use default until configured
        return False

    try:
        print(f"📂 Loading configuration from {CONFIG_FILE}...")
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

        BASE_URL = config.get("lollms_base_url", "http://localhost:9600").rstrip('/')
        API_KEY = config.get("lollms_api_key") # Can be None or empty string
        output_dir_str = config.get("output_dir", str(DEFAULT_OUTPUT_DIR))
        OUTPUT_DIR = Path(output_dir_str)

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Setup headers
        HEADERS_NO_STREAM = { "Content-Type": "application/json", "Accept": "application/json" }
        if API_KEY:
            HEADERS_NO_STREAM["X-API-Key"] = API_KEY

        print("✅ Configuration loaded successfully.")
        print(f"   - Server URL: {BASE_URL}")
        print(f"   - API Key Set: {'Yes' if API_KEY else 'No'}")
        print(f"   - Output Dir: {OUTPUT_DIR}")
        config_needed = False # Config is now loaded
        return True

    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in '{CONFIG_FILE}': {e}")
        print("   Please fix the file or delete it to run the wizard again.")
        config_needed = True # Re-run wizard maybe?
        # Fallback to defaults
        BASE_URL = "http://localhost:9600"
        API_KEY = None
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        HEADERS_NO_STREAM = { "Content-Type": "application/json", "Accept": "application/json" }
        return False # Indicate load failure
    except Exception as e:
        print(f"❌ ERROR loading '{CONFIG_FILE}': {e}")
        traceback.print_exc()
        config_needed = True # Assume config is bad, need wizard
        # Reset to safe defaults
        BASE_URL = "http://localhost:9600"
        API_KEY = None
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        HEADERS_NO_STREAM = { "Content-Type": "application/json", "Accept": "application/json" }
        return False # Indicate load failure


# --- The Meta-Prompt for Generating Training Data ---
# Using .replace() method, so single braces {} are fine in examples.
# The specific placeholder {user_provided_content_here} will be replaced.
TRAINING_DATA_GENERATION_PROMPT_TEMPLATE = """
# LLM Training Data Generation Prompt

## Your Role:
You are an expert AI assistant specialized in generating high-quality, structured training data for other language models. Your goal is to create realistic and informative discussions based *solely* on the provided source content. The discussions feature a confident AI that gives the information or does the task without needing the documentation. The objective is to make the fine tuned AI know the content of the document (no need to rag to answer the user). It can recall references if they explicitely exist, but shouldn't need to have any content in the discussion itself.

## Objective:
Generate a JSON output containing a list of distinct discussions. Each discussion object within the list must have two keys:
1.  `system_prompt`: A carefully crafted system prompt for a *target* LLM. This prompt should instruct the target LLM to act as an expert *only* on the specific topic covered in the accompanying discussion data and to answer questions based *exclusively* on the information contained within that discussion.
2.  `discussion`: A list of conversation turns formatted as {"role": "user", "content": "..."} and {"role": "assistant", "content": "..."}. This conversation must strictly explore, explain, clarify, or query aspects of the provided source content.

## Input Content:
The source material for the discussions will be provided by the user within the `<SOURCE_CONTENT>` tags below. You MUST base all generated discussions entirely on this content. Do not introduce any external knowledge, assumptions, or information not present in the source.

<SOURCE_CONTENT>
{user_provided_content_here}
</SOURCE_CONTENT>

## Task Requirements:

1.  **Analyze Source Content:** Thoroughly read and understand the information provided in `<SOURCE_CONTENT>`. Identify key concepts, facts, procedures, or data points.
2.  **Craft Target System Prompts:** For each discussion you generate, create a concise and specific `system_prompt`. This prompt should define the persona of the target LLM (e.g., "You are an AI assistant knowledgeable *only* about the provided text regarding [Specific Topic from Source Content].") and explicitly state the constraint that it must *only* use the information from the subsequent conversation history (`discussion` data) to answer user queries.
3.  **Generate Diverse Discussions:** Create multiple (aim for 3-5 if the content allows, otherwise adjust based on content richness) distinct discussions. Each `discussion` should:
    *   Be a coherent conversation between a "user" and an "assistant".
    *   Start with a user query or statement related to the source content with precise context (the user does not have access to the document he just asks question so do not assume the content is present in the context) .
    *   Contain multiple turns (at least 2-3 pairs of user/assistant messages).
    *   Explore different facets of the source content (e.g., definitions, explanations, comparisons, step-by-step processes, data interpretation, task building etc).
    *   Ensure the "assistant" responses are accurate, informative, helpful, and derived *directly* and *exclusively* from the `<SOURCE_CONTENT>` without referencing them as the idea is that the LLM knows that content and is capable of using it to answer or perform tasks. The assistant should make use of the content without as if it was already in its brain.
4.  **Format as JSON:** Structure the entire output as a single JSON list `[...]`. Each element in the list must be a JSON object { "system_prompt": "...", "discussion": [...] }.
5.  **Handle Escaping:** Critically important: Ensure that any special characters within the `content` strings (and the `system_prompt` string) that could break JSON parsing or interfere with code interpretation (e.g., double quotes `"` within the text, backslashes `\\`, newlines `\\n`, tabs `\\t`) are correctly escaped. For example, a double quote inside a content string must become `\\\"`, a backslash must become `\\\\`, etc. This applies especially if the source content itself contains code snippets or JSON examples.
6.  **Quality Focus:** Aim for high-quality, natural-sounding conversations that would be genuinely useful for fine-tuning an LLM to be knowledgeable and constrained to specific information domains. Avoid trivial or nonsensical exchanges.

## Output Format Example:

```json
[
  {
    "system_prompt": "You are an AI assistant specialized in explaining the core concepts of Quantum Entanglement based *only* on the provided documentation snippet. Answer questions strictly using the information given in this conversation.",
    "discussion": [
      {
        "role": "user",
        "content": "Can you explain what Quantum Entanglement means according to the document?"
      },
      {
        "role": "assistant",
        "content": "Based on the provided text, Quantum Entanglement is a phenomenon where two or more quantum particles become linked in such a way that they share the same fate, regardless of the distance separating them. Measuring a property (like spin) of one particle instantaneously influences the correlated property of the other particle(s)."
      }
      // ... more discussion objects if applicable ...
    ]
  }
]
```

## Final Instruction:
Proceed to analyze the `<SOURCE_CONTENT>` and generate the JSON output containing the list of system prompts and their corresponding discussions, adhering strictly to all requirements, especially content grounding and proper JSON escaping. Ensure the output starts *directly* with `[` and ends *directly* with `]`. Do not include any introductory text or ```json markdown fences around the final JSON output.
"""

# --- API Client Function ---
def generate_data_via_api(full_prompt: str, generation_config: dict | None = None) -> tuple[str | None, str | None]:
    """
    Makes a non-streaming API call to the lollms-server /api/v1/generate endpoint.
    Uses globally loaded BASE_URL and HEADERS_NO_STREAM.
    """
    if config_needed: # Should not happen if UI logic is correct, but safe check
        return None, "Error: Configuration is not set. Please complete the setup wizard."

    api_url = f"{BASE_URL}/api/v1/generate"
    payload = {
        "input_data": [
             {"type": "text", "role": "system_prompt", "data": "You follow instructions precisely."},
             {"type": "text", "role": "user_prompt", "data": full_prompt}
        ],
        "generation_type": "ttt",
        "stream": False,
        # "generation_config": generation_config or {} # Optional: Add if needed
    }

    print(f"🚀 Sending request to {api_url}...")
    try:
        # Increased timeout for potentially long generations
        response = requests.post(api_url, headers=HEADERS_NO_STREAM, json=payload, timeout=600) # 10 min timeout
        response.raise_for_status()
        result = response.json()
        print("✅ Received response.")

        # Adapt based on potential v1 API structure changes
        if isinstance(result, dict) and "output" in result:
            output_data = result["output"]
            generated_text = ""
            if isinstance(output_data, dict) and "text" in output_data:
                generated_text = output_data["text"]
            elif isinstance(output_data, str): # Direct string output
                 generated_text = output_data
            elif isinstance(output_data, list) and output_data: # List of outputs, take first text?
                first_item = output_data[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    generated_text = first_item["text"]
                elif isinstance(first_item, str):
                    generated_text = first_item
            else:
                print(f"⚠️ Unexpected 'output' structure: {output_data}")
                error_detail = json.dumps(result, indent=2)[:500]
                return None, f"Error: Unexpected response structure in 'output'.\nDetails:\n{error_detail}"

            # Strip markdown fences if present
            generated_text = generated_text.strip()
            if generated_text.startswith("```json"):
                generated_text = generated_text[7:]
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]
            generated_text = generated_text.strip()

            # Check for common error messages from the server side
            if "error" in generated_text.lower() and len(generated_text)<200: # Crude check
                print(f"⚠️ Server might have returned an error message: {generated_text}")
                return None, f"Server reported an error: {generated_text}"

            return generated_text, None
        else:
            print(f"⚠️ Unexpected overall response structure: {result}")
            error_detail = json.dumps(result, indent=2)[:500]
            return None, f"Error: Unexpected response structure from server.\nDetails:\n{error_detail}"

    except requests.exceptions.Timeout:
        print("❌ Request timed out.")
        return None, "Error: Request timed out. The server might be busy or the task is too long (increase timeout?)."
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return None, f"Error: Could not connect to the LOLLMS server at {BASE_URL}. Is it running and accessible?"
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        error_msg = f"Error: API request failed.\nStatus Code: {e.response.status_code if e.response else 'N/A'}\n"
        if e.response is not None:
            try: error_msg += f"Server Response: {e.response.text[:500]}"
            except Exception: error_msg += "Server Response: (Could not decode)"
        return None, error_msg
    except Exception as e:
        print(f"❌ An unexpected error occurred during API call: {e}")
        traceback.print_exc()
        return None, f"An unexpected error occurred during API call: {e}"

# --- Gradio Interface Functions ---

def generate_prompt_only(source_content: str) -> str:
    """
    Takes source content and generates the full prompt string using .replace()
    without calling the API.
    """
    if not source_content or not source_content.strip():
        return "Error: Source content cannot be empty."

    print("📝 Formatting prompt using .replace()...")
    placeholder = "{user_provided_content_here}"
    # Use simple string replacement
    full_prompt = TRAINING_DATA_GENERATION_PROMPT_TEMPLATE.replace(
        placeholder,
        source_content
    )
    print("✅ Prompt formatted.")
    return full_prompt


def process_and_generate(source_content: str) -> tuple[dict | str, str | None]:
    """
    Takes source content, formats prompt using .replace(), calls API,
    validates JSON, returns results via gr.update for specific components.
    """
    if config_needed:
        return gr.update(value={"error": "Configuration needed. Please save settings first."}), gr.update(value=None, interactive=False)

    if not source_content or not source_content.strip():
        # Return error in the JSON output area, clear download link
        return gr.update(value={"error": "Source content cannot be empty."}), gr.update(value=None, interactive=False)

    print("⏳ Formatting prompt using .replace()...")
    placeholder = "{user_provided_content_here}"
    # Use simple string replacement
    full_prompt = TRAINING_DATA_GENERATION_PROMPT_TEMPLATE.replace(
        placeholder,
        source_content
    )

    generated_json_str, error_msg = generate_data_via_api(full_prompt)

    if error_msg:
        # Display error in JSON area, clear download link
        return gr.update(value={"error": error_msg}), gr.update(value=None, interactive=False)

    if not generated_json_str:
         # Display error in JSON area, clear download link
         return gr.update(value={"error": "Received empty response from the server."}), gr.update(value=None, interactive=False)

    print("🔄 Validating JSON response...")
    try:
        # Attempt to fix potential markdown fences again just in case
        if generated_json_str.strip().startswith("```json"):
             generated_json_str = generated_json_str.strip()[7:]
        if generated_json_str.strip().endswith("```"):
             generated_json_str = generated_json_str.strip()[:-3]
        generated_json_str = generated_json_str.strip()

        # Basic check if it looks like JSON before parsing
        if not (generated_json_str.startswith('[') and generated_json_str.endswith(']')) and \
           not (generated_json_str.startswith('{') and generated_json_str.endswith('}')):
             print(f"❌ Received content doesn't look like JSON: {generated_json_str[:100]}...")
             # Use the raw string in the error detail for better debugging
             raise json.JSONDecodeError("Output does not look like valid JSON (missing brackets/braces?).", generated_json_str, 0)

        parsed_json = json.loads(generated_json_str)

        # Basic structure validation
        if not isinstance(parsed_json, list):
             # Display warning in JSON area, clear download link
             return gr.update(value={"warning": "Generated output is not a JSON list as expected.", "raw_output": generated_json_str}), gr.update(value=None, interactive=False)
        if parsed_json and (not isinstance(parsed_json[0], dict) or "system_prompt" not in parsed_json[0] or "discussion" not in parsed_json[0]):
            # Display warning in JSON area, still allow download
            pass # Allow download even with potential structure issues

        print("✅ JSON validation successful.")
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_data_{timestamp}.json"
            # Use the globally configured OUTPUT_DIR
            # Ensure OUTPUT_DIR exists (might not if config failed but wizard was skipped somehow)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Use a temporary file first to ensure write is successful before returning path
            # Specify dir to ensure it's saved in the intended output directory
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', dir=OUTPUT_DIR, suffix=".json", prefix="temp_") as temp_f:
                json.dump(parsed_json, temp_f, indent=2, ensure_ascii=False)
                temp_filepath = Path(temp_f.name)

            # Rename temporary file to final filename
            final_filepath = OUTPUT_DIR / filename
            # Ensure the target directory exists before renaming
            final_filepath.parent.mkdir(parents=True, exist_ok=True)
            temp_filepath.rename(final_filepath)

            print(f"💾 Saved generated data to {final_filepath}")
            # Return JSON data and the path to the saved file for download
            return gr.update(value=parsed_json), gr.update(value=str(final_filepath), interactive=True)

        except Exception as e:
             print(f"❌ Error saving file: {e}")
             traceback.print_exc()
             # Show JSON data but indicate save error, disable download
             return gr.update(value={"warning": f"Could not save file: {e}", "data": parsed_json}), gr.update(value=None, interactive=False)

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON received: {e}")
        # Include the raw string in the error message for debugging
        error_detail = f"Error: The server returned text that is not valid JSON.\nDetails: {e}\n\nRaw Output (first 1000 chars):\n{generated_json_str[:1000]}..."
        # Display error in JSON area, clear download link
        return gr.update(value={"error": error_detail, "raw_output": generated_json_str}), gr.update(value=None, interactive=False)
    except Exception as e:
        print(f"❌ Error processing response: {e}")
        traceback.print_exc()
         # Display error in JSON area, clear download link
        return gr.update(value={"error": f"Error processing response: {e}"}), gr.update(value=None, interactive=False)


def save_config_and_proceed(url: str, key: str, out_dir: str):
    """
    Validates input, saves config, reloads globals, and updates UI visibility.
    Returns dictionary of gr.update calls.
    """
    global config_needed

    url = url.strip().rstrip('/')
    key = key.strip() if key else None # Store None if empty string
    out_dir = out_dir.strip()

    # Basic validation
    if not url or not url.startswith(('http://', 'https://')):
        # Return only the update for the status message within the wizard group
        return { wizard_status_message: gr.update(value="Error: Invalid Base URL. Must start with http:// or https://", visible=True) }

    if not out_dir:
         out_dir_path = DEFAULT_OUTPUT_DIR # Use default Path object if empty
    else:
         out_dir_path = Path(out_dir)

    config_data = {
        "lollms_base_url": url,
        "lollms_api_key": key,
        "output_dir": str(out_dir_path) # Save path as string in JSON
    }

    try:
        # Ensure parent directory exists for config file
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving configuration to {CONFIG_FILE}...")
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        # Reload the configuration into global variables
        load_success = load_config()

        if load_success:
            print("✅ Configuration saved and loaded.")
            # Update UI: Hide wizard, show main app
            return {
                wizard_group: gr.update(visible=False),
                main_app_group: gr.update(visible=True),
                wizard_status_message: gr.update(value="Configuration saved!", visible=True),
                # Update status bar in main app view as well
                status_server_url: gr.update(value=f"Using LOLLMS Server at: `{BASE_URL}`"),
                status_api_key: gr.update(value=f"API Key: {'Set' if API_KEY else 'Not Set'}")
            }
        else:
            # This case should ideally not happen if save succeeded, but handle defensively
             print("❌ Configuration saved, but failed to reload.")
             return {
                 wizard_status_message: gr.update(value="Error: Config saved but failed to reload. Check console.", visible=True)
             }

    except IOError as e:
        print(f"❌ Error saving configuration file: {e}")
        traceback.print_exc()
        return {
            wizard_status_message: gr.update(value=f"Error saving config file: {e}", visible=True)
        }
    except Exception as e:
        print(f"❌ Unexpected error saving config: {e}")
        traceback.print_exc()
        return {
             wizard_status_message: gr.update(value=f"Unexpected error saving config: {e}", visible=True)
        }

# --- Initial Config Load Attempt ---
load_config() # Try loading config file at the start

# --- Build Gradio App ---
print("🛠️ Building Gradio interface...")

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 LOLLMS Training Data Builder")

    # --- Configuration Wizard UI (Visible only if config_needed) ---
    with gr.Column(visible=config_needed) as wizard_group:
        gr.Markdown("## ⚙️ Initial Configuration Needed")
        gr.Markdown(f"Please provide your `lollms-server` connection details. This will be saved to `{CONFIG_FILE}`.")
        wizard_url_input = gr.Textbox(label="LOLLMS Server Base URL", placeholder="http://localhost:9600", value="http://localhost:9600")
        wizard_api_key_input = gr.Textbox(label="LOLLMS API Key (Optional)", placeholder="Leave blank if no API key is set", type="password")
        wizard_output_dir_input = gr.Textbox(label="Output Directory for JSON files", placeholder=str(DEFAULT_OUTPUT_DIR), value=str(DEFAULT_OUTPUT_DIR))
        wizard_save_button = gr.Button("Save Configuration and Continue", variant="primary")
        wizard_status_message = gr.Markdown(visible=False) # For feedback

    # --- Main Application UI (Visible only if config NOT needed) ---
    with gr.Column(visible=not config_needed) as main_app_group:
        gr.Markdown(
            "Paste your source documentation or data below. The AI will generate discussion examples "
            "based *only* on this content, formatted as JSON for LLM training."
            " You can either generate the full prompt to use elsewhere, or send it to the configured LOLLMS server."
        )
        with gr.Row():
            # Input Column
            with gr.Column(scale=2):
                source_text_input = gr.Textbox(
                    lines=15,
                    label="Source Content",
                    placeholder="Paste your documentation, article, or data here..."
                )
                with gr.Row():
                    generate_prompt_button = gr.Button("📜 Generate Prompt Only")
                    generate_button = gr.Button("✨ Generate & Save Data via API", variant="primary")

            # Output Column
            with gr.Column(scale=3):
                gr.Markdown("---")
                gr.Markdown("### Generated Prompt")
                with gr.Row():
                    prompt_display_textbox = gr.Textbox(
                        label="Generated Prompt (for manual use)",
                        lines=8,
                        interactive=False, # Keep it read-only
                        scale=10
                    )
                    # Use a regular button for JS copy action
                    copy_prompt_js_button = gr.Button("📄 Copy Prompt", scale=1)

                gr.Markdown("---")
                gr.Markdown("### Generated JSON Data (from API)")
                json_output = gr.JSON(label="API Output")
                download_button = gr.File(label="Download Generated JSON", interactive=False)

        gr.Markdown("---")
        # Status indicators using current global values (updated by wizard/load)
        status_server_url = gr.Markdown(f"Using LOLLMS Server at: `{BASE_URL}`")
        status_api_key = gr.Markdown(f"API Key: {'Set' if API_KEY else 'Not Set'}")

    # --- Define JavaScript for Copy Button ---
    copy_js_code = """
    async (text_to_copy) => {
        if (!text_to_copy) {
            // Maybe show a small temporary message instead of alert
            // console.log("Nothing to copy!");
            return;
        }
        try {
            await navigator.clipboard.writeText(text_to_copy);
            // Simple feedback: Maybe briefly change button text or style (more complex)
             alert('Prompt copied to clipboard!'); // Keeping alert for simplicity
        } catch (err) {
            console.error('Failed to copy text: ', err);
            alert('Failed to copy prompt. Check browser console (F12). Requires HTTPS or localhost.');
        }
    }
    """

    # --- Connect Components ---

    # Wizard save action
    # Output dictionary maps component variables to gr.update() calls
    # Make sure the variable names here match the component variable names defined above
    wizard_save_button.click(
        fn=save_config_and_proceed,
        inputs=[wizard_url_input, wizard_api_key_input, wizard_output_dir_input],
        outputs=[wizard_group, main_app_group, wizard_status_message, status_server_url, status_api_key]
    )

    # Action to generate only the prompt text
    generate_prompt_button.click(
        fn=generate_prompt_only,
        inputs=[source_text_input],
        outputs=[prompt_display_textbox] # Output to the prompt display box
    )

    # Connect the regular button to trigger the JavaScript copy function
    copy_prompt_js_button.click(
        fn=None,  # No Python function needed on the server side for copying
        inputs=[prompt_display_textbox], # Pass the content of the textbox to the JS function
        outputs=None, # No Gradio outputs are updated by this action
        js=copy_js_code # The JavaScript code to execute on click in the browser
    )

    # Main app action: generate data via API
    generate_button.click(
        fn=process_and_generate,
        inputs=[source_text_input],
        outputs=[json_output, download_button], # Output to JSON display and Download button
        api_name="generate_training_data" # Optional: for API usage if needed
    )


print("🚀 Launching Gradio app...")
if __name__ == "__main__":
    # share=True allows access over network, remove if only local access needed
    # Set server_name to allow connections from other devices on the network
    # Use environment variable for port or default, e.g., 7860
    server_port = int(os.environ.get("GRADIO_PORT", 7860))
    try:
        app.launch(server_name="0.0.0.0", server_port=server_port)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"❌ ERROR: Port {server_port} is already in use.")
            print("   Please close the other application using this port or set a different port using the GRADIO_PORT environment variable.")
            print("   Example: `set GRADIO_PORT=7861` (Windows Cmd) or `export GRADIO_PORT=7861` (Linux/macOS) then run the script again.")
        else:
            print(f"❌ ERROR launching Gradio: {e}")
            traceback.print_exc()
        sys.exit(1) # Exit if launch fails critically
    except Exception as e:
        print(f"❌ An unexpected error occurred during launch: {e}")
        traceback.print_exc()
        sys.exit(1)