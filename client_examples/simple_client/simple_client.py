# client_examples/simple_client_menu.py
# MODIFIED VERSION using ascii_colors.Menu

import requests
import json
import sys
import os
import base64
import time
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

# --- Dependency Check ---
try:
    import pipmaster as pm

    pm.install_if_missing("requests")
    pm.install_if_missing("sseclient-py")
    pm.install_if_missing("Pillow")
    pm.install_if_missing("ascii_colors>=0.11.2") # Ensure new enough version for Menu
    print("Core dependencies checked/installed.")
    from PIL import Image
    from io import BytesIO
    from sseclient import SSEClient  # type: ignore
    # --- Import Menu and ASCIIColors ---
    from ascii_colors import ASCIIColors, Menu, trace_exception

    # Basic logging setup using ascii_colors
    import ascii_colors as logging
    logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')
    logger = logging.getLogger("LOLLMSClient")
    # --- End Import ---

except ImportError as e:
    print(f"CRITICAL ERROR: Missing core library ({e}).")
    print(
        "Please install required packages manually: pip install requests sseclient-py Pillow ascii_colors>=0.11.2"
    )
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during dependency check: {e}")
    print(
        "Please ensure required packages are installed: pip install requests sseclient-py Pillow ascii_colors>=0.11.2"
    )
    sys.exit(1)


# --- Configuration ---
BASE_URL = "http://localhost:9601/api/v1"  # Default, can be overridden
DEFAULT_TIMEOUT = 120
DEFAULT_TTT_BINDING = None
DEFAULT_TTT_MODEL = None
DEFAULT_TTI_BINDING = None
DEFAULT_TTI_MODEL = None
HISTORY_FILE = Path("chat_history.json")
SETTINGS_FILE = Path("settings.json")
IMAGE_DIR = Path("generated_images")

# --- Application State ---
API_KEY: Optional[str] = None  # Global variable to hold the current API key
current_personality: Optional[str] = None
current_ttt_binding: Optional[str] = None
current_ttt_model: Optional[str] = None
current_tti_binding: Optional[str] = DEFAULT_TTI_BINDING
current_tti_model: Optional[str] = DEFAULT_TTI_MODEL


# --- Helper Functions (using ASCIIColors for printing) ---
def print_system(message):
    """Prints system messages."""
    ASCIIColors.cyan(f"\n🤖 SYSTEM: {message}")


def print_ai_prefix(personality_name: Optional[str]):
    """Prints the AI prefix."""
    prefix = f"💡 ({personality_name or 'Default'}) AI: "
    ASCIIColors.print(prefix, end="", flush=True, color=ASCIIColors.color_bright_cyan)


def print_user(message):
    """Prints user input representation."""
    ASCIIColors.yellow(f"\n👤 YOU: {message}")


def print_ai_history(message):
    """Prints AI messages from history."""
    ASCIIColors.print(f"\n💡 AI: {message}", color=ASCIIColors.color_green)


def print_error(message):
    """Prints error messages."""
    ASCIIColors.error(f"\n❌ ERROR: {message}") # Use built-in error


def print_warning(message):
    """Prints warning messages."""
    ASCIIColors.warning(f"\n⚠️ WARNING: {message}") # Use built-in warning


def load_json_file(filepath: Path, default: Any = None) -> Any:
    """Loads JSON data from a file, returning default on error."""
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print_error(f"Failed to decode JSON file '{filepath}'. Using defaults.")
            return default
        except Exception as e:
            print_error(f"Error loading file '{filepath}': {e}")
            trace_exception(e)
            return default
    return default


def save_json_file(filepath: Path, data: Any):
    """Saves data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print_error(f"Error saving file '{filepath}': {e}")
        trace_exception(e)


def load_history() -> List[Dict[str, str]]:
    """Loads chat history, ensuring it's a list."""
    history = load_json_file(HISTORY_FILE, default=[])
    return history if isinstance(history, list) else []


def save_history(history: List[Dict[str, str]]):
    """Saves chat history."""
    save_json_file(HISTORY_FILE, history)


# --- Updated Settings Functions ---
def load_settings() -> Dict[str, Optional[str]]:
    """Loads application settings, including API key."""
    defaults = {
        "base_url": BASE_URL,
        "api_key": None,
        "personality": None,
        "ttt_binding": None,
        "ttt_model": None,
        "tti_binding": DEFAULT_TTI_BINDING,
        "tti_model": DEFAULT_TTI_MODEL,
    }
    loaded = load_json_file(SETTINGS_FILE, default={})
    if not isinstance(loaded, dict):
        print_error(f"Settings file '{SETTINGS_FILE}' has invalid format. Using defaults.")
        loaded = {}
    defaults.update(loaded) # Merge loaded settings over defaults
    return defaults


def save_settings(settings: Dict[str, Optional[str]]):
    """Saves application settings, including API key."""
    save_json_file(SETTINGS_FILE, settings)


def get_current_settings() -> Dict[str, Optional[str]]:
    """Returns a dictionary of the current in-memory settings."""
    return {
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "personality": current_personality,
        "ttt_binding": current_ttt_binding,
        "ttt_model": current_ttt_model,
        "tti_binding": current_tti_binding,
        "tti_model": current_tti_model,
    }


def update_current_settings(settings: Dict[str, Optional[str]]):
    """Updates the global state variables from settings."""
    global current_personality, current_ttt_binding, current_ttt_model
    global current_tti_binding, current_tti_model
    global API_KEY, BASE_URL

    BASE_URL = settings.get("base_url", BASE_URL)
    API_KEY = settings.get("api_key")
    current_personality = settings.get("personality")
    current_ttt_binding = settings.get("ttt_binding")
    current_ttt_model = settings.get("ttt_model")
    current_tti_binding = settings.get("tti_binding", DEFAULT_TTI_BINDING)
    current_tti_model = settings.get("tti_model", DEFAULT_TTI_MODEL)
    print_system("Settings loaded/updated in memory.")


# --- End Updated Settings Functions ---


def save_base64_image(b64_string: str, prompt: str) -> Optional[Path]:
    """Decodes base64 string and saves as PNG image."""
    try:
        img_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(img_data))
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(
            c for c in prompt if c.isalnum() or c in (" ", "_")
        ).rstrip()
        safe_prompt_short = (
            safe_prompt[:30].replace(" ", "_") if safe_prompt else "image"
        )
        filename = f"{timestamp}_{safe_prompt_short}.png"
        output_path = IMAGE_DIR / filename
        img.save(output_path, "PNG")
        print_system(f"Image saved successfully to: {output_path}")
        return output_path
    except base64.binascii.Error as e:
        print_error(f"Error decoding base64 string: {e}")
    except Exception as e:
        print_error(f"Error saving image: {e}")
        trace_exception(e)
    return None


# --- Updated API Call Functions ---
def build_headers(stream: bool) -> Dict[str, str]:
    """Builds request headers, adding API key if set."""
    headers = {"Content-Type": "application/json"}
    if stream:
        headers["Accept"] = "text/event-stream"
    else:
        headers["Accept"] = "application/json"
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def make_api_call(
    endpoint: str, method: str = "GET", payload: Optional[Dict] = None
) -> Optional[Any]:
    """Generic function to make non-streaming API calls."""
    url = f"{BASE_URL}{endpoint}"
    headers = build_headers(stream=False)
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(
                url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT
            )
        else:
            print_error(f"Unsupported HTTP method: {method}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print_error(f"API call failed to {url}: {e}")
        if hasattr(e, "response") and e.response is not None:
            status = e.response.status_code
            body = e.response.text
            detail = f"Status Code: {status}"
            try:
                detail += f", Detail: {e.response.json().get('detail', body)}"
            except json.JSONDecodeError:
                detail += f", Body: {body[:200]}..."
            print(f"  Server Response: {detail}")
        return None
    except Exception as e:
        print_error(f"Unexpected error during API call to {url}: {e}")
        trace_exception(e)
        return None


def make_generate_request(
    payload: Dict[str, Any], stream: bool
) -> Optional[Union[str, Dict, SSEClient]]:
    """Makes a request specifically to the /generate endpoint."""
    url = f"{BASE_URL}/generate"
    headers = build_headers(stream=stream)
    timeout = DEFAULT_TIMEOUT

    # Validate input_data structure
    if (
        "input_data" not in payload
        or not isinstance(payload["input_data"], list)
        or not payload["input_data"]
    ):
        if "prompt" in payload and isinstance(payload["prompt"], str):
            print_warning("Converting legacy 'prompt' field to 'input_data'.")
            payload["input_data"] = [
                {"type": "text", "role": "user_prompt", "data": payload["prompt"]}
            ]
            del payload["prompt"]
        else:
            print_error("Generate request payload missing valid 'input_data' list.")
            return None

    payload["stream"] = stream

    try:
        response = requests.post(
            url, headers=headers, json=payload, stream=stream, timeout=timeout
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if stream:
            if "text/event-stream" in content_type:
                return SSEClient(response)
            elif "text/plain" in content_type:
                print_system("Warning: Server returned plain text instead of stream.")
                return response.text
            else:
                print_error(
                    f"Server returned unexpected Content-Type '{content_type}' for streaming."
                )
                print(f"Body: {response.text[:500]}")
                return None
        else:
            if "application/json" in content_type:
                return response.json()
            else:
                print_system(f"Warning: Expected JSON but got {content_type}")
                return response.text
    except requests.exceptions.Timeout:
        print_error(f"Request timed out after {timeout} seconds to {url}")
    except requests.exceptions.ConnectionError as e:
        print_error(f"Could not connect to server at {url}. Is it running? Details: {e}")
    except requests.exceptions.RequestException as e:
        print_error(f"Generate request failed to {url}: {e}")
        if hasattr(e, "response") and e.response is not None:
            status = e.response.status_code
            body = e.response.text
            detail = f"Status Code: {status}"
            try:
                error_json = e.response.json()
                detail += f", Detail: {error_json.get('detail', body)}"
            except json.JSONDecodeError:
                detail += f", Body: {body[:200]}..."
            print(f"  Server Response: {detail}")
            if status in [401, 403]:
                print_warning("Authentication failed. Check API key in settings.json.")
    except Exception as e:
        print_error(f"An unexpected error occurred during generate request: {e}")
        trace_exception(e)
    return None


# --- End Updated API Call Functions ---


# --- Menu Navigation (using ASCIIColors.Menu) ---
def select_item_from_api_list(
    item_type: str,
    fetch_endpoint: str,
    current_value: Optional[str],
    display_key: str = "name",
    value_key: str = "name",
    list_json_key: Optional[str] = None,
    is_dict_list: bool = True,
    allow_none: bool = True,
    none_display_text: str = "Use Server Default",
) -> Optional[str]:
    """Fetches items, displays an interactive menu, returns selected value."""
    print_system(f"Fetching available {item_type}s...")
    data = make_api_call(fetch_endpoint)
    items = []

    if data:
        if list_json_key:
            raw_list_data = data.get(list_json_key)
            if is_dict_list and isinstance(raw_list_data, dict):
                items = list(raw_list_data.values())
            elif not is_dict_list and isinstance(raw_list_data, dict):
                items = list(raw_list_data.keys())
            elif isinstance(raw_list_data, list):
                items = raw_list_data
            else:
                print_warning(f"Could not find/parse key '{list_json_key}' in API response.")
        elif isinstance(data, list):
            items = data
        else:
            print_warning(f"Unexpected API response format for {item_type}s.")
    else:
        print_error(f"Failed to fetch {item_type}s.")
        return current_value # Keep current on fetch failure

    if not items:
        print_system(f"No {item_type}s available.")
        return None if allow_none else current_value

    # --- Build Menu ---
    menu = Menu(
        title=f"Select {item_type}",
        mode="select_single",
        enable_filtering=True, # Useful for long lists
        item_color=ASCIIColors.color_cyan,
        selected_background=ASCIIColors.color_bg_blue,
        title_color=ASCIIColors.color_bright_yellow
    )

    # Add the 'None' option first if allowed
    if allow_none:
        none_marker = " [*]" if current_value is None else ""
        menu.add_choice(text=f"{none_display_text}{none_marker}", value=None)

    # Add items from API
    valid_items_added = False
    for item in items:
        display_val = None
        actual_val = None
        if is_dict_list:
            if isinstance(item, dict):
                display_val = item.get(display_key)
                actual_val = item.get(value_key)
        elif isinstance(item, str):
            display_val = item
            actual_val = item

        if display_val and actual_val:
            marker = " [*]" if actual_val == current_value else ""
            menu.add_choice(text=f"{display_val}{marker}", value=actual_val)
            valid_items_added = True
        else:
            print_warning(f"Skipping invalid item in {item_type} list: {item}")

    if not valid_items_added and not allow_none:
        print_system(f"No valid {item_type}s could be listed.")
        return current_value

    # --- Run Menu ---
    selected_value = menu.run()

    # --- Process Result ---
    if selected_value is None and allow_none: # If 'None' option was chosen or cancelled
        if menu.last_selected_index == 0 and allow_none: # Explicitly chose the 'None' option
            print_system(f"{item_type} set to: {none_display_text}")
            return None
        else: # Cancelled with Ctrl+C or Esc
            print_system("Selection cancelled, keeping current value.")
            return current_value
    elif selected_value is not None: # A valid item was chosen
        print_system(f"{item_type} set to: {selected_value}")
        return selected_value
    else: # Cancelled when 'None' wasn't allowed
        print_system("Selection cancelled, keeping current value.")
        return current_value

# --- Menu Handlers (use the refactored select_item_from_api_list) ---
def handle_personality_menu():
    global current_personality
    current_personality = select_item_from_api_list(
        item_type="Personality",
        fetch_endpoint="/list_personalities",
        current_value=current_personality,
        list_json_key="personalities",
        is_dict_list=True,
        allow_none=True,
        none_display_text="None (No Personality)",
    )

def handle_binding_menu():
    global current_ttt_binding, current_ttt_model
    old_binding = current_ttt_binding
    new_binding = select_item_from_api_list(
        item_type="TTT Binding",
        fetch_endpoint="/list_bindings",
        current_value=current_ttt_binding,
        list_json_key="binding_instances",
        is_dict_list=False, # Keys are instance names
        allow_none=True,
        none_display_text="Use Server Default Binding",
    )
    if new_binding != old_binding:
        print_system(
            f"Binding changed from '{old_binding or 'Default'}' to '{new_binding or 'Default'}'. Resetting model."
        )
        current_ttt_binding = new_binding
        current_ttt_model = None

def handle_model_menu():
    global current_ttt_model
    if current_ttt_binding is None:
        print_system("Cannot select model: Server Default Binding active. Select specific TTT Binding first.")
        return
    current_ttt_model = select_item_from_api_list(
        item_type="TTT Model",
        fetch_endpoint=f"/list_available_models/{current_ttt_binding}",
        current_value=current_ttt_model,
        list_json_key="models",
        is_dict_list=True,
        allow_none=True,
        none_display_text=f"Use Default for '{current_ttt_binding}'",
    )

# --- Display and Server/API Key Settings (use ASCIIColors.prompt) ---
def display_current_settings():
    """Displays the currently active settings."""
    settings = get_current_settings()
    ASCIIColors.print("\n--- Current Settings ---", color=ASCIIColors.color_bright_cyan)
    ASCIIColors.print(f"  Server URL:  {settings['base_url']}")
    ASCIIColors.print(f"  API Key:     {'Set (********)' if settings['api_key'] else 'Not Set'}")
    ASCIIColors.print(f"  Personality: {settings['personality'] or 'None'}")
    ASCIIColors.print(f"  TTT Binding: {settings['ttt_binding'] or 'Default'}")
    ASCIIColors.print(f"  TTT Model:   {settings['ttt_model'] or 'Default'}")
    ASCIIColors.print(f"  TTI Binding: {settings['tti_binding'] or 'Default'}")
    ASCIIColors.print(f"  TTI Model:   {settings['tti_model'] or 'Default'}")
    ASCIIColors.print("-" * 24)
    ASCIIColors.prompt("Press Enter to continue...") # Use prompt to wait

def handle_server_url_setting():
    """Allows user to change the server URL."""
    global BASE_URL
    current_url = BASE_URL
    new_url = ASCIIColors.prompt(
        f"Current Server URL: {current_url}\nEnter new server URL (or Enter to keep current): ",
        color=ASCIIColors.color_yellow
    ).strip()

    if new_url:
        if new_url.lower().startswith("http://") or new_url.lower().startswith("https://"):
            BASE_URL = new_url.rstrip('/')
            print_system(f"Server URL set to: {BASE_URL}")
        else:
            print_error("Invalid URL format. Must start with http:// or https://.")
    else:
        print_system("Server URL unchanged.")

def handle_api_key_setting():
    """Allows user to set or clear the API key."""
    global API_KEY
    current_key_status = 'Set (********)' if API_KEY else 'Not Set'
    new_key = ASCIIColors.prompt(
        f"Current API Key: {current_key_status}\nEnter new API Key (or leave blank to clear): ",
        color=ASCIIColors.color_yellow,
        hide_input=True # Use hidden input for keys
    ).strip()

    if new_key:
        API_KEY = new_key
        print_system("API Key set.")
    else:
        API_KEY = None
        print_system("API Key cleared.")

# --- Refactored Settings Menu Loop (using ASCIIColors.Menu) ---
def run_settings_menu():
    """Runs the main settings menu loop."""
    while True:
        settings = get_current_settings() # Get fresh settings each loop iteration
        settings_menu = Menu(
            title="Settings",
            item_color=ASCIIColors.color_green,
            selected_background=ASCIIColors.color_bg_blue,
            title_color=ASCIIColors.color_bright_yellow
        )

        settings_menu.add_action(f"Server URL    : ({settings['base_url']})", handle_server_url_setting)
        settings_menu.add_action(f"API Key       : ({'Set' if settings['api_key'] else 'Not Set'})", handle_api_key_setting)
        settings_menu.add_action(f"Personality   : ({settings['personality'] or 'None'})", handle_personality_menu)
        settings_menu.add_action(f"TTT Binding   : ({settings['ttt_binding'] or 'Server Default'})", handle_binding_menu)
        settings_menu.add_action(f"TTT Model     : ({settings['ttt_model'] or 'Server Default'})", handle_model_menu)
        # Add TTI Binding/Model later if needed
        # settings_menu.add_action(f"TTI Binding   : ({settings['tti_binding'] or 'Server Default'})", handle_tti_binding_menu)
        # settings_menu.add_action(f"TTI Model     : ({settings['tti_model'] or 'Server Default'})", handle_tti_model_menu)
        settings_menu.add_separator()
        settings_menu.add_action("Show Current Settings", display_current_settings)
        settings_menu.add_action("Save & Back", lambda: "save_back") # Action returns a value

        selected_action_result = settings_menu.run()

        if selected_action_result == "save_back" or selected_action_result is None: # User selected Save/Back or Quit (Ctrl+C)
            break # Exit the settings loop

    save_settings(get_current_settings()) # Save all current settings on exit
    print_system("Settings saved.")

# --- Main Application Logic ---
if __name__ == "__main__":
    chat_history = load_history()
    saved_settings = load_settings()
    update_current_settings(saved_settings) # Load settings into globals

    # Health Check and API Key Handling
    print_system("Checking server status...")
    health_info = make_api_call("/health")
    if not health_info:
        print_error("Could not reach server. Please ensure it's running and the URL is correct.")
        print_error(f"Current URL: {BASE_URL}")
        sys.exit(1)

    server_version = health_info.get("version", "Unknown")
    key_required_by_server = health_info.get("api_key_required", True)
    print_system(f"Server OK (Version: {server_version}). API Key Required: {key_required_by_server}")

    if key_required_by_server:
        if API_KEY:
            print_system("Using API key found in settings.json.")
        else:
            print_warning("Server requires an API key, but none is set in settings.json.")
            while not API_KEY:
                try:
                    user_key = ASCIIColors.prompt(
                        "Enter API Key for the server: ",
                        color=ASCIIColors.color_yellow,
                        hide_input=True
                    ).strip()
                    if user_key:
                        API_KEY = user_key
                        print_system("API Key accepted.")
                        current_settings = get_current_settings()
                        save_settings(current_settings)
                        print_system("API Key saved to settings.json.")
                        break
                    else:
                        print_warning("API Key cannot be empty.")
                except (EOFError, KeyboardInterrupt):
                    print_error("\nAPI Key entry cancelled. Cannot proceed without API Key.")
                    sys.exit(1)
    else:
        if API_KEY:
            print_warning("Server does not require API key, but one is set locally. Clearing.")
            API_KEY = None
        else:
            print_system("Server does not require an API key.")

    # Welcome and History
    print_system("Welcome to LOLLMS Console Chat!")
    print_system("Type '/imagine [prompt]' to generate an image.")
    print_system("Type '/settings' to change URL/Key/Personality/Model.")
    print_system("Type '/quit' or '/exit' to end the chat.")
    print_system("-" * 30)
    if chat_history:
        print_system("Resuming previous conversation:")
        for entry in chat_history:
            if entry.get("role") == "user":
                print_user(entry.get("content", ""))
            elif entry.get("role") == "assistant":
                print_ai_history(entry.get("content", ""))
            elif entry.get("role") == "system":
                print_system(entry.get("content", ""))
        print_system("-" * 30)

    # --- Main Chat Loop ---
    while True:
        try:
            user_input = ASCIIColors.prompt(
                f"\n👤 YOU: ", color=ASCIIColors.color_bright_green
            ).strip()
        except EOFError:
            print_system("Exiting...")
            break
        except KeyboardInterrupt:
            print_system("\nExiting...")
            break
        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit"]:
            print_system("Saving state and exiting...")
            break
        elif user_input.lower() == "/settings":
            run_settings_menu()
            continue

        chat_history.append({"role": "user", "content": user_input})

        # --- Image Generation Logic ---
        if user_input.lower().startswith("/imagine "):
            image_prompt = user_input[len("/imagine ") :].strip()
            if not image_prompt:
                print_system("Please provide a prompt after /imagine.")
                chat_history.append(
                    {"role": "system", "content": "Image generation skipped (no prompt)."}
                )
                continue

            print_system(f"Generating image for: '{image_prompt}'...")
            payload = {
                "input_data": [{"type": "text", "role": "user_prompt", "data": image_prompt}],
                "generation_type": "tti",
                "binding_name": current_tti_binding,
                "model_name": current_tti_model,
                "stream": False,
            }
            response_data = make_generate_request(payload, stream=False)

            if response_data and isinstance(response_data, dict) and "output" in response_data:
                output_list = response_data.get("output", [])
                image_found = False
                for item in output_list:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "image"
                        and item.get("data")
                    ):
                        image_b64 = item["data"]
                        metadata = item.get("metadata", {})
                        prompt_used = metadata.get("prompt_used", image_prompt)
                        saved_path = save_base64_image(image_b64, prompt_used)
                        msg = (
                            f"Image generated and saved to {saved_path.name}"
                            if saved_path
                            else "Image generation succeeded but saving failed."
                        )
                        chat_history.append({"role": "system", "content": msg})
                        image_found = True
                        break
                    elif isinstance(item, dict) and item.get("type") == "error":
                        error_msg = item.get("data", "Unknown error from server.")
                        print_error(f"Image generation failed: {error_msg}")
                        chat_history.append(
                            {"role": "system", "content": f"Image generation failed: {error_msg}"}
                        )
                        image_found = True
                        break
                if not image_found:
                    print_error("Response did not contain valid image data.")
                    chat_history.append(
                        {"role": "system", "content": "Image generation failed: No image data in response."}
                    )
            elif response_data:
                print_error(
                    f"Received unexpected response format for image generation:\n{response_data}"
                )
                chat_history.append(
                    {"role": "system", "content": "Image generation failed (unexpected response format)."}
                )
            else:
                chat_history.append(
                    {"role": "system", "content": "Image generation request failed."}
                )
        else:
            # --- Text Generation Logic ---
            print_ai_prefix(current_personality)
            payload = {
                "input_data": [{"type": "text", "role": "user_prompt", "data": user_input}],
                "generation_type": "ttt",
                "personality": current_personality,
                "binding_name": current_ttt_binding,
                "model_name": current_ttt_model,
                "stream": True,
            }
            result = make_generate_request(payload, stream=True)

            if isinstance(result, str): # Handle non-stream fallback
                print(result)
                chat_history.append({"role": "assistant", "content": result})
            elif isinstance(result, SSEClient): # Handle stream
                sse_client = result
                full_ai_response = ""
                stream_error_occurred = False
                final_content_list = []
                try:
                    for event in sse_client.events():
                        if event.event == 'message' and event.data:
                            try:
                                chunk_data = json.loads(event.data)
                                chunk_type = chunk_data.get("type")
                                content = chunk_data.get("content")
                                metadata = chunk_data.get("metadata", {})

                                if chunk_type == "chunk" and content and isinstance(content, str):
                                    ASCIIColors.print(content, end="", flush=True, color=ASCIIColors.color_green)
                                    full_ai_response += content
                                elif chunk_type == "error":
                                    error_msg = f"\n--- Stream Error: {content} ---"
                                    ASCIIColors.print(error_msg, color=ASCIIColors.color_bright_red)
                                    if full_ai_response:
                                        chat_history.append({"role": "assistant", "content": full_ai_response})
                                    chat_history.append({"role": "system", "content": f"Stream error occurred: {content}"})
                                    full_ai_response = ""
                                    stream_error_occurred = True
                                    break
                                elif chunk_type == "info":
                                    ASCIIColors.print(f"\n--- Stream Info: {content} ---", color=ASCIIColors.color_blue, flush=True)
                                elif chunk_type == "final":
                                    final_content_list = content if isinstance(content, list) else []
                                    print() # Newline
                                    break
                            except json.JSONDecodeError:
                                ASCIIColors.print(f"\nError: Received non-JSON data: {event.data}", color=ASCIIColors.color_red)
                            except Exception as e:
                                ASCIIColors.print(f"\nError processing chunk: {e}", color=ASCIIColors.color_red)
                                trace_exception(e)
                    # After stream ends
                    if not stream_error_occurred:
                        final_text = ""
                        for item in final_content_list:
                            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("data"), str):
                                final_text += item["data"]
                            elif isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                                print_system("Received image data in final stream chunk.")
                                save_base64_image(item['data'], f"stream_image_{int(time.time())}")

                        final_response_to_save = final_text if final_text else full_ai_response
                        if final_response_to_save:
                            chat_history.append({"role": "assistant", "content": final_response_to_save.strip()})
                        elif not final_content_list:
                            print_warning("Stream finished without content.")
                            chat_history.append({"role": "system", "content": "(Stream finished without generating content)"})

                except requests.exceptions.ChunkedEncodingError:
                    print_error("Stream connection broken.")
                    if full_ai_response:
                        chat_history.append({"role": "assistant", "content": full_ai_response + " [Stream Interrupted]"})
                except Exception as e:
                    print_error(f"Error processing stream: {e}")
                    trace_exception(e)
                    if full_ai_response:
                        chat_history.append({"role": "assistant", "content": full_ai_response + " [Stream Error]"})
            else: # Handle non-stream, non-sse result (e.g., error)
                print()
                chat_history.append({"role": "system", "content": "Text generation request failed."})
        save_history(chat_history) # Save after each turn

    save_history(chat_history)
    save_settings(get_current_settings())
    print_system("Goodbye!")