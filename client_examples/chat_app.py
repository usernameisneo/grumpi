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
    print("Core dependencies checked/installed.")
    # Import PIL *after* ensuring it's installed
    from PIL import Image
    from io import BytesIO
    # Import sseclient *after* ensuring it's installed
    from sseclient import SSEClient # type: ignore
except ImportError:
    print("CRITICAL ERROR: pipmaster not found or failed.")
    print("Please install required packages manually:")
    print("  pip install requests sseclient-py Pillow")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during dependency check: {e}")
    print("Please ensure required packages are installed:")
    print("  pip install requests sseclient-py Pillow")
    sys.exit(1)


# --- Configuration ---
# Server settings
BASE_URL = "http://localhost:9600/api/v1"
API_KEY = "user1_key_abc123" # Use a valid key from your config.toml
DEFAULT_TIMEOUT = 120

# Default bindings/models (used if nothing selected/saved or if server default is desired)
DEFAULT_TTT_BINDING = None
DEFAULT_TTT_MODEL = None
DEFAULT_TTI_BINDING = None
DEFAULT_TTI_MODEL = None

# App settings
HISTORY_FILE = Path("chat_history.json")
SETTINGS_FILE = Path("settings.json")
IMAGE_DIR = Path("generated_images")
HEADERS_STREAM = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}
HEADERS_NO_STREAM = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# --- Application State (Global) ---
current_personality: Optional[str] = None
current_ttt_binding: Optional[str] = None
current_ttt_model: Optional[str] = None
# Use defaults initially, will be loaded from settings
current_tti_binding: Optional[str] = DEFAULT_TTI_BINDING
current_tti_model: Optional[str] = DEFAULT_TTI_MODEL

# --- Helper Functions ---

def print_system(message):
    """Prints system messages."""
    print(f"\n🤖 SYSTEM: {message}")

def print_ai_prefix(personality_name: Optional[str]):
    """Prints the AI prefix including the personality."""
    prefix = f"💡 ({personality_name or 'Default'}) AI: "
    print(prefix, end="", flush=True)

def print_user(message):
    """Prints user input representation for history display."""
    print(f"\n👤 YOU: {message}")

def print_ai_history(message):
    """Prints AI messages from history."""
    print(f"\n💡 AI: {message}")

def print_error(message):
    """Prints error messages."""
    print(f"\n❌ ERROR: {message}")

def print_warning(message):
    """Prints warning messages."""
    print(f"\n⚠️ WARNING: {message}")

def load_json_file(filepath: Path, default: Any = None) -> Any:
    """Loads JSON data from a file, returning default on any error."""
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print_error(f"Failed to decode JSON file '{filepath}'. Using defaults.")
            return default
        except Exception as e: # Catch other potential errors like permission issues
            print_error(f"Error loading file '{filepath}': {e}")
            return default
    return default

def save_json_file(filepath: Path, data: Any):
    """Saves data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print_error(f"Error saving file '{filepath}': {e}")

def load_history() -> List[Dict[str, str]]:
    """Loads chat history, ensuring it's always a list."""
    history = load_json_file(HISTORY_FILE, default=[])
    # Ensure the loaded data is actually a list, otherwise return default
    return history if isinstance(history, list) else []

def save_history(history: List[Dict[str, str]]):
    """Saves chat history."""
    save_json_file(HISTORY_FILE, history)

def load_settings() -> Dict[str, Optional[str]]:
    """Loads application settings, ensuring default keys exist."""
    defaults = {
        "personality": None,
        "ttt_binding": None,
        "ttt_model": None,
        "tti_binding": DEFAULT_TTI_BINDING, # Load TTI settings too
        "tti_model": DEFAULT_TTI_MODEL,
    }
    loaded = load_json_file(SETTINGS_FILE, default={}) # Load empty dict if file missing/bad
    # Ensure the loaded data is a dictionary
    if not isinstance(loaded, dict):
        print_error(f"Settings file '{SETTINGS_FILE}' has invalid format. Using defaults.")
        loaded = {}
    # Merge loaded settings with defaults to ensure all keys are present
    defaults.update(loaded)
    return defaults

def save_settings(settings: Dict[str, Optional[str]]):
    """Saves application settings."""
    save_json_file(SETTINGS_FILE, settings)

def get_current_settings() -> Dict[str, Optional[str]]:
    """Returns a dictionary of the current in-memory settings."""
    return {
        "personality": current_personality,
        "ttt_binding": current_ttt_binding,
        "ttt_model": current_ttt_model,
        "tti_binding": current_tti_binding,
        "tti_model": current_tti_model,
    }

def update_current_settings(settings: Dict[str, Optional[str]]):
    """Updates the global state variables from a settings dictionary."""
    global current_personality, current_ttt_binding, current_ttt_model
    global current_tti_binding, current_tti_model
    current_personality = settings.get("personality")
    current_ttt_binding = settings.get("ttt_binding")
    current_ttt_model = settings.get("ttt_model")
    current_tti_binding = settings.get("tti_binding", DEFAULT_TTI_BINDING) # Use default if missing
    current_tti_model = settings.get("tti_model", DEFAULT_TTI_MODEL)     # Use default if missing
    print_system("Settings loaded/updated in memory.")

def save_base64_image(b64_string: str, prompt: str) -> Optional[Path]:
    """Decodes base64 string and saves as PNG image in the IMAGE_DIR."""
    try:
        img_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(img_data))
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
        safe_prompt_short = safe_prompt[:30].replace(" ", "_") if safe_prompt else "image"
        filename = f"{timestamp}_{safe_prompt_short}.png"
        output_path = IMAGE_DIR / filename
        img.save(output_path, "PNG")
        print_system(f"Image saved successfully to: {output_path}")
        return output_path
    except base64.binascii.Error as e:
        print_error(f"Error decoding base64 string: {e}")
    except Exception as e:
        print_error(f"Error saving image: {e}")
    return None

def make_api_call(endpoint: str, method: str = "GET", payload: Optional[Dict] = None) -> Optional[Any]:
    """Generic function to make non-streaming API calls."""
    url = f"{BASE_URL}{endpoint}"
    headers = HEADERS_NO_STREAM # Use non-streaming headers
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
        else:
            print_error(f"Unsupported HTTP method: {method}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print_error(f"API call failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            status = e.response.status_code
            body = e.response.text
            detail = f"Status Code: {status}"
            try: detail += f", Detail: {e.response.json().get('detail', body)}"
            except json.JSONDecodeError: detail += f", Body: {body[:200]}..."
            print(f"  Server Response: {detail}")
        return None
    except Exception as e:
        print_error(f"Unexpected error during API call to {url}: {e}")
        return None


def make_generate_request(payload: Dict[str, Any], stream: bool) -> Optional[Union[str, Dict, SSEClient]]:
    """Makes a request specifically to the /generate endpoint."""
    url = f"{BASE_URL}/generate"
    headers = HEADERS_STREAM if stream else HEADERS_NO_STREAM
    timeout = DEFAULT_TIMEOUT

    try:
        response = requests.post(url, headers=headers, json=payload, stream=stream, timeout=timeout)
        # print(f"DEBUG: Server Status Code: {response.status_code}")
        # print(f"DEBUG: Server Headers: {response.headers}")
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if stream:
            if "text/event-stream" in content_type:
                return SSEClient(response)
            elif "text/plain" in content_type:
                print_system("\nWarning: Server returned plain text instead of stream. Displaying full response.")
                return response.text
            else:
                print_error(f"Server returned unexpected Content-Type '{content_type}' for streaming request.")
                try: print(f"DEBUG: Body:\n{response.content[:500].decode('utf-8', errors='replace')}")
                except Exception as e: print(f"DEBUG: Error reading response content: {e}")
                return None
        else: # Non-streaming
            if "application/json" in content_type:
                return response.json()
            else:
                 print_system(f"Warning: Expected JSON but got {content_type}")
                 return response.text

    except requests.exceptions.Timeout:
        print_error(f"Request timed out after {timeout} seconds connecting to {url}")
    except requests.exceptions.ConnectionError as e:
         print_error(f"Could not connect to server at {url}. Is lollms_server running? Details: {e}")
    except requests.exceptions.RequestException as e:
        print_error(f"Generate request failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            status = e.response.status_code
            body = e.response.text
            detail = f"Status Code: {status}"
            try: detail += f", Detail: {e.response.json().get('detail', body)}"
            except json.JSONDecodeError: detail += f", Body: {body[:200]}..."
            print(f"  Server Response: {detail}")
    except Exception as e:
        print_error(f"An unexpected error occurred during generate request: {e}")
    return None


# --- Menu Navigation and Selection Helpers ---

def display_menu(title: str, options: List[str], show_back=True):
    """Displays a numbered menu."""
    print(f"\n--- {title} ---")
    for i, option in enumerate(options):
        print(f"  {i+1}. {option}")
    if show_back:
        print(f"\n  0. Back / Cancel")
    print("-" * (len(title) + 6))

def get_menu_choice(max_option: int, allow_zero=True) -> Optional[int]:
    """Gets and validates user's numeric menu choice."""
    while True:
        try:
            prompt = "Enter selection number"
            if allow_zero: prompt += " (0 to cancel/go back)"
            prompt += ": "
            choice_str = input(prompt).strip()
            if not choice_str: continue # Handle empty input

            choice = int(choice_str)
            min_option = 0 if allow_zero else 1
            if min_option <= choice <= max_option:
                return choice
            else:
                print_error(f"Invalid choice. Please enter a number between {min_option} and {max_option}.")
        except ValueError:
            print_error("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print_system("\nOperation cancelled.")
            return None # Indicate cancellation

def select_item_from_api_list(
    item_type: str,
    fetch_endpoint: str,
    current_value: Optional[str],
    display_key: str = 'name',
    value_key: str = 'name',
    list_json_key: Optional[str] = None,
    is_dict_list: bool = True,
    allow_none: bool = True, # Allow selecting 'None' (Server Default)
    none_display_text: str = "Use Server Default"
) -> Optional[str]:
    """Fetches items, displays a menu, and returns the selected value."""
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
                 print_warning(f"Could not find or parse key '{list_json_key}' in API response.")
        elif isinstance(data, list):
            items = data
        else:
            print_warning(f"Unexpected API response format for {item_type}s.")
    else:
        print_error(f"Failed to fetch {item_type}s.")
        # Keep current value if fetch fails
        return current_value

    if not items:
        print_system(f"No {item_type}s available.")
        # If no items available, force selection to None if allowed, else keep current
        return None if allow_none else current_value

    options = []
    values = []
    current_selection_text = current_value or "None (Server Default)"

    # Build options list
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
            # Mark current selection
            marker = " [*]" if actual_val == current_value else ""
            options.append(f"{display_val}{marker}")
            values.append(actual_val)
        else:
            print_warning(f"Skipping invalid item in {item_type} list: {item}")

    if not options:
        print_system(f"No valid {item_type}s could be listed.")
        return None if allow_none else current_value

    title = f"Select {item_type}"
    menu_options = options
    if allow_none:
        none_marker = " [*]" if current_value is None else ""
        menu_options = [f"{none_display_text}{none_marker}"] + menu_options

    display_menu(title, menu_options, show_back=True)

    choice = get_menu_choice(len(options), allow_zero=True) # Allow 0 for "None" or "Back"

    if choice is None: # Cancelled
        return current_value
    elif choice == 0:
        if allow_none:
            print_system(f"{item_type} set to: {none_display_text}")
            return None # User explicitly selected None/Default
        else:
            print_system("Keeping current selection.")
            return current_value # 0 means Back/Cancel if None not allowed
    else:
        selected_value = values[choice - 1]
        print_system(f"{item_type} set to: {selected_value}")
        return selected_value

# --- Menu Handlers ---

def handle_personality_menu():
    """Handles the personality selection submenu."""
    global current_personality
    current_personality = select_item_from_api_list(
        item_type="Personality",
        fetch_endpoint="/list_personalities",
        current_value=current_personality,
        display_key='name',
        value_key='name',
        list_json_key='personalities',
        is_dict_list=True,
        allow_none=True, # Allow using no personality
        none_display_text="None (No Personality)"
    )

def handle_binding_menu():
    """Handles the TTT binding selection submenu."""
    global current_ttt_binding, current_ttt_model
    old_binding = current_ttt_binding
    new_binding = select_item_from_api_list(
        item_type="TTT Binding",
        fetch_endpoint="/list_bindings",
        current_value=current_ttt_binding,
        list_json_key='binding_instances',
        is_dict_list=False, # List of string keys
        allow_none=True,
        none_display_text="Use Server Default Binding"
    )
    if new_binding != old_binding:
        print_system(f"Binding changed from '{old_binding or 'Default'}' to '{new_binding or 'Default'}'. Resetting model.")
        current_ttt_binding = new_binding
        current_ttt_model = None

def handle_model_menu():
    """Handles the TTT model selection submenu."""
    global current_ttt_model
    if current_ttt_binding is None:
        print_system("Cannot select model: Server Default Binding is active.")
        print_system("Select a specific TTT Binding first to choose a model.")
        return

    current_ttt_model = select_item_from_api_list(
        item_type="TTT Model",
        fetch_endpoint=f"/list_available_models/{current_ttt_binding}",
        current_value=current_ttt_model,
        display_key='name',
        value_key='name',
        list_json_key='models',
        is_dict_list=True,
        allow_none=True, # Allow using binding's default model
        none_display_text=f"Use Default for '{current_ttt_binding}'"
    )

def display_current_settings():
    """Displays the currently active settings."""
    settings = get_current_settings()
    print("\n--- Current Settings ---")
    print(f"  Personality: {settings['personality'] or 'None (No Personality)'}")
    print(f"  TTT Binding: {settings['ttt_binding'] or 'Server Default'}")
    print(f"  TTT Model:   {settings['ttt_model'] or 'Server Default'}")
    print(f"  TTI Binding: {settings['tti_binding'] or 'Server Default'}")
    print(f"  TTI Model:   {settings['tti_model'] or 'Server Default'}")
    print("-" * 24)
    input("Press Enter to continue...")


def run_settings_menu():
    """Runs the main settings menu loop."""
    while True:
        menu_options = [
            f"Personality : ({current_personality or 'None'})",
            f"TTT Binding : ({current_ttt_binding or 'Server Default'})",
            f"TTT Model   : ({current_ttt_model or 'Server Default'})",
            "Show Current Settings",
            # Add TTI selection later if needed
        ]
        display_menu("Settings", menu_options, show_back=True)
        choice = get_menu_choice(len(menu_options), allow_zero=True)

        if choice is None or choice == 0: # Back or cancelled
            break
        elif choice == 1:
            handle_personality_menu()
        elif choice == 2:
            handle_binding_menu()
        elif choice == 3:
            handle_model_menu()
        elif choice == 4:
            display_current_settings()

    # Save settings when exiting the menu
    save_settings(get_current_settings())
    print_system("Settings saved.")

# --- Main Application Logic ---

if __name__ == "__main__":
    # Load initial state
    # FIXED: Ensure chat_history is always a list
    chat_history = load_history()
    if not isinstance(chat_history, list):
        print_error("History file was corrupted. Starting fresh.")
        chat_history = []

    saved_settings = load_settings()
    update_current_settings(saved_settings)

    # --- Initial messages ---
    print_system("Welcome to LOLLMS Console Chat!")
    print_system("Type '/imagine [prompt]' to generate an image.")
    print_system("Type '/settings' to change personality/model.")
    print_system("Type '/quit' or '/exit' to end the chat.")
    print_system("-" * 30)

    # --- History Display ---
    if chat_history:
        print_system("Resuming previous conversation:")
        for entry in chat_history:
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user": print_user(content)
            elif role == "assistant": print_ai_history(content) # Use specific history print
            elif role == "system": print_system(content)
        print_system("-" * 30)


    while True:
        prompt_prefix = "👤 YOU: "
        try:
            user_input = input(f"\n{prompt_prefix}").strip()
        except EOFError: print_system("Exiting..."); break
        except KeyboardInterrupt: print_system("Exiting..."); break

        if not user_input: continue

        # --- Command handling ---
        if user_input.lower() in ["/quit", "/exit"]:
            print_system("Saving state and exiting...")
            break
        elif user_input.lower() == "/settings":
            run_settings_menu()
            continue

        # Add user message to history *before* processing
        chat_history.append({"role": "user", "content": user_input})

        if user_input.lower().startswith("/imagine "):
            # --- Image Generation Logic ---
            image_prompt = user_input[len("/imagine "):].strip()
            if not image_prompt:
                print_system("Please provide a prompt after /imagine.")
                chat_history.append({"role": "system", "content": "Image generation skipped (no prompt)." })
                continue

            print_system(f"Generating image for: '{image_prompt}'...")
            payload = {
                "prompt": image_prompt,
                "generation_type": "tti",
                "binding_name": current_tti_binding, # Use current TTI settings
                "model_name": current_tti_model,     # Use current TTI settings
                "stream": False
            }
            response_data = make_generate_request(payload, stream=False)

            # --- Process image response ---
            if response_data and isinstance(response_data, dict):
                image_b64 = response_data.get("image_base64")
                if image_b64:
                    saved_path = save_base64_image(image_b64, image_prompt)
                    msg = f"Image generated and saved to {saved_path.name}" if saved_path else "Image generation succeeded but saving failed."
                    chat_history.append({"role": "system", "content": msg})
                else:
                    error_msg = response_data.get("error", "Response did not contain 'image_base64' data.")
                    print_error(f"Image generation failed: {error_msg}")
                    chat_history.append({"role": "system", "content": f"Image generation failed: {error_msg}"})
            elif response_data:
                print_error(f"Received unexpected response format for image generation:\n{response_data}")
                chat_history.append({"role": "system", "content": f"Image generation failed (unexpected response format)."})
            else: # Error already printed by make_generate_request
                chat_history.append({"role": "system", "content": "Image generation request failed."})
            # --- End Image Processing ---

        else: # Regular Text-to-Text chat
            # --- Print AI Prefix ---
            print_ai_prefix(current_personality)

            payload = {
                "prompt": user_input,
                "generation_type": "ttt",
                "personality": current_personality,
                "binding_name": current_ttt_binding,
                "model_name": current_ttt_model,
                "stream": True,
                # Add history if needed: "extra_data": {"history": chat_history[:-1]}
            }
            result = make_generate_request(payload, stream=True)

            # --- Process TTT response ---
            if isinstance(result, str): # Handle plain text fallback
                print(result)
                chat_history.append({"role": "assistant", "content": result})
            elif isinstance(result, SSEClient): # Handle stream
                sse_client = result
                full_ai_response = ""
                stream_error_occurred = False
                try:
                    for event in sse_client.events():
                        if event.event == 'message':
                            if not event.data: continue
                            try:
                                chunk_data = json.loads(event.data)
                                chunk_type = chunk_data.get("type")
                                content = chunk_data.get("content")

                                if chunk_type == "chunk" and content:
                                    print(content, end="", flush=True)
                                    full_ai_response += content
                                elif chunk_type == "error":
                                    error_msg = f"\n--- Stream Error: {content} ---"
                                    print(error_msg, end="", flush=True)
                                    if full_ai_response: chat_history.append({"role": "assistant", "content": full_ai_response})
                                    chat_history.append({"role": "system", "content": f"Stream error occurred: {content}"})
                                    full_ai_response = ""
                                    stream_error_occurred = True
                                    break
                                elif chunk_type == "final":
                                    final_content = chunk_data.get("content", full_ai_response)
                                    if not full_ai_response and final_content:
                                        print(final_content, end="", flush=True)
                                        full_ai_response = final_content
                                    print() # Newline
                                    if full_ai_response and not stream_error_occurred:
                                        chat_history.append({"role": "assistant", "content": full_ai_response})
                                    break # End loop
                            except json.JSONDecodeError: print(f"\nError: Received non-JSON data: {event.data}")
                            except Exception as e: print(f"\nError processing chunk: {e}")

                    # Save if stream ended abruptly without final/error
                    if full_ai_response and not stream_error_occurred and (not chat_history or chat_history[-1].get("content") != full_ai_response):
                         print() # Ensure newline
                         chat_history.append({"role": "assistant", "content": full_ai_response})

                except requests.exceptions.ChunkedEncodingError:
                    print_error("Stream connection broken unexpectedly.")
                    if full_ai_response and not stream_error_occurred: chat_history.append({"role": "assistant", "content": full_ai_response + " [Stream Interrupted]"})
                except Exception as e:
                    print_error(f"An error occurred while processing the stream: {e}")
                    if full_ai_response and not stream_error_occurred: chat_history.append({"role": "assistant", "content": full_ai_response + " [Stream Error]"})
            else: # Handle request failure
                print() # Add newline because AI prefix was printed
                chat_history.append({"role": "system", "content": "Text generation request failed."})
            # --- End TTT Response Processing ---

        # Save history after each turn
        save_history(chat_history)

    # Final save before exiting
    save_history(chat_history)
    save_settings(get_current_settings())
    print_system("Goodbye!")