# client_examples/simple_client.py
import requests
import json
import sys
import textwrap
from typing import Optional

# --- Configuration ---
BASE_URL = "http://localhost:9600/api/v1"
API_KEY = "user1_key_abc123" # Default key from example config
HEADERS = { "X-API-Key": API_KEY, "Content-Type": "application/json", "Accept": "application/json" }
DEFAULT_TIMEOUT = 60
SEPARATOR = "=" * 60
INDENT = "  "

# --- Helper Functions ---
def print_json(data, indent=INDENT):
    """Prints JSON data with indentation."""
    print(textwrap.indent(json.dumps(data, indent=2), indent))

def print_heading(title):
    """Prints a formatted heading."""
    print(f"\n{SEPARATOR}\n{INDENT}{title.upper()}\n{SEPARATOR}")

def print_error(message):
    """Prints an error message."""
    print(f"\n{INDENT}ERROR: {message}")

def make_request(method, endpoint, **kwargs):
    """Makes a request to the server and handles basic errors."""
    url = f"{BASE_URL}{endpoint}"; timeout = kwargs.pop('timeout', DEFAULT_TIMEOUT)
    try:
        response = requests.request(method, url, headers=HEADERS, timeout=timeout, **kwargs)
        response.raise_for_status(); content_type = response.headers.get("content-type", "")
        if "application/json" in content_type: return response.json()
        elif "text/plain" in content_type: return response.text
        else: return response.content
    except requests.exceptions.Timeout: print_error(f"Request timed out ({timeout}s) to {url}"); return None
    except requests.exceptions.ConnectionError as e: print_error(f"Connection error to {url}: {e}"); return None
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"{INDENT}Server Status: {e.response.status_code}")
            try: error_detail = e.response.json().get('detail', e.response.text); print(f"{INDENT}Server Detail: {error_detail}")
            except Exception: print(f"{INDENT}Server Body: {e.response.text}")
        return None
    except Exception as e: print_error(f"Unexpected error: {e}"); return None

def select_from_list(items: list, item_type: str, display_key: str = 'name', can_skip=False) -> Optional[str]:
    """Helper to prompt user to select an item from a list."""
    if not items: print(f"{INDENT}No {item_type}s found."); return None
    print(f"{INDENT}Available {item_type}s:")
    for i, item in enumerate(items):
        display_value = item if isinstance(item, str) else item.get(display_key, f"Item {i+1}")
        print(f"{INDENT}  {i+1}. {display_value}")
    prompt_text = f"{INDENT}Select {item_type} number (1-{len(items)})"
    prompt_text += " or 0 to skip/use default: " if can_skip else ": "
    while True:
        try:
            choice_idx = int(input(prompt_text))
            if can_skip and choice_idx == 0: return None
            choice_idx -= 1
            if 0 <= choice_idx < len(items): return items[choice_idx] if isinstance(items[choice_idx], str) else items[choice_idx].get(display_key)
            else: print(f"{INDENT}Invalid selection.")
        except ValueError: print(f"{INDENT}Invalid input. Please enter a number.")

# --- Main Execution ---
if __name__ == "__main__":
    print_heading("Connecting to lollms_server")
    print(f"{INDENT}Discovering Bindings...")
    bindings_response = make_request("GET", "/list_bindings")
    if not bindings_response or 'binding_instances' not in bindings_response: print_error("Could not fetch bindings. Exiting."); sys.exit(1)
    configured_bindings = bindings_response['binding_instances']; binding_names = list(configured_bindings.keys())
    print(f"{INDENT}Found {len(binding_names)} configured binding instances.")

    print(f"\n{INDENT}Discovering Personalities...")
    personalities_response = make_request("GET", "/list_personalities")
    if not personalities_response or 'personalities' not in personalities_response: print_error("Could not fetch personalities. Exiting."); sys.exit(1)
    loaded_personalities_dict = personalities_response['personalities']; personality_names = list(loaded_personalities_dict.keys())
    print(f"{INDENT}Found {len(personality_names)} loaded personalities.")

    print_heading("Build Generation Request")
    payload = {"stream": False, "input_data": []} # Use input_data list

    print(f"\n{INDENT}Step 1: Choose Personality (or None)")
    chosen_personality_name = select_from_list(personality_names, "Personality", can_skip=True)
    payload['personality'] = chosen_personality_name

    print(f"\n{INDENT}Step 2: Choose Binding and Model")
    use_defaults = input(f"{INDENT}Use server default binding/model for TTT? (y/n): ").lower()
    if use_defaults != 'y':
        chosen_binding_name = select_from_list(binding_names, "Binding Instance")
        if not chosen_binding_name: print_error("Binding selection failed. Exiting."); sys.exit(1)
        payload['binding_name'] = chosen_binding_name
        print(f"\n{INDENT}Fetching models for binding '{chosen_binding_name}'...")
        models_response = make_request("GET", f"/list_available_models/{chosen_binding_name}")
        available_models = models_response.get('models', []) if models_response else []
        if available_models:
             chosen_model_name = select_from_list(available_models, "Model", display_key='name', can_skip=True)
             if chosen_model_name: payload['model_name'] = chosen_model_name
             else: print(f"{INDENT}No model selected, using default for '{chosen_binding_name}'.")
        else:
             print(f"{INDENT}Warning: Could not fetch models for '{chosen_binding_name}'.")
             model_input = input(f"{INDENT}Enter exact model name manually (or leave blank for default): ")
             if model_input: payload['model_name'] = model_input
             else: print(f"{INDENT}No model specified, server default will be used.")

    print(f"\n{INDENT}Step 3: Enter Prompt")
    user_prompt = input(f"{INDENT}Prompt: ")
    if not user_prompt: print_error("Prompt cannot be empty. Exiting."); sys.exit(1)
    # Add prompt to input_data
    payload['input_data'].append({"type": "text", "role": "user_prompt", "data": user_prompt})

    # --- REMOVED: Extra Data (use input_data with specific roles instead) ---

    print(f"\n{INDENT}Step 4: Add Custom Parameters (Optional)")
    add_params = input(f"{INDENT}Add custom generation parameters (e.g., temperature)? (y/n): ").lower()
    if add_params == 'y':
         print(f"{INDENT}Enter parameters as key=value (e.g., temperature=0.5). Empty line when finished."); custom_params = {}
         while True:
              line = input(f"{INDENT}  key=value: ")
              if not line: break
              parts = line.split('=', 1)
              if len(parts) == 2:
                   key, value_str = parts[0].strip(), parts[1].strip()
                   if key:
                        try: value = float(value_str)
                        except ValueError: 
                            try:
                                value = int(value_str)
                            except ValueError:
                                value = value_str
                        custom_params[key] = value
                   else: print(f"{INDENT}  Invalid format.")
              else: print(f"{INDENT}  Invalid format.")
         if custom_params: payload['parameters'] = custom_params

    print_heading("Sending Generation Request")
    print(f"{INDENT}Using Payload:")
    print_json(payload)
    response_content = make_request("POST", "/generate", json=payload)

    print_heading("Generation Result")
    if response_content is not None:
        if isinstance(response_content, dict): print_json(response_content) # Print JSON structure
        else: print(response_content) # Print raw text
    else: print_error("Generation failed or returned no content.")

    print_heading("Client Finished")