# client_examples/simple_client.py
import requests
import json
import sys
import textwrap # For better formatting
from typing import Optional
# --- Configuration ---
BASE_URL = "http://localhost:9600/api/v1" # Your lollms_server address
API_KEY = "user1_key_abc123" # Use a valid API key from your config.toml
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}
DEFAULT_TIMEOUT = 60 # Increased default timeout for potentially long generations

# --- Styling ---
SEPARATOR = "=" * 60
INDENT = "  "

# --- Helper Functions ---
def print_json(data, indent=INDENT):
    """Prints JSON data with indentation."""
    print(textwrap.indent(json.dumps(data, indent=2), indent))

def print_heading(title):
    """Prints a formatted heading."""
    print(f"\n{SEPARATOR}")
    print(f"{INDENT}{title.upper()}")
    print(SEPARATOR)

def print_error(message):
    """Prints an error message."""
    print(f"\n{INDENT}ERROR: {message}")

def make_request(method, endpoint, **kwargs):
    """Makes a request to the server and handles basic errors."""
    url = f"{BASE_URL}{endpoint}"
    timeout = kwargs.pop('timeout', DEFAULT_TIMEOUT)
    try:
        response = requests.request(method, url, headers=HEADERS, timeout=timeout, **kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        elif "text/plain" in content_type:
            return response.text
        else:
            return response.content
    except requests.exceptions.Timeout:
        print_error(f"Request timed out after {timeout} seconds connecting to {url}")
        return None
    except requests.exceptions.ConnectionError as e:
         print_error(f"Could not connect to server at {url}. Is it running? Details: {e}")
         return None
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"{INDENT}Server Response Status: {e.response.status_code}")
            try:
                error_detail = e.response.json().get('detail', e.response.text)
                print(f"{INDENT}Server Response Detail: {error_detail}")
            except Exception:
                print(f"{INDENT}Server Response Body: {e.response.text}")
        return None
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        return None

def select_from_list(items: list, item_type: str, display_key: str = 'name', can_skip=False) -> Optional[str]:
    """Helper to prompt user to select an item from a list."""
    if not items:
        print(f"{INDENT}No {item_type}s found.")
        return None

    print(f"{INDENT}Available {item_type}s:")
    for i, item in enumerate(items):
        display_value = item if isinstance(item, str) else item.get(display_key, f"Item {i+1}")
        print(f"{INDENT}  {i+1}. {display_value}")

    prompt_text = f"{INDENT}Select {item_type} number (1-{len(items)})"
    if can_skip:
        prompt_text += " or 0 to skip/use default: "
    else:
        prompt_text += ": "

    while True:
        try:
            choice_idx = int(input(prompt_text))
            if can_skip and choice_idx == 0:
                return None # User chose to skip
            choice_idx -= 1 # Adjust to 0-based index
            if 0 <= choice_idx < len(items):
                 return items[choice_idx] if isinstance(items[choice_idx], str) else items[choice_idx].get(display_key)
            else:
                print(f"{INDENT}Invalid selection.")
        except ValueError:
            print(f"{INDENT}Invalid input. Please enter a number.")


# --- Main Execution ---
if __name__ == "__main__":
    print_heading("Connecting to lollms_server")

    # 1. List Bindings
    print(f"{INDENT}Discovering Bindings...")
    bindings_response = make_request("GET", "/list_bindings")
    if not bindings_response or 'binding_instances' not in bindings_response:
        print_error("Could not fetch bindings or no instances found. Exiting.")
        sys.exit(1)
    configured_bindings = bindings_response['binding_instances']
    binding_names = list(configured_bindings.keys())
    print(f"{INDENT}Found {len(binding_names)} configured binding instances.")

    # 2. List Personalities
    print(f"\n{INDENT}Discovering Personalities...")
    personalities_response = make_request("GET", "/list_personalities")
    if not personalities_response or 'personalities' not in personalities_response:
        print_error("Could not fetch personalities or none found. Exiting.")
        sys.exit(1)
    loaded_personalities_dict = personalities_response['personalities']
    personality_names = list(loaded_personalities_dict.keys())
    print(f"{INDENT}Found {len(personality_names)} loaded personalities.")

    # --------------------------
    # Build Generation Request
    # --------------------------
    print_heading("Build Generation Request")
    payload = {"stream": False} # Default to non-streaming for this client
    generate_request = True

    # --- Choose Personality ---
    print(f"\n{INDENT}Step 1: Choose Personality (or None)")
    chosen_personality_name = select_from_list(personality_names, "Personality", can_skip=True)
    payload['personality'] = chosen_personality_name # Will be None if skipped

    # --- Choose Binding & Model ---
    print(f"\n{INDENT}Step 2: Choose Binding and Model")
    use_defaults = input(f"{INDENT}Use server default binding/model for TTT? (y/n): ").lower()
    if use_defaults != 'y':
        chosen_binding_name = select_from_list(binding_names, "Binding Instance")
        if not chosen_binding_name:
            print_error("Binding selection failed. Exiting.")
            sys.exit(1)
        payload['binding_name'] = chosen_binding_name

        # List and select model
        print(f"\n{INDENT}Fetching models for binding '{chosen_binding_name}'...")
        models_response = make_request("GET", f"/list_available_models/{chosen_binding_name}")
        available_models = models_response.get('models', []) if models_response else []

        if available_models:
             chosen_model_info = select_from_list(available_models, "Model", display_key='name', can_skip=True)
             if chosen_model_info: # User selected a model
                 payload['model_name'] = chosen_model_info # chosen_model_info is the name string here
             else: # User chose 0 or selection failed
                  print(f"{INDENT}No model selected, will use default for binding '{chosen_binding_name}' (if defined on server).")
        else:
             print(f"{INDENT}Warning: Could not fetch models for binding '{chosen_binding_name}'.")
             model_input = input(f"{INDENT}Enter exact model name manually (or leave blank for server default): ")
             if model_input: payload['model_name'] = model_input
             else: print(f"{INDENT}No model specified, server default will be used.")

    # --- Get Prompt ---
    print(f"\n{INDENT}Step 3: Enter Prompt")
    payload['prompt'] = input(f"{INDENT}Prompt: ")
    if not payload['prompt']:
         print_error("Prompt cannot be empty. Exiting.")
         sys.exit(1)

    # --- Add Extra Data (Optional) ---
    print(f"\n{INDENT}Step 4: Add Extra Data (Optional)")
    add_extra = input(f"{INDENT}Add extra data (e.g., for RAG)? (y/n): ").lower()
    if add_extra == 'y':
         print(f"{INDENT}Enter extra data as key=value pairs, one per line (e.g., context=some text).")
         print(f"{INDENT}Enter an empty line when finished.")
         extra_data = {}
         while True:
              line = input(f"{INDENT}  key=value: ")
              if not line: break
              parts = line.split('=', 1)
              if len(parts) == 2:
                   key, value = parts[0].strip(), parts[1].strip()
                   if key: extra_data[key] = value
                   else: print(f"{INDENT}  Invalid format, skipping line.")
              else:
                   print(f"{INDENT}  Invalid format (must be key=value), skipping line.")
         if extra_data:
              payload['extra_data'] = extra_data


    # --- Add Custom Parameters (Optional) ---
    print(f"\n{INDENT}Step 5: Add Custom Parameters (Optional)")
    add_params = input(f"{INDENT}Add custom generation parameters (e.g., temperature, max_tokens)? (y/n): ").lower()
    if add_params == 'y':
         print(f"{INDENT}Enter parameters as key=value pairs, one per line (e.g., temperature=0.5).")
         print(f"{INDENT}Enter an empty line when finished.")
         custom_params = {}
         while True:
              line = input(f"{INDENT}  key=value: ")
              if not line: break
              parts = line.split('=', 1)
              if len(parts) == 2:
                   key, value_str = parts[0].strip(), parts[1].strip()
                   if key:
                        # Try converting to number if possible
                        try: value = float(value_str)
                        except ValueError:
                             try: value = int(value_str)
                             except ValueError: value = value_str # Keep as string
                        custom_params[key] = value
                   else: print(f"{INDENT}  Invalid format, skipping line.")
              else:
                   print(f"{INDENT}  Invalid format (must be key=value), skipping line.")
         if custom_params:
              payload['parameters'] = custom_params

    # --------------------------
    # Perform Generation
    # --------------------------
    print_heading("Sending Generation Request")
    print(f"{INDENT}Using Payload:")
    print_json(payload)

    response_text = make_request("POST", "/generate", json=payload)

    print_heading("Generation Result")
    if response_text is not None:
        print(response_text)
    else:
        print_error("Generation failed or returned no content.")

    print_heading("Client Finished")