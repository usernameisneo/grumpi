# client_examples/image_client.py
import requests
import json
import base64
from PIL import Image
from io import BytesIO
import os
import time

# --- Configuration ---
BASE_URL = "http://localhost:9601/api/v1" # Your lollms_server address
API_KEY = "user1_key_abc123" # Use a valid API key from your config.toml
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}
OUTPUT_DIR = "generated_images" # Directory to save images

# --- Helper Functions ---
def print_json(data):
    """Prints JSON data with indentation."""
    print(json.dumps(data, indent=2))

def save_base64_image(b64_string, filename):
    """Decodes base64 string and saves as PNG image."""
    try:
        img_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(img_data))

        # Ensure output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_path = os.path.join(OUTPUT_DIR, filename)
        img.save(output_path, "PNG")
        print(f"Image saved successfully to: {output_path}")
        return output_path
    except base64.binascii.Error as e:
        print(f"Error decoding base64 string: {e}")
        return None
    except Exception as e:
        print(f"Error saving image '{filename}': {e}")
        return None

def make_request(method, endpoint, **kwargs):
    """Makes a request (same as simple_client)."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, headers=HEADERS, **kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        else:
             print(f"Warning: Expected JSON response but got {content_type}")
             return response.text # Or content
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server Response Status: {e.response.status_code}")
            try:
                print(f"Server Response Body: {e.response.text}")
            except Exception:
                print("Server Response Body: (Could not decode)")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Performing Image Generation (TTI) ---")
    payload = {
        "personality": "my_coolbot", # Personality choice might affect prompt rewriting if enabled later
        "prompt": "Impressionist painting of a cat programmer working late at night, coffee cup nearby",
        "binding_name": "my_dalle_binding", # Use the DALL-E binding instance name from your config
        "generation_type": "tti",
        "parameters": {
            "model": "dall-e-3", # Can specify model here or rely on binding default
            "size": "1024x1024",
            "quality": "standard",
            "style": "vivid"
        },
        "stream": False # TTI is never streamed
    }

    start_time = time.time()
    response_data = make_request("POST", "/generate", json=payload)
    end_time = time.time()

    if response_data and isinstance(response_data, dict):
        print(f"\nGeneration request took {end_time - start_time:.2f} seconds.")
        print("Response Metadata:")
        print_json({k: v for k, v in response_data.items() if k != 'image_base64'}) # Print details except the long b64 string

        image_b64 = response_data.get("image_base64")
        if image_b64:
            # Create a unique filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            save_base64_image(image_b64, filename)
        else:
            print("\nError: Response did not contain 'image_base64' data.")

    elif response_data:
         print(f"\nReceived unexpected response format:\n{response_data}")