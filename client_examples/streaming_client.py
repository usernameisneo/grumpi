# client_examples/streaming_client.py
import requests
import json
from sseclient import SSEClient

# --- Configuration ---
BASE_URL = "http://localhost:9600/api/v1" # Your lollms_server address
API_KEY = "user1_key_abc123" # Use a valid API key from your config.toml
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "text/event-stream" # Important for streaming
}

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Performing Streaming Text Generation ---")
    payload = {
        "personality": "my_coolbot", # Use a personality loaded on your server
        "prompt": "Write a short story about a robot discovering music.",
        "model_name": "phi3:mini", # Specify model or rely on default
        "binding_name": "default_ollama", # Specify binding or rely on default
        "stream": True, # <<< Enable streaming
        "parameters": {
            "max_tokens": 200,
            "temperature": 0.7
        }
    }
    url = f"{BASE_URL}/generate"

    try:
        # Use requests with stream=True and sseclient
        response = requests.post(url, headers=HEADERS, json=payload, stream=True)
        response.raise_for_status() # Check for initial errors like 401, 404

        client = SSEClient(response)
        full_response = ""

        print("\nStreaming Result:")
        for event in client.events():
            if event.event == 'message': # Default event type if not specified
                if not event.data:
                    continue
                try:
                    # Parse the JSON data payload
                    chunk_data = json.loads(event.data)
                    chunk_type = chunk_data.get("type")
                    content = chunk_data.get("content")
                    metadata = chunk_data.get("metadata", {})

                    if chunk_type == "chunk" and content:
                        print(content, end="", flush=True) # Print chunk content immediately
                        full_response += content
                    elif chunk_type == "final":
                        print("\n--- Stream Finished ---")
                        print(f"Final Metadata: {metadata}")
                        # Optional: verify final content matches accumulated content
                        # print(f"Accumulated: {full_response}")
                        # print(f"Final Content: {content}")
                    elif chunk_type == "error":
                        print(f"\n--- Stream Error ---")
                        print(f"Error: {content}")
                    elif chunk_type == "info":
                         print(f"\n--- Stream Info ---")
                         print(f"Info: {content}")
                    # Handle other chunk types if defined ('function_call'?)

                except json.JSONDecodeError:
                    print(f"\nError: Received non-JSON data: {event.data}")
                except Exception as e:
                    print(f"\nError processing stream chunk: {e}")

    except requests.exceptions.RequestException as e:
        print(f"\nError connecting or during request to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server Response Status: {e.response.status_code}")
            try:
                print(f"Server Response Body: {e.response.text}")
            except Exception:
                 print("Server Response Body: (Could not decode)")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\n--- End of Streaming Client ---")