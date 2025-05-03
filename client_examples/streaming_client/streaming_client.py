# client_examples/streaming_client.py
import requests
import json
from sseclient import SSEClient

# --- Configuration ---
BASE_URL = "http://localhost:9601/api/v1"
API_KEY = "user1_key_abc123" # Use a valid API key
HEADERS = { "X-API-Key": API_KEY, "Content-Type": "application/json", "Accept": "text/event-stream" }

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Performing Streaming Text Generation ---")
    text_prompt = "Write a short story about a robot discovering music."
    payload = {
        "input_data": [{"type": "text", "role": "user_prompt", "data": text_prompt}], # Use input_data
        "personality": None, # Example: No personality
        # "model_name": "mistral:latest", # Example model (Optional)
        # "binding_name": "default_ollama", # Example binding (Optional)
        "stream": True,
        "generation_type": "ttt",
        "parameters": { "max_tokens": 200, "temperature": 0.7 }
    }
    url = f"{BASE_URL}/generate"

    try:
        response = requests.post(url, headers=HEADERS, json=payload, stream=True)
        response.raise_for_status() # Check for HTTP errors
        client = SSEClient(response)
        full_response_text = ""
        final_output_list = []
        print("\nStreaming Result:")

        for event in client.events():
            if event.event == 'message' and event.data:
                try:
                    chunk_data = json.loads(event.data)
                    chunk_type = chunk_data.get("type")
                    content = chunk_data.get("content")
                    metadata = chunk_data.get("metadata", {})

                    if chunk_type == "chunk" and content:
                        # Only print text chunks directly for TTT
                        if isinstance(content, str):
                             print(content, end="", flush=True)
                             full_response_text += content
                        # You could handle other chunk types here if needed (e.g., audio chunks)
                    elif chunk_type == "final":
                         # The content of the final chunk is List[OutputData]
                         final_output_list = content if isinstance(content, list) else []
                         print("\n--- Stream Finished ---")
                         print(f"Final Metadata: {metadata}")
                         # Optionally process the final_output_list here
                         print("Final Output Structure:")
                         print(json.dumps(final_output_list, indent=2))
                         break # End loop on final chunk
                    elif chunk_type == "error":
                         print(f"\n--- Stream Error: {content} ---")
                         break # Stop on error
                    elif chunk_type == "info":
                         print(f"\n--- Stream Info: {content} ---")
                except json.JSONDecodeError:
                    print(f"\nError: Received non-JSON data: {event.data}")
                except Exception as e:
                    print(f"\nError processing chunk: {e}")
            # Handle other SSE event types if necessary
            # elif event.event == 'some_other_event': ...

    except requests.exceptions.RequestException as e:
        print(f"\nError requesting {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f" Server Response: {e.response.status_code} - {e.response.text[:200]}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    print("\n--- End of Streaming Client ---")