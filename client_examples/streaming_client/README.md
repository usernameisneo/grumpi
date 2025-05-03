# Streaming Client Example (`streaming_client.py`)

## Purpose

This script demonstrates how to interact with the `lollms_server` API for **streaming** Text-to-Text (TTT) generation using Server-Sent Events (SSE).

## Prerequisites

*   Python 3.7+
*   `requests` library (`pip install requests`)
*   `sseclient-py` library (`pip install sseclient-py`)
*   A running `lollms_server` instance.
*   A valid API key configured in the `lollms_server`'s `config.toml`.
*   A TTT binding/model configured and working on the server (e.g., Ollama).

## Configuration

Edit the following variables at the top of `streaming_client.py`:

*   `BASE_URL`: Set this to the URL of your running `lollms_server` (e.g., `"http://localhost:9601/api/v1"`).
*   `API_KEY`: Set this to a valid API key allowed by your server.
*   **Payload:** Modify the `payload` dictionary within the `if __name__ == "__main__":` block to specify:
    *   `input_data`: The prompt and any other inputs.
    *   `personality` (optional).
    *   `model_name` (optional, uses server default otherwise).
    *   `binding_name` (optional, uses server default otherwise).
    *   `parameters` (optional generation parameters like `max_tokens`, `temperature`).
    *   **`stream` must be set to `True`**.

## Running the Example

1.  Make sure your `lollms_server` is running and configured with a TTT binding.
2.  Ensure your Python environment has the `requests` and `sseclient-py` libraries installed.
3.  Navigate to the `lollms_server` project root directory in your terminal.
4.  Run the script:
    ```bash
    python client_examples/streaming_client.py
    ```

## Expected Behavior

1.  The script sends a POST request to the `/generate` endpoint with `stream: True`.
2.  It connects to the Server-Sent Events stream provided by the server.
3.  As the server generates text, the script receives JSON chunks via SSE.
4.  It parses each chunk and prints the `content` of `type: "chunk"` events directly to the console, creating the appearance of the text streaming in.
5.  It handles other event types like `final`, `error`, and `info`, printing relevant information.

## Notes

*   This client specifically demonstrates streaming TTT. TTI and other generation types are typically non-streaming.
*   The example payload uses hardcoded values for the prompt, model, etc. Adapt these as needed.