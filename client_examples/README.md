# Simple Client Example (`simple_client.py`)

## Purpose

This script demonstrates basic interaction with the `lollms_server` API for:

1.  Listing available binding instances and personalities.
2.  Prompting the user to select a personality, binding, and model (or use defaults).
3.  Sending a non-streaming Text-to-Text (TTT) generation request based on user input.
4.  Printing the JSON or text response received from the server.

## Prerequisites

*   Python 3.7+
*   `requests` library (`pip install requests`)
*   A running `lollms_server` instance.
*   A valid API key configured in the `lollms_server`'s `config.toml`.

## Configuration

Edit the following variables at the top of `simple_client.py`:

*   `BASE_URL`: Set this to the URL of your running `lollms_server` (e.g., `"http://localhost:9600/api/v1"`).
*   `API_KEY`: Set this to a valid API key allowed by your server.

## Running the Example

1.  Make sure your `lollms_server` is running.
2.  Ensure your Python environment has the `requests` library installed.
3.  Navigate to the `lollms_server` project root directory in your terminal.
4.  Run the script:
    ```bash
    python client_examples/simple_client.py
    ```

## Expected Behavior

1.  The script will connect to the server and fetch available bindings and personalities.
2.  It will prompt you to:
    *   Choose a personality (or skip).
    *   Decide whether to use the server's default binding/model or select manually.
    *   If selecting manually, choose a binding instance and then optionally a specific model for that binding.
    *   Enter your text prompt.
    *   Optionally add custom generation parameters (like `temperature`).
3.  The script will construct the JSON payload for the `/generate` request and print it.
4.  It will send the request to the server.
5.  Finally, it will print the server's response (either formatted JSON or plain text).

## Notes

*   This client does not support streaming responses. See `streaming_client.py`.
*   It does not handle multimodal inputs (images, etc.).
*   Error handling is basic. Server connection issues or API errors will be printed to the console.