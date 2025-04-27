# Console Chat App Example (`chat_app.py`)

## Purpose

This script provides a more interactive console-based chat application for interacting with the `lollms_server`. It features:

*   Persistent chat history (`chat_history.json`).
*   Persistent settings (`settings.json`) for personality, TTT binding, and TTT model.
*   A `/settings` command to change the active personality and model via interactive menus.
*   A `/imagine` command to trigger Text-to-Image (TTI) generation.
*   Support for streaming Text-to-Text (TTT) responses.
*   Saving of generated images to `generated_images/`.
*   Basic dependency checking using `pipmaster`.

## Prerequisites

*   Python 3.7+
*   `requests` library (`pip install requests`)
*   `sseclient-py` library (`pip install sseclient-py`)
*   `Pillow` library (`pip install Pillow`)
*   `pipmaster` library (`pip install pipmaster`)
*   A running `lollms_server` instance.
*   A valid API key configured in the `lollms_server`'s `config.toml`.
*   Working TTT and (optionally) TTI bindings configured on the server.

## Configuration

Edit the following variables near the top of `chat_app.py`:

*   `BASE_URL`: Set this to the **API URL** of your running `lollms_server` (e.g., `"http://localhost:9600/api/v1"`).
*   `API_KEY`: Set this to a valid API key allowed by your server.
*   `DEFAULT_TTI_BINDING`, `DEFAULT_TTI_MODEL` (Optional): Set default TTI binding/model if desired, otherwise leave as `None`.

The application uses two JSON files within the `client_examples` directory:

*   `chat_history.json`: Stores the conversation history. Created automatically.
*   `settings.json`: Stores the user's selected personality, TTT binding, and TTT model. Created automatically.

## Running the Example

1.  Make sure your `lollms_server` is running with working bindings.
2.  Ensure your Python environment has the required libraries (`requests`, `sseclient-py`, `Pillow`, `pipmaster`) installed. `pipmaster` should attempt to install missing ones if it's present.
3.  Navigate to the `lollms_server` project root directory in your terminal.
4.  Run the script:
    ```bash
    python client_examples/chat_app.py
    ```

## Usage

*   **Chat:** Simply type your message and press Enter to interact with the TTT model using the current settings.
*   **Image Generation:** Type `/imagine <your image prompt>` and press Enter. The script will use the configured TTI defaults (or server defaults) and save the image to `client_examples/generated_images/`.
*   **Settings:** Type `/settings` and press Enter to access the configuration menu:
    *   Select Personality
    *   Select TTT Binding
    *   Select TTT Model (only available after selecting a specific binding)
    *   Show Current Settings
    *   Use `0` to go back or save changes.
*   **Quit:** Type `/quit` or `/exit` (or press Ctrl+C/Ctrl+D).

## Expected Behavior

*   On first run, it will start a new chat. On subsequent runs, it loads previous history and settings.
*   Chat messages stream in token by token for TTT.
*   `/imagine` requests print status messages and save images.
*   `/settings` allows interactive selection from options fetched live from the server API.
*   Chat history and settings are saved automatically.

## Notes

*   Error handling is present but may not cover all edge cases.
*   The TTI functionality uses the default TTI binding/model configured in the script or on the server; it doesn't have a separate settings menu option in this version.