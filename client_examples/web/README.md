# Web Client Example (`client_examples/web/`)

## Purpose

This example provides a very basic web frontend (HTML, CSS, JavaScript) to interact with the `lollms_server` API. It demonstrates:

*   Connecting to the server and fetching initial data (bindings, personalities).
*   Populating dropdown menus for selection.
*   Sending Text-to-Text (TTT) and Text-to-Image (TTI) requests.
*   Handling both streaming (SSE) and non-streaming responses for TTT.
*   Displaying text results and generated images.
*   Basic configuration (server URL, API key) via the UI.

## Prerequisites

*   A modern web browser (Chrome, Firefox, Edge, Safari).
*   A running `lollms_server` instance.
*   **Crucially:** The `lollms_server` must be configured with appropriate CORS settings in `config.toml` to allow requests from the origin where you are serving this web client (e.g., `http://localhost:8080` if using a local server, or `null` if opening `index.html` directly via `file://`). Example `config.toml` snippet:
    ```toml
    [server]
    # ... other settings ...
    # Allow access from file:// and a common dev server port
    allowed_origins = ["null", "http://localhost:8080", "http://127.0.0.1:8080"]
    ```
*   Working TTT and TTI bindings configured on the server.
*   A valid `lollms_server` API key (if required by the server).

## Configuration

Configuration is done directly in the web interface:

*   **Server URL:** Enter the base URL of your running `lollms_server` (e.g., `http://localhost:9600`). **Do NOT include `/api/v1` here.**
*   **API Key:** Enter a valid API key if your server requires one.

## Running the Example

You have two main options:

**1. Open `index.html` Directly (Simple, Requires CORS `null`):**

*   Ensure your `lollms_server` `config.toml` includes `"null"` in `[server].allowed_origins`.
*   Navigate to the `lollms_server/client_examples/web/` directory in your file explorer.
*   Double-click `index.html` to open it in your web browser. The URL will start with `file://`.

**2. Serve via a Simple HTTP Server (Recommended for Development):**

*   This avoids potential issues with `file://` origins and mimics a real web deployment better.
*   Ensure your `lollms_server` `config.toml` includes the origin of your dev server (e.g., `"http://localhost:8080"`) in `[server].allowed_origins`.
*   Open your terminal in the `lollms_server/client_examples/web/` directory.
*   Run Python's built-in HTTP server (or any other simple server):
    ```bash
    # For Python 3
    python -m http.server 8080
    ```
*   Open your browser to `http://localhost:8080`.

## Usage

1.  Enter the Server URL and API Key (if needed).
2.  Click "Connect & Load Data". The status should update, and the dropdowns below should populate if successful.
3.  Select Personality, Binding, Model (optional, uses defaults otherwise).
4.  Choose Generation Type (TTT or TTI).
5.  Enter your prompt.
6.  Check the "Stream Response" box if you want a streaming TTT response.
7.  (Optional) Click "Add Parameters" or "Add Extra Data" to reveal text areas where you can enter JSON for advanced configuration (passed directly to the server).
8.  Click "Generate".
9.  Monitor the "Generate Status" and view results in the "Results" section. Text streams in for TTT streaming, JSON metadata appears below, and images are displayed for TTI.

## Files

*   `index.html`: The main HTML structure.
*   `style.css`: Basic styling for the page.
*   `lollms-client.js`: A reusable JavaScript class (`LollmsClient`) for making API calls (fetch, SSE handling).
*   `script.js`: Contains the UI logic that uses `LollmsClient` to interact with the DOM elements and the API.

## Notes

*   This is a basic example, not a production-ready UI.
*   Error handling is minimal. Check the browser's developer console (F12) for detailed errors.
*   The `LollmsClient` class could be further developed and potentially published as a separate library.
*   Styling is very basic.
*   Currently lacks support for visualizing audio/video outputs.