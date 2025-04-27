# Image Client Example (`image_client.py`)

## Purpose

This script demonstrates how to interact with the `lollms_server` API to perform Text-to-Image (TTI) generation.

## Prerequisites

*   Python 3.7+
*   `requests` library (`pip install requests`)
*   `Pillow` library (`pip install Pillow`)
*   A running `lollms_server` instance.
*   A **TTI-capable binding** (e.g., `dalle_binding`, a Stable Diffusion API binding) configured correctly in the `lollms_server`'s `config.toml`, including any necessary API keys for the binding itself.
*   A valid `lollms_server` API key configured in `config.toml [security]`.

## Configuration

Edit the following variables at the top of `image_client.py`:

*   `BASE_URL`: Set this to the URL of your running `lollms_server` (e.g., `"http://localhost:9600/api/v1"`).
*   `API_KEY`: Set this to a valid `lollms_server` API key.
*   `OUTPUT_DIR`: The directory where generated images will be saved (default: `"generated_images"` within the `client_examples` folder).
*   **Payload:** Modify the `payload` dictionary within the `if __name__ == "__main__":` block:
    *   `prompt`: The text description for the image.
    *   `binding_name`: **Required.** The name of your configured TTI binding instance from `config.toml`.
    *   `model_name` (optional): Specify a model if your TTI binding supports multiple (e.g., `"dall-e-3"`, `"sdxl"`).
    *   `parameters` (optional): Include TTI-specific parameters like `size`, `quality`, `style`, `negative_prompt`, `steps`, etc., depending on what your chosen binding supports.
    *   `generation_type` must be `"tti"`.
    *   `stream` should be `false` (TTI is non-streaming).

## Running the Example

1.  Make sure your `lollms_server` is running and correctly configured with a working TTI binding.
2.  Ensure your Python environment has the `requests` and `Pillow` libraries installed.
3.  Navigate to the `lollms_server` project root directory in your terminal.
4.  Run the script:
    ```bash
    python client_examples/image_client.py
    ```

## Expected Behavior

1.  The script sends a POST request to the `/generate` endpoint with `generation_type: "tti"` and your configured payload.
2.  It waits for the server to process the request and generate the image.
3.  If successful, the server returns a JSON response containing an `image_base64` field.
4.  The script decodes the base64 string.
5.  It saves the decoded image as a PNG file in the specified `OUTPUT_DIR`, using a filename based on the current timestamp.
6.  Metadata from the response (excluding the base64 string) is printed to the console.

## Notes

*   Ensure the `binding_name` in the payload matches exactly the name you defined under `[bindings]` in your `config.toml`.
*   The specific `parameters` you can use depend entirely on the TTI binding you are targeting. Consult the binding's documentation or implementation.