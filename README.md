# lollms_server
A multi modal lollms personalities compatible web server

## Client Examples

This project includes example Python clients in the `client_examples/` directory to demonstrate how to interact with the `lollms_server` API.

**Installation:**

Before running the examples, install their specific dependencies:

```bash
pip install -r client_examples/requirements.txt
```

**Configuration:**

Edit the client script files (`simple_client.py`, `streaming_client.py`, `image_client.py`) and update the following variables at the top:

*   `BASE_URL`: The address of your running `lollms_server` (e.g., `http://localhost:9600/api/v1`).
*   `API_KEY`: A valid API key configured in your server's `config.toml` under `[security].allowed_api_keys`.

Also, ensure the `personality`, `binding_name`, and `model_name` values used within the example payloads match personalities and bindings configured and loaded on your server.

**Running Examples:**

Navigate to the project root directory in your terminal and run the examples like standard Python scripts:

```bash
# Example: Run the simple client
python client_examples/simple_client.py

# Example: Run the streaming client
python client_examples/streaming_client.py

# Example: Run the image generation client
python client_examples/image_client.py
```