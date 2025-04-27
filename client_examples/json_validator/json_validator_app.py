import gradio as gr
import json
import io # Used for handling the file object efficiently

def validate_json_file(file_obj):
    """
    Validates the content of an uploaded JSON file.

    Args:
        file_obj: A Gradio file object (contains temp file path).

    Returns:
        A string indicating whether the JSON is valid or detailing the error.
        Formatted as Markdown for better display.
    """
    if file_obj is None:
        return "<p style='color:orange;'>⚠️ Please upload a JSON file first.</p>"

    try:
        # Gradio file_obj gives access to the temporary file path via .name
        file_path = file_obj.name

        # Read the content of the uploaded file
        # Use utf-8-sig to handle potential BOM (Byte Order Mark)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            json_content = f.read()

        # Handle empty file case
        if not json_content.strip():
             return "<p style='color:orange;'>⚠️ The uploaded file is empty.</p>"

        # Attempt to parse the JSON content
        json.loads(json_content)

        # If parsing succeeds, the JSON is valid
        return "<p style='color:green; font-weight:bold;'>✅ JSON is valid!</p>"

    except json.JSONDecodeError as e:
        # If parsing fails, capture the error details
        error_message = f"""
        <p style='color:red; font-weight:bold;'>❌ JSON Invalid!</p>
        <hr>
        <p><b>Error:</b> {e.msg}</p>
        <p><b>Line:</b> {e.lineno}</p>
        <p><b>Column:</b> {e.colno}</p>
        """
        # Optionally show context (be careful with large files/sensitive data)
        # lines = json_content.splitlines()
        # context_start = max(0, e.lineno - 3)
        # context_end = min(len(lines), e.lineno + 2)
        # context = "\n".join(lines[context_start:context_end])
        # error_message += f"<p><b>Context (around line {e.lineno}):</b></p><pre><code>{context}</code></pre>"

        return error_message

    except FileNotFoundError:
        return "<p style='color:red;'>❌ Error: Could not find the uploaded temporary file.</p>"
    except Exception as e:
        # Catch any other unexpected errors during file reading/processing
        return f"<p style='color:red;'>❌ An unexpected error occurred: {str(e)}</p>"

# --- Create the Gradio Interface ---
iface = gr.Interface(
    fn=validate_json_file,
    inputs=gr.File(
        label="Upload JSON File",
        file_types=[".json"]  # Restrict to .json files
        ),
    outputs=gr.Markdown(label="Validation Result"), # Use Markdown for better formatting
    title="JSON File Validator",
    description="Upload a `.json` file to check if its syntax is correct. Errors will be reported with line and column numbers.",
    examples=[
        # You can create small example files to link here if needed
        # ["./example_valid.json"],
        # ["./example_invalid.json"]
    ],
    allow_flagging='never' # Optional: Disables the flagging feature
)

# --- Launch the App ---
if __name__ == "__main__":
    print("Starting Gradio app... Access it at the URL provided below.")
    iface.launch()
    # Use iface.launch(share=True) if you want a temporary public link