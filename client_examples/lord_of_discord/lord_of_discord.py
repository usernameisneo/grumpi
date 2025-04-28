# client_examples/lord_of_discord.py
import pipmaster as pm
pm.ensure_packages({
    "discord":"",
    "requests":"",
    "sseclient-py":"",
    "Pillow":"",
    "ascii_colors":"",
}) # Corrected package name

import discord
from discord.ext import commands
from discord import app_commands, SelectOption, Embed, Color, File, Intents, Game, HTTPException, ui, LoginFailure

import requests
import json
import sys
import os
import base64
import time
import datetime
from pathlib import Path
from io import BytesIO
import asyncio
import tempfile
from typing import List, Dict, Optional, Any, Union, Callable, Coroutine, Literal
from ascii_colors import ASCIIColors # For colored console output
import ascii_colors as logging

logger = logging.getLogger(__name__) # Use the same logger hierarchy

# --- Dependency Check ---
try:
    # Ensure core dependencies are checked (already done above)
    from PIL import Image, UnidentifiedImageError
    print("Core dependencies seem okay.")
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependency - {e}")
    print("Please install required packages manually: pip install discord.py requests sseclient-py Pillow ascii_colors")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during dependency check: {e}")
    sys.exit(1)


# --- Configuration Loading ---
CONFIG_FILE = Path("bot_config.json")
SETTINGS_FILE = Path("discord_settings.json")
IMAGE_DIR = Path("generated_images")

def ensure_bot_config() -> bool:
    """Ensures bot_config.json exists and loads token/url/key."""
    global DISCORD_TOKEN, BASE_URL, API_KEY, HEADERS_NO_STREAM
    if not CONFIG_FILE.exists():
        ASCIIColors.error(f"'{CONFIG_FILE}' not found.")
        ASCIIColors.yellow("Please create it with the following content:")
        print("""
{
  "discord_token": "YOUR_DISCORD_BOT_TOKEN_HERE",
  "lollms_base_url": "http://localhost:9600",
  "lollms_api_key": "YOUR_LOLLMS_SERVER_API_KEY_HERE"
}
""")
        return False
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        DISCORD_TOKEN = config.get("discord_token")
        BASE_URL = config.get("lollms_base_url", "http://localhost:9600").rstrip('/') + "/api/v1"
        API_KEY = config.get("lollms_api_key")
        if not DISCORD_TOKEN or not API_KEY:
            ASCIIColors.error("ERROR: 'discord_token' or 'lollms_api_key' missing in bot_config.json")
            return False
        # Define headers now that API_KEY is loaded
        HEADERS_NO_STREAM = { "X-API-Key": API_KEY, "Content-Type": "application/json", "Accept": "application/json" }
        ASCIIColors.info("Bot configuration loaded.")
        return True
    except Exception as e:
        ASCIIColors.error(f"Error loading '{CONFIG_FILE}': {e}")
        return False

# --- Global Variables & Defaults ---
DISCORD_TOKEN: Optional[str] = None
BASE_URL: str = "http://localhost:9600/api/v1" # Will be updated by ensure_bot_config
API_KEY: Optional[str] = None
DEFAULT_TIMEOUT = 120
DEFAULT_TTT_BINDING = None; DEFAULT_TTT_MODEL = None
DEFAULT_TTI_BINDING = None; DEFAULT_TTI_MODEL = None
HEADERS_NO_STREAM: Dict[str, str] = {}
current_personality: Optional[str] = None; current_ttt_binding: Optional[str] = None
current_ttt_model: Optional[str] = None; current_tti_binding: Optional[str] = DEFAULT_TTI_BINDING
current_tti_model: Optional[str] = DEFAULT_TTI_MODEL; initial_settings_setup_needed: bool = False

# --- Helper Functions ---
def build_discord_context_string(interaction: discord.Interaction) -> str:
    """Builds a string containing context about the Discord environment from the bot's perspective."""
    bot_user = interaction.client.user
    user = interaction.user
    channel = interaction.channel
    guild = interaction.guild

    # --- Changed phrasing to "I am" ---
    context_parts = [
        "## My Current Discord Context",
        f"I am running as a Discord bot named **'{bot_user.display_name if bot_user else 'UnknownBot'}'** (ID: `{bot_user.id if bot_user else 'N/A'}`).",
        f"I am currently interacting with user **'{user.display_name}'** (Username: `{user.name}`, ID: `{user.id}`)."
    ]

    if isinstance(channel, discord.TextChannel):
        context_parts.append(f"This interaction is taking place in the text channel **'#{channel.name}'** (ID: `{channel.id}`).")
        if channel.topic:
            context_parts.append(f"* Channel Topic: \"{channel.topic}\"")
    elif isinstance(channel, discord.Thread):
        context_parts.append(f"This interaction is taking place in the thread **'{channel.name}'** (ID: `{channel.id}`) inside the channel '{channel.parent.name if channel.parent else 'Unknown'}'.")
    elif isinstance(channel, discord.DMChannel):
        context_parts.append("This interaction is taking place in a Direct Message.")
    else:
        context_parts.append(f"This interaction is taking place in a channel of type **'{type(channel).__name__}'** (ID: `{channel.id if channel else 'N/A'}`).")

    if guild:
        context_parts.append(f"We are on the server (guild) named **'{guild.name}'** (ID: `{guild.id}`).")
    else:
         context_parts.append("This interaction is happening outside of a server (e.g., in a DM).")

    # --- Changed phrasing for capabilities ---
    context_parts.extend([
        "\n## My Capabilities Here",
        "Users interact with me using Slash Commands:",
        "- `/lollms [prompt]`: For general chat and text generation (like this interaction).",
        "- `/lollms_imagine [prompt]`: To generate an image based on a description.",
        "- `/lollms_settings [subcommand]`: (Owner only) To configure my backend settings.",
        "I should use this context to provide relevant and helpful responses within this Discord environment." # More direct instruction
    ])

    return "\n".join(context_parts)

# Use ASCIIColors for logging
def log_system(message): ASCIIColors.info(f"🤖 SYSTEM: {message}")
def log_info(message): ASCIIColors.info(f"ℹ️ INFO: {message}")
def log_warning(message): ASCIIColors.warning(f"⚠️ WARNING: {message}")
def log_error(message): ASCIIColors.error(f"❌ ERROR: {message}")

def load_json_file(filepath: Path, default: Any = None) -> Any:
    """Loads JSON data from a file."""
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except json.JSONDecodeError: log_error(f"Failed decode {filepath}. Using defaults."); return default
        except Exception as e: log_error(f"Error loading {filepath}: {e}"); return default
    return default

def save_json_file(filepath: Path, data: Any) -> bool:
    """Saves data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e: log_error(f"Error saving {filepath}: {e}"); return False

def load_operational_settings() -> Dict[str, Optional[str]]:
    """Loads bot operational settings."""
    defaults = { "personality": None, "ttt_binding": None, "ttt_model": None, "tti_binding": DEFAULT_TTI_BINDING, "tti_model": DEFAULT_TTI_MODEL, }
    loaded = load_json_file(SETTINGS_FILE, default={})
    if not isinstance(loaded, dict): log_error(f"Settings file {SETTINGS_FILE} invalid. Using defaults."); loaded = {}
    defaults.update(loaded); return defaults

def save_operational_settings(settings: Dict[str, Optional[str]]):
    """Saves bot operational settings."""
    if not save_json_file(SETTINGS_FILE, settings): log_error("Failed to save operational settings!")
    else: log_system("Operational settings saved.")

def get_current_operational_settings() -> Dict[str, Optional[str]]:
    """Returns current operational settings."""
    return { "personality": current_personality, "ttt_binding": current_ttt_binding, "ttt_model": current_ttt_model, "tti_binding": current_tti_binding, "tti_model": current_tti_model, }

def update_current_operational_settings(settings: Dict[str, Optional[str]]):
    """Updates global operational settings."""
    global current_personality, current_ttt_binding, current_ttt_model, current_tti_binding, current_tti_model
    current_personality = settings.get("personality"); current_ttt_binding = settings.get("ttt_binding")
    current_ttt_model = settings.get("ttt_model"); current_tti_binding = settings.get("tti_binding", DEFAULT_TTI_BINDING)
    current_tti_model = settings.get("tti_model", DEFAULT_TTI_MODEL); log_system("Operational settings updated in memory.")

# --- API Client Functions ---
async def make_api_call_async(endpoint: str, method: str = "GET", payload: Optional[Dict] = None) -> Optional[Any]:
    """Makes async non-streaming API calls using requests in executor."""
    url = f"{BASE_URL}{endpoint}"
    loop = asyncio.get_running_loop()
    try:
        # Use a new session per call or manage sessions better if needed
        with requests.Session() as session:
            if method.upper() == "GET":
                response = await loop.run_in_executor(None, lambda: session.get(url, headers=HEADERS_NO_STREAM, timeout=DEFAULT_TIMEOUT))
            elif method.upper() == "POST":
                response = await loop.run_in_executor(None, lambda: session.post(url, headers=HEADERS_NO_STREAM, json=payload, timeout=DEFAULT_TIMEOUT))
            else: log_error(f"Unsupported HTTP method: {method}"); return None
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        log_error(f"API call failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None: log_error(f" Server Response: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e: log_error(f"Unexpected error during API call to {url}: {e}"); return None

async def make_generate_request_async(payload: Dict[str, Any]) -> Optional[Union[str, Dict]]:
    """Makes async non-streaming /generate requests."""
    url = f"{BASE_URL}/generate"
    headers = HEADERS_NO_STREAM
    payload['stream'] = False # Ensure stream is false for bot

    # --- VALIDATE/ENSURE input_data structure ---
    if "input_data" not in payload or not isinstance(payload["input_data"], list) or not payload["input_data"]:
        # Attempt to convert old prompt field if present
        if "prompt" in payload and isinstance(payload["prompt"], str):
             log_warning("Converting legacy 'prompt' field to 'input_data'.")
             payload["input_data"] = [{"type": "text", "role": "user_prompt", "data": payload["prompt"]}]
             del payload["prompt"]
        else:
            log_error("Generate request payload missing valid 'input_data' list.")
            return None
    # --- END VALIDATION ---

    loop = asyncio.get_running_loop()
    try:
        # Use requests within executor for thread safety if needed, or httpx if preferred
        with requests.Session() as session:
            response = await loop.run_in_executor(None, lambda: session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json() # Expect JSON for non-stream generate
    except requests.exceptions.Timeout:
        log_error(f"Request timed out to {url}")
    except requests.exceptions.ConnectionError as e:
        log_error(f"Could not connect to server at {url}. Details: {e}")
    except requests.exceptions.RequestException as e:
        log_error(f"Generate request failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            log_error(f" Server Response: {e.response.status_code} - {e.response.text[:200]}")
    except Exception as e:
        log_error(f"Unexpected error during generate request: {e}")
    return None # Indicate failure

# --- Discord Bot Setup ---
intents = Intents.default(); intents.guilds = True; intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Helper for Splitting Messages ---
def split_message(content: str, max_len: int = 1980) -> List[str]:
    """Splits long messages for Discord."""
    if len(content) <= max_len: return [content]
    chunks = []; current_chunk = ""; lines = content.splitlines(keepends=True)
    for line in lines:
        if len(current_chunk) + len(line) > max_len: chunks.append(current_chunk); current_chunk = line
        else: current_chunk += line
    if current_chunk: chunks.append(current_chunk)
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_len: final_chunks.append(chunk[:max_len]); chunk = chunk[max_len:]
        if chunk: final_chunks.append(chunk)
    return final_chunks

# --- Bot Events ---
@bot.event
async def on_ready():
    """Called when the bot is ready."""
    global initial_settings_setup_needed
    log_system(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    if not SETTINGS_FILE.exists():
        log_warning(f"'{SETTINGS_FILE}' not found. Run /lollms_settings setup"); initial_settings_setup_needed = True; save_operational_settings({})
    else: initial_settings_setup_needed = False
    update_current_operational_settings(load_operational_settings())
    log_system('Syncing slash commands...')
    try:
        await bot.change_presence(activity=Game(name="/lollms help"))
        bot.tree.add_command(settings_group)
        synced = await bot.tree.sync()
        log_system(f"Synced {len(synced)} slash commands globally.")
    except Exception as e: log_error(f"Failed to sync commands: {e}")
    log_system('Lord of Discord is ready.')
    if initial_settings_setup_needed: log_system("Waiting for owner: /lollms_settings setup")
    else: log_system(f"Current Settings: P:{current_personality or 'Default'} B:{current_ttt_binding or 'Default'} M:{current_ttt_model or 'Default'}")

# --- Slash Commands (/lollms, /lollms_imagine) ---
@bot.tree.command(name="lollms", description="Chat with the LOLLMS agent.")
@app_commands.describe(prompt="Your message or question for the agent.")
async def lollms_chat(interaction: discord.Interaction, prompt: str):
    """Handles the main chat command, now supporting multimodal responses."""
    try:
        await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e:
        log_error(f"Defer failed for /lollms: {e}")
        return

    # Build Discord Context (as before)
    discord_context = build_discord_context_string(interaction)

    payload = {
        "input_data": [
             {"type": "text", "role": "system_context", "data": discord_context},
             {"type": "text", "role": "system_context", "data": "Acknowledge that you are a Discord bot in your response and use the provided context."},
             {"type": "text", "role": "user_prompt", "data": prompt}
        ],
        "generation_type": "ttt", # Keep as ttt, personality decides output types
        "personality": current_personality,
        "binding_name": current_ttt_binding,
        "model_name": current_ttt_model,
        "stream": False # Bot uses non-streaming
    }
    log_info(f"Sending payload to /generate:\nPersonality: {current_personality}\nBinding: {current_ttt_binding}\nModel: {current_ttt_model}")

    result = await make_generate_request_async(payload)

    # --- Process potentially multimodal response ---
    response_parts: List[Union[str, discord.File]] = [] # Store text strings and file objects
    reply_status_emoji = "💡"
    ai_display_name = current_personality or "Default"
    final_error_message: Optional[str] = None
    temp_files_to_clean: List[Path] = [] # Keep track of temp files

    if isinstance(result, dict) and "output" in result:
        output_list = result.get("output", [])
        if not output_list: # Handle empty output list
             final_error_message = "(Received empty response from server)"
             reply_status_emoji = "🤷"
        else:
            for item in output_list:
                if not isinstance(item, dict): continue # Skip invalid items

                item_type = item.get("type")
                item_data = item.get("data")
                item_metadata = item.get("metadata", {})

                if item_type == "text" and isinstance(item_data, str):
                    response_parts.append(item_data.strip())
                elif item_type == "image" and isinstance(item_data, str):
                    try:
                        img_data = base64.b64decode(item_data)
                        img_buffer = BytesIO(img_data)
                        mime_type = item.get("mime_type", "image/png")
                        file_extension = mime_type.split('/')[-1] if mime_type else 'png'
                        img_prompt_used = item_metadata.get("prompt_used", "generated_image")

                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}", dir=IMAGE_DIR) as tmp_f:
                             # Pillow check optional here, discord.File handles bytes directly
                             tmp_f.write(img_buffer.read())
                             tmp_file_path = Path(tmp_f.name)
                             temp_files_to_clean.append(tmp_file_path) # Add to cleanup list

                        if tmp_file_path and tmp_file_path.exists():
                             safe_filename = "".join(c for c in img_prompt_used if c.isalnum() or c in (' ', '_')).rstrip()[:50].replace(' ','_') + f".{file_extension}"
                             discord_file = File(tmp_file_path, filename=safe_filename)
                             response_parts.append(discord_file) # Add the File object
                             response_parts.append(f"*(Image prompt used: `{img_prompt_used[:150]}{'...' if len(img_prompt_used)>150 else ''}`)*") # Add context
                        else:
                            response_parts.append("(Error saving image temporarily for upload)")
                            reply_status_emoji = "⚠️"
                    except Exception as img_err:
                         log_error(f"Failed processing image item: {img_err}")
                         response_parts.append(f"(Error processing image: {img_err})")
                         reply_status_emoji = "⚠️"
                elif item_type == "error" and isinstance(item_data, str):
                    response_parts.append(f"**Error from Generation:** {item_data}")
                    reply_status_emoji = "❌"
                # Add elif for 'audio', 'video' etc. if needed
                else:
                    # Append unhandled types as text/code block for visibility
                    response_parts.append(f"*(Unsupported output type: {item_type})*\n```json\n{json.dumps(item, indent=2)[:500]}\n```")

    elif isinstance(result, dict): # Missing 'output' key
        final_error_message = f"Unexpected JSON structure (missing 'output'):\n```json\n{json.dumps(result, indent=2)[:1500]}\n```"
        reply_status_emoji = "⚠️"
    elif result is None: # API call failed
        final_error_message = "Failed to get response from LOLLMS server."
        reply_status_emoji = "❌"
    else: # Unexpected return type
        final_error_message = f"Unexpected response type: {type(result)}"
        log_error(f"Unexpected type from API: {type(result)}")
        reply_status_emoji = "❌"

    # Now we have the emoji and display name determined
    reply_prefix = f"👤 **{interaction.user.display_name}:** {prompt}\n\n{reply_status_emoji} **({ai_display_name}) AI:**"
    # --- Construct and Send Multi-Part Reply ---
    base_prefix = f"👤 **{interaction.user.display_name}:** {prompt}\n\n{reply_status_emoji} **({ai_display_name}) AI:**"
    message_parts_to_send: List[Union[str, discord.File]] = []
    current_text_content = base_prefix

    # Add the error message first if it exists
    if final_error_message:
        current_text_content += "\n" + final_error_message

    # Append text parts and collect files
    files_to_attach: List[discord.File] = []
    for part in response_parts:
        if isinstance(part, str):
            # Check if adding this text part exceeds limit
            if len(current_text_content) + len(part) + 1 > 2000: # +1 for newline
                 # Save current text part, start new one
                 message_parts_to_send.append(current_text_content)
                 current_text_content = part # Start new text block
            else:
                 current_text_content += "\n" + part # Add text to current block
        elif isinstance(part, discord.File):
            files_to_attach.append(part) # Collect files

    # Add the last text part
    if current_text_content != base_prefix or not files_to_attach: # Add if it contains more than just the prefix, or if there are no files (need to send something)
         message_parts_to_send.append(current_text_content.strip())

    # Send messages
    try:
        first_message = True
        for i, part_content in enumerate(message_parts_to_send):
             # Attach all files to the *last* message part if possible,
             # or the first if it's the only one. Discord limitation.
             attach_files_now = files_to_attach if (i == len(message_parts_to_send) - 1) else []

             if first_message:
                 # Ensure content isn't empty if sending only files
                 content_to_send = part_content if part_content else (reply_prefix if not final_error_message else final_error_message) # Send prefix or error if text is empty
                 if len(content_to_send) > 2000: content_to_send=content_to_send[:1997]+"..." # Truncate if somehow still too long
                 await interaction.followup.send(content_to_send, files=attach_files_now or None)
                 first_message = False
             else:
                 # Send subsequent parts in the channel
                 if len(part_content) > 2000: part_content=part_content[:1997]+"..."
                 await interaction.channel.send(part_content, files=attach_files_now or None) # type: ignore

    except HTTPException as e:
        log_error(f"Failed sending multipart response to Discord: {e}")
        try: await interaction.followup.send("❌ Error: Could not send the full response to Discord.", ephemeral=True)
        except Exception: pass
    finally:
        # Clean up all temporary files
        for f_path in temp_files_to_clean:
             if f_path.exists():
                 try: f_path.unlink(); logger.debug(f"Cleaned up temp file: {f_path}")
                 except OSError as e_unlink: log_error(f"Error deleting temp file {f_path}: {e_unlink}")

@bot.tree.command(name="lollms_imagine", description="Generate an image with LOLLMS.")
@app_commands.describe(prompt="The description of the image to generate.")
async def lollms_imagine(interaction: discord.Interaction, prompt: str):
    """Handles the image generation command."""
    try:
        await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e:
        log_error(f"Defer failed for /lollms_imagine: {e}")
        return

    # Build Discord Context (as before)
    discord_context = build_discord_context_string(interaction)

    payload = {
        "input_data": [
             {"type": "text", "role": "system_context", "data": discord_context},
             {"type": "text", "role": "user_prompt", "data": prompt}
        ],
        "generation_type": "tti",
        "binding_name": current_tti_binding,
        "model_name": current_tti_model,
        "stream": False
    }
    log_info(f"Sending payload to /generate for TTI:\nTTI Binding: {current_tti_binding}\nTTI Model: {current_tti_model}")

    response_data = await make_generate_request_async(payload)

    tmp_file_path: Optional[Path] = None
    status_emoji = "🖼️"
    # --- Construct initial reply prefix (without long prompt yet) ---
    reply_prefix = f"👤 **{interaction.user.display_name}:** /imagine {prompt[:100]}{'...' if len(prompt)>100 else ''}\n\n{status_emoji} **Image Result:**" # Truncate original prompt display
    reply_content = reply_prefix # Start with the prefix
    file_to_send: Optional[File] = None
    error_occurred = False

    try:
        # Process response structure
        if response_data and isinstance(response_data, dict) and "output" in response_data:
            output_list = response_data.get("output", [])
            image_found = False

            for item in output_list:
                if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                    image_b64 = item["data"]
                    metadata = item.get("metadata", {})
                    mime_type = item.get("mime_type", "image/png")
                    # Use prompt from metadata if available, truncate if needed for display
                    prompt_used = metadata.get("prompt_used", prompt)
                    prompt_display = prompt_used[:150] + ('...' if len(prompt_used) > 150 else '') # Truncate prompt used display
                    file_extension = mime_type.split('/')[-1] if mime_type else 'png'

                    try:
                        img_data = base64.b64decode(image_b64)
                        img_buffer = BytesIO(img_data)
                        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}", dir=IMAGE_DIR) as tmp_f:
                            try:
                                img = Image.open(img_buffer)
                                img.save(tmp_f, format=img.format or file_extension.upper())
                            except (UnidentifiedImageError, ValueError, OSError):
                                log_warning(f"Pillow couldn't identify/save image type '{mime_type}', saving raw bytes.")
                                img_buffer.seek(0)
                                tmp_f.write(img_buffer.read())
                            tmp_file_path = Path(tmp_f.name)

                        if tmp_file_path and tmp_file_path.exists():
                             safe_filename = "".join(c for c in prompt_used if c.isalnum() or c in (' ', '_')).rstrip()[:50].replace(' ','_') + f".{file_extension}"
                             file_to_send = File(tmp_file_path, filename=safe_filename)
                             # --- Add potentially truncated prompt to reply content ---
                             reply_content += f"\n*(Prompt used: `{prompt_display}`)*"
                             # -------------------------------------------------------
                             image_found = True
                        else:
                             status_emoji = "❌"; reply_content += "\nError saving image temporarily."; error_occurred = True

                    except (base64.binascii.Error, UnidentifiedImageError) as img_err:
                         log_error(f"Invalid image data received: {img_err}"); status_emoji = "❌"; reply_content += f"\nError: Invalid image data received ({type(img_err).__name__})."; error_occurred = True
                    except Exception as e:
                         log_error(f"Error processing image: {e}"); status_emoji = "❌"; reply_content += f"\nError processing image: {e}"; error_occurred = True
                    break # Process first image

                elif isinstance(item, dict) and item.get("type") == "error":
                     error_msg = item.get('data', 'Unknown error from server.')
                     status_emoji = "❌"; reply_content += f"\nError: Server failed: {error_msg}"; error_occurred = True
                     image_found = True
                     break

            if not image_found and not error_occurred:
                status_emoji = "🤷"; reply_content += "\nNo image data found in the server response."
                error_occurred = True

        elif isinstance(response_data, dict):
            status_emoji = "❌"; reply_content += f"\nError: Unexpected JSON structure from server:\n```\n{json.dumps(response_data, indent=2)[:1000]}\n```"; error_occurred = True
        else:
            status_emoji = "❌"; reply_content += "\nError: Failed to get valid response from LOLLMS server."; error_occurred = True

        # --- Check length *before* sending ---
        MAX_DISCORD_LEN = 2000
        if len(reply_content) > MAX_DISCORD_LEN:
            log_warning(f"Reply content exceeds {MAX_DISCORD_LEN} chars, truncating...")
            # Truncate, keeping the prefix and adding an indicator
            chars_to_keep = MAX_DISCORD_LEN - len("... (content truncated)")
            reply_content = reply_content[:chars_to_keep] + "... (content truncated)"
            # We might lose the "(Prompt used: ...)" part, but better than failing

        # Send the final message
        if file_to_send:
            await interaction.followup.send(reply_content, file=file_to_send)
        else:
            # Even if there was an error processing image, send the text part
            await interaction.followup.send(reply_content)

    finally:
        if tmp_file_path and tmp_file_path.exists():
            try: tmp_file_path.unlink()
            except OSError as e: log_error(f"Error deleting temp image {tmp_file_path}: {e}")

# --- Settings Select Menu Helper ---
async def create_classic_select_menu( interaction_or_ctx: Union[discord.Interaction], setting_type: str, fetch_endpoint: str, current_value: Optional[str], list_json_key: Optional[str], display_key: str = 'name', value_key: str = 'name', is_dict_list: bool = True, allow_none: bool = True, none_display_text: str = "Use Server Default", on_complete_callback: Optional[Callable[[discord.Interaction, Optional[str]], Coroutine[Any, Any, None]]] = None ) -> Optional[ui.View]:
    """Creates a Discord Select menu view for settings."""
    log_system(f"Creating select menu for: {setting_type}")
    options_data = await make_api_call_async(fetch_endpoint)
    items = []
    if options_data:
        if list_json_key:
            raw_list_data = options_data.get(list_json_key)
            if is_dict_list and isinstance(raw_list_data, dict): items = list(raw_list_data.values())
            elif not is_dict_list and isinstance(raw_list_data, dict): items = list(raw_list_data.keys())
            elif isinstance(raw_list_data, list): items = raw_list_data
        elif isinstance(options_data, list): items = options_data
        else: log_warning(f"Unexpected API response for {setting_type}.")
    if not items and not allow_none: log_warning(f"No items for {setting_type}, menu cannot be created."); return None
    select_options: List[SelectOption] = []
    if allow_none: select_options.append(SelectOption(label=none_display_text, value="__NONE__", default=current_value is None))
    for item in items:
        label = None; value = None
        if is_dict_list and isinstance(item, dict): label = item.get(display_key); value = item.get(value_key)
        elif isinstance(item, str): label = item; value = item
        if label and value: select_options.append(SelectOption(label=label[:100], value=value[:100], default=value == current_value))
    if not select_options: log_warning(f"No valid options for {setting_type}"); return None
    if len(select_options)>25: log_warning("Truncating options to 25."); select_options = select_options[:25]

    class SettingsSelect(ui.Select):
        def __init__(self): super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)
        async def callback(self, interaction: discord.Interaction):
            selected_value = self.values[0]; final_value = None if selected_value == "__NONE__" else selected_value
            log_system(f"{setting_type} selected: {final_value}")
            global current_personality, current_ttt_binding, current_ttt_model, current_tti_binding, current_tti_model
            updated = False
            if setting_type == "Personality": current_personality = final_value; updated = True
            elif setting_type == "TTT Binding":
                if current_ttt_binding != final_value: current_ttt_binding = final_value; current_ttt_model = None; updated = True
            elif setting_type == "TTT Model": current_ttt_model = final_value; updated = True
            elif setting_type == "TTI Binding":
                if current_tti_binding != final_value: current_tti_binding = final_value; current_tti_model = None; updated = True
            elif setting_type == "TTI Model": current_tti_model = final_value; updated = True
            if updated: save_operational_settings(get_current_operational_settings())
            await interaction.response.edit_message(content=f"✅ {setting_type} set to: `{final_value or none_display_text}`", view=None)
            if on_complete_callback: await on_complete_callback(interaction, final_value)

    view = ui.View(timeout=180); view.add_item(SettingsSelect()); return view

async def create_select_menu(
    interaction_or_ctx: Union[discord.Interaction],
    setting_type: str,
    # Removed fetch_endpoint, data will be passed in
    # fetch_endpoint: str,
    items_data: Union[List, Dict], # Pass the actual data (list or dict)
    binding_types_data: Optional[Dict] = None, # Pass binding type info when filtering
    filter_capability: Optional[Literal['tti', 'ttt', 'tts', 'stt']] = None, # New filter param
    current_value: Optional[str] = None,
    # Removed list_json_key and is_dict_list, handle structure based on type
    # list_json_key: Optional[str],
    display_key: str = 'name',
    value_key: str = 'name',
    # is_dict_list: bool = True, # No longer needed
    allow_none: bool = True,
    none_display_text: str = "Use Server Default",
    on_complete_callback: Optional[Callable[[discord.Interaction, Optional[str]], Coroutine[Any, Any, None]]] = None
) -> Optional[ui.View]:
    """Creates a Discord Select menu view for settings, with optional capability filtering."""
    log_system(f"Creating select menu for: {setting_type}")
    select_options: List[SelectOption] = []
    valid_items_map: Dict[str, str] = {} # Store value -> label mapping

    # Add the "None" option first if allowed
    if allow_none:
        select_options.append(SelectOption(label=none_display_text, value="__NONE__", default=current_value is None))

    items_to_process = []
    if isinstance(items_data, dict):
        items_to_process = list(items_data.items()) # List of (key, value_dict) tuples
    elif isinstance(items_data, list):
        items_to_process = items_data # List of dicts or strings

    if not items_to_process:
         log_warning(f"No items provided for {setting_type} menu.")
         # Return view with only None option if applicable
         if allow_none:
            view = ui.View(timeout=180)
            # Need to redefine the Select class locally if we return here
            class EmptySettingsSelect(ui.Select):
                def __init__(self): super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)
                async def callback(self, interaction: discord.Interaction): await interaction.response.edit_message(content=f"No valid options available.", view=None)
            view.add_item(EmptySettingsSelect())
            return view
         else:
            return None # No items and None not allowed

    # --- Filtering Logic ---
    for item_key_or_obj in items_to_process:
        label = None
        value = None
        item_config = None # Store config for capability check

        if isinstance(item_key_or_obj, tuple): # Came from a dict
            value, item_config = item_key_or_obj
            label = value # Use key as label/value by default for binding instances
        elif isinstance(item_key_or_obj, dict): # Came from a list of dicts (like models, personalities)
            value = item_key_or_obj.get(value_key)
            label = item_key_or_obj.get(display_key) or value
            item_config = item_key_or_obj # The item itself might have config/details
        elif isinstance(item_key_or_obj, str): # Came from a list of strings
            value = item_key_or_obj
            label = value

        if not label or not value:
            log_warning(f"Skipping invalid item for {setting_type}: {item_key_or_obj}")
            continue

        # --- Apply Capability Filter ---
        should_include = True
        if filter_capability and setting_type.endswith("Binding"): # Only filter bindings for now
            if binding_types_data and isinstance(item_config, dict):
                binding_type_name = item_config.get("type")
                binding_type_info = binding_types_data.get(binding_type_name) if binding_type_name else None
                if binding_type_info and isinstance(binding_type_info, dict):
                    # Check if the binding type supports the required output modality
                    supported_outputs = binding_type_info.get("supported_output_modalities", [])
                    if filter_capability == 'tti' and 'image' not in supported_outputs:
                        should_include = False
                        logger.debug(f"Filtering out binding '{value}' (type: {binding_type_name}): Doesn't support 'image' output.")
                    elif filter_capability == 'ttt' and 'text' not in supported_outputs:
                        should_include = False
                        logger.debug(f"Filtering out binding '{value}' (type: {binding_type_name}): Doesn't support 'text' output.")
                    # Add elif for tts (audio), stt (text input? maybe filter input too?) etc.
                else:
                    log_warning(f"Could not find type info for binding '{value}' (type: {binding_type_name}) to apply filter.")
                    # Optionally filter out if info missing? Or include? Let's include for now.
            else:
                log_warning(f"Cannot filter binding '{value}': Missing type info or binding_types_data.")
        # --- End Filtering ---

        if should_include:
             valid_items_map[value] = label


    # Create options from filtered items
    for value, label in valid_items_map.items():
         select_options.append(SelectOption(label=label[:100], value=value[:100], default=value == current_value))

    # Check if any valid options remain after filtering (besides potentially "__NONE__")
    if len(select_options) <= (1 if allow_none else 0):
         log_warning(f"No valid options remain for {setting_type} after filtering for '{filter_capability}'.")
         # Return menu with only None option if applicable
         if allow_none:
            view = ui.View(timeout=180)
            class FilteredEmptySettingsSelect(ui.Select):
                 def __init__(self): super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)
                 async def callback(self, interaction: discord.Interaction): await interaction.response.edit_message(content=f"No options match the filter '{filter_capability}'.", view=None)
            view.add_item(FilteredEmptySettingsSelect())
            return view
         else:
            return None # No items and None not allowed

    if len(select_options) > 25:
        log_warning(f"Truncating options for {setting_type} to 25.")
        select_options = select_options[:25]

    # --- Define Select Class and View ---
    class SettingsSelect(ui.Select):
        def __init__(self):
            super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)

        async def callback(self, interaction: discord.Interaction):
            selected_value = self.values[0]
            final_value = None if selected_value == "__NONE__" else selected_value
            # Use valid_items_map to get the display label corresponding to the selected value
            display_label = none_display_text if final_value is None else valid_items_map.get(final_value, final_value) # Fallback to value if label not found

            log_system(f"{setting_type} selected: {final_value}")
            global current_personality, current_ttt_binding, current_ttt_model, current_tti_binding, current_tti_model
            updated = False
            setting_changed = False # Track if the specific setting being targeted was changed

            # Update global state based on setting_type
            if setting_type == "Personality":
                if current_personality != final_value: setting_changed = True
                current_personality = final_value; updated = True
            elif setting_type == "TTT Binding":
                if current_ttt_binding != final_value: current_ttt_binding = final_value; current_ttt_model = None; updated = True; setting_changed = True
            elif setting_type == "TTT Model":
                if current_ttt_model != final_value: setting_changed = True
                current_ttt_model = final_value; updated = True
            elif setting_type == "TTI Binding":
                 if current_tti_binding != final_value: current_tti_binding = final_value; current_tti_model = None; updated = True; setting_changed = True
            elif setting_type == "TTI Model":
                 if current_tti_model != final_value: setting_changed = True
                 current_tti_model = final_value; updated = True

            if updated:
                save_operational_settings(get_current_operational_settings())

            # Only call the callback if the setting actually changed and a callback exists
            if setting_changed and on_complete_callback:
                 await interaction.response.edit_message(content=f"✅ {setting_type} set to: `{display_label}`. Processing next step...", view=None)
                 await on_complete_callback(interaction, final_value)
            else:
                 # Just confirm the selection if no callback or setting didn't change
                 await interaction.response.edit_message(content=f"✅ {setting_type} set to: `{display_label}`", view=None)

    view = ui.View(timeout=180)
    view.add_item(SettingsSelect())
    return view

# --- Initial Setup Wizard ---
async def run_initial_setup_wizard(interaction: discord.Interaction):
    """Guides the owner through the initial setup."""
    log_system("Starting initial setup wizard...")
    await interaction.response.send_message("Starting initial setup wizard...", ephemeral=True)
    await setup_step_personality(interaction)

async def setup_step_personality(interaction: discord.Interaction):
    """Setup Step 1: Select Personality."""
    view = await create_classic_select_menu(interaction, "Personality", "/list_personalities", None, 'personalities', is_dict_list=True, allow_none=True, none_display_text="None (No Personality)", on_complete_callback=setup_step_ttt_binding)
    if view: await interaction.followup.send("Step 1: Select Personality:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error creating personality menu.", ephemeral=True)

async def setup_step_ttt_binding(interaction: discord.Interaction, _selected_personality: Optional[str]):
    """Setup Step 2: Select TTT Binding."""
    view = await create_classic_select_menu(interaction, "TTT Binding", "/list_bindings", None, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default", on_complete_callback=setup_step_ttt_model)
    if view: await interaction.followup.send("Step 2: Select TTT Binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error creating TTT binding menu.", ephemeral=True)

async def setup_step_ttt_model(interaction: discord.Interaction, selected_ttt_binding: Optional[str]):
    """Setup Step 3: Select TTT Model."""
    if not selected_ttt_binding: await setup_step_tti_binding(interaction, None); return
    view = await create_classic_select_menu(interaction, "TTT Model", f"/list_available_models/{selected_ttt_binding}", None, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{selected_ttt_binding}'", on_complete_callback=setup_step_tti_binding)
    if view: await interaction.followup.send(f"Step 3: Select TTT Model for `{selected_ttt_binding}`:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error TTT model menu.", ephemeral=True); await setup_step_tti_binding(interaction, None)

async def setup_step_tti_binding(interaction: discord.Interaction, _selected_ttt_model: Optional[str]):
    """Setup Step 4: Select TTI Binding."""
    view = await create_classic_select_menu(interaction, "TTI Binding", "/list_bindings", None, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default", on_complete_callback=setup_step_tti_model)
    if view: await interaction.followup.send("Step 4: Select TTI Binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error creating TTI binding menu.", ephemeral=True); await finish_setup(interaction, None) # Proceed even on error

async def setup_step_tti_model(interaction: discord.Interaction, selected_tti_binding: Optional[str]):
    """Setup Step 5: Select TTI Model."""
    if not selected_tti_binding: await finish_setup(interaction, None); return
    view = await create_classic_select_menu(interaction, "TTI Model", f"/list_available_models/{selected_tti_binding}", None, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{selected_tti_binding}'", on_complete_callback=finish_setup)
    if view: await interaction.followup.send(f"Step 5: Select TTI Model for `{selected_tti_binding}`:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error TTI model menu.", ephemeral=True); await finish_setup(interaction, None)

async def finish_setup(interaction: discord.Interaction, _selected_tti_model: Optional[str]):
    """Final step of the setup wizard."""
    global initial_settings_setup_needed
    initial_settings_setup_needed = False
    save_operational_settings(get_current_operational_settings())
    await interaction.followup.send("✅ Initial setup complete! Settings saved.", ephemeral=True)
    log_system("Initial setup wizard completed.")

# --- Settings Command Group & Subcommands ---
settings_group = app_commands.Group(name="lollms_settings", description="View or setup Lord of Discord settings.")

@settings_group.command(name="setup", description="Run the initial setup wizard (owner only).")
@app_commands.checks.has_permissions()
async def setup_wizard(interaction: discord.Interaction):
    """Starts the setup wizard."""
    if initial_settings_setup_needed: await run_initial_setup_wizard(interaction)
    else: await interaction.response.send_message("✅ Setup already completed.", ephemeral=True)

@settings_group.command(name="view", description="View current LOLLMS settings (owner only).")
@app_commands.checks.has_permissions()
async def view_settings(interaction: discord.Interaction):
    """Displays the current settings."""
    await interaction.response.defer(ephemeral=True)
    if initial_settings_setup_needed: await interaction.followup.send("Run `/lollms_settings setup` first.", ephemeral=True); return
    settings = get_current_operational_settings(); embed = Embed(title="Current LOLLMS Settings", color=Color.blue())
    embed.add_field(name="Personality", value=f"`{settings['personality'] or 'None'}`", inline=False); embed.add_field(name="TTT Binding", value=f"`{settings['ttt_binding'] or 'Default'}`", inline=False); embed.add_field(name="TTT Model", value=f"`{settings['ttt_model'] or 'Default'}`", inline=False); embed.add_field(name="TTI Binding", value=f"`{settings['tti_binding'] or 'Default'}`", inline=False); embed.add_field(name="TTI Model", value=f"`{settings['tti_model'] or 'Default'}`", inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)

@settings_group.command(name="set_personality", description="Select active personality (owner only).")
@app_commands.checks.has_permissions()
async def set_personality(interaction: discord.Interaction):
    """Allows setting the personality."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    view = await create_classic_select_menu(interaction, "Personality", "/list_personalities", current_personality, 'personalities', is_dict_list=True, allow_none=True, none_display_text="None (No Personality)")
    if view: await interaction.followup.send("Choose personality:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_ttt_binding", description="Select TTT binding (owner only).")
@app_commands.checks.has_permissions()
async def set_ttt_binding(interaction: discord.Interaction):
    """Allows setting the TTT binding."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    view = await create_classic_select_menu(interaction, "TTT Binding", "/list_bindings", current_ttt_binding, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default")
    if view: await interaction.followup.send("Choose TTT binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_ttt_model", description="Select TTT model (owner only).")
@app_commands.checks.has_permissions()
async def set_ttt_model(interaction: discord.Interaction):
    """Allows setting the TTT model."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    if not current_ttt_binding: await interaction.followup.send("❌ Select TTT Binding first.", ephemeral=True); return
    view = await create_classic_select_menu(interaction, "TTT Model", f"/list_available_models/{current_ttt_binding}", current_ttt_model, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{current_ttt_binding}'")
    if view: await interaction.followup.send(f"Choose TTT model for `{current_ttt_binding}`:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_tti_binding", description="Select TTI binding (owner only).")
@app_commands.checks.has_permissions()
async def set_tti_binding(interaction: discord.Interaction):
    """Allows setting the TTI binding, filtering for image output capability."""
    if initial_settings_setup_needed:
        await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    # Fetch full binding data first
    bindings_data = await make_api_call_async("/list_bindings")
    if not bindings_data:
        await interaction.followup.send("❌ Error fetching binding list from server.", ephemeral=True)
        return

    binding_instances = bindings_data.get("binding_instances", {})
    binding_types = bindings_data.get("binding_types", {}) # Need type info for filtering

    # Create the menu, passing data and filter criteria
    view = await create_select_menu(
        interaction,
        setting_type="TTI Binding",
        items_data=binding_instances, # Pass the instances dict
        binding_types_data=binding_types, # Pass the types dict
        filter_capability='tti', # Filter for image output
        current_value=current_tti_binding,
        # display_key/value_key not needed when items_data is dict
        allow_none=True,
        none_display_text="Server Default",
        on_complete_callback=lambda i, val: asyncio.create_task(reset_tti_model_on_binding_change(i, val))
    )

    if view:
        await interaction.followup.send("Choose TTI binding (showing image-capable):", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu or no TTI bindings found.", ephemeral=True)

async def reset_tti_model_on_binding_change(interaction: discord.Interaction, new_binding_name: Optional[str]):
    """Callback to reset TTI model when TTI binding changes."""
    global current_tti_model
    if current_tti_binding != new_binding_name: # Check if it actually changed
        log_info(f"TTI binding changed to '{new_binding_name}', resetting TTI model.")
        current_tti_model = None
        save_operational_settings(get_current_operational_settings())
        # No need to send message here, the menu callback already confirms the binding change

@settings_group.command(name="set_tti_model", description="Select TTI model (owner only).")
@app_commands.checks.has_permissions()
async def set_tti_model(interaction: discord.Interaction):
    """Allows setting the TTI model."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)

    # --- FIX: Check and use current_tti_binding ---
    if not current_tti_binding:
        await interaction.followup.send("❌ Select TTI Binding first using `/lollms_settings set_tti_binding`.", ephemeral=True)
        return

    # --- FIX: Use current_tti_binding in the endpoint URL ---
    api_endpoint = f"/list_available_models/{current_tti_binding}"

    view = await create_classic_select_menu(
        interaction,
        "TTI Model",
        api_endpoint, # Use the corrected endpoint
        current_tti_model, # Use the current TTI model value
        'models', # Key in JSON response
        is_dict_list=True, # Model list has dicts
        allow_none=True,
        # --- FIX: Use current_tti_binding in the display text ---
        none_display_text=f"Default for '{current_tti_binding}'"
    )
    if view:
        # --- FIX: Use current_tti_binding in the display text ---
        await interaction.followup.send(f"Choose TTI model for `{current_tti_binding}`:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu (maybe the selected TTI binding has no models?).", ephemeral=True)

# --- Error Handling ---
@bot.event
async def on_application_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Handles application command errors."""
    respond_method = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
    if isinstance(error, app_commands.CommandNotFound): pass
    elif isinstance(error, app_commands.CheckFailure) or isinstance(error, commands.NotOwner): await respond_method("❌ You don't have permission.", ephemeral=True)
    elif isinstance(error, app_commands.MissingRequiredArgument): await respond_method(f"❌ Missing argument: `{error.param.name}`.", ephemeral=True)
    elif isinstance(error, HTTPException) and error.code == 50035: await respond_method("❌ Error: Response too long.", ephemeral=True)
    else:
        cmd_name = interaction.command.qualified_name if interaction.command else 'unknown'
        log_error(f"Unhandled error in '{cmd_name}': {error}")
        try: await respond_method("❌ An unexpected error occurred.", ephemeral=True)
        except Exception as resp_err: log_error(f"Failed sending error response: {resp_err}")

# --- Run Bot ---
if __name__ == "__main__":
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    log_system("Initializing Lord of Discord...")
    if not ensure_bot_config(): sys.exit(1)
    log_system("Starting Discord Bot...")
    if not DISCORD_TOKEN: log_error("Discord Token missing."); sys.exit(1)
    try: bot.run(DISCORD_TOKEN)
    except LoginFailure: log_error("Login Failed. Check Discord token.")
    except Exception as e: log_error(f"Bot runtime error: {e}")