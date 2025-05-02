# client_examples/lord_of_discord.py
import pipmaster as pm
pm.ensure_packages({
    "discord.py": ">=2.0.0",
    "requests": "",
    "sseclient-py": "",
    "Pillow": "",
    "ascii_colors": ">=0.11.2",
})

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

import ascii_colors as logging
from ascii_colors import ASCIIColors, Menu, trace_exception

logger = logging.getLogger("lord_of_discord")

try:
    from PIL import Image, UnidentifiedImageError
    ASCIIColors.info("Core dependencies seem okay.")
except ImportError as e:
    ASCIIColors.error(f"CRITICAL ERROR: Missing dependency - {e}")
    ASCIIColors.error("Please ensure required packages are installed: pip install discord.py requests sseclient-py Pillow ascii_colors")
    sys.exit(1)
except Exception as e:
    ASCIIColors.error(f"CRITICAL ERROR during dependency check: {e}")
    trace_exception(e)
    sys.exit(1)

CONFIG_FILE = Path("bot_config.json")
SETTINGS_FILE = Path("discord_settings.json")
IMAGE_DIR = Path("generated_images")
HISTORY_LIMIT = 15 # Max number of past messages to include in context


DISCORD_TOKEN: Optional[str] = None
BASE_URL: str = "http://localhost:9600"
API_KEY: Optional[str] = None
DEFAULT_TIMEOUT = 120
DEFAULT_TTT_BINDING = None
DEFAULT_TTT_MODEL = None
DEFAULT_TTI_BINDING = None
DEFAULT_TTI_MODEL = None

current_personality: Optional[str] = None
current_ttt_binding: Optional[str] = None
current_ttt_model: Optional[str] = None
current_tti_binding: Optional[str] = DEFAULT_TTI_BINDING
current_tti_model: Optional[str] = DEFAULT_TTI_MODEL
initial_settings_setup_needed: bool = False

def build_discord_context_string(interaction: discord.Interaction) -> str:
    """Builds a string containing context about the Discord environment from the bot's perspective."""
    bot_user = interaction.client.user
    user = interaction.user
    channel = interaction.channel
    guild = interaction.guild
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

    context_parts.extend([
        "\n## My Capabilities Here",
        "Users interact with me using Slash Commands:",
        "- `/lollms [prompt]`: For general chat and text generation (like this interaction).",
        "- `/lollms_imagine [prompt]`: To generate an image based on a description.",
        "- `/lollms_settings [subcommand]`: (Owner only) To configure my backend settings.",
        "I should use this context to provide relevant and helpful responses within this Discord environment."
    ])
    return "\n".join(context_parts)

def log_system(message):
    ASCIIColors.info(f"🤖 SYSTEM: {message}")

def log_debug(message):
    ASCIIColors.debug(f"ℹ️ DEBUG: {message}")

def log_info(message):
    ASCIIColors.info(f"ℹ️ INFO: {message}")

def log_warning(message):
    ASCIIColors.warning(f"⚠️ WARNING: {message}")

def log_error(message):
    ASCIIColors.error(f"❌ ERROR: {message}")

def load_json_file(filepath: Path, default: Any = None) -> Any:
    """Loads JSON data from a file."""
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            log_error(f"Failed decode {filepath}. Returning default.")
            return default
        except Exception as e:
            log_error(f"Error loading {filepath}: {e}")
            return default
    return default

def save_json_file(filepath: Path, data: Any) -> bool:
    """Saves data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_error(f"Error saving {filepath}: {e}")
        return False
async def build_history_context(interaction: discord.Interaction, limit: int = HISTORY_LIMIT) -> str:
    """Fetches and formats recent message history from the channel."""
    if not interaction.channel:
        return "" # Cannot fetch history without a channel

    history_str_parts = ["## Recent Conversation History (Oldest first)"]
    try:
        # Fetch messages before the current interaction command
        messages = [msg async for msg in interaction.channel.history(limit=limit + 1) if msg.id != interaction.id]
        messages.reverse() # Oldest first

        if not messages:
            history_str_parts.append("(No recent messages found)")
            return "\n".join(history_str_parts)

        bot_user = interaction.client.user
        human_user = interaction.user

        for msg in messages:
            # Determine role prefix
            role_prefix = ""
            if msg.author == bot_user:
                role_prefix = "Assistant:"
            elif msg.author == human_user:
                role_prefix = "User:"
            else:
                role_prefix = f"OtherUser ({msg.author.display_name}):" # Include other users for context

            # Get message content
            content = msg.content or "(Message had no text content)"
            if msg.attachments:
                content += " " + " ".join(f"[Attachment: {att.filename}]" for att in msg.attachments)

            history_str_parts.append(f"{role_prefix} {content.strip()}")

        return "\n".join(history_str_parts)

    except discord.Forbidden:
        log_error("Missing 'Read Message History' permission to build context.")
        return "## Recent Conversation History\n(Error: Missing permission to read history)"
    except Exception as e:
        log_error(f"Error fetching message history: {e}")
        trace_exception(e)
        return f"## Recent Conversation History\n(Error fetching history: {e})"
    

def run_configuration_wizard() -> bool:
    """Runs an interactive wizard to create bot_config.json."""
    ASCIIColors.red("--- Lord of Discord Configuration Wizard ---")
    ASCIIColors.yellow("Configuration file 'bot_config.json' not found.")
    ASCIIColors.print("Let's set up the basic connection details.")
    config_data = {}
    while True:
        token = ASCIIColors.prompt("Enter your Discord Bot Token (required): ", hide_input=True).strip()
        if token:
            config_data["discord_token"] = token
            break
        else:
            ASCIIColors.warning("Discord Token cannot be empty.")
    default_url = "http://localhost:9600"
    url_input = ASCIIColors.prompt(f"Enter the LOLLMS Server Base URL [Default: {default_url}]: ").strip()
    config_data["lollms_base_url"] = url_input or default_url
    api_key_input = ASCIIColors.prompt("Enter your LOLLMS Server API Key (leave blank if none required): ", hide_input=True).strip()
    config_data["lollms_api_key"] = api_key_input or None
    ASCIIColors.print("\n--- Configuration Summary ---")
    ASCIIColors.print(f"Discord Token: {'*' * len(config_data['discord_token'])}")
    ASCIIColors.print(f"LOLLMS URL:    {config_data['lollms_base_url']}")
    ASCIIColors.print(f"LOLLMS API Key: {'Present' if config_data['lollms_api_key'] else 'Not Provided'}")
    if ASCIIColors.confirm("\nSave this configuration to bot_config.json?", default_yes=True):
        if save_json_file(CONFIG_FILE, config_data):
            ASCIIColors.success(f"Configuration saved to {CONFIG_FILE}")
            return True
        else:
            ASCIIColors.error("Failed to save configuration file.")
            return False
    else:
        ASCIIColors.yellow("Configuration not saved. Exiting.")
        return False

def ensure_bot_config() -> bool:
    """Ensures bot_config.json exists (runs wizard if not) and loads token/url/key."""
    global DISCORD_TOKEN, BASE_URL, API_KEY
    if not CONFIG_FILE.exists():
        if not run_configuration_wizard():
            return False
    config = load_json_file(CONFIG_FILE)
    if not isinstance(config, dict):
        log_error(f"Invalid format in {CONFIG_FILE}. Please fix or delete it to re-run the wizard.")
        return False
    DISCORD_TOKEN = config.get("discord_token")
    BASE_URL = config.get("lollms_base_url", "http://localhost:9600").rstrip('/') + "/api/v1"
    API_KEY = config.get("lollms_api_key")
    if not DISCORD_TOKEN:
        log_error(f"ERROR: 'discord_token' missing or empty in {CONFIG_FILE}.")
        return False
    log_info("Bot configuration loaded successfully.")
    log_info(f"  > LOLLMS URL: {BASE_URL}")
    log_info(f"  > API Key: {'Present' if API_KEY else 'Not Provided'}")
    return True

def load_operational_settings() -> Dict[str, Optional[str]]:
    """Loads bot operational settings."""
    defaults = { "personality": None, "ttt_binding": None, "ttt_model": None, "tti_binding": DEFAULT_TTI_BINDING, "tti_model": DEFAULT_TTI_MODEL, }
    loaded = load_json_file(SETTINGS_FILE, default={})
    if not isinstance(loaded, dict):
        log_error(f"Settings file {SETTINGS_FILE} invalid. Using defaults.")
        loaded = {}
    defaults.update(loaded)
    return defaults

def save_operational_settings(settings: Dict[str, Optional[str]]):
    """Saves bot operational settings."""
    if not save_json_file(SETTINGS_FILE, settings):
        log_error("Failed to save operational settings!")
    else:
        log_system("Operational settings saved.")

def get_current_operational_settings() -> Dict[str, Optional[str]]:
    """Returns current operational settings."""
    return { "personality": current_personality, "ttt_binding": current_ttt_binding, "ttt_model": current_ttt_model, "tti_binding": current_tti_binding, "tti_model": current_tti_model, }

def update_current_operational_settings(settings: Dict[str, Optional[str]]):
    """Updates global operational settings."""
    global current_personality, current_ttt_binding, current_ttt_model, current_tti_binding, current_tti_model
    current_personality = settings.get("personality")
    current_ttt_binding = settings.get("ttt_binding")
    current_ttt_model = settings.get("ttt_model")
    current_tti_binding = settings.get("tti_binding", DEFAULT_TTI_BINDING)
    current_tti_model = settings.get("tti_model", DEFAULT_TTI_MODEL)
    log_system("Operational settings updated in memory.")

def _get_api_headers(include_key: bool = True) -> Dict[str, str]:
    """Constructs headers, conditionally including API key."""
    headers = { "Content-Type": "application/json", "Accept": "application/json" }
    if include_key and API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers

async def make_api_call_async(endpoint: str, method: str = "GET", payload: Optional[Dict] = None) -> Optional[Any]:
    """Makes async non-streaming API calls using requests in executor."""
    url = f"{BASE_URL}{endpoint}"
    loop = asyncio.get_running_loop()
    headers = _get_api_headers()
    try:
        with requests.Session() as session:
            if method.upper() == "GET":
                response = await loop.run_in_executor(None, lambda: session.get(url, headers=headers, timeout=DEFAULT_TIMEOUT))
            elif method.upper() == "POST":
                response = await loop.run_in_executor(None, lambda: session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT))
            else:
                log_error(f"Unsupported HTTP method: {method}")
                return None
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        log_error(f"API call failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 401 or e.response.status_code == 403:
                log_error(" -> Authorization Error: Check if server requires an API key and if it's correct in bot_config.json")
            else:
                log_error(f" Server Response: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        log_error(f"Unexpected error during API call to {url}: {e}")
        return None

async def make_generate_request_async(payload: Dict[str, Any]) -> Optional[Union[str, Dict]]:
    """Makes async non-streaming /generate requests."""
    url = f"{BASE_URL}/generate"
    headers = _get_api_headers()
    payload['stream'] = False
    if "input_data" not in payload or not isinstance(payload["input_data"], list) or not payload["input_data"]:
        if "prompt" in payload and isinstance(payload["prompt"], str):
            log_warning("Converting legacy 'prompt' field to 'input_data'.")
            payload["input_data"] = [{"type": "text", "role": "user_prompt", "data": payload["prompt"]}]
            del payload["prompt"]
        else:
            log_error("Generate request payload missing valid 'input_data' list.")
            return None
    loop = asyncio.get_running_loop()
    try:
        with requests.Session() as session:
            response = await loop.run_in_executor(None, lambda: session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        log_error(f"Request timed out to {url}")
    except requests.exceptions.ConnectionError as e:
        log_error(f"Could not connect to server at {url}. Details: {e}")
    except requests.exceptions.RequestException as e:
        log_error(f"Generate request failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 401 or e.response.status_code == 403:
                log_error(" -> Authorization Error: Check API key.")
            else:
                log_error(f" Server Response: {e.response.status_code} - {e.response.text[:200]}")
    except Exception as e:
        log_error(f"Unexpected error during generate request: {e}")
    return None

intents = Intents.default()
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

def split_message(content: str, max_len: int = 1980) -> List[str]:
    """Splits long messages for Discord."""
    if len(content) <= max_len:
        return [content]
    chunks = []
    current_chunk = ""
    lines = content.splitlines(keepends=True)
    for line in lines:
        if len(current_chunk) + len(line) > max_len:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line
    if current_chunk:
        chunks.append(current_chunk)
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_len:
            final_chunks.append(chunk[:max_len])
            chunk = chunk[max_len:]
        if chunk:
            final_chunks.append(chunk)
    return final_chunks

@bot.event
async def on_ready():
    """Called when the bot is ready."""
    global initial_settings_setup_needed
    log_system(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    if not SETTINGS_FILE.exists():
        log_warning(f"'{SETTINGS_FILE}' not found. Operational settings will use defaults.")
        initial_settings_setup_needed = True
        save_operational_settings({})
    else:
        initial_settings_setup_needed = False
    update_current_operational_settings(load_operational_settings())
    log_system('Syncing slash commands...')
    try:
        await bot.change_presence(activity=Game(name="/lollms help"))
        bot.tree.add_command(settings_group)
        synced = await bot.tree.sync()
        log_system(f"Synced {len(synced)} slash commands globally.")
    except Exception as e:
        log_error(f"Failed to sync commands: {e}")
    log_system('Lord of Discord is ready.')
    if initial_settings_setup_needed:
        log_system("Waiting for owner to run: /lollms_settings setup")
    else:
        log_system(f"Current Settings: P:{current_personality or 'Default'} | TTT B:{current_ttt_binding or 'Default'} M:{current_ttt_model or 'Default'} | TTI B:{current_tti_binding or 'Default'} M:{current_tti_model or 'Default'}")

@bot.tree.command(name="lollms", description="Chat with the LOLLMS agent.")
@app_commands.describe(prompt="Your message or question for the agent.")
async def lollms_chat(interaction: discord.Interaction, prompt: str):
    """Handles the main chat command, now supporting multimodal responses and history."""
    try:
        await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e:
        log_error(f"Defer failed for /lollms: {e}")
        # Try sending an ephemeral message if defer fails
        try:
            await interaction.response.send_message("Processing your request...", ephemeral=True)
        except Exception:
            log_error("Failed to send ephemeral thinking message either.")
            return # Cannot proceed if we can't acknowledge
        # Note: If defer fails, we can't use interaction.followup later,
        # we'd have to use interaction.edit_original_response() or send new messages to channel.
        # For simplicity, we'll assume defer usually works.

    discord_context = build_discord_context_string(interaction)
    history_context = await build_history_context(interaction) # <<< Fetch history

    # --- Build input_data with history ---
    input_items = [
        {"type": "text", "role": "system_context", "data": discord_context},
        {"type": "text", "role": "system_context", "data": "Acknowledge that you are a Discord bot in your response and use the provided context."},
        {"type": "text", "role": "system_context", "data": history_context}, # <<< Add history here
        {"type": "text", "role": "user_prompt", "data": prompt}
    ]
    # --- End input_data build ---

    payload = {
        "input_data": input_items,
        "generation_type": "ttt",
        "personality": current_personality,
        "binding_name": current_ttt_binding,
        "model_name": current_ttt_model,
        "stream": False
    }

    log_info(f"Sending payload to /generate:\nPersonality: {current_personality}\nBinding: {current_ttt_binding}\nModel: {current_ttt_model}")
    log_debug(f"History context sent (first 100 chars): {history_context[:100]}") # Log start of history

    result = await make_generate_request_async(payload)

    response_parts: List[Union[str, discord.File]] = []
    reply_status_emoji = "💡"
    ai_display_name = current_personality or "Default"
    final_error_message: Optional[str] = None
    temp_files_to_clean: List[Path] = []

    if isinstance(result, dict) and "output" in result:
        output_list = result.get("output", [])
        if not output_list:
            final_error_message = "(Received empty response from server)"
            reply_status_emoji = "🤷"
        else:
            for item in output_list:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                item_data = item.get("data")
                item_metadata = item.get("metadata", {})
                item_thoughts = item.get("thoughts") # <<< Get thoughts

                # --- Handle Thoughts ---
                if item_thoughts and isinstance(item_thoughts, str):
                    # Format thoughts inside Discord spoiler tags
                    formatted_thoughts = f"||```\n🧠 Thinking:\n{item_thoughts.strip()}\n```||"
                    response_parts.append(formatted_thoughts)
                # --- End Handle Thoughts ---

                if item_type == "text" and isinstance(item_data, str):
                    response_parts.append(item_data.strip()) # Append main text after thoughts
                elif item_type == "image" and isinstance(item_data, str):
                    # --- Image handling (remains the same) ---
                    try:
                        img_data = base64.b64decode(item_data)
                        img_buffer = BytesIO(img_data)
                        mime_type = item.get("mime_type", "image/png")
                        file_extension = mime_type.split('/')[-1] if mime_type else 'png'
                        img_prompt_used = item_metadata.get("prompt_used", "generated_image")
                        IMAGE_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}", dir=IMAGE_DIR) as tmp_f:
                            tmp_f.write(img_buffer.read())
                            tmp_file_path = Path(tmp_f.name)
                            temp_files_to_clean.append(tmp_file_path)
                        if tmp_file_path and tmp_file_path.exists():
                            safe_filename = "".join(c for c in img_prompt_used if c.isalnum() or c in (' ', '_')).rstrip()[:50].replace(' ','_') + f".{file_extension}"
                            discord_file = File(tmp_file_path, filename=safe_filename)
                            response_parts.append(discord_file)
                            response_parts.append(f"*(Image prompt used: `{img_prompt_used[:150]}{'...' if len(img_prompt_used)>150 else ''}`)*")
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
                # --- Consider adding handling for other types if needed ---
                elif item_type not in ["text", "image", "error"]: # Handle unknown types gracefully
                    response_parts.append(f"*(Unsupported output type received: {item_type})*\n```json\n{json.dumps(item, indent=2)[:500]}\n```")

    elif isinstance(result, dict): # Handle case where API returned dict but no 'output' key
        final_error_message = f"Unexpected JSON structure (missing 'output'):\n```json\n{json.dumps(result, indent=2)[:1500]}\n```"
        reply_status_emoji = "⚠️"
    elif result is None: # Handle API call failure
        final_error_message = "Failed to get response from LOLLMS server. Check server logs and connection."
        reply_status_emoji = "❌"
    else: # Handle unexpected return type from API call
        final_error_message = f"Unexpected response type: {type(result)}"
        log_error(f"Unexpected type from API: {type(result)}")
        reply_status_emoji = "❌"

    # --- Message Sending Logic (adjusted slightly for clarity) ---
    base_prefix = f"👤 **{interaction.user.display_name}:** {prompt}\n\n{reply_status_emoji} **({ai_display_name}) AI:**"
    message_parts_to_send: List[Union[str, discord.File]] = []
    current_text_content = "" # Start with empty text part
    files_to_attach: List[discord.File] = []

    # Prepend the prefix to the first text part if available, otherwise send it alone
    first_text_added = False
    for part in response_parts:
        if isinstance(part, str):
            if not first_text_added:
                current_text_content += base_prefix + "\n" + part # Add prefix before first text
                first_text_added = True
            else:
                 current_text_content += "\n" + part # Add subsequent text with newline
        elif isinstance(part, discord.File):
            files_to_attach.append(part) # Collect files

    # If no text was added but there's a prefix (e.g., only images/thoughts generated)
    if not first_text_added and base_prefix:
        current_text_content = base_prefix

    # Add final error message if one occurred
    if final_error_message:
        current_text_content += ("\n" if current_text_content else "") + final_error_message

    # Split the potentially long text content into valid Discord messages
    if current_text_content:
        text_chunks = split_message(current_text_content.strip())
        message_parts_to_send.extend(text_chunks)
    elif not files_to_attach: # Handle case where absolutely nothing was generated
         message_parts_to_send.append(base_prefix + "\n(No response content generated)")


    # Send the messages/files
    try:
        first_message_sent = False
        for i, part_content_or_file in enumerate(message_parts_to_send):
            # Attach all files only to the *last* message part containing text
            # or send files alone if there's no text.
            attach_files_now: Optional[List[discord.File]] = None
            content_to_send: Optional[str] = None

            if isinstance(part_content_or_file, str):
                content_to_send = part_content_or_file
                if i == len(message_parts_to_send) - 1: # If this is the last part
                    attach_files_now = files_to_attach
            # This loop structure assumes text comes first, then files are attached at the end.
            # A more robust approach might send text parts first, then files in a final separate message if needed.
            # Let's stick to attaching files to the last text part for now.

            if content_to_send or attach_files_now: # Only send if there's something to send
                if not first_message_sent:
                    await interaction.followup.send(content=content_to_send, files=attach_files_now or None)
                    first_message_sent = True
                else:
                    # Use interaction.channel.send for subsequent parts
                    # Ensure channel is TextChannel or similar
                    if isinstance(interaction.channel, (discord.TextChannel, discord.Thread, discord.DMChannel)):
                         await interaction.channel.send(content=content_to_send, files=attach_files_now or None)
                    else:
                        log_warning("Cannot send followup messages in this channel type.")
                        # Send remaining files in the initial followup if possible
                        if attach_files_now and not first_message_sent:
                           await interaction.followup.send(files=attach_files_now)
                           first_message_sent = True # Mark as sent

        # If there were files but no text parts to attach them to
        if files_to_attach and not first_message_sent:
             await interaction.followup.send(files=files_to_attach)

    except HTTPException as e:
        log_error(f"Failed sending response to Discord: {e} (Code: {e.code}, Status: {e.status})")
        # Try to send a simple error message if possible
        try:
            error_followup_msg = "❌ Error: Could not send the full response to Discord."
            if e.code == 50035: # Value exceeds maximum length
                 error_followup_msg += " (Response was too long)"
            await interaction.followup.send(error_followup_msg, ephemeral=True)
        except Exception:
            pass # Avoid error loops
    finally:
        # Cleanup temporary image files
        for f_path in temp_files_to_clean:
            if f_path.exists():
                try:
                    f_path.unlink()
                    log_debug(f"Cleaned up temp file: {f_path}")
                except OSError as e_unlink:
                    log_error(f"Error deleting temp file {f_path}: {e_unlink}")

@bot.tree.command(name="lollms_imagine", description="Generate an image with LOLLMS.")
@app_commands.describe(prompt="The description of the image to generate.")
async def lollms_imagine(interaction: discord.Interaction, prompt: str):
    """Handles the image generation command."""
    try:
        await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e:
        log_error(f"Defer failed for /lollms_imagine: {e}")
        return

    discord_context = build_discord_context_string(interaction)
    history_context = await build_history_context(interaction) # <<< Fetch history (Optional)

    # --- Build input_data with history ---
    input_items = [
        {"type": "text", "role": "system_context", "data": discord_context},
        {"type": "text", "role": "system_context", "data": history_context}, # <<< Add history (Optional)
        {"type": "text", "role": "user_prompt", "data": prompt}
    ]
    # --- End input_data build ---

    payload = {
        "input_data": input_items, # Use updated input
        "generation_type": "tti",
        "binding_name": current_tti_binding,
        "model_name": current_tti_model,
        "stream": False
    }
    log_info(f"Sending payload to /generate for TTI:\nTTI Binding: {current_tti_binding}\nTTI Model: {current_tti_model}")
    response_data = await make_generate_request_async(payload)
    tmp_file_path: Optional[Path] = None
    status_emoji = "🖼️"
    reply_prefix = f"👤 **{interaction.user.display_name}:** /imagine {prompt[:100]}{'...' if len(prompt)>100 else ''}\n\n{status_emoji} **Image Result:**"
    reply_content = reply_prefix
    file_to_send: Optional[File] = None
    error_occurred = False

    try:
        if response_data and isinstance(response_data, dict) and "output" in response_data:
            output_list = response_data.get("output", [])
            image_found = False
            for item in output_list:
                if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                    image_b64 = item["data"]
                    metadata = item.get("metadata", {})
                    mime_type = item.get("mime_type", "image/png")
                    prompt_used = metadata.get("prompt_used", prompt)
                    prompt_display = prompt_used[:150] + ('...' if len(prompt_used) > 150 else '')
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
                            reply_content += f"\n*(Prompt used: `{prompt_display}`)*"
                            image_found = True
                        else:
                            status_emoji = "❌"
                            reply_content += "\nError saving image temporarily."
                            error_occurred = True
                    except (base64.binascii.Error, UnidentifiedImageError) as img_err:
                        log_error(f"Invalid image data received: {img_err}")
                        status_emoji = "❌"
                        reply_content += f"\nError: Invalid image data received ({type(img_err).__name__})."
                        error_occurred = True
                    except Exception as e:
                        log_error(f"Error processing image: {e}")
                        status_emoji = "❌"
                        reply_content += f"\nError processing image: {e}"
                        error_occurred = True
                    break
                elif isinstance(item, dict) and item.get("type") == "error":
                    error_msg = item.get('data', 'Unknown error from server.')
                    status_emoji = "❌"
                    reply_content += f"\nError: Server failed: {error_msg}"
                    error_occurred = True
                    image_found = True
                    break
            if not image_found and not error_occurred:
                status_emoji = "🤷"
                reply_content += "\nNo image data found in the server response."
                error_occurred = True
        elif isinstance(response_data, dict):
            status_emoji = "❌"
            reply_content += f"\nError: Unexpected JSON structure from server:\n```\n{json.dumps(response_data, indent=2)[:1000]}\n```"
            error_occurred = True
        else:
            status_emoji = "❌"
            reply_content += "\nError: Failed to get valid response from LOLLMS server."
            error_occurred = True

        MAX_DISCORD_LEN = 2000
        if len(reply_content) > MAX_DISCORD_LEN:
            log_warning(f"Reply content exceeds {MAX_DISCORD_LEN} chars, truncating...")
            chars_to_keep = MAX_DISCORD_LEN - len("... (content truncated)")
            reply_content = reply_content[:chars_to_keep] + "... (content truncated)"

        if file_to_send:
            await interaction.followup.send(reply_content, file=file_to_send)
        else:
            await interaction.followup.send(reply_content)
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
            except OSError as e:
                log_error(f"Error deleting temp image {tmp_file_path}: {e}")

async def create_select_menu(
    interaction_or_ctx: Union[discord.Interaction],
    setting_type: str,
    items_data: Union[List, Dict],
    binding_types_data: Optional[Dict] = None,
    filter_capability: Optional[Literal['tti', 'ttt', 'tts', 'stt']] = None,
    current_value: Optional[str] = None,
    display_key: str = 'name',
    value_key: str = 'name',
    allow_none: bool = True,
    none_display_text: str = "Use Server Default",
    on_complete_callback: Optional[Callable[[discord.Interaction, Optional[str]], Coroutine[Any, Any, None]]] = None
) -> Optional[ui.View]:
    """Creates a Discord Select menu view for settings, with optional capability filtering."""
    log_system(f"Creating select menu for: {setting_type}")
    select_options: List[SelectOption] = []
    valid_items_map: Dict[str, str] = {}
    if allow_none:
        select_options.append(SelectOption(label=none_display_text, value="__NONE__", default=current_value is None))
    items_to_process = []
    if isinstance(items_data, dict):
        items_to_process = list(items_data.items())
    elif isinstance(items_data, list):
        items_to_process = items_data
    if not items_to_process:
        log_warning(f"No items provided for {setting_type} menu.")
        if allow_none:
            view = ui.View(timeout=180)
            class EmptySettingsSelect(ui.Select):
                def __init__(self):
                    super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)
                async def callback(self, interaction: discord.Interaction):
                    await interaction.response.edit_message(content=f"No valid options available.", view=None)
            view.add_item(EmptySettingsSelect())
            return view
        else:
            return None
    for item_key_or_obj in items_to_process:
        label = None
        value = None
        item_config = None
        if isinstance(item_key_or_obj, tuple):
            value, item_config = item_key_or_obj
            label = value
        elif isinstance(item_key_or_obj, dict):
            value = item_key_or_obj.get(value_key)
            label = item_key_or_obj.get(display_key) or value
            item_config = item_key_or_obj
        elif isinstance(item_key_or_obj, str):
            value = item_key_or_obj
            label = value
        if not label or not value:
            log_warning(f"Skipping invalid item for {setting_type}: {item_key_or_obj}")
            continue
        should_include = True
        if filter_capability and setting_type.endswith("Binding"):
            if binding_types_data and isinstance(item_config, dict):
                binding_type_name = item_config.get("type")
                binding_type_info = binding_types_data.get(binding_type_name) if binding_type_name else None
                if binding_type_info and isinstance(binding_type_info, dict):
                    supported_outputs = binding_type_info.get("supported_output_modalities", [])
                    if filter_capability == 'tti' and 'image' not in supported_outputs:
                        should_include = False
                        logger.debug(f"Filtering out binding '{value}' (type: {binding_type_name}): Doesn't support 'image' output.")
                    elif filter_capability == 'ttt' and 'text' not in supported_outputs:
                        should_include = False
                        logger.debug(f"Filtering out binding '{value}' (type: {binding_type_name}): Doesn't support 'text' output.")
                else:
                    log_warning(f"Could not find type info for binding '{value}' (type: {binding_type_name}) to apply filter.")
            else:
                log_warning(f"Cannot filter binding '{value}': Missing type info or binding_types_data.")
        if should_include:
            valid_items_map[value] = label
    for value, label in valid_items_map.items():
        select_options.append(SelectOption(label=label[:100], value=value[:100], default=value == current_value))
    if len(select_options) <= (1 if allow_none else 0):
        log_warning(f"No valid options remain for {setting_type} after filtering for '{filter_capability}'.")
        if allow_none:
            view = ui.View(timeout=180)
            class FilteredEmptySettingsSelect(ui.Select):
                def __init__(self):
                    super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)
                async def callback(self, interaction: discord.Interaction):
                    await interaction.response.edit_message(content=f"No options match the filter '{filter_capability}'.", view=None)
            view.add_item(FilteredEmptySettingsSelect())
            return view
        else:
            return None
    if len(select_options) > 25:
        log_warning(f"Truncating options for {setting_type} to 25.")
        select_options = select_options[:25]

    class SettingsSelect(ui.Select):
        def __init__(self):
            super().__init__(placeholder=f'Select {setting_type}...', min_values=1, max_values=1, options=select_options)
        async def callback(self, interaction: discord.Interaction):
            selected_value = self.values[0]
            final_value = None if selected_value == "__NONE__" else selected_value
            display_label = none_display_text if final_value is None else valid_items_map.get(final_value, final_value)
            log_system(f"{setting_type} selected: {final_value}")
            global current_personality, current_ttt_binding, current_ttt_model, current_tti_binding, current_tti_model
            updated = False
            setting_changed = False
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
            if setting_changed and on_complete_callback:
                 await interaction.response.edit_message(content=f"✅ {setting_type} set to: `{display_label}`. Processing next step...", view=None)
                 await on_complete_callback(interaction, final_value)
            else:
                 await interaction.response.edit_message(content=f"✅ {setting_type} set to: `{display_label}`", view=None)

    view = ui.View(timeout=180)
    view.add_item(SettingsSelect())
    return view

async def run_setup_wizard(interaction: discord.Interaction):
    """Guides the owner through the setup."""
    log_system("Starting interactive setup wizard...")
    personalities_data = await make_api_call_async("/list_personalities")
    bindings_data = await make_api_call_async("/list_bindings")
    if not personalities_data or not bindings_data:
        await interaction.followup.send("❌ Error fetching initial data from LOLLMS server. Cannot start setup.", ephemeral=True)
        return
    await interaction.followup.send("Starting interactive setup...", ephemeral=True)
    await setup_step_personality(interaction, personalities_data)

async def setup_step_personality(interaction: discord.Interaction, personalities_data: Dict):
    """Setup Step 1: Select Personality."""
    current_settings = get_current_operational_settings()
    view = await create_select_menu(
        interaction, "Personality", personalities_data.get('personalities',{}),
        current_value=current_settings['personality'], allow_none=True, none_display_text="None (No Personality)",
        on_complete_callback=setup_step_ttt_binding
    )
    if view:
        await interaction.followup.send("Step 1: Select Personality:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("❌ Error creating personality menu or no personalities found.", ephemeral=True)

async def setup_step_ttt_binding(interaction: discord.Interaction, _selected_personality: Optional[str]):
    """Setup Step 2: Select TTT Binding."""
    current_settings = get_current_operational_settings()
    bindings_data = await make_api_call_async("/list_bindings")
    ASCIIColors.magenta(bindings_data)
    if not bindings_data:
        await interaction.followup.send("❌ Error fetching bindings.", ephemeral=True)
        return
    view = await create_select_menu(
        interaction, "TTT Binding", bindings_data.get('binding_instances',{}),
        binding_types_data=bindings_data.get('binding_types',{}), filter_capability='ttt',
        current_value=current_settings['ttt_binding'], allow_none=True, none_display_text="Server Default",
        on_complete_callback=setup_step_ttt_model
    )
    if view:
        await interaction.followup.send("Step 2: Select TTT Binding (Text-capable):", view=view, ephemeral=True)
    else:
        await interaction.followup.send("❌ Error creating TTT binding menu or no text bindings found.", ephemeral=True)

async def setup_step_ttt_model(interaction: discord.Interaction, selected_ttt_binding: Optional[str]):
    """Setup Step 3: Select TTT Model."""
    current_settings = get_current_operational_settings()
    if not selected_ttt_binding:
        await setup_step_tti_binding(interaction, None)
        return
    models_data = await make_api_call_async(f"/list_available_models/{selected_ttt_binding}")
    if not models_data:
        await interaction.followup.send(f"❌ Error fetching models for {selected_ttt_binding}.", ephemeral=True)
        await setup_step_tti_binding(interaction, None)
        return
    view = await create_select_menu(
        interaction, "TTT Model", models_data.get('models',[]),
        current_value=current_settings['ttt_model'], allow_none=True, none_display_text=f"Default for '{selected_ttt_binding}'",
        on_complete_callback=setup_step_tti_binding
    )
    if view:
        await interaction.followup.send(f"Step 3: Select TTT Model for `{selected_ttt_binding}`:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("❌ Error creating TTT model menu or no models found.", ephemeral=True)
        await setup_step_tti_binding(interaction, None)

async def setup_step_tti_binding(interaction: discord.Interaction, _selected_ttt_model: Optional[str]):
    """Setup Step 4: Select TTI Binding."""
    current_settings = get_current_operational_settings()
    bindings_data = await make_api_call_async("/list_bindings")
    if not bindings_data:
        await interaction.followup.send("❌ Error fetching bindings.", ephemeral=True)
        await finish_setup(interaction, None)
        return
    view = await create_select_menu(
        interaction, "TTI Binding", bindings_data.get('binding_instances',{}),
        binding_types_data=bindings_data.get('binding_types',{}), filter_capability='tti',
        current_value=current_settings['tti_binding'], allow_none=True, none_display_text="Server Default",
        on_complete_callback=setup_step_tti_model
    )
    if view:
        await interaction.followup.send("Step 4: Select TTI Binding (Image-capable):", view=view, ephemeral=True)
    else:
        await interaction.followup.send("❌ Error creating TTI binding menu or no image bindings found.", ephemeral=True)
        await finish_setup(interaction, None)

async def setup_step_tti_model(interaction: discord.Interaction, selected_tti_binding: Optional[str]):
    """Setup Step 5: Select TTI Model."""
    current_settings = get_current_operational_settings()
    if not selected_tti_binding:
        await finish_setup(interaction, None)
        return
    models_data = await make_api_call_async(f"/list_available_models/{selected_tti_binding}")
    if not models_data:
        await interaction.followup.send(f"❌ Error fetching models for {selected_tti_binding}.", ephemeral=True)
        await finish_setup(interaction, None)
        return
    view = await create_select_menu(
        interaction, "TTI Model", models_data.get('models',[]),
        current_value=current_settings['tti_model'], allow_none=True, none_display_text=f"Default for '{selected_tti_binding}'",
        on_complete_callback=finish_setup
    )
    if view:
        await interaction.followup.send(f"Step 5: Select TTI Model for `{selected_tti_binding}`:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("❌ Error creating TTI model menu or no models found.", ephemeral=True)
        await finish_setup(interaction, None)

async def finish_setup(interaction: discord.Interaction, _selected_tti_model: Optional[str]):
    """Final step of the setup wizard."""
    global initial_settings_setup_needed
    initial_settings_setup_needed = False
    save_operational_settings(get_current_operational_settings())
    await interaction.followup.send("✅ Interactive setup complete! Settings saved.", ephemeral=True)
    log_system("Interactive setup wizard completed.")

settings_group = app_commands.Group(name="lollms_settings", description="View or setup Lord of Discord settings.")

@settings_group.command(name="setup", description="Run the initial setup wizard (owner only).")
@app_commands.checks.has_permissions()
async def setup_wizard(interaction: discord.Interaction):
    """Starts the setup wizard."""
    await interaction.response.defer(ephemeral=True, thinking=True)
    await run_setup_wizard(interaction)

@settings_group.command(name="view", description="View current LOLLMS settings (owner only).")
@app_commands.checks.has_permissions()
async def view_settings(interaction: discord.Interaction):
    """Displays the current settings."""
    await interaction.response.defer(ephemeral=True)
    if initial_settings_setup_needed:
        await interaction.followup.send("Run `/lollms_settings setup` first.", ephemeral=True)
        return
    settings = get_current_operational_settings()
    embed = Embed(title="Current LOLLMS Settings", color=Color.blue())
    embed.add_field(name="Personality", value=f"`{settings['personality'] or 'None'}`", inline=False)
    embed.add_field(name="TTT Binding", value=f"`{settings['ttt_binding'] or 'Default'}`", inline=False)
    embed.add_field(name="TTT Model", value=f"`{settings['ttt_model'] or 'Default'}`", inline=False)
    embed.add_field(name="TTI Binding", value=f"`{settings['tti_binding'] or 'Default'}`", inline=False)
    embed.add_field(name="TTI Model", value=f"`{settings['tti_model'] or 'Default'}`", inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)

@settings_group.command(name="set_personality", description="Select active personality (owner only).")
@app_commands.checks.has_permissions()
async def set_personality(interaction: discord.Interaction):
    """Allows setting the personality."""
    if initial_settings_setup_needed:
        await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    personalities_data = await make_api_call_async("/list_personalities")
    if not personalities_data:
        await interaction.followup.send("❌ Error fetching personalities.", ephemeral=True)
        return
    view = await create_select_menu(
        interaction, "Personality", personalities_data.get('personalities',{}),
        current_value=current_personality, allow_none=True, none_display_text="None (No Personality)"
    )
    if view:
        await interaction.followup.send("Choose personality:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu or no personalities found.", ephemeral=True)

@settings_group.command(name="set_ttt_binding", description="Select TTT binding (owner only).")
@app_commands.checks.has_permissions()
async def set_ttt_binding(interaction: discord.Interaction):
    """Allows setting the TTT binding."""
    if initial_settings_setup_needed:
        await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    bindings_data = await make_api_call_async("/list_bindings")
    if not bindings_data:
        await interaction.followup.send("❌ Error fetching bindings.", ephemeral=True)
        return
    view = await create_select_menu(
        interaction, "TTT Binding", bindings_data.get('binding_instances',{}),
        binding_types_data=bindings_data.get('binding_types',{}), filter_capability='ttt',
        current_value=current_ttt_binding, allow_none=True, none_display_text="Server Default"
    )
    if view:
        await interaction.followup.send("Choose TTT binding (Text-capable):", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu or no text bindings found.", ephemeral=True)

@settings_group.command(name="set_ttt_model", description="Select TTT model (owner only).")
@app_commands.checks.has_permissions()
async def set_ttt_model(interaction: discord.Interaction):
    """Allows setting the TTT model."""
    if initial_settings_setup_needed:
        await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    if not current_ttt_binding:
        await interaction.followup.send("❌ Select TTT Binding first.", ephemeral=True)
        return
    models_data = await make_api_call_async(f"/list_available_models/{current_ttt_binding}")
    if not models_data:
        await interaction.followup.send(f"❌ Error fetching models for {current_ttt_binding}.", ephemeral=True)
        return
    view = await create_select_menu(
        interaction, "TTT Model", models_data.get('models',[]),
        current_value=current_ttt_model, allow_none=True, none_display_text=f"Default for '{current_ttt_binding}'"
    )
    if view:
        await interaction.followup.send(f"Choose TTT model for `{current_ttt_binding}`:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu or no models found.", ephemeral=True)

@settings_group.command(name="set_tti_binding", description="Select TTI binding (owner only).")
@app_commands.checks.has_permissions()
async def set_tti_binding(interaction: discord.Interaction):
    """Allows setting the TTI binding."""
    if initial_settings_setup_needed:
        await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    bindings_data = await make_api_call_async("/list_bindings")
    if not bindings_data:
        await interaction.followup.send("❌ Error fetching bindings.", ephemeral=True)
        return
    view = await create_select_menu(
        interaction, "TTI Binding", bindings_data.get('binding_instances',{}),
        binding_types_data=bindings_data.get('binding_types',{}), filter_capability='tti',
        current_value=current_tti_binding, allow_none=True, none_display_text="Server Default",
        on_complete_callback=lambda i, val: asyncio.create_task(reset_tti_model_on_binding_change(i, val))
    )
    if view:
        await interaction.followup.send("Choose TTI binding (Image-capable):", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu or no image bindings found.", ephemeral=True)

async def reset_tti_model_on_binding_change(interaction: discord.Interaction, new_binding_name: Optional[str]):
    """Callback to reset TTI model when TTI binding changes."""
    global current_tti_model
    if current_tti_binding != new_binding_name:
        log_info(f"TTI binding changed to '{new_binding_name}', resetting TTI model.")
        current_tti_model = None
        save_operational_settings(get_current_operational_settings())

@settings_group.command(name="set_tti_model", description="Select TTI model (owner only).")
@app_commands.checks.has_permissions()
async def set_tti_model(interaction: discord.Interaction):
    """Allows setting the TTI model."""
    if initial_settings_setup_needed:
        await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    if not current_tti_binding:
        await interaction.followup.send("❌ Select TTI Binding first using `/lollms_settings set_tti_binding`.", ephemeral=True)
        return
    api_endpoint = f"/list_available_models/{current_tti_binding}"
    models_data = await make_api_call_async(api_endpoint)
    if not models_data:
        await interaction.followup.send(f"❌ Error fetching models for {current_tti_binding}.", ephemeral=True)
        return
    view = await create_select_menu(
        interaction, "TTI Model", models_data.get('models',[]),
        current_value=current_tti_model, allow_none=True,
        none_display_text=f"Default for '{current_tti_binding}'"
    )
    if view:
        await interaction.followup.send(f"Choose TTI model for `{current_tti_binding}`:", view=view, ephemeral=True)
    else:
        await interaction.followup.send("Could not create menu (maybe the selected TTI binding has no models?).", ephemeral=True)

@bot.event
async def on_application_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Handles application command errors."""
    respond_method = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
    if isinstance(error, app_commands.CommandNotFound):
        pass
    elif isinstance(error, app_commands.CheckFailure) or isinstance(error, commands.NotOwner):
        await respond_method("❌ You don't have permission.", ephemeral=True)
    elif isinstance(error, app_commands.MissingRequiredArgument):
        await respond_method(f"❌ Missing argument: `{error.param.name}`.", ephemeral=True)
    elif isinstance(error, HTTPException) and error.code == 50035:
        await respond_method("❌ Error: Response too long.", ephemeral=True)
    else:
        cmd_name = interaction.command.qualified_name if interaction.command else 'unknown'
        log_error(f"Unhandled error in '{cmd_name}': {error}")
        trace_exception(error)
        try:
            await respond_method("❌ An unexpected error occurred.", ephemeral=True)
        except Exception as resp_err:
            log_error(f"Failed sending error response: {resp_err}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='{asctime} | {levelname:<8} | {name} | {message}', style='{')
    logger.info("--------------------------------------------------")
    logger.info("Initializing Lord of Discord...")
    logger.info("--------------------------------------------------")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not ensure_bot_config():
        ASCIIColors.error("Bot configuration could not be loaded or created. Exiting.")
        sys.exit(1)

    if not SETTINGS_FILE.exists():
        log_warning(f"'{SETTINGS_FILE}' not found. Operational settings need setup.")
        initial_settings_setup_needed = True
        save_operational_settings({})
    else:
        initial_settings_setup_needed = False

    update_current_operational_settings(load_operational_settings())

    log_system("Starting Discord Bot...")
    if not DISCORD_TOKEN:
        log_error("Discord Token missing after config load.")
        sys.exit(1)

    ASCIIColors.cyan("\n--- Setup and Deployment ---")
    ASCIIColors.print("1.  **Discord Bot Setup:**")
    ASCIIColors.print("    - Go to the Discord Developer Portal: https://discord.com/developers/applications")
    ASCIIColors.print("    - Create a 'New Application'. Give it a name (e.g., 'MyLOLLOMSBot').")
    ASCIIColors.print("    - Go to the 'Bot' tab on the left.")
    ASCIIColors.print("    - Click 'Add Bot' and confirm.")
    ASCIIColors.print("    - **IMPORTANT:** Enable ALL 'Privileged Gateway Intents' (Presence, Server Members, Message Content).")
    ASCIIColors.print("    - Click 'Reset Token' and copy the token immediately. **This is your `discord_token`**. Store it securely.")
    ASCIIColors.print("    - (Optional) Customize the bot's icon and description.")
    ASCIIColors.print("2.  **Configuration (`bot_config.json`):**")
    ASCIIColors.print(f"    - This file stores your bot token and LOLLMS server details. It should be in the same directory as this script (`{CONFIG_FILE.resolve()}`).")
    ASCIIColors.print("    - If the file is missing, the Configuration Wizard will run when you start the bot.")
    ASCIIColors.print("    - Ensure `lollms_base_url` points to your running `lollms_server` (e.g., 'http://127.0.0.1:9600').")
    ASCIIColors.print("    - Enter the `lollms_api_key` if your server requires one, otherwise leave it blank or `null`.")
    ASCIIColors.print("3.  **Invite Bot:**")
    ASCIIColors.print("    - Go back to the Developer Portal -> Your Application -> OAuth2 -> URL Generator.")
    ASCIIColors.print("    - Select the following scopes: `bot` and `applications.commands`.")
    ASCIIColors.print("    - Select the following Bot Permissions: `Send Messages`, `Read Messages/View Channels`, `Embed Links`, `Attach Files` (and others if needed).")
    ASCIIColors.print("    - Copy the generated URL and paste it into your browser.")
    ASCIIColors.print("    - Select the server you want to add the bot to and authorize it.")
    ASCIIColors.print("4.  **Run the Bot:**")
    ASCIIColors.print(f"    - Make sure your `lollms_server` is running and accessible at the configured URL ({BASE_URL}).")
    ASCIIColors.print(f"    - Execute this script: `python {Path(__file__).name}`")
    ASCIIColors.print("    - The bot should log in and sync commands.")
    ASCIIColors.print("5.  **Initial Setup (if needed):**")
    ASCIIColors.print("    - If this is the first run or `discord_settings.json` was missing, the bot owner needs to run `/lollms_settings setup` in Discord.")
    ASCIIColors.print("6.  **Deployment (Running Persistently):**")
    ASCIIColors.print("    - To keep the bot running after you close the terminal, use tools like:")
    ASCIIColors.print("      - `screen` or `tmux`: Run the bot inside a session.")
    ASCIIColors.print("      - `systemd` (Linux): Create a service file to manage the bot process.")
    ASCIIColors.print("      - Docker: Containerize the bot application.")
    ASCIIColors.print("      - Process managers like `pm2` (Node.js ecosystem, but can manage Python scripts).")
    ASCIIColors.cyan("--- End Setup Instructions ---\n")

    try:
        bot.run(DISCORD_TOKEN)
    except LoginFailure:
        log_error("Login Failed. Verify the 'discord_token' in bot_config.json is correct.")
    except Exception as e:
        log_error(f"Bot runtime error: {e}")
        trace_exception(e)

    log_system("Lord of Discord stopped.")