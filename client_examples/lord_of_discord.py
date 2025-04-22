# client_examples/lord_of_discord.py
import pipmaster as pm
pm.install_if_missing("discord") # Corrected package name
pm.install_if_missing("requests")
pm.install_if_missing("sseclient-py")
pm.install_if_missing("Pillow")
pm.install_if_missing("ascii_colors") # Added missing dependency

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
from typing import List, Dict, Optional, Any, Union, Callable, Coroutine
from ascii_colors import ASCIIColors # For colored console output

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
    url = f"{BASE_URL}/generate"; headers = HEADERS_NO_STREAM
    payload['stream'] = False # Ensure stream is false for bot
    loop = asyncio.get_running_loop()
    try:
        with requests.Session() as session:
            response = await loop.run_in_executor(None, lambda: session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT))
        response.raise_for_status()
        return response.json() # Expect JSON for non-stream generate
    except requests.exceptions.Timeout: log_error(f"Request timed out to {url}")
    except requests.exceptions.ConnectionError as e: log_error(f"Could not connect to server at {url}. Details: {e}")
    except requests.exceptions.RequestException as e:
        log_error(f"Generate request failed to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None: log_error(f" Server Response: {e.response.status_code} - {e.response.text[:200]}")
    except Exception as e: log_error(f"Unexpected error during generate request: {e}")
    return None

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
    """Handles the main chat command."""
    try: await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e: log_error(f"Defer failed for /lollms: {e}"); return

    # Use new input_data format
    payload = {
        "input_data": [{"type": "text", "role": "user_prompt", "data": prompt}],
        "generation_type": "ttt", "personality": current_personality,
        "binding_name": current_ttt_binding, "model_name": current_ttt_model,
        "stream": False # Bot uses non-streaming
    }
    result = await make_generate_request_async(payload)
    response_text = ""; status_emoji = "💡"

    # Process the JSON response which should contain {"output": {"text": "..."}}
    if isinstance(result, dict) and "output" in result and "text" in result["output"]: response_text = result["output"]["text"]
    elif isinstance(result, dict): response_text = f"Unexpected JSON:\n```json\n{json.dumps(result, indent=2)[:1500]}\n```"; status_emoji = "⚠️"
    elif result is None: response_text = "Failed to get response from LOLLMS server."; status_emoji = "❌"
    else: response_text = f"Unexpected response type: {type(result)}"; log_error(f"Unexpected type: {type(result)}"); status_emoji = "❌"
    if not response_text.strip(): response_text = "(Received empty response)"; status_emoji = "🤷"

    base_reply = f"👤 **{interaction.user.display_name}:** {prompt}\n\n{status_emoji} **({current_personality or 'Default'}) AI:** "
    chunks = split_message(response_text)
    try:
        first_chunk_content = base_reply + (chunks[0] if chunks else "(No content)")
        remaining_chunks = chunks[1:]
        if len(first_chunk_content) > 2000:
            await interaction.followup.send(base_reply); split_first = split_message(chunks[0])
            for sub_chunk in split_first: await interaction.channel.send(sub_chunk) # type: ignore
        else: await interaction.followup.send(first_chunk_content)
        for chunk in remaining_chunks: await interaction.channel.send(chunk) # type: ignore
    except HTTPException as e: 
        log_error(f"Failed sending response: {e}")
        try: 
            await interaction.followup.send("❌ Error sending response.")
        except Exception: pass


@bot.tree.command(name="lollms_imagine", description="Generate an image with LOLLMS.")
@app_commands.describe(prompt="The description of the image to generate.")
async def lollms_imagine(interaction: discord.Interaction, prompt: str):
    """Handles the image generation command."""
    try: await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e: log_error(f"Defer failed for /lollms_imagine: {e}"); return

    # Use new input_data format
    payload = {
        "input_data": [{"type": "text", "role": "user_prompt", "data": prompt}],
        "generation_type": "tti",
        "binding_name": current_tti_binding, "model_name": current_tti_model,
        "stream": False
    }
    response_data = await make_generate_request_async(payload)
    tmp_file_path: Optional[Path] = None
    status_emoji = "🖼️"; reply_content = f"👤 **{interaction.user.display_name}:** /imagine {prompt}\n\n{status_emoji} **Image Result:**"

    try:
        # Process the JSON response which should contain {"output": {"image_base64": "...", ...}}
        if response_data and isinstance(response_data, dict) and "output" in response_data:
            output_content = response_data["output"]
            image_b64 = output_content.get("image_base64")
            if image_b64:
                try:
                    img_data = base64.b64decode(image_b64); img_buffer = BytesIO(img_data)
                    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=IMAGE_DIR) as tmp_f:
                        img = Image.open(img_buffer); img.save(tmp_f, format="PNG"); tmp_file_path = Path(tmp_f.name)
                    if tmp_file_path and tmp_file_path.exists():
                        safe_filename = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()[:50].replace(' ','_') + ".png"
                        discord_file = File(tmp_file_path, filename=safe_filename)
                        await interaction.followup.send(reply_content, file=discord_file)
                    else: status_emoji = "❌"; reply_content += "\nError saving image."; await interaction.followup.send(reply_content)
                except (base64.binascii.Error, UnidentifiedImageError) as img_err: log_error(f"Invalid image data: {img_err}"); status_emoji = "❌"; reply_content += f"\nError: Invalid image data ({type(img_err).__name__})."; await interaction.followup.send(reply_content)
                except Exception as e: log_error(f"Error processing image: {e}"); status_emoji = "❌"; reply_content += f"\nError processing image: {e}"; await interaction.followup.send(reply_content)
            else: error_msg = output_content.get("error", "No 'image_base64' in output."); status_emoji = "❌"; reply_content += f"\nError: Server failed: {error_msg}"; await interaction.followup.send(reply_content)
        elif isinstance(response_data, dict): status_emoji = "❌"; reply_content += f"\nError: Unexpected JSON structure:\n```\n{json.dumps(response_data, indent=2)[:1000]}\n```"; await interaction.followup.send(reply_content)
        else: status_emoji = "❌"; reply_content += "\nError: Failed to get valid image response."; await interaction.followup.send(reply_content)
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try: tmp_file_path.unlink()
            except OSError as e: log_error(f"Error deleting temp image {tmp_file_path}: {e}")

# --- Settings Select Menu Helper ---
async def create_select_menu( interaction_or_ctx: Union[discord.Interaction], setting_type: str, fetch_endpoint: str, current_value: Optional[str], list_json_key: Optional[str], display_key: str = 'name', value_key: str = 'name', is_dict_list: bool = True, allow_none: bool = True, none_display_text: str = "Use Server Default", on_complete_callback: Optional[Callable[[discord.Interaction, Optional[str]], Coroutine[Any, Any, None]]] = None ) -> Optional[ui.View]:
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

# --- Initial Setup Wizard ---
async def run_initial_setup_wizard(interaction: discord.Interaction):
    """Guides the owner through the initial setup."""
    log_system("Starting initial setup wizard...")
    await interaction.response.send_message("Starting initial setup wizard...", ephemeral=True)
    await setup_step_personality(interaction)

async def setup_step_personality(interaction: discord.Interaction):
    """Setup Step 1: Select Personality."""
    view = await create_select_menu(interaction, "Personality", "/list_personalities", None, 'personalities', is_dict_list=True, allow_none=True, none_display_text="None (No Personality)", on_complete_callback=setup_step_ttt_binding)
    if view: await interaction.followup.send("Step 1: Select Personality:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error creating personality menu.", ephemeral=True)

async def setup_step_ttt_binding(interaction: discord.Interaction, _selected_personality: Optional[str]):
    """Setup Step 2: Select TTT Binding."""
    view = await create_select_menu(interaction, "TTT Binding", "/list_bindings", None, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default", on_complete_callback=setup_step_ttt_model)
    if view: await interaction.followup.send("Step 2: Select TTT Binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error creating TTT binding menu.", ephemeral=True)

async def setup_step_ttt_model(interaction: discord.Interaction, selected_ttt_binding: Optional[str]):
    """Setup Step 3: Select TTT Model."""
    if not selected_ttt_binding: await setup_step_tti_binding(interaction, None); return
    view = await create_select_menu(interaction, "TTT Model", f"/list_available_models/{selected_ttt_binding}", None, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{selected_ttt_binding}'", on_complete_callback=setup_step_tti_binding)
    if view: await interaction.followup.send(f"Step 3: Select TTT Model for `{selected_ttt_binding}`:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error TTT model menu.", ephemeral=True); await setup_step_tti_binding(interaction, None)

async def setup_step_tti_binding(interaction: discord.Interaction, _selected_ttt_model: Optional[str]):
    """Setup Step 4: Select TTI Binding."""
    view = await create_select_menu(interaction, "TTI Binding", "/list_bindings", None, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default", on_complete_callback=setup_step_tti_model)
    if view: await interaction.followup.send("Step 4: Select TTI Binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("❌ Error creating TTI binding menu.", ephemeral=True); await finish_setup(interaction, None) # Proceed even on error

async def setup_step_tti_model(interaction: discord.Interaction, selected_tti_binding: Optional[str]):
    """Setup Step 5: Select TTI Model."""
    if not selected_tti_binding: await finish_setup(interaction, None); return
    view = await create_select_menu(interaction, "TTI Model", f"/list_available_models/{selected_tti_binding}", None, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{selected_tti_binding}'", on_complete_callback=finish_setup)
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
    view = await create_select_menu(interaction, "Personality", "/list_personalities", current_personality, 'personalities', is_dict_list=True, allow_none=True, none_display_text="None (No Personality)")
    if view: await interaction.followup.send("Choose personality:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_ttt_binding", description="Select TTT binding (owner only).")
@app_commands.checks.has_permissions()
async def set_ttt_binding(interaction: discord.Interaction):
    """Allows setting the TTT binding."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    view = await create_select_menu(interaction, "TTT Binding", "/list_bindings", current_ttt_binding, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default")
    if view: await interaction.followup.send("Choose TTT binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_ttt_model", description="Select TTT model (owner only).")
@app_commands.checks.has_permissions()
async def set_ttt_model(interaction: discord.Interaction):
    """Allows setting the TTT model."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    if not current_ttt_binding: await interaction.followup.send("❌ Select TTT Binding first.", ephemeral=True); return
    view = await create_select_menu(interaction, "TTT Model", f"/list_available_models/{current_ttt_binding}", current_ttt_model, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{current_ttt_binding}'")
    if view: await interaction.followup.send(f"Choose TTT model for `{current_ttt_binding}`:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_tti_binding", description="Select TTI binding (owner only).")
@app_commands.checks.has_permissions()
async def set_tti_binding(interaction: discord.Interaction):
    """Allows setting the TTI binding."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    view = await create_select_menu(interaction, "TTI Binding", "/list_bindings", current_tti_binding, 'binding_instances', is_dict_list=False, allow_none=True, none_display_text="Server Default")
    if view: await interaction.followup.send("Choose TTI binding:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

@settings_group.command(name="set_tti_model", description="Select TTI model (owner only).")
@app_commands.checks.has_permissions()
async def set_tti_model(interaction: discord.Interaction):
    """Allows setting the TTI model."""
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings setup` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    if not current_tti_binding: await interaction.followup.send("❌ Select TTI Binding first.", ephemeral=True); return
    view = await create_select_menu(interaction, "TTI Model", f"/list_available_models/{current_tti_binding}", current_tti_model, 'models', is_dict_list=True, allow_none=True, none_display_text=f"Default for '{current_tti_binding}'")
    if view: await interaction.followup.send(f"Choose TTI model for `{current_tti_binding}`:", view=view, ephemeral=True)
    else: await interaction.followup.send("Could not create menu.", ephemeral=True)

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