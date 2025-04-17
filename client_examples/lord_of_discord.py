# --- Imports ---
import discord
from discord.ext import commands
# Import app_commands for tree-based slash commands and groups
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

# --- Dependency Check ---
# (Keep the pipmaster block as is)
try:
    import pipmaster as pm
    pm.install_if_missing("requests")
    pm.install_if_missing("sseclient-py")
    pm.install_if_missing("Pillow")
    pm.install_if_missing("discord.py") # Or py-cord
    print("Core dependencies checked/installed.")
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    # from sseclient import SSEClient # Not used directly
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependency - {e}")
    print("Please install required packages:")
    print("  pip install requests sseclient-py Pillow discord.py")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during dependency check: {e}")
    sys.exit(1)


# --- Configuration Loading ---
# (Keep CONFIG_FILE, SETTINGS_FILE, IMAGE_DIR definitions)
CONFIG_FILE = Path("bot_config.json")
SETTINGS_FILE = Path("discord_settings.json")
IMAGE_DIR = Path("generated_images")
# (Keep ensure_bot_config function as is - it sets globals DISCORD_TOKEN, BASE_URL, API_KEY)
def ensure_bot_config() -> bool: ... # (Implementation from previous step)

# --- Global Variables & Defaults ---
# (Keep DISCORD_TOKEN, BASE_URL, API_KEY definitions - populated by ensure_bot_config)
DISCORD_TOKEN: Optional[str] = None
BASE_URL: str = "http://localhost:9600/api/v1"
API_KEY: Optional[str] = None
DEFAULT_TIMEOUT = 120
DEFAULT_TTT_BINDING = None; DEFAULT_TTT_MODEL = None
DEFAULT_TTI_BINDING = None; DEFAULT_TTI_MODEL = None
HEADERS_NO_STREAM: Dict[str, str] = {}
# (Keep Bot Operational State globals: current_*, initial_settings_setup_needed)
current_personality: Optional[str] = None; current_ttt_binding: Optional[str] = None
current_ttt_model: Optional[str] = None; current_tti_binding: Optional[str] = DEFAULT_TTI_BINDING
current_tti_model: Optional[str] = DEFAULT_TTI_MODEL; initial_settings_setup_needed: bool = False

# --- Helper Functions ---
# (Keep log_*, load/save_json_file, load/save_operational_settings, get/update_current_operational_settings)
def log_system(message): ... # (Implementation from previous step)
def log_info(message): ...   # (Implementation from previous step)
def log_warning(message): ... # (Implementation from previous step)
def log_error(message): ...   # (Implementation from previous step)
def load_json_file(filepath: Path, default: Any = None) -> Any: ... # (Implementation from previous step)
def save_json_file(filepath: Path, data: Any) -> bool: ... # (Implementation from previous step)
def load_operational_settings() -> Dict[str, Optional[str]]: ... # (Implementation from previous step)
def save_operational_settings(settings: Dict[str, Optional[str]]): ... # (Implementation from previous step)
def get_current_operational_settings() -> Dict[str, Optional[str]]: ... # (Implementation from previous step)
def update_current_operational_settings(settings: Dict[str, Optional[str]]): ... # (Implementation from previous step)

# --- API Client Functions ---
# (Keep make_api_call_async, make_generate_request_async)
async def make_api_call_async(endpoint: str, method: str = "GET", payload: Optional[Dict] = None) -> Optional[Any]: ... # (Implementation from previous step)
async def make_generate_request_async(payload: Dict[str, Any]) -> Optional[Union[str, Dict]]: ... # (Implementation from previous step)

# --- Discord Bot Setup ---
# CORRECTED: Use commands.Bot
intents = Intents.default()
intents.guilds = True
intents.members = True # Required for user lookups in some contexts
# intents.message_content = True # Enable if needed for non-slash features
bot = commands.Bot(command_prefix="!", intents=intents) # Prefix needed for commands.Bot

# --- Helper for Splitting Messages ---
# (Keep split_message)
def split_message(content: str, max_len: int = 1980) -> List[str]: ... # (Implementation from previous step)

# --- Bot Events ---
@bot.event
async def on_ready():
    global initial_settings_setup_needed
    log_system(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    if not SETTINGS_FILE.exists():
        log_warning(f"'{SETTINGS_FILE}' not found. Initial operational setup required via /lollms_settings.")
        initial_settings_setup_needed = True
        save_operational_settings({})
    else: initial_settings_setup_needed = False
    update_current_operational_settings(load_operational_settings())
    log_system('Syncing slash commands...')
    try:
        await bot.change_presence(activity=Game(name="/lollms help"))
        # Register the command group with the tree HERE
        bot.tree.add_command(settings_group) # Add the group instance
        # Sync the tree
        synced = await bot.tree.sync()
        log_system(f"Synced {len(synced)} slash commands globally.")
    except Exception as e: log_error(f"Failed to sync commands: {e}")
    log_system('Lord of Discord is ready.')
    if initial_settings_setup_needed: log_system("Waiting for owner to run '/lollms_settings' for initial setup.")
    else: log_system(f"Current Settings: P:{current_personality or 'Default'} B:{current_ttt_binding or 'Default'} M:{current_ttt_model or 'Default'}")

# --- Slash Commands (/lollms, /lollms_imagine) ---

# CORRECTED: Use @bot.tree.command decorator
@bot.tree.command(name="lollms", description="Chat with the LOLLMS agent.")
# CORRECTED: Use @app_commands.describe for options
@app_commands.describe(prompt="Your message or question for the agent.")
async def lollms_chat(interaction: discord.Interaction, prompt: str):
    # Use interaction instead of ctx for app_commands
    try: await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e: log_error(f"Defer failed for /lollms: {e}"); return

    payload = {"prompt": prompt, "generation_type": "ttt", "personality": current_personality,
               "binding_name": current_ttt_binding, "model_name": current_ttt_model}
    result = await make_generate_request_async(payload)
    response_text = ""; status_emoji = "💡"

    # --- (Response handling logic is the same) ---
    if isinstance(result, str): response_text = result
    elif isinstance(result, dict): response_text = f"Unexpected JSON:\n```json\n{json.dumps(result, indent=2)[:1500]}\n```"; status_emoji = "⚠️"
    elif result is None: response_text = "Failed to get response from LOLLMS server."; status_emoji = "❌"
    else: response_text = "Unexpected internal response type."; log_error(f"Unexpected type: {type(result)}"); status_emoji = "❌"
    if not response_text.strip(): response_text = "(Received empty response)"; status_emoji = "🤷"

    base_reply = f"👤 **{interaction.user.display_name}:** {prompt}\n\n{status_emoji} **({current_personality or 'Default'}) AI:** "
    chunks = split_message(response_text)
    try:
        first_chunk_content = base_reply + (chunks[0] if chunks else "(No content)")
        remaining_chunks = chunks[1:]
        # Use interaction.followup for deferred responses
        if len(first_chunk_content) > 2000:
            await interaction.followup.send(base_reply) # Send initial part
            split_first = split_message(chunks[0])
            # Send overflows to the channel
            for sub_chunk in split_first: await interaction.channel.send(sub_chunk) # type: ignore <- interaction.channel might be None in DMs
        else: await interaction.followup.send(first_chunk_content)
        # Send remaining chunks to the channel
        for chunk in remaining_chunks: await interaction.channel.send(chunk) # type: ignore
    except HTTPException as e:
        log_error(f"Failed sending response: {e}")
        try: await interaction.followup.send("❌ Error sending full response.")
        except Exception: pass


# CORRECTED: Use @bot.tree.command decorator
@bot.tree.command(name="lollms_imagine", description="Generate an image with LOLLMS.")
# CORRECTED: Use @app_commands.describe for options
@app_commands.describe(prompt="The description of the image to generate.")
async def lollms_imagine(interaction: discord.Interaction, prompt: str):
    # Use interaction instead of ctx
    try: await interaction.response.defer(ephemeral=False, thinking=True)
    except HTTPException as e: log_error(f"Defer failed for /lollms_imagine: {e}"); return

    payload = {"prompt": prompt, "generation_type": "tti",
               "binding_name": current_tti_binding, "model_name": current_tti_model}
    response_data = await make_generate_request_async(payload)
    tmp_file_path: Optional[Path] = None
    status_emoji = "🖼️"; reply_content = f"👤 **{interaction.user.display_name}:** /imagine {prompt}\n\n{status_emoji} **Image Result:**"

    # --- (Image processing and sending logic is the same, use interaction.followup) ---
    try:
        if response_data and isinstance(response_data, dict):
            image_b64 = response_data.get("image_base64")
            if image_b64:
                try:
                    img_data = base64.b64decode(image_b64); img_buffer = BytesIO(img_data)
                    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=IMAGE_DIR) as tmp_f:
                        img = Image.open(img_buffer); img.save(tmp_f, format="PNG")
                        tmp_file_path = Path(tmp_f.name)
                    if tmp_file_path and tmp_file_path.exists():
                        safe_filename = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()[:50].replace(' ','_') + ".png"
                        discord_file = File(tmp_file_path, filename=safe_filename)
                        await interaction.followup.send(reply_content, file=discord_file) # Use interaction
                    else: status_emoji = "❌"; reply_content += "\nError saving image."; await interaction.followup.send(reply_content) # Use interaction
                except (base64.binascii.Error, UnidentifiedImageError) as img_err:
                    log_error(f"Invalid image data: {img_err}"); status_emoji = "❌"
                    reply_content += f"\nError: Invalid image data ({type(img_err).__name__})."; await interaction.followup.send(reply_content) # Use interaction
                except Exception as e:
                    log_error(f"Error processing image: {e}"); status_emoji = "❌"
                    reply_content += f"\nError processing image: {e}"; await interaction.followup.send(reply_content) # Use interaction
            else:
                error_msg = response_data.get("error", "No 'image_base64'.")
                status_emoji = "❌"; reply_content += f"\nError: Server failed: {error_msg}"; await interaction.followup.send(reply_content) # Use interaction
        elif isinstance(response_data, str):
             status_emoji = "❌"; reply_content += f"\nError: Server text response:\n```\n{response_data[:1000]}\n```"; await interaction.followup.send(reply_content) # Use interaction
        else: status_emoji = "❌"; reply_content += "\nError: Failed to get image response."; await interaction.followup.send(reply_content) # Use interaction
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try: tmp_file_path.unlink()
            except OSError as e: log_error(f"Error deleting temp image {tmp_file_path}: {e}")


# --- Settings Select Menu Helper ---
# (Keep create_select_menu function as is, it works with Interactions)
async def create_select_menu(
    interaction_or_ctx: Union[discord.Interaction, discord.ApplicationContext],
    setting_type: str, fetch_endpoint: str, current_value: Optional[str],
    list_json_key: Optional[str], display_key: str = 'name', value_key: str = 'name',
    is_dict_list: bool = True, allow_none: bool = True, none_display_text: str = "Use Server Default",
    on_complete_callback: Optional[Callable[[discord.Interaction, Optional[str]], Coroutine[Any, Any, None]]] = None
) -> Optional[ui.View]: ... # (Implementation from previous step)

# --- Initial Operational Setup Wizard Functions ---
# (Keep run_initial_setup_wizard, setup_step_*, finish_setup functions as they use create_select_menu)
async def run_initial_setup_wizard(ctx: discord.ApplicationContext): ... # (Implementation from previous step)
async def setup_step_personality(ctx_or_interaction: Union[discord.ApplicationContext, discord.Interaction]): ... # (Implementation from previous step)
async def setup_step_ttt_binding(interaction: discord.Interaction, _selected_personality: Optional[str]): ... # (Implementation from previous step)
async def setup_step_ttt_model(interaction: discord.Interaction, selected_ttt_binding: Optional[str]): ... # (Implementation from previous step)
async def setup_step_tti_binding(interaction: discord.Interaction, _selected_ttt_model: Optional[str]): ... # (Implementation from previous step)
async def setup_step_tti_model(interaction: discord.Interaction, selected_tti_binding: Optional[str]): ... # (Implementation from previous step)
async def finish_setup(interaction: discord.Interaction, _selected_tti_model: Optional[str]): ... # (Implementation from previous step)


# --- Settings Command Group & Subcommands ---
# CORRECTED: Define group using app_commands.Group
settings_group = app_commands.Group(name="lollms_settings", description="View or setup Lord of Discord operational settings.")

# Subcommands now use the group decorator
@settings_group.command(name="view", description="View the current LOLLMS settings.")
@app_commands.checks.is_owner() # Use app_commands check
async def view_settings(interaction: discord.Interaction):
    # Use interaction
    await interaction.response.defer(ephemeral=True)
    if initial_settings_setup_needed:
         await interaction.followup.send("Please complete the initial setup first by running the base `/lollms_settings` command.", ephemeral=True)
         return
    settings = get_current_operational_settings(); embed = Embed(title="Current LOLLMS Settings", color=Color.blue())
    embed.add_field(name="Personality", value=f"`{settings['personality'] or 'None'}`", inline=False)
    embed.add_field(name="TTT Binding", value=f"`{settings['ttt_binding'] or 'Default'}`", inline=False)
    embed.add_field(name="TTT Model", value=f"`{settings['ttt_model'] or 'Default'}`", inline=False)
    embed.add_field(name="TTI Binding", value=f"`{settings['tti_binding'] or 'Default'}`", inline=False)
    embed.add_field(name="TTI Model", value=f"`{settings['tti_model'] or 'Default'}`", inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)


@settings_group.command(name="set_personality", description="Select the active personality.")
@app_commands.checks.is_owner()
async def set_personality(interaction: discord.Interaction):
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True) # Defer response within the interaction
    view = await create_select_menu(interaction, "Personality", "/list_personalities", current_personality, list_json_key='personalities', is_dict_list=True, allow_none=True, none_display_text="None (No Personality)")
    # Send followup after deferral
    if view: await interaction.followup.send("Choose a personality:", view=view, ephemeral=True)
    elif not view: await interaction.followup.send("Could not create personality menu.", ephemeral=True) # Handle case where menu creation failed


@settings_group.command(name="set_ttt_binding", description="Select the binding for Text-to-Text.")
@app_commands.checks.is_owner()
async def set_ttt_binding(interaction: discord.Interaction):
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    view = await create_select_menu(interaction, "TTT Binding", "/list_bindings", current_ttt_binding, list_json_key='binding_instances', is_dict_list=False, allow_none=True, none_display_text="Use Server Default Binding")
    if view: await interaction.followup.send("Choose a TTT binding:", view=view, ephemeral=True)
    elif not view: await interaction.followup.send("Could not create TTT binding menu.", ephemeral=True)


@settings_group.command(name="set_ttt_model", description="Select the model for TTT (requires specific binding).")
@app_commands.checks.is_owner()
async def set_ttt_model(interaction: discord.Interaction):
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    if not current_ttt_binding: await interaction.followup.send("❌ Select a TTT Binding first.", ephemeral=True); return
    view = await create_select_menu(interaction, "TTT Model", f"/list_available_models/{current_ttt_binding}", current_ttt_model, list_json_key='models', is_dict_list=True, allow_none=True, none_display_text=f"Use Default for '{current_ttt_binding}'")
    if view: await interaction.followup.send(f"Choose TTT model for `{current_ttt_binding}`:", view=view, ephemeral=True)
    elif not view: await interaction.followup.send("Could not create TTT model menu.", ephemeral=True)


@settings_group.command(name="set_tti_binding", description="Select the binding for Text-to-Image.")
@app_commands.checks.is_owner()
async def set_tti_binding(interaction: discord.Interaction):
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    view = await create_select_menu(interaction, "TTI Binding", "/list_bindings", current_tti_binding, list_json_key='binding_instances', is_dict_list=False, allow_none=True, none_display_text="Use Server Default Binding")
    if view: await interaction.followup.send("Choose a TTI binding:", view=view, ephemeral=True)
    elif not view: await interaction.followup.send("Could not create TTI binding menu.", ephemeral=True)


@settings_group.command(name="set_tti_model", description="Select the model for TTI (requires specific binding).")
@app_commands.checks.is_owner()
async def set_tti_model(interaction: discord.Interaction):
    if initial_settings_setup_needed: await interaction.response.send_message("Run `/lollms_settings` first.", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    if not current_tti_binding: await interaction.followup.send("❌ Select a TTI Binding first.", ephemeral=True); return
    view = await create_select_menu(interaction, "TTI Model", f"/list_available_models/{current_tti_binding}", current_tti_model, list_json_key='models', is_dict_list=True, allow_none=True, none_display_text=f"Use Default for '{current_tti_binding}'")
    if view: await interaction.followup.send(f"Choose TTI model for `{current_tti_binding}`:", view=view, ephemeral=True)
    elif not view: await interaction.followup.send("Could not create TTI model menu.", ephemeral=True)

# --- Error Handling ---
# Keep on_application_command_error (uses bot.event, handles interaction errors)
@bot.event
async def on_application_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    # Use interaction context for responding
    respond_method = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
    if isinstance(error, app_commands.CommandNotFound): pass # Should be handled by Discord
    elif isinstance(error, app_commands.CheckFailure) or isinstance(error, commands.NotOwner): # Catch permission errors
        await respond_method("❌ You don't have permission to use this command.", ephemeral=True)
    elif isinstance(error, app_commands.MissingRequiredArgument): # Correct error type
        await respond_method(f"❌ Missing argument: `{error.param.name}`.", ephemeral=True) # Access param differently
    elif isinstance(error, HTTPException) and error.code == 50035: # Invalid Form Body (message too long)
        await respond_method("❌ Error: The response was too long to send.", ephemeral=True)
    else:
        command_name = interaction.command.qualified_name if interaction.command else 'unknown command'
        log_error(f"Unhandled error in '{command_name}': {error}")
        try: await respond_method("❌ An unexpected error occurred.", ephemeral=True)
        except Exception as resp_err: log_error(f"Failed to send error response: {resp_err}")


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