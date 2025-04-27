# Lord of Discord Example (`lord_of_discord.py`)

## Purpose

This script runs a Discord bot that acts as a frontend for the `lollms_server`. Users can interact with the bot using slash commands to:

*   Chat with the AI (`/lollms`).
*   Generate images (`/lollms_imagine`).
*   Configure the bot's default personality, TTT binding/model, and TTI binding/model via owner-only settings commands (`/lollms_settings ...`).

## Prerequisites

*   Python 3.7+
*   Libraries: `discord.py`, `requests`, `sseclient-py`, `Pillow`, `ascii_colors`, `pipmaster`. Run `pip install -r client_examples/lord_of_discord_requirements.txt` (You'll need to create this file or install manually).
*   A running `lollms_server` instance with working TTT and TTI bindings.
*   A valid `lollms_server` API key.
*   A **Discord Bot Token**. You need to create a bot application on the Discord Developer Portal ([https://discord.com/developers/applications](https://discord.com/developers/applications)).
    *   Enable the **Server Members Intent** and **Message Content Intent** for your bot in the portal.
    *   Invite the bot to your Discord server with `applications.commands` and necessary permissions (Send Messages, Read Message History, Attach Files).

## Configuration

1.  **`client_examples/bot_config.json`**: Create this file with the following structure:
    ```json
    {
      "discord_token": "YOUR_DISCORD_BOT_TOKEN_HERE",
      "lollms_base_url": "http://localhost:9600", // Base URL of lollms_server (NO /api/v1)
      "lollms_api_key": "YOUR_LOLLMS_SERVER_API_KEY_HERE"
    }
    ```
    Replace the placeholder values with your actual Discord bot token and `lollms_server` URL/API key.

2.  **`client_examples/discord_settings.json`**: This file stores the bot's operational settings (personality, bindings, models). It will be created automatically on the first run if it doesn't exist. You can configure it using the `/lollms_settings` commands in Discord.

## Running the Example

1.  Make sure your `lollms_server` is running with appropriate TTT and TTI bindings configured.
2.  Ensure you have created `bot_config.json` with your Discord token and server details.
3.  Ensure your Python environment has the required libraries installed (`pip install -r client_examples/lord_of_discord_requirements.txt` or install manually: `discord.py requests sseclient-py Pillow ascii_colors pipmaster`).
4.  Navigate to the `lollms_server` project root directory in your terminal.
5.  Run the script:
    ```bash
    python client_examples/lord_of_discord.py
    ```

## Usage (in Discord)

*   **Initial Setup (Owner Only):** If `discord_settings.json` doesn't exist, the bot owner needs to run `/lollms_settings setup` in Discord. This will guide through selecting the initial personality, bindings, and models via interactive menus.
*   **Chat:** `/lollms prompt:<your message>` - Sends the prompt to the server using the configured TTT settings.
*   **Image Generation:** `/lollms_imagine prompt:<your image description>` - Sends the prompt to the server using the configured TTI settings. The generated image is posted back to the channel.
*   **Settings (Owner Only):**
    *   `/lollms_settings view`: Shows the current configuration.
    *   `/lollms_settings set_personality`: Choose a new default personality.
    *   `/lollms_settings set_ttt_binding`: Choose a new TTT binding.
    *   `/lollms_settings set_ttt_model`: Choose a model for the selected TTT binding.
    *   `/lollms_settings set_tti_binding`: Choose a new TTI binding.
    *   `/lollms_settings set_tti_model`: Choose a model for the selected TTI binding.

## Expected Behavior

*   The bot logs into Discord and syncs slash commands.
*   It responds to `/lollms` and `/lollms_imagine` commands by querying the `lollms_server` API.
*   TTT responses are posted back as text (long responses might be split).
*   TTI responses are posted back as embedded images.
*   Settings commands allow the bot owner (or users with appropriate permissions) to change the backend configuration dynamically.
*   Settings are saved to `discord_settings.json`.

## Notes

*   The bot uses non-streaming requests for simplicity in Discord integration.
*   Error handling is included, but complex API or Discord issues might require debugging.
*   Ensure the bot has the necessary Discord permissions in the server it's added to.
*   Requires `pipmaster` for initial dependency checks.