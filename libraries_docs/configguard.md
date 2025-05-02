# ConfigGuard

[![PyPI version](https://img.shields.io/pypi/v/configguard.svg)](https://pypi.org/project/configguard/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/configguard.svg)](https://pypi.org/project/configguard/)
[![PyPI license](https://img.shields.io/pypi/l/configguard.svg)](https://github.com/ParisNeo/ConfigGuard/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/configguard)](https://pepy.tech/project/configguard)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://parisneo.github.io/ConfigGuard/)
<!-- [![Build Status](https://github.com/ParisNeo/ConfigGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/ParisNeo/ConfigGuard/actions/workflows/ci.yml) Placeholder -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

**Stop fighting inconsistent, error-prone, and insecure configuration files!** 🚀

**ConfigGuard** transforms your Python application's configuration management from a potential source of bugs and security risks into a robust, reliable, and developer-friendly system. Moving beyond simple dictionaries or basic file parsing, ConfigGuard introduces a **schema-driven fortress** for your settings, offering unparalleled control and safety.

Leverage a comprehensive suite of features designed for modern applications:

*   Define strict **Type Safety** and complex **Validation Rules** (`min`, `max`, `options`, `nullable`).
*   Protect sensitive data effortlessly with built-in, handler-transparent **Encryption**.
*   Manage configuration changes across application updates with seamless **Versioning** and automated **Migration**.
*   Choose your preferred **Storage Format** (JSON, YAML, TOML, SQLite included) without altering your core logic.
*   Organize complex configurations intuitively using **Nested Sections**.
*   Accommodate unpredictable structures with flexible **Dynamic Sections**.

**Why waste time debugging subtle configuration typos or managing insecure secrets manually?** ConfigGuard catches errors early, simplifies maintenance, and secures your sensitive data, allowing you to focus on building great features.

**Adopt ConfigGuard and configure with confidence!**

---

## ✨ Key Features

*   📝 **Schema-Driven:** Define your configuration's expected structure, types, defaults, and validation rules within a Python dictionary or a JSON file. This acts as the single source of truth, ensuring consistency and enabling static analysis benefits. Schema definitions include a mandatory `__version__` key for robust version tracking.
*   <0xF0><0x9F><0xA7><0xB1> **Nested Configuration:** Structure complex settings logically using **sections**, defined directly within your schema (`"type": "section"`). Access nested settings intuitively through standard attribute or dictionary notation (e.g., `config.database.connection.pool_size`, `config['server']['ssl']['enabled']`), promoting code readability and maintainability.
*   <0xF0><0x9F><0x94><0x91> **Dynamic Sections:** For scenarios requiring flexibility (like plugin settings or user-defined mappings), define sections with an empty schema (`"schema": {}`). These sections behave like standard Python dictionaries, allowing the addition, modification, and deletion of arbitrary key-value pairs at runtime, while still benefiting from ConfigGuard's saving, loading, and encryption mechanisms.
*   🔒 **Built-in Encryption:** Secure sensitive configuration values (API keys, passwords, tokens) transparently using Fernet symmetric encryption (requires the `cryptography` library). Encryption is handled automatically by the storage backend during save/load operations, meaning your application code always interacts with plain, decrypted values.
*   💾 **Multiple Backends:** Persist your configurations in various formats through an extensible handler system. ConfigGuard automatically detects the desired format based on the file extension (`.json`, `.yaml`, `.yml`, `.toml`, `.db`, `.sqlite`, `.sqlite3`). Default handlers are provided for JSON, YAML, TOML, and SQLite.
*   🔄 **Versioning & Migration:** Embed a version string (e.g., `"1.2.0"`) in your schema's `__version__` key. When loading configuration files, ConfigGuard compares the file's version with the instance's schema version. It prevents loading configurations newer than the application expects and gracefully handles older versions by merging existing values, applying new defaults, and skipping settings/sections no longer present in the current schema. This migration logic operates **recursively through nested sections**.
*   <0xF0><0x9F><0x97><0x84>️ **Flexible Save Modes:** Control the granularity of saved data:
    *   `mode='values'` (default): Saves only the current configuration values. Ideal for runtime updates, preserving the structure (including nesting and dynamic content) according to the chosen handler's capabilities.
    *   `mode='full'`: Saves the complete state: the schema version, the full schema *definition* (including nested structures and empty schemas for dynamic sections), and the current values. Best for backups, transferring configurations between environments, or providing comprehensive state to external tools or UIs.
*   <0xF0><0x9F><0xA7><0xB1> **Supported Types:** Define standard settings with common Python types: `str`, `int`, `float`, `bool`, or `list`. *(Note: Validation of individual element types within lists is not currently implemented)*. Dynamic sections can store any value that is serializable by the chosen backend handler (typically JSON-serializable types).
*   🐍 **Intuitive Access:** Interact with your configuration naturally. Access values using attribute (`config.section.setting`, `config.dynamic_section.key`) or dictionary (`config['section']['setting']`, `config['dynamic_section']['key']`) syntax. Retrieve schema details for *defined* settings using the `sc_` prefix (`config.sc_section.sc_setting`).
*   ✔️ **Automatic Validation:** ConfigGuard automatically validates values against the schema rules (type, `nullable`, `options`, `min_val`, `max_val`) whenever a standard setting is modified or when data is loaded. This prevents invalid data from entering your configuration state. **Values added to dynamic sections bypass this schema validation.**
*   📤 **Easy Export/Import:**
    *   `export_schema_with_values()`: Get a snapshot of the entire configuration state (schema definition + current values, including dynamic content) as a dictionary, suitable for populating UIs, sending over APIs, or debugging.
    *   `import_config(data, ignore_unknown=True)`: Update the configuration *values* from a (potentially nested) dictionary. This merges data into the existing structure, applying validation for standard settings and adding/updating keys in dynamic sections. The `ignore_unknown` flag controls whether unexpected keys cause errors.
*   🧩 **Extensible:** Built with a clear `StorageHandler` interface, allowing developers to easily implement and register support for additional storage backends (e.g., databases, cloud services, environment variables).

---

## 🤔 Why Choose ConfigGuard?

ConfigGuard addresses common pain points in configuration management:

*   **Eliminate Runtime Config Errors:** Instead of discovering a typo in a port number or an invalid logging level only when your application crashes, ConfigGuard catches these errors early – either when the schema is defined or when data is loaded/set – thanks to its strict validation against your predefined rules.
*   **Secure Your Secrets with Ease:** Stop storing sensitive API keys, database passwords, or tokens in plain text files or insecure environment variables. ConfigGuard's integrated encryption provides a simple, transparent mechanism to protect this data at rest, requiring only a single encryption key and the `cryptography` library.
*   **Future-Proof Your Application:** As your application evolves, so will its configuration needs. ConfigGuard's versioning system allows you to update your schema confidently. When loading older config files, it automatically attempts to migrate the data, preserving user settings where possible and applying new defaults, significantly reducing the friction of application updates.
*   **Improve Code Clarity and Maintainability:** Schemas act as self-documentation for your configuration settings. The explicit definition of types, defaults, validation rules, and help strings makes it much easier for developers (including your future self) to understand what each setting does and how to configure it correctly. Nested sections further enhance organization.
*   **Manage Complexity Effectively:** Modern applications often have numerous configuration options. ConfigGuard allows you to tame this complexity by organizing settings into logical, hierarchical sections (both predefined and dynamic), making the overall configuration easier to navigate and manage.
*   **Increase Developer Productivity:** Eliminate the need to write repetitive, error-prone boilerplate code for parsing different config file formats, validating data types, checking ranges, handling defaults for missing values, and managing encryption. ConfigGuard handles these common tasks robustly.
*   **Gain Storage Freedom:** Start with JSON for simplicity, move to YAML for readability, use TOML if preferred, or leverage SQLite for transactional saves – all without changing how your application code interacts with the configuration object. The backend is abstracted away by the handler system.

---

## 🚀 Installation

ConfigGuard requires Python 3.8 or later.

**Base Installation (includes JSON and SQLite support):**

```bash
pip install configguard
```

**With Optional Features (Extras):**

ConfigGuard uses "extras" to manage dependencies for optional features like encryption and specific file format handlers.

*   **Encryption:** Requires the `cryptography` library.
    ```bash
    pip install configguard[encryption]
    # or install separately:
    # pip install cryptography
    ```

*   **YAML Support:** Requires the `PyYAML` library.
    ```bash
    pip install configguard[yaml]
    # or install separately:
    # pip install PyYAML
    ```

*   **TOML Support:** Requires the `toml` library.
    ```bash
    pip install configguard[toml]
    # or install separately:
    # pip install toml
    ```

*   *(SQLite support uses Python's built-in `sqlite3` and needs no extra pip install).*

**Installing Multiple Extras:**

You can combine extras in a single command:

```bash
# Example: Install with encryption, YAML, and TOML support
pip install configguard[encryption,yaml,toml]
```

**Installing All Optional Features:**

```bash
pip install configguard[all]
```

**For Development:**

If you want to contribute to ConfigGuard or run tests, clone the repository and install in editable mode with development dependencies:

```bash
# Clone the repository
git clone https://github.com/ParisNeo/ConfigGuard.git
cd ConfigGuard

# Install in editable mode (-e) with the 'dev' extra
pip install -e .[dev]
```
This installs ConfigGuard itself, plus tools like `pytest`, `black`, `ruff`, `mypy`, and the dependencies needed for all built-in handlers and encryption.

---

## ⚡ Quick Start

This example demonstrates defining a schema with standard and dynamic sections, initializing ConfigGuard, accessing/modifying values, and saving.

```python
from configguard import ConfigGuard, ValidationError, generate_encryption_key
from pathlib import Path
import typing # Required for type hinting the schema dict

# 1. Define your schema: Includes version, standard section, dynamic section, top-level setting.
CONFIG_VERSION = "1.1.0"
my_schema: typing.Dict[str, typing.Any] = {
    "__version__": CONFIG_VERSION, # Mandatory version key
    "server": { # Standard section with defined settings
        "type": "section",
        "help": "Core web server settings.",
        "schema": {
            "host": { "type": "str", "default": "127.0.0.1", "help": "IP address to bind to." },
            "port": { "type": "int", "default": 8080, "min_val": 1024, "max_val": 65535, "help": "Port number." }
        }
    },
    "plugin_data": { # DYNAMIC section - allows arbitrary keys later
        "type": "section",
        "help": "Stores runtime data or settings for plugins.",
        "schema": {} # Empty schema marks it as dynamic
    },
    "log_level": { # Standard top-level setting
        "type": "str",
        "default": "INFO",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
        "help": "Application logging verbosity."
    }
}

# 2. Setup file path and optional encryption key
config_file = Path("my_app_config.yaml") # Using YAML handler (requires PyYAML)
# Generate a key ONCE and store it securely (e.g., env variable, secrets manager)
# key = generate_encryption_key()
# print(f"Generated Key (store securely!): {key.decode()}")
# For demonstration, use a hardcoded key (NOT FOR PRODUCTION!)
encryption_key = b'bZJc8p9zQXnR1hQ9tVxLr2aGzU7J5mGzO6l3X8wJdIo='

# 3. Initialize ConfigGuard instance
try:
    # Pass schema, path, and key. Handler is chosen based on path suffix.
    config = ConfigGuard(
        schema=my_schema,
        config_path=config_file,
        encryption_key=encryption_key # Remove if not encrypting
    )
    print(f"ConfigGuard initialized using {type(config._handler).__name__}.")
# Handle cases where optional handler dependencies are missing
except ImportError as e:
    print(f"ERROR: Missing dependency required for {config_file.suffix} files: {e}")
    print("Please install the required extra, e.g., 'pip install configguard[yaml]'")
    exit()
except Exception as e:
    print(f"ERROR: Failed to initialize ConfigGuard: {e}")
    exit()

# 4. Access values (defaults initially, unless config_file existed and was loaded)
print(f"Initial Server Host: {config.server.host}") # -> '127.0.0.1'
print(f"Initial Log Level: {config['log_level']}") # -> 'INFO'
# Accessing dynamic section (initially empty)
print(f"Initial Plugin Data: {config.plugin_data.get_config_dict()}") # -> {}

# 5. Access schema details for defined settings
print(f"Help for server port: {config.server.sc_port.help}")
# Schema for the dynamic section itself shows its type and empty inner schema
print(f"Schema object for plugin_data section: {config.sc_plugin_data}")

# 6. Modify values: standard settings are validated, dynamic keys are not.
try:
    # Modify standard setting (validated)
    config.server.port = 9090
    config['log_level'] = 'DEBUG' # Also validated

    # Add data to the DYNAMIC section (no validation applied here)
    config.plugin_data['active_plugin'] = 'analyzer_v2'
    config.plugin_data.user_prefs = {'theme': 'dark', 'notifications': False}
    config.plugin_data.request_counts = {'analyzer_v2': 105, 'reporter': 5}

    # Update dynamic data
    config.plugin_data.request_counts['analyzer_v2'] += 1

    # config.server.port = 80 # This would raise ValidationError (min_val: 1024)
except ValidationError as e:
    print(f"VALIDATION ERROR: {e}")

print(f"Updated Server Port: {config.server.port}") # -> 9090
print(f"Updated Active Plugin: {config.plugin_data['active_plugin']}") # -> 'analyzer_v2'
print(f"Updated Counts: {config.plugin_data.request_counts}") # -> {'analyzer_v2': 106, 'reporter': 5}

# 7. Delete keys from DYNAMIC section
del config.plugin_data.user_prefs
print(f"Plugin data keys after deletion: {list(config.plugin_data.keys())}") # -> ['active_plugin', 'request_counts']

# 8. Save the current configuration values (including dynamic data)
# Since we initialized with an encryption key, the output file will be encrypted.
config.save(mode='values')
print(f"Configuration saved to {config_file} (encrypted).")

# 9. Automatic Loading on next run
# If you run this script again, ConfigGuard will find 'my_app_config.yaml',
# decrypt it using the same key, and load the saved values (port 9090, debug level, plugin data etc.)
# config_reloaded = ConfigGuard(schema=my_schema, config_path=config_file, encryption_key=encryption_key)
# print(f"Reloaded Port: {config_reloaded.server.port}") # Would output 9090
```

---

## 📚 Core Concepts Detailed

*   **Schema:** The cornerstone of ConfigGuard.
    *   **Structure:** A Python dictionary defining the entire configuration layout.
    *   **`__version__` (Mandatory):** A top-level key holding a version string (e.g., `"1.0.0"`, `"2.1-beta"`). Must be parseable by the `packaging` library. Used for migration control.
    *   **Settings:** Keys in the schema dictionary represent setting names. Each setting has a definition dictionary specifying its properties.
    *   **Sections (`"type": "section"`):** Allows hierarchical grouping. Requires a nested `"schema"` dictionary which defines the contents of the section.
    *   **Dynamic Sections (`"schema": {}`):** A special type of section defined with an empty schema dictionary. These sections allow runtime addition/modification/deletion of arbitrary key-value pairs without schema validation for those pairs.
    *   **Setting Definition Keys:**
        *   `type` (str, Mandatory): One of `str`, `int`, `float`, `bool`, `list`, or `section`.
        *   `default` (Any): The value used if none is loaded from the config file. Mandatory for settings where `nullable` is `False`. Must be valid according to other schema rules.
        *   `help` (str, Recommended): A description of the setting's purpose. Used for documentation and potentially UIs.
        *   `nullable` (bool, Optional, Default: `False`): If `True`, allows the setting's value to be `None`. If `False`, a value (or the default) must always be present.
        *   `options` (list, Optional): A list of allowed values for the setting. The setting's value must be one of the items in this list.
        *   `min_val` (int/float, Optional): Minimum allowed value for `int` or `float` types.
        *   `max_val` (int/float, Optional): Maximum allowed value for `int` or `float` types.

*   **ConfigGuard Object:** The main object you interact with.
    *   **Initialization:** Created with the schema definition, an optional `config_path`, an optional `encryption_key`, and optional `autosave` flag. Automatically attempts to load data from `config_path` if provided.
    *   **Access:** Provides attribute (`config.setting`) and dictionary (`config['setting']`) access to top-level settings and sections.
    *   **Schema Access:** Use `config.sc_setting` or `config['sc_setting']` to get the `SettingSchema` object for a *defined* setting, or the schema dictionary for a section.

*   **ConfigSection Object:** Represents a nested section defined in the schema.
    *   **Access:** Provides the same attribute (`section.nested_setting`) and dictionary (`section['nested_setting']`) access for items defined within its schema, or for dynamic keys if it's a dynamic section.
    *   **Modification Rules:** **You interact with the *contents* of a `ConfigSection`. You CANNOT replace the section itself** by assigning a dictionary to `config.section_name` or `config['section_name']`. This restriction applies to both standard and dynamic sections and prevents accidental destruction of the managed object structure.

*   **Dynamic Sections (`"schema": {}`):**
    *   **Purpose:** Useful when you don't know all configuration keys beforehand, such as for plugin configurations, user-defined mappings, or runtime feature flags.
    *   **Behavior:** Acts like a nested dictionary within the ConfigGuard structure. You can add (`section.new_key = val`), update (`section.key = new_val`), and delete (`del section.key`) keys freely using attribute or item access.
    *   **Trade-off:** Offers flexibility at the cost of **no schema validation** for the keys and values added dynamically. Type safety and constraints are not enforced for dynamic content.
    *   **Integration:** Dynamic content is included in saves (values and full), loads, exports, and is subject to encryption.

*   **Storage Handlers:** Abstraction layer for persistence.
    *   **Selection:** Automatically chosen based on `config_path` suffix (`.json`, `.yaml`, `.yml`, `.toml`, `.db`, `.sqlite`, `.sqlite3`).
    *   **Encryption:** Handlers manage encryption/decryption transparently if an `encryption_key` is provided to `ConfigGuard`.
    *   **Structure Handling:**
        *   JSON, YAML, TOML: Naturally support nested dictionaries, preserving the section structure (including dynamic content) directly.
        *   SQLite: Uses a flat key-value table (`config(key TEXT PK, value BLOB)`). Nested structures (standard and dynamic) are flattened using dot-notation keys (e.g., `server.port`, `plugin_data.my_plugin.enabled`). Values are stored as JSON strings, which are then encrypted if a key is provided. This allows storing complex values (lists, dicts within dynamic sections) but means the database representation differs from the in-memory object structure.
    *   **Default Encrypted Handlers (`.bin`, `.enc`):** Default to `JsonHandler`, assuming the underlying encrypted data is JSON. If you encrypt a YAML file and save it as `.bin`, loading it will likely fail unless you explicitly provide a `YamlHandler` instance during `ConfigGuard` initialization.

*   **Save Modes (`values` vs `full`):**
    *   `mode='values'`: Saves a representation of the current *values* only. The structure depends on the handler (nested for JSON/YAML/TOML, flattened for SQLite). This mode does **not** include the schema version or definition in the output file. Use cases: Runtime saving of user changes, simple persistence.
    *   `mode='full'`: Saves a complete snapshot: the schema `__version__`, the entire schema *definition* dictionary (including nested schemas and `{}` for dynamic sections), and the current *values* (structured according to the handler). Use cases: Creating configuration backups, transferring complete configurations between environments, providing full state to external tools/UIs, ensuring version information is stored with the data.

*   **Versioning & Migration:** Facilitates managing configuration changes over time.
    *   **Mechanism:** Compares `__version__` in the loaded file with the `__version__` in the current `ConfigGuard` instance's schema.
    *   **Loading Newer:** Raises `SchemaError` to prevent loading data potentially incompatible with older code.
    *   **Loading Older:** Merges data recursively:
        *   Values for keys present in both the old file and current schema are loaded (and validated). Type coercion is attempted for compatible types (int/float/str).
        *   Keys present in the current schema but *not* in the old file receive their default values.
        *   Keys/Sections present in the old file but *not* in the current schema are skipped (a warning is logged).
        *   Dynamic section *content* is loaded if the dynamic section itself still exists in the current schema. No migration is performed on the dynamic content itself.
    *   **Logging:** Warnings are logged for skipped keys/sections and failed coercions during migration.

*   **Encryption:** Provides confidentiality for configuration data at rest.
    *   **Algorithm:** Uses Fernet (AES-128-CBC with HMAC-SHA256 authentication), provided by the `cryptography` library. Requires a URL-safe base64-encoded 32-byte key.
    *   **Key Management:** Use `configguard.generate_encryption_key()` to create a suitable key. **You must store this key securely** (e.g., environment variable, dedicated secrets management system, secure file permissions). Losing the key means losing access to encrypted data.
    *   **Transparency:** Encryption/decryption is handled by the storage handler. Your application code (`config.setting = value`, `value = config.setting`) always deals with plain data. The data is only encrypted when being written to disk/DB and decrypted when being read. This applies to all values, including those within dynamic sections.

---

## 📖 Detailed Usage

### 1. Defining the Schema

(See Core Concepts section for key details)

```python
# More complex schema example
import typing
CONFIG_VERSION = "3.0.0"

complex_schema: typing.Dict[str, typing.Any] = {
    "__version__": CONFIG_VERSION,
    "network": {
        "type": "section", "help": "Network settings",
        "schema": {
            "hostname": {"type": "str", "default": "auto", "help": "System hostname (or 'auto')"},
            "port": {"type": "int", "default": 8000, "min_val": 1, "max_val": 65535},
            "allowed_ips": {"type": "list", "default": ["127.0.0.1", "::1"], "help": "List of allowed client IPs"},
        }
    },
    "performance": {
        "type": "section", "help": "Performance tuning",
        "schema": {
            "worker_threads": {"type": "int", "default": 4, "min_val": 1, "max_val": 64},
            "cache": {
                "type": "section", "help": "Caching options",
                "schema": {
                    "enabled": {"type": "bool", "default": True},
                    "max_size_mb": {"type": "int", "default": 1024, "min_val": 0},
                    "strategy": {"type": "str", "default": "LRU", "options": ["LRU", "FIFO", "LFU"]},
                }
            }
        }
    },
    "user_scripts": { # Dynamic section
        "type": "section", "help": "Paths to user-provided scripts.",
        "schema": {}
    },
    "enable_analytics": { "type": "bool", "default": False, "help": "Enable anonymous usage analytics."}
}

# Optionally save schema to JSON for reuse or distribution
# import json
# with open("app_schema_v3.json", "w") as f:
#     json.dump(complex_schema, f, indent=2)
```

### 2. Initializing ConfigGuard

Load schema from dictionary or file path. Provide `config_path` for persistence and `encryption_key` for security.

```python
from configguard import ConfigGuard, generate_encryption_key, SchemaError, HandlerError, EncryptionError
from pathlib import Path

schema_source = complex_schema # Or Path("app_schema_v3.json")
config_file = Path("app_config_v3.db") # Using SQLite
enc_key = generate_encryption_key() # Store this securely!

try:
    config = ConfigGuard(
        schema=schema_source,
        config_path=config_file,
        encryption_key=enc_key
    )
    print("ConfigGuard initialized.")
    # If config_file existed, its (decrypted) data is now loaded.
    # If not, config holds default values from the schema.

except FileNotFoundError as e:
    # Schema file not found if schema_source was a path
    print(f"Schema file error: {e}")
except SchemaError as e:
    # Invalid schema definition (missing version, bad type, etc.)
    print(f"Schema definition error: {e}")
except (HandlerError, ImportError) as e:
    # Error initializing handler (missing dependency like pyyaml, or DB error)
    print(f"Configuration handler error: {e}")
except EncryptionError as e:
    # Invalid encryption key or corrupted encrypted file
    print(f"Encryption error during initial load: {e}")
except Exception as e:
    print(f"An unexpected initialization error occurred: {e}")
```

### 3. Accessing Settings and Schema

Use attribute or item access. Iterate sections like dictionaries.

```python
# Access standard nested setting
cache_strategy = config.performance.cache.strategy
print(f"Cache Strategy: {cache_strategy}")

# Access top-level list setting
allowed = config.network.allowed_ips
print(f"Allowed IPs: {allowed}")

# Access dynamic content (assuming 'on_startup.py' was added)
startup_script = config.user_scripts.get('on_startup.py') # Use .get for safer dynamic access
if startup_script:
    print(f"Startup script: {startup_script}")

# Iterate over a standard section's defined settings
print("Network Settings:")
for key in config.network: # Iterates over keys defined in schema for network
    print(f" - {key}: {config.network[key]}")

# Iterate over a dynamic section's keys
print("User Scripts:")
for script_name, script_path in config.user_scripts.items():
     print(f" - {script_name}: {script_path}")

# Access schema details
print(f"Cache Max Size Default: {config.performance.cache.sc_max_size_mb.default_value}")
# Schema for dynamic section confirms its nature
print(f"User Scripts Schema: {config.sc_user_scripts}")
```

### 4. Modifying Settings

Assign values to standard settings (triggers validation) or dynamic keys (no validation). Use `del` for dynamic keys. **Do not assign to section names.**

```python
# Modify standard setting
config.performance.worker_threads = 8
config.network.allowed_ips.append("192.168.1.100") # Modify list in-place

# Add/Modify dynamic keys
config.user_scripts['on_shutdown'] = '/opt/scripts/shutdown.sh'
config.user_scripts.data_processor = {'type': 'python', 'path': '~/scripts/process.py'}

# Delete dynamic key
if 'on_startup.py' in config.user_scripts:
    del config.user_scripts['on_startup.py']

# --- INVALID ASSIGNMENTS ---
# config.network = {'port': 8080} # Error: Cannot assign to standard section
# config.user_scripts = {}       # Error: Cannot assign to dynamic section
```

### 5. Saving & Loading

`save()` persists the current state. `load()` reads from disk (usually happens at init).

```python
# Save current values (including dynamic user_scripts) encrypted to the SQLite DB
config.save() # Defaults to mode='values', uses path from init

# Save a full backup to a different file (e.g., YAML, unencrypted)
try:
    backup_path = Path("config_v3_backup.yaml")
    # Need to create a temporary instance or re-init without key for unencrypted save
    temp_config_for_save = ConfigGuard(schema=config.schema, encryption_key=None)
    # Import current values into temp instance
    temp_config_for_save.import_config(config.get_config_dict())
    temp_config_for_save.save(filepath=backup_path, mode='full')
    print(f"Full unencrypted backup saved to {backup_path}")
except ImportError:
    print("Cannot save YAML backup, PyYAML not installed.")
except Exception as e:
    print(f"Failed to save backup: {e}")


# Manual Load (if needed, e.g., after external file change)
try:
    print(f"Manually reloading from {config.config_path}")
    config.load() # Reloads from the path config was initialized with
    print("Reload complete.")
    print(f"Worker threads after reload: {config.performance.worker_threads}")
except FileNotFoundError:
    print(f"Cannot load: File {config.config_path} not found.")
except Exception as e:
    print(f"Error during manual load: {e}")
```

### 6. Versioning & Migration

Handled automatically during load based on `__version__`. Check logs for warnings about skipped keys or sections from older files.

```python
# --- Simulation ---
# Imagine current schema is V2.0.0
# Load a config file saved with schema V1.0.0:
# config_v1_path = Path("old_config_v1.json")
# try:
#    config_v2_instance = ConfigGuard(schema=schema_v2_dict, config_path=config_v1_path)
#    # ConfigGuard logs warnings for settings in v1 file not in v2 schema.
#    # Settings in v2 schema but not v1 file get v2 defaults.
#    # Matching settings have their values loaded from v1 file.
#    # Dynamic section content (if section exists in both) is loaded from v1 file.
# except SchemaError as e: # e.g. if v1 file version > v2 schema version
#    print(f"Version mismatch: {e}")
```

### 7. Encryption

Provide `encryption_key` at init. Generation and storage are key.

```python
from configguard import generate_encryption_key, ConfigGuard

# Generate key (DO THIS ONCE and store securely!)
# new_key = generate_encryption_key()
# print(f"Store this key safely: {new_key.decode()}")

# Use stored key
stored_key = b'YOUR_SECURELY_STORED_32_BYTE_URLSAFE_BASE64_KEY'

secure_config = ConfigGuard(
    schema=complex_schema,
    config_path="secure_app.bin", # Use .bin or .enc for encrypted files
    encryption_key=stored_key
)

# Modify sensitive and non-sensitive data
secure_config.network.hostname = "prod.server.local"
secure_config.user_scripts.deploy_key = "ssh-rsa AAA..." # Dynamic sensitive data

# Save - data is encrypted on disk
secure_config.save()

# Loading automatically decrypts
# loaded_config = ConfigGuard(...)
# print(loaded_config.user_scripts.deploy_key) # -> Prints plain key
```

### 8. Handling Nested Configurations

Define sections within sections in your schema. Access follows the structure naturally. Modification rules apply at each level.

```python
# Accessing deeply nested setting (from complex_schema)
cache_size = config.performance.cache.max_size_mb
print(f"Cache size: {cache_size}")

# Modifying deeply nested setting
config.performance.cache.enabled = False
config['performance']['cache']['strategy'] = 'FIFO' # Item access also works

# Cannot assign to nested section
# config.performance.cache = {"enabled": False} # INVALID
```

### 9. Import/Export

`export_schema_with_values()` provides a full snapshot. `import_config()` merges value updates.

```python
# --- Export ---
full_state = config.export_schema_with_values()

# Example structure of full_state['settings']:
# {
#   "network": { "schema": { ... }, "value": { "hostname": "auto", ... } },
#   "performance": {
#     "schema": { ... },
#     "value": {
#       "worker_threads": 8,
#       "cache": { "enabled": False, "max_size_mb": 1024, "strategy": "FIFO" } # Value is nested
#     }
#   },
#   "user_scripts": { # Dynamic section
#      "schema": { "type": "section", "help": "...", "schema": {} },
#      "value": { # Value contains the dynamic keys
#          "on_shutdown": "/opt/scripts/shutdown.sh",
#          "data_processor": { "type": "python", "path": "~/scripts/process.py" }
#      }
#   },
#   "enable_analytics": { "schema": { ... }, "value": False }
# }

import json
# print(json.dumps(full_state, indent=2))

# --- Import ---
update_data = {
    "performance": {
        "cache": {
            "max_size_mb": 2048, # Update nested standard setting
            "unknown_cache_param": True # Ignored if ignore_unknown=True
        }
    },
    "user_scripts": { # Add/update dynamic keys
        "new_report_script": "/usr/local/bin/report.py",
        "data_processor": { "type": "rust", "path": "/opt/bin/process_rs" } # Update dynamic value
    },
    "unknown_section": True # Ignored if ignore_unknown=True
}

try:
    # Merge updates, ignore keys not in schema (unless in dynamic section)
    config.import_config(update_data, ignore_unknown=True)
    print(f"Cache size after import: {config.performance.cache.max_size_mb}") # -> 2048
    print(f"Data processor after import: {config.user_scripts.data_processor}")
except SettingNotFoundError as e:
     print(f"Import failed (ignore_unknown=False): {e}")
except Exception as e:
    print(f"Import failed: {e}")
```

---

## 💡 Use Cases

*   **Robust Application Settings:** Define and manage essential parameters like server ports, file paths, feature flags, logging levels with guaranteed type safety and validation. Organize settings by application component (e.g., `server`, `database`, `ui`, `tasks`) using nested sections.
*   **Secure Credential Storage:** Store sensitive data like API keys, database connection strings with passwords, OAuth tokens, or encryption keys within specific sections (e.g., `credentials.database`, `credentials.external_api`). Enable encryption (`encryption_key`) to protect this data at rest transparently.
*   **User Preferences:** Manage user-specific application settings like themes, language choices, layout configurations, notification preferences. A standard section can enforce known preference keys, while a dynamic section could store UI state or less critical, user-defined preferences.
*   **Microservice Configuration:** Each service can have its own `ConfigGuard` schema defining its unique requirements (database connections, message queue endpoints, cache settings, service discovery URLs). Shared settings could potentially be managed through includes or layering if needed (though not a built-in feature).
*   **Multi-Environment Deployment:** Maintain consistency across development, staging, and production environments by using the same schema but different configuration files (`dev.yaml`, `staging.db`, `prod.toml`). Use encryption for production secrets. Versioning helps manage updates across environments.
*   **Plugin and Extension Systems:** Use dynamic sections (`"schema": {}`) to allow plugins or extensions to store their own configuration data without requiring modifications to the core application schema. The core app can load/save the dynamic section content, while the plugin interprets its own keys/values.
*   **Generating Configuration UIs:** Use the output of `export_schema_with_values()` to dynamically generate web forms or GUI elements for editing configurations. The schema provides field types, help text, options (for dropdowns), and validation rules (min/max) to build intelligent editors.
*   **Complex Workflow/Pipeline Configuration:** Define parameters for multi-step processes, data pipelines, or scientific workflows, potentially using nested sections for different stages and dynamic sections for stage-specific parameters.

---

## 🔧 Advanced Topics

*   **Custom Storage Handlers:** Extend ConfigGuard's capabilities by creating your own storage backend.
    1.  Subclass `configguard.handlers.StorageHandler`.
    2.  Implement the abstract `load(self, filepath)` and `save(self, filepath, data, mode)` methods.
        *   Your `load` must return a `LoadResult` dictionary (`{'version': Optional[str], 'schema': Optional[dict], 'values': dict}`).
        *   Your `save` must handle the `data` payload (`{'instance_version', 'schema_definition', 'config_values'}`) and the `mode` ('values' or 'full').
        *   If your handler should support encryption, use `self._encrypt(bytes)` and `self._decrypt(bytes)` internally, which leverage the Fernet instance passed during `__init__`.
        *   Consider how your format represents nested structures and dynamic section content.
    3.  Register your handler by adding its file extension(s) and class to the `configguard.handlers.HANDLER_MAP` dictionary, or provide an instance directly during `ConfigGuard` initialization using the `handler` argument.
*   **(Potential Future) Custom Migration Functions:** For complex schema changes between versions (e.g., renaming keys, splitting sections, complex type transformations), a future enhancement could allow users to register custom Python functions to handle specific version-to-version migrations beyond the default key matching and default filling.
*   **(Potential Future) Schema Includes/Composition:** For very large configurations, a mechanism to include or compose schemas from multiple files could be considered.

---

## 🤝 Contributing

Contributions are highly welcome and appreciated! Help make ConfigGuard even better.

1.  **Found a Bug or Have an Idea?** Check the [Issue Tracker](https://github.com/ParisNeo/ConfigGuard/issues) to see if it's already reported. If not, please open a new issue, providing as much detail as possible (code examples, error messages, expected vs. actual behavior).
2.  **Ready to Contribute Code?**
    *   **Fork the Repository:** Create your own fork on GitHub.
    *   **Create a Branch:** Make a new branch in your fork for your changes (e.g., `feature/add-new-handler`, `bugfix/fix-validation-edge-case`).
    *   **Develop:** Write your code, ensuring it adheres to the project's quality standards:
        *   **Style:** Follow PEP 8 guidelines. Use **Black** for code formatting (`black .`).
        *   **Linting:** Use **Ruff** for linting (`ruff check .`). Address reported issues.
        *   **Typing:** Add **Type Hints** (`typing`) to all functions and methods. Check with **Mypy** (`mypy configguard`).
        *   **Docstrings:** Write clear, informative docstrings (Google style preferred) for all public modules, classes, functions, and methods. Explain parameters, return values, raised exceptions, and usage.
    *   **Testing:** Add **unit tests** using `pytest` in the `tests/` directory for any new features or bug fixes. Ensure existing tests pass. Aim for high test coverage (`pytest --cov=configguard`).
    *   **Commit:** Write clear, concise commit messages explaining your changes.
    *   **Push & Pull Request:** Push your branch to your fork and open a Pull Request against the `main` branch of the original `ParisNeo/ConfigGuard` repository. Describe your changes in the PR description and link any relevant issues.
3.  **Code of Conduct:** Please note that this project is released with a Contributor Code of Conduct. By participating, you are expected to uphold this code. (A formal CODE_OF_CONDUCT.md file may be added later).

---

## 📜 License

ConfigGuard is distributed under the terms of the **Apache License 2.0**.

This means you are free to use, modify, and distribute the software for commercial or non-commercial purposes, but you must include the original copyright notice and license text. See the [LICENSE](LICENSE) file in the repository for the full license text.

---

<p align="center">
  Built with ❤️ by ParisNeo with the help of Gemini 2.5
</p>