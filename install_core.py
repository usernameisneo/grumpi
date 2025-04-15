# install_core.py
import sys
import os
import subprocess
import venv
import shutil
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
CONFIG_EXAMPLE = PROJECT_ROOT / "config.toml.example"
CONFIG_DEST = PROJECT_ROOT / "config.toml"
MIN_PYTHON_VERSION = (3, 9)

# --- Helper Functions ---

def print_step(message):
    print(f"\n--- {message} ---")

def print_success(message):
    print(f"Success: {message}")

def print_warning(message):
    print(f"Warning: {message}")

def print_error(message, exit_code=1):
    print(f"ERROR: {message}", file=sys.stderr)
    if exit_code is not None:
        sys.exit(exit_code)

def check_python_version():
    print_step(f"Checking Python Version (>= {'.'.join(map(str, MIN_PYTHON_VERSION))})")
    if sys.version_info < MIN_PYTHON_VERSION:
        print_error(f"Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or higher is required. You have {platform.python_version()}. Please upgrade Python.")
    print_success(f"Python version {sys.version} is compatible.")

def create_virtual_environment():
    print_step(f"Creating Virtual Environment in '{VENV_DIR}'")
    if VENV_DIR.exists():
        print_warning(f"Virtual environment directory '{VENV_DIR}' already exists. Skipping creation.")
        return

    try:
        venv.create(VENV_DIR, with_pip=True)
        print_success("Virtual environment created.")
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")

def get_executable_path(env_dir, executable_name):
    """Gets the platform-specific path to an executable in the venv."""
    if sys.platform == "win32":
        path = env_dir / "Scripts" / f"{executable_name}.exe"
    else:
        path = env_dir / "bin" / executable_name
    return path

def install_dependencies():
    print_step(f"Installing Dependencies from '{REQUIREMENTS_FILE.name}'")
    if not REQUIREMENTS_FILE.exists():
        print_error(f"'{REQUIREMENTS_FILE.name}' not found. Cannot install dependencies.")

    python_exe = get_executable_path(VENV_DIR, "python")
    pip_exe = get_executable_path(VENV_DIR, "pip")

    if not python_exe.exists() or not pip_exe.exists():
        print_error("Could not find python or pip executables in the virtual environment. Setup might be incomplete.")

    command = [str(pip_exe), "install", "-r", str(REQUIREMENTS_FILE)]
    print(f"Running: {' '.join(command)}")

    try:
        # Use check=True to raise CalledProcessError on failure
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print_success("Core dependencies installed successfully.")
        # print(result.stdout) # Optionally print pip output
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies. Pip exited with code {e.returncode}.", exit_code=None)
        print("--- Pip Output ---")
        print(e.stdout)
        print("--- Pip Error ---")
        print(e.stderr)
        print("------------------")
        print_error("Dependency installation failed. Please check the errors above.", exit_code=1)
    except Exception as e:
         print_error(f"An unexpected error occurred during dependency installation: {e}")


def setup_configuration_file():
    print_step("Setting up Configuration File")
    if not CONFIG_EXAMPLE.exists():
        print_warning(f"'{CONFIG_EXAMPLE.name}' not found. Cannot create default config.")
        return

    if CONFIG_DEST.exists():
        print_warning(f"'{CONFIG_DEST.name}' already exists. Skipping creation.")
        print(f"{' '*9}Please review and edit '{CONFIG_DEST.name}' manually if needed.")
        return

    try:
        shutil.copyfile(CONFIG_EXAMPLE, CONFIG_DEST)
        print_success(f"Default configuration copied to '{CONFIG_DEST.name}'.")
        print(f"{' '*9}IMPORTANT: Edit '{CONFIG_DEST.name}' to set API keys, configure bindings, models, etc.")
    except Exception as e:
        print_error(f"Failed to copy configuration file: {e}")


# --- Main Installation Logic ---
if __name__ == "__main__":
    print("Starting lollms_server setup...")

    check_python_version()
    create_virtual_environment()
    install_dependencies()
    setup_configuration_file()

    print_step("Setup Complete!")
    print("\nNext Steps:")
    if sys.platform == "win32":
        print(f"1. Activate the virtual environment: .\\{VENV_DIR.name}\\Scripts\\activate")
        print(f"2. Edit the configuration file: notepad {CONFIG_DEST.name}")
        print(f"3. (Optional) Install extras for specific bindings (e.g., '.\\{VENV_DIR.name}\\Scripts\\pip install openai ollama pygame')")
        print(f"4. Run the server using the run script: .\\run.bat")
    else:
        print(f"1. Activate the virtual environment: source {VENV_DIR.name}/bin/activate")
        print(f"2. Edit the configuration file: nano {CONFIG_DEST.name}  (or your preferred editor)")
        print(f"3. (Optional) Install extras for specific bindings (e.g., './{VENV_DIR.name}/bin/pip install openai ollama pygame')")
        print(f"4. Run the server using the run script: ./run.sh")
    print(f"5. Access the server (usually at http://localhost:9600 if defaults are used).")