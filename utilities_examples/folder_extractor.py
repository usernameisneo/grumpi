# -*- coding: utf-8 -*-
"""
=========================================================================
 Standalone PyQt5 Folder Structure to Text Application (v1.3 - Custom Prompt)
=========================================================================
Based on the Lollms Function Call: Folder Structure to Text by ParisNeo
Adapted for PyQt5 by: Claude / AI Assistant
Creation Date: 2024-05-27
Last Update: 2024-05-29 (Added Custom Prompt Text Area)
Version: 1.3.0
Description:
  A PyQt5 application that takes a folder path and generates a Markdown-
  formatted text representation. Provides flexible exclusion options, max file size,
  saving output, enhanced environment management (save/load/recent/manage),
  remembers last used paths, and allows adding custom instructions to the output.
=========================================================================
"""
import sys
import fnmatch
import datetime
import re
import json
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict, Any
import functools # For passing arguments with signals

# --- Configuration Constants ---
DEFAULT_EXCLUDED_FOLDERS: Set[str] = {
    ".git", "__pycache__", "node_modules", "target", "dist", "build", "venv",
    ".venv", "env", ".env", ".vscode", ".idea", "logs", "temp", "tmp", "bin", "obj",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis", "*.egg-info"
}
DEFAULT_EXCLUDED_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".o", ".obj", ".class", ".dll", ".so", ".exe", ".bin",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".jar", ".war", ".ear", ".png", ".jpg",
    ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".ico", ".mp3", ".wav", ".ogg", ".mp4",
    ".avi", ".mov", ".webm", ".db", ".sqlite", ".sqlite3", ".lock", ".pdf", ".doc",
    ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp", ".ttf", ".otf",
    ".woff", ".woff2", ".DS_Store", ".ipynb_checkpoints",
}
ALLOWED_TEXT_EXTENSIONS: Set[str] = {
    ".txt", ".md", ".markdown", ".rst", ".adoc", ".asciidoc", ".py", ".java", ".js", ".ts",
    ".jsx", ".tsx", ".html", ".htm", ".css", ".scss", ".sass", ".less", ".c", ".cpp", ".h",
    ".hpp", ".cs", ".go", ".rs", ".swift", ".kt", ".kts", ".php", ".rb", ".pl", ".pm", ".lua",
    ".sh", ".bash", ".zsh", ".bat", ".ps1", ".psm1", ".sql", ".r", ".dart", ".groovy", ".scala",
    ".clj", ".cljs", ".cljc", ".edn", ".vb", ".vbs", ".f", ".for", ".f90", ".f95", ".json",
    ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf", ".properties", ".csv", ".tsv",
    ".env", ".dockerfile", "dockerfile", ".tf", ".tfvars", ".hcl", ".gradle", ".pom",
    ".csproj", ".vbproj", ".sln", ".gitignore", ".gitattributes", ".npmrc", ".yarnrc",
    ".editorconfig", ".babelrc", ".eslintrc", ".prettierrc", ".stylelintrc", ".makefile",
    "makefile", "Makefile", "CMakeLists.txt", ".tex", ".bib", ".sty", ".graphql", ".gql",
    ".vue", ".svelte", ".astro", ".liquid", ".njk", ".jinja", ".jinja2", ".patch", ".diff",
}
TREE_BRANCH, TREE_LAST, TREE_VLINE, TREE_SPACE = "├─ ", "└─ ", "│  ", "   "
FOLDER_ICON, FILE_ICON = "📁", "📄"
DEFAULT_MAX_FILE_SIZE_MB = 1.0
PRESET_EXCLUSIONS: Dict[str, List[str]] = {
    "Python Project": [
        "*.pyc", "*.pyo", "*.pyd", "__pycache__/", "venv/", ".venv/", "env/", ".env/",
        ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".hypothesis/", "build/",
        "dist/", "*.egg-info/", "htmlcov/", ".coverage", "instance/", "*.sqlite3", "*.db",
    ],
    "Node.js Project": [
        "node_modules/", "npm-debug.log*", "yarn-debug.log*", "yarn-error.log*", ".npm",
        ".yarn", "dist/", "build/", "out/", ".env*", "*.local", "coverage/", ".DS_Store",
    ],
    "C/C++ Project": [
        "*.o", "*.obj", "*.a", "*.lib", "*.so", "*.dylib", "*.dll", "*.exe", "build/",
        "bin/", "obj/", "Debug/", "Release/", "*.out", "*.gch", "*.stackdump", ".vscode/",
        ".ccls-cache/", ".cache/", "CMakeCache.txt", "CMakeFiles/", "cmake_install.cmake",
        "CTestTestfile.cmake", "compile_commands.json",
    ],
    "Rust Project": ["target/", "*.rlib", "*.so", "*.dylib", "*.dll", "*.a", "*.exe"],
    "Java Project": [
        "*.class", "*.jar", "*.war", "*.ear", "target/", "build/", "bin/", "out/",
        ".gradle/", ".mvn/", "hs_err_pid*.log", ".project", ".classpath", ".settings/",
        "*.iml", ".idea/",
    ],
}
PRESET_OPTIONS = ["None/Defaults", "Custom"] + sorted(PRESET_EXCLUSIONS.keys())
# --- End Constants ---

# --- Default Settings Dictionary (Includes new custom_prompt) ---
DEFAULT_SETTINGS: Dict[str, Any] = {
    "folder_path": "",
    "preset": "None/Defaults",
    "custom_folders": "",
    "custom_extensions": "",
    "custom_patterns": "",
    "dynamic_patterns": "",
    "max_file_size_mb": DEFAULT_MAX_FILE_SIZE_MB,
    "save_output_checked": False,
    "custom_prompt": "", # Added new setting
}
MAX_RECENT_ENVS = 10 # Max number of recent environment files to store

try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QDialogButtonBox,
        QLineEdit, QLabel, QFileDialog, QTextEdit, QComboBox, QGroupBox, QDialog, QListWidgetItem,
        QFormLayout, QDoubleSpinBox, QCheckBox, QMessageBox, QSplitter, QAction, QStatusBar,
        QMainWindow, QMenu
    )
    from PyQt5.QtCore import Qt, QSettings, QSize
    from PyQt5.QtGui import QFont, QIcon
except ImportError:
    print("Error: PyQt5 is required. Please install it using 'pip install PyQt5'")
    sys.exit(1)

# --- Helper Functions ---
def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip(' _')
    return sanitized or "default_folder_name"

# --- Core Processing Class (FolderProcessor - Unchanged) ---
class FolderProcessor:
    # --- Start FolderProcessor code (exactly as in v1.2) ---
    def __init__(self, status_callback=print, warning_callback=print, error_callback=print):
        self.status_callback = status_callback
        self.warning_callback = warning_callback
        self.error_callback = error_callback
        self._active_excluded_folders: Set[str] = set()
        self._active_excluded_extensions: Set[str] = set()
        self._active_excluded_patterns: List[str] = []
        self._max_file_size_bytes: int = 0

    def setup_exclusions_and_limits(
        self,
        preset_name: str,
        custom_folders_str: str,
        custom_exts_str: str,
        custom_patterns_str: str,
        dynamic_patterns_list: List[str],
        max_size_mb: float
    ):
        self._active_excluded_folders = set(DEFAULT_EXCLUDED_FOLDERS)
        self._active_excluded_extensions = set(DEFAULT_EXCLUDED_EXTENSIONS)
        self._active_excluded_patterns = []

        static_folders_set: Set[str] = set()
        static_exts_set: Set[str] = set()
        static_patterns_list: List[str] = []

        if preset_name == "Custom":
            static_folders_set = {f.strip().lower() for f in custom_folders_str.split(',') if f.strip()}
            static_exts_set = {e.strip().lower() for e in custom_exts_str.split(',') if e.strip() and e.strip().startswith('.')}
            static_patterns_list = [p.strip() for p in custom_patterns_str.split(',') if p.strip()]
        elif preset_name in PRESET_EXCLUSIONS:
            preset_patterns = PRESET_EXCLUSIONS.get(preset_name, [])
            static_patterns_list.extend(preset_patterns)
        elif preset_name != "None/Defaults":
            self.warning_callback(f"Unknown preset '{preset_name}'. Falling back to None/Defaults.")

        self._active_excluded_folders.update(static_folders_set)
        self._active_excluded_extensions.update(static_exts_set)
        self._active_excluded_patterns.extend(static_patterns_list)

        if dynamic_patterns_list:
            self._active_excluded_patterns.extend(dynamic_patterns_list)

        self._max_file_size_bytes = int(max(0.01, max_size_mb) * 1024 * 1024)

    def _is_excluded(self, item: Path) -> bool:
        item_name_lower = item.name.lower()
        item_suffix_lower = item.suffix.lower() if item.is_file() else ""
        if item.is_dir() and item_name_lower in self._active_excluded_folders: return True
        if item.is_file() and item_suffix_lower in self._active_excluded_extensions: return True
        for pattern in self._active_excluded_patterns:
            if fnmatch.fnmatchcase(item.name, pattern): return True
            if item.is_dir() and pattern.endswith('/') and fnmatch.fnmatchcase(item.name, pattern[:-1]): return True
        return False

    def _is_text_file(self, file: Path) -> bool:
        return file.suffix.lower() in ALLOWED_TEXT_EXTENSIONS

    def _read_file_content(self, file: Path) -> str:
        try:
            file_size = file.stat().st_size
            if file_size > self._max_file_size_bytes:
                 max_size_mb = self._max_file_size_bytes / (1024 * 1024)
                 return f"[File content omitted: Exceeds size limit ({max_size_mb:.2f} MB)]"
            if file_size == 0: return "[Empty file]"
            try:
                with open(file, "r", encoding="utf-8") as f: content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file, "r", encoding="latin-1") as f: content = f.read()
                except Exception as read_err:
                    self.warning_callback(f"Error reading file {file.name} with latin-1: {str(read_err)}")
                    return f"[Error reading file: Could not decode]"
            except Exception as read_err:
                self.warning_callback(f"Error reading file {file.name} with UTF-8: {str(read_err)}")
                return f"[Error reading file: {str(read_err)}]"
            return content if content else "[File appears empty after read]"
        except OSError as os_err:
            self.warning_callback(f"OS error accessing file {file.name}: {str(os_err)}")
            return f"[Error accessing file: {str(os_err)}]"
        except Exception as e:
            self.error_callback(f"Unexpected error reading file {file.name}: {str(e)}")
            return f"[Unexpected error reading file]"

    def _build_tree_and_collect_files(self, folder: Path, prefix: str = "") -> Tuple[List[str], List[Path]]:
        tree_lines, found_files = [], []
        try:
            filtered_items = [item for item in folder.iterdir() if not self._is_excluded(item)]
            items = sorted(filtered_items, key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
             tree_lines.append(f"{prefix}[Error: Permission denied]")
             self.warning_callback(f"Permission denied listing directory: {folder}")
             return tree_lines, found_files
        except OSError as e:
            tree_lines.append(f"{prefix}[Error: {e.strerror}]")
            self.warning_callback(f"OS error listing directory {folder}: {e}")
            return tree_lines, found_files
        num_items = len(items)
        for i, item in enumerate(items):
            is_last = (i == num_items - 1)
            connector = TREE_LAST if is_last else TREE_BRANCH
            line_prefix, child_prefix = prefix + connector, prefix + (TREE_SPACE if is_last else TREE_VLINE)
            if item.is_dir():
                tree_lines.append(f"{line_prefix}{FOLDER_ICON} {item.name}/")
                sub_tree_lines, sub_found_files = self._build_tree_and_collect_files(item, child_prefix)
                tree_lines.extend(sub_tree_lines); found_files.extend(sub_found_files)
            elif item.is_file():
                if self._is_text_file(item):
                    tree_lines.append(f"{line_prefix}{FILE_ICON} {item.name}"); found_files.append(item)
                else: tree_lines.append(f"{line_prefix}{FILE_ICON} {item.name} [Skipped: Non-text/Binary]")
        return tree_lines, found_files

    def _generate_file_contents_markdown(self, root_folder: Path, file_paths: List[Path]) -> List[str]:
        content_lines = ["", "---", "", "## File Contents"]
        if not file_paths:
            content_lines.append("\n*No text files found or included based on filters and size limits.*")
            return content_lines
        file_paths.sort(key=lambda p: p.relative_to(root_folder) if root_folder in p.parents else p.name)
        for file_path in file_paths:
            try: relative_path = file_path.relative_to(root_folder)
            except ValueError:
                 relative_path = file_path.name
                 self.warning_callback(f"Could not determine relative path for {file_path} (root: {root_folder}). Using filename.")
            content_lines.append(f"\n### `{relative_path}`")
            file_content = self._read_file_content(file_path)
            lang = file_path.suffix[1:].lower() if file_path.suffix else "text"
            lang = "".join(c for c in lang if c.isalnum()) or "text"
            aliases = {'py': 'python', 'js': 'javascript', 'md': 'markdown', 'sh': 'bash', 'yml': 'yaml',
                       'dockerfile': 'docker', 'h': 'c', 'hpp': 'cpp', 'cs': 'csharp', 'ts': 'typescript',
                       'rb': 'ruby', 'pl': 'perl', 'kt': 'kotlin', 'rs': 'rust', 'go': 'golang'}
            lang = aliases.get(lang, lang)
            content_lines.append(f"```{lang}")
            content_lines.extend(file_content.splitlines() if file_content else [""])
            content_lines.append("```")
        return content_lines

    def generate_structure_text(self, folder_path_str: str, preset_name: str, custom_folders_str: str,
                                custom_exts_str: str, custom_patterns_str: str, dynamic_exclude_str: str,
                                max_size_mb: float, custom_prompt: str = "") -> str: # Added custom_prompt argument
        """ Main processing method. Includes custom prompt at the end if provided. """
        if not folder_path_str: return "```error\nError: No folder path specified.\n```"
        try:
            folder = Path(folder_path_str).resolve()
            if not folder.exists(): return f"```error\nError: Folder not found: {folder}\n```"
            if not folder.is_dir(): return f"```error\nError: Path is not a directory: {folder}\n```"
        except Exception as e:
            self.error_callback(f"Error resolving path '{folder_path_str}': {e}")
            return f"```error\nError resolving path: {e}\n```"
        dynamic_patterns = [p.strip() for p in dynamic_exclude_str.split(',') if p.strip()]
        try:
            self.setup_exclusions_and_limits(preset_name, custom_folders_str, custom_exts_str,
                                            custom_patterns_str, dynamic_patterns, max_size_mb)
        except Exception as e:
            self.error_callback(f"Error setting up exclusions: {e}")
            return f"```error\nError setting up exclusions: {e}\n```"
        self.status_callback(f"Starting analysis for: {folder}...")
        try:
            tree_lines, found_files = self._build_tree_and_collect_files(folder, prefix="")
            structure_output_lines = [f"# Folder Structure: {folder.name}",
                f"*(Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})*",
                "", "```text", f"{FOLDER_ICON} {folder.name}/"]
            structure_output_lines.extend(tree_lines)
            structure_output_lines.append("```")
            content_output_lines = self._generate_file_contents_markdown(folder, found_files)
            full_output = "\n".join(structure_output_lines) + "\n" + "\n".join(content_output_lines)

            # --- Append Custom Prompt ---
            if custom_prompt and custom_prompt.strip():
                full_output += "\n\n---\n\n## Custom Instructions\n\n" + custom_prompt.strip()
            # --- End Append Custom Prompt ---

            self.status_callback(f"Analysis complete. Found {len(found_files)} text files to include.")
            return full_output.strip()
        except PermissionError as pe:
             self.error_callback(f"Permission error accessing path {folder}: {str(pe)}")
             return f"```error\nError: Permission denied accessing folder contents: {str(pe)}\n```"
        except Exception as e:
            self.error_callback(f"An unexpected error occurred during processing: {str(e)}")
            return f"```error\nError: An unexpected error occurred: {str(e)}\n```"
    # --- End FolderProcessor code ---


# --- Dialog for Managing Recent Environments (Unchanged from v1.2) ---
class ManageRecentDialog(QDialog):
    # --- Start ManageRecentDialog code (exactly as in v1.2) ---
    def __init__(self, recent_envs: List[Dict[str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Recent Environments")
        self.setMinimumWidth(400)
        self.recent_envs = recent_envs
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select environments to remove:"))
        self.list_widget = QListWidget()
        for env in self.recent_envs:
            item = QListWidgetItem(f"{env['name']} ({env['path']})")
            item.setData(Qt.UserRole, env['path'])
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected)
        layout.addWidget(remove_button)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def remove_selected(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items: return
        confirm = QMessageBox.question(self, "Confirm Removal",
                                       f"Are you sure you want to remove {len(selected_items)} selected entries?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            paths_to_remove = {item.data(Qt.UserRole) for item in selected_items}
            self.recent_envs = [env for env in self.recent_envs if env['path'] not in paths_to_remove]
            self.list_widget.clear()
            for env in self.recent_envs:
                item = QListWidgetItem(f"{env['name']} ({env['path']})")
                item.setData(Qt.UserRole, env['path'])
                self.list_widget.addItem(item)

    def get_updated_list(self) -> List[Dict[str, str]]:
        return self.recent_envs
    # --- End ManageRecentDialog code ---


# --- PyQt5 Main Application Window ---
class FolderStructureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyCompany", "FolderStructureApp")
        self.processor = FolderProcessor(
            status_callback=self.show_status,
            warning_callback=self.show_warning,
            error_callback=self.show_error
        )
        self.current_environment_file = None # Path object or None
        self.recent_environments: List[Dict[str, str]] = []

        self.initUI() # Initialize UI elements first
        self.load_persistent_settings() # Load paths and recent list
        self.create_actions() # Create actions
        self.create_menus() # Create menus
        self.create_status_bar()
        self._update_recent_menu() # Populate recent menu
        self.load_initial_environment() # Try loading last env or defaults


    def show_status(self, message):
        self.statusBar().showMessage(message, 5000)
        print(f"INFO: {message}")

    def show_warning(self, message):
        self.statusBar().showMessage(f"Warning: {message}", 8000)
        QMessageBox.warning(self, "Warning", message)
        print(f"WARNING: {message}")

    def show_error(self, message):
        self.statusBar().showMessage(f"Error: {message}", 10000)
        QMessageBox.critical(self, "Error", message)
        print(f"ERROR: {message}")

    def initUI(self):
        self.setWindowTitle('Folder Structure to Text v1.3')
        self.setGeometry(150, 150, 950, 800) # Slightly increased height for prompt area
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left Panel: Controls ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(15)

        # Folder Selection Group (Unchanged)
        folder_group = QGroupBox("Target Folder")
        folder_layout = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Select the target folder...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(browse_button)
        folder_group.setLayout(folder_layout)
        controls_layout.addWidget(folder_group)

        # Exclusion Settings Group (Unchanged)
        exclusion_group = QGroupBox("Exclusion Settings")
        exclusion_layout = QVBoxLayout()
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Exclusion Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESET_OPTIONS)
        self.preset_combo.currentIndexChanged.connect(self.update_custom_fields_state)
        preset_layout.addWidget(self.preset_combo)
        exclusion_layout.addLayout(preset_layout)
        self.custom_group = QGroupBox("Custom Static Exclusions (Enabled when Preset is 'Custom')")
        custom_form_layout = QFormLayout()
        self.custom_folders_edit = QLineEdit()
        self.custom_folders_edit.setPlaceholderText("e.g., docs,tests,temp")
        self.custom_extensions_edit = QLineEdit()
        self.custom_extensions_edit.setPlaceholderText("e.g., .log,.tmp,.bak")
        self.custom_patterns_edit = QLineEdit()
        self.custom_patterns_edit.setPlaceholderText("e.g., *.log,temp_*,cache_*/")
        custom_form_layout.addRow("Exclude Folders:", self.custom_folders_edit)
        custom_form_layout.addRow("Exclude Exts:", self.custom_extensions_edit)
        custom_form_layout.addRow("Exclude Patterns:", self.custom_patterns_edit)
        self.custom_group.setLayout(custom_form_layout)
        exclusion_layout.addWidget(self.custom_group)
        dynamic_layout = QFormLayout()
        self.dynamic_exclude_edit = QLineEdit()
        self.dynamic_exclude_edit.setPlaceholderText("e.g., *.private,credentials.json")
        dynamic_layout.addRow("Dynamic Exclude Patterns:", self.dynamic_exclude_edit)
        exclusion_layout.addLayout(dynamic_layout)
        exclusion_group.setLayout(exclusion_layout)
        controls_layout.addWidget(exclusion_group)

        # Other Options Group (Unchanged)
        options_group = QGroupBox("Options")
        options_layout = QFormLayout()
        self.max_size_spinbox = QDoubleSpinBox()
        self.max_size_spinbox.setSuffix(" MB")
        self.max_size_spinbox.setMinimum(0.01); self.max_size_spinbox.setMaximum(100.0)
        self.max_size_spinbox.setSingleStep(0.1)
        self.save_output_checkbox = QCheckBox("Save Markdown output on Generate")
        options_layout.addRow("Max File Size:", self.max_size_spinbox)
        options_layout.addRow(self.save_output_checkbox)
        options_group.setLayout(options_layout)
        controls_layout.addWidget(options_group)

        # --- NEW: Custom Instructions Prompt Group ---
        prompt_group = QGroupBox("Custom Instructions Prompt (Appended to Output)")
        prompt_layout = QVBoxLayout()
        self.custom_prompt_edit = QTextEdit()
        self.custom_prompt_edit.setPlaceholderText("Enter any additional instructions or context here...")
        self.custom_prompt_edit.setAcceptRichText(False) # Plain text only
        self.custom_prompt_edit.setFixedHeight(100) # Give it some initial height
        prompt_layout.addWidget(self.custom_prompt_edit)
        prompt_group.setLayout(prompt_layout)
        controls_layout.addWidget(prompt_group)
        # --- End Custom Prompt Group ---

        # Action Button (Unchanged)
        self.generate_button = QPushButton("Generate Structure Text")
        self.generate_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.generate_button.clicked.connect(self.generate_structure)
        controls_layout.addWidget(self.generate_button)

        controls_layout.addStretch(1) # Push controls to the top

        # --- Right Panel: Output (Unchanged) ---
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.addWidget(QLabel("Generated Markdown Output:"))
        self.output_textedit = QTextEdit()
        self.output_textedit.setReadOnly(True)
        self.output_textedit.setFont(QFont("Courier New", 10))
        self.output_textedit.setLineWrapMode(QTextEdit.NoWrap)
        clear_output_button = QPushButton("Clear Output")
        clear_output_button.clicked.connect(self.clear_output)
        output_layout.addWidget(self.output_textedit)
        output_layout.addWidget(clear_output_button, 0, Qt.AlignRight)

        splitter.addWidget(controls_widget)
        splitter.addWidget(output_widget)
        splitter.setSizes([380, 570]) # Keep same split ratio
        main_layout.addWidget(splitter)
        self.update_custom_fields_state()

    def create_actions(self):
        # Actions remain the same as v1.2
        self.load_env_action = QAction("&Load Environment...", self, shortcut="Ctrl+L",
                                       statusTip="Load configuration from a JSON file", triggered=self.load_environment_dialog)
        self.save_env_action = QAction("&Save Environment", self, shortcut="Ctrl+S",
                                       statusTip="Save current configuration to the current file", triggered=self.save_current_environment)
        self.save_env_as_action = QAction("Save Environment &As...", self, shortcut="Ctrl+Shift+S",
                                          statusTip="Save current configuration to a new JSON file", triggered=self.save_environment_as)
        self.manage_recent_action = QAction("&Manage Recent Environments...", self,
                                            statusTip="View and remove recent environment files", triggered=self.manage_recent_environments)
        self.exit_action = QAction("E&xit", self, shortcut="Ctrl+Q",
                                   statusTip="Exit the application", triggered=self.close)
        self.reset_settings_action = QAction("&Reset Settings", self, shortcut="Ctrl+R",
                                          statusTip="Reset all settings to their defaults", triggered=self.reset_settings)
        self.clear_output_action = QAction("&Clear Output", self, shortcut="Ctrl+Shift+C",
                                           statusTip="Clear the output text area", triggered=self.clear_output)
        self.save_env_action.setEnabled(self.current_environment_file is not None)

    def create_menus(self):
        # Menus remain the same as v1.2
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.load_env_action)
        self.recent_env_menu = file_menu.addMenu("Load Recent Environment")
        file_menu.addAction(self.manage_recent_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_env_action)
        file_menu.addAction(self.save_env_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.reset_settings_action)
        edit_menu.addAction(self.clear_output_action)

    def create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def browse_folder(self):
        # Unchanged from v1.2
        start_dir = self.settings.value("lastBrowseDir", str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.folder_path_edit.text() or start_dir)
        if folder:
            self.folder_path_edit.setText(folder)
            self.settings.setValue("lastBrowseDir", folder)
            self.settings.setValue("lastFolderPath", folder)

    def update_custom_fields_state(self):
        # Unchanged from v1.2
        is_custom = (self.preset_combo.currentText() == "Custom")
        self.custom_group.setEnabled(is_custom)

    def clear_output(self):
        # Unchanged from v1.2
        self.output_textedit.clear()
        self.show_status("Output cleared.")

    def reset_settings(self):
        # Modified to include resetting the custom prompt
        confirm = QMessageBox.question(self, "Confirm Reset",
                                       "Reset all settings to defaults? The current environment file association will be lost.",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.apply_settings_to_ui(DEFAULT_SETTINGS) # Includes clearing prompt
            self.current_environment_file = None
            self._update_window_title()
            self.save_env_action.setEnabled(False)
            self.show_status("Settings reset to defaults.")

    # --- State Management (Modified) ---
    def get_settings_from_ui(self) -> Dict[str, Any]:
        # Modified to include custom prompt
        return {
            "folder_path": self.folder_path_edit.text(),
            "preset": self.preset_combo.currentText(),
            "custom_folders": self.custom_folders_edit.text(),
            "custom_extensions": self.custom_extensions_edit.text(),
            "custom_patterns": self.custom_patterns_edit.text(),
            "dynamic_patterns": self.dynamic_exclude_edit.text(),
            "max_file_size_mb": self.max_size_spinbox.value(),
            "save_output_checked": self.save_output_checkbox.isChecked(),
            "custom_prompt": self.custom_prompt_edit.toPlainText(), # Get text from prompt area
        }

    def apply_settings_to_ui(self, settings_dict: Dict[str, Any]):
        # Modified to include custom prompt
        self.folder_path_edit.setText(settings_dict.get("folder_path", DEFAULT_SETTINGS["folder_path"]))
        self.preset_combo.setCurrentText(settings_dict.get("preset", DEFAULT_SETTINGS["preset"]))
        self.custom_folders_edit.setText(settings_dict.get("custom_folders", DEFAULT_SETTINGS["custom_folders"]))
        self.custom_extensions_edit.setText(settings_dict.get("custom_extensions", DEFAULT_SETTINGS["custom_extensions"]))
        self.custom_patterns_edit.setText(settings_dict.get("custom_patterns", DEFAULT_SETTINGS["custom_patterns"]))
        self.dynamic_exclude_edit.setText(settings_dict.get("dynamic_patterns", DEFAULT_SETTINGS["dynamic_patterns"]))
        self.max_size_spinbox.setValue(float(settings_dict.get("max_file_size_mb", DEFAULT_SETTINGS["max_file_size_mb"])))
        self.save_output_checkbox.setChecked(bool(settings_dict.get("save_output_checked", DEFAULT_SETTINGS["save_output_checked"])))
        self.custom_prompt_edit.setPlainText(settings_dict.get("custom_prompt", DEFAULT_SETTINGS["custom_prompt"])) # Set prompt text
        self.update_custom_fields_state()

    def load_persistent_settings(self):
        # Unchanged from v1.2
        last_folder = self.settings.value("lastFolderPath", "")
        if last_folder and Path(last_folder).is_dir():
             self.folder_path_edit.setText(last_folder)
        else:
             self.folder_path_edit.setText(DEFAULT_SETTINGS["folder_path"])
        recent_data = self.settings.value("recentEnvironments", [])
        if isinstance(recent_data, list):
            self.recent_environments = [
                item for item in recent_data
                if isinstance(item, dict) and 'name' in item and 'path' in item
            ]
        else:
            self.recent_environments = []

    def load_initial_environment(self):
        # Unchanged from v1.2
        loaded = False
        if self.recent_environments:
            most_recent_path_str = self.recent_environments[0].get('path')
            if most_recent_path_str:
                 most_recent_path = Path(most_recent_path_str)
                 if most_recent_path.exists():
                     if self._load_environment_from_path(most_recent_path):
                         loaded = True
                     else:
                         self.show_warning(f"Failed to load most recent environment '{most_recent_path.name}'. Removing from list.")
                         self._remove_recent_environment(str(most_recent_path))
        if not loaded:
            self.apply_settings_to_ui(DEFAULT_SETTINGS) # Applies default empty prompt too
            self.current_environment_file = None
            self._update_window_title()
            self.save_env_action.setEnabled(False)
            self.show_status("Loaded default settings.")

    def _update_window_title(self):
        # Unchanged from v1.2
        base_title = "Folder Structure to Text v1.3"
        if self.current_environment_file:
            self.setWindowTitle(f"{base_title} - [{self.current_environment_file.name}]")
        else:
            self.setWindowTitle(base_title)

    # --- Environment File Handling (Unchanged from v1.2) ---
    def _load_environment_from_path(self, file_path: Path) -> bool:
        try:
            with file_path.open('r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
            if not isinstance(loaded_settings, dict):
                raise TypeError("Loaded file is not a valid JSON dictionary.")
            self.apply_settings_to_ui(loaded_settings) # Now includes prompt
            self.current_environment_file = file_path
            self.settings.setValue("lastEnvDir", str(file_path.parent))
            self._add_recent_environment(file_path.name, str(file_path))
            self._update_window_title()
            self.save_env_action.setEnabled(True)
            self.show_status(f"Environment loaded: {file_path.name}")
            return True
        except (IOError, json.JSONDecodeError, TypeError, KeyError) as e:
            self.show_error(f"Failed to load environment file '{file_path.name}': {e}")
            # Don't reset UI here, leave potentially partially loaded state or previous state
            self.current_environment_file = None # Ensure file association is broken
            self._update_window_title()
            self.save_env_action.setEnabled(False)
            return False

    def load_environment_dialog(self):
        start_dir = self.settings.value("lastEnvDir", str(Path.home()))
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Load Environment", start_dir, "JSON Files (*.json);;All Files (*)"
        )
        if fileName:
            self._load_environment_from_path(Path(fileName))

    def save_current_environment(self):
        if not self.current_environment_file:
            self.save_environment_as()
            return
        current_settings = self.get_settings_from_ui() # Includes prompt
        try:
            with self.current_environment_file.open('w', encoding='utf-8') as f:
                json.dump(current_settings, f, indent=4)
            self.show_status(f"Environment saved: {self.current_environment_file.name}")
        except (IOError, TypeError) as e:
            self.show_error(f"Failed to save environment file '{self.current_environment_file.name}': {e}")

    def save_environment_as(self):
        current_settings = self.get_settings_from_ui() # Includes prompt
        start_dir = self.settings.value("lastEnvDir", str(Path.home()))
        default_name = f"{Path(self.folder_path_edit.text()).name}_env.json" if self.folder_path_edit.text() else "folder_structure_env.json"
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Environment As...", str(Path(start_dir) / sanitize_filename(default_name)),
            "JSON Files (*.json);;All Files (*)"
        )
        if fileName:
            file_path = Path(fileName)
            try:
                with file_path.open('w', encoding='utf-8') as f:
                    json.dump(current_settings, f, indent=4)
                self.current_environment_file = file_path
                self.settings.setValue("lastEnvDir", str(file_path.parent))
                self._add_recent_environment(file_path.name, str(file_path))
                self._update_window_title()
                self.save_env_action.setEnabled(True)
                self.show_status(f"Environment saved as: {file_path.name}")
            except (IOError, TypeError) as e:
                self.show_error(f"Failed to save environment file: {e}")

    # --- Recent Environment Management (Unchanged from v1.2) ---
    def _add_recent_environment(self, name: str, path_str: str):
        self.recent_environments = [env for env in self.recent_environments if env['path'] != path_str]
        self.recent_environments.insert(0, {'name': name, 'path': path_str})
        self.recent_environments = self.recent_environments[:MAX_RECENT_ENVS]
        self.settings.setValue("recentEnvironments", self.recent_environments)
        self._update_recent_menu()

    def _remove_recent_environment(self, path_str: str):
        initial_len = len(self.recent_environments)
        self.recent_environments = [env for env in self.recent_environments if env['path'] != path_str]
        if len(self.recent_environments) < initial_len:
            self.settings.setValue("recentEnvironments", self.recent_environments)
            self._update_recent_menu()

    def _update_recent_menu(self):
        self.recent_env_menu.clear()
        actions = []
        if self.recent_environments:
            for i, env in enumerate(self.recent_environments):
                action = QAction(f"&{i+1} {env['name']}", self,
                                 triggered=functools.partial(self.load_specific_recent_environment, env['path']))
                action.setStatusTip(f"Load {env['path']}")
                actions.append(action)
        else:
            actions.append(QAction("No Recent Files", self, enabled=False))
        self.recent_env_menu.addActions(actions)

    def load_specific_recent_environment(self, path_str: str):
        file_path = Path(path_str)
        if not file_path.exists():
            self.show_warning(f"Recent environment file not found: {path_str}. Removing from list.")
            self._remove_recent_environment(path_str)
            return
        self._load_environment_from_path(file_path)

    def manage_recent_environments(self):
        dialog = ManageRecentDialog(list(self.recent_environments), self)
        if dialog.exec_() == QDialog.Accepted:
            updated_list = dialog.get_updated_list()
            if updated_list != self.recent_environments:
                self.recent_environments = updated_list
                self.settings.setValue("recentEnvironments", self.recent_environments)
                self._update_recent_menu()
                self.show_status("Recent environments list updated.")

    # --- Core Action (Modified) ---
    def generate_structure(self):
        settings = self.get_settings_from_ui() # Gets prompt text as well
        folder_path = settings["folder_path"]
        if not folder_path:
            self.show_warning("Please select a target folder.")
            return
        self.settings.setValue("lastFolderPath", folder_path)

        self.generate_button.setEnabled(False)
        self.output_textedit.setPlainText("Generating, please wait...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        # Pass the custom prompt text to the processor
        generated_text = self.processor.generate_structure_text(
            folder_path, settings["preset"], settings["custom_folders"],
            settings["custom_extensions"], settings["custom_patterns"],
            settings["dynamic_patterns"], settings["max_file_size_mb"],
            settings["custom_prompt"] # Pass the new argument
        )

        self.output_textedit.setPlainText(generated_text)
        QApplication.restoreOverrideCursor()
        self.generate_button.setEnabled(True)

        if settings["save_output_checked"] and not generated_text.startswith("```error"):
            self.save_output_markdown_to_file(generated_text, Path(folder_path).name)

    def save_output_markdown_to_file(self, text_content, folder_name):
        # Unchanged from v1.2
        save_dir = self.settings.value("lastMarkdownSaveDir", str(Path(self.folder_path_edit.text()).parent))
        suggested_name = f"textraction_{sanitize_filename(folder_name)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Markdown Output", str(Path(save_dir) / suggested_name),
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        )
        if fileName:
            try:
                file_path = Path(fileName)
                file_path.write_text(text_content, encoding='utf-8')
                self.settings.setValue("lastMarkdownSaveDir", str(file_path.parent))
                self.show_status(f"Markdown output saved successfully to: {Path(fileName).name}")
            except Exception as e:
                self.show_error(f"Failed to save Markdown file: {e}")

    def closeEvent(self, event):
        # Unchanged from v1.2
        self.settings.sync()
        event.accept()


# --- Application Entry Point ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setOrganizationName("MyCompany")
    app.setApplicationName("FolderStructureApp")
    ex = FolderStructureApp()
    ex.show()
    sys.exit(app.exec_())