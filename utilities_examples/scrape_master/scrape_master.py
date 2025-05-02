# -*- coding: utf-8 -*-
# Project: ScrapeMaster GUI
# Author: ParisNeo with gemini 2.5
# Description: A PyQt5 GUI application for the ScrapeMaster library, allowing users to scrape web content, view it raw or rendered, save/load results, and manage scraping settings, with persistent theming.

import sys
import json
import os
from urllib.parse import urlparse
import time
import traceback
import functools

try:
    import pipmaster as pm
    pm.ensure_packages(["PyQt5", "markdown", "qt_material"])
except ImportError:
    print("Error: pipmaster not found. Please install it: pip install pipmaster")
    print("Then install required packages: pip install PyQt5 markdown qt_material")
    sys.exit(1)
except Exception as e:
    print(f"Error ensuring packages: {e}")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLineEdit, QPushButton, QTextEdit, QLabel, QAction, QFileDialog,
        QMessageBox, QMenu, QStatusBar, QTabWidget, QDialog, QCheckBox,
        QDialogButtonBox, QGridLayout, QSpinBox
    )
    from PyQt5.QtCore import QSettings, Qt, pyqtSignal
    from PyQt5.QtGui import QIcon
    import qt_material
except ImportError as e:
    print(f"Error: Missing required PyQt5 components, markdown, or qt_material. ({e})")
    print("Please ensure PyQt5, markdown, and qt_material are installed correctly.")
    sys.exit(1)

try:
    import markdown
except ImportError:
    print("Error: 'markdown' library not found. Please install it: 'pip install markdown'")
    sys.exit(1)

# --- Adjust path for ScrapeMaster ---
# Assume scrapemaster folder is one level up from where this script might be
# Or adjust as necessary depending on your project structure
try:
    # If running from within a package structure or development environment
    current_dir = Path(__file__).parent
    scrapemaster_path = current_dir.parent / 'scrapemaster' # Example path
    if not scrapemaster_path.exists():
         # Fallback if not found relative, maybe it's installed?
         pass # Let the direct import try
    else:
         sys.path.insert(0, str(current_dir.parent)) # Add parent to path

    from scrapemaster import ScrapeMaster
    from scrapemaster.core import SUPPORTED_STRATEGIES, DEFAULT_STRATEGY_ORDER
except ImportError as e:
     print(f"Error importing ScrapeMaster library: {e}")
     print("Please ensure the ScrapeMaster library is installed or accessible.")
     print("If running from source, ensure the path is correct.")
     print("If installed, ensure the installation is valid.")
     sys.exit(1)
except NameError: # If Path is not defined yet
    from pathlib import Path
    current_dir = Path(__file__).parent
    scrapemaster_path = current_dir.parent / 'scrapemaster'
    if not scrapemaster_path.exists():
         pass
    else:
         sys.path.insert(0, str(current_dir.parent))
    try:
        from scrapemaster import ScrapeMaster
        from scrapemaster.core import SUPPORTED_STRATEGIES, DEFAULT_STRATEGY_ORDER
    except ImportError as e:
        print(f"Error importing ScrapeMaster library after Path fix: {e}")
        sys.exit(1)


APP_NAME = "ScrapeMaster GUI"
ORG_NAME = "AICodeHelper" # Keep consistent with folder app if desired
MAX_RECENT_FILES = 10
SETTINGS_RECENT_FILES = "recentFiles"
SETTINGS_STRATEGY_PREFIX = "settings/strategyEnabled_"
SETTINGS_HEADLESS = "settings/headlessMode"
SETTINGS_CRAWL_DEPTH = "settings/crawlDepth"
SETTINGS_SELECTED_THEME = "selectedTheme"
SETTINGS_WINDOW_GEOMETRY = "window/geometry"
SETTINGS_WINDOW_STATE = "window/state"
DEFAULT_THEME = "light_cyan_500.xml"


def is_valid_url_for_gui(url_string: str) -> bool:
    if not isinstance(url_string, str): return False
    try:
        result = urlparse(url_string)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except ValueError: return False


class SettingsDialog(QDialog):
    settingsChanged = pyqtSignal()

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Scraper Settings")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        grid_layout = QGridLayout()

        self.strategy_checkboxes = {}
        grid_layout.addWidget(QLabel("<b>Scraping Strategies (in order):</b>"), 0, 0, 1, 2)
        row = 1
        for strategy in DEFAULT_STRATEGY_ORDER:
            if strategy not in SUPPORTED_STRATEGIES: continue
            checkbox = QCheckBox(strategy.capitalize())
            setting_key = f"{SETTINGS_STRATEGY_PREFIX}{strategy}"
            is_enabled = self.settings.value(setting_key, True, type=bool)
            checkbox.setChecked(is_enabled)
            grid_layout.addWidget(checkbox, row, 0)
            self.strategy_checkboxes[strategy] = checkbox
            row += 1

        grid_layout.addWidget(QLabel("<b>Options:</b>"), row, 0, 1, 2)
        row += 1
        self.headless_checkbox = QCheckBox("Run browser headless (no visible window)")
        headless_enabled = self.settings.value(SETTINGS_HEADLESS, True, type=bool)
        self.headless_checkbox.setChecked(headless_enabled)
        grid_layout.addWidget(self.headless_checkbox, row, 0, 1, 2)
        row += 1

        grid_layout.addWidget(QLabel("Crawl Depth:"), row, 0)
        self.crawl_depth_spinbox = QSpinBox()
        self.crawl_depth_spinbox.setRange(0, 10)
        self.crawl_depth_spinbox.setToolTip("0 = Scrape only the entered URL.\n>0 = Follow links up to this depth within the same domain.")
        current_depth = self.settings.value(SETTINGS_CRAWL_DEPTH, 0, type=int)
        self.crawl_depth_spinbox.setValue(current_depth)
        grid_layout.addWidget(self.crawl_depth_spinbox, row, 1)
        row += 1

        layout.addLayout(grid_layout)
        layout.addStretch(1)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.save_settings)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

    def save_settings(self):
        at_least_one_strategy = False
        for strategy, checkbox in self.strategy_checkboxes.items():
            setting_key = f"{SETTINGS_STRATEGY_PREFIX}{strategy}"
            is_enabled = checkbox.isChecked()
            self.settings.setValue(setting_key, is_enabled)
            if is_enabled: at_least_one_strategy = True

        if not at_least_one_strategy:
            QMessageBox.warning(self, "Settings Error", "Please select at least one scraping strategy.")
            return

        self.settings.setValue(SETTINGS_HEADLESS, self.headless_checkbox.isChecked())
        self.settings.setValue(SETTINGS_CRAWL_DEPTH, self.crawl_depth_spinbox.value())
        self.settingsChanged.emit()
        self.accept()


class DocScraperAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.setOrganizationName(ORG_NAME)
        QApplication.setApplicationName(APP_NAME)
        self.settings = QSettings()
        self.current_file_path = None
        self.current_theme = DEFAULT_THEME
        self.theme_actions = []

        self.load_persistent_settings()
        self.initUI()
        self.update_recent_files_menu()
        self.apply_theme(self.current_theme, startup=True)
        self.load_window_state()


    def initUI(self):
        self.setWindowTitle(APP_NAME)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter documentation URL here...")
        self.url_input.returnPressed.connect(self.scrape_url_action)
        scrape_button = QPushButton("Scrape")
        scrape_button.clicked.connect(self.scrape_url_action)
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(scrape_button)
        main_layout.addLayout(url_layout)

        self.tab_widget = QTabWidget()
        self.raw_markdown_output = QTextEdit()
        self.raw_markdown_output.setPlaceholderText("Raw scraped Markdown content will appear here...")
        self.raw_markdown_output.setReadOnly(False)
        self.raw_markdown_output.setAcceptRichText(False)
        self.rendered_output = QTextEdit()
        self.rendered_output.setPlaceholderText("Rendered view (if Markdown is scraped successfully).")
        self.rendered_output.setReadOnly(True)
        self.tab_widget.addTab(self.raw_markdown_output, "Raw Markdown")
        self.tab_widget.addTab(self.rendered_output, "Rendered")
        main_layout.addWidget(self.tab_widget)

        button_layout = QHBoxLayout()
        self.copy_button = QPushButton("Copy Raw Markdown")
        self.copy_button.clicked.connect(self.copy_markdown_action)
        self.copy_button.setEnabled(False)
        button_layout.addWidget(self.copy_button)
        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)

        self.create_menus()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        self.update_title()

    def show_status(self, message: str, timeout: int = 3000):
        self.statusBar.showMessage(message, timeout)
        print(f"INFO: {message}")

    def show_warning(self, message: str):
        self.statusBar.showMessage(f"Warning: {message}", 5000)
        QMessageBox.warning(self, "Warning", message)
        print(f"WARNING: {message}")

    def show_error(self, message: str):
         self.statusBar.showMessage(f"Error: {message}", 8000)
         QMessageBox.critical(self, "Error", message)
         print(f"ERROR: {message}")

    def create_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        load_action = QAction('&Load JSON...', self); load_action.setShortcut('Ctrl+O'); load_action.setStatusTip('Load scraped data from a JSON file'); load_action.triggered.connect(self.load_file_dialog); file_menu.addAction(load_action)
        save_as_action = QAction('Save As &JSON...', self); save_as_action.setShortcut('Ctrl+S'); save_as_action.setStatusTip('Save URL and Markdown to a JSON file'); save_as_action.triggered.connect(self.save_file_dialog); file_menu.addAction(save_as_action)
        export_md_action = QAction('&Export as Markdown (.md)...', self)
        export_md_action.setStatusTip('Export the raw Markdown content to a .md file')
        export_md_action.triggered.connect(self.export_markdown_dialog)
        file_menu.addAction(export_md_action)
        file_menu.addSeparator()
        self.recent_files_menu = QMenu('&Recent Files', self); file_menu.addMenu(self.recent_files_menu)
        file_menu.addSeparator()
        exit_action = QAction('&Exit', self); exit_action.setShortcut('Ctrl+Q'); exit_action.setStatusTip('Exit application'); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)

        options_menu = menubar.addMenu('&Options')
        settings_action = QAction('&Settings...', self)
        settings_action.setStatusTip('Configure scraping strategies and options')
        settings_action.triggered.connect(self.open_settings_dialog)
        options_menu.addAction(settings_action)

        view_menu = menubar.addMenu('&View')
        theme_menu = view_menu.addMenu("&Theme")
        available_themes = [
            'dark_blue.xml', 'dark_cyan.xml', 'dark_teal.xml', 'dark_amber.xml', 'dark_red.xml',
            'light_blue.xml', 'light_cyan.xml', 'light_cyan_500.xml', 'light_teal.xml', 'light_amber.xml', 'light_red.xml'
        ]
        self.theme_actions.clear()
        for theme_file in sorted(available_themes):
            theme_name = theme_file.replace('.xml', '').replace('_', ' ').title()
            action = QAction(theme_name, self, checkable=True,
                             triggered=functools.partial(self.change_theme, theme_file))
            self.theme_actions.append(action)
            theme_menu.addAction(action)

    def load_persistent_settings(self):
        self.current_theme = self.settings.value(SETTINGS_SELECTED_THEME, DEFAULT_THEME)
        # Recent files are loaded dynamically by update_recent_files_menu
        # Other settings (strategy, headless, depth) are loaded by SettingsDialog or when used

    def load_window_state(self):
        geometry = self.settings.value(SETTINGS_WINDOW_GEOMETRY); state = self.settings.value(SETTINGS_WINDOW_STATE)
        if geometry: self.restoreGeometry(geometry)
        if state: self.restoreState(state)
        else: self.setGeometry(100, 100, 800, 600)

    def save_window_state(self):
        self.settings.setValue(SETTINGS_WINDOW_GEOMETRY, self.saveGeometry()); self.settings.setValue(SETTINGS_WINDOW_STATE, self.saveState())

    def update_title(self):
        title = APP_NAME;
        if self.current_file_path: title += f" - {os.path.basename(self.current_file_path)}"
        self.setWindowTitle(title)

    def apply_theme(self, theme_file: str, startup: bool = False):
        try:
            if not theme_file:
                theme_file = DEFAULT_THEME
                print(f"Warning: Invalid theme '{theme_file}' provided, falling back to default {DEFAULT_THEME}.")

            if not startup:
                self.show_status(f"Applying theme: {theme_file}...")

            qt_material.apply_stylesheet(
                QApplication.instance(),
                theme=theme_file,
                invert_secondary= ('dark' in theme_file)
            )

            self.current_theme = theme_file
            self.settings.setValue(SETTINGS_SELECTED_THEME, theme_file)

            for action in self.theme_actions:
                action.setChecked(action.text().lower().replace(' ', '_') + ".xml" == theme_file)

            if not startup:
                self.show_status(f"Theme '{theme_file}' applied.")

        except Exception as e:
            self.show_error(f"Failed to apply theme '{theme_file}': {e}")
            if theme_file != DEFAULT_THEME:
                self.apply_theme(DEFAULT_THEME) # Fallback to default

    def change_theme(self, theme_file: str):
        self.apply_theme(theme_file)

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self)
        # Connect signal if needed, but settings are saved within the dialog
        # dialog.settingsChanged.connect(self.handle_settings_change) # Example
        dialog.exec_()

    def scrape_url_action(self):
        url = self.url_input.text().strip()
        if not url: self.show_warning("Please enter a URL."); return

        if not is_valid_url_for_gui(url):
            if not url.startswith(('http://', 'https://')):
                url_https = f"https://{url}"; url_http = f"http://{url}"
                if is_valid_url_for_gui(url_https): url = url_https; self.url_input.setText(url); print(f"Assuming HTTPS: {url}")
                elif is_valid_url_for_gui(url_http): url = url_http; self.url_input.setText(url); print(f"Assuming HTTP: {url}")
                else: self.show_warning(f"Invalid URL: {self.url_input.text()}"); return
            else: self.show_warning(f"Invalid URL: {self.url_input.text()}"); return

        active_strategies = [s for s in DEFAULT_STRATEGY_ORDER if self.settings.value(f"{SETTINGS_STRATEGY_PREFIX}{s}", True, type=bool) and s in SUPPORTED_STRATEGIES]
        if not active_strategies: self.show_error("No scraping strategies enabled!"); return
        headless_mode = self.settings.value(SETTINGS_HEADLESS, True, type=bool)
        crawl_depth = self.settings.value(SETTINGS_CRAWL_DEPTH, 0, type=int)

        crawl_msg = f", Crawl Depth: {crawl_depth}" if crawl_depth > 0 else ""
        self.show_status(f"Scraping {url} (Strategies: {active_strategies}, Headless: {headless_mode}{crawl_msg})...", 60000) # Longer timeout for status
        self.raw_markdown_output.setPlaceholderText(f"Scraping {url}{crawl_msg}...\nPlease wait...")
        self.rendered_output.setPlaceholderText("Waiting for content...")
        self.raw_markdown_output.clear(); self.rendered_output.clear()
        self.copy_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        markdown_content = None
        html_content = None
        final_status_message = "Scraping finished."
        failed_urls_during_crawl = []

        try:
            scraper = ScrapeMaster(url, strategy=active_strategies, headless=headless_mode)
            results = scraper.scrape_all(
                max_depth=crawl_depth,
                convert_to_markdown=True
            )

            if results:
                markdown_content = results.get('markdown')
                failed_urls_during_crawl = results.get('failed_urls', [])

                if markdown_content:
                    self.raw_markdown_output.setPlainText(markdown_content)
                    self.copy_button.setEnabled(True)
                    crawl_info = f", {len(results.get('visited_urls',[]))} pages scraped" if crawl_depth > 0 else ""
                    final_status_message = f"Success! (Strategy: {scraper.last_strategy_used or 'N/A'}{crawl_info})"
                    self.raw_markdown_output.setPlaceholderText("Raw scraped Markdown content.")
                    try:
                        html_content = markdown.markdown(markdown_content, extensions=['fenced_code', 'tables', 'extra'])
                        self.rendered_output.setHtml(html_content)
                        self.rendered_output.setPlaceholderText("Rendered Markdown content.")
                    except Exception as e_render:
                        self.rendered_output.setPlainText(f"[Render Error: {e_render}]")
                        self.rendered_output.setPlaceholderText("Failed to render Markdown.")

                else:
                    fallback_text = "\n---\n".join(results.get('texts', []))
                    message = "[INFO] No primary Markdown content generated."
                    if fallback_text:
                         message += " Displaying all text fragments."
                         self.raw_markdown_output.setPlainText(fallback_text)
                         self.copy_button.setEnabled(True)
                    else:
                         message += " No text fragments found either."
                         self.raw_markdown_output.setPlainText(message)
                         self.copy_button.setEnabled(False)
                    final_status_message = message
                    self.raw_markdown_output.setPlaceholderText(message)
                    self.rendered_output.setPlaceholderText("No Markdown content to render.")

            else:
                error_message = scraper.get_last_error() or "Unknown scraping error."
                self.raw_markdown_output.setPlainText(f"Error:\n{error_message}")
                final_status_message = f"Scraping failed. Check Raw tab."
                self.raw_markdown_output.setPlaceholderText("Scraping failed. See error message above.")
                self.rendered_output.setPlaceholderText("Scraping failed.")
                if scraper.last_error and ("All scraping strategies failed" in scraper.last_error or "Could not initialize" in scraper.last_error):
                     self.show_error(error_message) # Show critical errors as dialog

        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = f"Critical Application Error:\n{e}\n\nDetails:\n{error_details}"
            self.raw_markdown_output.setPlainText(error_msg); self.rendered_output.setPlainText(f"[App Error: {e}]")
            final_status_message = "A critical error occurred."
            self.raw_markdown_output.setPlaceholderText("Critical error."); self.rendered_output.setPlaceholderText("Critical error.")
            self.show_error(f"An unexpected error occurred:\n{e}")

        finally:
             QApplication.restoreOverrideCursor()

        # Report failed URLs after cursor is restored
        if failed_urls_during_crawl:
            failed_list_str = "\n - ".join(failed_urls_during_crawl)
            print(f"Warning: Failed to scrape the following URLs during crawl:\n - {failed_list_str}")
            self.show_warning(f"Failed to scrape {len(failed_urls_during_crawl)} URLs during crawl. Check console log for details.")

        self.show_status(final_status_message, 5000)
        self.current_file_path = None
        self.update_title()

    def copy_markdown_action(self):
        clipboard = QApplication.clipboard(); raw_markdown_text = self.raw_markdown_output.toPlainText()
        clipboard.setText(raw_markdown_text); self.show_status("Raw Markdown copied to clipboard!", 2000)

    def save_file_dialog(self):
        url = self.url_input.text(); raw_markdown = self.raw_markdown_output.toPlainText()
        if not url or not raw_markdown or raw_markdown.startswith("Error:") or raw_markdown.startswith("[INFO]") or not raw_markdown.strip():
            self.show_warning("Need valid URL and content to save."); return
        try:
            parsed_url = urlparse(url); safe_domain = parsed_url.netloc.replace('.', '_'); safe_path = parsed_url.path.replace('/', '_').strip('_');
            if not safe_path: safe_path = 'index'; suggested_name = f"{safe_domain}_{safe_path}.json"; suggested_name = "".join(c for c in suggested_name if c.isalnum() or c in ('_', '-')).rstrip()[:100] + ".json"
        except Exception: suggested_name = "scraped_data.json"
        options = QFileDialog.Options(); file_path, _ = QFileDialog.getSaveFileName(self, "Save Scraped Data (JSON)", suggested_name, "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            if not file_path.lower().endswith('.json'): file_path += '.json'
            self.save_file(file_path)

    def save_file(self, file_path):
        data = { "url": self.url_input.text(), "markdown": self.raw_markdown_output.toPlainText() }
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
            self.current_file_path = file_path; self.add_recent_file(file_path); self.update_title(); self.show_status(f"Data saved to {os.path.basename(file_path)}", 3000)
        except Exception as e: self.show_error(f"Could not save JSON file:\n{e}"); self.show_status("Save failed.", 3000)

    def export_markdown_dialog(self):
        raw_markdown = self.raw_markdown_output.toPlainText()
        url = self.url_input.text()

        if not raw_markdown or raw_markdown.startswith("Error:") or raw_markdown.startswith("[INFO]") or not raw_markdown.strip():
            self.show_warning("No valid Markdown content available to export.")
            return

        suggested_name = "scraped_content.md"
        if url and is_valid_url_for_gui(url):
            try:
                parsed_url = urlparse(url); safe_domain = parsed_url.netloc.replace('.', '_'); safe_path = parsed_url.path.replace('/', '_').strip('_');
                if not safe_path: safe_path = 'index'; base_name = f"{safe_domain}_{safe_path}"
                suggested_name = "".join(c for c in base_name if c.isalnum() or c in ('_', '-')).rstrip()[:100] + ".md"
            except Exception: pass
        else:
             timestamp = time.strftime("%Y%m%d_%H%M%S")
             suggested_name = f"scraped_content_{timestamp}.md"

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Raw Markdown",
            suggested_name,
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
            options=options
        )

        if file_path:
            if not file_path.lower().endswith(('.md', '.txt')):
                file_path += '.md'

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(raw_markdown)
                self.show_status(f"Markdown exported to {os.path.basename(file_path)}", 3000)
            except IOError as e:
                self.show_error(f"Could not export Markdown file:\n{e}")
                self.show_status("Export failed.", 3000)
            except Exception as e:
                 self.show_error(f"An unexpected error occurred during export:\n{e}")
                 self.show_status("Export failed.", 3000)

    def load_file_dialog(self):
        options = QFileDialog.Options(); file_path, _ = QFileDialog.getOpenFileName(self, "Load Scraped Data (JSON)", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path: self.load_file(file_path)

    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if "url" not in data or "markdown" not in data: raise ValueError("JSON file missing 'url' or 'markdown' key.")
            loaded_url = data.get("url", ""); loaded_markdown = data.get("markdown", "")
            self.url_input.setText(loaded_url); self.raw_markdown_output.setPlainText(loaded_markdown)
            is_valid_content = bool(loaded_markdown and not loaded_markdown.startswith("Error:") and not loaded_markdown.startswith("[INFO]"))
            if is_valid_content:
                try:
                    html_content = markdown.markdown(loaded_markdown, extensions=['fenced_code', 'tables', 'extra'])
                    self.rendered_output.setHtml(html_content); self.rendered_output.setPlaceholderText("Rendered.")
                except Exception as e_render: self.rendered_output.setPlainText(f"[Render Error: {e_render}]"); self.rendered_output.setPlaceholderText("Failed render.")
                self.copy_button.setEnabled(True)
            else:
                self.rendered_output.clear(); self.rendered_output.setPlaceholderText("Loaded file has error/no content."); self.copy_button.setEnabled(False)
            self.current_file_path = file_path; self.add_recent_file(file_path); self.update_title(); self.show_status(f"Loaded {os.path.basename(file_path)}", 3000)
        except FileNotFoundError: self.show_error(f"File not found:\n{file_path}"); self.remove_recent_file(file_path); self.show_status("Load failed: File not found.", 3000)
        except json.JSONDecodeError: self.show_error(f"Could not decode JSON file:\n{file_path}"); self.show_status("Load failed: Invalid JSON.", 3000)
        except ValueError as e: self.show_error(f"Invalid file format: {e}\n{file_path}"); self.show_status("Load failed: Invalid format.", 3000)
        except Exception as e: self.show_error(f"An unexpected error occurred during load:\n{e}"); self.show_status("Load failed.", 3000)

    def get_recent_files(self) -> list[str]: return self.settings.value(SETTINGS_RECENT_FILES, [], type=list)
    def set_recent_files(self, files: list[str]): self.settings.setValue(SETTINGS_RECENT_FILES, files)
    def add_recent_file(self, file_path: str):
        if not file_path: return; recent_files = self.get_recent_files();
        try: recent_files.remove(file_path)
        except ValueError: pass
        recent_files.insert(0, file_path); del recent_files[MAX_RECENT_FILES:]; self.set_recent_files(recent_files); self.update_recent_files_menu()
    def remove_recent_file(self, file_path: str):
        if not file_path: return; recent_files = self.get_recent_files();
        try: recent_files.remove(file_path); self.set_recent_files(recent_files); self.update_recent_files_menu()
        except ValueError: pass
    def update_recent_files_menu(self):
        self.recent_files_menu.clear(); recent_files = self.get_recent_files(); actions = []
        for i, file_path in enumerate(recent_files):
            if not file_path: continue
            action = QAction(f"&{i+1} {os.path.basename(file_path)}", self); action.setData(file_path); action.triggered.connect(self.open_recent_file); action.setToolTip(file_path) ; actions.append(action)
        if actions: self.recent_files_menu.addActions(actions); self.recent_files_menu.setEnabled(True)
        else: no_recent_action = QAction("(No Recent Files)", self); no_recent_action.setEnabled(False); self.recent_files_menu.addAction(no_recent_action); self.recent_files_menu.setEnabled(False)
    def open_recent_file(self):
        action = self.sender();
        if action and action.data():
            file_path = action.data()
            if os.path.exists(file_path): self.load_file(file_path)
            else: self.show_warning(f"File not found: {os.path.basename(file_path)}"); self.remove_recent_file(file_path)

    def closeEvent(self, event):
        self.save_window_state()
        self.settings.setValue(SETTINGS_SELECTED_THEME, self.current_theme)
        self.settings.sync()
        self.show_status("Settings saved. Exiting...")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        mainWin = DocScraperAppGUI()
        mainWin.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("\n--- Unhandled Exception ---")
        print(f"Error: {e}")
        print(traceback.format_exc())
        try:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setWindowTitle("Application Crash")
            msgBox.setText(f"An unexpected error occurred:\n\n{e}\n\nPlease see console output for details.")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec_()
        except:
            pass
        sys.exit(1)