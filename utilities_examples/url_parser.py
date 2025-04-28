# --- Add/Ensure these imports are at the top ---
import sys
import json
import os
from urllib.parse import urlparse
import time
import re # For checking blocker pages

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel, QAction, QFileDialog,
    QMessageBox, QMenu, QStatusBar
)
from PyQt5.QtCore import QSettings, Qt, QUrl
from PyQt5.QtGui import QIcon

import pipmaster as pm
pm.ensure_packages({
    "selenium":"",
    "webdriver-manager":"",
    "undetected-chromedriver":"",
    "lxml":"",
    "bs4":"",
    "markdownify":""
})
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Attempt to import undetected_chromedriver, set a flag if successful
try:
    import undetected_chromedriver as uc
    UNDETECTED_AVAILABLE = True
    print(">>> undetected-chromedriver library found.") # More visible confirmation
except ImportError:
    UNDETECTED_AVAILABLE = False
    print(">>> undetected-chromedriver library not found. Advanced stealth scraping disabled.")


# --- Constants ---
APP_NAME = "Doc Scraper"
ORG_NAME = "MyCompany"
MAX_RECENT_FILES = 10
SETTINGS_RECENT_FILES = "recentFiles"

# Enhanced Headers - Mimic a common browser setup
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"'
}

# Selectors for finding the main content area (add more as needed)
CONTENT_SELECTORS = [
    '.main-content', 'main', '.docs-body', 'article', '#main-content',
    '.content', '#content', '[role="main"]', '.post-content', '.entry-content'
]

# Selectors for removing noisy elements within the main content
NOISY_SELECTORS = [
    'nav', 'footer', 'aside', '.sidebar', '#sidebar', '.toc', '.table-of-contents',
    '.docs-header', '.docs-sidebar', '.edit-page-link', '.feedback-widget',
    'script', 'style', 'header', 'iframe', 'form', '.ads', '.advertisement',
    'button', 'input', '[role="search"]', '[role="navigation"]', '[role="complementary"]',
    '.metadata', '.post-meta', '.breadcrumbs', '.breadcrumb', '.page-navigation',
    '.related-posts', '.comments', '#comments'
]

# Phrases indicating a JavaScript/Cookie blocker page (case-insensitive)
BLOCKER_PHRASES = [
    "enable javascript", "enable cookies", "checking your browser",
    "redirecting", "cloudflare", "verify you are human", "js challenge",
    "please wait", "needs javascript to function", "one moment please"
]
BLOCKER_REGEX = re.compile('|'.join(BLOCKER_PHRASES), re.IGNORECASE)

# --- Global flag for debugging non-headless mode ---
DEBUG_NON_HEADLESS = False # Set to True ONLY if Strategy 3 also fails, to see the browser window

# --- Helper Functions (is_valid_url, _check_for_blocker, _parse_and_markdownify) ---
# (Keep these functions exactly as they were in the previous 'multi-strategy' code block)
def is_valid_url(url_string: str) -> bool:
    try:
        result = urlparse(url_string)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except ValueError:
        return False

def _check_for_blocker(html_content: str) -> bool:
    if not html_content: return False
    text_sample = html_content[:4096].lower() # Check beginning of content
    is_blocker = BLOCKER_REGEX.search(text_sample)
    if is_blocker:
        print("Detected potential blocker phrase.")
    return bool(is_blocker)

def _parse_and_markdownify(html_content: str) -> tuple[str | None, str | None]:
    if not html_content: return None, "Error: Received empty HTML content."
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        main_content_element = None
        used_selector = None
        for selector in CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element:
                print(f"Found content container using selector: '{selector}'")
                main_content_element = element
                used_selector = selector
                break
        if not main_content_element:
            print("Could not find specific main content tag using selectors, falling back to <body>")
            main_content_element = soup.body
            used_selector = 'body (fallback)'
            if not main_content_element: return None, "Error: Could not find <body> tag in the HTML."

        count = 0
        for noisy_selector in NOISY_SELECTORS:
            try:
                for element in main_content_element.select(noisy_selector):
                    element.decompose(); count += 1
            except Exception as e_decompose: print(f"Warning: Error decomposing '{noisy_selector}': {e_decompose}")
        print(f"Removed {count} noisy elements from '{used_selector}' container.")

        cleaned_text_sample = main_content_element.get_text(strip=True)[:500].lower()
        if BLOCKER_REGEX.search(cleaned_text_sample):
             print("Warning: Content container still seems to hold blocker message after cleaning.")
             # Don't return error yet, let the calling function decide based on context
             # Return the blocker content for now, maybe user wants it? No, better to signal failure.
             # Let's signal specific blocker failure from here.
             return None, f"Blocker identified within '{used_selector}' after cleaning."

        print("Converting to Markdown...")
        markdown_text = md(str(main_content_element), heading_style="ATX", escape_underscores=False)
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()
        print("Markdown extraction complete.")
        return markdown_text, None
    except Exception as e:
        import traceback
        print(f"Error during parsing/markdownify:\n{traceback.format_exc()}")
        return None, f"Error processing HTML: {e}"

# --- Scraping Strategy Functions (_try_requests, _run_selenium_attempt, _try_selenium) ---
# (Keep _try_requests exactly as before)
def _try_requests(url: str) -> tuple[str | None, str | None]:
    print("\n--- Strategy 1: Trying simple HTTP request (requests) ---")
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        response = session.get(url, timeout=20)
        print(f"Response Status Code: {response.status_code}")
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
             print(f"Warning: Received non-HTML content type: {content_type}. Aborting requests strategy.")
             return None, None
        response.raise_for_status()
        html_content = response.content
        if _check_for_blocker(html_content.decode(response.encoding or 'utf-8', errors='ignore')):
            print("Requests strategy failed: Blocker page detected.")
            return None, None # Blocker - try next strategy
        print("Content doesn't immediately look like a blocker. Parsing...")
        # Parse and check again inside _parse_and_markdownify
        markdown, error = _parse_and_markdownify(html_content)
        if error and "Blocker identified" in error:
             print("Requests strategy failed: Blocker identified during parsing.")
             return None, None # Blocker confirmed - try next strategy
        return markdown, error # Return result or parsing error
    except requests.exceptions.RequestException as e:
        print(f"Requests strategy failed: {e}")
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
            return None, None # 403 Blocker - try next strategy
        else: return None, f"Error fetching URL (requests): {e}" # Other request error
    except Exception as e:
        print(f"Unexpected error during requests strategy: {e}")
        return None, f"Unexpected error (requests): {e}"

# (Keep _run_selenium_attempt exactly as before)
def _run_selenium_attempt(driver, url: str) -> tuple[str | None, str | None]:
    try:
        driver.set_page_load_timeout(45)
        driver.implicitly_wait(2)
        print("Navigating to URL...")
        try: driver.get(url)
        except TimeoutException: return None, f"Error: Page load timed out after 45 seconds."
        except WebDriverException as e: return None, f"Error: WebDriver error during get: {e}"

        wait_time = 25
        print(f"Waiting up to {wait_time}s for potential content containers {CONTENT_SELECTORS}...")
        wait = WebDriverWait(driver, wait_time)
        found_container_selector = None
        try:
            container_found = False
            for selector in CONTENT_SELECTORS:
                # print(f"  Checking for: {selector}") # Can be noisy
                try:
                    # Use a short wait for each check within the loop
                    WebDriverWait(driver, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    print(f"  Container '{selector}' found quickly.")
                    found_container_selector = selector
                    container_found = True
                    break # Found one, proceed
                except TimeoutException: continue # Try next selector

            # If loop finished without finding one, do a longer wait for the first/primary selector
            if not container_found:
                 primary_selector = CONTENT_SELECTORS[0]
                 print(f"  No container found quickly, doing longer wait for primary selector '{primary_selector}'")
                 wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, primary_selector)))
                 found_container_selector = primary_selector # Assume the primary one appeared
                 print(f"  Primary container '{primary_selector}' appeared after longer wait.")

            print(f"Content container ({found_container_selector or 'N/A'}) likely present.")
            time.sleep(1) # Optional short pause

        except TimeoutException:
            print(f"Timed out waiting for any content container.")
            html_content_on_timeout = driver.page_source
            if _check_for_blocker(html_content_on_timeout): return None, None # Blocker failure
            else:
                print("Timeout waiting for specific container, but page might not be a blocker. Attempting parse anyway...")
                return _parse_and_markdownify(html_content_on_timeout) # Try parsing what we got
        except WebDriverException as e:
             print(f"WebDriver error during explicit wait: {e}")
             return None, f"Error: WebDriver error while waiting for content: {e}"

        print("Retrieving page source...")
        html_content = driver.page_source
        if not html_content: return None, "Error: Selenium retrieved empty page source after waiting."

        # Check for blocker *after* wait and retrieving source
        if _check_for_blocker(html_content):
            print("Selenium strategy failed: Blocker page detected even after wait.")
            return None, None # Blocker failure

        print("Parsing content retrieved by Selenium...")
        # Parse and check again inside _parse_and_markdownify
        markdown, error = _parse_and_markdownify(html_content)
        if error and "Blocker identified" in error:
             print("Selenium strategy failed: Blocker identified during parsing.")
             return None, None # Blocker confirmed
        return markdown, error # Return result or parsing error
    finally:
        pass
    # Keep finally block outside _run_selenium_attempt if called by _try_selenium
    # Let _try_selenium handle driver.quit()

# (Use the _try_selenium function from the previous thought block - with DEBUG_NON_HEADLESS)
def _try_selenium(url: str, use_undetected: bool = False) -> tuple[str | None, str | None]:
    """Attempts to fetch and process URL using Selenium (standard or undetected)."""
    driver_type = "undetected-chromedriver" if use_undetected else "standard Selenium"
    print(f"\n--- Strategy {2 if not use_undetected else 3}: Trying {driver_type} ---")

    if use_undetected and not UNDETECTED_AVAILABLE:
        print("Skipping undetected-chromedriver: library not installed.")
        return None, "Error: Undetected-chromedriver library not available."

    options = webdriver.ChromeOptions()
    if not DEBUG_NON_HEADLESS:
        options.add_argument("--headless=new")
    else:
         print("*** RUNNING IN NON-HEADLESS (VISIBLE BROWSER) MODE FOR DEBUGGING ***")

    options.add_argument("--disable-gpu"); options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage"); options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    options.add_argument("window-size=1920,1080"); options.add_argument("--disable-blink-features=AutomationControlled")

    if not use_undetected:
        options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        prefs = {"profile.default_content_setting_values.cookies": 1, "profile.default_content_setting_values.javascript": 1}
        options.add_experimental_option("prefs", prefs)

    driver = None
    driver_path = None # Variable to store the driver path

    try:
        # --- CHANGE HERE: Get driver path first for both cases ---
        try:
            print("Getting ChromeDriver path via webdriver-manager...")
            driver_path = ChromeDriverManager().install()
            print(f"Using ChromeDriver at: {driver_path}")
        except Exception as e_wdm:
             # Handle case where webdriver-manager itself fails
             print(f"Error getting ChromeDriver path via webdriver-manager: {e_wdm}")
             # If we can't even get the path, both Selenium strategies will likely fail.
             return None, f"Error: Could not find/install ChromeDriver using webdriver-manager: {e_wdm}"
        # --- END CHANGE ---


        if use_undetected:
            print(f"Initializing {driver_type}...")
            try:
                # --- CHANGE HERE: Add use_subprocess=True ---
                driver = uc.Chrome(
                    driver_executable_path=driver_path,
                    options=options,
                    use_subprocess=True, # <--- Add this argument
                    version_main=119 # Keep or remove based on testing
                )
                # --- END CHANGE ---
                print(f"{driver_type} initialized.")
            except Exception as e_uc_init:
                # Handle uc initialization errors specifically if needed
                print(f"Error setting up {driver_type}: {e_uc_init}")
                return None, f"Error: Could not initialize {driver_type}. {e_uc_init}"
        else:
            print(f"Initializing {driver_type} WebDriver...")
            try:
                 # --- CHANGE HERE: Use the pre-fetched driver_path ---
                 service = ChromeService(executable_path=driver_path)
                 driver = webdriver.Chrome(service=service, options=options)
                 # --- END CHANGE ---
                 print(f"{driver_type} WebDriver initialized.")
            except Exception as e_std_sel: # Catch specific init error
                 print(f"Error setting up {driver_type} WebDriver: {e_std_sel}")
                 return None, f"Error: Could not initialize {driver_type} WebDriver using path {driver_path}. {e_std_sel}"


        markdown, error = _run_selenium_attempt(driver, url)
        if markdown is None and error is None:
             print(f"{driver_type} strategy completed but detected blocker page.")
             return None, None
        return markdown, error

    except WebDriverException as e:
        err_msg = f"Error interacting with browser ({driver_type}): {e}"
        if use_undetected and "cannot connect to chrome" in str(e):
             err_msg += "\nHint: Check firewall/antivirus, Chrome/ChromeDriver compatibility, or try clearing uc cache."
        elif use_undetected and "cannot parse capability" in str(e):
             err_msg += "\nHint: This might be due to incompatible Chrome options or browser/driver version mismatch."
        print(f"A WebDriver error occurred during {driver_type} initialization/run: {e}")
        return None, err_msg
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during {driver_type} processing:\n{traceback.format_exc()}")
        return None, f"Error processing content ({driver_type}): {e}"
    finally:
        if driver:
            driver.quit()
            print(f"{driver_type} WebDriver closed.")

# --- Main Fetching Function (Use the version from the previous thought block - with improved logging) ---
def fetch_and_extract_markdown(url: str) -> str:
    """
    Fetches URL using multiple strategies, extracts main content, and converts to Markdown.
    """
    start_time = time.time()
    print(f"\n=== Starting multi-strategy scrape for: {url} ===")
    last_error_or_status = "No strategies completed successfully." # Default message

    # Strategy 1: Requests
    markdown, error = _try_requests(url)
    if markdown is not None:
        print(f"=== Success (requests) in {time.time() - start_time:.2f} seconds ===")
        return markdown
    elif error is not None: # Definite error from requests
        print(f"--- Requests failed definitively: {error} ---")
        last_error_or_status = error
        # Decide if we should stop or continue
        if "Error fetching URL" in error and "403" not in error:
             pass # Option to return error here
    else: # (None, None) returned -> Blocker or non-HTML
         print("--- Requests strategy yielded no content or detected blocker ---")
         last_error_or_status = "Requests: Blocked or non-HTML content"

    # Strategy 2: Standard Selenium
    print("\n>>> Proceeding to Strategy 2: Standard Selenium <<<") # Explicit log
    markdown, error = _try_selenium(url, use_undetected=False)
    if markdown is not None:
        print(f"=== Success (standard Selenium) in {time.time() - start_time:.2f} seconds ===")
        return markdown
    elif error is not None: # Definite error from standard selenium
        print(f"--- Standard Selenium failed definitively: {error} ---")
        last_error_or_status = error
        if "Could not initialize" in error:
             print("!!! Aborting further strategies due to WebDriver initialization failure. !!!")
             return last_error_or_status # Return critical init error
    else: # (None, None) returned -> Blocker detected by standard selenium
        print("--- Standard Selenium strategy completed but detected blocker page ---")
        last_error_or_status = "Standard Selenium: Blocked"


    # Strategy 3: Undetected Chromedriver
    if UNDETECTED_AVAILABLE:
        print("\n>>> Proceeding to Strategy 3: Undetected-Chromedriver <<<") # Explicit log
        markdown, error = _try_selenium(url, use_undetected=True)
        if markdown is not None:
            print(f"=== Success (undetected-chromedriver) in {time.time() - start_time:.2f} seconds ===")
            return markdown
        elif error is not None: # Definite error from undetected
             print(f"--- Undetected-chromedriver failed definitively: {error} ---")
             last_error_or_status = error
        else: # (None, None) returned -> Blocker detected by undetected
            print("--- Undetected-chromedriver strategy completed but detected blocker page ---")
            last_error_or_status = "Undetected-chromedriver: Blocked"
    else:
        print("--- Skipping Undetected-chromedriver: Library not available ---")
        # Update last status only if it was previously non-committal
        if last_error_or_status.endswith("Blocked"):
            last_error_or_status += " (Undetected-chromedriver unavailable)"
        elif last_error_or_status == "No strategies completed successfully.":
             last_error_or_status = "Undetected-chromedriver library not available."


    # If all strategies failed
    final_error_message = "Error: All scraping strategies failed. The site may be heavily protected, require login, or content structure is unrecognized."
    print(f"=== Failure (All Strategies) in {time.time() - start_time:.2f} seconds ===")
    final_error_message += f"\nLast status: {last_error_or_status}" # Append the last known status/error

    return final_error_message


# --- Main Application Class (DocScraperApp) ---
# (Keep the DocScraperApp class exactly as it was in the previous 'multi-strategy' code block)
class DocScraperApp(QMainWindow):
    # ... (initUI, create_menus, scrape_url, etc. - NO CHANGES NEEDED HERE) ...
    def __init__(self):
        super().__init__()
        # Check confirmation message for undetected-chromedriver during startup
        print("-" * 30)
        if UNDETECTED_AVAILABLE: print("Undetected-chromedriver IS available.")
        else: print("Undetected-chromedriver IS NOT available. Strategy 3 will be skipped.")
        print("-" * 30)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.current_file_path = None
        self.initUI()
        self.update_recent_files_menu()

    def initUI(self):
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter documentation URL here...")
        self.url_input.returnPressed.connect(self.scrape_url)
        scrape_button = QPushButton("Scrape")
        scrape_button.clicked.connect(self.scrape_url)
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(scrape_button)
        main_layout.addLayout(url_layout)
        self.markdown_output = QTextEdit()
        self.markdown_output.setPlaceholderText("Scraped Markdown content will appear here...")
        self.markdown_output.setReadOnly(False)
        main_layout.addWidget(self.markdown_output)
        button_layout = QHBoxLayout()
        self.copy_button = QPushButton("Copy Markdown")
        self.copy_button.clicked.connect(self.copy_markdown)
        self.copy_button.setEnabled(False)
        button_layout.addWidget(self.copy_button)
        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)
        self.create_menus()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        self.update_title()

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        load_action = QAction('&Load...', self); load_action.setShortcut('Ctrl+O'); load_action.setStatusTip('Load scraped data from a JSON file'); load_action.triggered.connect(self.load_file_dialog); file_menu.addAction(load_action)
        save_as_action = QAction('&Save As...', self); save_as_action.setShortcut('Ctrl+S'); save_as_action.setStatusTip('Save URL and Markdown to a JSON file'); save_as_action.triggered.connect(self.save_file_dialog); file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        self.recent_files_menu = QMenu('&Recent Files', self); file_menu.addMenu(self.recent_files_menu)
        file_menu.addSeparator()
        exit_action = QAction('&Exit', self); exit_action.setShortcut('Ctrl+Q'); exit_action.setStatusTip('Exit application'); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)

    def update_title(self):
        title = APP_NAME
        if self.current_file_path: title += f" - {os.path.basename(self.current_file_path)}"
        self.setWindowTitle(title)

    def scrape_url(self):
        url = self.url_input.text().strip()
        if not url: QMessageBox.warning(self, "Input Error", "Please enter a URL."); return
        if not is_valid_url(url):
            if not url.startswith(('http://', 'https://')):
                url_https = f"https://{url}"
                url_http = f"http://{url}"
                if is_valid_url(url_https): url = url_https; self.url_input.setText(url)
                elif is_valid_url(url_http): url = url_http; self.url_input.setText(url)
                else: QMessageBox.warning(self, "Input Error", f"Invalid URL format: {self.url_input.text()}"); return
        self.statusBar.showMessage(f"Scraping {url} using multiple strategies...")
        self.markdown_output.setPlaceholderText(f"Scraping {url}...\nPlease wait, this might take up to a minute depending on the site.")
        self.markdown_output.clear(); self.copy_button.setEnabled(False)
        QApplication.processEvents()
        markdown_content = fetch_and_extract_markdown(url) # Call the main fetch function
        self.markdown_output.setPlainText(markdown_content)
        if markdown_content.startswith("Error:"):
             self.markdown_output.setPlaceholderText("Scraping failed. See error message above.")
             self.copy_button.setEnabled(False)
             self.statusBar.showMessage(f"Scraping failed. Check message.", 5000)
             if "All scraping strategies failed" in markdown_content or "Could not initialize" in markdown_content:
                  QMessageBox.critical(self, "Scraping Error", markdown_content)
        else:
             self.markdown_output.setPlaceholderText("Scraped Markdown content will appear here...")
             self.copy_button.setEnabled(True)
             self.statusBar.showMessage("Scraping successful!", 3000)
        self.current_file_path = None
        self.update_title()

    def copy_markdown(self):
        clipboard = QApplication.clipboard(); clipboard.setText(self.markdown_output.toPlainText()); self.statusBar.showMessage("Markdown copied to clipboard!", 2000)
    def save_file_dialog(self):
        url = self.url_input.text(); markdown = self.markdown_output.toPlainText()
        if not url or not markdown or markdown.startswith("Error:"): QMessageBox.warning(self, "Save Error", "Need a valid URL and successfully scraped Markdown content to save."); return
        try:
            parsed_url = urlparse(url); safe_domain = parsed_url.netloc.replace('.', '_'); safe_path = parsed_url.path.replace('/', '_').strip('_');
            if not safe_path: safe_path = 'index'; suggested_name = f"{safe_domain}_{safe_path}.json"; suggested_name = "".join(c for c in suggested_name if c.isalnum() or c in ('_', '-')).rstrip()[:100] + ".json"
        except Exception: suggested_name = "scraped_data.json"
        options = QFileDialog.Options(); file_path, _ = QFileDialog.getSaveFileName(self, "Save Scraped Data", suggested_name, "JSON Files (*.json);;All Files (*)", options=options )
        if file_path:
            if not file_path.lower().endswith('.json'): file_path += '.json'
            self.save_file(file_path)
    def save_file(self, file_path):
        data = { "url": self.url_input.text(), "markdown": self.markdown_output.toPlainText() }
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
            self.current_file_path = file_path; self.add_recent_file(file_path); self.update_title(); self.statusBar.showMessage(f"Data saved to {os.path.basename(file_path)}", 3000)
        except IOError as e: QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}"); self.statusBar.showMessage("Save failed.", 3000)
        except Exception as e: QMessageBox.critical(self, "Save Error", f"An unexpected error occurred during save:\n{e}"); self.statusBar.showMessage("Save failed.", 3000)
    def load_file_dialog(self):
        options = QFileDialog.Options(); file_path, _ = QFileDialog.getOpenFileName( self, "Load Scraped Data", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path: self.load_file(file_path)
    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if "url" not in data or "markdown" not in data: raise ValueError("JSON file is missing 'url' or 'markdown' key.")
            self.url_input.setText(data.get("url", "")); self.markdown_output.setPlainText(data.get("markdown", "")); self.copy_button.setEnabled(bool(data.get("markdown") and not data.get("markdown").startswith("Error:")))
            self.current_file_path = file_path; self.add_recent_file(file_path); self.update_title(); self.statusBar.showMessage(f"Loaded data from {os.path.basename(file_path)}", 3000)
        except FileNotFoundError: QMessageBox.critical(self, "Load Error", f"File not found:\n{file_path}"); self.remove_recent_file(file_path); self.statusBar.showMessage("Load failed: File not found.", 3000)
        except json.JSONDecodeError: QMessageBox.critical(self, "Load Error", f"Could not decode JSON file:\n{file_path}"); self.statusBar.showMessage("Load failed: Invalid JSON.", 3000)
        except ValueError as e: QMessageBox.critical(self, "Load Error", f"Invalid file format: {e}\n{file_path}"); self.statusBar.showMessage("Load failed: Invalid format.", 3000)
        except IOError as e: QMessageBox.critical(self, "Load Error", f"Could not read file:\n{e}"); self.statusBar.showMessage("Load failed.", 3000)
        except Exception as e: QMessageBox.critical(self, "Load Error", f"An unexpected error occurred during load:\n{e}"); self.statusBar.showMessage("Load failed.", 3000)
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
            action = QAction(f"&{i+1} {os.path.basename(file_path)}", self); action.setData(file_path); action.triggered.connect(self.open_recent_file); actions.append(action)
        if actions: self.recent_files_menu.addActions(actions); self.recent_files_menu.setEnabled(True)
        else: self.recent_files_menu.setEnabled(False)
    def open_recent_file(self):
        action = self.sender();
        if action:
            file_path = action.data()
            if file_path and os.path.exists(file_path): self.load_file(file_path)
            elif file_path: QMessageBox.warning(self, "File Not Found", f"The file '{os.path.basename(file_path)}' could not be found.\nIt might have been moved or deleted."); self.remove_recent_file(file_path)
    def closeEvent(self, event): self.settings.sync(); event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    # Needed for undetected-chromedriver when running packaged apps sometimes
    # import multiprocessing
    # multiprocessing.freeze_support() # Uncomment if packaging with PyInstaller/cx_Freeze
    app = QApplication(sys.argv)
    # Optional: app.setStyle('Fusion')
    mainWin = DocScraperApp()
    mainWin.show()
    sys.exit(app.exec_())