# ==============================================================================
#      Standalone Local Translator v1.0
# ==============================================================================
# Description:
# - A completely standalone, offline translation tool.
# - Uses a local AI model running on localhost:8087 (e.g., via Ollama/LiteLLM).
# - Provides two distinct hotkeys: one to translate to English, one to Chinese.
# - Works seamlessly with Fedora's system hotkeys via the file signal method.
# - The result is copied directly to the clipboard.
# ==============================================================================

import os
import sys
import re
import threading
import subprocess
import time
from pathlib import Path

# --- Core Dependencies ---
import pyperclip
from langchain_openai import ChatOpenAI

# ==============================================================================
#      Proxy Cleanup Module
# ==============================================================================
# Temporarily clear any system-wide proxy environment variables.
# This ensures that the script can connect directly to localhost services
# (like Ollama on port 8087) without being redirected by the proxy.
for proxy_var in [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    if proxy_var in os.environ:
        del os.environ[proxy_var]
# ==============================================================================

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ==============================================================================
#      1. CONFIGURATION
# ==============================================================================

# --- IPC (Inter-Process Communication) Signal Files ---
# This is where the script will "listen" for your hotkey triggers.
IPC_DIR = Path.home() / ".local_translator_ipc"
TRIGGER_FILES = {
    "to_english": IPC_DIR / "trigger_to_en",
    "to_chinese": IPC_DIR / "trigger_to_zh",
}

# --- Local Model API Endpoint ---
LOCAL_API_BASE = "http://127.0.0.1:8087/v1"

# ==============================================================================
#      2. CORE COMPONENTS
# ==============================================================================

# --- Global Variables & State Controller ---
llm_client = None


class AppController:
    """A simple lock to prevent multiple tasks from running at once."""

    def __init__(self):
        self.is_processing = False
        self.lock = threading.Lock()


app_controller = AppController()


# --- Helper Functions ---
def clean_text(text: str) -> str:
    """Removes non-printable characters from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def safe_notification(title: str, message: str):
    """Uses the native Linux notify-send command to show a desktop notification."""
    try:
        subprocess.run(
            ["notify-send", title, message, "-a", "Local Translator", "-t", "4000"],
            check=True,
        )
    except (FileNotFoundError, Exception):
        pass  # Fails silently if notify-send is not available.


# --- Local AI Initialization ---
def setup_local_llm():
    """Initializes the connection to the local LLM service."""
    global llm_client
    try:
        print(f">> [AI] Connecting to local model API at [{LOCAL_API_BASE}]...")
        llm_client = ChatOpenAI(
            openai_api_base=LOCAL_API_BASE,
            openai_api_key="not-needed",
            model_name="local-model",
            temperature=0.1,
            request_timeout=90,
        )
        # Test the connection with a simple request
        llm_client.invoke("Hi")
        print("✅ [AI] Local AI model connection successful.")
        return True
    except Exception as e:
        print(f"❌ [AI] FATAL ERROR: Failed to connect to local model: {e}")
        print(
            "   -> Ensure your local AI service (e.g., Ollama + LiteLLM) is running and accessible."
        )
        return False


# ==============================================================================
#      3. TRANSLATION LOGIC
# ==============================================================================


def translate_text(text: str, target_language: str):
    """Generic translation function that calls the local model."""
    if target_language == "en":
        prompt = f"Translate the following text to English. Output only the translated text, without any explanations or formatting:\n\n{clean_text(text)}"
        log_prefix = "[Translate->EN]"
        notification_title = "Translated to English"
    elif target_language == "zh":
        prompt = f"将以下内容翻译成中文，只输出翻译后的纯文本，不要任何解释或格式:\n\n{clean_text(text)}"
        log_prefix = "[Translate->ZH]"
        notification_title = "Translated to Chinese"
    else:
        return

    try:
        with app_controller.lock:
            if app_controller.is_processing:
                safe_notification(
                    "Translator Busy", "Please wait for the previous task to finish."
                )
                return
            app_controller.is_processing = True

        print(f"\n{log_prefix} Translating: “{text[:40].strip()}...”")
        response = llm_client.invoke(prompt)
        translated_text = response.content.strip()

        pyperclip.copy(translated_text)
        print(
            f"{log_prefix} Success (copied to clipboard):\n---\n{translated_text}\n---"
        )
        safe_notification(notification_title, "Result copied to clipboard!")

    except Exception as e:
        print(f"!! {log_prefix} Error during translation: {e}")
        safe_notification("Translation Failed", str(e))
    finally:
        with app_controller.lock:
            app_controller.is_processing = False


# ==============================================================================
#      4. TRIGGER AND EXECUTION
# ==============================================================================


def create_handler(target_func, *args):
    """
    Creates a handler that gets text from the user's selection
    and then calls the target function.
    """

    def handler():
        def task_runner():
            text_to_process = ""
            try:
                # Prioritize Wayland's clipboard tool for selected text
                result = subprocess.run(
                    ["wl-paste", "-p"], capture_output=True, text=True, check=True
                )
                text_to_process = result.stdout
            except (FileNotFoundError, subprocess.CalledProcessError):
                try:
                    # Fallback to X11's clipboard tool for selected text
                    result = subprocess.run(
                        ["xclip", "-o", "-selection", "primary"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    text_to_process = result.stdout
                except (FileNotFoundError, subprocess.CalledProcessError):
                    try:
                        # Final fallback to the general clipboard
                        text_to_process = pyperclip.paste()
                    except Exception:
                        pass

            if text_to_process and not text_to_process.isspace():
                # Call the target function (e.g., translate_text) with its arguments
                target_func(text_to_process, *args)
            else:
                print(">> [Handler] No text found in selection or clipboard.")
                safe_notification("No Text Found", "Please select or copy text first.")

        threading.Thread(target=task_runner, daemon=True).start()

    return handler


class FileTriggerHandler(FileSystemEventHandler):
    """Watches for the creation of trigger files and calls the correct function."""

    def __init__(self):
        self.function_map = {
            str(TRIGGER_FILES["to_english"]): create_handler(translate_text, "en"),
            str(TRIGGER_FILES["to_chinese"]): create_handler(translate_text, "zh"),
        }

    def on_created(self, event):
        if not event.is_directory and event.src_path in self.function_map:
            print(f"\n>> [Signal] Trigger received: {Path(event.src_path).name}")
            handler_func = self.function_map[event.src_path]
            handler_func()
            try:
                # Clean up the signal file to prevent re-triggering
                time.sleep(0.1)
                os.unlink(event.src_path)
            except OSError:
                pass


# ==============================================================================
#      5. MAIN PROGRAM
# ==============================================================================

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 60)
    print("        Standalone Local Translator v1.0")
    print("=" * 60)

    if not WATCHDOG_AVAILABLE:
        print("!! FATAL ERROR: The 'watchdog' library is not installed.")
        input(
            "   Please run 'pip install watchdog' and try again. Press Enter to exit."
        )
        sys.exit(1)

    if not setup_local_llm():
        input(
            "   Local AI connection failed. Please check the error and press Enter to exit."
        )
        sys.exit(1)

    # Set up the IPC directory and clean any old trigger files
    IPC_DIR.mkdir(exist_ok=True)
    for f in TRIGGER_FILES.values():
        if f.exists():
            f.unlink()

    # Start the file system watcher
    event_handler = FileTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, str(IPC_DIR), recursive=False)
    observer.start()

    print("\n" + "=" * 60)
    print("  ✅ [System Ready] Local Translator is running in the background.")
    print("\n  Set up your system-wide hotkeys with these commands:")
    print(f"  - Translate to English: touch {TRIGGER_FILES['to_english']}")
    print(f"  - Translate to Chinese: touch {TRIGGER_FILES['to_chinese']}")
    print("\n  Press Ctrl+C in this terminal to stop the script.")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [System] Shutdown signal received, stopping...")
    finally:
        observer.stop()
        observer.join()
        print(">> [System] Translator has been shut down safely.")
