# ==============================================================================
#      Bilingual Dialogue Turn Builder v5.3 - State-Aware & Robust
# ==============================================================================
# Version Notes:
# - [【【【 V5.3 Logic Fix & Robustness 】】】
#   - Added a state-aware check to the "My Turn" (Alt+Q) function.
#   - The program now prevents creating a new dialogue turn if a previous one
#     is still open and waiting for a reply.
#   - This solves the issue of creating accidental duplicate or "orphaned"
#     records due to network lag or repeated key presses.
# - [【【【 V5.2 UI Enhancement 】】】
#   - Added colored terminal output for better readability.
# ==============================================================================
import os
from dotenv import load_dotenv
import json

# Network Fix Module
load_dotenv()
if "all_proxy" in os.environ:
    del os.environ["all_proxy"]
if "ALL_PROXY" in os.environ:
    del os.environ["ALL_PROXY"]

import sys
import threading
import re
import time
import pyperclip
import google.generativeai as genai
import sqlite3
import subprocess
import uuid
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False


# --- ANSI color codes for terminal output ---
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"  # Added Red for warnings
    ENDC = "\033[0m"  # Reset color


# --- 1. Configuration Area ---
API_KEY = os.getenv("GOOGLE_AI_KEY")
IPC_DIR = Path.home() / ".ai_ecosystem_ipc"

TRIGGER_FILES = {
    "my_turn": IPC_DIR / "trigger_translate_to_en",
    "their_turn": IPC_DIR / "trigger_translate_to_zh",
}

# --- Database Configuration ---
CORPUS_DB = Path.home() / "ai_dialogue_turns.sqlite"
QDRANT_COLLECTION_NAME = "dialogue_turns_v1"
VECTOR_DIMENSION = 384


# --- 2. Global State and Control ---
last_turn_record_id = None
db_lock = threading.Lock()
gemini_model = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_QDRANT_DB_READY = False


class AppController:
    def __init__(self):
        self.is_processing = False
        self.lock = threading.Lock()


app_controller = AppController()


# --- 3. Initialization and Setup Module ---


def setup_api():
    global gemini_model
    if not API_KEY or "..." in API_KEY:
        return False
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
        print("✅ [API] Gemini API initialized successfully.")
        return True
    except Exception:
        return False


def setup_corpus_database():
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS dialogue_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_pair TEXT,
                output_pair TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
            conn.commit()
            conn.close()
        print(
            f"✅ [SQLite] Bilingual dialogue turn database initialized successfully (Path: {CORPUS_DB})"
        )
        return True
    except Exception as e:
        print(f"❌ [SQLite] Severe error: Database initialization failed: {e}")
        return False


def setup_vector_database():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_QDRANT_DB_READY
    if not VECTOR_DB_AVAILABLE:
        return
    try:
        print(">> [System] Preparing vector database and model...")
        QDRANT_CLIENT = QdrantClient("localhost", port=6333)
        collections = [c.name for c in QDRANT_CLIENT.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            QDRANT_CLIENT.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION, distance=models.Distance.COSINE
                ),
            )
        EMBEDDING_MODEL = SentenceTransformer("./all-MiniLM-L6-v2")
        IS_QDRANT_DB_READY = True
        print("✅ [System] Vector database and model are ready.")
    except Exception as e:
        print(f"❌ [Qdrant/Embedding] Initialization failed: {e}")
        IS_QDRANT_DB_READY = False


# --- 4. Core Functionality Module ---


def add_turn_to_vector_db(record_id):
    if not IS_QDRANT_DB_READY:
        return
    try:
        # (Logic unchanged)
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT input_pair, output_pair FROM dialogue_turns WHERE id = ?",
                (record_id,),
            )
            record = cursor.fetchone()
            conn.close()

        if not record:
            return

        input_pair_json, output_pair_json = record
        input_pair = json.loads(input_pair_json)
        output_pair = json.loads(output_pair_json)

        text_to_embed = (
            f"My Input (Chinese): {input_pair.get('zh', '')}\n"
            f"My Input (English): {input_pair.get('en', '')}\n"
            f"Their Reply (English): {output_pair.get('en', '')}\n"
            f"Their Reply (Chinese): {output_pair.get('zh', '')}"
        )

        vector = EMBEDDING_MODEL.encode(text_to_embed).tolist()
        payload = {"source_id": record_id, "full_turn": text_to_embed}
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(record_id)))

        QDRANT_CLIENT.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
            wait=True,
        )
        print(
            f"✅ [Vectorization] Successfully stored dialogue turn ID: {record_id} in Qdrant!"
        )
    except Exception as e:
        print(f"!! [Vectorization] Error processing turn ID {record_id}: {e}")


def record_my_turn(chinese_text):
    global last_turn_record_id
    with app_controller.lock:
        if app_controller.is_processing:
            return
        app_controller.is_processing = True

    try:
        # --- [NEW] State Check ---
        if last_turn_record_id is not None:
            warning_message = f"A dialogue turn (ID: {last_turn_record_id}) is already open. Please complete it with Alt+W before starting a new one."
            print(f"\n{Colors.RED}[!] ACTION BLOCKED: {warning_message}{Colors.ENDC}")
            safe_notification("Action Blocked", warning_message)
            return  # Exit the function immediately

        print(f"\n[My Turn] Received Chinese intent: “{chinese_text[:30].strip()}...”")
        prompt = f"Translate the following Chinese text to fluent, natural-sounding English. Output only the translated text:\n\n{clean_text(chinese_text)}"
        response = gemini_model.generate_content(
            prompt, request_options={"timeout": 30}
        )
        english_text = response.text.strip()
        pyperclip.copy(english_text)

        print(
            f"{Colors.YELLOW}--- [ English Translation | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.CYAN}{english_text}{Colors.ENDC}")
        print(
            f"{Colors.YELLOW}----------------------------------------------------{Colors.ENDC}"
        )

        input_pair = {"zh": chinese_text, "en": english_text}
        input_pair_json = json.dumps(input_pair, ensure_ascii=False)

        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO dialogue_turns (input_pair) VALUES (?)", (input_pair_json,)
            )
            conn.commit()
            last_turn_record_id = cursor.lastrowid
            conn.close()

        print(
            f"✅ [SQLite] New dialogue turn created (ID: {last_turn_record_id}), waiting for reply."
        )
        safe_notification(
            "Recorded My Turn (Input Pair)",
            f"ID: {last_turn_record_id} waiting for their turn.",
        )

    except Exception as e:
        safe_notification("Failed to record my turn", str(e))
    finally:
        with app_controller.lock:
            app_controller.is_processing = False


def record_their_turn(english_reply):
    global last_turn_record_id
    with app_controller.lock:
        if app_controller.is_processing:
            return
        app_controller.is_processing = True

    try:
        # (Logic unchanged, as this function should only work if a turn is open)
        print(
            f"\n[Their Turn] Received English reply: “{english_reply[:50].strip()}...”"
        )

        if last_turn_record_id is None:
            safe_notification(
                "Failed to record their turn", "No pending record found for pairing."
            )
            return

        prompt = f"Translate the following English to Chinese, outputting only the translated plain text:\n\n{clean_text(english_reply)}"
        response = gemini_model.generate_content(
            prompt, request_options={"timeout": 30}
        )
        chinese_translation = response.text.strip()
        pyperclip.copy(chinese_translation)

        print(
            f"{Colors.YELLOW}--- [ Chinese Translation | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.GREEN}{chinese_translation}{Colors.ENDC}")
        print(
            f"{Colors.YELLOW}-----------------------------------------------------{Colors.ENDC}"
        )

        output_pair = {"en": english_reply, "zh": chinese_translation}
        output_pair_json = json.dumps(output_pair, ensure_ascii=False)

        record_id_to_complete = last_turn_record_id

        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE dialogue_turns SET output_pair = ? WHERE id = ?",
                (output_pair_json, record_id_to_complete),
            )
            conn.commit()
            conn.close()

        print(
            f"✅ [SQLite] Successfully paired their turn to record (ID: {record_id_to_complete})."
        )
        safe_notification(
            "Recorded Their Turn (Output Pair)",
            f"Dialogue turn {record_id_to_complete} completed!",
        )

        threading.Thread(
            target=add_turn_to_vector_db, args=(record_id_to_complete,), daemon=True
        ).start()

        last_turn_record_id = None  # Reset state after completion

    except Exception as e:
        safe_notification("Failed to record their turn", str(e))
    finally:
        with app_controller.lock:
            app_controller.is_processing = False


# --- 5. Auxiliary and Signal Handling ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def safe_notification(title, message):
    try:
        subprocess.run(
            ["notify-send", title, message, "-a", "Dialogue Builder", "-t", "4000"],
            check=True,
        )
    except:
        pass


class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        self.function_map = {
            str(TRIGGER_FILES["my_turn"]): create_hotkey_handler(record_my_turn),
            str(TRIGGER_FILES["their_turn"]): create_hotkey_handler(record_their_turn),
        }

    def on_created(self, event):
        if not event.is_directory and event.src_path in self.function_map:
            print(f"\n>> [Signal] Received task signal: {Path(event.src_path).name}")
            self.function_map[event.src_path]()
            try:
                time.sleep(0.2)
                os.unlink(event.src_path)
            except OSError:
                pass


def create_hotkey_handler(target_func):
    def handler():
        def task_with_selected_text():
            text_to_process = ""
            try:
                text_to_process = subprocess.run(
                    ["wl-paste", "-p"], capture_output=True, text=True, check=True
                ).stdout
            except:
                try:
                    text_to_process = subprocess.run(
                        ["xclip", "-o", "-selection", "primary"],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                except:
                    text_to_process = pyperclip.paste()
            if text_to_process and not text_to_process.isspace():
                target_func(text_to_process)
            else:
                safe_notification("No text found", "Please select or copy text first")

        threading.Thread(target=task_with_selected_text, daemon=True).start()

    return handler


# --- 6. Main Program Entry ---
if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("      Bilingual Dialogue Turn Builder v5.3 - State-Aware & Robust")
    print("=" * 70)

    if not all([WATCHDOG_AVAILABLE, VECTOR_DB_AVAILABLE]):
        sys.exit(1)

    if not setup_api():
        sys.exit(1)
    if not setup_corpus_database():
        sys.exit(1)
    setup_vector_database()

    IPC_DIR.mkdir(exist_ok=True)
    for file_path in TRIGGER_FILES.values():
        if file_path.exists():
            file_path.unlink()

    observer = Observer()
    observer.schedule(FileTriggerHandler(), str(IPC_DIR), recursive=False)
    observer.start()

    print("\n" + "=" * 70)
    print(
        "  [System Ready] Background service started, listening for dialogue turn capture signals..."
    )
    print("\n  Workflow (one complete dialogue turn):")
    print(
        "  1. Select your [Chinese Intent], press [Alt+Q] -> Records your turn (Input Pair)."
    )
    print(
        "  2. Select the other party's [English Reply], press [Alt+W] -> Records their turn (Output Pair)."
    )
    print(
        "\n  Please ensure your system hotkeys have not been modified. The program is now listening on:"
    )
    print(f"  - Record My Turn (Alt+Q):   {TRIGGER_FILES['my_turn']}")
    print(f"  - Record Their Turn (Alt+W): {TRIGGER_FILES['their_turn']}")
    print("\n  Press Ctrl+C to safely exit the program.")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [System] Exit command received, shutting down...")
    finally:
        observer.stop()
        observer.join()
        print(">> [System] Exited safely.")
        print(">> [System] Exited safely.")
