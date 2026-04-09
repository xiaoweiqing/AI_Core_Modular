# ==============================================================================
#      Bilingual Dialogue Turn Builder v6.2 - Self-Maintaining
# ==============================================================================
# Version Notes:
# - [【【【 V6.2 Self-Maintaining Feature 】】】
#   - Automatic Cleanup on Startup: The program now automatically deletes
#     incomplete "orphaned" records that are older than 24 hours upon launch.
#   - This keeps the database clean without deleting recent, pending work.
#   - The separate cleanup_db.py script is NO LONGER NEEDED.
# - [【【【 V6.1 Final Bugfix 】】】
#   - Re-added missing helper functions to prevent crashes.
# ==============================================================================
import os
from dotenv import load_dotenv
import json
import re
import traceback
import sys
import threading
import time
import pyperclip
import google.generativeai as genai
import sqlite3
import subprocess
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Network Fix Module
load_dotenv()
if "all_proxy" in os.environ:
    del os.environ["all_proxy"]
if "ALL_PROXY" in os.environ:
    del os.environ["ALL_PROXY"]

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


# --- ANSI color codes ---
class Colors:
    CYAN, GREEN, YELLOW, RED, ENDC = (
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[91m",
        "\033[0m",
    )


# --- 1. Configuration Area ---
API_KEY = os.getenv("GOOGLE_AI_KEY")
IPC_DIR = Path.home() / ".ai_ecosystem_ipc"

TRIGGER_FILES = {
    "my_turn": IPC_DIR / "trigger_translate_to_en",
    "their_turn": IPC_DIR / "trigger_translate_to_zh",
    "cancel_turn": IPC_DIR / "trigger_cancel_last_turn",
}
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


# --- 3. Initialization and Setup ---
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
                id INTEGER PRIMARY KEY AUTOINCREMENT, input_pair TEXT, output_pair TEXT,
                created_at TEXT
            )"""
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


# --- 4. Core Functionality ---


def get_local_time_str():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def intelligent_translate(text, target_lang):
    placeholder_template = "___CODE_BLOCK_{}___"
    code_blocks = re.findall(r"(```.*?```)", text, re.DOTALL)
    plain_text_with_placeholders = text
    for i, block in enumerate(code_blocks):
        plain_text_with_placeholders = plain_text_with_placeholders.replace(
            block, placeholder_template.format(i), 1
        )
    if plain_text_with_placeholders.strip():
        if target_lang == "en":
            prompt = f"Translate the following Chinese text to fluent, natural-sounding English. Preserve the '___CODE_BLOCK_n___' placeholders exactly as they are. Output only the translated text:\n\n{clean_text(plain_text_with_placeholders)}"
        else:
            prompt = f"将以下英文翻译成中文，但请务必保持 '___CODE_BLOCK_n___' 占位符原样不动。只输出翻译后的纯文本:\n\n{clean_text(plain_text_with_placeholders)}"
        response = gemini_model.generate_content(
            prompt, request_options={"timeout": 60}
        )
        translated_text_with_placeholders = response.text.strip()
    else:
        translated_text_with_placeholders = plain_text_with_placeholders
    final_text = translated_text_with_placeholders
    for i, block in enumerate(code_blocks):
        final_text = final_text.replace(placeholder_template.format(i), block, 1)
    return final_text


def record_my_turn(chinese_text_with_code):
    global last_turn_record_id
    with app_controller.lock:
        if app_controller.is_processing:
            return
        app_controller.is_processing = True

    try:
        if last_turn_record_id is not None:
            warning_message = f"A turn (ID: {last_turn_record_id}) is already open. Press Alt+C to cancel it, or Alt+W to complete it."
            print(f"\n{Colors.RED}[!] ACTION BLOCKED: {warning_message}{Colors.ENDC}")
            safe_notification("Action Blocked", warning_message)
            return

        print(
            f"\n[My Turn] Received mixed content input: “{chinese_text_with_code[:40].strip()}...”"
        )
        english_text = intelligent_translate(chinese_text_with_code, "en")
        pyperclip.copy(english_text)

        print(
            f"{Colors.YELLOW}--- [ English Translation (Code Preserved) | Copied ] ---{Colors.ENDC}"
        )
        print(f"{Colors.CYAN}{english_text}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*52}{Colors.ENDC}")

        input_pair = {"zh": chinese_text_with_code, "en": english_text}
        input_pair_json = json.dumps(input_pair, ensure_ascii=False)
        current_time = get_local_time_str()

        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO dialogue_turns (input_pair, created_at) VALUES (?, ?)",
                (input_pair_json, current_time),
            )
            conn.commit()
            last_turn_record_id = cursor.lastrowid
            conn.close()

        print(
            f"✅ [SQLite] New dialogue turn created (ID: {last_turn_record_id}) at {current_time}, waiting for reply."
        )
        safe_notification(
            "Recorded My Turn (Input Pair)", f"ID: {last_turn_record_id} waiting."
        )
    except Exception as e:
        safe_notification("Failed to record my turn", str(e))
        traceback.print_exc()
    finally:
        with app_controller.lock:
            app_controller.is_processing = False


def record_their_turn(english_reply_with_code):
    global last_turn_record_id
    with app_controller.lock:
        if app_controller.is_processing:
            return
        app_controller.is_processing = True
    try:
        print(
            f"\n[Their Turn] Received mixed content reply: “{english_reply_with_code[:50].strip()}...”"
        )
        if last_turn_record_id is None:
            safe_notification("Failed to record their turn", "No pending record found.")
            return

        chinese_translation = intelligent_translate(english_reply_with_code, "zh")
        pyperclip.copy(chinese_translation)

        print(
            f"{Colors.YELLOW}--- [ Chinese Translation (Code Preserved) | Copied ] ---{Colors.ENDC}"
        )
        print(f"{Colors.GREEN}{chinese_translation}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*53}{Colors.ENDC}")

        output_pair = {"en": english_reply_with_code, "zh": chinese_translation}
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
            f"✅ [SQLite] Successfully paired turn to record (ID: {record_id_to_complete})."
        )
        safe_notification(
            "Recorded Their Turn (Output Pair)",
            f"Turn {record_id_to_complete} completed!",
        )
        threading.Thread(
            target=add_turn_to_vector_db, args=(record_id_to_complete,), daemon=True
        ).start()
        last_turn_record_id = None
    except Exception as e:
        safe_notification("Failed to record their turn", str(e))
        traceback.print_exc()
    finally:
        with app_controller.lock:
            app_controller.is_processing = False


def cancel_last_turn():
    global last_turn_record_id
    with app_controller.lock:
        if app_controller.is_processing:
            safe_notification("Action Blocked", "System is busy, please wait.")
            return

    print("\n[Cancel Turn] Received request to cancel last action.")
    if last_turn_record_id is None:
        print(f"{Colors.YELLOW}[!] No open turn to cancel.{Colors.ENDC}")
        safe_notification("Cancel Failed", "There is no open turn to cancel.")
        return

    try:
        record_id_to_delete = last_turn_record_id
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM dialogue_turns WHERE id = ?", (record_id_to_delete,)
            )
            conn.commit()
        last_turn_record_id = None
        print(
            f"{Colors.GREEN}✅ [SQLite] Successfully deleted orphaned record (ID: {record_id_to_delete}).{Colors.ENDC}"
        )
        safe_notification(
            "Action Canceled", f"Deleted pending turn ID: {record_id_to_delete}."
        )
    except Exception as e:
        print(f"!! [Cancel Turn] Error while deleting record: {e}")
        safe_notification("Cancel Failed", "Error deleting record from database.")


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


def add_turn_to_vector_db(record_id):
    if not IS_QDRANT_DB_READY:
        return
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT input_pair, output_pair FROM dialogue_turns WHERE id = ?",
                (record_id,),
            )
            record = cursor.fetchone()
            conn.close()
        if not record or not record[1]:
            return
        input_pair, output_pair = json.loads(record[0]), json.loads(record[1])
        text_to_embed = (
            f"My Input (CN): {input_pair.get('zh', '')}\nMy Input (EN): {input_pair.get('en', '')}\n"
            f"Their Reply (EN): {output_pair.get('en', '')}\nTheir Reply (CN): {output_pair.get('zh', '')}"
        )
        vector = EMBEDDING_MODEL.encode(text_to_embed).tolist()
        payload = {"source_id": record_id, "full_turn": text_to_embed}
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(record_id)))
        QDRANT_CLIENT.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
            wait=True,
        )
        print(f"✅ [Vectorization] Successfully stored turn ID: {record_id} in Qdrant!")
    except Exception as e:
        print(f"!! [Vectorization] Error processing turn ID {record_id}: {e}")


# [NEW] Automatic cleanup function
def cleanup_old_orphans():
    """Deletes incomplete records older than 24 hours on startup."""
    print(">> [System] Performing automatic cleanup of old, incomplete records...")
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()

            # Calculate the cutoff time (24 hours ago)
            cutoff_time = datetime.now(timezone(timedelta(hours=8))) - timedelta(
                hours=24
            )
            cutoff_time_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")

            # Find how many records will be deleted before deleting them
            cursor.execute(
                "SELECT COUNT(*) FROM dialogue_turns WHERE output_pair IS NULL AND created_at < ?",
                (cutoff_time_str,),
            )
            count = cursor.fetchone()[0]

            if count > 0:
                print(
                    f"{Colors.YELLOW}   -> Found {count} orphaned record(s) older than 24 hours. Deleting...{Colors.ENDC}"
                )
                cursor.execute(
                    "DELETE FROM dialogue_turns WHERE output_pair IS NULL AND created_at < ?",
                    (cutoff_time_str,),
                )
                conn.commit()
                print(f"   ✅ [Cleanup] Deleted {count} orphaned record(s).")
            else:
                print("   -> No old orphaned records found. Database is clean.")

    except Exception as e:
        print(f"   !! [Cleanup] An error occurred during automatic cleanup: {e}")


class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        self.function_map = {
            str(TRIGGER_FILES["my_turn"]): create_hotkey_handler(record_my_turn),
            str(TRIGGER_FILES["their_turn"]): create_hotkey_handler(record_their_turn),
            str(TRIGGER_FILES["cancel_turn"]): cancel_last_turn,
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
    print("      Bilingual Dialogue Turn Builder v6.2 - Self-Maintaining")
    print("=" * 70)

    if not all([WATCHDOG_AVAILABLE, VECTOR_DB_AVAILABLE]):
        sys.exit(1)
    if not setup_api():
        sys.exit(1)
    if not setup_corpus_database():
        sys.exit(1)

    # --- [NEW] Call the cleanup function on startup ---
    cleanup_old_orphans()

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
        "  [System Ready] Background service started, listening for capture signals..."
    )
    print("\n  Workflow:")
    print("  1. Select [Input] (Chinese + Code), press [Alt+Q] -> Records your turn.")
    print("  2. Select [Reply] (English + Code), press [Alt+W] -> Completes the turn.")
    print(
        f"  3. Made a mistake? Press [Alt+C] -> {Colors.YELLOW}Cancels the last open turn.{Colors.ENDC}"
    )
    print("\n  Program is now listening on:")
    print(f"  - Record My Turn (Alt+Q):   {TRIGGER_FILES['my_turn']}")
    print(f"  - Record Their Turn (Alt+W): {TRIGGER_FILES['their_turn']}")
    print(
        f"  - {Colors.YELLOW}Cancel Last Turn (Alt+C): {TRIGGER_FILES['cancel_turn']}{Colors.ENDC}"
    )
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
