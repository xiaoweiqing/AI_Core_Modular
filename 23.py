# ==============================================================================
#      Bilingual Dialogue Turn Builder v7.3 - Feature Update
# ==============================================================================
# Version Notes:
# - [【【【 V7.3 Feature - AI Prompt Optimizer 】】】
#   - ADDED: A new hotkey [Alt+E] to automatically transform selected text into
#     a high-quality, structured AI prompt without changing the original intent.
# - [【【【 V7.2 Feature - High-Quality Annotation 】】】
#   - ADDED: A new hotkey [Alt+F] to mark the most recently completed dialogue
#     pair as "high-quality" in the database.
# - [【【【 V7.1 Stability & Hardening Update 】】】
#   - FIXED: Corrected potential race conditions and instability with a more
#     robust locking mechanism and refined threading model.
#   - ENHANCED: Added more granular error handling for all background tasks.
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

# --- Initialization & Setup ---
load_dotenv()

# --- Dependency Checks ---
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print(
        "❌ Critical Error: 'watchdog' library not found. Please run 'pip install watchdog'."
    )

try:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print(
        "❌ Critical Error: Vector libraries not found. Please run 'pip install qdrant-client sentence-transformers'."
    )


# --- ANSI Color Codes ---
class Colors:
    CYAN, GREEN, YELLOW, RED, ENDC, MAGENTA, BLUE = (
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[91m",
        "\033[0m",
        "\033[95m",
        "\033[94m",
    )


# --- 1. Configuration Area ---
API_KEY = os.getenv("GOOGLE_AI_KEY")
IPC_DIR = Path.home() / ".ai_ecosystem_ipc"

TRIGGER_FILES = {
    "translate_to_en": IPC_DIR / "trigger_translate_to_en",  # Alt+Q
    "translate_to_zh": IPC_DIR / "trigger_translate_to_zh",  # Alt+W
    "optimize_prompt": IPC_DIR
    / "trigger_optimize_prompt",  # ### --- NEW KEYBIND --- ### Alt+E
    "save_input": IPC_DIR / "trigger_save_input",  # Alt+S
    "save_output": IPC_DIR / "trigger_save_output",  # Alt+D
    "cancel_turn": IPC_DIR / "trigger_cancel_last_turn",  # Alt+C
    "mark_high_quality": IPC_DIR / "trigger_mark_high_quality",  # Alt+F
}

CORPUS_DB = Path.home() / "ai_training_corpus.sqlite"
DB_TABLE_NAME = "training_data"
QDRANT_COLLECTION_NAME = "dialogue_pairs_v2"
VECTOR_DIMENSION = 384

# --- 2. Global State and Control ---
last_record_id = None
last_completed_id = None
db_lock = threading.Lock()
gemini_model = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_QDRANT_DB_READY = False
app_controller_lock = threading.Lock()


# --- 3. Initialization and Setup ---
def setup_api():
    global gemini_model
    if not API_KEY or "..." in API_KEY:
        print(
            f"❌ {Colors.RED}[API Error] Google AI API Key is not configured.{Colors.ENDC}"
        )
        return False
    try:
        if "all_proxy" in os.environ:
            del os.environ["all_proxy"]
        if "ALL_PROXY" in os.environ:
            del os.environ["ALL_PROXY"]
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
        print(
            f"✅ {Colors.GREEN}[API] Gemini API initialized successfully.{Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(
            f"❌ {Colors.RED}[API Error] Failed to initialize Gemini API: {e}{Colors.ENDC}"
        )
        return False


def setup_corpus_database():
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                output_text TEXT,
                metadata TEXT,
                status TEXT NOT NULL
            )"""
            )
            try:
                cursor.execute(
                    f"ALTER TABLE {DB_TABLE_NAME} ADD COLUMN quality_label TEXT"
                )
                print(f"   -> [SQLite] Added 'quality_label' column for annotations.")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()
        print(
            f"✅ {Colors.GREEN}[SQLite] Training Corpus DB initialized (Path: {CORPUS_DB}){Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(
            f"❌ {Colors.RED}[SQLite Error] Database initialization failed: {e}{Colors.ENDC}"
        )
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
        print(
            f"✅ {Colors.GREEN}[System] Vector database and model are ready.{Colors.ENDC}"
        )
    except Exception as e:
        print(
            f"❌ {Colors.RED}[Qdrant/Embedding Error] Initialization failed. Is the Qdrant Docker container running? Error: {e}{Colors.ENDC}"
        )
        IS_QDRANT_DB_READY = False


# --- 4. Core AI and Helper Functions ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def is_primarily_chinese(text):
    return True if re.search(r"[\u4e00-\u9fff]", text) else False


def get_local_time_str():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def run_ai_task(prompt):
    try:
        response = gemini_model.generate_content(
            prompt, request_options={"timeout": 90}
        )
        return response.text.strip()
    except Exception as e:
        print(f"{Colors.RED}[AI Error] Task failed: {e}{Colors.ENDC}")
        return None


# --- 5. AI-Powered Workflows (Translate, Optimize) ---


### --- NEW FUNCTION START --- ###
def optimize_prompt_for_ai(text):
    """
    Takes user's raw text and transforms it into a high-quality, structured prompt for an AI.
    The core principle is to ENHANCE the prompt without ALTERING the original intent.
    """
    if not app_controller_lock.acquire(blocking=False):
        safe_notification(
            "System Busy", "Another operation is in progress. Please wait."
        )
        return
    try:
        print(
            f"\n{Colors.BLUE}[AI Prompt Optimizer] Received text for optimization...{Colors.ENDC}"
        )

        # This is the "Meta-Prompt" that instructs the AI how to behave.
        meta_prompt = f"""
# Role
You are a world-class Prompt Engineering expert. Your task is to take a user's raw text and transform it into a highly effective, structured, and detailed prompt that will yield the best possible response from an AI model (like a large language model).

# Crucial Rule
**You MUST strictly preserve the original core meaning and intent of the user's text.** Do not add new ideas, change the fundamental request, or interpret beyond the provided text. Your goal is to enrich, clarify, and structure the request, not to alter it.

# Task
Analyze the user's raw text below and enhance it into a professional prompt using the following structure. If a section is not applicable or cannot be inferred from the raw text, omit it.

1.  **Role:** Assign a clear, expert role to the AI (e.g., "You are a senior data scientist," "You are a creative copywriter").
2.  **Context:** Provide necessary background information that the AI needs to understand the request.
3.  **Task/Instruction:** State the primary goal in a clear, direct, and unambiguous way. If the original task is complex, break it down into logical steps.
4.  **Parameters/Constraints:** Define the boundaries. Specify style, tone, length, keywords to include/exclude, and other specific requirements.
5.  **Output Format:** Explicitly define the desired output structure (e.g., "Provide the answer in a Markdown table," "Output a JSON object with the following schema," "Write a 500-word article").

# User's Raw Text
---
{clean_text(text)}
---

# Your Output
Now, based on the rules and the user's text, generate the optimized prompt. Present it clearly within a code block.
"""
        optimized_prompt = run_ai_task(meta_prompt)

        if not optimized_prompt:
            safe_notification(
                "Optimization Failed", "AI model did not return a response."
            )
            return

        # Clean up the response to only get the code block content
        code_block_match = re.search(
            r"```(?:\w+\n)?(.*)```", optimized_prompt, re.DOTALL
        )
        if code_block_match:
            final_prompt = code_block_match.group(1).strip()
        else:
            final_prompt = optimized_prompt  # Fallback

        pyperclip.copy(final_prompt)
        print(
            f"{Colors.YELLOW}--- [ Prompt Optimization Complete | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.GREEN}{final_prompt}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*65}{Colors.ENDC}")
        safe_notification("Prompt Optimized", "Result copied to clipboard.")

    finally:
        app_controller_lock.release()


### --- NEW FUNCTION END --- ###


def stateless_translate(text, target_lang):
    if not app_controller_lock.acquire(blocking=False):
        safe_notification(
            "System Busy", "Another operation is in progress. Please wait."
        )
        return
    try:
        print(
            f"\n{Colors.BLUE}[Stateless Translate] Received text for translation to {target_lang}...{Colors.ENDC}"
        )
        placeholder_template = "___CODE_BLOCK_{}___"
        code_blocks = re.findall(r"(```.*?```)", text, re.DOTALL)
        plain_text = text
        for i, block in enumerate(code_blocks):
            plain_text = plain_text.replace(block, placeholder_template.format(i), 1)

        prompt_text = clean_text(plain_text)
        if target_lang == "en":
            prompt = f"Translate the following text to fluent, natural-sounding English. Preserve the '___CODE_BLOCK_n___' placeholders exactly as they are. Output only the translated text:\n\n{prompt_text}"
            color = Colors.CYAN
        else:
            prompt = f"将以下文本翻译成中文，但请务必保持 '___CODE_BLOCK_n___' 占位符原样不动。只输出翻译后的纯文本:\n\n{prompt_text}"
            color = Colors.GREEN

        translated_text_placeholders = run_ai_task(prompt)
        if not translated_text_placeholders:
            safe_notification(
                "Translation Failed", "AI model did not return a response."
            )
            return

        final_text = translated_text_placeholders
        for i, block in enumerate(code_blocks):
            final_text = final_text.replace(placeholder_template.format(i), block, 1)

        pyperclip.copy(final_text)
        print(
            f"{Colors.YELLOW}--- [ Translation Complete | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{color}{final_text}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*54}{Colors.ENDC}")
        safe_notification("Translation Complete", "Result copied to clipboard.")
    finally:
        app_controller_lock.release()


# --- 6. Data Recording Workflow (Alt+S, Alt+D, Alt+C, Alt+F) ---
def save_input(text):
    global last_record_id
    with app_controller_lock:
        if last_record_id is not None:
            msg = f"Input (ID: {last_record_id}) is pending. Use Alt+D to complete or Alt+C to cancel."
            print(f"\n{Colors.RED}[!] ACTION BLOCKED: {msg}{Colors.ENDC}")
            safe_notification("Action Blocked", msg)
            return

        print(
            f"\n{Colors.MAGENTA}[Save Input] Capturing new input record...{Colors.ENDC}"
        )
        metadata = {"created_at": get_local_time_str()}
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO {DB_TABLE_NAME} (input_text, metadata, status) VALUES (?, ?, ?)",
                (text, metadata_json, "pending_output"),
            )
            conn.commit()
            last_record_id = cursor.lastrowid
            conn.close()

        print(
            f"✅ {Colors.GREEN}[SQLite] Input saved as ID: {last_record_id}. Waiting for output (Alt+D).{Colors.ENDC}"
        )
        safe_notification("Input Saved", f"Record ID: {last_record_id} is pending.")
        threading.Thread(
            target=process_metadata_and_vectorize,
            args=(last_record_id, "input"),
            daemon=True,
        ).start()


def save_output(text):
    global last_record_id, last_completed_id
    with app_controller_lock:
        if last_record_id is None:
            msg = "No input pending. Use Alt+S to save an input first."
            print(f"\n{Colors.RED}[!] ACTION BLOCKED: {msg}{Colors.ENDC}")
            safe_notification("Action Blocked", msg)
            return

        print(
            f"\n{Colors.MAGENTA}[Save Output] Pairing output with record ID: {last_record_id}...{Colors.ENDC}"
        )
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {DB_TABLE_NAME} SET output_text = ?, status = ? WHERE id = ?",
                (text, "pending_summaries", last_record_id),
            )
            conn.commit()
            conn.close()

        print(
            f"✅ {Colors.GREEN}[SQLite] Output saved. Record {last_record_id} is now complete.{Colors.ENDC}"
        )
        safe_notification("Output Saved", f"Record {last_record_id} completed!")

        record_id_to_process = last_record_id
        last_completed_id = last_record_id
        last_record_id = None
        threading.Thread(
            target=process_metadata_and_vectorize,
            args=(record_id_to_process, "output"),
            daemon=True,
        ).start()


def cancel_last_turn():
    global last_record_id
    with app_controller_lock:
        if last_record_id is None:
            msg = "No pending input to cancel."
            print(f"\n{Colors.YELLOW}[!] {msg}{Colors.ENDC}")
            safe_notification("Cancel Failed", msg)
            return

        print(
            f"\n{Colors.YELLOW}[Cancel] Deleting pending record ID: {last_record_id}...{Colors.ENDC}"
        )
        record_id_to_delete = last_record_id
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {DB_TABLE_NAME} WHERE id = ?", (record_id_to_delete,)
            )
            conn.commit()
        last_record_id = None
        print(
            f"✅ {Colors.GREEN}[SQLite] Deleted pending record ID: {record_id_to_delete}.{Colors.ENDC}"
        )
        safe_notification(
            "Action Canceled", f"Deleted pending record ID: {record_id_to_delete}."
        )


def mark_as_high_quality():
    global last_completed_id
    with app_controller_lock:
        if last_completed_id is None:
            msg = "No recently completed record to mark. Complete a pair with Alt+D first."
            print(f"\n{Colors.YELLOW}[!] {msg}{Colors.ENDC}")
            safe_notification("Annotation Failed", msg)
            return

        print(
            f"\n{Colors.BLUE}[Annotate] Marking record ID: {last_completed_id} as 'high-quality'...{Colors.ENDC}"
        )
        record_id_to_mark = last_completed_id
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {DB_TABLE_NAME} SET quality_label = ? WHERE id = ?",
                ("high-quality", record_id_to_mark),
            )
            conn.commit()
            conn.close()

        print(
            f"✅ {Colors.GREEN}[SQLite] Record {record_id_to_mark} has been marked as high-quality.{Colors.ENDC}"
        )
        safe_notification(
            "Annotation Successful",
            f"Record ID: {record_id_to_mark} marked as high-quality.",
        )
        last_completed_id = None


# --- 7. Asynchronous Backend Processing ---
def process_metadata_and_vectorize(record_id, stage):
    print(f">> [Async] Starting background job for record {record_id} (Stage: {stage})")
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT input_text, output_text, metadata FROM {DB_TABLE_NAME} WHERE id = ?",
                (record_id,),
            )
            record = cursor.fetchone()
            conn.close()
        if not record:
            return

        input_text, output_text, metadata_json = record
        metadata = json.loads(metadata_json)

        if stage == "input":
            print(
                f"   -> Record {record_id}: Detecting language and summarizing input..."
            )
            metadata["language"] = "zh" if is_primarily_chinese(input_text) else "en"
            summary_prompt = f"Summarize the following in one short sentence, in the same language as the input:\n\n{clean_text(input_text)}"
            summary = run_ai_task(summary_prompt)
            if summary:
                metadata["input_summary"] = summary

        elif stage == "output" and output_text:
            print(f"   -> Record {record_id}: Summarizing output...")
            summary_prompt = f"Summarize the following in one short sentence, in the same language as the response:\n\n{clean_text(output_text)}"
            summary = run_ai_task(summary_prompt)
            if summary:
                metadata["output_summary"] = summary

        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {DB_TABLE_NAME} SET metadata = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), record_id),
            )
            conn.commit()
            conn.close()
        print(f"   -> Record {record_id}: Metadata updated in database.")

        if output_text and IS_QDRANT_DB_READY:
            print(f"   -> Record {record_id}: Vectorizing complete record...")
            full_text = f"User Input: {input_text}\n\nAI Response: {output_text}"
            vector = EMBEDDING_MODEL.encode(full_text).tolist()
            payload = {"source_id": record_id, "full_turn": full_text, **metadata}
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(record_id)))

            QDRANT_CLIENT.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                ],
                wait=True,
            )

            with db_lock:
                conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute(
                    f"UPDATE {DB_TABLE_NAME} SET status = ? WHERE id = ?",
                    ("complete", record_id),
                )
                conn.commit()
                conn.close()
            print(
                f"✅ {Colors.GREEN}[Async Job Complete] Successfully processed and indexed record ID: {record_id}!{Colors.ENDC}"
            )
    except Exception as e:
        print(
            f"❌ {Colors.RED}[Async Job Error] Background processing failed for record {record_id}: {e}{Colors.ENDC}"
        )
        traceback.print_exc()


# --- 8. System Utilities and Signal Handling ---
def safe_notification(title, message):
    try:
        subprocess.run(
            ["notify-send", title, message, "-a", "Dialogue Builder", "-t", "4000"],
            check=True,
            capture_output=True,
        )
    except:
        pass


def cleanup_old_orphans():
    print(">> [System] Performing startup cleanup of old, incomplete records...")
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, metadata FROM {DB_TABLE_NAME} WHERE status = 'pending_output'"
            )
            orphans = cursor.fetchall()
            ids_to_delete = []
            for record_id, metadata_json in orphans:
                try:
                    created_at_str = json.loads(metadata_json).get("created_at")
                    if created_at_str:
                        created_at = datetime.strptime(
                            created_at_str, "%Y-%m-%d %H:%M:%S"
                        ).replace(tzinfo=timezone(timedelta(hours=8)))
                        if created_at.astimezone(timezone.utc) < cutoff_time:
                            ids_to_delete.append((record_id,))
                except (json.JSONDecodeError, TypeError):
                    print(
                        f"{Colors.YELLOW}   -> Warning: Found malformed metadata for old record ID {record_id}. Skipping.{Colors.ENDC}"
                    )
                    continue
            if ids_to_delete:
                print(
                    f"{Colors.YELLOW}   -> Found {len(ids_to_delete)} orphaned record(s) older than 24 hours. Deleting...{Colors.ENDC}"
                )
                cursor.executemany(
                    f"DELETE FROM {DB_TABLE_NAME} WHERE id = ?", ids_to_delete
                )
                conn.commit()
                print(
                    f"   ✅ [Cleanup] Deleted {len(ids_to_delete)} orphaned record(s)."
                )
            else:
                print("   -> No old orphaned records found. Database is clean.")
            conn.close()
    except Exception as e:
        print(f"   ❌ {Colors.RED}[Cleanup Error] An error occurred: {e}{Colors.ENDC}")


class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        self.function_map = {
            str(TRIGGER_FILES["translate_to_en"]): create_hotkey_handler(
                stateless_translate, "en"
            ),
            str(TRIGGER_FILES["translate_to_zh"]): create_hotkey_handler(
                stateless_translate, "zh"
            ),
            ### --- MODIFIED BLOCK START --- ###
            str(TRIGGER_FILES["optimize_prompt"]): create_hotkey_handler(
                optimize_prompt_for_ai
            ),
            str(TRIGGER_FILES["save_input"]): create_hotkey_handler(save_input),
            str(TRIGGER_FILES["save_output"]): create_hotkey_handler(save_output),
            str(TRIGGER_FILES["cancel_turn"]): cancel_last_turn,
            str(TRIGGER_FILES["mark_high_quality"]): mark_as_high_quality,
            ### --- MODIFIED BLOCK END --- ###
        }

    def on_created(self, event):
        if not event.is_directory and event.src_path in self.function_map:
            print(f"\n>> [Signal] Received task: {Path(event.src_path).name}")
            self.function_map[event.src_path]()
            try:
                time.sleep(0.1)
                os.unlink(event.src_path)
            except OSError:
                pass


def create_hotkey_handler(target_func, *args):
    def handler():
        def task_with_selected_text():
            text_to_process = ""
            try:
                text_to_process = subprocess.run(
                    ["xclip", "-o", "-selection", "primary"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            except Exception:
                try:
                    text_to_process = subprocess.run(
                        ["wl-paste", "-p"], capture_output=True, text=True, check=True
                    ).stdout
                except Exception:
                    text_to_process = ""

            if not text_to_process or text_to_process.isspace():
                text_to_process = pyperclip.paste()

            if text_to_process and not text_to_process.isspace():
                if args:
                    target_func(text_to_process, *args)
                else:
                    target_func(text_to_process)
            else:
                safe_notification("No Text Found", "Please select or copy text first.")

        threading.Thread(target=task_with_selected_text, daemon=True).start()

    return handler


# --- 9. Main Program Entry ---
if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("      Bilingual Dialogue Turn Builder v7.3 - Feature Update")
    print("=" * 70)

    if not all([WATCHDOG_AVAILABLE, VECTOR_DB_AVAILABLE]):
        sys.exit(1)
    if not setup_api():
        sys.exit(1)
    if not setup_corpus_database():
        sys.exit(1)

    setup_vector_database()
    cleanup_old_orphans()

    IPC_DIR.mkdir(exist_ok=True)
    for file_path in TRIGGER_FILES.values():
        if file_path.exists():
            file_path.unlink()

    observer = Observer()
    observer.schedule(FileTriggerHandler(), str(IPC_DIR), recursive=False)
    observer.start()

    print("\n" + "=" * 70)
    print(f"  [System Ready] Background service started. Listening for signals...")
    print("\n  WORKFLOW:")
    print(f"  {Colors.BLUE}--- AI Power Tools ---{Colors.ENDC}")
    ### --- MODIFIED BLOCK START --- ###
    print(
        f"  1. Select text, press [Alt+Q] -> Translates to {Colors.CYAN}English{Colors.ENDC} (No Save)"
    )
    print(
        f"  2. Select text, press [Alt+W] -> Translates to {Colors.GREEN}Chinese{Colors.ENDC} (No Save)"
    )
    print(
        f"  3. Select text, press [Alt+E] -> {Colors.BLUE}Optimizes it into a High-Quality Prompt{Colors.ENDC}"
    )
    print(f"\n  {Colors.MAGENTA}--- AI Training Data Center ---{Colors.ENDC}")
    print(
        f"  4. Select [Input],  press [Alt+S] -> {Colors.MAGENTA}Saves Input{Colors.ENDC} to database."
    )
    print(
        f"  5. Select [Output], press [Alt+D] -> {Colors.MAGENTA}Saves Output{Colors.ENDC}, completes pair."
    )
    print(
        f"  6. Mistake?      Press [Alt+C] -> {Colors.YELLOW}Cancels{Colors.ENDC} pending input."
    )
    print(
        f"  7. After pair,   press [Alt+F] -> Marks as {Colors.BLUE}High-Quality{Colors.ENDC}."
    )
    ### --- MODIFIED BLOCK END --- ###
    print("\n  Press Ctrl+C to safely exit the program.")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [System] Exit command received, shutting down gracefully...")
    finally:
        observer.stop()
        observer.join()
        print(">> [System] Exited safely.")
