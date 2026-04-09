# ==============================================================================
#      Bilingual Dialogue Turn Builder v7.6 - Context Export
# ==============================================================================
# Version Notes:
# - [【【【 V7.6 Feature - Context Continuation Export 】】】 ### --- NEW NOTE --- ###
#   - ADDED: A new hotkey [Alt+G] to export the last 4 interactions from the
#     daily log to a `session_context.json` file, enabling context
#     continuation in new AI sessions.
# - [【【【 V7.4 Feature - Daily Activity Logger 】】】
#   - ADDED: A parallel database `gemini_daily_records.sqlite` to automatically
#     log every translation and prompt optimization for personal analysis.
#   - ADDED: AI-powered summarization for each logged activity, running in the background.
# - [【【【 V7.3 Feature - AI Prompt Optimizer 】】】
#   - ADDED: A new hotkey [Alt+E] to automatically transform selected text into
#     a high-quality, structured AI prompt without changing the original intent.
# ==============================================================================
from gtts import gTTS
import tempfile

# Make sure 'subprocess' is imported. It should already be there, but double-check.
import subprocess
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

# ---> 【【【 ADD ALL THESE NEW IMPORTS 】】】 <---
from queue import Queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# ---> 【【【 ADDITION ENDS 】】】 <---
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
# ---> 【【【 ADD THIS PROXY BLOCK IF YOU NEED IT 】】】 <---
PROXY_URL = "http://127.0.0.1:2080"
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL
print(f"✅ {Colors.GREEN}[System] HTTP/HTTPS Proxy set to {PROXY_URL}{Colors.ENDC}")
# ---> 【【【 ADDITION ENDS 】】】 <---


API_KEY = os.getenv("GOOGLE_AI_KEY")
IPC_DIR = Path.home() / ".ai_ecosystem_ipc"


# ---> 【【【 ADD THESE NEW LINES FOR MEETING MODE 】】】 <---
AUDIO_ARCHIVE_DIR = Path.home() / "audio_archives"
KEEP_AUDIO_FILES = True
RECORD_CHUNK_SECONDS = 60
# ---> 【【【 ADDITION ENDS 】】】 <---
### --- MODIFIED BLOCK START --- ###
# In Section 1: Configuration Area

### --- MODIFIED BLOCK START --- ###
# In Section 1: Configuration Area

TRIGGER_FILES = {
    "translate_to_en": IPC_DIR / "trigger_translate_to_en",  # Alt+Q
    "translate_to_zh": IPC_DIR / "trigger_translate_to_zh",  # Alt+W
    "optimize_prompt": IPC_DIR / "trigger_optimize_prompt",  # Alt+E
    "export_range_context": IPC_DIR
    / "trigger_export_range_context",  # ### --- NEW HOTKEY --- ### Alt+V
    "read_aloud": IPC_DIR
    / "trigger_read_aloud",  # ### --- ADD THIS NEW LINE (Alt+R) --- ###
    # --- 【【【 在这里添加新热键 (例如 Alt+T) 】】】 ---
    "toggle_read_aloud": IPC_DIR / "trigger_toggle_read_aloud_mode",
    # --- 【【【 添加结束 】】】 --
    "save_input": IPC_DIR / "trigger_save_input",  # Alt+S
    "save_output": IPC_DIR / "trigger_save_output",  # Alt+D
    "cancel_turn": IPC_DIR / "trigger_cancel_last_turn",  # Alt+C
    "mark_high_quality": IPC_DIR / "trigger_mark_high_quality",  # Alt+F
    "personal_risk_analysis": IPC_DIR / "trigger_personal_risk_analysis",  # Alt+X
    # ---> 【【【 ADD THESE TWO NEW HOTKEYS 】】】 <---
    "voice_to_text": IPC_DIR / "trigger_voice_to_text",  # Alt+G
    "meeting_mode": IPC_DIR / "trigger_meeting_mode",  # Alt+B
}
### --- MODIFIED BLOCK END --- ###
### --- MODIFIED BLOCK END --- ###

# --- Database Paths ---
CORPUS_DB = Path.home() / "ai_training_corpus.sqlite"
DAILY_RECORDS_DB = (
    Path.home() / "gemini_daily_records.sqlite"
)  # New database for daily logs
RISK_ASSESSMENT_DB = Path.home() / "personal_risk_assessments.sqlite"  # <-- ADD THIS

# ---> 【【【 ADD THIS NEW DATABASE PATH 】】】 <---
VOICE_TRANSCRIPTS_DB = Path.home() / "voice_transcripts.sqlite"
DB_TABLE_NAME = "training_data"
DAILY_RECORDS_TABLE_NAME = "records"  # New table name
RISK_ASSESSMENT_TABLE_NAME = "assessments"  # <-- ADD THIS

# ---> 【【【 ADD THIS NEW TABLE NAME 】】】 <---
VOICE_TRANSCRIPTS_TABLE_NAME = "transcripts"
QDRANT_COLLECTION_NAME = "dialogue_pairs_v2"
VECTOR_DIMENSION = 384


# --- 2. Global State and Control ---
last_record_id = None
last_completed_id = None
db_lock = threading.Lock()
daily_db_lock = threading.Lock()
gemini_model = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_QDRANT_DB_READY = False
app_controller_lock = threading.Lock()
risk_db_lock = threading.Lock()

# ---> 【【【 在这里添加新的一行 】】】 <---
tts_process = None  # Will hold the currently running TTS playback process
# ---> 【【【 添加结束 】】】 <---

# ---> 【【【 ADD ALL THESE NEW GLOBAL VARIABLES 】】】 <---
WHISPER_MODEL = None
IS_RECORDING = False
MEETING_MODE_ACTIVE = False
transcription_queue = Queue()
recorder_thread = None
# ---> 【【【 ADDITION ENDS 】】】 <---
# --- 【【【 新增的全局开关 】】】 ---
# This will be our in-memory switch, False by default.
READ_ALOUD_MODE_ENABLED = False
# --- 【【【 添加结束 】】】 ---


# --- 3. Initialization and Setup ---

# --- 3. Initialization and Setup ---

# ... (place these two new functions alongside your other setup_* functions) ...


### --- NEW: SETUP WHISPER MODEL (Loads from Local Folder) --- ###
def setup_whisper_model():
    """Loads the faster-whisper model directly from a local project folder."""
    global WHISPER_MODEL
    try:
        # IMPORTANT: Make sure this path is correct for your API script's location
        # If your model folder is inside the same directory as the API script, this is correct.
        local_model_path = "./faster-whisper-large-v3-local"
        if not os.path.isdir(local_model_path):
            print(
                f"❌ {Colors.RED}[Whisper Error] Model folder not found at '{local_model_path}'!{Colors.ENDC}"
            )
            return False
        print(
            f">> [Whisper] Loading 'large-v3' model from local path: {local_model_path}"
        )
        WHISPER_MODEL = WhisperModel(
            local_model_path, device="cpu", compute_type="int8"
        )
        print(
            f"✅ {Colors.GREEN}[Whisper] Model 'large-v3' loaded successfully.{Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(
            f"❌ {Colors.RED}[Whisper Error] Failed to load local model: {e}{Colors.ENDC}"
        )
        return False


### --- NEW: SETUP VOICE TRANSCRIPTS DB --- ###
def setup_voice_transcripts_database():
    """Initializes the database for storing voice transcripts."""
    try:
        with sqlite3.connect(VOICE_TRANSCRIPTS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {VOICE_TRANSCRIPTS_TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    language_detected TEXT NOT NULL,
                    transcribed_text TEXT NOT NULL
                )"""
            )
        print(
            f"✅ {Colors.GREEN}[SQLite] Voice Transcripts DB initialized (Path: {VOICE_TRANSCRIPTS_DB}){Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(
            f"❌ {Colors.RED}[SQLite Error] Voice Transcripts DB initialization failed: {e}{Colors.ENDC}"
        )
        return False


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
        
        # This line is corrected. No 'request_options' here.
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


def setup_daily_records_database():
    """Initializes the new database for logging daily activities."""
    try:
        with daily_db_lock:
            conn = sqlite3.connect(DAILY_RECORDS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS {DAILY_RECORDS_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                original_text TEXT,
                processed_text TEXT,
                meta_prompt TEXT,
                ai_summary TEXT,
                tags TEXT
            )"""
            )
            conn.commit()
            conn.close()
        print(
            f"✅ {Colors.GREEN}[SQLite] Daily Records DB initialized (Path: {DAILY_RECORDS_DB}){Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(
            f"❌ {Colors.RED}[SQLite Error] Daily Records DB initialization failed: {e}{Colors.ENDC}"
        )
        return False


def setup_risk_assessment_database():
    """Initializes the database for storing personal risk and opportunity analyses."""
    try:
        with risk_db_lock:
            conn = sqlite3.connect(RISK_ASSESSMENT_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS {RISK_ASSESSMENT_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_situation TEXT NOT NULL,
                ai_full_response TEXT NOT NULL
            )"""
            )
            conn.commit()
            conn.close()
        print(
            f"✅ {Colors.GREEN}[SQLite] Risk Assessment DB initialized (Path: {RISK_ASSESSMENT_DB}){Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(
            f"❌ {Colors.RED}[SQLite Error] Risk Assessment DB initialization failed: {e}{Colors.ENDC}"
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
        # --- THIS IS THE CORRECTED LINE ---
        if QDRANT_COLLECTION_NAME not in collections:
            QDRANT_CLIENT.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION, distance=models.Distance.COSINE
                ),
            )
        model_path = str(Path(__file__).parent / "all-MiniLM-L6-v2")
        EMBEDDING_MODEL = SentenceTransformer(model_path)
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
        # This is the correct place for the timeout with the new library version.
        response = gemini_model.generate_content(
            prompt, request_options={"timeout": 120}
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"{Colors.RED}[AI Error] Task failed: {e}{Colors.ENDC}")
        return None


def log_daily_record(event_type, original_text, processed_text, meta_prompt=""):
    """
    Asynchronously logs the details of an operation to the daily records database.
    This includes generating an AI summary of the activity.
    """
    print(f">> [Async Log] Recording '{event_type}' event to daily records...")
    try:
        summary_prompt = f"""
# Task
Analyze the user's action below and provide a concise, one-sentence summary in Chinese.
This summary should capture the core intent of the user's original text.

# Data
- **Action Type:** {event_type}
- **User's Original Text:** "{clean_text(original_text)}"
- **AI's Processed Text:** "{clean_text(processed_text)}"

# Your Output
Provide only the one-sentence Chinese summary.
"""
        ai_summary = run_ai_task(summary_prompt)
        if not ai_summary:
            ai_summary = "AI summary generation failed."

        with daily_db_lock:
            conn = sqlite3.connect(DAILY_RECORDS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {DAILY_RECORDS_TABLE_NAME}
                (timestamp, event_type, original_text, processed_text, meta_prompt, ai_summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    get_local_time_str(),
                    event_type,
                    original_text,
                    processed_text,
                    meta_prompt,
                    ai_summary,
                ),
            )
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()

        print(
            f"✅ {Colors.GREEN}[Async Log Complete] Saved event as Record ID: {record_id} in daily logs.{Colors.ENDC}"
        )

    except Exception as e:
        print(
            f"❌ {Colors.RED}[Async Log Error] Failed to log daily record: {e}{Colors.ENDC}"
        )
        traceback.print_exc()


# --- 5. AI-Powered Workflows (Translate, Optimize) ---
### --- NEW FUNCTION START --- ###
### --- FINAL UNIFIED ONLINE TTS & TRANSLATION BLOCK (V1) --- ###
# In Section 5, alongside other workflow functions.
# --- 5. AI-Powered Workflows (Translate, Optimize) ---

### --- NEW: VOICE-TO-TEXT & MEETING RECORDER WORKFLOWS --- ###

### --- CORRECTED & IMPROVED VOICE WORKFLOWS (API Version) --- ###
### --- CORRECTED & FINAL VOICE WORKFLOWS BLOCK --- ###

# Global stream objects for recording
short_rec_stream = None
short_rec_frames = []
meeting_rec_stream = None
meeting_rec_frames = []


# ==============================================================================
#      API版本修复 1: 替换旧的 voice_to_text_workflow 函数
#      API Version Fix 1: Replace the old voice_to_text_workflow function
# ==============================================================================
def voice_to_text_workflow():
    """处理短录音 (Alt+G) 的开启和关闭，并永久保存音频文件。"""
    global IS_RECORDING, short_rec_stream, short_rec_frames

    # --- 这是停止录音的逻辑 ---
    if IS_RECORDING:
        if not short_rec_stream:
            IS_RECORDING = False
            return
        try:
            short_rec_stream.stop()
            short_rec_stream.close()
        except Exception as e:
            print(f"停止短录音流时出错: {e}")

        IS_RECORDING = False
        print(f"\n{Colors.GREEN}>> 短录音已停止。{Colors.ENDC}")
        safe_notification("录音已停止", "正在处理音频...")

        # --- 关键修复: 检查录音是否为空 ---
        if not short_rec_frames:
            print(
                f"   -> {Colors.YELLOW}[警告] 未录制到任何音频，操作已取消。{Colors.ENDC}"
            )
            # 如果主程序锁被占用了，一定要释放它
            if app_controller_lock.locked():
                app_controller_lock.release()
            return

        # --- 实现你想要的【永久保存】逻辑 ---
        try:
            AUDIO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"short_rec_{timestamp}.wav"  # 使用 "short_rec_" 前缀
            filepath = AUDIO_ARCHIVE_DIR / filename
            sf.write(filepath, np.concatenate(short_rec_frames, axis=0), 16000)
            print(
                f"   -> {Colors.GREEN}[永久保存] 音频已保存至: {filepath}{Colors.ENDC}"
            )

            # --- 恢复原有的核心功能：将文件路径传递给转录任务 ---
            # 注意: API 版本使用的是 transcribe_audio_task 帮助函数
            threading.Thread(
                target=transcribe_audio_task, args=(str(filepath), False)
            ).start()

        except Exception as e:
            print(f"❌ {Colors.RED}[保存短录音错误] 无法保存音频文件: {e}{Colors.ENDC}")
            if app_controller_lock.locked():
                app_controller_lock.release()

        return  # 结束停止录音的逻辑

    # --- 这是开始录音的逻辑 (这部分没有改变) ---
    if not app_controller_lock.acquire(blocking=False):
        safe_notification("系统繁忙", "另一个任务正在进行中。")
        return
    try:
        IS_RECORDING = True
        short_rec_frames = []
        print(f"\n{Colors.YELLOW}>> 短录音已开始... 再次按下 Alt+G 停止。{Colors.ENDC}")
        safe_notification("录音已开始", "再次按下 Alt+G 停止。")

        def cb(i, f, t, s):
            short_rec_frames.append(i.copy())

        short_rec_stream = sd.InputStream(samplerate=16000, channels=1, callback=cb)
        short_rec_stream.start()
    except Exception as e:
        print(f"❌ {Colors.RED}[语音启动错误] {e}{Colors.ENDC}")
        IS_RECORDING = False
        app_controller_lock.release()


def toggle_meeting_mode():
    """Starts or stops the long-form meeting recording mode (Alt+B)."""
    global MEETING_MODE_ACTIVE, recorder_thread
    if MEETING_MODE_ACTIVE:
        MEETING_MODE_ACTIVE = False  # Signal the thread to stop
        print(
            f"\n{Colors.RED}>> [Meeting Mode] STOP signal received. Processing final audio chunk...{Colors.ENDC}"
        )
        safe_notification("Meeting Mode", "Stopping...")
        return
    if not app_controller_lock.acquire(blocking=False):
        safe_notification("System Busy", "Cannot start Meeting Mode.")
        return
    try:
        MEETING_MODE_ACTIVE = True
        print(
            f"\n{Colors.GREEN}>> [Meeting Mode] STARTED. Recording continuously.{Colors.ENDC}"
        )
        safe_notification("Meeting Mode Active", "Recording started.")
        recorder_thread = threading.Thread(target=recorder_task, daemon=True)
        recorder_thread.start()
    finally:
        app_controller_lock.release()


def recorder_task():
    """A non-blocking, callback-based background thread for continuous meeting recording."""
    global meeting_rec_stream, meeting_rec_frames, MEETING_MODE_ACTIVE
    AUDIO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    meeting_rec_frames = []

    def meeting_cb(indata, frames, time, status):
        meeting_rec_frames.append(indata.copy())

    try:
        meeting_rec_stream = sd.InputStream(
            samplerate=16000, channels=1, callback=meeting_cb
        )
        meeting_rec_stream.start()
        print(
            f"{Colors.YELLOW}   -> [Recorder] Stream opened. Recording...{Colors.ENDC}"
        )
        while MEETING_MODE_ACTIVE:
            time.sleep(RECORD_CHUNK_SECONDS)
            if not MEETING_MODE_ACTIVE:
                break
            current_frames = meeting_rec_frames
            meeting_rec_frames = []
            if current_frames:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fn = f"rec_{ts}.wav"
                fp = AUDIO_ARCHIVE_DIR / fn
                sf.write(fp, np.concatenate(current_frames, axis=0), 16000)
                print(f"   -> [Recorder] Saved chunk: {fn}{Colors.ENDC}")
                transcription_queue.put(str(fp))
    except Exception as e:
        print(f"❌ {Colors.RED}[Recorder Error] {e}{Colors.ENDC}")
    finally:
        if meeting_rec_stream:
            try:
                meeting_rec_stream.stop()
                meeting_rec_stream.close()
            except Exception as e:
                print(f"Error stopping meeting stream: {e}")
        if meeting_rec_frames:
            print("   -> [Recorder] Saving final audio chunk...")
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fn = f"rec_final_{ts}.wav"
            fp = AUDIO_ARCHIVE_DIR / fn
            sf.write(fp, np.concatenate(meeting_rec_frames, axis=0), 16000)
            print(f"   -> [Recorder] Saved final chunk: {fn}{Colors.ENDC}")
            transcription_queue.put(str(fp))
        print(
            f"{Colors.GREEN}>> [Meeting Mode] Recorder thread stopped gracefully.{Colors.ENDC}"
        )


# ==============================================================================
#      API版本修复 2: 替换旧的 processor_task 函数
#      API Version Fix 2: Replace the old processor_task function
# ==============================================================================
def processor_task():
    """后台线程，从队列中获取音频文件并进行转录。"""
    while True:
        try:
            path = transcription_queue.get()  # 等待任务

            # --- 【【【 这就是关键的修复点 】】】 ---
            # 用 startswith 来精确匹配，而不是用 in
            is_meeting_chunk = Path(path).name.startswith("rec_")
            # --- 【【【 修复结束 】】】 ---

            print(
                f"\n{Colors.BLUE}>> [Processor] Picked up: {Path(path).name}{Colors.ENDC}"
            )
            safe_notification("New Transcription", "Processing audio...")
            segs, info = WHISPER_MODEL.transcribe(path)
            txt = " ".join(s.text for s in segs).strip()

            if txt:
                prefix = "[Meeting] " if is_meeting_chunk else ""
                print(f"   -> [Processor] Done (Lang: {info.language}). Saving.")
                try:
                    with sqlite3.connect(VOICE_TRANSCRIPTS_DB) as conn:
                        conn.execute(
                            f"INSERT INTO {VOICE_TRANSCRIPTS_TABLE_NAME} (timestamp, language_detected, transcribed_text) VALUES (?, ?, ?)",
                            (get_local_time_str(), info.language, f"{prefix}{txt}"),
                        )
                    print(f"   -> [Processor] ✅ Transcript saved.{Colors.ENDC}")

                    # --- 【【【 恢复了原有的核心逻辑 】】】 ---
                    # 如果不是会议区块（也就是我们的短录音），则复制到剪贴板
                    if not is_meeting_chunk:
                        pyperclip.copy(txt)
                        safe_notification(
                            "Transcription Complete", "Text copied to clipboard."
                        )
                        print(
                            f"   -> {Colors.GREEN}[Clipboard] ✅ Transcription copied!{Colors.ENDC}"
                        )

                except Exception as db_e:
                    print(
                        f"   -> [Processor] ❌ {Colors.RED}DB Error: {db_e}{Colors.ENDC}"
                    )
            else:
                print(
                    f"   -> [Processor] {Colors.YELLOW}No speech detected.{Colors.ENDC}"
                )

            # 文件删除逻辑保持不变 (由 KEEP_AUDIO_FILES 控制)
            if not KEEP_AUDIO_FILES and os.path.exists(path):
                try:
                    os.unlink(path)
                    print(f"   -> [Processor] Deleted audio file.{Colors.ENDC}")
                except Exception as del_e:
                    print(
                        f"   -> {Colors.RED}[Error] Failed to delete temp file: {del_e}{Colors.ENDC}"
                    )

            transcription_queue.task_done()
        except Exception as e:
            print(f"❌ {Colors.RED}[Processor Error] {e}{Colors.ENDC}")
            traceback.print_exc()


def transcribe_audio_task(audio_path, is_meeting):
    """Puts a single audio file onto the main processing queue and releases lock if needed."""
    transcription_queue.put(audio_path)
    # Only the short recording mode holds the main lock
    if not is_meeting and app_controller_lock.locked():
        app_controller_lock.release()


### --- END OF VOICE WORKFLOWS BLOCK --- ###

### --- END OF NEW VOICE FUNCTIONS BLOCK --- ###


### --- 【【【 新增：朗读模式切换功能 (ALT+T) 】】】 ---
def toggle_read_aloud_mode():
    """
    Toggles the global read-aloud mode ON or OFF.
    This function doesn't require selected text.
    """
    global READ_ALOUD_MODE_ENABLED

    # Invert the current state (True becomes False, False becomes True)
    READ_ALOUD_MODE_ENABLED = not READ_ALOUD_MODE_ENABLED

    if READ_ALOUD_MODE_ENABLED:
        status_message = "[朗读模式] 已开启"
        color = Colors.GREEN
        print(f"\n{color}>> [System] {status_message}{Colors.ENDC}")
        safe_notification("模式切换", "翻译后将自动朗读 (Alt+Q/W)。")
    else:
        status_message = "[朗读模式] 已关闭"
        color = Colors.YELLOW
        print(f"\n{color}>> [System] {status_message}{Colors.ENDC}")
        safe_notification("模式切换", "翻译后将不再自动朗读。")


### --- 【【【 新增功能结束 】】】 ---


### --- 【【【 用这个新版本，完整替换掉旧的 read_text_aloud 函数 】】】 ---
### --- 【【【 用这个新版本，完整替换掉旧的 read_text_aloud 函数 V2.0 - API版 】】】 ---
def read_text_aloud(text):
    """
    Master function for gTTS (Online). (Version 2.0 with Robust Long Text Handling)
    - It is fully asynchronous.
    - Handles interruptions gracefully.
    - It writes long text to a file before processing, though gTTS is generally robust.
    """
    global tts_process

    # --- 中断逻辑 (这部分不变) ---
    if tts_process and tts_process.poll() is None:
        print(
            f"\n{Colors.YELLOW}[TTS Stop] Received stop signal. Terminating playback...{Colors.ENDC}"
        )
        try:
            tts_process.terminate()
            tts_process.wait(timeout=1)
            print(f"   -> {Colors.GREEN}Playback stopped successfully.{Colors.ENDC}")
        except Exception as e:
            print(f"   -> {Colors.RED}Error stopping process: {e}{Colors.ENDC}")
        finally:
            tts_process = None
        return

    # --- 启动新朗读 (在后台线程中运行) ---
    def task():
        global tts_process
        temp_audio_file = None
        temp_text_file = None  # <-- 我们也为API版本添加临时文本文件
        try:
            print(
                f"\n{Colors.BLUE}[Online Read Aloud] Generating audio with gTTS for text of length {len(text)}...{Colors.ENDC}"
            )

            lang_code = "zh-CN" if is_primarily_chinese(text) else "en"
            print(f"   -> Detected language code: {lang_code}")

            # --- 【【【 核心修复点在这里 】】】 ---
            # 1. 创建一个临时文件来写入我们的长文本
            #    虽然 gTTS 可以直接处理变量，但这是一种更稳妥的编程习惯。
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as fp_text:
                temp_text_file = fp_text.name
                fp_text.write(text)

            # 2. 初始化 gTTS 对象
            tts = gTTS(text=text, lang=lang_code, slow=False)

            # 3. 创建用于输出的音频临时文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp_audio:
                temp_audio_file = fp_audio.name

            # 4. 下载音频文件
            print("   -> Requesting audio from Google TTS service...")
            tts.save(temp_audio_file)
            print("   -> Audio downloaded, starting playback...")

            # 启动播放，并将进程对象存到全局变量
            tts_process = subprocess.Popen(
                ["mpg123", "-q", temp_audio_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tts_process.wait()  # 等待播放结束
            print(f"   -> {Colors.GREEN}Playback complete.{Colors.ENDC}")

        except Exception as e:
            print(
                f"   -> {Colors.RED}An unexpected error occurred during online TTS: {e}{Colors.ENDC}"
            )
            traceback.print_exc()
        finally:
            # 确保清理所有临时文件
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            if temp_text_file and os.path.exists(temp_text_file):
                os.remove(temp_text_file)
            tts_process = None

    # 立即启动后台线程
    threading.Thread(target=task, daemon=True).start()


### --- 【【【 替换结束 】】】 ---

### --- 【【【 替换结束 】】】 ---


# In Section 5, REPLACE the existing translate_and_read_aloud function.


# In Section 5, REPLACE the existing translate_and_read_aloud function with this one.


def translate_and_read_aloud(text, target_lang):
    """
    A unified workflow that checks the IN-MEMORY toggle to decide whether to read aloud.
    (Version 2: Restores the success notification).
    """
    if not app_controller_lock.acquire(blocking=False):
        safe_notification("System Busy", "Another operation is in progress.")
        return

    try:
        action_log = "[Translate & Read]" if READ_ALOUD_MODE_ENABLED else "[Translate]"

        if target_lang == "en":
            print(f"\n{Colors.BLUE}{action_log} Translating to English...{Colors.ENDC}")
            prompt = f"Translate the following text to fluent, natural-sounding English. Preserve code blocks. Output only the translated text:\n\n{clean_text(text)}"
        else:  # target_lang == 'zh'
            print(f"\n{Colors.BLUE}{action_log} Translating to Chinese...{Colors.ENDC}")
            prompt = f"将以下文本翻译成流畅、自然的中文。保留代码块。只输出翻译后的文本:\n\n{clean_text(text)}"

        final_text = run_ai_task(prompt)
        if not final_text:
            safe_notification(
                "Translation Failed", "AI model did not return a response."
            )
            return

        pyperclip.copy(final_text)

        # --- 【【【【【 THIS IS THE MISSING LINE 】】】】】 ---
        # We are adding the success notification right here!
        safe_notification("Translation Complete", "Result copied to clipboard.")
        # --- 【【【【【 FIX COMPLETE 】】】】】 ---

        color = Colors.CYAN if target_lang == "en" else Colors.GREEN
        print(
            f"{Colors.YELLOW}--- [ Translation Complete | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{color}{final_text}{Colors.ENDC}")

        # The rest of the logic for reading aloud or skipping remains the same
        if READ_ALOUD_MODE_ENABLED:
            print("   -> 朗读模式已开启，正在启动后台 TTS...")
            read_text_aloud(final_text)
        else:
            print("   -> 朗读模式已关闭，跳过朗读。")

        # Database logging also remains the same
        threading.Thread(
            target=log_daily_record, args=("TRANSLATE", text, final_text), daemon=True
        ).start()

    finally:
        app_controller_lock.release()


### --- END OF FINAL ONLINE TTS BLOCK --- ###

### --- NEW FUNCTION END --- ###

### --- NEW FUNCTION START --- ###
### --- NEW FUNCTION START --- ###
# (Replace the old export_recent_context function in Section 5 with this)


### --- NEW FUNCTION: CONTEXT EXPORT BY ID RANGE (ALT+V) --- ###
def export_context_by_range(range_text):
    """
    Exports interactions from a specific ID range (e.g., "340-347") provided
    from the clipboard to a uniquely named JSON file.
    """
    if not app_controller_lock.acquire(blocking=False):
        safe_notification(
            "System Busy", "Another operation is in progress. Please wait."
        )
        return

    try:
        print(
            f"\n{Colors.BLUE}[Manual Export] Received range: '{range_text}'...{Colors.ENDC}"
        )

        # 1. Parse the range string like "340 - 347"
        match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", range_text)
        if not match:
            msg = f"Invalid format. Please copy a range like '340-347'."
            print(f"❌ {Colors.RED}[Export Error] {msg}{Colors.ENDC}")
            safe_notification("Export Failed", msg)
            return

        start_id, end_id = sorted([int(match.group(1)), int(match.group(2))])
        print(f"   -> Querying for record IDs between {start_id} and {end_id}.")

        # 2. Fetch data from the main training corpus database
        # This function should query the main CORPUS_DB, not the daily logs.
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, input_text, output_text
                FROM {DB_TABLE_NAME}
                WHERE id BETWEEN ? AND ?
                ORDER BY id ASC
                """,
                (start_id, end_id),
            )
            records = cursor.fetchall()
            conn.close()

        if not records:
            safe_notification(
                "Export Failed", f"No records found in range {start_id}-{end_id}."
            )
            return

        # 3. Format the data into a clear JSON structure
        context_data = [
            {"id": r[0], "user_input": r[1], "ai_output": r[2]} for r in records
        ]

        # 4. Define the output path and save the file
        base_path = Path("/home/weiyubin/projects/AI-Ecosystem-Core/context")
        base_path.mkdir(parents=True, exist_ok=True)
        export_path = base_path / f"session_context_range_{start_id}_to_{end_id}.json"

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(context_data, f, ensure_ascii=False, indent=4)

        # 5. Notify the user of success
        final_message = f"Saved {len(records)} interactions to:\n{export_path}"
        print(f"✅ {Colors.GREEN}[Export Complete] {final_message}{Colors.ENDC}")
        safe_notification("Manual Export Complete", final_message)

    except Exception as e:
        safe_notification("Export Failed", str(e))
        traceback.print_exc()
    finally:
        app_controller_lock.release()


### --- NEW FUNCTION END --- ###


def optimize_prompt_for_ai(text):
    """
    Takes user's raw text and transforms it into a high-quality, structured prompt for an AI.
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
        meta_prompt = f"""
# Role
You are a world-class Prompt Engineering expert. Your task is to take a user's raw text and transform it into a highly effective, structured, and detailed prompt that will yield the best possible response from an AI model (like a large language model).

# Crucial Rule
**You MUST strictly preserve the original core meaning and intent of the user's text.** Do not add new ideas, change the fundamental request, or interpret beyond the provided text. Your goal is to enrich, clarify, and structure the request, not to alter it.

# Task
Analyze the user's raw text below and enhance it into a professional prompt using the following structure. If a section is not applicable or cannot be inferred from the raw text, omit it.
1.  **Role:** Assign a clear, expert role to the AI.
2.  **Context:** Provide necessary background information.
3.  **Task/Instruction:** State the primary goal clearly.
4.  **Parameters/Constraints:** Define boundaries (style, tone, length).
5.  **Output Format:** Define the desired output structure.

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

        code_block_match = re.search(
            r"```(?:\w+\n)?(.*)```", optimized_prompt, re.DOTALL
        )
        final_prompt = (
            code_block_match.group(1).strip() if code_block_match else optimized_prompt
        )

        pyperclip.copy(final_prompt)
        print(
            f"{Colors.YELLOW}--- [ Prompt Optimization Complete | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.GREEN}{final_prompt}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*65}{Colors.ENDC}")
        safe_notification("Prompt Optimized", "Result copied to clipboard.")

        threading.Thread(
            target=log_daily_record,
            args=("OPTIMIZE_PROMPT", text, final_prompt, meta_prompt),
            daemon=True,
        ).start()
    finally:
        app_controller_lock.release()


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

        threading.Thread(
            target=log_daily_record, args=("TRANSLATE", text, final_text), daemon=True
        ).start()
    finally:
        app_controller_lock.release()


### --- NEW & REWRITTEN FUNCTION V2 (Less Rigid): PERSONAL AI SECURITY ANALYSIS --- ###
def run_personal_risk_analysis(situation_text):
    """
    Analyzes a situation based on the user's Core Principles (Personal Constitution)
    and saves the analysis to a dedicated risk assessment database.
    This version uses an improved prompt for a more natural, advisor-like tone.
    """
    if not IS_QDRANT_DB_READY:
        safe_notification("Error", "Vector Database is not ready for analysis.")
        return

    if not app_controller_lock.acquire(blocking=False):
        safe_notification(
            "System Busy", "Another operation is in progress. Please wait."
        )
        return

    try:
        print(
            f"\n{Colors.MAGENTA}[Personal Constitution Analysis] Analyzing new situation...{Colors.ENDC}"
        )

        # 1. 检索核心原则 (这部分不变)
        query_vector = EMBEDDING_MODEL.encode(situation_text).tolist()
        search_results = QDRANT_CLIENT.search(
            collection_name="personal_constitution",
            query_vector=query_vector,
            limit=5,
        )
        retrieved_principles = ""
        for result in search_results:
            retrieved_principles += result.payload.get("text_chunk", "") + "\n---\n"

        # 2. 【核心优化】构建一个更流畅、更顾问式的指令 (Prompt)
        prompt = f"""
# 你的角色
你是我专属的、高度智能的AI战略顾问。你的分析风格应该是深刻、自然、一针见血，像一位经验丰富的导师在给我提供咨询建议，而不是一份死板的报告。我的最终目标是成为以现实利益为导向的顶尖AI工程师。

# 我的个人宪法（思考时必须遵循的核心原则）
---
{retrieved_principles}
---

# 需要分析的新情况
"{situation_text}"

# 你的任务
请深入理解我的个人宪法，并对“新情况”进行全面的分析。请将你的洞察有机地融合在一篇结构清晰、易于阅读的分析报告中。**请不要使用生硬的“分析/结论”模板**，而是用流畅的语言，自然地阐述以下几个方面：

1.  **首要考量 (安全与合规)**：首先明确指出是否存在任何法律风险或欺诈迹象。这是底线。
2.  **潜在的风险与机遇**：接着，深入剖析此情况对我个人（职业、声誉、时间等）可能带来的具体风险，以及其中蕴含的、能助我成长的机遇。
3.  **利益与回报评估**：量化或描述此情况可能带来的实际利益，包括金钱、知识、人脉等。
4.  **最终行动建议**：在综合以上所有分析后，给出一个清晰、可执行的核心行动建议。

请自由地使用加粗、项目符号来突出重点，让报告既专业又易于理解。
"""
        # 3. 获取AI分析结果 (这部分不变, 仍然调用 run_ai_task)
        analysis_response = run_ai_task(prompt)
        if not analysis_response:
            safe_notification(
                "Analysis Failed", "The AI model did not return a response."
            )
            return

        # 4. 保存到数据库 (这部分不变)
        with risk_db_lock:
            conn = sqlite3.connect(RISK_ASSESSMENT_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO {RISK_ASSESSMENT_TABLE_NAME} (timestamp, input_situation, ai_full_response) VALUES (?, ?, ?)",
                (get_local_time_str(), situation_text, analysis_response),
            )
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()

        print(
            f"✅ {Colors.GREEN}[Analysis Complete] Analysis saved to Risk DB as ID: {record_id}.{Colors.ENDC}"
        )

        # 5. 显示结果 (这部分不变)
        pyperclip.copy(analysis_response)
        safe_notification(
            "Analysis Complete", f"Saved as ID: {record_id}. Results copied."
        )
        print(
            f"{Colors.YELLOW}--- [ Personal Constitution Analysis | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.CYAN}{analysis_response}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*65}{Colors.ENDC}")

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
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id FROM {DB_TABLE_NAME} WHERE input_text = ?", (text,)
            )
            existing_record = cursor.fetchone()
            conn.close()
        if existing_record:
            msg = f"This input already exists as Record ID: {existing_record[0]}. Save operation cancelled."
            print(f"\n{Colors.YELLOW}[!] DUPLICATE FOUND: {msg}{Colors.ENDC}")
            safe_notification("Duplicate Found", msg)
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
        record_id_to_delete = last_record_id
        print(
            f"\n{Colors.YELLOW}[Cancel] Deleting pending record ID: {record_id_to_delete}...{Colors.ENDC}"
        )
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {DB_TABLE_NAME} WHERE id = ?", (record_id_to_delete,)
            )
            conn.commit()
            conn.close()  # Added missing conn.close()
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
        record_id_to_mark = last_completed_id
        print(
            f"\n{Colors.BLUE}[Annotate] Marking record ID: {record_id_to_mark} as 'high-quality'...{Colors.ENDC}"
        )
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
            metadata["language"] = "zh" if is_primarily_chinese(input_text) else "en"
            summary_prompt = f"Summarize the following in one short sentence, in the same language as the input:\n\n{clean_text(input_text)}"
            summary = run_ai_task(summary_prompt)
            if summary:
                metadata["input_summary"] = summary
        elif stage == "output" and output_text:
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
        if output_text and IS_QDRANT_DB_READY:
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
                f"✅ {Colors.GREEN}[Async Job Complete] Processed and indexed record ID: {record_id}!{Colors.ENDC}"
            )
    except Exception as e:
        print(
            f"❌ {Colors.RED}[Async Job Error] Background processing failed for record {record_id}: {e}{Colors.ENDC}"
        )
        traceback.print_exc()


# --- 8. System Utilities and Signal Handling ---


### --- 【【【 新增功能：启动时处理待办任务 】】】 --- ###
def process_pending_tasks_on_startup():
    """
    Checks the database on startup for any records that were saved but not
    fully processed, and queues them for background processing.
    """
    print(">> [System] Checking for any pending background tasks from last session...")
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            # 找到所有状态为“等待总结”的记录
            cursor.execute(
                f"SELECT id FROM {DB_TABLE_NAME} WHERE status = 'pending_summaries'"
            )
            pending_records = cursor.fetchall()
            conn.close()

        if pending_records:
            print(
                f"   -> {Colors.YELLOW}Found {len(pending_records)} pending task(s). Queueing them for processing...{Colors.ENDC}"
            )
            for record in pending_records:
                record_id = record[0]
                print(f"      - Queueing task for Record ID: {record_id}")
                # 为每一个待办任务启动一个独立的后台线程
                threading.Thread(
                    target=process_metadata_and_vectorize,
                    args=(record_id, "output"),  # 我们总是从 "output" 阶段重试
                    daemon=True,
                ).start()
        else:
            print("   -> No pending tasks found. All records are fully processed.")

    except Exception as e:
        print(
            f"   ❌ {Colors.RED}[Pending Task Error] Failed to process pending tasks: {e}{Colors.ENDC}"
        )


### --- 【【【 新增功能结束 】】】 ---


def safe_notification(title, message):
    try:
        subprocess.run(
            ["notify-send", title, message, "-a", "Dialogue Builder", "-t", "4000"],
            check=True,
            capture_output=True,
        )
    except:
        pass


def cleanup_duplicates():
    print(">> [System] Performing startup deduplication check...")
    try:
        with db_lock:
            conn = sqlite3.connect(CORPUS_DB, check_same_thread=False)
            cursor = conn.cursor()
            query = f"""
                DELETE FROM {DB_TABLE_NAME}
                WHERE id IN (
                    SELECT id FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (PARTITION BY input_text ORDER BY id) as rn
                        FROM {DB_TABLE_NAME}
                    ) t
                    WHERE t.rn > 1
                )
            """
            cursor.execute(query)
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
        if deleted_count > 0:
            print(f"   ✅ [Cleanup] Deleted {deleted_count} duplicate record(s).")
    except Exception as e:
        print(
            f"   ❌ {Colors.RED}[Deduplication Error] An error occurred: {e}{Colors.ENDC}"
        )


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
                    continue
            if ids_to_delete:
                cursor.executemany(
                    f"DELETE FROM {DB_TABLE_NAME} WHERE id = ?", ids_to_delete
                )
                conn.commit()
                print(
                    f"   ✅ [Cleanup] Deleted {len(ids_to_delete)} old orphaned record(s)."
                )
            conn.close()
    except Exception as e:
        print(f"   ❌ {Colors.RED}[Cleanup Error] An error occurred: {e}{Colors.ENDC}")


# In Section 8: System Utilities and Signal Handling


# In Section 8: System Utilities and Signal Handling


class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        self.function_map = {
            str(TRIGGER_FILES["translate_to_en"]): create_hotkey_handler(
                translate_and_read_aloud, "en"
            ),
            str(TRIGGER_FILES["translate_to_zh"]): create_hotkey_handler(
                translate_and_read_aloud, "zh"
            ),
            str(TRIGGER_FILES["optimize_prompt"]): create_hotkey_handler(
                optimize_prompt_for_ai
            ),
            str(TRIGGER_FILES["export_range_context"]): create_hotkey_handler(
                export_context_by_range
            ),
            str(TRIGGER_FILES["read_aloud"]): create_hotkey_handler(read_text_aloud),
            str(TRIGGER_FILES["toggle_read_aloud"]): toggle_read_aloud_mode,
            str(TRIGGER_FILES["save_input"]): create_hotkey_handler(save_input),
            str(TRIGGER_FILES["save_output"]): create_hotkey_handler(save_output),
            str(TRIGGER_FILES["cancel_turn"]): cancel_last_turn,
            str(TRIGGER_FILES["mark_high_quality"]): mark_as_high_quality,
            str(TRIGGER_FILES["personal_risk_analysis"]): create_hotkey_handler(
                run_personal_risk_analysis
            ),
            # --- Correctly added mappings with proper commas ---
            str(TRIGGER_FILES["voice_to_text"]): voice_to_text_workflow,
            str(TRIGGER_FILES["meeting_mode"]): toggle_meeting_mode,
        }

    ### --- MODIFIED BLOCK END --- ###

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
                (
                    target_func(text_to_process, *args)
                    if args
                    else target_func(text_to_process)
                )
            else:
                safe_notification("No Text Found", "Please select or copy text first.")

        threading.Thread(target=task_with_selected_text, daemon=True).start()

    return handler


# --- 9. Main Program Entry ---
if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("      Bilingual Dialogue Turn Builder v7.6 - Context Export")
    print("=" * 70)

    if not all([WATCHDOG_AVAILABLE, VECTOR_DB_AVAILABLE]):
        sys.exit(1)
    if not setup_api():
        sys.exit(1)
    # ---> 【【【 ADD THESE TWO SETUP CALLS 】】】 <---
    if not setup_voice_transcripts_database():
        sys.exit(1)
    if not setup_whisper_model():
        sys.exit(1)
    # ---> 【【【 ADDITION ENDS 】】】 <---
    if not setup_corpus_database():
        sys.exit(1)
    if not setup_daily_records_database():
        sys.exit(1)
    if not setup_risk_assessment_database():
        sys.exit(1)

    setup_vector_database()
    cleanup_duplicates()
    cleanup_old_orphans()
    # --- 【【【 在这里加上新的一行 】】】 ---
    process_pending_tasks_on_startup()
    # --- 【【【 添加结束 】】】 ---
    # ---> 【【【 THIS LINE IS CRITICAL - MAKE SURE IT IS HERE 】】】 <---
    print(">> [System] Starting background transcription processor...")
    threading.Thread(target=processor_task, daemon=True).start()
    print("✅ [System] Background processor is running.")
    # ---> 【【【 ADDITION ENDS 】】】 <---

    IPC_DIR.mkdir(exist_ok=True)
    for file_path in TRIGGER_FILES.values():
        if file_path.exists():
            file_path.unlink()

    observer = Observer()
    observer.schedule(FileTriggerHandler(), str(IPC_DIR), recursive=False)
    observer.start()

    ### --- MODIFIED BLOCK START --- ###
    # In Section 9: Main Program Entry

    ### --- 【【【 修改这里：更新启动界面的帮助信息 】】】 --- ###

    print("\n" + "=" * 70)
    print(f"  [System Ready] Background service started. Listening for signals...")

    # --- 新增一个专门的语言学习工具区域 ---
    print(f"\n  {Colors.GREEN}--- Language Learning & Tools ---{Colors.ENDC}")
    print(f"  🎤 [Alt+G] -> Record & Transcribe SHORT Voice to Text (Start/Stop)")
    print(
        f"  🔴 [Alt+B] -> {Colors.RED}TOGGLE Long-Form Meeting Recorder (Saves Audio & Text){Colors.ENDC}"
    )
    print(
        f"  ✨ [Alt+R] -> {Colors.GREEN}Reads selected text aloud (EN/ZH Auto-Detect){Colors.ENDC}"
    )

    # --- 【【【 在这里添加新的说明 】】】 ---
    print(
        f"  ✨ [Alt+T] -> {Colors.YELLOW}切换 [翻译后自动朗读] 模式 (开/关){Colors.ENDC}"
    )
    # --- 【【【 添加结束 】】】 ---
    print(f"\n  {Colors.BLUE}--- AI Power Tools (Auto-Logged & Read) ---{Colors.ENDC}")
    print(
        f"  1. [Alt+Q] -> Translates to {Colors.CYAN}English & Reads Aloud{Colors.ENDC}"
    )
    print(
        f"  2. [Alt+W] -> Translates to {Colors.GREEN}Chinese & Reads Aloud{Colors.ENDC}"
    )
    print(
        f"  3. [Alt+E] -> {Colors.BLUE}Optimizes into a High-Quality Prompt{Colors.ENDC}"
    )

    print(f"\n  {Colors.MAGENTA}--- Personal Security & Context ---{Colors.ENDC}")
    # --- 修正风险分析的快捷键 ---
    print(
        f"  ✨ [Alt+Z] -> {Colors.MAGENTA}Analyzes text for personal risks & opportunities{Colors.ENDC}"
    )
    print(
        f"  ✨ Copy 'X-Y' & press [Alt+V] -> {Colors.BLUE}Exports dialogue history in range X-Y{Colors.ENDC}"
    )

    print(f"\n  {Colors.MAGENTA}--- AI Training Data Center ---{Colors.ENDC}")
    print(f"  4. [Alt+S] -> {Colors.MAGENTA}Saves Input{Colors.ENDC} to corpus.")
    print(f"  5. [Alt+D] -> {Colors.MAGENTA}Saves Output{Colors.ENDC}, completes pair.")
    print(f"  6. [Alt+C] -> {Colors.YELLOW}Cancels{Colors.ENDC} pending input.")
    print(f"  7. [Alt+F] -> Marks as {Colors.BLUE}High-Quality{Colors.ENDC}.")

    print("\n  Press Ctrl+C to safely exit the program.")
    print("=" * 70 + "\n")

    ### --- 【【【 替换结束 】】】 --- ###
    ### --- MODIFIED BLOCK END --- ###

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [System] Exit command received, shutting down gracefully...")
    finally:
        observer.stop()
        observer.join()
        print(">> [System] Exited safely.")
