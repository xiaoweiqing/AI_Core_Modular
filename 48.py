# ==============================================================================
#      Bilingual Dialogue Turn Builder v7.3 - Feature Update
# ==============================================================================
# Version Notes:
# - [【【【 V7.4 Feature - Daily Activity Logger 】】】 ### --- NEW NOTE --- ###
#   - ADDED: A parallel database `gemini_daily_records.sqlite` to automatically
#     log every translation and prompt optimization for personal analysis.
#   - ADDED: AI-powered summarization for each logged activity, running in the background.
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
import tempfile  # <-- ADD THIS LINE
from queue import Queue
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# ---> 【【【 ADD THIS IMPORT STATEMENT 】】】 <---
import numpy as np

# ---> 【【【 ADDITION ENDS 】】】 <-
# import google.generativeai as genai
from langchain_openai import ChatOpenAI  # <-- ADD THIS NEW IMPORT

# import google.generativeai as genai
from langchain_openai import ChatOpenAI  # <-- ADD THIS NEW IMPORT
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

# --- [CRITICAL] Proxy Cleaner for Local Model Connection ---
# This block ensures that any system-wide HTTP/HTTPS proxy settings
# do not interfere with the connection to the local model server.
for proxy_var in [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    if proxy_var in os.environ:
        print(f">> [Proxy Cleaner] Found and removed system proxy: {proxy_var}")
        del os.environ[proxy_var]
# --- Proxy Cleaner End ---


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

# ---> 【【【 ADD THESE NEW LINES FOR MEETING MODE 】】】 <---
AUDIO_ARCHIVE_DIR = Path.home() / "audio_archives"  # Folder to save meeting recordings
KEEP_AUDIO_FILES = (
    True  # Set to False if you want to auto-delete WAV files after transcription
)
RECORD_CHUNK_SECONDS = 60  # Record in 60-second chunks. You can change this.
# ---> 【【【 ADDITION ENDS 】】】 <---
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
    # --- 【【【 添加结束 】】】 ---
    "save_input": IPC_DIR / "trigger_save_input",  # Alt+S
    "save_output": IPC_DIR / "trigger_save_output",  # Alt+D
    "cancel_turn": IPC_DIR / "trigger_cancel_last_turn",  # Alt+C
    "mark_high_quality": IPC_DIR / "trigger_mark_high_quality",  # Alt+F
    "personal_risk_analysis": IPC_DIR / "trigger_personal_risk_analysis",  # Alt+X
    # ---> 【【【 ADD THIS NEW HOTKEY 】】】 <---
    "voice_to_text": IPC_DIR / "trigger_voice_to_text",  # Alt+G
    # ---> 【【【 ADD THIS NEW HOTKEY 】】】 <---
    "meeting_mode": IPC_DIR / "trigger_meeting_mode",  # Alt+B
}

# --- Database Paths --- ### --- MODIFIED BLOCK START --- ###
CORPUS_DB = Path.home() / "ai_training_corpus.sqlite"
DAILY_RECORDS_DB = (
    Path.home() / "gemini_daily_records.sqlite"
)  # New database for daily logs
RISK_ASSESSMENT_DB = Path.home() / "personal_risk_assessments.sqlite"  # <-- ADD THIS

# ---> 【【【 ADD THIS NEW LINE 】】】 <---
VOICE_TRANSCRIPTS_DB = Path.home() / "voice_transcripts.sqlite"

DB_TABLE_NAME = "training_data"
DAILY_RECORDS_TABLE_NAME = "records"  # New table name
# ### --- MODIFIED BLOCK END --- ###
RISK_ASSESSMENT_TABLE_NAME = "assessments"  # <-- ADD THIS

# ---> 【【【 ADD THIS NEW LINE 】】】 <---
VOICE_TRANSCRIPTS_TABLE_NAME = "transcripts"
QDRANT_COLLECTION_NAME = "dialogue_pairs_v2"
VECTOR_DIMENSION = 384


# --- 2. Global State and Control ---
last_record_id = None
last_completed_id = None
db_lock = threading.Lock()
daily_db_lock = (
    threading.Lock()
)  ### --- NEW --- ### A dedicated lock for the new database
gemini_model = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_QDRANT_DB_READY = False
app_controller_lock = threading.Lock()
risk_db_lock = threading.Lock()  # <-- ADD THIS

# ---> 【【【 在这里添加新的一行 】】】 <---
tts_process = None  # Will hold the currently running TTS playback process
# ---> 【【【 添加结束 】】】 <---
# --- 【【【 新增的全局开关 (本地版) 】】】 ---
# ---> 【【【 ADD THIS NEW LINE 】】】 <---
WHISPER_MODEL = None  # A global placeholder for the loaded Whisper model
# 这个变量将作为内存中的开关，默认是关闭的

# ---> 【【【 ADD THIS NEW LINE 】】】 <---
IS_RECORDING = False  # State tracker for voice recording
# ---> 【【【 ADDITION ENDS 】】】 <---
# ---> 【【【 ADD THESE NEW GLOBAL VARIABLES 】】】 <---
MEETING_MODE_ACTIVE = False  # State tracker for the new meeting mode
transcription_queue = Queue()  # The queue to hold file paths for processing
recorder_thread = None  # A placeholder for our recorder thread
# ---> 【【【 ADDITION ENDS 】】】 <---


READ_ALOUD_MODE_ENABLED = False
# --- 【【【 添加结束 】】】 ---


# --- 3. Initialization and Setup ---
# --- 为本地模型修改 ---
from faster_whisper import (
    WhisperModel,
)  # <-- Make sure this import is at the top of your script
import sounddevice as sd  # <-- Add this import
import soundfile as sf  # <-- Add this import


### --- NEW FUNCTION: SETUP WHISPER MODEL --- ###
### --- NEW & IMPROVED setup_whisper_model (Loads from Local Folder) --- ###
def setup_whisper_model():
    """Loads the faster-whisper model directly from a local project folder."""
    global WHISPER_MODEL

    try:
        # Define the path to your manually downloaded model folder
        # The './' means it's relative to where the script is running.
        local_model_path = "./faster-whisper-large-v3-local"

        if not os.path.isdir(local_model_path):
            print(
                f"❌ {Colors.RED}[Whisper Error] Model folder not found at '{local_model_path}'!{Colors.ENDC}"
            )
            print(
                f"   {Colors.YELLOW}Please ensure you have downloaded the model files into that folder.{Colors.ENDC}"
            )
            return False

        print(f">> [Whisper] Loading 'base' model from local path: {local_model_path}")

        # We now pass the FOLDER PATH directly to WhisperModel.
        # This prevents it from ever trying to connect to the internet.
        WHISPER_MODEL = WhisperModel(
            local_model_path, device="cpu", compute_type="int8"
        )

        print(
            f"✅ {Colors.GREEN}[Whisper] Model 'base' loaded successfully from local files.{Colors.ENDC}"
        )
        return True

    except Exception as e:
        print(
            f"❌ {Colors.RED}[Whisper Error] Failed to load local model: {e}{Colors.ENDC}"
        )
        print(
            f"   {Colors.YELLOW}Please ensure all model files (config.json, model.bin, etc.) are in the folder.{Colors.ENDC}"
        )
        return False


### --- NEW FUNCTION: SETUP VOICE TRANSCRIPTS DB --- ###
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
    global llm  # 我们将全局变量从 gemini_model 改为更通用的 llm

    # 您指定的本地服务器地址
    local_api_url = "http://127.0.0.1:8087/v1"

    try:
        print(
            f"{Colors.BLUE}>> [本地AI] 正在尝试连接本地模型: {local_api_url}{Colors.ENDC}"
        )

        # 使用 ChatOpenAI 来连接本地服务
        llm = ChatOpenAI(
            openai_api_base=local_api_url,
            openai_api_key="na",  # 本地模型不需要API Key
            model_name="local-model",  # 模型名称可以任意填写
            temperature=0.3,  # 调整模型的创造性
            request_timeout=300,  # 增加超时时间以应对复杂任务
            streaming=False,  # 对于这个程序，我们不需要流式输出
        )

        # 测试连接
        llm.invoke("Hi")

        print(
            f"✅ {Colors.GREEN}[本地AI] 连接成功！已准备好使用本地模型。{Colors.ENDC}"
        )
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[本地AI错误] 连接本地模型服务器失败。{Colors.ENDC}")
        print(
            f"   {Colors.YELLOW}请检查您的本地AI服务（如 LM Studio）是否已启动并正在运行？错误: {e}{Colors.ENDC}"
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


### --- NEW FUNCTION START --- ###
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


### --- NEW FUNCTION END --- ###
# (Paste this block into Section 3 of 25.py)


### --- NEW FUNCTION: SETUP RISK ASSESSMENT DB --- ###
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


### --- NEW FUNCTION: PERSONAL AI SECURITY ANALYSIS (ALT+X) --- ###
def run_personal_risk_analysis(situation_text):
    """
    Analyzes a situation using a personal knowledge base to identify risks and opportunities.
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
            f"\n{Colors.MAGENTA}[Personal Oracle] Analyzing new situation...{Colors.ENDC}"
        )

        # 1. Retrieve Relevant Personal History from Vector DB
        query_vector = EMBEDDING_MODEL.encode(situation_text).tolist()
        search_results = QDRANT_CLIENT.search(
            collection_name="personal_constitution",  # Use the collection we just created
            query_vector=query_vector,
            limit=3,
        )
        retrieved_context = ""
        for result in search_results:
            retrieved_context += result.payload.get("text_chunk", "") + "\n---\n"

        # 2. Build the Powerful Prompt for Gemini
        prompt = f"""
# Your Role
You are a Personal Strategic Advisor and Risk Analyst. Your sole mission is to protect the user from bad decisions by analyzing a new situation against their personal resume and history, provided below.

# User's Personal Profile (Retrieved from Memory)
---
{retrieved_context}
---

# New Situation to Analyze
"{situation_text}"

# Your Task
Based ONLY on the user's provided profile, analyze the new situation.
1.  **Identify Risks:** Point out any similarities to past negative experiences or conflicts with their skills and values.
2.  **Identify Opportunities:** Highlight any alignment with their skills or stated goals.
3.  **Recommend Actions:** Suggest concrete next steps or questions the user should ask.

Provide a clear, structured analysis.
"""
        # 3. Get the Analysis from the AI
        analysis_response = run_ai_task(prompt)
        if not analysis_response:
            safe_notification(
                "Analysis Failed", "The AI model did not return a response."
            )
            return

        # 4. Save the Analysis to the Dedicated Database
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
            f"✅ {Colors.GREEN}[Oracle] Analysis complete. Saved as Assessment ID: {record_id}.{Colors.ENDC}"
        )

        # 5. Display the Results and Notify the User
        pyperclip.copy(analysis_response)
        safe_notification(
            "Analysis Complete", f"Saved as ID: {record_id}. Results copied."
        )
        print(
            f"{Colors.YELLOW}--- [ Personal Analysis Complete | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.CYAN}{analysis_response}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*65}{Colors.ENDC}")

    finally:
        app_controller_lock.release()


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
        # --- ROBUST MODEL PATH ---
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


# --- 为本地模型修改 (V2 - 增加输出清理功能) ---
# --- 为本地模型修改 (V3 - 终极版，支持多种格式的思考过程清理) ---
def run_ai_task(prompt):
    try:
        response = llm.invoke(prompt)
        raw_output = response.content

        # 我们从原始输出开始，一步步进行清理
        cleaned_text = raw_output

        # --- 【【【 核心升级点在这里 】】】 ---

        # 第1步：检查并移除新的 <think>...</think> 格式
        # 我们用 </think> 作为分割点，因为它标志着思考过程的结束
        if "</think>" in cleaned_text:
            # 分割字符串，只取 </think> 之后的部分
            # 这样就把整个 <think>...</think> 块都丢掉了
            _, cleaned_text = cleaned_text.split("</think>", 1)

        # 第2步：在清理过的文本上，再检查并移除旧的 <|channel|>... 格式
        # 这确保了即使某个模型疯狂到两种格式都输出，也能正确处理
        final_answer_marker = "<|channel|>final<|message|>"
        if final_answer_marker in cleaned_text:
            # 再次分割，取标记之后的部分
            _, cleaned_text = cleaned_text.split(final_answer_marker, 1)

        # 第3步：返回最终清理干净、并移除首尾空格的文本
        return cleaned_text.strip()
        # --- 【【【 修改结束 】】】 ---

    except Exception as e:
        print(f"{Colors.RED}[AI 错误] 任务执行失败: {e}{Colors.ENDC}")
        traceback.print_exc()
        return None


### --- NEW FUNCTION START --- ###
def log_daily_record(event_type, original_text, processed_text, meta_prompt=""):
    """
    Asynchronously logs the details of an operation to the daily records database.
    This includes generating an AI summary of the activity.
    """
    print(f">> [Async Log] Recording '{event_type}' event to daily records...")
    try:
        # 1. Generate AI Summary
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
            ai_summary = "AI summary generation failed."  # Fallback

        # 2. Save to Database
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


### --- NEW FUNCTION END --- ###


# --- 5. AI-Powered Workflows (Translate, Optimize) ---

# 在 Section 5, 和其他工作流函数放在一起

### --- NEW: LONG-FORM MEETING RECORDER WORKFLOW (ALT+H) --- ###


def toggle_meeting_mode():
    """Starts or stops the long-form meeting recording mode."""
    global MEETING_MODE_ACTIVE, recorder_thread

    # --- STOP MEETING MODE ---
    if MEETING_MODE_ACTIVE:
        MEETING_MODE_ACTIVE = False
        print(
            f"\n{Colors.RED}>> [Meeting Mode] STOP signal received. Finishing final recording chunk...{Colors.ENDC}"
        )
        safe_notification(
            "Meeting Mode", "Stopping... The final audio chunk is being processed."
        )
        # The recorder_thread will see the flag and stop on its own.
        return

    # --- START MEETING MODE ---
    if not app_controller_lock.acquire(blocking=False):
        safe_notification(
            "System Busy", "Cannot start Meeting Mode while another task is running."
        )
        return

    try:
        MEETING_MODE_ACTIVE = True
        print(
            f"\n{Colors.GREEN}>> [Meeting Mode] STARTED. Recording in {RECORD_CHUNK_SECONDS}-second chunks.{Colors.ENDC}"
        )
        print(f"   Audio files will be saved to: {AUDIO_ARCHIVE_DIR}")
        print(f"   Press Alt+H again to stop.{Colors.ENDC}")
        safe_notification(
            "Meeting Mode Active", "Recording has started in the background."
        )

        # Start the dedicated recorder thread
        recorder_thread = threading.Thread(target=recorder_task, daemon=True)
        recorder_thread.start()

    finally:
        # The main lock is released immediately, allowing other tasks to run.
        app_controller_lock.release()


def recorder_task():
    """
    Runs in a background thread, continuously records audio in chunks,
    saves them to files, and puts them on the transcription queue.
    """
    AUDIO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_RATE = 16000

    while MEETING_MODE_ACTIVE:
        try:
            print(
                f"{Colors.YELLOW}   -> [Recorder] Starting new {RECORD_CHUNK_SECONDS}s chunk...{Colors.ENDC}"
            )

            # Record one chunk of audio
            recording = sd.rec(
                int(RECORD_CHUNK_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
            )
            sd.wait()

            # If the stop signal was received during recording, discard this chunk and exit
            if not MEETING_MODE_ACTIVE:
                break

            # Create a unique filename and save the chunk
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"recording_{timestamp}.wav"
            filepath = AUDIO_ARCHIVE_DIR / filename

            sf.write(filepath, recording, SAMPLE_RATE)

            print(f"   -> [Recorder] Saved chunk: {filename}{Colors.ENDC}")

            # Add the new file to the queue for the processor thread to handle
            transcription_queue.put(str(filepath))

        except Exception as e:
            print(
                f"❌ {Colors.RED}[Recorder Error] An error occurred in the recording loop: {e}{Colors.ENDC}"
            )
            time.sleep(5)  # Wait a bit before retrying

    print(
        f"{Colors.GREEN}>> [Meeting Mode] Recorder thread has stopped gracefully.{Colors.ENDC}"
    )


def processor_task():
    """
    Runs continuously in the background from script start.
    Waits for audio files to appear in the queue and transcribes them.
    """
    while True:
        try:
            audio_path_str = (
                transcription_queue.get()
            )  # This will block until a file is available

            print(
                f"\n{Colors.BLUE}>> [Processor] Picked up new task: {Path(audio_path_str).name}{Colors.ENDC}"
            )
            safe_notification("New Transcription", f"Processing audio chunk...")

            segments, info = WHISPER_MODEL.transcribe(audio_path_str)
            full_text = " ".join(segment.text for segment in segments).strip()

            # In processor_task function in Section 5

            # ... (inside the try block after getting the full_text) ...
            if full_text:
                print(
                    f"   -> [Processor] Transcription complete (Lang: {info.language}). Saving to DB."
                )
                # Save the transcript to the database
                try:
                    with sqlite3.connect(VOICE_TRANSCRIPTS_DB) as conn:
                        cursor = conn.cursor()
                        # ---> 【【【 CONFIRM THIS PREFIX EXISTS 】】】 <---
                        cursor.execute(
                            f"INSERT INTO {VOICE_TRANSCRIPTS_TABLE_NAME} (timestamp, language_detected, transcribed_text) VALUES (?, ?, ?)",
                            (
                                get_local_time_str(),
                                info.language,
                                f"[Meeting Audio Chunk] {full_text}",
                            ),
                        )
                    print(
                        f"   -> [Processor] ✅ Transcript saved to database.{Colors.ENDC}"
                    )
                # ...
                except Exception as db_e:
                    print(
                        f"   -> [Processor] ❌ {Colors.RED}Failed to save transcript to DB: {db_e}{Colors.ENDC}"
                    )
            else:
                print(
                    f"   -> [Processor] {Colors.YELLOW}No speech detected in this chunk.{Colors.ENDC}"
                )

            # Finally, decide whether to keep or delete the audio file
            if not KEEP_AUDIO_FILES:
                try:
                    os.unlink(audio_path_str)
                    print(
                        f"   -> [Processor] Deleted audio file: {Path(audio_path_str).name}{Colors.ENDC}"
                    )
                except Exception as del_e:
                    print(
                        f"   -> [Processor] ❌ {Colors.RED}Failed to delete audio file: {del_e}{Colors.ENDC}"
                    )

            transcription_queue.task_done()

        except Exception as e:
            print(
                f"❌ {Colors.RED}[Processor Error] A critical error occurred in the processor thread: {e}{Colors.ENDC}"
            )
            traceback.print_exc()


### --- NEW FUNCTION: VOICE-TO-TEXT WORKFLOW (ALT+G) --- ###
### --- NEW & IMPROVED VOICE-TO-TEXT WORKFLOW (ALT+G) --- ###
# This new workflow uses a state machine logic (start/stop)

# Global stream object for recording
### --- 5. AI-Powered Workflows (Translate, Optimize) --- ###

### --- NEW: VOICE-TO-TEXT & MEETING RECORDER WORKFLOWS --- ###
### --- (这是从您的API版本复制的、经过验证的完美逻辑) --- ###

# 用于录音的全局流对象
short_rec_stream = None
short_rec_frames = []
meeting_rec_stream = None
meeting_rec_frames = []


# ==============================================================================
#      请用下面这个【完整】的函数，替换掉你脚本里旧的 voice_to_text_workflow 函数
#      REPLACE your old voice_to_text_workflow function with this COMPLETE new one
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

        # --- 关键修复 1: 检查录音是否为空 ---
        # 如果 short_rec_frames 列表是空的，就说明录音时间太短，直接退出，防止崩溃
        if not short_rec_frames:
            print(
                f"   -> {Colors.YELLOW}[警告] 未录制到任何音频，操作已取消。{Colors.ENDC}"
            )
            # 如果主程序锁被占用了，一定要释放它
            if app_controller_lock.locked():
                app_controller_lock.release()
            return

        # --- 关键修复 2: 实现你想要的【永久保存】逻辑 ---
        # 这段代码现在可以安全运行，因为我们已经确认录音不为空
        try:
            # 确保存档目录存在
            AUDIO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

            # 为短录音创建一个唯一的、带时间戳的文件名
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"short_rec_{timestamp}.wav"  # 使用 "short_rec_" 前缀以作区分
            filepath = AUDIO_ARCHIVE_DIR / filename

            # 将录音数据写入这个永久路径
            sf.write(filepath, np.concatenate(short_rec_frames, axis=0), 16000)

            print(
                f"   -> {Colors.GREEN}[永久保存] 音频已保存至: {filepath}{Colors.ENDC}"
            )

            # --- 恢复原有的核心功能：将文件路径放入转录队列 ---
            transcription_queue.put(str(filepath))

        except Exception as e:
            print(f"❌ {Colors.RED}[保存短录音错误] 无法保存音频文件: {e}{Colors.ENDC}")
            # 如果保存失败，也要释放锁
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
    """启动或停止长时会议录音模式 (Alt+B)。"""
    global MEETING_MODE_ACTIVE, recorder_thread
    if MEETING_MODE_ACTIVE:
        MEETING_MODE_ACTIVE = False  # 向线程发送停止信号
        print(
            f"\n{Colors.RED}>> [会议模式] 收到停止信号。正在处理最后一个音频块...{Colors.ENDC}"
        )
        safe_notification("会议模式", "正在停止...")
        return
    if not app_controller_lock.acquire(blocking=False):
        safe_notification("系统繁忙", "无法启动会议模式。")
        return
    try:
        MEETING_MODE_ACTIVE = True
        print(f"\n{Colors.GREEN}>> [会议模式] 已启动。正在持续录音。{Colors.ENDC}")
        safe_notification("会议模式已激活", "录音已开始。")
        recorder_thread = threading.Thread(target=recorder_task, daemon=True)
        recorder_thread.start()
    finally:
        app_controller_lock.release()


def recorder_task():
    """一个非阻塞的、基于回调的后台线程，用于持续的会议录音。"""
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
        print(f"{Colors.YELLOW}   -> [录音机] 流已打开。正在录音...{Colors.ENDC}")
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
                print(f"   -> [录音机] 已保存区块: {fn}{Colors.ENDC}")
                transcription_queue.put(str(fp))
    except Exception as e:
        print(f"❌ {Colors.RED}[录音机错误] {e}{Colors.ENDC}")
    finally:
        if meeting_rec_stream:
            try:
                meeting_rec_stream.stop()
                meeting_rec_stream.close()
            except Exception as e:
                print(f"停止会议流时出错: {e}")
        if meeting_rec_frames:
            print("   -> [录音机] 正在保存最后一个音频块...")
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fn = f"rec_final_{ts}.wav"
            fp = AUDIO_ARCHIVE_DIR / fn
            sf.write(fp, np.concatenate(meeting_rec_frames, axis=0), 16000)
            print(f"   -> [录音机] 已保存最后区块: {fn}{Colors.ENDC}")
            transcription_queue.put(str(fp))
        print(f"{Colors.GREEN}>> [会议模式] 录音机线程已平稳停止。{Colors.ENDC}")


# ==============================================================================
#      请用下面这个【完整】的函数，替换掉你脚本里旧的 processor_task 函数
#      REPLACE your old processor_task function with this COMPLETE new one
# ==============================================================================
def processor_task():
    """后台线程，从队列中获取音频文件并进行转录。"""
    while True:
        try:
            path = transcription_queue.get()  # 等待任务

            # --- 【【【 这就是关键的修复点 】】】 ---
            # 我们必须用 startswith 来精确匹配，而不是用 in 来模糊匹配
            # This is the critical fix. We must use startswith for a precise match, not 'in'.
            is_meeting_chunk = Path(path).name.startswith("rec_")
            # --- 【【【 修复结束 】】】 ---

            print(
                f"\n{Colors.BLUE}>> [处理器] 已拾取任务: {Path(path).name}{Colors.ENDC}"
            )
            safe_notification("新的转录任务", "正在处理音频...")

            # 使用 Whisper 模型进行转录
            segs, info = WHISPER_MODEL.transcribe(path)
            txt = " ".join(s.text for s in segs).strip()

            if txt:
                # 根据是否为会议区块添加前缀
                prefix = "[Meeting] " if is_meeting_chunk else ""
                final_text = f"{prefix}{txt}"
                print(f"   -> [处理器] 完成 (语言: {info.language}). 正在保存。")

                try:
                    # 将结果保存到数据库
                    with sqlite3.connect(VOICE_TRANSCRIPTS_DB) as conn:
                        conn.execute(
                            f"INSERT INTO {VOICE_TRANSCRIPTS_TABLE_NAME} (timestamp, language_detected, transcribed_text) VALUES (?, ?, ?)",
                            (get_local_time_str(), info.language, final_text),
                        )
                    print(f"   -> [处理器] ✅ 转录稿已保存。{Colors.ENDC}")

                    # --- 【【【 恢复了原有的核心逻辑 】】】 ---
                    # 如果不是会议区块（也就是我们的短录音），则复制到剪贴板
                    # The original core logic is now restored.
                    if not is_meeting_chunk:
                        pyperclip.copy(txt)
                        safe_notification("转录完成", "文本已复制到剪贴板。")
                        print(
                            f"   -> {Colors.GREEN}[剪贴板] ✅ 转录文本已复制!{Colors.ENDC}"
                        )

                except Exception as db_e:
                    print(
                        f"   -> [处理器] ❌ {Colors.RED}数据库错误: {db_e}{Colors.ENDC}"
                    )
            else:
                print(f"   -> [处理器] {Colors.YELLOW}未检测到语音。{Colors.ENDC}")

            # 清理音频文件
            if not KEEP_AUDIO_FILES and os.path.exists(path):
                os.unlink(path)
                print(f"   -> [处理器] 已删除音频文件。{Colors.ENDC}")

            # 释放主程序锁（仅短录音会持有）
            if not is_meeting_chunk and app_controller_lock.locked():
                app_controller_lock.release()

            transcription_queue.task_done()
        except Exception as e:
            print(f"❌ {Colors.RED}[处理器错误] {e}{Colors.ENDC}")
            traceback.print_exc()


### --- 语音工作流替换结束 --- ###


### --- 【【【 新增：朗读模式切换功能 (ALT+T) 】】】 ---
def toggle_read_aloud_mode():
    """
    Toggles the global read-aloud mode ON or OFF.
    这个函数不需要选中的文本。
    """
    global READ_ALOUD_MODE_ENABLED

    # 反转当前状态 (True 变 False, False 变 True)
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


### --- NEW FUNCTION for Translate & Read --- ###
### --- FINAL UNIFIED TTS FUNCTIONS (V4) --- ###


### --- FINAL UNIFIED TTS & TRANSLATION BLOCK (V5 - Bilingual) --- ###


### --- 【【【 用这个新版本，完整替换掉旧的 read_text_aloud 函数 】】】 ---
def read_text_aloud(text):
    """
    Master function for Piper TTS.
    - If called while nothing is playing, it starts reading the text.
    - If called while something IS playing, it stops the current playback.
    - It is fully asynchronous and does not block other tasks like translation.
    """
    global tts_process  # 声明我们将修改全局变量

    # --- 核心逻辑 1: 中断功能 ---
    # 检查是否有一个正在播放的进程，并且它还在运行
    if tts_process and tts_process.poll() is None:
        print(
            f"\n{Colors.YELLOW}[TTS Stop] Received stop signal. Terminating playback...{Colors.ENDC}"
        )
        try:
            tts_process.terminate()  # 强制停止进程
            tts_process.wait(timeout=1)  # 等待进程结束
            print(f"   -> {Colors.GREEN}Playback stopped successfully.{Colors.ENDC}")
        except Exception as e:
            print(f"   -> {Colors.RED}Error stopping process: {e}{Colors.ENDC}")
        finally:
            tts_process = None  # 清理全局变量
        return  # 中断后，直接退出函数

    # --- 核心逻辑 2: 启动新朗读 (在后台线程中运行) ---
    def task():
        global tts_process  # 在线程中也要声明
        temp_audio_file = None
        try:
            print(
                f"\n{Colors.BLUE}[Async TTS] Generating audio with Piper...{Colors.ENDC}"
            )

            script_dir = Path(__file__).parent
            piper_exe = script_dir / "piper" / "piper"

            if is_primarily_chinese(text):
                model_file = script_dir / "zh_CN-huayan-medium.onnx"
            else:
                model_file = script_dir / "en_US-lessac-medium.onnx"

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp_audio:
                temp_audio_file = fp_audio.name

            piper_command = [
                str(piper_exe),
                "--model",
                str(model_file),
                "--output_file",
                temp_audio_file,
            ]

            # 生成音频
            subprocess.run(
                piper_command,
                input=text,
                text=True,
                encoding="utf-8",
                check=True,
                capture_output=True,
                timeout=60,
            )

            # 启动播放，并把进程对象存到全局变量中
            tts_process = subprocess.Popen(
                ["aplay", temp_audio_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tts_process.wait()  # 等待播放结束
            print(f"   -> {Colors.GREEN}Playback complete.{Colors.ENDC}")

        except Exception as e:
            print(
                f"   -> {Colors.RED}An unexpected error occurred during TTS: {e}{Colors.ENDC}"
            )
        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            tts_process = None  # 播放结束后，清理全局变量

    # 立即启动后台线程
    threading.Thread(target=task, daemon=True).start()


### --- 【【【 替换结束 】】】 ---

# 在 Section 5, 替换掉旧的 translate_and_read_aloud 函数


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


### --- END OF FINAL TTS BLOCK --- ###


### --- END OF FINAL TTS BLOCK --- ###

### --- END OF NEW FUNCTION --- ###

### --- END OF NEW FUNCTION --- ###


### --- NEW FUNCTION: CONTEXT EXPORT (ALT+G) --- ###
# (Add this new function in Section 5: AI-Powered Workflows)


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
        with db_lock:  # Using the lock for ai_training_corpus.sqlite
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

        # 4. Define the output path from your image and save the file
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

        code_block_match = re.search(
            r"```(?:\w+\n)?(.*)```", optimized_prompt, re.DOTALL
        )
        if code_block_match:
            final_prompt = code_block_match.group(1).strip()
        else:
            final_prompt = optimized_prompt

        pyperclip.copy(final_prompt)
        print(
            f"{Colors.YELLOW}--- [ Prompt Optimization Complete | Copied to Clipboard ] ---{Colors.ENDC}"
        )
        print(f"{Colors.GREEN}{final_prompt}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-'*65}{Colors.ENDC}")
        safe_notification("Prompt Optimized", "Result copied to clipboard.")

        ### --- NEW --- ###
        # After successfully optimizing, log this event to the daily records DB in the background.
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

        # Log the translation event in the background
        threading.Thread(
            target=log_daily_record, args=("TRANSLATE", text, final_text), daemon=True
        ).start()

        # ### --- THIS IS THE NEW, SMART LOGIC --- ###
        # If the target language was English, start reading it aloud in a separate thread.
        if target_lang == "en":
            print("   -> [Auto-Read] Starting offline TTS in the background...")
            threading.Thread(
                target=read_text_aloud, args=(final_text,), daemon=True
            ).start()

    finally:
        app_controller_lock.release()


# --- 6. Data Recording Workflow (Alt+S, Alt+D, Alt+C, Alt+F) ---
# ... (The rest of this section remains unchanged, as it pertains to the corpus database)
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
# ... (This section also remains unchanged as it's for the corpus database)
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
# ... (This section remains unchanged)
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
            print(
                f"   ✅ [Cleanup] Found and deleted {deleted_count} duplicate record(s)."
            )
        else:
            print("   -> No duplicate records found. Database is clean.")
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


# In the FileTriggerHandler class


# In Section 8: System Utilities and Signal Handling


class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        self.function_map = {
            # --- 【【【 这是修改过的部分 】】】 ---
            str(TRIGGER_FILES["translate_to_en"]): create_hotkey_handler(
                translate_and_read_aloud, "en"  # 调用统一功能，并告诉它目标是英文
            ),
            str(TRIGGER_FILES["translate_to_zh"]): create_hotkey_handler(
                translate_and_read_aloud, "zh"  # 调用统一功能，并告诉它目标是中文
            ),
            # --- 【【【 修改结束 】】】 ---
            str(TRIGGER_FILES["optimize_prompt"]): create_hotkey_handler(
                optimize_prompt_for_ai
            ),
            # ### --- THIS IS THE MODIFIED LINE --- ###
            str(TRIGGER_FILES["export_range_context"]): create_hotkey_handler(
                export_context_by_range
            ),
            # ### --- ADD THIS NEW LINE --- ###
            str(TRIGGER_FILES["read_aloud"]): create_hotkey_handler(read_text_aloud),
            # ### -------------------------- ###
            # --- 【【【 在这里添加新的映射关系 】】】 ---
            str(TRIGGER_FILES["toggle_read_aloud"]): toggle_read_aloud_mode,
            # --- 【【【 添加结束 】】】 ---
            str(TRIGGER_FILES["save_input"]): create_hotkey_handler(save_input),
            str(TRIGGER_FILES["save_output"]): create_hotkey_handler(save_output),
            str(TRIGGER_FILES["cancel_turn"]): cancel_last_turn,
            str(TRIGGER_FILES["mark_high_quality"]): mark_as_high_quality,
            str(TRIGGER_FILES["personal_risk_analysis"]): create_hotkey_handler(
                run_personal_risk_analysis
            ),
            # ---> 【【【 ADD THIS NEW MAPPING 】】】 <---
            # Note: No create_hotkey_handler, we call it directly.
            str(TRIGGER_FILES["voice_to_text"]): voice_to_text_workflow,
            # ---> 【【【 ADD THIS NEW MAPPING 】】】 <---
            str(TRIGGER_FILES["meeting_mode"]): toggle_meeting_mode,
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
    # ### --- MODIFIED --- ###
    print("      Bilingual Dialogue Turn Builder v7.5 - Personal Security Advisor")
    print("=" * 70)

    if not all([WATCHDOG_AVAILABLE, VECTOR_DB_AVAILABLE]):
        sys.exit(1)
    if not setup_api():
        sys.exit(1)
    # ---> 【【【 ADD THESE TWO LINES 】】】 <---
    if not setup_voice_transcripts_database():
        sys.exit(1)
    if not setup_whisper_model():
        sys.exit(1)
    # ---> 【【【 ADDITION ENDS 】】】 <---
    if not setup_corpus_database():
        sys.exit(1)
    if not setup_daily_records_database():
        sys.exit(1)
    if not setup_risk_assessment_database():  # <-- CORRECTLY INCLUDED
        sys.exit(1)

    setup_vector_database()
    cleanup_duplicates()
    cleanup_old_orphans()
    # --- 【【【 在这里加上新的一行 】】】 ---
    process_pending_tasks_on_startup()

    # ---> 【【【 在这里添加启动后台处理器的关键代码 】】】 <---
    print(">> [System] Starting background transcription processor...")
    threading.Thread(target=processor_task, daemon=True).start()
    print("✅ [System] Background processor is running.")
    # ---> 【【【 添加结束 】】】 <---
    # ---> 【【【 ADDITION ENDS 】】】 <---
    # --- 【【【 添加结束 】】】 ---
    IPC_DIR.mkdir(exist_ok=True)
    for file_path in TRIGGER_FILES.values():
        if file_path.exists():
            file_path.unlink()

    observer = Observer()
    observer.schedule(FileTriggerHandler(), str(IPC_DIR), recursive=False)
    observer.start()

    # In Section 9: Main Program Entry

    ### --- 【【【 修改这里：更新启动界面的帮助信息 】】】 --- ###

    # (在第9节的末尾，替换掉旧的 print() 块)
    print("\n" + "=" * 70)
    print(f"  [System Ready] Background service started. Listening for signals...")

    print(f"\n  {Colors.GREEN}--- Language Learning & Tools ---{Colors.ENDC}")
    print(f"  🎤 [Alt+G] -> Record & Transcribe SHORT Voice to Text (Start/Stop)")
    print(
        f"  🔴 [Alt+B] -> {Colors.RED}TOGGLE Long-Form Meeting Recorder (Saves Audio & Text){Colors.ENDC}"
    )
    print(
        f"  ✨ [Alt+R] -> {Colors.GREEN}Reads selected text aloud (EN/ZH Auto-Detect){Colors.ENDC}"
    )
    print(
        f"  ✨ [Alt+T] -> {Colors.YELLOW}切换 [翻译后自动朗读] 模式 (开/关){Colors.ENDC}"
    )
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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [System] Exit command received, shutting down gracefully...")
    finally:
        observer.stop()
        observer.join()
        print(">> [System] Exited safely.")
