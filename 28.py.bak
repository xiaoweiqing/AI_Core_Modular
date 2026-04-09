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

TRIGGER_FILES = {
    "translate_to_en": IPC_DIR / "trigger_translate_to_en",  # Alt+Q
    "translate_to_zh": IPC_DIR / "trigger_translate_to_zh",  # Alt+W
    "optimize_prompt": IPC_DIR
    / "trigger_optimize_prompt",  # ### --- NEW KEYBIND --- ### Alt+E
    "save_input": IPC_DIR / "trigger_save_input",  # Alt+S
    "save_output": IPC_DIR / "trigger_save_output",  # Alt+D
    "cancel_turn": IPC_DIR / "trigger_cancel_last_turn",  # Alt+C
    "mark_high_quality": IPC_DIR / "trigger_mark_high_quality",  # Alt+F
    "personal_risk_analysis": IPC_DIR
    / "trigger_personal_risk_analysis",  # <-- ADD THIS for Alt+X
}

# --- Database Paths --- ### --- MODIFIED BLOCK START --- ###
CORPUS_DB = Path.home() / "ai_training_corpus.sqlite"
DAILY_RECORDS_DB = (
    Path.home() / "gemini_daily_records.sqlite"
)  # New database for daily logs
RISK_ASSESSMENT_DB = Path.home() / "personal_risk_assessments.sqlite"  # <-- ADD THIS

DB_TABLE_NAME = "training_data"
DAILY_RECORDS_TABLE_NAME = "records"  # New table name
# ### --- MODIFIED BLOCK END --- ###
RISK_ASSESSMENT_TABLE_NAME = "assessments"  # <-- ADD THIS

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


# --- 3. Initialization and Setup ---
# --- 为本地模型修改 ---
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
            request_timeout=120,  # 增加超时时间以应对复杂任务
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

        ### --- NEW --- ###
        # After a successful translation, log this event to the daily records DB in the background.
        threading.Thread(
            target=log_daily_record, args=("TRANSLATE", text, final_text), daemon=True
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


class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        self.function_map = {
            str(TRIGGER_FILES["translate_to_en"]): create_hotkey_handler(
                stateless_translate, "en"
            ),
            str(TRIGGER_FILES["translate_to_zh"]): create_hotkey_handler(
                stateless_translate, "zh"
            ),
            str(TRIGGER_FILES["optimize_prompt"]): create_hotkey_handler(
                optimize_prompt_for_ai
            ),
            str(TRIGGER_FILES["save_input"]): create_hotkey_handler(save_input),
            str(TRIGGER_FILES["save_output"]): create_hotkey_handler(save_output),
            str(TRIGGER_FILES["cancel_turn"]): cancel_last_turn,
            str(TRIGGER_FILES["mark_high_quality"]): mark_as_high_quality,
            str(TRIGGER_FILES["personal_risk_analysis"]): create_hotkey_handler(
                run_personal_risk_analysis
            ),  # <-- ADD THIS
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
    if not setup_corpus_database():
        sys.exit(1)
    if not setup_daily_records_database():
        sys.exit(1)
    if not setup_risk_assessment_database():  # <-- CORRECTLY INCLUDED
        sys.exit(1)

    setup_vector_database()
    cleanup_duplicates()
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
    print(f"\n  {Colors.MAGENTA}--- Personal Security Advisor ---{Colors.ENDC}")
    print(
        f"  ✨ Select any text, press [Alt+Z] -> {Colors.MAGENTA}Analyzes for personal risks & opportunities{Colors.ENDC}"
    )
    print(f"\n  {Colors.BLUE}--- AI Power Tools (Auto-Logged) ---{Colors.ENDC}")
    print(f"  1. [Alt+Q] -> Translates to {Colors.CYAN}English{Colors.ENDC}")
    print(f"  2. [Alt+W] -> Translates to {Colors.GREEN}Chinese{Colors.ENDC}")
    print(
        f"  3. [Alt+E] -> {Colors.BLUE}Optimizes into a High-Quality Prompt{Colors.ENDC}"
    )
    print(f"\n  {Colors.MAGENTA}--- AI Training Data Center ---{Colors.ENDC}")
    print(f"  4. [Alt+S] -> {Colors.MAGENTA}Saves Input{Colors.ENDC} to corpus.")
    print(f"  5. [Alt+D] -> {Colors.MAGENTA}Saves Output{Colors.ENDC}, completes pair.")
    print(f"  6. [Alt+C] -> {Colors.YELLOW}Cancels{Colors.ENDC} pending input.")
    print(f"  7. [Alt+F] -> Marks as {Colors.BLUE}High-Quality{Colors.ENDC}.")
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
