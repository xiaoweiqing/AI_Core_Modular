# ==============================================================================
#      Standalone Local Translator v2.1 - Pro (Independent Storage)
# ==============================================================================
# Description:
# - A fast, reliable, and completely standalone offline translation tool.
# - 【Independent Storage】:
#   - Instantly saves every translation to its own dedicated SQLite database (`translations.db`).
#   - Asynchronously vectorizes each record into its own dedicated Qdrant collection (`translator_v1`).
# - 【Clear CLI Output】: The full translation result is always printed in the command line.
# - 【Asynchronous by Design】: All storage and vectorization happens in the background,
#   ensuring the translation function is always instantly responsive.
# - All known bugs have been fixed.
# ==============================================================================

import os
import sys
import re
import threading
import subprocess
import time
import sqlite3
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- Core Dependencies ---
import pyperclip
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ==============================================================================
#      Proxy Cleanup Module
# ==============================================================================
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
os.environ["HF_HUB_OFFLINE"] = "1"

# ==============================================================================
#      1. CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
IPC_DIR = SCRIPT_DIR / ".local_translator_ipc"
TRIGGER_FILES = {
    "to_english": IPC_DIR / "trigger_to_en",
    "to_chinese": IPC_DIR / "trigger_to_zh",
}
# --- 独立存储配置 ---
DB_FILE = SCRIPT_DIR / "translations.db"
QDRANT_COLLECTION_NAME = "translator_v1"
EMBEDDING_MODEL_DIR = SCRIPT_DIR / "all-MiniLM-L6-v2"

LOCAL_API_BASE = "http://127.0.0.1:8087/v1"

# ==============================================================================
#      2. CORE COMPONENTS
# ==============================================================================
llm_client = None
qdrant_client = None
embedding_model = None
IS_STORAGE_ENABLED = False
background_task_loop = None


class AppController:
    def __init__(self):
        self.is_processing = False
        self.lock = threading.Lock()


app_controller = AppController()


# --- 辅助函数 ---
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def safe_notification(title: str, message: str):
    try:
        subprocess.run(
            ["notify-send", title, message, "-a", "Local Translator", "-t", "4000"],
            check=True,
        )
    except Exception:
        pass


# --- 系统初始化 ---
def setup_systems():
    print("=" * 10 + " Standalone Local Translator v2.1 Pro is starting up " + "=" * 10)
    if not (setup_local_llm() and setup_storage()):
        input("!! Core system initialization failed. Press Enter to exit.")
        return False

    # 启动后台任务的专用线程
    def run_background_loop():
        global background_task_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        background_task_loop = loop
        loop.run_forever()

    thread = threading.Thread(target=run_background_loop, daemon=True)
    thread.start()
    return True


def setup_local_llm():
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
        llm_client.invoke("Hi")
        print("✅ [AI] Local AI model connection successful.")
        return True
    except Exception as e:
        print(f"❌ [AI] FATAL ERROR: Failed to connect to local model: {e}")
        return False


def setup_storage():
    """初始化 SQLite 和 Qdrant"""
    global qdrant_client, embedding_model, IS_STORAGE_ENABLED
    try:
        print(f">> [DB] Initializing translation log at: {DB_FILE.resolve()}")
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, original_text TEXT,
                    translated_text TEXT, target_language TEXT, is_vectorized INTEGER DEFAULT 0
                )
            """
            )
        print(f"✅ [DB] Database ready: {DB_FILE.name}")

        print(f">> [Vector] Loading embedding model from: {EMBEDDING_MODEL_DIR}")
        if not EMBEDDING_MODEL_DIR.is_dir():
            print(
                f"❌ [Vector] FATAL ERROR: Embedding model folder not found at '{EMBEDDING_MODEL_DIR}'!"
            )
            return False
        embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_DIR))
        vector_size = embedding_model.get_sentence_embedding_dimension()

        print(">> [Vector] Connecting to Qdrant (localhost:6333)...")
        qdrant_client = QdrantClient(host="localhost", port=6333, timeout=10)
        qdrant_client.get_collections()  # Test connection
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        print(
            f"✅ [Vector] Qdrant connection successful. Using collection: '{QDRANT_COLLECTION_NAME}'"
        )
        IS_STORAGE_ENABLED = True
        return True
    except Exception as e:
        print(f"⚠️ [Storage] WARNING: Could not initialize SQLite/Qdrant: {e}")
        print(
            "   -> Storage and vectorization will be disabled. Translation will still work."
        )
        IS_STORAGE_ENABLED = False
        return True  # 返回 True 以允许程序在无存储模式下运行


# ==============================================================================
# 3. 核心逻辑
# ==============================================================================


def instant_save_and_schedule_background_tasks(original, translated, language):
    """秒存到SQLite，并安排后台向量化任务"""
    if not IS_STORAGE_ENABLED:
        return

    try:
        with sqlite3.connect(DB_FILE) as conn:
            params = (
                datetime.now(timezone(timedelta(hours=8))).isoformat(),
                original,
                translated,
                language,
            )
            cursor = conn.execute(
                "INSERT INTO translations (timestamp, original_text, translated_text, target_language) VALUES (?, ?, ?, ?)",
                params,
            )
            db_id = cursor.lastrowid
            conn.commit()
        print(f"   -> [DB] Translation record saved instantly with ID: {db_id}.")

        # 将耗时任务提交到后台事件循环
        if background_task_loop:
            asyncio.run_coroutine_threadsafe(
                vectorize_record(db_id, original, translated), background_task_loop
            )
    except Exception as e:
        print(f"   !! [DB] Error during instant save: {e}")


async def vectorize_record(db_id, original, translated):
    """后台执行的向量化任务"""
    print(f"   -> [Vector] Background vectorization started for ID: {db_id}...")
    try:
        loop = asyncio.get_running_loop()
        text_to_embed = f"Original: {original}\nTranslated: {translated}"
        vector = await loop.run_in_executor(None, embedding_model.encode, text_to_embed)

        payload = {"db_id": db_id, "original": original, "translated": translated}

        await loop.run_in_executor(
            None,
            lambda: qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=db_id, vector=vector.tolist(), payload=payload
                    )
                ],
                wait=True,
            ),
        )

        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "UPDATE translations SET is_vectorized = 1 WHERE id = ?", (db_id,)
            )

        print(f"   ✅ [Vector] Vectorization for ID: {db_id} completed.")
    except Exception as e:
        print(f"   !! [Vector] Error during vectorization for ID {db_id}: {e}")


def translate_text(text: str, target_language: str):
    """主翻译函数：快速响应，然后触发后台存储"""
    if target_language == "en":
        prompt = f"Translate the following text to English. Output only the translated text, without any explanations or formatting:\n\n{clean_text(text)}"
        log_prefix = "[Translate->EN]"
        notification_title = "Translated to English"
    else:  # 'zh'
        prompt = f"将以下内容翻译成中文，只输出翻译后的纯文本，不要任何解释或格式:\n\n{clean_text(text)}"
        log_prefix = "[Translate->ZH]"
        notification_title = "Translated to Chinese"

    try:
        with app_controller.lock:
            if app_controller.is_processing:
                safe_notification("Translator Busy", "Please wait.")
                return
            app_controller.is_processing = True

        print(f"\n{log_prefix} Translating: “{text[:40].strip()}...”")
        response = llm_client.invoke(prompt)
        translated_text = response.content.strip()

        # 1. 立即响应用户：复制并打印
        pyperclip.copy(translated_text)
        print(f"{log_prefix} Success (copied to clipboard):")
        print("---")
        print(translated_text)
        print("---")

        # 2. 触发后台任务，不阻塞主线程
        threading.Thread(
            target=instant_save_and_schedule_background_tasks,
            args=(text, translated_text, target_language),
            daemon=True,
        ).start()

        safe_notification(notification_title, "Result copied! Saving in background...")

    except Exception as e:
        print(f"!! {log_prefix} Error during translation: {e}")
        safe_notification("Translation Failed", str(e))
    finally:
        with app_controller.lock:
            app_controller.is_processing = False


# ==============================================================================
#      4. 信号处理与主循环
# ==============================================================================
# ... (这部分与之前的版本几乎相同) ...
def create_handler(target_func, *args):
    def handler():
        def task_runner():
            text_to_process = ""
            try:
                result = subprocess.run(
                    ["wl-paste", "-p"], capture_output=True, text=True, check=True
                )
                text_to_process = result.stdout
            except (FileNotFoundError, subprocess.CalledProcessError):
                try:
                    result = subprocess.run(
                        ["xclip", "-o", "-selection", "primary"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    text_to_process = result.stdout
                except (FileNotFoundError, subprocess.CalledProcessError):
                    try:
                        text_to_process = pyperclip.paste()
                    except Exception:
                        pass
            if text_to_process:
                target_func(text_to_process, *args)
            else:
                safe_notification("No Text Found", "Please select or copy text first.")

        # 使用 threading 来确保UI不会被阻塞
        threading.Thread(target=task_runner, daemon=True).start()

    return handler


# 主循环使用 Watchdog，因为它比手写的异步循环更健壮
def main_with_watchdog():
    if not setup_systems():
        return

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print(
            "!! FATAL ERROR: The 'watchdog' library is not installed. Please run 'pip install watchdog'."
        )
        input("Press Enter to exit.")
        return

    class FileTriggerHandler(FileSystemEventHandler):
        def __init__(self):
            self.function_map = {
                str(TRIGGER_FILES["to_english"]): create_handler(translate_text, "en"),
                str(TRIGGER_FILES["to_chinese"]): create_handler(translate_text, "zh"),
            }

        def on_created(self, event):
            if not event.is_directory and event.src_path in self.function_map:
                print(f"\n>> [Signal] Trigger received: {Path(event.src_path).name}")
                self.function_map[event.src_path]()
                try:
                    time.sleep(0.1)
                    os.unlink(event.src_path)
                except OSError:
                    pass

    IPC_DIR.mkdir(exist_ok=True)
    for f in TRIGGER_FILES.values():
        if f.exists():
            f.unlink()

    event_handler = FileTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, str(IPC_DIR), recursive=False)
    observer.start()

    print("\n" + "=" * 60)
    print("  ✅ [System Ready] Local Translator Pro is running in the background.")
    print("\n  Set up your system-wide hotkeys with these commands:")
    print(f"  - Translate to English: touch {TRIGGER_FILES['to_english'].resolve()}")
    print(f"  - Translate to Chinese: touch {TRIGGER_FILES['to_chinese'].resolve()}")
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
        if background_task_loop:
            background_task_loop.call_soon_threadsafe(background_task_loop.stop)
        print(">> [System] Translator has been shut down safely.")


if __name__ == "__main__":
    main_with_watchdog()
