# ==============================================================================
#      【【【 THIS IS THE COMPLETE, MODIFIED FILE 】】】
# ==============================================================================
# core/database_manager.py
import os  # <--- 【【【 添加这一行 】】】
import asyncio
import aiosqlite
import traceback
from pathlib import Path
from qdrant_client import QdrantClient, models

# Import our custom modules
from core import state
from config import settings # Import the settings object
from utils.helpers import Colors

# In core/database_manager.py
# REPLACE your existing setup_all_databases function with this ENTIRE block.

# core/database_manager.py

async def setup_all_databases():
    """
    [ASYNC MASTER] Initializes all databases concurrently for maximum startup speed.
    [MODIFIED] This version now includes the setup for our new Search Archive database.
    """
    print("\n" + "="*70)
    print("--- Initializing All Databases Concurrently ---")

    # Define the SQL for the Constitution DB (existing)
    constitution_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {settings.CONSTITUTION_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            principle_text TEXT NOT NULL UNIQUE
        )
    """

    # Define the SQL for the Phrase Expander DB (existing)
    phrase_expander_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {settings.PHRASE_EXPANDER_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phrase TEXT NOT NULL UNIQUE,
            full_content TEXT NOT NULL
        )
    """

    # ==============================================================================
    #      【【【 1. Define the SQL for our NEW Search Archive table 】】】
    # ==============================================================================
    search_archive_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {settings.SEARCH_ARCHIVE_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_query TEXT NOT NULL,
            search_context TEXT,
            ai_answer TEXT
        )
    """

    tasks = [
        _initialize_sqlite_db("Corpus DB", settings.CORPUS_DB, f"CREATE TABLE IF NOT EXISTS {settings.CORPUS_TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, input_text TEXT, output_text TEXT, metadata TEXT, status TEXT NOT NULL, quality_label TEXT)"),
        _initialize_sqlite_db("Daily Records DB", settings.DAILY_RECORDS_DB, f"CREATE TABLE IF NOT EXISTS {settings.DAILY_RECORDS_TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, event_type TEXT NOT NULL, original_text TEXT, processed_text TEXT, meta_prompt TEXT, ai_summary TEXT, tags TEXT)"),
        _initialize_sqlite_db("Risk Assessment DB", settings.RISK_ASSESSMENT_DB, f"CREATE TABLE IF NOT EXISTS {settings.RISK_ASSESSMENT_TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, input_situation TEXT NOT NULL, ai_full_response TEXT NOT NULL)"),
        _setup_voice_transcripts_db_with_migration(),
        _initialize_sqlite_db("Concise Q&A DB", settings.CONCISE_QA_DB, f"CREATE TABLE IF NOT EXISTS {settings.CONCISE_QA_TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, user_query TEXT NOT NULL, ai_response TEXT NOT NULL)"),
        _initialize_sqlite_db("Constitution DB", settings.CONSTITUTION_DB, constitution_table_sql),
        _initialize_sqlite_db("Phrase Expander DB", settings.PHRASE_EXPANDER_DB, phrase_expander_table_sql),
        
        # ==============================================================================
        #      【【【 2. Add the task to create the new Search Archive DB 】】】
        # ==============================================================================
        _initialize_sqlite_db("Search Archive DB", settings.SEARCH_ARCHIVE_DB, search_archive_table_sql),
        
        setup_vector_database()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("="*70 + "\n")

    all_successful = True
    for result in results:
        if isinstance(result, Exception) or result is False:
            all_successful = False
            if isinstance(result, Exception):
                print(f"❌ {Colors.RED}[DB Manager] A task failed with an exception: {result}{Colors.ENDC}")
                traceback.print_exc()
    
    if not all_successful:
        print(f"❌ {Colors.RED}[DB Manager] One or more database initializations failed.{Colors.ENDC}")

    return all_successful

async def _initialize_sqlite_db(db_name: str, db_path: Path, create_sql: str) -> bool:
    """
    [ASYNC HELPER] Initializes a single SQLite database table.
    """
    try:
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute(create_sql)
            await conn.commit()
        print(f"✅ {Colors.GREEN}[DB Manager] {db_name} is ready. (Path: {db_path}){Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[DB Error] {db_name} initialization failed: {e}{Colors.ENDC}")
        traceback.print_exc()
        return False

async def _setup_voice_transcripts_db_with_migration() -> bool:
    """
    [ASYNC HELPER] Initializes the Voice Transcripts DB and ensures the
    'session_id' column exists for meeting recordings.
    """
    db_name = "Voice Transcripts DB"
    db_path = settings.VOICE_TRANSCRIPTS_DB
    table_name = settings.VOICE_TRANSCRIPTS_TABLE_NAME
    try:
        async with aiosqlite.connect(str(db_path)) as conn:
            # Step 1: Create the table if it doesn't exist (original logic)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    language_detected TEXT NOT NULL,
                    transcribed_text TEXT NOT NULL
                )""")

            # Step 2: Try to add the new session_id column
            try:
                await conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN session_id TEXT"
                )
                print(f"   -> [DB Migration] Added 'session_id' column to transcripts table.")
            except aiosqlite.OperationalError as e:
                # This error is expected if the column already exists. We can ignore it.
                if "duplicate column name" not in str(e):
                    raise # Re-raise any other unexpected errors

            await conn.commit()
        print(f"✅ {Colors.GREEN}[DB Manager] {db_name} is ready. (Path: {db_path}){Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[DB Error] {db_name} initialization failed: {e}{Colors.ENDC}")
        traceback.print_exc()
        return False

# In core/database_manager.py

# In core/database_manager.py

async def setup_vector_database():
    """
    【【【 终极修复版 V8 - 诊断与恢复逻辑 】】】
    [ASYNC] Initializes the Qdrant client non-blockingly.
    - Implements the "Temporarily Clear -> Connect -> Restore" logic for proxies.
    - This ensures local Qdrant connection succeeds without interfering with Google API's proxy needs.
    """
    def _sync_setup_qdrant():
        # Step 1: 诊断并备份原始代理设置
        print(">> [Qdrant Connect] Starting connection diagnostic for local DB...")
        original_proxies = {}
        proxy_keys = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        
        for key in proxy_keys:
            if key in os.environ:
                original_proxies[key] = os.environ.get(key)

        # Step 2: 临时清除代理，以确保能连接到本地的 Qdrant
        if original_proxies:
            print("   -> [Proxy Manager] Temporarily clearing system proxies for local connection...")
            for key in original_proxies.keys():
                del os.environ[key]
        else:
            print("   -> [Proxy Manager] No system proxies found. Proceeding directly.")

        try:
            # Step 3: 在没有代理的环境下，执行连接 Qdrant 的操作
            print(">> [Vector DB] Connecting to local Qdrant service (localhost:6333)...")
            state.QDRANT_CLIENT = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
            # 检查和创建 collections 的逻辑保持不变
            collections = [c.name for c in state.QDRANT_CLIENT.get_collections().collections]
            def create_if_missing(collection_name):
                if collection_name not in collections:
                    print(f"   -> [Vector DB] Collection '{collection_name}' not found. Creating...")
                    state.QDRANT_CLIENT.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(size=settings.VECTOR_DIMENSION, distance=models.Distance.COSINE)
                    )
            create_if_missing(settings.QDRANT_COLLECTION_NAME)
            create_if_missing(settings.QDRANT_RISK_ANALYSIS_COLLECTION)
            create_if_missing(settings.QDRANT_CONSTITUTION_COLLECTION)

            state.IS_QDRANT_DB_READY = True
            print(f"✅ {Colors.GREEN}[Vector DB] Qdrant client connection successful!{Colors.ENDC}")
            return True

        except Exception as e:
            print(f"❌ {Colors.RED}[Vector DB Error] Initialization failed. Is Qdrant running? Error: {e}{Colors.ENDC}")
            state.IS_QDRANT_DB_READY = False
            traceback.print_exc()
            return False

        finally:
            # Step 4: 无论连接成功还是失败，都必须恢复原始的代理设置！
            if original_proxies:
                print("   -> [Proxy Manager] Restoring original system proxy settings...")
                for key, value in original_proxies.items():
                    os.environ[key] = value
                print("   -> [Proxy Manager] Proxies restored.")

    return await asyncio.to_thread(_sync_setup_qdrant)
# ==============================================================================
#      【【【 ADD THIS ENTIRE NEW BLOCK TO THE END OF THE FILE 】】】
# ==============================================================================

# Import the necessary modules at the top of core/database_manager.py
from core import ai_services
from utils.helpers import get_local_time_str, clean_text

async def log_daily_record(event_type: str, original_text: str, processed_text: str, meta_prompt: str = ""):
    """
    [ASYNC BG TASK] Logs an operation to the daily records database and
    generates an AI summary for it in the background.
    """
    print(f">> [Async Log] Recording '{event_type}' event to daily records DB...")
    try:
        # 1. Generate AI Summary Asynchronously
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
        # We now await the async version of the AI task
        ai_summary = await ai_services.run_ai_task(summary_prompt)
        if not ai_summary:
            ai_summary = "AI summary generation failed."  # Fallback

        # 2. Save to Database Asynchronously
        async with aiosqlite.connect(str(settings.DAILY_RECORDS_DB)) as conn:
            await conn.execute(
                f"""
                INSERT INTO {settings.DAILY_RECORDS_TABLE_NAME}
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
            await conn.commit()
            
        print(f"✅ {Colors.GREEN}[Async Log Complete] Saved '{event_type}' event to daily logs.{Colors.ENDC}")

    except Exception as e:
        print(f"❌ {Colors.RED}[Async Log Error] Failed to log daily record: {e}{Colors.ENDC}")
        traceback.print_exc()
# ==============================================================================
#      【【【 END OF ADDITION 】】】
# ==============================================================================
