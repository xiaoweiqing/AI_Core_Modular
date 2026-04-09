#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
#      灵感记忆系统 (MuseBox) v1.4 - 最终稳定版
# ===================================================================================
# 功能:
# - 【【【 v1.4 核心修复: 解决了多线程环境下异步事件循环的传递问题 】】】
# - 采用文件信号触发，稳定且无冲突。
# - AI 驱动，自动对内容进行分类、摘要和结构化信息提取。
# - ...
# ===================================================================================

import os
import sys
import json
import sqlite3
import re
import time
import subprocess
import hashlib
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- 核心AI与数据库库 ---
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Watchdog 库用于文件系统监控 ---
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ... (前面的代码，直到信号处理部分，都完全一样)
# --- 代理与离线模式设置 ---
for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]
os.environ['HF_HUB_OFFLINE'] = '1'
# ==============================================================================
# 0. 用户配置区
# ==============================================================================
IPC_DIR = Path.home() / ".musebox_ipc"
TRIGGER_FILE = IPC_DIR / "capture_inspiration"
STORAGE_DIR = Path.home() / "MuseBox_Storage"
DB_FILE = STORAGE_DIR / "musebox_memory.db"
MARKDOWN_NOTES_DIR = STORAGE_DIR / "灵感笔记"
# --- 模型与数据库配置 ---
llm = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_RAG_ENABLED = False
QDRANT_COLLECTION_NAME = "musebox_inspirations_v1"
WORKER_COUNT = 1
# ==============================================================================
# 1. 核心AI指令 (Prompt)
# ==============================================================================
MUSE_PROCESSOR_PROMPT = """
# 角色
你是一个顶级的知识管理专家和信息架构师。你的任务是分析一段非结构化的文本，并将其转化为一份高质量、结构清晰的知识卡片。

# 核心规则
1.  **内容类型判断 (content_type)**: 首先，准确判断文本的类型。可选值包括：["文章片段", "代码片段", "个人笔记", "对话记录", "待办事项", "网页链接", "其他"]。
2.  **标题生成 (title)**: 为这段内容生成一个简洁、精炼、能概括核心思想的标题（不超过15个字）。
3.  **标签提取 (tags)**: 提取或生成3-5个最能代表内容核心主题的关键词标签。
4.  **核心摘要 (summary)**: 用1-3句话，对原文内容进行高质量的总结，提炼出其最有价值的核心信息。
5.  **结构化提取 (structured_data)**: 这是最关键的部分。根据内容类型，尽力提取结构化信息。
    - 如果是【文章片段】，尝试提取：`"author": "...", "source": "...", "key_arguments": ["...", "..."]`
    - 如果是【代码片段】，尝试提取：`"language": "...", "functionality": "...", "dependencies": ["...", "..."]`
    - 如果是其他类型，可以为空对象 `{}` 或自定义提取。
6.  **忠于原文**: 所有分析都必须基于原文，不允许凭空捏造。

# 输入材料
【原始文本】
{raw_text}

# 输出格式 (严格遵循此JSON结构，不要输出任何额外文字)
```json
{{
  "title": "...",
  "content_type": "...",
  "tags": ["...", "...", "..."],
  "summary": "...",
  "structured_data": {{}}
}}
"""
# ==============================================================================
# 2. 系统设置与辅助函数
# ==============================================================================
def setup_systems():
    """统一初始化所有外部系统（数据库，AI模型等）。"""
    print("="*10 + " 灵感记忆系统 (MuseBox) v1.4 正在启动 " + "="*10)
    STORAGE_DIR.mkdir(exist_ok=True)
    MARKDOWN_NOTES_DIR.mkdir(exist_ok=True)
    if not (setup_local_database() and setup_local_llm() and setup_qdrant_and_embedding()):
        input("!! 核心系统初始化失败，请检查错误信息后按回车键退出。")
        return False
    return True

def setup_local_database():
    print(f">> [DB] 正在初始化本地记忆库: {DB_FILE.name}...")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS inspirations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_time TEXT,
                title TEXT,
                content_type TEXT,
                tags TEXT,
                summary TEXT,
                structured_data_json TEXT,
                raw_text TEXT,
                text_hash TEXT UNIQUE,
                markdown_path TEXT
            )
            """)
        print(f"✅ [DB] 记忆库初始化成功。")
        return True
    except Exception as e:
        print(f"❌ [DB] 严重错误: 初始化SQLite失败: {e}")
        return False

def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        local_model_path = Path(__file__).parent / 'all-MiniLM-L6-v2'
        print(f">> [向量] 正在从本地路径加载 embedding 模型: {local_model_path}")
        if not local_model_path.is_dir():
            print(f"❌ [向量] 严重错误: 模型文件夹不存在！ '{local_model_path}'")
            return False
        EMBEDDING_MODEL = SentenceTransformer(str(local_model_path))
        vector_size = EMBEDDING_MODEL.get_sentence_embedding_dimension()
        
        print(">> [向量] 正在连接到向量数据库 (localhost:6333)...")
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        
        collections = [c.name for c in QDRANT_CLIENT.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            print(f">> [向量] 集合 '{QDRANT_COLLECTION_NAME}' 不存在，正在创建...")
            QDRANT_CLIENT.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
        
        count = QDRANT_CLIENT.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
        print(f"✅ [向量] 成功连接！当前灵感库中有 {count} 个向量记忆。")
        IS_RAG_ENABLED = True
        return True
    except Exception as e:
        print(f"❌ [向量] 严重错误: 无法连接或设置 Qdrant: {e}")
        IS_RAG_ENABLED = False
        return False

def setup_local_llm():
    global llm
    try:
        print(f">> [AI] 正在连接到本地模型 API at [http://127.0.0.1:8087/v1]...")
        llm = ChatOpenAI(
            openai_api_base="http://127.0.0.1:8087/v1", openai_api_key="na",
            model_name="local", temperature=0.2, max_tokens=4096, request_timeout=300
        )
        llm.invoke("Hi")
        print(f"✅ [AI] 本地大语言模型连接成功。")
        return True
    except Exception as e:
        print(f"❌ [AI] 严重错误: 连接本地模型失败: {e}")
        return False

def get_content_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def clean_json_response(text: str) -> str:
    match = re.search(r"json\s*(.*?)\s*", text, re.DOTALL)
    if match: return match.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start: return text[start:end+1]
    return "{}"

# ==============================================================================
# 3. 核心处理流程
# ==============================================================================
async def process_inspiration_text(raw_text):
    """完整的灵感处理流水线"""
    print("\n" + "-"*20 + f" 🔥 新灵感捕捉 [{datetime.now().strftime('%H:%M:%S')}] " + "-"*20)
    text_hash = get_content_hash(raw_text)
    with sqlite3.connect(DB_FILE) as conn:
        if conn.execute("SELECT id FROM inspirations WHERE text_hash = ?", (text_hash,)).fetchone():
            print(">> [去重] 这条灵感已经存在于记忆库中，已跳过。")
            return
    print(">> [AI分析] 正在请求AI对内容进行解析和结构化...")
    prompt = MUSE_PROCESSOR_PROMPT.format(raw_text=raw_text)
    try:
        response_text = await llm.ainvoke(prompt)
        ai_json_str = clean_json_response(response_text.content)
        ai_data = json.loads(ai_json_str)
        print("✅ [AI分析] 成功提取结构化信息！")
    except Exception as e:
        print(f"!! [AI分析] AI处理失败: {e}。将仅保存原文。")
        ai_data = {"title": raw_text[:20] + "...", "content_type": "未分类", "tags": [], "summary": "AI分析失败，请手动整理。", "structured_data": {}}
    print(">> [笔记生成] 正在创建清晰的 Markdown 笔记...")
    md_path = create_markdown_note(raw_text, ai_data)
    print(f"✅ [笔记生成] 笔记已保存到: {md_path.name}")
    print(">> [数据库] 正在将所有信息存入本地记忆库...")
    db_id = save_to_sqlite(raw_text, text_hash, ai_data, str(md_path))
    if db_id:
        print(f"✅ [数据库] 灵感已成功存档！数据库ID: {db_id}")
        asyncio.create_task(vectorize_inspiration(db_id, raw_text, ai_data))
    print("-" * (62 + len(datetime.now().strftime('%H:%M:%S'))))

def create_markdown_note(raw_text, ai_data):
    title = ai_data.get("title", "未命名灵感")
    safe_filename = re.sub(r'[\/*?:"<>|]', "", title)
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_filename}.md"
    filepath = MARKDOWN_NOTES_DIR / filename
    tags_str = ' '.join([f'`{tag}`' for tag in ai_data.get("tags", [])])
    content = f"""# {title}\n类型: {ai_data.get("content_type", "N/A")}\n标签: {tags_str}\n捕获时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## 核心摘要\n{ai_data.get("summary", "N/A")}\n\n## 结构化信息\n```json\n{json.dumps(ai_data.get("structured_data", {}), indent=2, ensure_ascii=False)}\n```\n\n## 原始文本\n---\n{raw_text}\n"""
    with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
    return filepath

def save_to_sqlite(raw_text, text_hash, ai_data, md_path):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            params = (datetime.now(timezone(timedelta(hours=8))).isoformat(), ai_data.get("title"), ai_data.get("content_type"), json.dumps(ai_data.get("tags"), ensure_ascii=False), ai_data.get("summary"), json.dumps(ai_data.get("structured_data"), ensure_ascii=False, indent=2), raw_text, text_hash, md_path)
            cursor = conn.execute("INSERT INTO inspirations (capture_time, title, content_type, tags, summary, structured_data_json, raw_text, text_hash, markdown_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
            return cursor.lastrowid
    except sqlite3.IntegrityError: return None
    except Exception as e:
        print(f"!! [数据库] 写入SQLite时出错: {e}")
        return None

async def vectorize_inspiration(db_id, raw_text, ai_data):
    if not IS_RAG_ENABLED: return
    print(f"   -> [后台向量化] 正在为 '{ai_data.get('title')}' 创建向量记忆...")
    try:
        text_to_embed = f"摘要: {ai_data.get('summary')}\n\n原文: {raw_text}"
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, EMBEDDING_MODEL.encode, text_to_embed)
        payload = {"db_id": db_id, "title": ai_data.get("title"), "summary": ai_data.get("summary")}
        await loop.run_in_executor(None, lambda: QDRANT_CLIENT.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[models.PointStruct(id=db_id, vector=vector.tolist(), payload=payload)], wait=True))
        print(f"   ✅ [后台向量化] 向量记忆创建成功！")
    except Exception as e:
        print(f"   !! [后台向量化] 向量化时出错: {e}")

# ==============================================================================
# 4. 主程序、信号处理与启动流程  <<< 【【【 核心修复区 】】】
# ==============================================================================
is_processing_task = False

# <<< MODIFIED: 函数现在接收 loop 对象作为参数
def trigger_capture_task(loop):
    """由文件信号触发的任务函数"""
    global is_processing_task
    if is_processing_task:
        print("\n>> [系统] 正在处理上一个灵感，请稍候...")
        return
    is_processing_task = True
    try:
        clipboard_content = subprocess.check_output(['xclip', '-o', '-selection', 'clipboard'], text=True)
        if clipboard_content and not clipboard_content.isspace():
            # <<< MODIFIED: 直接使用传入的 loop，而不是 asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(process_inspiration_text(clipboard_content), loop)
        else:
            print("\n>> [系统] 剪贴板内容为空，已忽略。")
    except FileNotFoundError:
        print("\n!! [错误] 未找到 'xclip' 工具。请在您的Linux系统上安装它 (例如: sudo dnf install xclip)")
    except Exception as e:
        print(f"\n!! [错误] 捕捉灵感时发生未知错误: {e}")
    finally:
        time.sleep(1)
        is_processing_task = False

# <<< MODIFIED: 处理器在初始化时需要接收 loop 对象
class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop  # <<< NEW: 存储 loop 的引用

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path) == TRIGGER_FILE:
            print(f"\n>> [信号] 收到任务信号: {TRIGGER_FILE.name}")
            # <<< MODIFIED: 调用任务时，把存储的 loop 传进去
            trigger_capture_task(self.loop)
            try:
                time.sleep(0.2) 
                os.unlink(event.src_path)
            except OSError:
                pass

def main():
    """主函数，用于设置和运行事件循环及文件监听器"""
    if not setup_systems(): return
    if not WATCHDOG_AVAILABLE:
        input("!! [严重错误] watchdog 库未安装！请运行 'pip install watchdog' 后重试。")
        sys.exit(1)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def run_loop_in_thread(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    thread = threading.Thread(target=run_loop_in_thread, args=(loop,), daemon=True)
    thread.start()

    IPC_DIR.mkdir(exist_ok=True)
    if TRIGGER_FILE.exists(): TRIGGER_FILE.unlink()

    # <<< MODIFIED: 创建处理器实例时，把 loop 作为参数传进去
    event_handler = FileTriggerHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, str(IPC_DIR), recursive=False)
    observer.start()

    full_trigger_path = TRIGGER_FILE.resolve()
    print("\n" + "="*70)
    print("  ✅ [系统就绪] MuseBox 后台服务已启动 (最终稳定版)")
    print("  请在您的系统设置中，将喜欢的快捷键绑定到下面的【完整命令】：")
    print(f"\n    touch {full_trigger_path}\n")
    print("  例如: Alt + S  ->  (绑定命令)  ->  复制并粘贴上面的完整命令")
    print("\n  按 Ctrl+C 即可安全退出程序。")
    print("="*70)
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [系统] 收到退出指令，正在关闭...")
    finally:
        observer.stop()
        observer.join()
        loop.call_soon_threadsafe(loop.stop)
        print(">> [系统] 已安全退出。")

if __name__ == "__main__":
    main()