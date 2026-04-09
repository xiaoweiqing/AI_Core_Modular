#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
#      灵感记忆系统 (MuseBox) v2.0 - 秒存模式最终版
# ===================================================================================
# 功能:
# - 【【【 v2.0 架构革命 - 秒存模式 】】】
#   - 实现了“先保存，后处理”的异步流程，用户体验零延迟。
#   - 按下快捷键后，原文立刻存入数据库，AI分析在后台进行。
# - 采用文件信号触发，稳定且无冲突。
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

# ... (前面的代码，直到核心处理流程，都完全一样)
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
    print("="*10 + " 灵感记忆系统 (MuseBox) v2.0 正在启动 " + "="*10)
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
            QDRANT_CLIENT.recreate_collection(collection_name=QDRANT_COLLECTION_NAME, vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE))
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
        llm = ChatOpenAI(openai_api_base="http://127.0.0.1:8087/v1", openai_api_key="na", model_name="local", temperature=0.2, max_tokens=4096, request_timeout=300)
        llm.invoke("Hi")
        print(f"✅ [AI] 本地大语言模型连接成功。")
        return True
    except Exception as e:
        print(f"❌ [AI] 严重错误: 连接本地模型失败: {e}")
        return False

def get_content_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def clean_json_response(text: str) -> str:
    match = re.search(r"json\s*(.*?)\s*", text, re.DOTALL)
    if match: return match.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start: return text[start:end+1]
    return "{}"

# ==============================================================================
# 3. 核心处理流程  <<< 【【【 架构重大变更区 】】】
# ==============================================================================

# <<< NEW: 步骤1 - 极速保存原文到数据库
def instant_save_raw_text(raw_text):
    """立刻保存原文和时间戳，返回新记录的ID。"""
    text_hash = get_content_hash(raw_text)
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # 首先检查是否重复
            if conn.execute("SELECT id FROM inspirations WHERE text_hash = ?", (text_hash,)).fetchone():
                print(">> [去重] 这条灵感已存在，跳过。")
                return None
            
            # 插入一条只有基础信息的记录
            params = (
                datetime.now(timezone(timedelta(hours=8))).isoformat(),
                raw_text,
                text_hash,
                "处理中...", # title 占位符
                "处理中..."  # summary 占位符
            )
            cursor = conn.execute(
                "INSERT INTO inspirations (capture_time, raw_text, text_hash, title, summary) VALUES (?, ?, ?, ?, ?)",
                params
            )
            print(f"✅ 灵感已秒存！数据库ID: {cursor.lastrowid}，后台处理中...")
            return cursor.lastrowid
    except Exception as e:
        print(f"!! [秒存] 写入数据库时发生初始错误: {e}")
        return None

# <<< NEW: 步骤2 - 在后台进行AI分析、文件创建和数据更新
async def process_and_update_in_background(db_id, raw_text):
    """获取AI分析结果，并更新数据库和创建文件。"""
    print(f"   -> [后台处理] 正在为 ID:{db_id} 的灵感请求AI分析...")
    prompt = MUSE_PROCESSOR_PROMPT.format(raw_text=raw_text)
    try:
        response_text = await llm.ainvoke(prompt)
        ai_json_str = clean_json_response(response_text.content)
        ai_data = json.loads(ai_json_str)
        print(f"   -> [后台处理] AI分析完成: “{ai_data.get('title')}”")
    except Exception as e:
        print(f"   !! [后台处理] AI分析失败: {e}。将仅保存原文。")
        ai_data = {"title": raw_text[:20] + "...", "content_type": "未分类", "tags": [], "summary": "AI分析失败", "structured_data": {}}

    # 创建Markdown笔记
    md_path = create_markdown_note(raw_text, ai_data)
    print(f"   -> [后台处理] Markdown笔记已创建: {md_path.name}")

    # 更新数据库记录
    try:
        with sqlite3.connect(DB_FILE) as conn:
            params = (
                ai_data.get("title"),
                ai_data.get("content_type"),
                json.dumps(ai_data.get("tags"), ensure_ascii=False),
                ai_data.get("summary"),
                json.dumps(ai_data.get("structured_data"), ensure_ascii=False, indent=2),
                str(md_path),
                db_id
            )
            conn.execute("""
                UPDATE inspirations 
                SET title=?, content_type=?, tags=?, summary=?, structured_data_json=?, markdown_path=?
                WHERE id=?
            """, params)
        print(f"   -> [后台处理] 数据库记录 ID:{db_id} 已更新。")
    except Exception as e:
        print(f"   !! [后台处理] 更新数据库时出错: {e}")
    
    # 最后进行向量化
    await vectorize_inspiration(db_id, raw_text, ai_data)


def create_markdown_note(raw_text, ai_data):
    title = ai_data.get("title", "未命名灵感")
    safe_filename = re.sub(r'[\/*?:"<>|]', "", title)
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_filename}.md"
    filepath = MARKDOWN_NOTES_DIR / filename
    tags_str = ' '.join([f'`{tag}`' for tag in ai_data.get("tags", [])])
    content = f"""# {title}\n类型: {ai_data.get("content_type", "N/A")}\n标签: {tags_str}\n捕获时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## 核心摘要\n{ai_data.get("summary", "N/A")}\n\n## 结构化信息\n```json\n{json.dumps(ai_data.get("structured_data", {}), indent=2, ensure_ascii=False)}\n```\n\n## 原始文本\n---\n{raw_text}\n"""
    with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
    return filepath

async def vectorize_inspiration(db_id, raw_text, ai_data):
    if not IS_RAG_ENABLED: return
    print(f"   -> [后台向量化] 正在为 ID:{db_id} 创建向量记忆...")
    try:
        text_to_embed = f"摘要: {ai_data.get('summary')}\n\n原文: {raw_text}"
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, EMBEDDING_MODEL.encode, text_to_embed)
        payload = {"db_id": db_id, "title": ai_data.get("title"), "summary": ai_data.get("summary")}
        await loop.run_in_executor(None, lambda: QDRANT_CLIENT.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[models.PointStruct(id=db_id, vector=vector.tolist(), payload=payload)], wait=True))
        print(f"   ✅ [后台处理] ID:{db_id} 全部流程处理完毕。")
    except Exception as e:
        print(f"   !! [后台向量化] 向量化时出错: {e}")

# ==============================================================================
# 4. 主程序、信号处理与启动流程
# ==============================================================================
is_processing_task = False

def trigger_capture_task(loop):
    """由文件信号触发的任务函数，现在只负责调度。"""
    global is_processing_task
    if is_processing_task:
        print("\n>> [系统] 正忙，请等待上一个任务完成。")
        return
    is_processing_task = True
    try:
        clipboard_content = subprocess.check_output(['xclip', '-o', '-selection', 'clipboard'], text=True)
        if clipboard_content and not clipboard_content.isspace():
            # <<< MODIFIED: 调用新的两步流程
            # 1. 立刻同步保存，获取ID
            db_id = instant_save_raw_text(clipboard_content)
            # 2. 如果保存成功 (不是重复内容)，则将慢速任务扔到后台
            if db_id:
                asyncio.run_coroutine_threadsafe(process_and_update_in_background(db_id, clipboard_content), loop)
        else:
            print("\n>> [系统] 剪贴板内容为空，已忽略。")
    except FileNotFoundError:
        print("\n!! [错误] 未找到 'xclip' 工具。")
    except Exception as e:
        print(f"\n!! [错误] 捕捉灵感时发生未知错误: {e}")
    finally:
        is_processing_task = False # 锁可以更快释放

class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path) == TRIGGER_FILE:
            print(f"\n>> [信号] 收到任务信号: {TRIGGER_FILE.name}")
            trigger_capture_task(self.loop)
            try:
                time.sleep(0.2); os.unlink(event.src_path)
            except OSError: pass

def main():
    if not setup_systems(): return
    if not WATCHDOG_AVAILABLE:
        input("!! [严重错误] watchdog 库未安装！请运行 'pip install watchdog' 后重试。")
        sys.exit(1)

    try: loop = asyncio.get_running_loop()
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

    event_handler = FileTriggerHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, str(IPC_DIR), recursive=False)
    observer.start()

    full_trigger_path = TRIGGER_FILE.resolve()
    print("\n" + "="*70)
    print("  ✅ [系统就绪] MuseBox 后台服务已启动 (秒存模式)")
    print("  请在您的系统设置中，将喜欢的快捷键绑定到下面的【完整命令】：")
    print(f"\n    touch {full_trigger_path}\n")
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