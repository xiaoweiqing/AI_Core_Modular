#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
#      灵感记忆系统 (MuseBox) v2.4 - 通知增强版
# ===================================================================================
# 功能:
# - 【【【 v2.4 新增: 在后台任务处理完毕后，发送桌面通知 】】】
# - 【 v2.3 核心修复: 重构为纯异步架构，解决AI调用卡死问题 】】】
# - 实现了“秒存模式”，用户体验零延迟。
# ===================================================================================

import os
import sys
import json
import sqlite3
import re
import asyncio
import hashlib
import subprocess  # <<< 【新增】
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- 核心AI与数据库库 ---
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- 代理与离线模式设置 ---
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
__RAW_TEXT__

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
    print("=" * 10 + " 灵感记忆系统 (MuseBox) v2.4 正在启动 " + "=" * 10)
    STORAGE_DIR.mkdir(exist_ok=True)
    MARKDOWN_NOTES_DIR.mkdir(exist_ok=True)
    if not (
        setup_local_database() and setup_local_llm() and setup_qdrant_and_embedding()
    ):
        input("!! 核心系统初始化失败，请检查错误信息后按回车键退出。")
        return False
    return True


def setup_local_database():
    print(f">> [DB] 正在初始化本地记忆库: {DB_FILE.name}...")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            # 创建主表
            cursor.execute(
                """
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
            """
            )

            # --- 【【【 核心修改：检查并自动添加新列 】】】 ---
            print(">> [DB] 正在检查数据库结构...")
            columns = [
                info[1]
                for info in cursor.execute("PRAGMA table_info(inspirations)").fetchall()
            ]

            if "ai_raw_output_json" not in columns:
                print("   -> [DB] 正在添加 'ai_raw_output_json' 列...")
                cursor.execute(
                    "ALTER TABLE inspirations ADD COLUMN ai_raw_output_json TEXT"
                )

            if "is_reviewed" not in columns:
                print("   -> [DB] 正在添加 'is_reviewed' 列...")
                cursor.execute(
                    "ALTER TABLE inspirations ADD COLUMN is_reviewed INTEGER DEFAULT 0"
                )

            if "task_type" not in columns:
                print("   -> [DB] 正在添加 'task_type' 列...")
                cursor.execute(
                    "ALTER TABLE inspirations ADD COLUMN task_type TEXT DEFAULT 'knowledge_card'"
                )

            conn.commit()
            print(f"✅ [DB] 记忆库结构已是最新，初始化成功。")
            return True
    except Exception as e:
        print(f"❌ [DB] 严重错误: 初始化或升级SQLite失败: {e}")
        return False


def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = current_dir / "all-MiniLM-L6-v2"

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
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        count = QDRANT_CLIENT.count(
            collection_name=QDRANT_COLLECTION_NAME, exact=True
        ).count
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
            openai_api_base="http://127.0.0.1:8087/v1",
            openai_api_key="na",
            model_name="local",
            temperature=0.2,
            max_tokens=4096,
            request_timeout=120,
            max_retries=1,
        )
        llm.invoke("Hi")
        print(f"✅ [AI] 本地大语言模型连接成功。")
        return True
    except Exception as e:
        print(f"❌ [AI] 严重错误: 连接本地模型失败: {e}")
        print("   -> 请确保您的本地AI模型服务 (如Ollama) 正在运行。")
        return False


def get_content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_json_response(text: str) -> str:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return "{}"


# --- 【【【 新增：原生通知函数 】】】 ---
def safe_notification(title, message):
    """
    使用 notify-send 命令发送桌面通知。
    """
    try:
        # -a 指定了应用名称, -t 指定了通知显示时间(毫秒)
        subprocess.run(
            ["notify-send", title, message, "-a", "MuseBox", "-t", "5000"], check=True
        )
    except FileNotFoundError:
        # 如果系统中没有 notify-send，则静默失败
        pass
    except Exception as e:
        print(f"!! [通知] 发送通知时出错: {e}")


# ==============================================================================
# 3. 核心处理流程
# ==============================================================================
def instant_save_raw_text(raw_text):
    text_hash = get_content_hash(raw_text)
    try:
        with sqlite3.connect(DB_FILE) as conn:
            if conn.execute(
                "SELECT id FROM inspirations WHERE text_hash = ?", (text_hash,)
            ).fetchone():
                print(">> [去重] 这条灵感已存在，跳过。")
                return None
            params = (
                datetime.now(timezone(timedelta(hours=8))).isoformat(),
                raw_text,
                text_hash,
                "处理中...",
                "处理中...",
            )
            cursor = conn.execute(
                "INSERT INTO inspirations (capture_time, raw_text, text_hash, title, summary) VALUES (?, ?, ?, ?, ?)",
                params,
            )
            conn.commit()
            print(f"✅ 灵感已秒存！数据库ID: {cursor.lastrowid}，后台处理中...")
            return cursor.lastrowid
    except Exception as e:
        print(f"!! [秒存] 写入数据库时发生初始错误: {e}")
        return None


async def process_and_update_in_background(db_id, raw_text):
    print(f"   -> [后台处理] 正在为 ID:{db_id} 的灵感请求AI分析...")
    prompt = MUSE_PROCESSOR_PROMPT.replace("__RAW_TEXT__", raw_text)

    ai_data = None
    ai_json_str = "{}"  # 默认值，防止AI调用失败时出错

    try:
        response_text = await asyncio.wait_for(llm.ainvoke(prompt), timeout=300.0)
        ai_json_str = clean_json_response(response_text.content)
        ai_data = json.loads(ai_json_str)
        print(f"   -> [后台处理] AI分析完成: “{ai_data.get('title')}”")
    except asyncio.TimeoutError:
        print(
            f"   !! [后台处理] 严重错误: AI模型在5分钟内未响应 (Timeout)！ID:{db_id} 的分析任务已中断。"
        )
    except Exception as e:
        print(f"   !! [后台处理] AI分析失败: {e}。")

    if ai_data is None:
        ai_data = {
            "title": raw_text[:20] + "...",
            "content_type": "未分类",
            "tags": [],
            "summary": "错误：AI分析失败或超时",
            "structured_data": {},
        }

    loop = asyncio.get_running_loop()
    md_path = await loop.run_in_executor(None, create_markdown_note, raw_text, ai_data)
    print(f"   -> [后台处理] Markdown笔记已创建: {md_path.name}")

    # 将 ai_json_str 传递给 update_database_record
    await loop.run_in_executor(
        None, update_database_record, db_id, ai_data, md_path, ai_json_str
    )

    await vectorize_inspiration(db_id, raw_text, ai_data)

    # --- 【【【 新增：在任务结束时调用通知 】】】 ---
    notification_title = "灵感处理完成"
    notification_message = f"“{ai_data.get('title', '...')[:40]}” 已被成功处理并归档。"
    # 使用 run_in_executor 来运行同步的通知函数，避免阻塞
    await loop.run_in_executor(
        None, safe_notification, notification_title, notification_message
    )


def update_database_record(db_id, ai_data, md_path, ai_raw_json_str):  # <<< 新增参数
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # --- 【【【 核心修改：在 UPDATE 语句中加入新列 】】】 ---
            params = (
                ai_data.get("title"),
                ai_data.get("content_type"),
                json.dumps(ai_data.get("tags"), ensure_ascii=False),
                ai_data.get("summary"),
                json.dumps(
                    ai_data.get("structured_data"), ensure_ascii=False, indent=2
                ),
                str(md_path),
                ai_raw_json_str,  # <<< 新增的值
                db_id,
            )
            conn.execute(
                """
                UPDATE inspirations 
                SET title=?, content_type=?, tags=?, summary=?, structured_data_json=?, markdown_path=?, ai_raw_output_json=? 
                WHERE id=?
            """,
                params,
            )
            conn.commit()
        print(f"   -> [后台处理] 数据库记录 ID:{db_id} 已更新。")
    except Exception as e:
        print(f"   !! [后台处理] 更新数据库时出错: {e}")


def create_markdown_note(raw_text, ai_data):
    title = ai_data.get("title", "未命名灵感")
    safe_filename = re.sub(r'[\/*?:"<>|]', "", title)
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_filename}.md"
    filepath = MARKDOWN_NOTES_DIR / filename
    tags_str = " ".join([f"`{tag}`" for tag in ai_data.get("tags", [])])
    content = f"""# {title}\n类型: {ai_data.get("content_type", "N/A")}\n标签: {tags_str}\n捕获时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## 核心摘要\n{ai_data.get("summary", "N/A")}\n\n## 结构化信息\n```json\n{json.dumps(ai_data.get("structured_data", {}), indent=2, ensure_ascii=False)}\n```\n\n## 原始文本\n---\n{raw_text}\n"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


async def vectorize_inspiration(db_id, raw_text, ai_data):
    if not IS_RAG_ENABLED:
        return
    print(f"   -> [后台向量化] 正在为 ID:{db_id} 创建向量记忆...")
    try:
        text_to_embed = f"摘要: {ai_data.get('summary')}\n\n原文: {raw_text}"
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, EMBEDDING_MODEL.encode, text_to_embed)
        payload = {
            "db_id": db_id,
            "title": ai_data.get("title"),
            "summary": ai_data.get("summary"),
        }
        await loop.run_in_executor(
            None,
            lambda: QDRANT_CLIENT.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=db_id, vector=vector.tolist(), payload=payload
                    )
                ],
                wait=True,
            ),
        )
        print(f"   ✅ [后台处理] ID:{db_id} 全部流程处理完毕。")
    except Exception as e:
        print(f"   !! [后台向量化] 向量化时出错: {e}")


# ==============================================================================
# 4. 主程序、信号处理与启动流程 (纯异步重构)
# ==============================================================================
async def trigger_capture_task():
    """
    这是一个异步版本的任务触发器
    """
    print(f"\n>> [信号] 收到任务信号: {TRIGGER_FILE.name}")
    try:
        process = await asyncio.create_subprocess_exec(
            "xclip",
            "-o",
            "-selection",
            "clipboard",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise FileNotFoundError(f"xclip 执行失败: {stderr.decode()}")

        clipboard_content = stdout.decode("utf-8")

        if clipboard_content and not clipboard_content.isspace():
            loop = asyncio.get_running_loop()
            db_id = await loop.run_in_executor(
                None, instant_save_raw_text, clipboard_content
            )

            if db_id:
                asyncio.create_task(
                    process_and_update_in_background(db_id, clipboard_content)
                )
        else:
            print("\n>> [系统] 剪贴板内容为空，已忽略。")

    except FileNotFoundError:
        print(
            "\n!! [错误] 未找到 'xclip' 工具。请确保已安装 (sudo dnf install xclip)。"
        )
    except Exception as e:
        print(f"\n!! [错误] 捕捉灵感时发生未知错误: {e}")


async def watch_for_trigger():
    """
    一个简单的异步文件监视器
    """
    IPC_DIR.mkdir(exist_ok=True)
    if TRIGGER_FILE.exists():
        TRIGGER_FILE.unlink()

    print(f">> [监控中] 正在监控触发文件: {TRIGGER_FILE}")
    while True:
        try:
            if TRIGGER_FILE.exists():
                asyncio.create_task(trigger_capture_task())
                await asyncio.sleep(0.2)
                if TRIGGER_FILE.exists():
                    TRIGGER_FILE.unlink()
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"!! [监控循环] 发生错误: {e}")
            await asyncio.sleep(5)


async def main_async():
    if not setup_systems():
        return

    full_trigger_path = TRIGGER_FILE.resolve()
    print("\n" + "=" * 70)
    print("  ✅ [系统就绪] MuseBox 后台服务已启动 (通知增强版 v2.4)")
    print("  请在您的系统设置中，将喜欢的快捷键绑定到下面的【完整命令】：")
    print(f"\n    touch {full_trigger_path}\n")
    print("\n  按 Ctrl+C 即可安全退出程序。")
    print("=" * 70)

    try:
        await watch_for_trigger()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n>> [系统] 收到退出指令，正在关闭...")
    finally:
        print(">> [系统] 已安全退出。")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
