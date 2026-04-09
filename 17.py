#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
#      AI Workbench v1.1 - 高响应异步版
# ===================================================================================
# 版本说明:
# - 【【【 v1.1 核心架构修正: 解决任务拥堵问题 】】】
#   - 真正异步: 重构任务调度逻辑。“秒存”操作在主线程中瞬间完成，不再排队。
#   - 响应优先: 只有真正耗时的AI调用和向量化任务进入后台线程池，确保翻译等快速操作的即时响应。
#   - 清理输出: 优化了命令行日志，使其清晰、准确，不再输出混乱信息。
#   - 修复所有已知Bug (包括 `clean_text` 和 `pyperclip` 的 a'name not defined' 错误)。
# - 【 v1.0 终极整合: MuseBox + Local Translator 】
# ===================================================================================

import os
import sys
import json
import sqlite3
import re
import asyncio
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- 核心AI与数据库库 ---
import pyperclip
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
IPC_DIR = Path(__file__).parent / ".ai_workbench_ipc"
TRIGGER_FILES = {
    "input": IPC_DIR / "capture_input",
    "output": IPC_DIR / "capture_output",
    "translate_en": IPC_DIR / "translate_to_en",
    "translate_zh": IPC_DIR / "translate_to_zh",
}
STORAGE_DIR = Path(__file__).parent / "AI_Workbench_Storage"
DB_FILE = STORAGE_DIR / "workbench_memory.db"
MARKDOWN_NOTES_DIR = STORAGE_DIR / "知识卡片"

# --- 模型与数据库配置 ---
llm = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_RAG_ENABLED = False
QDRANT_COLLECTION_NAME = "ai_workbench_v1"


class StateManager:
    def __init__(self):
        self.last_input_id = None


state_manager = StateManager()

# ==============================================================================
# 1. 核心AI指令 (Prompts)
# ==============================================================================
PAIR_ANALYSIS_PROMPT = """
# 角色: AI训练数据标注专家
# 任务: 分析“输入/输出”数据对，提取核心价值。
# 核心规则:
1.  **任务类型判断 (task_type)**: ["代码生成", "文本润色", "翻译", "创意写作", "摘要总结", "问答", "数据提取", "其他"]。
2.  **标题生成 (title)**: 为交互生成精炼标题（不超过20字）。
3.  **标签提取 (tags)**: 3-5个核心关键词标签。
4.  **核心摘要 (summary)**: 总结“解决了什么问题”及“得出了什么关键结论”。
5.  **质量评估 (quality_rating)**: {"score": 整数(1-5), "reason": "简短理由"}。
# 输入材料:
【输入/问题】
__INPUT_TEXT__
【输出/答案】
__OUTPUT_TEXT__
# 输出格式 (严格遵循此JSON结构):
```json
{"title": "...", "task_type": "...", "tags": ["...", "..."], "summary": "...", "quality_rating": {"score": 5, "reason": "..."}}
"""


# ==============================================================================
# 2. 系统设置与辅助函数
# ==============================================================================
def setup_systems():
    print("=" * 10 + " AI Workbench v1.1 - 高响应异步版正在启动 " + "=" * 10)
    STORAGE_DIR.mkdir(exist_ok=True)
    MARKDOWN_NOTES_DIR.mkdir(exist_ok=True)
    if not (
        setup_local_database() and setup_local_llm() and setup_qdrant_and_embedding()
    ):
        input("!! 核心系统初始化失败，请检查错误信息后按回车键退出。")
        return False
    return True


def setup_local_database():
    print(f">> [DB] 正在初始化统一记忆库于: {DB_FILE.resolve()}")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_time TEXT NOT NULL,
                record_type TEXT NOT NULL,
                pair_id INTEGER,
                raw_text TEXT NOT NULL,
                processed_text TEXT,
                text_hash TEXT UNIQUE NOT NULL,
                title TEXT, task_type TEXT, tags TEXT, summary TEXT,
                quality_score INTEGER, quality_reason TEXT,
                is_processed INTEGER DEFAULT 0,
                markdown_path TEXT
            )
            """
            )
            conn.commit()
            print(f"✅ [DB] 数据库准备就绪: {DB_FILE.name}")
            return True
    except Exception as e:
        print(f"❌ [DB] 严重错误: 初始化SQLite失败: {e}")
        return False


# 其他 setup 函数保持不变...
def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        current_dir = Path(__file__).parent
        local_model_path = current_dir / "all-MiniLM-L6-v2"
        if not local_model_path.is_dir():
            print(f"❌ [向量] 严重错误: 模型文件夹不存在！ '{local_model_path}'")
            return False
        EMBEDDING_MODEL = SentenceTransformer(str(local_model_path))
        vector_size = EMBEDDING_MODEL.get_sentence_embedding_dimension()
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333, timeout=20)
        QDRANT_CLIENT.get_collections()
        collections = [c.name for c in QDRANT_CLIENT.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            QDRANT_CLIENT.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        count = QDRANT_CLIENT.count(
            collection_name=QDRANT_COLLECTION_NAME, exact=False
        ).count
        print(f"✅ [向量] 成功连接！当前记忆库中有约 {count} 个向量。")
        IS_RAG_ENABLED = True
        return True
    except Exception as e:
        print(f"❌ [向量] 严重错误: 无法连接或设置 Qdrant: {e}")
        IS_RAG_ENABLED = False
        return False


def setup_local_llm():
    global llm
    try:
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
        return False


# 【v1.1 修复】确保所有辅助函数都已定义
def get_content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def clean_json_response(text: str) -> str:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return "{}"


def safe_notification(title, message):
    try:
        subprocess.run(
            ["notify-send", title, message, "-a", "AI Workbench", "-t", "5000"],
            check=True,
        )
    except Exception:
        pass


# ==============================================================================
# 3. 核心处理流程 (统一入口)
# ==============================================================================


# 【v1.1 架构修改】秒存函数现在是同步的，以保证即时性
def instant_save(raw_text: str, record_type: str):
    # ... (此函数逻辑不变, 但调用方式改变)
    text_hash = get_content_hash(raw_text)
    db_id, pair_id = None, None
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            if (
                record_type != "translation"
                and cursor.execute(
                    "SELECT id FROM records WHERE text_hash = ?", (text_hash,)
                ).fetchone()
            ):
                print(">> [去重] 这条记录已存在，跳过。")
                return None, None
            current_time = datetime.now(timezone(timedelta(hours=8))).isoformat()
            if record_type == "input":
                params = (current_time, record_type, raw_text, text_hash, "等待输出...")
                cursor.execute(
                    "INSERT INTO records (capture_time, record_type, raw_text, text_hash, title) VALUES (?, ?, ?, ?, ?)",
                    params,
                )
                db_id = cursor.lastrowid
                state_manager.last_input_id = db_id
                print(f"✅ [输入] 已秒存！ID: {db_id}。等待捕获输出...")
                safe_notification("输入已捕获", f"“{raw_text[:30]}...” 已记录。")
            elif record_type == "output":
                pair_id = state_manager.last_input_id
                if pair_id is None:
                    print("!! [警告] 未找到对应的输入。此记录将不会配对。")
                    safe_notification("输出捕获警告", "未找到对应的输入。")
                    return None, None
                print(f">> [输出] 正在与上一个输入 (ID: {pair_id}) 进行配对...")
                params = (
                    current_time,
                    record_type,
                    pair_id,
                    raw_text,
                    text_hash,
                    "处理中...",
                )
                cursor.execute(
                    "INSERT INTO records (capture_time, record_type, pair_id, raw_text, text_hash, title) VALUES (?, ?, ?, ?, ?, ?)",
                    params,
                )
                db_id = cursor.lastrowid
                state_manager.last_input_id = None
                print(f"✅ [输出] 已秒存！ID: {db_id}。后台AI分析任务已启动...")
            elif record_type == "translation":
                params = (current_time, record_type, raw_text, text_hash, "翻译中...")
                cursor.execute(
                    "INSERT INTO records (capture_time, record_type, raw_text, text_hash, title) VALUES (?, ?, ?, ?, ?)",
                    params,
                )
                db_id = cursor.lastrowid
                print(f"✅ [翻译] 原文已秒存！ID: {db_id}。后台翻译任务已启动...")
            conn.commit()
            return db_id, pair_id
    except Exception as e:
        print(f"!! [秒存] 写入数据库时发生错误: {e}")
        return None, None


# --- 后台任务分发器 ---
async def process_in_background(db_id, raw_text, record_type, pair_id=None, **kwargs):
    if record_type == "output" and pair_id:
        await process_pair_task(db_id, raw_text, pair_id)
    elif record_type == "translation":
        await process_translation_task(db_id, raw_text, kwargs.get("target_language"))


# --- 处理“数据对”的后台任务 ---
# --- 处理“数据对”的后台任务 ---
async def process_pair_task(output_id, output_text, input_id):
    print(
        f"   -> [AI分析] 正在为数据对 (Input:{input_id}, Output:{output_id}) 请求AI分析..."
    )
    try:
        # 从数据库获取输入文本
        with sqlite3.connect(DB_FILE) as conn:
            # 在同步代码块中执行数据库查询
            input_row = conn.execute(
                "SELECT raw_text FROM records WHERE id = ?", (input_id,)
            ).fetchone()

        if not input_row:
            print(f"   !! [后台任务] 严重错误: 找不到 ID 为 {input_id} 的输入记录。")
            return

        input_text = input_row[0]

        # 准备 Prompt
        prompt = PAIR_ANALYSIS_PROMPT.replace("__INPUT_TEXT__", input_text).replace(
            "__OUTPUT_TEXT__", output_text
        )

        # 【【【核心修改】】】 使用原生的异步方法 ainvoke
        response_text = await llm.ainvoke(prompt)
        ai_data = json.loads(clean_json_response(response_text.content))
        print(f"   -> [AI分析] 分析完成: “{ai_data.get('title')}”")

        # 更新数据库和文件是IO操作，可以在executor中运行以避免阻塞
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            update_db_and_markdown_for_pair,
            input_id,
            output_id,
            input_text,
            output_text,
            ai_data,
        )

        text_to_embed = (
            f"问题: {input_text}\n答案: {output_text}\n总结: {ai_data.get('summary')}"
        )
        await vectorize_record(output_id, text_to_embed, ai_data)
        safe_notification(
            "数据对处理完成", f"“{ai_data.get('title', '...')[:40]}” 已完全归档。"
        )
    except Exception as e:
        print(f"   !! [后台任务] 处理数据对时失败: {e}")


# --- 处理“翻译”的后台任务 ---
# --- 处理“翻译”的后台任务 ---
async def process_translation_task(db_id, original_text, target_language):
    lang_map = {"en": "英文", "zh": "中文"}
    print(
        f"   -> [翻译->{lang_map.get(target_language)}] 正在为 ID:{db_id} 请求翻译..."
    )
    try:
        if target_language == "en":
            prompt = f"Translate the following text to English. Output only the translated text, without any explanations or formatting:\n\n{clean_text(original_text)}"
        else:
            prompt = f"将以下内容翻译成中文，只输出翻译后的纯文本，不要任何解释或格式:\n\n{clean_text(original_text)}"

        # 【【【核心修改】】】 使用原生的异步方法 ainvoke
        response = await llm.ainvoke(prompt)
        translated_text = response.content.strip()

        # pyperclip 是一个快速的同步操作，可以直接调用
        pyperclip.copy(translated_text)
        print(f"   -> [翻译->{lang_map.get(target_language)}] 成功 (已复制):")
        print("---")
        print(translated_text)
        print("---")

        # 更新数据库是IO操作，可以在executor中运行以避免阻塞
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, update_db_for_translation, db_id, translated_text
        )

        text_to_embed = f"原文: {original_text}\n译文: {translated_text}"
        ai_data_for_vector = {
            "title": f"翻译记录: {original_text[:20]}...",
            "summary": translated_text,
        }
        await vectorize_record(db_id, text_to_embed, ai_data_for_vector)

        safe_notification(
            f"翻译到{lang_map.get(target_language)}完成", "结果已复制并已完全归档。"
        )
    except Exception as e:
        print(f"   !! [后台任务] 处理翻译时失败: {e}")


# --- 数据库/文件更新函数 ---
def update_db_and_markdown_for_pair(
    input_id, output_id, input_text, output_text, ai_data
):
    # ... (此函数逻辑不变)
    try:
        title = ai_data.get("title", "未命名数据对")
        safe_filename = re.sub(r'[\/*?:"<>|]', "", title)
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_filename}.md"
        filepath = MARKDOWN_NOTES_DIR / filename
        tags_str = " ".join([f"`{tag}`" for tag in ai_data.get("tags", [])])
        quality = ai_data.get("quality_rating", {})
        content = f"""# {title}\n**任务类型**: {ai_data.get("task_type", "N/A")} | **质量评分**: {quality.get("score", "N/A")}/5\n**标签**: {tags_str}\n\n## 核心摘要\n{ai_data.get("summary", "N/A")}\n\n**AI评估理由**: *{quality.get("reason", "N/A")}*\n\n---\n\n## 问答详情\n### ➡️ 输入 (ID: {input_id})\n```\n{input_text}\n```\n\n### ✅ 输出 (ID: {output_id})\n```\n{output_text}\n```\n"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   -> [归档] Markdown笔记已创建: {filepath.name}")
        with sqlite3.connect(DB_FILE) as conn:
            params_output = (
                title,
                ai_data.get("task_type"),
                json.dumps(ai_data.get("tags"), ensure_ascii=False),
                ai_data.get("summary"),
                quality.get("score"),
                quality.get("reason"),
                1,
                str(filepath),
                output_id,
            )
            conn.execute(
                "UPDATE records SET title=?, task_type=?, tags=?, summary=?, quality_score=?, quality_reason=?, is_processed=1, markdown_path=? WHERE id=?",
                params_output,
            )
            params_input = (
                title,
                ai_data.get("task_type"),
                json.dumps(ai_data.get("tags"), ensure_ascii=False),
                ai_data.get("summary"),
                1,
                str(filepath),
                input_id,
            )
            conn.execute(
                "UPDATE records SET title=?, task_type=?, tags=?, summary=?, is_processed=1, markdown_path=? WHERE id=?",
                params_input,
            )
            conn.commit()
        print(f"   -> [DB] 数据对 (I:{input_id}, O:{output_id}) 已在数据库中完全更新。")
    except Exception as e:
        print(f"   !! [DB/归档] 更新或创建文件时出错: {e}")


def update_db_for_translation(db_id, translated_text):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            params = (translated_text, 1, f"翻译记录", db_id)
            conn.execute(
                "UPDATE records SET processed_text=?, is_processed=?, title=? WHERE id=?",
                params,
            )
            conn.commit()
        print(f"   -> [DB] 翻译记录 ID:{db_id} 已在数据库中更新。")
    except Exception as e:
        print(f"   !! [DB] 更新翻译记录时出错: {e}")


async def vectorize_record(record_id, text_to_embed, ai_data):
    # ... (此函数逻辑不变)
    if not IS_RAG_ENABLED:
        return
    print(f"   -> [向量化] 正在为记录 ID:{record_id} 创建向量...")
    try:
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, EMBEDDING_MODEL.encode, text_to_embed)
        payload = {
            "db_id": record_id,
            "title": ai_data.get("title"),
            "summary": ai_data.get("summary"),
        }
        await loop.run_in_executor(
            None,
            lambda: QDRANT_CLIENT.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=record_id, vector=vector.tolist(), payload=payload
                    )
                ],
                wait=True,
            ),
        )
        print(f"   -> [向量化] ID:{record_id} 向量化成功。")
    except Exception as e:
        print(f"   !! [向量化] 出错: {e}")


# ==============================================================================
# 4. 主程序、信号处理与启动流程
# ==============================================================================
# 【v1.1 架构修改】trigger_task 现在是纯异步的，只调用同步的秒存
async def trigger_task(record_type, **kwargs):
    print(f"\n>> [信号] 收到任务信号: {record_type}")
    try:
        # 剪贴板操作是快速的IO，可以在主线程中执行
        clipboard_content = ""
        try:
            # 优先使用 Wayland 的工具
            proc = await asyncio.create_subprocess_exec(
                "wl-paste",
                "-n",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                clipboard_content = stdout.decode("utf-8").strip()
        except FileNotFoundError:
            pass  # wl-paste 不存在，继续尝试 xclip

        if not clipboard_content:
            try:
                # 回退到 X11 的工具
                proc = await asyncio.create_subprocess_exec(
                    "xclip",
                    "-o",
                    "-selection",
                    "clipboard",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    clipboard_content = stdout.decode("utf-8").strip()
            except FileNotFoundError:
                print("!! [警告] 未找到 wl-paste 或 xclip，剪贴板功能受限。")

        if clipboard_content:
            # 【关键】秒存是同步的，直接调用，瞬间完成
            db_id, pair_id = instant_save(clipboard_content, record_type)

            if db_id:
                # 只有秒存成功，才创建后台任务
                asyncio.create_task(
                    process_in_background(
                        db_id, clipboard_content, record_type, pair_id, **kwargs
                    )
                )
        else:
            print(">> [系统] 剪贴板内容为空，已忽略。")
    except Exception as e:
        print(f"!! [错误] 捕捉时发生未知错误: {e}")


# watch_for_triggers 和 main_async 函数保持不变...
async def watch_for_triggers():
    IPC_DIR.mkdir(exist_ok=True)
    for f in TRIGGER_FILES.values():
        if f.exists():
            f.unlink()
    print(">> [监控中] 正在监控所有工作台信号...")
    while True:
        try:
            if TRIGGER_FILES["input"].exists():
                await trigger_task("input")
                TRIGGER_FILES["input"].unlink()
            if TRIGGER_FILES["output"].exists():
                await trigger_task("output")
                TRIGGER_FILES["output"].unlink()
            if TRIGGER_FILES["translate_en"].exists():
                await trigger_task("translation", target_language="en")
                TRIGGER_FILES["translate_en"].unlink()
            if TRIGGER_FILES["translate_zh"].exists():
                await trigger_task("translation", target_language="zh")
                TRIGGER_FILES["translate_zh"].unlink()
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"!! [监控循环] 发生错误: {e}")
            await asyncio.sleep(5)


async def main_async():
    if not setup_systems():
        return
    print("\n" + "=" * 70)
    print("  ✅ [系统就绪] AI Workbench v1.1 (高响应异步版) 已启动")
    print("  请在系统设置中，为您的快捷键绑定下面的【完整命令】：")
    print(f"\n  捕获 输入 (推荐 Alt+D):    touch {TRIGGER_FILES['input'].resolve()}")
    print(f"  捕获 输出 (推荐 Alt+S):    touch {TRIGGER_FILES['output'].resolve()}")
    print(
        f"  翻译到英文:                touch {TRIGGER_FILES['translate_en'].resolve()}"
    )
    print(
        f"  翻译到中文:                touch {TRIGGER_FILES['translate_zh'].resolve()}\n"
    )
    print("\n  按 Ctrl+C 即可安全退出程序。")
    print("=" * 70)
    try:
        await watch_for_triggers()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n>> [系统] 收到退出指令，正在关闭...")
    finally:
        print(">> [系统] 已安全退出。")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
