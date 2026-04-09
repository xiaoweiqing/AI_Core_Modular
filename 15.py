#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
#      灵感记忆系统 (MuseBox) v3.0 - 数据对生产线
# ===================================================================================
# 功能:
# - 【【【 v3.0 核心重构: 引入“输入/输出”数据对捕捉模式 】】】
#   - 双热键系统: 一个快捷键 (`capture_input`) 记录问题/输入，另一个 (`capture_output`) 记录答案/输出。
#   - 自动配对: 记录“输出”时，会自动与上一个“输入”在数据库中关联，形成数据对。
#   - 数据库升级: 全新表结构，支持存储 `input`, `output`, `standalone` 三种记录类型。
#   - 智能分析升级: 当数据对形成时，触发高级AI分析，对“问答质量”和“主题”进行总结。
#   - 状态管理: 系统会记住上一个输入的ID，确保配对准确无误。
#   - 完整反馈: 所有后台任务（AI分析, 向量化, 归档）完成后，发送桌面通知。
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
# 【v3.0 修改】拆分信号文件，以对应 Alt+D 和 Alt+S
IPC_DIR = Path(__file__).parent / ".musebox_ipc"
TRIGGER_FILES = {
    "input": IPC_DIR / "capture_input",
    "output": IPC_DIR / "capture_output",
}
STORAGE_DIR = Path(__file__).parent / "MuseBox_Storage_v3"
DB_FILE = STORAGE_DIR / "musebox_memory_v3.db"  # 使用新数据库文件，与旧版完全隔离
MARKDOWN_NOTES_DIR = STORAGE_DIR / "知识卡片"

# --- 模型与数据库配置 ---
llm = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_RAG_ENABLED = False
QDRANT_COLLECTION_NAME = "musebox_v3"


# 【v3.0 新增】状态管理器，用于记住上一个输入的ID
class StateManager:
    def __init__(self):
        self.last_input_id = None


state_manager = StateManager()

# ==============================================================================
# 1. 核心AI指令 (Prompts)
# ==============================================================================
# 【v3.0 新增】用于分析“数据对”的全新Prompt
PAIR_ANALYSIS_PROMPT = """
# 角色
你是一个顶级的AI训练数据标注专家和知识蒸馏师。你的任务是分析一个“输入/输出”数据对，并提取其核心价值。

# 核心规则
1.  **任务类型判断 (task_type)**: 分析这个数据对解决了什么问题。可选值包括：["代码生成", "文本润色", "翻译", "创意写作", "摘要总结", "问答", "数据提取", "其他"]。
2.  **标题生成 (title)**: 为这个“输入/输出”交互生成一个精炼、概括核心主题的标题（不超过20个字）。
3.  **标签提取 (tags)**: 提取或生成3-5个最能代表这个交互核心主题的关键词标签。
4.  **核心摘要 (summary)**: 用1-3句话，高质量地总结这个交互“解决了什么问题”以及“得出了什么关键结论”。
5.  **质量评估 (quality_rating)**: 作为一个专家，从1到5给这个“输出”的质量打分（1=很差, 5=非常出色），并简要说明理由。
    - `"score": 整数 (1-5)`
    - `"reason": "简短的理由说明"`

# 输入材料
【输入/问题】
__INPUT_TEXT__

【输出/答案】
__OUTPUT_TEXT__

# 输出格式 (严格遵循此JSON结构，不要输出任何额外文字)
```json
{{
  "title": "...",
  "task_type": "...",
  "tags": ["...", "...", "..."],
  "summary": "...",
  "quality_rating": {{
    "score": 5,
    "reason": "..."
  }}
}}
"""


# ==============================================================================
# 2. 系统设置与辅助函数
# ==============================================================================
def setup_systems():
    print("=" * 10 + " 灵感记忆系统 (MuseBox) v3.0 正在启动 " + "=" * 10)
    STORAGE_DIR.mkdir(exist_ok=True)
    MARKDOWN_NOTES_DIR.mkdir(exist_ok=True)
    if not (
        setup_local_database() and setup_local_llm() and setup_qdrant_and_embedding()
    ):
        input("!! 核心系统初始化失败，请检查错误信息后按回车键退出。")
        return False
    return True


def setup_local_database():
    # 【【【 核心修正点在这里 】】】
    # 我们把打印语句修改一下，加入 DB_FILE 变量
    print(f">> [DB] 正在初始化v3.0记忆库于: {DB_FILE.resolve()}")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_time TEXT NOT NULL,
                record_type TEXT NOT NULL,      -- 'input' or 'output'
                pair_id INTEGER,                -- 如果是 'output'，这里是对应的 'input' 的 ID
                raw_text TEXT NOT NULL,
                text_hash TEXT UNIQUE NOT NULL,
                -- AI分析后填充的字段
                title TEXT,
                task_type TEXT,
                tags TEXT,
                summary TEXT,
                quality_score INTEGER,
                quality_reason TEXT,
                is_processed INTEGER DEFAULT 0, -- 0=未处理, 1=已处理
                markdown_path TEXT
            )
            """
            )
            conn.commit()
            # 【【【 核心修正点在这里 】】】
            # 把成功日志也修改一下
            print(f"✅ [DB] 数据库准备就绪: {DB_FILE.name}")
            return True
    except Exception as e:
        print(f"❌ [DB] 严重错误: 初始化SQLite失败: {e}")
        return False


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
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        collections = [c.name for c in QDRANT_CLIENT.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            QDRANT_CLIENT.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        count = QDRANT_CLIENT.count(
            collection_name=QDRANT_COLLECTION_NAME, exact=True
        ).count
        print(f"✅ [向量] 成功连接！当前记忆库中有 {count} 个向量。")
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


def get_content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            ["notify-send", title, message, "-a", "MuseBox", "-t", "5000"], check=True
        )
    except Exception:
        pass


# ==============================================================================
# 3. 【【【 v3.0 核心处理流程重构 】】】
# ==============================================================================


# --- 步骤 1: 秒存函数 (统一入口) ---
def instant_save(raw_text: str, record_type: str):
    text_hash = get_content_hash(raw_text)
    db_id = None
    pair_id = None

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            if cursor.execute(
                "SELECT id FROM records WHERE text_hash = ?", (text_hash,)
            ).fetchone():
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
                state_manager.last_input_id = db_id  # 【关键】记住这个输入ID
                print(f"✅ [输入] 已秒存！ID: {db_id}。等待捕获输出...")
                safe_notification(
                    "输入已捕获", f"“{raw_text[:30]}...” 已记录，等待输出。"
                )

            elif record_type == "output":
                if state_manager.last_input_id is None:
                    print(
                        "!! [警告] 未找到对应的输入。此记录将不会配对，也不会进行AI分析。"
                    )
                    safe_notification(
                        "输出捕获警告", "未找到对应的输入，此记录未配对。"
                    )
                    # 仍然保存，但作为一个孤立的输出，不触发后续流程
                    params = (
                        current_time,
                        record_type,
                        None,
                        raw_text,
                        text_hash,
                        "孤立的输出",
                    )
                    cursor.execute(
                        "INSERT INTO records (capture_time, record_type, pair_id, raw_text, text_hash, title) VALUES (?, ?, ?, ?, ?, ?)",
                        params,
                    )
                    db_id = None  # 返回None，阻止后台任务
                else:
                    pair_id = state_manager.last_input_id
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
                    state_manager.last_input_id = None  # 【关键】配对完成，清空状态
                    print(f"✅ [输出] 已秒存！ID: {db_id}。后台AI分析任务已启动...")

            conn.commit()
            return db_id, pair_id

    except Exception as e:
        print(f"!! [秒存] 写入数据库时发生错误: {e}")
        return None, None


# --- 步骤 2: 后台AI分析任务 (统一入口) ---
async def process_pair_in_background(output_id, output_text, input_id):
    print(
        f"   -> [AI分析] 正在为数据对 (Input:{input_id}, Output:{output_id}) 请求AI分析..."
    )
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # 【【【 修正点 】】】从元组中正确提取字符串
            input_text_tuple = conn.execute(
                "SELECT raw_text FROM records WHERE id = ?", (input_id,)
            ).fetchone()
            if not input_text_tuple:
                print(
                    f"   !! [严重错误] 在数据库中找不到 ID 为 {input_id} 的输入记录！分析中止。"
                )
                safe_notification("处理失败", f"数据库中找不到配对的输入 ID:{input_id}")
                return
            input_text = input_text_tuple[0]  # 正确提取字符串

        prompt = PAIR_ANALYSIS_PROMPT.replace("__INPUT_TEXT__", input_text).replace(
            "__OUTPUT_TEXT__", output_text
        )

        response_text = await asyncio.wait_for(llm.ainvoke(prompt), timeout=300.0)
        ai_data = json.loads(clean_json_response(response_text.content))
        print(f"   -> [AI分析] 分析完成: “{ai_data.get('title')}”")

        loop = asyncio.get_running_loop()

        # 异步执行数据库更新和文件创建
        await loop.run_in_executor(
            None,
            update_database_and_create_markdown,
            input_id,
            output_id,
            input_text,
            output_text,
            ai_data,
        )

        # 异步执行向量化
        text_to_embed = (
            f"问题: {input_text}\n答案: {output_text}\n总结: {ai_data.get('summary')}"
        )
        await vectorize_record(output_id, text_to_embed, ai_data)

        # 【最终反馈】所有任务完成后发送通知
        safe_notification(
            "数据对处理完成",
            f"“{ai_data.get('title', '...')[:40]}” 已被完全处理并归档。",
        )

    except Exception as e:
        print(f"   !! [后台任务] 严重错误: 处理数据对时失败: {e}")
        safe_notification(
            "处理失败", f"处理数据对(I:{input_id}, O:{output_id})时发生错误。"
        )


def update_database_and_create_markdown(
    input_id, output_id, input_text, output_text, ai_data
):
    """在一个函数中处理所有同步IO，减少数据库连接次数"""
    try:
        title = ai_data.get("title", "未命名数据对")
        # 创建Markdown笔记
        safe_filename = re.sub(r'[\/*?:"<>|]', "", title)
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_filename}.md"
        filepath = MARKDOWN_NOTES_DIR / filename
        tags_str = " ".join([f"`{tag}`" for tag in ai_data.get("tags", [])])
        quality = ai_data.get("quality_rating", {})

        content = f"""# {title}\n**任务类型**: {ai_data.get("task_type", "N/A")} | **质量评分**: {quality.get("score", "N/A")}/5\n**标签**: {tags_str}\n\n## 核心摘要\n{ai_data.get("summary", "N/A")}\n\n**AI评估理由**: *{quality.get("reason", "N/A")}*\n\n---\n\n## 问答详情\n### ➡️ 输入 (ID: {input_id})\n```\n{input_text}\n```\n\n### ✅ 输出 (ID: {output_id})\n```\n{output_text}\n```\n"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   -> [归档] Markdown笔记已创建: {filepath.name}")

        # 更新数据库
        with sqlite3.connect(DB_FILE) as conn:
            # 【【【 修正点 1: 更新 output 记录 】】】
            # SQL语句需要8个 '?'
            # 参数元组也必须正好有8个值
            params_output = (
                title,
                ai_data.get("task_type"),
                json.dumps(ai_data.get("tags"), ensure_ascii=False),
                ai_data.get("summary"),
                quality.get("score"),
                quality.get("reason"),
                str(filepath),
                output_id,
            )
            conn.execute(
                """
                UPDATE records SET title=?, task_type=?, tags=?, summary=?, quality_score=?, quality_reason=?, is_processed=1, markdown_path=?
                WHERE id=?
            """,
                params_output,
            )

            # 【【【 修正点 2: 更新 input 记录 】】】
            # SQL语句需要6个 '?'
            # 参数元组也必须正好有6个值
            params_input = (
                title,
                ai_data.get("task_type"),
                json.dumps(ai_data.get("tags"), ensure_ascii=False),
                ai_data.get("summary"),
                str(filepath),
                input_id,
            )
            conn.execute(
                """
                UPDATE records SET title=?, task_type=?, tags=?, summary=?, is_processed=1, markdown_path=?
                WHERE id=?
            """,
                params_input,
            )

            conn.commit()
        print(f"   -> [DB] 数据对 (I:{input_id}, O:{output_id}) 已在数据库中完全更新。")
    except Exception as e:
        print(f"   !! [DB/归档] 更新或创建文件时出错: {e}")


async def vectorize_record(record_id, text_to_embed, ai_data):
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
async def trigger_capture_task(record_type):
    print(f"\n>> [信号] 收到任务信号: capture_{record_type}")
    try:
        # 兼容 Wayland 和 X11
        clipboard_content = ""
        try:
            proc = await asyncio.create_subprocess_exec(
                "wl-paste", "-n", stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            clipboard_content = stdout.decode("utf-8").strip()
        except FileNotFoundError:
            proc = await asyncio.create_subprocess_exec(
                "xclip", "-o", "-selection", "clipboard", stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            clipboard_content = stdout.decode("utf-8").strip()

        if clipboard_content:
            loop = asyncio.get_running_loop()
            db_id, pair_id = await loop.run_in_executor(
                None, instant_save, clipboard_content, record_type
            )

            if db_id and pair_id and record_type == "output":
                # 只有当成功创建了一个“输出”记录并找到了配对的“输入”时，才启动后台处理
                asyncio.create_task(
                    process_pair_in_background(db_id, clipboard_content, pair_id)
                )
        else:
            print(">> [系统] 剪贴板内容为空，已忽略。")
    except Exception as e:
        print(f"!! [错误] 捕捉时发生未知错误: {e}")


async def watch_for_triggers():
    IPC_DIR.mkdir(exist_ok=True)
    for f in TRIGGER_FILES.values():
        if f.exists():
            f.unlink()

    print(">> [监控中] 正在监控输入/输出信号...")
    while True:
        try:
            if TRIGGER_FILES["input"].exists():
                await trigger_capture_task("input")
                await asyncio.sleep(0.1)
                TRIGGER_FILES["input"].unlink()

            if TRIGGER_FILES["output"].exists():
                await trigger_capture_task("output")
                await asyncio.sleep(0.1)
                TRIGGER_FILES["output"].unlink()

            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"!! [监控循环] 发生错误: {e}")
            await asyncio.sleep(5)


async def main_async():
    if not setup_systems():
        return

    print("\n" + "=" * 70)
    print("  ✅ [系统就绪] MuseBox v3.0 (数据对生产线) 已启动")
    print("  请在系统设置中，为您的一对快捷键绑定下面的【完整命令】：")
    print(f"\n  捕获 输入 (推荐 Alt+D):    touch {TRIGGER_FILES['input'].resolve()}")
    print(f"  捕获 输出 (推荐 Alt+S):    touch {TRIGGER_FILES['output'].resolve()}\n")
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
