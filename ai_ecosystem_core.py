# ==============================================================================
#      AI Ecosystem Core v16.0 - Fedora 原生增强版
# ==============================================================================
# 版本说明:
# - 【【【 V16.0 架构升级 - 原生、精准、高效 】】】
#   - 核心数据库: 从 ChromaDB 迁移到性能更强、更专业的 Qdrant 向量数据库。
#   - 原生化集成: 放弃跨平台库(tkinter, plyer)，全面采用 Linux 原生命令行工具
#     (xclip, gnome-screenshot, notify-send)，运行更稳定、响应更迅速。
#   - 交互模式变革: 从“自动监控剪贴板”转变为“用户通过精确热键调用功能”模式，
#     意图明确，杜绝误触发。
#   - 热键系统: 升级到 pynput 库，提升在 Linux 环境下的可靠性。
#   - 知识管理: 引入“集合”(Collection)概念，实现对不同来源知识的结构化存储
#     (如 "ai_ecosystem_core", "ai_recruitment_assistant")。
# ==============================================================================
# ==============================================================================
#      【【【 v16.1 终极网络修复模块 】】】
# ==============================================================================
import os
from dotenv import load_dotenv
load_dotenv() 
# 这会加载 .env 文件中的变量到环境变量
# 清除所有可能导致内部库（如qdrant_client）混淆的SOCKS代理环境变量
# 确保程序能正确连接到 localhost 上的 Docker 容器
if 'all_proxy' in os.environ: del os.environ['all_proxy']
if 'ALL_PROXY' in os.environ: del os.environ['ALL_PROXY']
# ==============================================================================
import pyperclip
import time
import google.generativeai as genai
import sys
import threading
import os
import re
import logging
import json
import notion_client
from googleapiclient.discovery import build
from datetime import datetime, timezone, timedelta
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import uuid
# 【【【 新增：Watchdog 和 Path 库 】】】
from pathlib import Path
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
# v16.0 新增 Imports
import subprocess # 用于调用原生命令行工具

from qdrant_client import QdrantClient, models # 新的数据库客户端
from sentence_transformers import SentenceTransformer # 用于生成向量
# 【【【 新增：定义IPC信号文件路径 】】】
IPC_DIR = Path.home() / ".ai_ecosystem_ipc"
# 为每个功能定义一个信号文件
TRIGGER_FILES = {
    "rag_qa": IPC_DIR / "trigger_rag_qa",
    "smart_reply": IPC_DIR / "trigger_smart_reply",
    "text_analysis": IPC_DIR / "trigger_text_analysis",
    "optimize_post": IPC_DIR / "trigger_optimize_post",
    "translate": IPC_DIR / "trigger_translate",
    "optimize_comm": IPC_DIR / "trigger_optimize_comm",
    "quick_save": IPC_DIR / "trigger_quick_save",
    "process_jd": IPC_DIR / "trigger_process_jd",
    "analyze_screenshot": IPC_DIR / "trigger_analyze_screenshot",
    "quick_ocr": IPC_DIR / "trigger_quick_ocr"
}

# --- 0. 【【【 用户配置区 - v16.0 统一版 (从 .env 加载) 】】】 ---
# 程序将从项目根目录下的 .env 文件自动加载所有配置

# 1. 您的 Google AI API Key
API_KEY = os.getenv("GOOGLE_AI_KEY")

# 2. 您的 Notion Internal Integration Token
NOTION_TOKEN = os.getenv("NOTION_TOKEN")

# --- 核心数据库ID ---
CORE_BRAIN_DATABASE_ID = os.getenv("CORE_BRAIN_DATABASE_ID")
TOOLBOX_LOG_DATABASE_ID = os.getenv("TOOLBOX_LOG_DATABASE_ID")
CANDIDATE_DB_ID = os.getenv("CANDIDATE_DB_ID")
DAILY_REVIEW_DATABASE_ID = os.getenv("DAILY_REVIEW_DATABASE_ID")
TRAINING_HUB_DATABASE_ID = os.getenv("TRAINING_HUB_DATABASE_ID")
INBOX_DATABASE_ID = os.getenv("INBOX_DATABASE_ID")

# --- 招聘工作流数据库 ---
JD_HUB_DATABASE_ID = os.getenv("JD_HUB_DATABASE_ID")
CANDIDATE_PROFILE_HUB_DB_ID = os.getenv("CANDIDATE_PROFILE_HUB_DB_ID")

# --- Google Search API ---
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

# --- 添加一个启动检查，确保关键密钥已加载 ---
if not API_KEY or not NOTION_TOKEN:
    print("❌ [严重错误] 关键的 API_KEY 或 NOTION_TOKEN 未能在 .env 文件中找到！")
    print("   -> 请确保您的 .env 文件存在，并且包含了 GOOGLE_AI_KEY 和 NOTION_TOKEN。")
    sys.exit(1) # 直接退出程序

# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.0 数据库升级模块 (Qdrant) 】】】 ⬇⬇⬇
# ==============================================================================
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_QDRANT_DB_AVAILABLE = False
# 定义向量维度，"all-MiniLM-L6-v2"模型的维度是384
VECTOR_DIMENSION = 384
# 【【【 新增：为这个程序定义专属的 Collection 名称列表 】】】
ECOSYSTEM_COLLECTIONS = ["ai_ecosystem_core", "ai_recruitment_assistant"]
def setup_database_and_embedding():
    """
    初始化Qdrant客户端、加载本地嵌入模型，并确保所需的集合存在。
    """
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_QDRANT_DB_AVAILABLE
    try:
        print(">> [Qdrant] 正在连接到本地向量数据库 (localhost:6333)...")
        QDRANT_CLIENT = QdrantClient("localhost", port=6333)
        # 检查与服务器的连通性
        QDRANT_CLIENT.get_collections()
        print("✅ [Qdrant] 成功连接到数据库服务器！")

        print(">> [Embedding] 正在加载本地句子转换模型 (all-MiniLM-L6-v2)...")
        # 首次运行时会自动下载模型，可能需要一些时间
        EMBEDDING_MODEL = SentenceTransformer('./all-MiniLM-L6-v2') # 由启动器自动修改为本地路径
        print("✅ [Embedding] 嵌入模型加载成功！")

        # 检查并创建所需的集合
        collections_to_ensure = ECOSYSTEM_COLLECTIONS
        existing_collections = [col.name for col in QDRANT_CLIENT.get_collections().collections]
        print(f">> [Qdrant] 当前存在的集合: {existing_collections}")

        for collection_name in collections_to_ensure:
            if collection_name not in existing_collections:
                print(f">> [Qdrant] 集合 '{collection_name}' 不存在，正在创建...")
                QDRANT_CLIENT.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
                )
                print(f"✅ [Qdrant] 集合 '{collection_name}' 创建成功！")

        IS_QDRANT_DB_AVAILABLE = True
    except Exception as e:
        print(f"❌ [Qdrant/Embedding] 严重错误: 初始化失败: {e}")
        print("   -> 请确保您已通过 Docker 启动了 Qdrant 容器！")
        print("   -> 推荐命令: docker run -p 6333:6333 qdrant/qdrant")
        IS_QDRANT_DB_AVAILABLE = False
# ==============================================================================

# --- 【【【 Notion属性名精准配置区 - 全生态版 】】】 ---
RELATION_BRAIN_PROPERTY_NAME = "🧠 智能关联"
RELATION_LOG_PROPERTY_NAME = "💬 相关日志"
RELATION_CANDIDATE_PROPERTY_NAME = "👤 相关候选人"
RELATION_REVIEW_PROPERTY_NAME = "📈 相关复盘"
RELATION_TRAINING_PROPERTY_NAME = "⚙️ 相关训练"
RELATION_LINK_BRAIN = "源链接-作战室"
RELATION_LINK_LOG = "源链接-互动日志"
RELATION_LINK_CANDIDATE = "源链接-候选人中心"
RELATION_LINK_REVIEW = "源链接-每日复盘"


# --- 1. 全局状态与控制 ---
logging.getLogger('pynput').setLevel(logging.WARNING)

class AppController:
    def __init__(self):
        self.is_processing = False
        self.should_exit = False
        self.lock = threading.Lock()
        # v16.0 中， capture_mode_active 和 monitoring_active 已被精确热键取代，不再需要

    def trigger_exit(self):
        print("\n>> 收到退出指令 (Alt+Esc)，系统正在安全关闭... <<")
        with self.lock:
            self.should_exit = True

app_controller = AppController()

def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
# 【【【 新增：文件信号处理器 】】】
class FileTriggerHandler(FileSystemEventHandler):
    def __init__(self):
        # 将信号文件名映射到对应的处理函数
        self.function_map = {
            str(TRIGGER_FILES["rag_qa"]): create_hotkey_handler(answer_question_with_rag),
            str(TRIGGER_FILES["smart_reply"]): create_hotkey_handler(get_ai_reply),
            str(TRIGGER_FILES["text_analysis"]): create_hotkey_handler(get_ai_analysis),
            str(TRIGGER_FILES["optimize_post"]): create_hotkey_handler(optimize_social_post),
            str(TRIGGER_FILES["translate"]): create_hotkey_handler(translate_text),
            str(TRIGGER_FILES["optimize_comm"]): create_hotkey_handler(optimize_communication),
            str(TRIGGER_FILES["quick_save"]): create_hotkey_handler(quick_save_to_inbox),
            str(TRIGGER_FILES["process_jd"]): create_hotkey_handler(process_clipboard_jd),
            str(TRIGGER_FILES["analyze_screenshot"]): trigger_deep_analysis,
            str(TRIGGER_FILES["quick_ocr"]): trigger_quick_ocr,
        }

    def on_created(self, event):
        if not event.is_directory:
            src_path = event.src_path
            # 检查触发的文件是否在我们的功能列表里
            if src_path in self.function_map:
                print(f"\n>> [信号] 收到任务信号: {Path(src_path).name}")
                # 获取并调用对应的处理函数
                target_function = self.function_map[src_path]
                target_function()
                # 立即删除信号文件，防止重复触发
                try:
                    time.sleep(0.2) # 等待一小会确保文件句柄释放
                    os.unlink(src_path)
                except OSError:
                    pass
# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.0 核心辅助函数 (原生化改造) 】】】 ⬇⬇⬇
# ==============================================================================
def safe_notification(title, message, **kwargs):
    """
    【v16.0 原生版】使用 notify-send 命令发送桌面通知，更稳定可靠。
    """
    try:
        subprocess.run(['notify-send', title, message, '-a', 'AI Ecosystem Core', '-t', '4000'], check=True)
    except FileNotFoundError:
        # 如果系统中没有 notify-send，则静默失败
        pass
    except Exception as e:
        print(f"!! [通知] 发送通知时出错: {e}")

def _capture_screen_area():
    """
    【v16.0 原生版】使用 gnome-screenshot 进行区域截图，稳定且符合Fedora原生体验。
    返回一个 PIL.Image 对象或 None。
    """
    try:
        # 创建一个临时文件来保存截图
        temp_file_path = f"/tmp/screenshot_{uuid.uuid4()}.png"
        
        # 调用 gnome-screenshot 的区域截图功能 (-a)，并将输出保存到文件 (-f)
        # check=True 会在命令失败时抛出异常
        subprocess.run(['gnome-screenshot', '-a', '-f', temp_file_path], check=True)
        
        # 检查文件是否真的被创建
        if os.path.exists(temp_file_path):
            with Image.open(temp_file_path) as img:
                # 复制图像数据，以便我们可以删除临时文件
                img_copy = img.copy()
            os.remove(temp_file_path) # 清理临时文件
            return img_copy
        else:
            # 用户可能按下了 Esc 取消了截图
            return None
    except FileNotFoundError:
        print("!! [截图工具] 错误: 'gnome-screenshot' 命令未找到。请确保它已安装。")
        safe_notification("截图失败", "未找到 gnome-screenshot 工具")
        return None
    except subprocess.CalledProcessError:
        # 当用户取消截图时，gnome-screenshot会以非零状态码退出，导致此异常
        print(">> [截图工具] 截图操作已取消。")
        return None
    except Exception as e:
        print(f"!! [截图工具] 截图时发生未知错误: {e}")
        return None

def create_hotkey_handler(target_func):
    """
    【v16.0 原生版】为所有热键创建一个处理器。
    优先使用 xclip 获取当前鼠标选中的文本，失败则回退到读取剪贴板。
    这种方式响应更即时，意图更明确。
    """
    def handler():
        # 检查是否已有任务在运行
        with app_controller.lock:
            if app_controller.is_processing:
                print("!! [系统] 正忙，请等待上一个任务完成。")
                safe_notification("AI 系统正忙", "请稍后再试")
                return

        def task_with_selected_text():
            text_to_process = ""
            try:
                # 优先尝试获取鼠标高亮的文本 (Linux X11 特有)
                result = subprocess.run(['xclip', '-o', '-selection', 'primary'], capture_output=True, text=True, check=True)
                text_to_process = result.stdout
            except (FileNotFoundError, subprocess.CalledProcessError):
                # 如果 xclip 失败或没有选中内容，则回退到读取剪贴板
                try:
                    text_to_process = pyperclip.paste()
                except Exception as e_pyperclip:
                    print(f"!! [热键处理] 读取剪贴板时也失败了: {e_pyperclip}")

            if text_to_process and not text_to_process.isspace():
                # 使用线程处理，避免阻塞UI
                threading.Thread(target=target_func, args=(text_to_process,), daemon=True).start()
            else:
                print(">> [热键处理] 未获取到有效文本。")

        # 立即在后台线程中执行，不阻塞热键监听器
        threading.Thread(target=task_with_selected_text, daemon=True).start()

    return handler
# ==============================================================================

def generate_search_query(question, max_length=60):
    """【智能查询优化器】"""
    try:
        print(f">> [查询优化] 正在为问题 “{question[:30]}...” 提炼核心搜索词...")
        prompt = f"""从以下用户问题中，提炼出最核心、最适合用于Google搜索的关键词。只返回关键词，不要任何解释或多余的文字。用户问题：“{question}”\n\n优化后的搜索词："""
        response = gemini_model.generate_content(prompt, request_options={'timeout': 10})
        search_query = response.text.strip().replace("\n", " ").replace("\"", "")
        if len(search_query) > max_length:
            search_query = search_query[:max_length]
        print(f"✅ [查询优化] 优化后的搜索词: “{search_query}”")
        return search_query
    except Exception as e:
        print(f"!! [查询优化] 提炼搜索词时出错: {e}。将使用原始问题进行搜索。")
        return question

def google_search_for_context(query, num_results=3):
    """【Google 搜索集成模块】"""
    if not GOOGLE_SEARCH_API_KEY or "..." in GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX: return ""
    print(f">> [Google搜索] 正在为 “{query[:30]}...” 进行实时网络搜索...")
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_SEARCH_CX, num=num_results).execute()
        if 'items' not in res: return ""
        search_results = [f"【网络来源 {i+1}: {item.get('title', '无标题')}】\n摘要: {item.get('snippet', '无摘要').replace(chr(10), ' ')}\n链接: {item.get('link', '#')}" for i, item in enumerate(res['items'])]
        print(f"✅ [Google搜索] 成功获取 {len(search_results)} 条网络信息。")
        return "\n\n---\n\n".join(search_results)
    except Exception as e:
        print(f"!! [Google搜索] 网络搜索时发生错误: {e}")
        if 'quota' in str(e).lower(): print("   -> 提示：可能是每日免费搜索配额(100次)已用完。")
        return ""

# --- 2. API 初始化 ---
gemini_model = None
def setup_api():
    global gemini_model
    if "..." in API_KEY or len(API_KEY) < 30:
        print("!! 错误：请先在代码的【用户配置区】填入你自己的API Key !!")
        return False
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
        print(">> [API] Gemini API 初始化成功。")
        return True
    except Exception as e:
        print(f"!! [API] 初始化失败: {e}")
        return False

# --- 3. 写入AI训练中心 ---
def write_to_training_hub(task_type, input_text, output_text, source_db_name, source_page_id):
    if not TRAINING_HUB_DATABASE_ID or len(TRAINING_HUB_DATABASE_ID) < 32: return
    try:
        notion = notion_client.Client(auth=NOTION_TOKEN)
        safe_input_text = str(input_text)[:1990]
        safe_output_text = str(output_text)[:1990]
        training_title = f"【{task_type}】{safe_input_text[:60]}..."
        properties_data = {
            "训练任务": {"title": [{"text": {"content": training_title}}]},
            "任务类型": {"select": {"name": task_type}},
            "源数据 (Input)": {"rich_text": [{"text": {"content": safe_input_text}}]},
            "理想输出 (Output)": {"rich_text": [{"text": {"content": safe_output_text}}]},
            "标注状态": {"select": {"name": "待审核"}}
        }
        relation_column_map = {'CoreBrain': RELATION_LINK_BRAIN, 'Log': RELATION_LINK_LOG, 'Candidate': RELATION_LINK_CANDIDATE}
        if source_db_name in relation_column_map and source_page_id:
            properties_data[relation_column_map[source_db_name]] = {"relation": [{"id": source_page_id}]}
        notion.pages.create(parent={"database_id": TRAINING_HUB_DATABASE_ID}, properties=properties_data)
        print(f">> [训练中心] 已成功记录一条 '{task_type}' 训练数据。")
    except Exception as e:
        print(f"!! [训练中心] 写入时出错: {e}")

# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.0 数据库写入函数 (Qdrant) 】】】 ⬇⬇⬇
# ==============================================================================
# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.0 数据库写入函数 (Qdrant) - 增强反馈版 】】】 ⬇⬇⬇
# ==============================================================================
def add_knowledge_to_qdrant(collection_name, document_id, document_text, metadata):
    if not IS_QDRANT_DB_AVAILABLE or not document_text or not EMBEDDING_MODEL:
        return
    try:
        print(f">> [Qdrant] 正在向量化知识: “{metadata.get('source_title', 'Untitled')[:30]}...”")
        vector = EMBEDDING_MODEL.encode(clean_text(document_text)).tolist()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(document_id)))
        payload = {"text": clean_text(document_text), "metadata": metadata}
        
        QDRANT_CLIENT.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
            wait=True 
        )
        
        # --- 【【【 新增代码：获取并显示当前总数 】】】 ---
        collection_info = QDRANT_CLIENT.get_collection(collection_name=collection_name)
        current_count = collection_info.points_count
        # ---------------------------------------------------
        
        # 修改打印信息，加入总数
        print(f"✅ [Qdrant] 知识已成功添加/更新到 '{collection_name}'！当前总量: {current_count}")
        
    except Exception as e:
        print(f"!! [Qdrant] 添加或更新知识时出错: {e}")
# --- 5. 写入AI互动日志 (适配v16.0) ---
def save_log_to_notion(log_type: str, input_text: str, output_text: str):
    if len(TOOLBOX_LOG_DATABASE_ID) < 32: return
    try:
        notion = notion_client.Client(auth=NOTION_TOKEN)
        page_title = f"【{log_type}】{input_text[:80]}"
        print(f">> [Notion日志] 正在保存 '{log_type}' 记录到Notion...")
        children_blocks = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}} for chunk in [output_text[i:i + 1900] for i in range(0, len(output_text), 1900)]]
        current_date_iso = datetime.now(timezone(timedelta(hours=8))).isoformat()
        properties_data = { "主题": {"title": [{"text": {"content": page_title}}]}, "类型": {"select": {"name": log_type}}, "输入内容": {"rich_text": [{"text": {"content": input_text[:1900]}}]}, "输出摘要": {"rich_text": [{"text": {"content": output_text[:1900]}}]}, "记录日期": {"date": {"start": current_date_iso}}}
        new_log_page = notion.pages.create(parent={"database_id": TOOLBOX_LOG_DATABASE_ID}, properties=properties_data, children=children_blocks)
        print(">> [Notion日志] 保存成功！")
        log_page_id = new_log_page.get('id')
        
        # 异步写入训练中心
        task_type_map = {"智能回复": "风格化写作", "文本分析": "摘要生成", "内容优化": "风格化写作", "翻译": "翻译", "沟通优化": "风格化写作", "图像分析": "图像理解与结构化输出", "RAG问答": "问答配对"}
        threading.Thread(target=write_to_training_hub, args=(task_type_map.get(log_type, "通用任务"), input_text, output_text, 'Log', log_page_id), daemon=True).start()
        
        # 【v16.0 修改】异步写入Qdrant核心知识库
        knowledge_text = f"这是一条'{log_type}'类型的AI互动日志。\n\n【用户输入】:\n{input_text}\n\n【AI的回答或分析】:\n{output_text}"
        metadata = {"source_db": "log", "task_type": log_type, "source_title": page_title, "original_id": log_page_id}
        threading.Thread(target=add_knowledge_to_qdrant, args=("ai_ecosystem_core", log_page_id, knowledge_text, metadata), daemon=True).start()
    except Exception as e:
        print(f"!! [Notion日志] 保存时出错: {e}")

# --- 6. 快速剪藏到Inbox (【【【 最终双功能增强版 】】】) ---
# --- 6. 快速剪藏到Inbox (【【【 最终 V16.2 兼容版 】】】) ---
def quick_save_to_inbox(text_to_save):
    if not INBOX_DATABASE_ID or len(INBOX_DATABASE_ID) < 32:
        safe_notification('快速剪藏失败', '请先在代码中配置Inbox数据库ID')
        return
    try:
        notion = notion_client.Client(auth=NOTION_TOKEN)
        
        cleaned_text = text_to_save.strip()
        if not cleaned_text:
            print(">> [快速剪藏] 剪藏内容为空，已跳过。")
            return

        # 从第一行生成标题
        page_title = cleaned_text.split('\n')[0][:100]
        # 【修正】使用完整的原文作为页面内容
        page_content = cleaned_text

        print(f"\n>> [快速剪藏] 正在保存内容到'灵感剪藏箱': “{page_title}”")
        
        children_blocks = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}} for chunk in [page_content[i:i+1900] for i in range(0, len(page_content), 1900)]] if page_content else []
        
        current_date_iso = datetime.now(timezone(timedelta(hours=8))).isoformat()
        properties_data = {"标题": {"title": [{"text": {"content": page_title}}]}, "日期": {"date": {"start": current_date_iso}}}
        
        new_inbox_page = notion.pages.create(parent={"database_id": INBOX_DATABASE_ID}, properties=properties_data, children=children_blocks)
        
        print(">> [快速剪藏] 保存到 Inbox 成功！")
        safe_notification('快速剪藏成功', f'内容已存入: {page_title}')
        inbox_page_id = new_inbox_page.get('id')
        
        # 【【【 关键修正 】】】调用当前版本正确的 Qdrant 函数
        metadata = {"source_db": "inbox", "task_type": "quick_capture", "source_title": page_title, "original_id": inbox_page_id}
        threading.Thread(target=add_knowledge_to_qdrant, args=("ai_ecosystem_core", inbox_page_id, cleaned_text, metadata), daemon=True).start()

        # --- 【功能联动】将同样的内容提交给“核心大脑”处理 ---
        print(">> [功能联动] 已将该内容提交给“核心大脑”进行深度处理和归档...")
        # 注意：这里我们传递 cleaned_text 保证数据一致性
        threading.Thread(target=process_data_packet, args=(cleaned_text,), daemon=True).start()
        # ---------------------------------------------------

    except Exception as e:
        print(f"!! [快速剪藏] 保存时出错: {e}")
        safe_notification('快速剪藏失败', f"错误: {e}")
# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.0 RAG 主函数 (Qdrant) 】】】 ⬇⬇⬇
# ==============================================================================
def retrieve_from_multiple_collections(question_text, collection_names, top_n=3):
    if not IS_QDRANT_DB_AVAILABLE or not EMBEDDING_MODEL:
        print("!! [RAG检索] Qdrant不可用，无法进行知识库检索。")
        return ""
    
    print(f">> [RAG检索-Qdrant] 正在为问题“{question_text[:30]}...”高速检索上下文...")
    try:
        # 1. 将查询问题转换为向量
        query_vector = EMBEDDING_MODEL.encode(question_text).tolist()
        
        all_contexts = []
        # 2. 遍历所有指定的集合进行搜索
        for collection_name in collection_names:
            print(f"   -> 正在搜索集合: '{collection_name}'...")
            search_result = QDRANT_CLIENT.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_n
            )
            # 3. 从搜索结果中提取文本内容
            collection_contexts = [hit.payload["text"] for hit in search_result if "text" in hit.payload]
            all_contexts.extend(collection_contexts)

        if not all_contexts:
            print(">> [RAG检索-Qdrant] 在所有指定集合中均未找到相关上下文。")
            return ""
            
        print(f">> [RAG检索-Qdrant] 成功找到 {len(all_contexts)} 条相关上下文。")
        return "\n\n---\n\n".join(all_contexts)
    except Exception as e:
        print(f"!! [RAG检索-Qdrant] 查询时发生错误: {e}")
        return ""

def answer_question_with_rag(question):
    with app_controller.lock:
        if app_controller.is_processing:
            print("!! [RAG问答] 系统正忙，请稍后。")
            safe_notification('AI系统正忙', '请等待上一个任务完成后再试')
            return
        app_controller.is_processing = True
    try:
        optimized_query = generate_search_query(question)
        
        # 【v16.0 修改】从多个Qdrant集合中检索
        collections_to_search = ECOSYSTEM_COLLECTIONS
        personal_context = retrieve_from_multiple_collections(
            question,
            collection_names=collections_to_search,
            top_n=3
        )
        
        web_context_future = ThreadPoolExecutor(max_workers=1).submit(google_search_for_context, optimized_query)
        web_context = web_context_future.result()

        combined_context = ""
        if personal_context: combined_context += "--- 【个人知识库信息】 ---\n" + personal_context
        if web_context: combined_context += "\n\n--- 【实时网络搜索结果】 ---\n" + web_context

        if not combined_context.strip():
            prompt = f"""# 指令：作为通用问答助手\n请用你的通用知识回答以下问题。你的输出必须是一个完整的、无额外解释的JSON对象。\n# JSON输出结构:\n{{ "answer": "你基于通用知识给出的详细回答", "source_used": ["general_knowledge"], "confidence_score": 0.5, "evidence": [], "thoughts": "由于用户的个人知识库和实时网络搜索中都没有相关内容，我将使用我的通用知识库来回答这个问题。" }}\n# 问题:\n{question}"""
            print(">> [RAG问答] 个人知识库与网络均无资料，使用LLM通用知识。")
        else:
            prompt = f"""# 角色: 你是一个【信息整合与应答引擎】。\n## 核心指令\n你的唯一任务是根据下面提供的【上下文信息】来回答用户的【问题】。你【必须】直接利用这些信息进行回答。\n## 强制规则\n1. **绝对优先【实时网络搜索结果】**。\n2. **禁止回避**: 严禁以任何理由拒绝回答。\n3. **具体化回答**: 如果网络搜索结果中提到了任何具体的例子或数据，你【必须】将它们提取并整合到你的回答中。\n## 输出协议: 你的输出必须是一个完整的、无额外解释的JSON对象。\n## JSON输出结构定义:\n{{ "answer": "你的最终回答。", "source_used": "一个描述信息来源的字符串数组。如果使用了网络搜索，'google_search'必须包含在内。", "confidence_score": "一个0.0到1.0之间的浮点数，表示回答基于上下文的程度。", "evidence": ["在此引用1-3条最关键的原文证据。"], "thoughts": "你的思考过程。" }}\n---\n{combined_context}\n---\n【问题】\n{question}"""
        
        print(">> [RAG问答] 已构建混合上下文Prompt，正在请求LLM生成答案...")
        response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"}, request_options={'timeout': 300})
        result_data = json.loads(response.text)
        
        print("\n" + "="*25 + " 混合 RAG 智能问答结果 " + "="*25)
        print(f"您的提问: {question}\n> AI 的回答: {result_data.get('answer', '未提供答案')}\n")
        source_map = {"personal_knowledge_base": "✅ 个人知识库", "google_search": "🌐 实时网络搜索", "general_knowledge": "🧠 LLM通用知识"}
        sources = ", ".join([source_map.get(s, s) for s in result_data.get('source_used', [])])
        print(f"信息来源: {sources}")
        print(f"📈 置信度 (基于上下文): {result_data.get('confidence_score', 0) * 100:.1f}%")
        if evidences := result_data.get('evidence', []):
            print("\n🔍 关键证据:")
            for i, ev in enumerate(evidences): print(f"  [{i+1}] “...{ev[:80]}...”")
        print(f"\n🧠 AI思考过程: “{result_data.get('thoughts', '...')}”\n" + "="*68 + "\n")
        
        full_answer_for_log = f"【回答】\n{result_data.get('answer')}\n\n【来源】\n{sources}\n\n【证据】\n{json.dumps(result_data.get('evidence',[]), ensure_ascii=False)}"
        threading.Thread(target=save_log_to_notion, args=("RAG问答", question, full_answer_for_log), daemon=True).start()
        safe_notification('RAG智能问答完成', f'对“{question[:30]}...”的回答已生成')
    except (json.JSONDecodeError, Exception) as e: 
        print(f"\n!! [RAG问答] 处理时发生错误: {e}")
    finally:
        with app_controller.lock: 
            app_controller.is_processing = False

# --- 8. 图像分析 (适配v16.0) ---
def analyze_screenshot_deep_logic(img):
    """这是截图分析的实际AI处理逻辑。"""
    try:
        print(">> [深度分析] 截图成功！正在发送给Gemini进行深度分析...")
        prompt = """# 角色: 全能视觉分析与策略洞察专家\n# 核心能力: 你是一位精通UI/UX设计、数据可视化、信息架构和中文语境的视觉分析专家。你的任务是接收一张截图，然后从多个维度进行深度分析，并以结构化的JSON格式输出你的洞察。\n# 分析维度与指令:\n1. **图片类型 (Image Type)**: 快速判断并归类这张截图。\n2. **文本提取 (Text Extraction)**: 使用你的OCR能力，提取图片中所有清晰可读的中文和英文文本。\n3. **核心摘要 (Core Summary)**: **综合**图片内容和提取的文本，用一到两句精炼的中文总结这张截图的核心信息。\n4. **行动洞察 (Actionable Insights)**: 这是最重要的部分。发掘出**潜在的机会、风险、优化点或下一步可以采取的具体行动**。\n# 输出协议: 你的输出**必须**是一个完整的、严格符合规范的JSON对象。\n# JSON输出结构:\n{ "图片类型": "string", "提取文本": "string", "核心摘要": "string", "行动洞察": [ "string", "string" ] }"""
        response = gemini_model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"}, request_options={'timeout': 180})
        result_data = json.loads(response.text)
        print("\n" + "="*25 + " 智能截图分析结果 " + "="*25)
        print(f"🖼️  图片类型: {result_data.get('图片类型', '未知')}")
        print(f"📝 核心摘要: {result_data.get('核心摘要', '未能生成摘要')}")
        if insights := result_data.get('行动洞察', []):
            print("💡 行动洞察:")
            for insight in insights: print(f"   - {insight}")
        if extracted_text := result_data.get('提取文本', ''):
            print(f"\n📋 提取文本 (部分预览):\n---\n{extracted_text[:200]}...\n---")
        print("="*68 + "\n")
        log_input_text = f"【图像分析】对一张 '{result_data.get('图片类型', '未知')}' 类型的截图进行分析。"
        log_output_text = json.dumps(result_data, ensure_ascii=False, indent=2)
        save_log_to_notion("图像分析", log_input_text, log_output_text)
    except Exception as e:
        print(f"!! [深度分析] AI处理时出错: {e}")
    finally:
        with app_controller.lock:
            app_controller.is_processing = False

def quick_ocr_logic(img):
    """这是极速OCR的实际AI处理逻辑。"""
    try:
        print(">> [极速OCR] 截图成功！正在发送给Gemini进行极速文字提取...")
        prompt = "# 指令：请提取并返回这张图片中的所有文字，只返回纯文本，不要任何其他内容。"
        response = gemini_model.generate_content([prompt, img], request_options={'timeout': 30})
        extracted_text = response.text.strip()
        if extracted_text:
            pyperclip.copy(extracted_text)
            print("\n" + "="*25 + " 极速OCR结果 (已复制) " + "="*25)
            print(extracted_text); print("="*68 + "\n")
        else:
            print("\n>> [极速OCR] 未能从图片中提取到任何文字。")
    except Exception as e:
        print(f"!! [极速OCR] AI处理时出错: {e}")
    finally:
        with app_controller.lock:
            app_controller.is_processing = False

def run_screenshot_flow(logic_func, flow_name):
    """通用截图流程，避免阻塞主程序。"""
    with app_controller.lock:
        if app_controller.is_processing:
            print(f"!! [{flow_name}] 系统正忙，请稍后。")
            return
        app_controller.is_processing = True

    try:
        print(f"\n>> [{flow_name}] 请用鼠标拖拽选择区域 (按Esc取消)...")
        img = _capture_screen_area()

        if img:
            threading.Thread(target=logic_func, args=(img,), daemon=True).start()
        else:
            print(f">> [{flow_name}] 操作已取消或失败。")
            with app_controller.lock:
                app_controller.is_processing = False
    except Exception as e:
        print(f"!! [截图流程错误] 运行 {flow_name} 流程时发生严重错误: {e}")
        with app_controller.lock:
            app_controller.is_processing = False

def trigger_deep_analysis():
    """热键触发器：深度分析"""
    threading.Thread(target=run_screenshot_flow, args=(analyze_screenshot_deep_logic, "深度分析"), daemon=True).start()

def trigger_quick_ocr():
    """热键触发器：极速OCR"""
    threading.Thread(target=run_screenshot_flow, args=(quick_ocr_logic, "极速OCR"), daemon=True).start()


# --- 9. 核心大脑与智能关联 (适配v16.0) ---
def save_to_core_brain_db(title, category, summarized_question, answer):
    if len(CORE_BRAIN_DATABASE_ID) < 32: return
    try:
        notion = notion_client.Client(auth=NOTION_TOKEN)
        print(f">> [核心大脑] 正在归档知识: “{title}”")
        if not isinstance(answer, str):
            answer = f"【系统归档错误】AI未能生成有效的回答文本。原始数据类型为: {type(answer).__name__}"
        MAX_CONTENT_LENGTH = 180000 
        if len(answer) > MAX_CONTENT_LENGTH:
            answer = answer[:MAX_CONTENT_LENGTH] + "\n\n...【警告：内容过长，已被系统自动截断】..."
        children_blocks = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}} for chunk in [answer[i:i + 1900] for i in range(0, len(answer), 1900)]]
        current_date_iso = datetime.now(timezone(timedelta(hours=8))).isoformat()
        properties_data = { "主题": {"title": [{"text": {"content": title}}]}, "我的问题": {"rich_text": [{"text": {"content": summarized_question}}]}, "分类": {"select": {"name": category}}, "记录日期": {"date": {"start": current_date_iso}}}
        new_page = notion.pages.create(parent={"database_id": CORE_BRAIN_DATABASE_ID}, properties=properties_data, children=children_blocks)
        print(">> [核心大脑] 知识归档成功！")
        safe_notification('核心大脑归档成功', f'主题: {title}')
        new_page_id = new_page.get('id')
        
        # 【v16.0 修改】异步写入Qdrant核心知识库
        text_to_embed = f"主题: {title}\n问题: {summarized_question}\n答案: {answer}".strip()
        metadata = {"source_db": "core_brain", "task_type": "knowledge_capture", "source_title": title, "original_id": new_page_id}
        threading.Thread(target=add_knowledge_to_qdrant, args=("ai_ecosystem_core", new_page_id, text_to_embed, metadata), daemon=True).start()
        
        threading.Thread(target=write_to_training_hub, args=("问答配对", summarized_question, answer, 'CoreBrain', new_page_id), daemon=True).start()
    except Exception as e:
        print(f"!! [核心大脑] 归档时出错: {e}")

def process_data_packet(full_text):
    prompt = f"""# 角色: 你是我的【AI核心大脑】系统的**中央处理器 (Central Processor)**。\n# 核心指令: 你的任务是接收并处理一个包含“原始问题”和“AI回答”的数据包，将其转化为结构化的知识。\n# 待处理数据包: --- {clean_text(full_text)} ---\n# 输出协议: 你必须严格遵循JSON格式进行输出，不得包含任何额外解释。\n{{ "title": "知识主题", "category": "Resource", "summarized_question": "精炼后的核心问题", "ai_answer": "完整的AI回答" }}"""
    try:
        with app_controller.lock: app_controller.is_processing = True
        print(f"\n>> [核心大脑] 中央处理器正在处理数据包...")
        response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"}, request_options={'timeout': 60})
        result = json.loads(response.text)
        save_to_core_brain_db(
            title=result.get('title', '知识主题生成失败'), category=result.get('category', 'Resource'),
            summarized_question=result.get('summarized_question', '问题精炼失败'), answer=result.get('ai_answer', full_text))
    except Exception as e:
        print(f"!! [核心大脑] 系统处理异常: {e}")
        save_to_core_brain_db(title=f"【未处理数据包】{full_text[:30]}...", category='Task', summarized_question='系统处理失败，请手动整理', answer=full_text)
    finally:
        with app_controller.lock: app_controller.is_processing = False

# --- 10. AI工具箱 (适配v16.0) ---
def get_ai_reply(text):
    prompt = f"你是一个情商很高、善于沟通的对话助手。请根据对方说的这句话：“{clean_text(text)}”，生成一句得体、自然、友好的简短回复。"
    try:
        with app_controller.lock: app_controller.is_processing = True
        print(f"\n[智能回复] 正在为“{text[:40]}...”生成回复...")
        response = gemini_model.generate_content(prompt, request_options={'timeout': 20})
        ai_response = response.text.strip()
        pyperclip.copy(ai_response)
        print(f"[智能回复] 回复已生成并复制:\n---\n{ai_response}\n---")
        threading.Thread(target=save_log_to_notion, args=("智能回复", text, ai_response), daemon=True).start()
        safe_notification('AI 智能回复', '回复已生成，可直接粘贴！')
    except Exception as e: print(f"!! [智能回复] 调用API时出错: {e}")
    finally:
        with app_controller.lock: app_controller.is_processing = False

def get_ai_analysis(text):
    prompt = f"""# 角色: 精通中英双语的沟通策略与心理分析专家。\n# 待分析内容: “{clean_text(text)}”\n# 核心任务: 深入分析这段话的真实含义、潜台词，并给出行动建议。\n# 输出格式: 请严格按照“1. 真实含义解析 (字面意思, 潜台词) 2. 行动建议 (思维调整, 具体回应)”的结构输出。"""
    try:
        with app_controller.lock: app_controller.is_processing = True
        print(f"\n[文本分析] 正在分析: “{text[:50]}...”")
        response = gemini_model.generate_content(prompt, request_options={'timeout': 120}, stream=True)
        print("\n--- AI分析中... ---\n")
        full_response_text = "".join(chunk.text for chunk in response if hasattr(chunk, 'text') and chunk.text)
        print(full_response_text)
        print("\n\n--- 分析完成 ---\n")
        threading.Thread(target=save_log_to_notion, args=("文本分析", text, full_response_text), daemon=True).start()
    except Exception as e: print(f"\n!! [文本分析] 调用API时出错: {e}")
    finally:
        with app_controller.lock: app_controller.is_processing = False

def optimize_social_post(text):
    prompt = f"""# 任务：深度重塑与双语呈现\n将以下内容，重塑为一条具有深度、能引发思考的洞见。同时提供精炼的中文和优雅的英文版本。\n# 待处理原文：“{clean_text(text)}”\n# 输出格式：严格按照“中文:\n[内容]\n\nEnglish:\n[Content]”格式。"""
    try:
        with app_controller.lock: app_controller.is_processing = True
        print(f"\n[内容优化] 正在优化: “{text[:40]}...”")
        response = gemini_model.generate_content(prompt, request_options={'timeout': 60})
        optimized_text = response.text.strip()
        pyperclip.copy(optimized_text)
        print(f"[内容优化] 成功 (已复制):\n---\n{optimized_text}\n---")
        threading.Thread(target=save_log_to_notion, args=("内容优化", text, optimized_text), daemon=True).start()
        safe_notification('AI 内容优化', '优化后的内容已复制！')
    except Exception as e: print(f"!! [内容优化] 时出错: {e}")
    finally:
        with app_controller.lock: app_controller.is_processing = False

def translate_text(text):
    prompt = f"Translate the following Chinese text to English. Output only the translated text:\n\n{clean_text(text)}" if re.search(r'[\u4e00-\u9fa5]', clean_text(text)) else f"将以下英文内容翻译成中文，只输出翻译后的纯文本:\n\n{clean_text(text)}"
    try:
        with app_controller.lock: app_controller.is_processing = True
        print(f"\n[翻译] 正在翻译: “{text[:40]}...”")
        response = gemini_model.generate_content(prompt, request_options={'timeout': 30})
        translated_text = response.text.strip()
        pyperclip.copy(translated_text)
        print(f"[翻译] 成功 (已复制):\n---\n{translated_text}\n---")
        threading.Thread(target=save_log_to_notion, args=("翻译", text, translated_text), daemon=True).start()
        safe_notification('AI 翻译', '翻译结果已复制！')
    except Exception as e: print(f"!! [翻译] 时出错: {e}")
    finally:
        with app_controller.lock: app_controller.is_processing = False

def optimize_communication(text):
    prompt = f"""# 指令：启动“沟通优化大师”模式\n你现在将扮演一个名为“沟通优化大师”的 AI 助手。你的核心能力是深度分析我提供的任何文本，并自动将其优化。从现在起，请严格遵循以下工作流程：\n## 你的分析框架\n1. 推断核心意图\n2. 评估和建议语气\n3. 猜测沟通对象与场景\n4. 识别并保留关键信息\n## 你的输出格式\n---\n**【优化方案】**\n**▶️ 版本一：[ 版本名 ]**\n> [ 优化后的文本 ]\n**▶️ 版本二：[ 版本名 ]**\n> [ 优化后的文本 ]\n**💡【润色解析】**\n* **核心思路：**\n* **为什么这么改：**\n* **适用场景建议：**\n---\n# 待优化文本\n{clean_text(text)}"""
    try:
        with app_controller.lock: app_controller.is_processing = True
        print(f"\n[沟通优化] 正在调用沟通优化大师分析: “{text[:40]}...”")
        response = gemini_model.generate_content(prompt, request_options={'timeout': 120})
        optimized_text = response.text.strip()
        pyperclip.copy(optimized_text)
        print(f"[沟通优化] 优化方案已生成 (已复制):\n---\n{optimized_text}\n---")
        threading.Thread(target=save_log_to_notion, args=("沟通优化", text, optimized_text), daemon=True).start()
        safe_notification('AI 沟通优化大师', '优化方案已生成并复制！')
    except Exception as e: print(f"!! [沟通优化] 时出错: {e}")
    finally:
        with app_controller.lock: app_controller.is_processing = False


# --- 11 & 12. 招聘工作流 (适配v16.0) ---
def analyze_and_optimize_jd(jd_text):
    print("   -> 正在将JD发送给Gemini进行提炼与优化...")
    prompt = f"""# ROLE: Top-tier Recruitment Strategy Advisor & Information Architect\n# TASK: You MUST perform three critical actions on the provided raw Job Description (JD) text. Your output MUST be a single, valid JSON object and nothing else. All three keys (`position_title`, `analysis_notes`, `optimized_jd`) MUST be present and contain non-empty string values.\n# INSTRUCTIONS:\n1. **`position_title`**: Extract the precise job title from the JD.\n2. **`analysis_notes`**: Act as a consultant. Provide a detailed analysis of the JD in Chinese.\n3. **`optimized_jd`**: Rewrite and optimize the original JD in Chinese based on your analysis.\n# INPUT (Raw JD):\n{jd_text}\n# OUTPUT (Strict JSON format required):\n{{ "position_title": "string", "analysis_notes": "string", "optimized_jd": "string" }}"""
    try:
        response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"}, request_options={'timeout': 180})
        result = json.loads(response.text)
        result["original_jd"] = jd_text
        print(f"   ✅ Gemini处理完成！提炼出的职位名称为: '{result.get('position_title')}'")
        return result
    except Exception as e:
        print(f"   !! 调用Gemini时出错: {e}"); return None

def create_jd_page_in_notion(analysis_result):
    print("   -> 正在Notion中创建新页面并填充内容...")
    title = analysis_result.get("position_title") or f"剪贴板JD {datetime.now().strftime('%H:%M')}"
    try:
        notion = notion_client.Client(auth=NOTION_TOKEN)
        properties = {"职位名称": {"title": [{"text": {"content": title}}]}, "状态": {"select": {"name": "Open"}}}
        children = [ {"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "原始JD文本"}}]}}, {"type": "code", "code": {"rich_text": [{"text": {"content": analysis_result.get("original_jd","")[:1990]}}], "language": "plain text"}}, {"type": "divider", "divider": {}}, {"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "🤖 AI招聘顾问分析与建议"}}]}}, {"type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": analysis_result.get("analysis_notes","")[:1990]}}]}}, {"type": "heading_3", "heading_3": {"rich_text": [{"text": {"content": "优化后的JD版本"}}]}}, {"type": "code", "code": {"rich_text": [{"text": {"content": analysis_result.get("optimized_jd","")[:1990]}}], "language": "plain text"}} ]
        new_page = notion.pages.create(parent={"database_id": JD_HUB_DATABASE_ID}, properties=properties, children=children)
        page_id = new_page.get("id")
        analysis_result["final_position_title"] = title
        print(f"   ✅ Notion页面创建成功！(ID: {page_id})")
        return page_id, analysis_result
    except Exception as e:
        print(f"   !! 创建Notion页面时出错: {e}"); return None, analysis_result

def process_clipboard_jd(text):
    """处理剪贴板中JD文本的完整流程。"""
    with app_controller.lock:
        if app_controller.is_processing: print("!! [JD处理] 系统正忙，请稍后。"); return
        app_controller.is_processing = True
    try:
        print("\n>>>>> (热键触发) 开始处理JD文本 <<<<<")
        if not text or len(text.strip()) < 50:
            print("!! [JD处理] 错误：内容过短或为空。"); return
        analysis_result = analyze_and_optimize_jd(text)
        if not analysis_result: return
        page_id, result = create_jd_page_in_notion(analysis_result)
        if not page_id: return
        
        # 【v16.0 修改】异步写入Qdrant招聘知识库
        knowledge_text = result.get("optimized_jd") or result.get("original_jd")
        if knowledge_text:
            metadata = {"source_db": "job_description", "task_type": "jd_indexing", "source_title": result.get("final_position_title"), "original_id": page_id}
            threading.Thread(target=add_knowledge_to_qdrant, args=("ai_recruitment_assistant", page_id, knowledge_text, metadata), daemon=True).start()

        print(f"\n>>>>> JD '{result.get('final_position_title')}' 已处理并全自动归档完毕！ <<<<<")
        safe_notification('JD处理完成', f"职位 '{result.get('final_position_title')}' 已归档")
    finally:
        with app_controller.lock: app_controller.is_processing = False

# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.0 主程序入口 (pynput 热键监听) 】】】 ⬇⬇⬇
# ==============================================================================
# ==============================================================================
#      ⬇⬇⬇ 【【【 v16.2 主程序入口 (Watchdog 文件信号版) 】】】 ⬇⬇⬇
# ==============================================================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 70)
    print("  AI Ecosystem Core v16.2 - 文件信号增强版")
    print("=" * 70)
    
    if not WATCHDOG_AVAILABLE:
        input("!! [严重错误] watchdog 库未安装！请运行 'pip install watchdog' 后重试。")
        sys.exit(1)

    print(">> [系统] 正在初始化 API...")
    if not setup_api():
        input("API初始化失败，请检查配置。按回车退出。")
        sys.exit(1)

    print("\n>> [系统] 正在初始化向量数据库...")
    setup_database_and_embedding()
    if not IS_QDRANT_DB_AVAILABLE:
        input("向量数据库初始化失败，RAG功能将不可用。按回车退出。")
        sys.exit(1)

    # 创建并清理信号站目录
    IPC_DIR.mkdir(exist_ok=True)
    for file_path in TRIGGER_FILES.values():
        if file_path.exists():
            file_path.unlink()
    
    # 启动 Watchdog 观察者
    event_handler = FileTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, str(IPC_DIR), recursive=False)
    observer.start()

    print("\n" + "=" * 70)
    print("  [系统就绪] 后台服务已启动，正在通过 Watchdog 监听信号...")
    print("  请在系统设置中，将您喜欢的快捷键绑定到下面的 'touch' 命令：")
    print(f"  - RAG 问答:     touch {TRIGGER_FILES['rag_qa']}")
    print(f"  - 智能回复:     touch {TRIGGER_FILES['smart_reply']}")
    print(f"  - 文本分析:     touch {TRIGGER_FILES['text_analysis']}")
    print(f"  - 内容优化:     touch {TRIGGER_FILES['optimize_post']}")
    print(f"  - 快速翻译:     touch {TRIGGER_FILES['translate']}")
    print(f"  - 沟通优化:     touch {TRIGGER_FILES['optimize_comm']}")
    print(f"  - 快速剪藏:     touch {TRIGGER_FILES['quick_save']}")
    print(f"  - JD 处理:      touch {TRIGGER_FILES['process_jd']}")
    print(f"  - 截图分析:     touch {TRIGGER_FILES['analyze_screenshot']}")
    print(f"  - 截图OCR:      touch {TRIGGER_FILES['quick_ocr']}")
    print("\n  按 Ctrl+C 即可安全退出程序。")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>> [系统] 收到退出指令，正在关闭...")
    finally:
        observer.stop()
        observer.join()
        print(">> [系统] 已安全退出。")