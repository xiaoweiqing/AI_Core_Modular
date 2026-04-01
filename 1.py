# core/ai_services.py (最终修复版 V5 - 强制代理注入 & 真正本地LLM)

import requests 
from typing import Optional 
import os
import traceback
import asyncio
from pathlib import Path
from langchain_core.messages import HumanMessage

# 【【【 1. 导入所有正确的库，包括新增的 httpx 和 LlamaCpp 】】】
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
# --- [修改] 我们不再需要 ChatOpenAI，而是需要 LlamaCpp ---
from langchain_community.llms import LlamaCpp 
from sentence_transformers import SentenceTransformer
from config import settings, ACTIVE_AI_PROVIDER, CURRENT_MODEL_NAME, ACTIVE_EMBEDDING_CONFIG
from core import state
from utils.helpers import Colors

# ==============================================================================
#      【【【 Master Setup & Runner (保持不变) 】】】
# ==============================================================================
async def setup_api():
    if ACTIVE_AI_PROVIDER == "google":
        return await _setup_google_api_langchain()
    elif ACTIVE_AI_PROVIDER == "local":
        # --- [修改] 这里现在会调用我们新的本地加载函数 ---
        return await _setup_local_llm_direct_vulkan() 
    else:
        print(f"❌ {Colors.RED}[Config Error] Invalid AI Provider: '{ACTIVE_AI_PROVIDER}'{Colors.ENDC}")
        return False

# ==============================================================================
#      【【【 run_ai_task 函数 (保持不变) 】】】
#      这个函数的设计非常棒，无需任何改动就能兼容新的本地模型对象
# ==============================================================================
async def run_ai_task(prompt: str, provider: Optional[str] = None) -> str | None:
    try:
        llm_instance = None
        target_provider = provider or ACTIVE_AI_PROVIDER

        if target_provider == "google":
            llm_instance = state.llm_google
        elif target_provider == "local":
            llm_instance = state.llm_local

        if not llm_instance:
            print(f"❌ {Colors.RED}[AI Task Error] LLM for '{target_provider}' not ready.{Colors.ENDC}")
            return None

        messages = [HumanMessage(content=prompt)]
        response = await llm_instance.ainvoke(messages)
        
        # --- [小修正] .content 可能不是字符串，做个安全转换 ---
        cleaned_text = response.content.strip() if isinstance(response.content, str) else str(response)
        
        if "</think>" in cleaned_text: _, cleaned_text = cleaned_text.split("</think>", 1)
        if "<|channel|>final<|message|>" in cleaned_text: _, cleaned_text = cleaned_text.split("<|channel|>final<|message|>", 1)
        
        return cleaned_text.strip()
        
    except Exception as e:
        print(f"❌ {Colors.RED}[AI Task Error] Task failed for '{target_provider}': {e}{Colors.ENDC}")
        traceback.print_exc()
    return None

# ==============================================================================
#      【【【 4. Google Gemini 的最终实现 (保持不变) 】】】
# ==============================================================================
async def _setup_google_api_langchain():
    print(f"{Colors.BLUE}>> [AI Service] Initializing Google Gemini (Back-to-Basics Proxy Mode)...{Colors.ENDC}")
    if not settings.google_ai_key:
        print(f"❌ {Colors.RED}[AI Error] GOOGLE_AI_KEY not found in .env.{Colors.ENDC}")
        return False
    try:
        def _connect_sync():
            proxy_url = settings.https_proxy or settings.http_proxy
            if not proxy_url:
                raise ValueError("Proxy URL (HTTPS_PROXY) not found in .env file, which is required for Google API mode.")
            print(f">> [Proxy Force-Inject] Setting OS environment proxy to: {proxy_url}")
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['HTTP_PROXY'] = proxy_url
            
            print(f">> [Google AI] Configuring with model '{settings.GOOGLE_MODEL_NAME}'...")
            llm = ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL_NAME,
                temperature=0.7,
                google_api_key=settings.google_ai_key,
                request_timeout=60
            )
            return llm

        state.llm_google = await asyncio.to_thread(_connect_sync)
        print(f"✅ {Colors.GREEN}[AI Service] Google Gemini configured successfully! (Model: {settings.GOOGLE_MODEL_NAME}){Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[AI Error] Failed to configure Google Gemini: {e}{Colors.ENDC}")
        traceback.print_exc()
        return False

# ==============================================================================
#      【【【 5. 本地 LLM 的实现 (这是唯一被替换的部分) 】】】
# ==============================================================================
async def _setup_local_llm_direct_vulkan():
    """
    [MODIFIED] This function REPLACES the old _setup_local_api.
    It directly loads the GGUF model using LlamaCpp with Vulkan acceleration,
    making it truly offline and serverless.
    """
    MODEL_PATH = "/mnt/data/model/Qwen3-Coder-30B-A3B-Instruct-UD-TQ1_0.gguf"
    
    print(f"{Colors.BLUE}>> [AI Service] Initializing TRUE LOCAL LLM via Vulkan...{Colors.ENDC}")
    print(f">> [AI Service] Model Path: {MODEL_PATH}")

    try:
        def _load_model_sync():
            print(">> [Proxy Cleaner] Local mode: removing any system proxy settings...")
            for proxy_var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
                if proxy_var in os.environ: del os.environ[proxy_var]
            
            # 使用 LlamaCpp 包装器使其与 LangChain 兼容
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                # --- 这是您在启动脚本中设置的关键参数 ---
                n_ctx=62144,
                n_gpu_layers=99,      # 强制使用 Vulkan/GPU 加速
                
                # --- 建议的性能和采样参数 ---
                n_batch=512,          # 提高提示处理速度
                temperature=0.3,      # 匹配您原来的设置
                
                # --- 其他设置 ---
                verbose=True,         # 在日志中确认 Vulkan 设备是否被找到
                streaming=False,
            )
            return llm

        state.llm_local = await asyncio.to_thread(_load_model_sync)
        print(f"✅ {Colors.GREEN}[AI Service] True Local LLM loaded and configured.{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[AI Error] CRITICAL - Failed to load local GGUF model: {e}{Colors.ENDC}")
        traceback.print_exc()
        return False

# ==============================================================================
#      【【【 6. Embedding 模型加载 (保持不变) 】】】
#      (您有两个同名函数，我将它们都保留下来)
# ==============================================================================
async def setup_embedding_model():
    """[ASYNC] 根据 config.py 中的总开关，加载指定的本地模型。"""
    try:
        model_path = ACTIVE_EMBEDDING_CONFIG["path"]
        if not Path(model_path).is_dir():
            print(f"❌ {Colors.RED}[Embedding Error] Model folder not found at '{model_path}'!{Colors.ENDC}")
            return False

        print(f">> [Async Embedding] Loading model '{CURRENT_MODEL_NAME}' from: '{model_path}'...")
        state.EMBEDDING_MODEL = await asyncio.to_thread(SentenceTransformer, model_path)
        print(f"✅ {Colors.GREEN}[Async Embedding] Model '{CURRENT_MODEL_NAME}' is ready.{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[Async Embedding Error] Failed to load local model: {e}{Colors.ENDC}")
        return False

async def generate_text_vector(text: str) -> list[float] | None:
    """[ASYNC] 生成文本的向量嵌入。"""
    try:
        if not state.EMBEDDING_MODEL:
            print(f"{Colors.RED}[Async Embedding Error] Embedding model not initialized.{Colors.ENDC}")
            return None
        full_vector = await asyncio.to_thread(state.EMBEDDING_MODEL.encode, text)
        return full_vector.tolist()
    except Exception as e:
        print(f"❌ {Colors.RED}[Async Embedding Error] Failed to encode text: {e}{Colors.ENDC}")
        traceback.print_exc()
        return None

# ==============================================================================
#      【【【 NEW: Google Search API Service (保持不变) 】】】
# ==============================================================================
async def google_search_task(query: str) -> str:
    print("   -> [API Call] Performing Google Search...")
    api_key = settings.google_search_api_key
    cx = settings.google_search_cx
    if not api_key or not cx:
        return "Error: Google Search API Key or CX is not configured in .env file."
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cx, 'q': query, 'num': 5}
    def _sync_search():
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            if "items" not in search_results:
                return "No search results found."
            formatted_context = ""
            for i, item in enumerate(search_results["items"]):
                title = item.get('title', 'No Title')
                snippet = item.get('snippet', 'No Snippet').replace('\n', ' ')
                formatted_context += f"Result [{i+1}]: {title}\nSnippet: {snippet}\n\n"
            return formatted_context
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to connect to Google Search API. Details: {e}"
        except Exception as e:
            return f"Error: An unexpected error occurred during the search. Details: {e}"
    return await asyncio.to_thread(_sync_search)
