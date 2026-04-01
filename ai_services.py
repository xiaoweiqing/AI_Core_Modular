import requests # <--- MAKE SURE THIS LINE IS PRESENT
from typing import Optional # <--- AND THIS ONE FOR THE NEXT STEP
import os
import traceback
import asyncio
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from config import settings, ACTIVE_AI_PROVIDER, CURRENT_MODEL_NAME, ACTIVE_EMBEDDING_CONFIG
from core import state
from utils.helpers import Colors

async def setup_api():
    # 【【【 错误已修复 】】】 下面的代码块已经正确缩进
    if ACTIVE_AI_PROVIDER == "google":
        return await _setup_google_api_langchain()
    elif ACTIVE_AI_PROVIDER == "local":
        return await _setup_local_api()
    else:
        print(f"❌ {Colors.RED}[Config Error] Invalid AI Provider: '{ACTIVE_AI_PROVIDER}'{Colors.ENDC}")
        return False

async def run_ai_task(prompt: str, provider: Optional[str] = None) -> str | None:
    """
    [MODIFIED] Now accepts an optional 'provider' argument to override the global setting.
    - This allows special features like Power Search to force the use of the Google model.
    - Uses LangChain v1.0's recommended asynchronous method ainvoke.
    """
    try:
        # Determine which LLM instance to use
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
        
        # Use ainvoke for true async operation
        response = await llm_instance.ainvoke(messages)
        
        cleaned_text = response.content.strip()
        
        # Compatibility for local models that might add extra tags
        if "</think>" in cleaned_text: _, cleaned_text = cleaned_text.split("</think>", 1)
        if "<|channel|>final<|message|>" in cleaned_text: _, cleaned_text = cleaned_text.split("<|channel|>final<|message|>", 1)
        
        return cleaned_text.strip()
        
    except Exception as e:
        print(f"❌ {Colors.RED}[AI Task Error] Task failed for '{target_provider}': {e}{Colors.ENDC}")
        traceback.print_exc()
        return None

# ==============================================================================
# 【【【 4. Google Gemini 的最终实现 (强制代理注入) 】】】
# ==============================================================================
async def _setup_google_api_langchain():
    """
    【【【 终极返璞归真版 V9 - 绝对可靠方案 】】】
    - 放弃所有手动的 httpx.Client 注入，因为这可能与 langchain 库存在兼容性问题。
    - 回归到最简单、最有效的方法：在初始化 LLM 之前，强制设置环境变量。
    - 这直接模仿了您那个“好用的程序”的成功模式。
    """
    print(f"{Colors.BLUE}>> [AI Service] Initializing Google Gemini (Back-to-Basics Proxy Mode)...{Colors.ENDC}")
    if not settings.google_ai_key:
        print(f"❌ {Colors.RED}[AI Error] GOOGLE_AI_KEY not found in .env.{Colors.ENDC}")
        return False

    try:
        def _connect_sync():
            proxy_url = settings.https_proxy or settings.http_proxy
            if not proxy_url:
                raise ValueError("Proxy URL (HTTPS_PROXY) not found in .env file, which is required for Google API mode.")

            # ==============================================================================
            #      【【【 这是唯一且全部的逻辑：强制设置环境变量 】】】
            # ==============================================================================
            print(f">> [Proxy Force-Inject] Setting OS environment proxy to: {proxy_url}")
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['HTTP_PROXY'] = proxy_url
            # ==============================================================================
            
            print(f">> [Google AI] Configuring with model '{settings.GOOGLE_MODEL_NAME}'...")
            
            # 【【【 关键改动：不再创建和传递 client 对象！ 】】】
            # 我们让 LangChain 自己处理网络，它会自动读取上面设置的环境变量。
            llm = ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL_NAME,
                temperature=0.7,
                google_api_key=settings.google_ai_key,
                request_timeout=60 # 设置一个合理的超时时间
            )
            
            return llm

        state.llm_google = await asyncio.to_thread(_connect_sync)
        print(f"✅ {Colors.GREEN}[AI Service] Google Gemini configured successfully! (Model: {settings.GOOGLE_MODEL_NAME}){Colors.ENDC}")
        return True
        
    except Exception as e:
        print(f"❌ {Colors.RED}[AI Error] Failed to configure Google Gemini: {e}{Colors.ENDC}")
        print(f"   {Colors.YELLOW}Please double-check your proxy service is running and configured correctly in .env.{Colors.ENDC}")
        traceback.print_exc()
        return False

# ==============================================================================
# 【【【 5. 本地 LLM 的实现 (保持不变) 】】】
# ==============================================================================
async def _setup_local_api():
    # 这个函数已经是正确的，保持不变
    print(f"{Colors.BLUE}>> [AI Service] Connecting to local model: {settings.local_api_url}{Colors.ENDC}")
    try:
        def _connect_sync():
            print(">> [Proxy Cleaner] Local mode: removing system proxy settings...")
            for proxy_var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
                if proxy_var in os.environ: del os.environ[proxy_var]

            llm = ChatOpenAI(
                openai_api_base=settings.local_api_url,
                openai_api_key="na", model_name="local-model",
                temperature=0.3, request_timeout=300
            )
            return llm
            
        state.llm_local = await asyncio.to_thread(_connect_sync)
        print(f"✅ {Colors.GREEN}[AI Service] Connection to local LLM configured.{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[AI Error] Failed to connect to local model server: {e}{Colors.ENDC}")
        return False

# ==============================================================================
# 【【【 6. Embedding 模型加载 (保持不变) 】】】
# =====================================================================
async def setup_embedding_model():
    """[ASYNC] 根据 config.py 中的总开关，加载指定的本地模型。"""
    try:
        # 从我们的“控制中心”动态读取模型路径
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
    """
    [ASYNC] 生成文本的向量嵌入。
    这个版本会直接返回模型生成的、未经修改的完整向量。
    """
    try:
        if not state.EMBEDDING_MODEL:
            print(f"{Colors.RED}[Async Embedding Error] Embedding model not initialized.{Colors.ENDC}")
            return None

        # 直接生成并返回完整的向量
        full_vector = await asyncio.to_thread(state.EMBEDDING_MODEL.encode, text)

        return full_vector.tolist()

    except Exception as e:
        print(f"❌ {Colors.RED}[Async Embedding Error] Failed to encode text: {e}{Colors.ENDC}")
        traceback.print_exc()
        return None

# ==============================================================================
# 【【【 NEW: Google Search API Service 】】】
# ==============================================================================
async def google_search_task(query: str) -> str:
    """
    [ASYNC] Performs a Google search and returns a formatted string of results.
    """
    print("   -> [API Call] Performing Google Search...")
    api_key = settings.google_search_api_key
    cx = settings.google_search_cx

    if not api_key or not cx:
        return "Error: Google Search API Key or CX is not configured in .env file."

    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cx, 'q': query, 'num': 5} # Fetch top 5 results

    def _sync_search():
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
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
            print(f"❌ {Colors.RED}[Google Search Error] Network or API error: {e}{Colors.ENDC}")
            return f"Error: Failed to connect to Google Search API. Details: {e}"
        except Exception as e:
            print(f"❌ {Colors.RED}[Google Search Error] An unexpected error occurred: {e}{Colors.ENDC}")
            traceback.print_exc()
            return f"Error: An unexpected error occurred during the search. Details: {e}"

    return await asyncio.to_thread(_sync_search)
