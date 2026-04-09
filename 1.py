#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#      Apple FastVLM 截图深度分析工具 v1.0
# ==============================================================================
# 功能说明:
# - 核心引擎: 搭载 Apple 的 FastVLM-0.5B 多模态视觉语言模型。
# - 触发方式: 在后台监听全局热键 (Ctrl+Alt+S)，一键启动分析。
# - 截图工具: 调用 Linux 原生的 'gnome-screenshot' 进行区域截图，稳定可靠。
# - 深度分析: 不仅仅是OCR，更能理解图片内容、总结核心信息、提供行动洞察。
# - 独立运行: 这是一个专门的工具，无任何外部数据库或API依赖。
# ==============================================================================

import os
import sys
import threading
import subprocess
import time
import uuid

# --- 尝试导入核心库，如果失败则提供清晰的指引 ---
try:
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pynput import keyboard
except ImportError as e:
    print(f"!! [严重错误] 缺少关键的Python库: {e.name}")
    print("   -> 请确保您已经通过 'pip install -r requirements.txt' 安装了所有依赖。")
    sys.exit(1)

# --- 全局配置 ---
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'

# 定义触发截图分析的热键
HOTKEY_COMBINATION = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode.from_char('s')}
current_keys = set()

# --- 全局变量 ---
VLM_MODEL, VLM_TOKENIZER = None, None

# --- 应用状态控制器 ---
class AppController:
    def __init__(self):
        self.is_processing = False
        self.lock = threading.Lock()

app_controller = AppController()

# ==============================================================================
# --- 核心功能模块 ---
# ==============================================================================

def setup_vlm_model():
    """
    在程序启动时加载 Apple FastVLM 多模态模型。
    首次运行时会自动从Hugging Face下载，可能需要较长时间。
    """
    global VLM_MODEL, VLM_TOKENIZER
    try:
        print(f"{Colors.CYAN}>> [VLM] 正在加载 Apple's FastVLM-0.5B 模型...{Colors.RESET}")
        print("   (首次运行会自动下载模型，根据网络情况可能需要几分钟，请耐心等待)")
        
        model_id = "apple/FastVLM-0.5B"
        
        # 自动选择设备 (GPU > CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f">> [VLM] 检测到设备: {device.upper()}, 使用数据类型: {str(torch_dtype)}")

        VLM_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        VLM_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        
        print(f"{Colors.GREEN}✅ [VLM] FastVLM 模型加载成功！{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ [VLM] 严重错误: 加载 FastVLM 模型失败: {e}{Colors.RESET}")
        print("   -> 请检查您的网络连接是否可以访问 Hugging Face。")
        return False

def capture_screen_area():
    """
    调用 gnome-screenshot 进行区域截图，返回一个 PIL.Image 对象或 None。
    """
    try:
        temp_file_path = f"/tmp/fastvlm_screenshot_{uuid.uuid4()}.png"
        subprocess.run(['gnome-screenshot', '-a', '-f', temp_file_path], check=True)
        
        if os.path.exists(temp_file_path):
            with Image.open(temp_file_path) as img:
                img_copy = img.copy()
            os.remove(temp_file_path)
            return img_copy
        return None
    except FileNotFoundError:
        print(f"{Colors.RED}!! [截图工具] 错误: 'gnome-screenshot' 命令未找到。{Colors.RESET}")
        return None
    except subprocess.CalledProcessError:
        print(f"{Colors.YELLOW}>> [截图工具] 截图操作已取消。{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}!! [截图工具] 截图时发生未知错误: {e}{Colors.RESET}")
        return None

def analyze_screenshot_with_fastvlm(img: Image.Image):
    """
    使用加载的 FastVLM 模型对 PIL 图像进行深度分析。
    """
    with app_controller.lock:
        app_controller.is_processing = True

    try:
        print(f"\n{Colors.CYAN}>> [FastVLM] 收到截图，开始进行深度分析...{Colors.RESET}")
        
        # --- 改造自 v16.0 的强大分析 Prompt ---
        vlm_prompt_text = (
            "<image>\n"
            "# 角色: 全能视觉分析与策略洞察专家\n"
            "# 核心任务: 接收一张截图，从多个维度进行深度分析，并输出你的洞察。\n"
            "# 分析维度:\n"
            "1. **文本提取**: 提取图片中所有清晰可读的中文和英文文本。\n"
            "2. **核心摘要**: 综合图片内容和文本，用一句话总结截图的核心信息。\n"
            "3. **行动洞察**: 这是最重要的部分。发掘出潜在的机会、风险、优化点或下一步可以采取的具体行动。\n"
            "# 请开始你的分析:"
        )
        
        # --- FastVLM 推理逻辑 (源自 v27.0) ---
        tok = VLM_TOKENIZER
        model = VLM_MODEL
        IMAGE_TOKEN_INDEX = -200
        
        messages = [{"role": "user", "content": vlm_prompt_text}]
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        
        input_ids = torch.cat([
            tok(pre, return_tensors="pt", add_special_tokens=False).input_ids,
            torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=torch.long),
            tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        ], dim=1).to(model.device)
        
        px = model.get_vision_tower().image_processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
        px = px.to(model.device, dtype=model.dtype)
        
        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                images=px,
                max_new_tokens=2048,
                do_sample=False, # 使用确定性解码
            )
        
        # 清理和格式化输出
        response_text = tok.decode(out[0], skip_special_tokens=True)
        # 找到prompt的结束位置，只取之后的内容
        clean_response = response_text.split("请开始你的分析:")[-1].strip()
        
        print("\n" + "="*25 + f" {Colors.BOLD}{Colors.MAGENTA}Apple FastVLM 分析结果{Colors.RESET} " + "="*25)
        print(clean_response)
        print("="*78 + "\n")
        print(f"{Colors.YELLOW}>> 分析完成。再次按下 {Colors.BOLD}Ctrl+Alt+S{Colors.RESET}{Colors.YELLOW} 可进行新的分析。{Colors.RESET}")

    except Exception as e:
        print(f"{Colors.RED}!! [FastVLM 分析] AI处理时出错: {e}{Colors.RESET}")
    finally:
        with app_controller.lock:
            app_controller.is_processing = False

def trigger_analysis():
    """
    由热键调用的主流程函数。
    """
    if app_controller.is_processing:
        print(f"{Colors.YELLOW}!! [系统] 正忙，请等待上一个分析任务完成。{Colors.RESET}")
        return

    print(f"\n{Colors.GREEN}>> [热键触发] 请用鼠标拖拽选择截图区域 (按Esc可取消)...{Colors.RESET}")
    
    # 截图操作可能会阻塞，但这是预期的行为
    captured_image = capture_screen_area()
    
    if captured_image:
        # 在新线程中运行分析，以防万一有任何阻塞，不影响键盘监听器
        analysis_thread = threading.Thread(target=analyze_screenshot_with_fastvlm, args=(captured_image,), daemon=True)
        analysis_thread.start()

# --- 热键监听器回调 ---
def on_press(key):
    if key in HOTKEY_COMBINATION:
        current_keys.add(key)
        if all(k in current_keys for k in HOTKEY_COMBINATION):
            trigger_analysis()

def on_release(key):
    try:
        current_keys.remove(key)
    except KeyError:
        pass

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print(f"  {Colors.BOLD}Apple FastVLM 截图深度分析工具 v1.0{Colors.RESET}")
    print("=" * 70)
    
    if not setup_vlm_model():
        input("核心模型初始化失败，程序无法启动。请检查错误信息后重试。按回车退出。")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print(f"  {Colors.GREEN}[系统就绪] 后台服务已启动，正在监听热键...{Colors.RESET}")
    print(f"  - {Colors.BOLD}按下 Ctrl + Alt + S{Colors.RESET} 即可开始截图分析。")
    print(f"  - 在终端窗口按下 {Colors.BOLD}Ctrl + C{Colors.RESET} 即可安全退出程序。")
    print("=" * 70 + "\n")
    
    # 启动热键监听器
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\n>> [系统] 收到退出指令，正在关闭...")
            print(">> [系统] 已安全退出。")