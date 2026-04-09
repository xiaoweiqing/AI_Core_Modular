#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#      Apple FastVLM 截图深度分析工具 v1.1 (文件信号触发版)
# ==============================================================================
# v1.1 更新:
# - 【架构升级】: 移除 pynput 库，不再监听全局热键。
# - 【触发方式】: 改为使用 watchdog 库监控指定目录下的信号文件。
# - 【系统集成】: 用户可在操作系统中自定义任何快捷键，并绑定命令来触发本程序。
# ==============================================================================

import os
import sys
import threading
import subprocess
import time
import uuid
from pathlib import Path

# --- 尝试导入核心库 ---
try:
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError as e:
    print(f"!! [严重错误] 缺少关键的Python库: {e.name}")
    print("   -> 请确保您已经通过 'pip install -r requirements.txt' 安装了所有依赖。")
    sys.exit(1)

# --- 全局配置 ---
class Colors:
    RESET = '\033[0m'; BOLD = '\033[1m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    RED = '\033[91m'; CYAN = '\033[96m'; MAGENTA = '\033[95m'

# 【【【 新增：定义IPC信号文件路径 】】】
IPC_DIR = Path.home() / ".cache" / "fastvlm_analyzer"
TRIGGER_FILE = IPC_DIR / "trigger_analysis"

# --- 全局变量 ---
VLM_MODEL, VLM_TOKENIZER = None, None

# --- 应用状态控制器 ---
class AppController:
    def __init__(self):
        self.is_processing = False
        self.lock = threading.Lock()

app_controller = AppController()

# ==============================================================================
# --- 核心功能模块 (这部分函数与v1.0版本完全相同) ---
# ==============================================================================
def setup_vlm_model():
    global VLM_MODEL, VLM_TOKENIZER
    try:
        print(f"{Colors.CYAN}>> [VLM] 正在加载 Apple's FastVLM-0.5B 模型...{Colors.RESET}")
        model_id = "apple/FastVLM-0.5B"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        print(f">> [VLM] 检测到设备: {device.upper()}, 使用数据类型: {str(torch_dtype)}")
        VLM_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        VLM_MODEL = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
        print(f"{Colors.GREEN}✅ [VLM] FastVLM 模型加载成功！{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ [VLM] 严重错误: 加载 FastVLM 模型失败: {e}{Colors.RESET}")
        return False

def capture_screen_area():
    try:
        temp_file_path = f"/tmp/fastvlm_screenshot_{uuid.uuid4()}.png"
        subprocess.run(['gnome-screenshot', '-a', '-f', temp_file_path], check=True)
        if os.path.exists(temp_file_path):
            with Image.open(temp_file_path) as img: img_copy = img.copy()
            os.remove(temp_file_path)
            return img_copy
        return None
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"{Colors.YELLOW}>> [截图工具] 截图操作已取消或失败。{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}!! [截图工具] 截图时发生未知错误: {e}{Colors.RESET}")
        return None
def analyze_screenshot_with_fastvlm(img: Image.Image):
    """
    使用加载的 FastVLM 模型对 PIL 图像进行分析。
    提供了两种模式：纯OCR 和 轻量级分析。
    """
    with app_controller.lock:
        app_controller.is_processing = True

    try:
        print(f"\n{Colors.CYAN}>> [FastVLM] 收到截图，开始进行分析...{Colors.RESET}")
        
        # ======================================================================
        # --- 【【【 您可以在这里二选一 】】】 ---
        # ======================================================================

        # 【方案一：终极OCR模式】 - 最快、最可靠
        # 任务：只提取所有文字。
        vlm_prompt_text = "<image>\n请提取并返回这张图片中的所有文字，只返回纯文本，不要任何其他内容。"
        
        # ----------------------------------------------------------------------

        # 【方案二：轻量级结构化分析模式】 - 推荐尝试
        # 任务：提取文字，并做一个非常简单的总结。指令具体且简单。
        # vlm_prompt_text = (
        #     "<image>\n"
        #     "1. 提取图片中的所有文字。\n"
        #     "2. 用一句话总结这张截图的核心内容。"
        # )

        # ======================================================================
        
        # --- FastVLM 推理逻辑 (保持不变) ---
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
                max_new_tokens=4096, # 增加token以应对超长文本
                do_sample=False,
            )
        
        response_text = tok.decode(out[0], skip_special_tokens=True)
        # 从 prompt 的末尾开始截取，获得最干净的输出
        clean_response = response_text.split(vlm_prompt_text)[-1].strip()
        
        print("\n" + "="*25 + f" {Colors.BOLD}{Colors.MAGENTA}Apple FastVLM 分析结果{Colors.RESET} " + "="*25)
        print(clean_response)
        print("="*78 + "\n")
        print(f"{Colors.YELLOW}>> 分析完成。等待下一次快捷键信号...{Colors.RESET}")

    except Exception as e:
        print(f"{Colors.RED}!! [FastVLM 分析] AI处理时出错: {e}{Colors.RESET}")
    finally:
        with app_controller.lock:
            app_controller.is_processing = False

def trigger_analysis():
    if app_controller.is_processing:
        print(f"{Colors.YELLOW}!! [系统] 正忙，请等待上一个分析任务完成。{Colors.RESET}")
        return
    print(f"\n{Colors.GREEN}>> [信号触发] 请用鼠标拖拽选择截图区域 (按Esc可取消)...{Colors.RESET}")
    captured_image = capture_screen_area()
    if captured_image:
        threading.Thread(target=analyze_screenshot_with_fastvlm, args=(captured_image,), daemon=True).start()

# ==============================================================================
# --- 【【【 新增：Watchdog 文件信号处理器 】】】 ---
# ==============================================================================
class FileTriggerHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and Path(event.src_path) == TRIGGER_FILE:
            print(f"\n>> [信号] 收到任务信号: {TRIGGER_FILE.name}")
            trigger_analysis()
            try:
                # 立即删除信号文件，为下一次触发做准备
                time.sleep(0.1)
                os.unlink(event.src_path)
            except OSError:
                pass

# ==============================================================================
# --- 【【【 主程序入口 (Watchdog 监听版) 】】】 ---
# ==============================================================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print(f"  {Colors.BOLD}Apple FastVLM 截图深度分析工具 v1.1 (文件信号触发版){Colors.RESET}")
    print("=" * 70)
    
    if not setup_vlm_model():
        input("核心模型初始化失败，程序无法启动。按回车退出。")
        sys.exit(1)
    
    # 创建并清理信号站目录
    IPC_DIR.mkdir(exist_ok=True)
    if TRIGGER_FILE.exists():
        TRIGGER_FILE.unlink()
    
    # 启动 Watchdog 观察者
    event_handler = FileTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, str(IPC_DIR), recursive=False)
    observer.start()

    print("\n" + "=" * 70)
    print(f"  {Colors.GREEN}[系统就绪] 后台服务已启动，正在通过 Watchdog 监听信号...{Colors.RESET}")
    print(f"  - 监控目录: {IPC_DIR}")
    print(f"  - 信号文件: {TRIGGER_FILE.name}")
    print(f"  - {Colors.BOLD}请在系统设置中，将您的快捷键 (如 Alt+S) 绑定到下面的命令。{Colors.RESET}")
    print(f"  - 在终端窗口按下 {Colors.BOLD}Ctrl + C{Colors.RESET} 即可安全退出程序。")
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