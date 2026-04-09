# ==============================================================================
#      沉浸式翻译器 v2.0 (Super Translator)
# ==============================================================================
# 功能:
# - 这是一个独立的、由热键触发的翻译工具。
# - 【核心功能】: 翻译完成后，会自动模拟键盘“粘贴”操作，
#   用翻译结果“原地替换”掉您选中的原文。
# - 提供两个独立的热键，分别用于翻译成中文和英文。
# - 自动获取鼠标选中的文本，无需手动复制。
#
# 作者: Gemini
# ==============================================================================

import os
import re
import sys
import threading
import subprocess
import pyperclip
import google.generativeai as genai
from dotenv import load_dotenv
from pynput import keyboard

# ==============================================================================
#      1. 【【【 用户配置区 】】】
# ==============================================================================

# --- 加载环境变量 ---
# 这会从脚本所在目录的 .env 文件中加载配置。
load_dotenv()

# --- 你的 Google AI API Key ---
# 请确保你的 .env 文件里有这一行: GOOGLE_AI_KEY="你的API密钥"
API_KEY = os.getenv("GOOGLE_AI_KEY")

# --- 【【【 自定义你的快捷键 】】】 ---
# 你可以使用 'alt', 'ctrl', 'shift', 'cmd' 以及 'f1'-'f12' 或普通字母 'a', 'b', 'c'...
# 注意：组合键的顺序是 <修饰键>+<普通键>
# 示例: '<alt>+t', '<ctrl>+<alt>+j'

# 将选中文字翻译成“英文”的热键
HOTKEY_TO_ENGLISH = "<alt>+e"

# 将选中文字翻译成“中文”的热键
HOTKEY_TO_CHINESE = "<alt>+c"

# ==============================================================================
#      2. 核心组件
# ==============================================================================


# --- 全局状态锁，防止快速连按导致任务冲突 ---
class AppController:
    def __init__(self):
        self.is_processing = False
        self.lock = threading.Lock()


app_controller = AppController()
gemini_model = None
keyboard_controller = keyboard.Controller()


# --- 文本清理 ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)


# --- 桌面通知 (Linux/macOS) ---
def safe_notification(title, message):
    try:
        # 优先使用 notify-send (Linux)
        subprocess.run(
            ["notify-send", title, message, "-a", "SuperTranslator", "-t", "3000"],
            check=True,
        )
    except FileNotFoundError:
        # 回退到 osascript (macOS)
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "{title}"',
                ],
                check=True,
            )
        except FileNotFoundError:
            pass  # 如果两个都不支持，则静默失败


# --- API 初始化 ---
def setup_api():
    global gemini_model
    if not API_KEY or "你的API密钥" in API_KEY:
        print("❌ [严重错误] 未找到或未配置有效的 GOOGLE_AI_KEY！")
        print("   -> 请在脚本同目录下创建一个 .env 文件，并填入你的API密钥。")
        return False
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        print("✅ [API] Google Gemini API 初始化成功。")
        return True
    except Exception as e:
        print(f"❌ [API] API 初始化失败: {e}")
        return False


# ==============================================================================
#      3. 【【【 核心翻译与替换逻辑 】】】
# ==============================================================================


def translate_and_replace(target_language: str):
    """
    完整执行“获取选中 -> 翻译 -> 替换原文”的流程。
    :param target_language: 'en' (英文) or 'zh' (中文)
    """
    # 检查是否已有任务在运行
    with app_controller.lock:
        if app_controller.is_processing:
            print("!! [系统] 正忙，请等待上一个任务完成。")
            safe_notification("翻译器正忙", "请稍后再试")
            return
        app_controller.is_processing = True

    try:
        # 1. 获取选中的文本
        selected_text = ""
        try:
            # 优先尝试获取鼠标高亮的文本 (Linux X11 特有)
            result = subprocess.run(
                ["xclip", "-o", "-selection", "primary"],
                capture_output=True,
                text=True,
                check=True,
            )
            selected_text = result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            # 如果 xclip 失败或没有选中内容，则回退到读取剪贴板
            try:
                selected_text = pyperclip.paste()
            except Exception as e_pyperclip:
                print(f"!! [错误] 读取剪贴板失败: {e_pyperclip}")

        if not selected_text or selected_text.isspace():
            print(">> [系统] 未获取到有效文本。")
            safe_notification("未找到文本", "请先用鼠标选中您想翻译的文字")
            return

        print(
            f"\n>> [翻译任务] 目标语言: {target_language.upper()} | 原文: “{selected_text[:30].strip()}...”"
        )

        # 2. 构建 Prompt 并调用 API
        if target_language == "en":
            prompt = f"Translate the following text to English. Output only the translated text, without any additional explanations or formatting:\n\n{clean_text(selected_text)}"
        else:  # 'zh'
            prompt = f"将以下内容翻译成中文，只输出翻译后的纯文本，不要任何解释或格式:\n\n{clean_text(selected_text)}"

        response = gemini_model.generate_content(
            prompt, request_options={"timeout": 30}
        )
        translated_text = response.text.strip()

        if not translated_text:
            print(f"!! [错误] API返回为空。")
            safe_notification("翻译失败", "AI未能生成翻译结果。")
            return

        print(f"✅ [翻译成功] 结果: “{translated_text[:30].strip()}...”")

        # 3. 将翻译结果放入剪贴板
        pyperclip.copy(translated_text)

        # 4. 【关键步骤】模拟键盘 "Ctrl+V" (粘贴) 操作
        # 这会用剪贴板里的新内容（译文）覆盖掉之前选中的原文
        print(">> [系统] 正在模拟粘贴操作以替换原文...")
        with keyboard_controller.pressed(keyboard.Key.ctrl):
            keyboard_controller.press("v")
            keyboard_controller.release("v")

        safe_notification("翻译并替换成功！", f'"{selected_text[:20]}..." 已被替换')

    except Exception as e:
        print(f"!! [严重错误] 翻译流程出错: {e}")
        safe_notification("翻译失败", str(e))
    finally:
        # 任务完成，释放锁
        with app_controller.lock:
            app_controller.is_processing = False


# ==============================================================================
#      4. 主程序入口与热键监听
# ==============================================================================

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 60)
    print("        沉浸式翻译器 v2.0 - 正在启动...")
    print("=" * 60)

    if not setup_api():
        input("API 初始化失败，请检查配置后重试。按回车退出。")
        sys.exit(1)

    # 定义热键和它们对应的功能
    hotkey_actions = {
        HOTKEY_TO_ENGLISH: lambda: translate_and_replace("en"),
        HOTKEY_TO_CHINESE: lambda: translate_and_replace("zh"),
    }

    print("\n" + "=" * 60)
    print("  ✅ [系统就绪] 翻译器已在后台运行。")
    print("\n  【使用方法】:")
    print("   1. 用鼠标选中任意您想翻译的文字。")
    print(f"   2. 按下热键 '{HOTKEY_TO_ENGLISH}' 将其翻译成英文。")
    print(f"   3. 或按下热键 '{HOTKEY_TO_CHINESE}' 将其翻译成中文。")
    print("\n  原文将被自动替换为翻译结果！")
    print("\n  (在此终端窗口按 Ctrl+C 即可退出程序)")
    print("=" * 60 + "\n")

    # 启动全局热键监听器
    try:
        with keyboard.GlobalHotKeys(hotkey_actions) as listener:
            listener.join()
    except Exception as e:
        print(f"!! [热键错误] 无法启动热键监听器: {e}")
        print("   -> 可能原因：另一个程序占用了快捷键，或者权限不足。")
        input("   按回车键退出。")
