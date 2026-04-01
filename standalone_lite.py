import sys
import os
import sqlite3
import subprocess
from pathlib import Path

# --- 配置 ---
DB_PATH = Path.home() / "phrase_expander.sqlite"
TABLE_NAME = "phrase_mappings"
LAST_ID_FILE = "/tmp/musebox_last_id"

def send_notification(title, message):
    try: subprocess.run(["notify-send", "-t", "2000", title, message])
    except: pass

def get_selection_text():
    try:
        res = subprocess.run(["wl-paste", "--primary", "--no-newline"], capture_output=True, text=True, timeout=0.5)
        if res.returncode == 0 and res.stdout.strip(): return res.stdout.strip()
        res = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True, timeout=0.5)
        return res.stdout.strip()
    except: return ""

def copy_to_clipboard(text):
    try: subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
    except: send_notification("错误", "无法写入剪贴板")

# --- 功能函数 ---
def add_phrase(text):
    if not text:
        send_notification("⚠️ 失败", "选中文本为空")
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"INSERT OR IGNORE INTO {TABLE_NAME} (phrase, full_content) VALUES (?, ?)", (text, "[PENDING]"))
        cursor = conn.execute(f"SELECT id FROM {TABLE_NAME} WHERE phrase = ?", (text,))
        row = cursor.fetchone()
        if row:
            with open(LAST_ID_FILE, "w") as f: f.write(str(row[0]))
            send_notification("✅ 短语锁定", f"关键字: {text[:20]}...")

def add_full_content(text):
    if not os.path.exists(LAST_ID_FILE):
        send_notification("⚠️ 顺序错误", "请先按 Alt+S 锁定短语")
        return
    with open(LAST_ID_FILE, "r") as f: last_id = f.read().strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"UPDATE {TABLE_NAME} SET full_content = ? WHERE id = ?", (text, last_id))
        send_notification("✅ 映射成功", "内容已保存")
    if os.path.exists(LAST_ID_FILE): os.remove(LAST_ID_FILE)

def phrase_expander(phrase):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(f"SELECT full_content FROM {TABLE_NAME} WHERE phrase = ?", (phrase,))
        row = cursor.fetchone()
        if row and row[0]:
            copy_to_clipboard(row[0])
            send_notification("📖 提取成功", "内容已复制")
        else:
            send_notification("❌ 未找到", f"短语: {phrase}")

def delete_phrase_mapping(phrase):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME} WHERE phrase = ?", (phrase,))
        send_notification("🗑️ 已删除", f"映射: {phrase}")

# --- 主逻辑 ---
if __name__ == "__main__":
    # 获取触发文件路径
    trigger_path = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # [关键改进]：一进来先删掉触发文件，防止 systemd 疯狂重启
    if trigger_path and os.path.exists(trigger_path):
        os.remove(trigger_path)

    # 如果没有路径则退出
    if not trigger_path:
        sys.exit(0)

    captured_text = get_selection_text()

    # 路由任务
    if "trigger_add_phrase" in trigger_path:
        add_phrase(captured_text)
    elif "trigger_add_full_content" in trigger_path:
        add_full_content(captured_text)
    elif "trigger_phrase_expander" in trigger_path:
        phrase_expander(captured_text)
    elif "trigger_delete_phrase_mapping" in trigger_path:
        delete_phrase_mapping(captured_text)
