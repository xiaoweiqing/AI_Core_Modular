import sqlite3
import json
import os
from pathlib import Path
import textwrap

DB_FILE = Path.home() / "MuseBox_Storage" / "musebox_memory.db"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pretty_print_json(json_data):
    """接收Python字典或JSON字符串，美化打印"""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，返回原始字符串，但进行缩进处理
        return textwrap.indent(str(json_data), "  ")


def get_multiline_input(initial_content=""):
    """一个更健壮的多行输入函数，支持粘贴和编辑"""
    print(">>> 进入多行编辑模式。完成后，在新的一行输入 'END' 并回车 <<<")
    print(">>> 您可以直接粘贴内容进来。 <<<")

    lines = []
    # 如果有初始内容，先显示
    if initial_content and initial_content.strip() != "None":
        print("--- 初始内容 (可复制后修改) ---")
        print(initial_content)
        print("------------------------------")

    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:  # 处理 Ctrl+D 结束输入的情况
            break
    return "\n".join(lines)


def review_entries():
    if not DB_FILE.exists():
        print(f"❌ 错误: 数据库文件不存在于 {DB_FILE}")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    while True:  # 使用循环，直到没有未批改的条目
        cursor.execute(
            "SELECT id, raw_text, ai_raw_output_json, task_type FROM inspirations WHERE is_reviewed = 0 LIMIT 1"
        )
        entry = cursor.fetchone()

        if not entry:
            clear_screen()
            print("🎉 恭喜！所有灵感都已批改完毕！")
            break

        db_id, raw_text, ai_raw_json, current_task_type = entry

        # ----------- 开始处理单个条目 -----------
        clear_screen()
        print("=" * 25 + f" 批改作业 (ID: {db_id}) " + "=" * 25)

        print("\n--- 【1. 原始文本 (Input)】 " + "-" * 48)
        print(raw_text)

        print("\n--- 【2. AI的初步分析 (AI Raw Output)】 " + "-" * 37)
        print(pretty_print_json(ai_raw_json))

        # 步骤1: 分类任务
        print("\n--- 【3. 请分类任务】 " + "-" * 53)
        new_task_type_input = input(
            f"当前任务类型是 '{current_task_type}'，要修改吗？(直接回车不改): "
        ).strip()
        new_task_type = (
            new_task_type_input if new_task_type_input else current_task_type
        )

        # 步骤2: 修改为完美输出
        print("\n--- 【4. 请修改或创建完美的JSON输出】 " + "-" * 35)
        perfect_json_str = get_multiline_input(initial_content=ai_raw_json)

        # 步骤3: 确认并更新
        while True:
            clear_screen()
            print("=" * 25 + " 请确认您的修改 " + "=" * 25)
            print(f"【数据库ID】: {db_id}")
            print(f"【任务类型】: {new_task_type}")
            print("\n--- 【最终的完美输出】 " + "-" * 50)

            # 尝试解析并美化打印，如果失败则提示错误
            is_valid_json = False
            try:
                parsed_json = json.loads(perfect_json_str)
                print(pretty_print_json(parsed_json))
                is_valid_json = True
            except json.JSONDecodeError:
                print("!!!!!!!!!! 警告: 您输入的不是有效的JSON格式 !!!!!!!!!!!")
                print(perfect_json_str)

            print("=" * 70)

            if not is_valid_json:
                confirm = (
                    input("输入无效！(e)dit-重新编辑, (s)kip-跳过本条: ")
                    .strip()
                    .lower()
                )
                if confirm == "s":
                    break
                else:  # 重新编辑
                    perfect_json_str = get_multiline_input(
                        initial_content=perfect_json_str
                    )
                    continue

            confirm = (
                input("确认保存？(y)es / (n)o / (s)kip / (e)dit: ").strip().lower()
            )

            if confirm in ["y", "yes"]:
                try:
                    cursor.execute(
                        "UPDATE inspirations SET task_type = ?, structured_data_json = ?, is_reviewed = 1 WHERE id = ?",
                        (new_task_type, perfect_json_str, db_id),
                    )
                    conn.commit()
                    print(f"✅ ID: {db_id} 已成功批改并保存！")
                    input("按回车键继续批改下一条...")
                    break  # 跳出确认循环，进入下一个条目
                except Exception as e:
                    print(f"❌ 数据库更新失败: {e}")
                    input("按回车键重试...")

            elif confirm in ["n", "no"]:
                print("已取消本次修改。")
                input("按回车键继续...")
                break  # 重新开始处理本条
            elif confirm in ["s", "skip"]:
                print("已跳过本条。")
                input("按回- 车键继续...")
                break  # 进入下一个条目
            elif confirm in ["e", "edit"]:
                perfect_json_str = get_multiline_input(initial_content=perfect_json_str)
            else:
                print("无效输入，请重试。")

    conn.close()


if __name__ == "__main__":
    review_entries()
