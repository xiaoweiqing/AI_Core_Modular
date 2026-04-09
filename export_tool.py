import sqlite3
import json
from pathlib import Path

# --- 配置区 ---
DB_FILE = Path.home() / "MuseBox_Storage" / "musebox_memory.db"
# 您可以在这里定义不同任务的专属指令
TASK_INSTRUCTIONS = {
    "knowledge_card": "你是一个顶级的知识管理专家...",  # 您的知识卡片Prompt
    "resume_analysis": "请从以下非结构化简历文本中，精确地提取出候选人的核心信息，并以JSON格式输出。",  # 简历分析的Prompt
    "default": "请分析以下文本并提取关键信息。",
}


def export_training_data():
    if not DB_FILE.exists():
        print(f"❌ 错误: 数据库文件不存在于 {DB_FILE}")
        return

    # 1. 让用户选择要导出的任务类型
    print("🔍 正在从 MuseBox 数据库导出训练数据...")
    task_to_export = input(
        "请输入您想导出的任务类型 (例如: knowledge_card, resume_analysis): "
    ).strip()

    if not task_to_export:
        print("❌ 未输入任务类型，已取消。")
        return

    output_file = f"finetune_data_{task_to_export}.jsonl"

    all_formatted_data = []
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            # 2. 从数据库中精确筛选数据
            query = "SELECT raw_text, structured_data_json FROM inspirations WHERE is_reviewed = 1 AND task_type = ?"
            cursor.execute(query, (task_to_export,))

            for raw_text, perfect_json in cursor.fetchall():
                instruction = TASK_INSTRUCTIONS.get(
                    task_to_export, TASK_INSTRUCTIONS["default"]
                )

                # 3. 组装成模型喜欢的格式
                formatted_entry = {
                    "instruction": instruction,
                    "input": raw_text,
                    "output": perfect_json,  # 注意：这里直接用您批改后的完美JSON
                }
                all_formatted_data.append(formatted_entry)
    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")
        return

    # 4. 保存为 JSONL 文件
    if not all_formatted_data:
        print(f"⚠️ 在数据库中没有找到类型为 '{task_to_export}' 的、已批改的训练数据。")
        return

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in all_formatted_data:
                # 确保 output 是一个字符串
                if not isinstance(entry["output"], str):
                    entry["output"] = json.dumps(entry["output"], ensure_ascii=False)

                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + "\n")

        print(f"\n✅ 数据导出成功！共找到 {len(all_formatted_data)} 条高质量训练数据。")
        print(f"🎉 文件已保存到: {output_file}")
        print("现在，您可以将这个文件用于您的大模型微调流程了！")

    except IOError as e:
        print(f"\n❌ 文件写入失败: {e}")


if __name__ == "__main__":
    export_training_data()
