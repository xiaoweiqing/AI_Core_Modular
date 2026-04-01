# cleanup_qdrant.py (V2 - 带代理清理器)

import os
from qdrant_client import QdrantClient

# ==============================================================================
#      【【【 这是新增的“代理清理器”，和你的主程序里的一模一样 】】】
# ==============================================================================
print(">> [Proxy Cleaner] 正在检查并清理系统代理设置...")
proxy_cleaned = False
for proxy_var in [
    "http_proxy", "https_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
]:
    if proxy_var in os.environ:
        print(f"   -> 发现并移除了代理: {proxy_var}")
        del os.environ[proxy_var]
        proxy_cleaned = True
if not proxy_cleaned:
    print("   -> 未发现系统代理，环境干净。")
# ==============================================================================


# --- 你要删除的、被污染的集合名字 (保持不变) ---
COLLECTION_TO_DELETE = "gemma_dialogue_pairs_768d_v1"
RISK_COLLECTION_TO_DELETE = "gemma_risk_analysis_768d_v1"

def main():
    print("\n--- Qdrant 数据库清理工具 ---")
    
    try:
        client = QdrantClient(host="localhost", port=6333)
        print("✅ 成功连接到 Qdrant 服务。")

        # --- 删除主集合 ---
        print(f"\n正在尝试删除集合: '{COLLECTION_TO_DELETE}'...")
        try:
            delete_result = client.delete_collection(collection_name=COLLECTION_TO_DELETE)
            if delete_result:
                print(f"✅ 成功删除集合 '{COLLECTION_TO_DELETE}'。")
            else:
                print(f"🟡 集合 '{COLLECTION_TO_DELETE}' 删除失败或已不存在，跳过。")
        except Exception as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                 print(f"🟢 集合 '{COLLECTION_TO_DELETE}' 本来就不存在，无需删除。")
            else:
                print(f"❌ 删除时发生未知错误: {e}")

        # --- 删除风险分析集合 ---
        print(f"\n正在尝试删除集合: '{RISK_COLLECTION_TO_DELETE}'...")
        try:
            delete_result = client.delete_collection(collection_name=RISK_COLLECTION_TO_DELETE)
            if delete_result:
                print(f"✅ 成功删除集合 '{RISK_COLLECTION_TO_DELETE}'。")
            else:
                print(f"🟡 集合 '{RISK_COLLECTION_TO_DELETE}' 删除失败或已不存在，跳过。")
        except Exception as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                 print(f"🟢 集合 '{RISK_COLLECTION_TO_DELETE}' 本来就不存在，无需删除。")
            else:
                print(f"❌ 删除时发生未知错误: {e}")

    except Exception as e:
        print(f"❌ 无法连接到 Qdrant 服务: {e}")
        print("   请确保你的 Qdrant Docker 容器正在运行。")

    print("\n--- 清理完成 ---")

if __name__ == "__main__":
    main()
