# cleanup_vectors.py (v2 - with Proxy Cleaner)
import os
from qdrant_client import QdrantClient

# --- [CRITICAL] Proxy Cleaner ---
# 这段代码确保系统代理不会干扰到与本地Qdrant的连接
# It is copied from your main 50.py script.
for proxy_var in [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    if proxy_var in os.environ:
        print(f">> [Proxy Cleaner] Found and removed system proxy: {proxy_var}")
        del os.environ[proxy_var]
# --- Proxy Cleaner End ---


# 集合名称，与您的主脚本中定义的一致
COLLECTION_TO_DELETE = "personal_constitution"

print("正在连接 Qdrant 服务...")
client = QdrantClient("localhost", port=6333)

print(f"正在尝试删除集合: '{COLLECTION_TO_DELETE}'...")

try:
    result = client.delete_collection(collection_name=COLLECTION_TO_DELETE)
    if result:
        print(f"✅ 成功删除集合 '{COLLECTION_TO_DELETE}'！")
    else:
        print(f"🟡 集合 '{COLLECTION_TO_DELETE}' 可能原本就不存在。")

except Exception as e:
    print(f"❌ 删除时发生错误: {e}")
    print("   请确认 Qdrant Docker 容器是否正在运行。")
