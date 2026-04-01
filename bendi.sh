#!/bin/bash

# ==============================================================================
#      AI Core Modular - Launcher v9.2 ("True Local Vulkan" Edition)
# ==============================================================================
# v9.2 更新:
# - 【架构升级】: 完全移除对外部 llama-server 的依赖。
# - 【启动简化】: 主程序 main.py 现在直接加载模型，不再需要复杂的
#   服务器检查、清理和等待逻辑。
# - 【完全离线】: 整个启动和运行过程无需任何网络连接。
# ==============================================================================

# --- 1. 项目配置 (已简化) ---
PYTHON_SCRIPT_NAME="main.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_DIR="all-MiniLM-L6-v2" # Embedding 模型目录依然需要检查
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# --- 脚本主逻辑 ---
clear
echo "================================================================="
echo "   AI Core Launcher v9.2 (\"True Local Vulkan\" Edition)"
echo "================================================================="
echo ""

# --- 步骤 1: 检查、创建并激活虚拟环境 ---
echo "[1/4] 正在设置并激活虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未找到虚拟环境 'venv'，正在创建..."
    python3 -m venv venv || { echo "❌ 严重错误: 创建虚拟环境失败！"; read -p "按任意键退出..."; exit 1; }
fi
source venv/bin/activate
echo "✅ 虚拟环境已激活。"
echo ""

# --- 步骤 2: 安装/更新 Python 依赖 ---
echo "[2/4] 正在安装/更新依赖..."
# 注意：请确保 requirements.txt 中已包含 'langchain-community'
pip install -r requirements.txt -i ${PIP_MIRROR} || { echo "❌ 严重错误: Python 依赖安装失败！"; read -p "按任意键退出..."; exit 1; }
echo "✅ Python 依赖已是最新。"
echo ""

# --- 步骤 3: 检查数据库容器 ---
echo "[3/4] 正在检查数据库容器 (${DB_CONTAINER_NAME})..."
if [ -n "$(docker ps -q -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo "✅ 数据库正在运行。"
elif [ -n "$(docker ps -aq -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo ">> 检测到已停止的数据库，正在重启..."
    docker start ${DB_CONTAINER_NAME}
    echo "✅ 数据库已启动。"
else
    echo "❌ 严重错误: 未找到 Docker 容器 '${DB_CONTAINER_NAME}'！"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 4: 检查 Embedding 模型 ---
echo "[4/4] 正在检查本地 Embedding 模型..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ Embedding 模型文件夹 '${MODEL_DIR}' 已找到。"
else
    echo "❌ 错误: Embedding 模型文件夹 '${MODEL_DIR}' 丢失！"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# launcher.sh 的最后部分

# --- [已移除] 步骤 5: 启动 LLM 服务器 ---
# 这个步骤已不再需要，因为模型现在由 main.py 直接加载。

# --- 最后一步: 启动主程序 ---
echo ">>> 准备启动 Python 主程序: ${PYTHON_SCRIPT_NAME}..."
echo "    (大语言模型将在程序内部进行加载，请耐心等待...)"
echo "    (强制指定使用 Vulkan 设备 0: AMD Radeon 780M)" # <--- 添加这行提示
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""

# +++ 这是核心改动：在运行python前设置环境变量 +++
GGML_VULKAN_DEVICE=0 python "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [  程序日志结束  ] --------------------------"
read -p "程序已执行完毕。按任意键退出..."
