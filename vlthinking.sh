#!/bin/bash

# ==============================================================================
#      AI Core Modular - Launcher v7.0 ("sj-Thinking" 模型版)
# ==============================================================================
# v7.0 更新:
# - 【模型更换】: 已按照您的最新要求，将模型路径和参数更新为
#   "sjQwen3-VL-30B-A3B-Thinking"。
#
# v6.0 核心功能:
# - 【智能等待】: 脚本会每隔3秒检查一次服务器状态，一旦启动成功便立即继续，
#   极大提升了启动效率。
# - 【自我修复】: 自动清理旧的或错误的模型进程。
# ==============================================================================

# --- 1. 项目配置 ---
PYTHON_SCRIPT_NAME="main.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_DIR="all-MiniLM-L6-v2"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ##############################################################################
# ### --- 本地视觉模型服务器配置 (已更新为 "sj-Thinking" 模型) --- ###
# ##############################################################################
MODEL_SERVER_DIR="$HOME/llama.cpp/build-vulkan-new"
# 新的 "sj-Thinking" 模型 GGUF 文件路径
MODEL_PATH="/mnt/data/models/Qwen3-VL-30B-A3B-Thinking/sjQwen3-VL-30B-A3B-Thinking-UD-TQ1_0.gguf"
# 新的 "sj-Thinking" 模型 mmproj 文件路径
MMPROJ_PATH="/mnt/data/models/Qwen3-VL-30B-A3B-Thinking/sjmmproj-30B-Thinking-F16.gguf"
MODEL_SERVER_EXEC="./bin/llama-server"

# 使用数组定义您指定的全新参数 (c=14096, ngl=50)，确保100%正确
SERVER_ARGS=(
    --mmproj "${MMPROJ_PATH}"
    -c 14096
    -ngl 50
    --host 0.0.0.0
    --port 8087
    --repeat-penalty 1.1
    --presence-penalty 0.5
    --top-k 40
    --top-p 0.95
    --jinja
)


# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI Core Modular - Launcher v7.0 (\"sj-Thinking\" 模型版)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查、创建并激活虚拟环境 ---
echo "[1/6] 正在设置并激活虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未找到虚拟环境 'venv'，正在创建..."
    python3 -m venv venv || { echo "❌ 严重错误: 创建虚拟环境失败！"; read -p "按任意键退出..."; exit 1; }
fi
source venv/bin/activate
echo "✅ 虚拟环境已激活。"
echo ""

# --- 步骤 2: 安装/更新 Python 依赖 ---
echo "[2/6] 正在安装/更新依赖..."
pip install -r requirements.txt -i ${PIP_MIRROR} || { echo "❌ 严重错误: Python 依赖安装失败！"; read -p "按任意键退出..."; exit 1; }
echo "✅ Python 依赖已是最新。"
echo ""

# --- 步骤 3: 检查数据库容器 ---
echo "[3/6] 正在检查数据库容器 (${DB_CONTAINER_NAME})..."
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
echo "[4/6] 正在检查本地 Embedding 模型..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ Embedding 模型文件夹 '${MODEL_DIR}' 已找到。"
else
    echo "❌ 错误: Embedding 模型文件夹 '${MODEL_DIR}' 丢失！"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 5: 智能检查并启动 LLM 服务器 ---
echo "[5/6] 正在智能检查并准备本地 LLM 服务器..."

# 智能清理：检查是否有任何 llama-server 进程在运行
if pgrep -f "llama-server" > /dev/null; then
    if pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
        echo "✅ 您指定的新模型服务器已在运行。"
    else
        echo ">> 检测到正在运行的是旧模型或未知模型，将强制终止..."
        killall -9 llama-server
        sleep 2
        echo "✅ 旧进程已清理。"
    fi
fi

# 再次检查，如果需要则启动
if ! pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
    echo ">> 服务器未运行，正在后台启动您的新模型..."
    if [ ! -d "${MODEL_SERVER_DIR}" ]; then
        echo "❌ 严重错误: 模型服务器目录 '${MODEL_SERVER_DIR}' 不存在！"
        read -p "按任意键退出..."
        exit 1
    fi
    
    (
      cd "${MODEL_SERVER_DIR}" && nohup ${MODEL_SERVER_EXEC} -m "${MODEL_PATH}" "${SERVER_ARGS[@]}" > llama-server.log 2>&1 &
    )
    
    # 【核心优化】智能等待逻辑
    MAX_WAIT_SECONDS=45
    CHECK_INTERVAL=3
    elapsed_time=0
    server_started=false

    echo -n ">> 正在智能等待服务器初始化 (最长 ${MAX_WAIT_SECONDS} 秒): "
    
    while [ ${elapsed_time} -lt ${MAX_WAIT_SECONDS} ]; do
        if pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
            if ss -tlnp | grep -q ":8087"; then
                server_started=true
                break
            fi
        fi
        echo -n "."
        sleep ${CHECK_INTERVAL}
        elapsed_time=$((elapsed_time + CHECK_INTERVAL))
    done
    echo "" # 换行

    if ${server_started}; then
        echo "✅ 服务器在 ${elapsed_time} 秒内成功启动！日志: ${MODEL_SERVER_DIR}/llama-server.log"
    else
        echo "❌ 严重错误: 等待 ${MAX_WAIT_SECONDS} 秒后，服务器仍未成功启动！"
        echo "   可能原因：1. 模型路径错误 2. 显存不足 3. llama.cpp版本问题"
        echo "   请检查日志获取详细信息: ${MODEL_SERVER_DIR}/llama-server.log"
        tail -n 20 "${MODEL_SERVER_DIR}/llama-server.log"
        read -p "按任意键退出..."
        exit 1
    fi
fi
echo ""

# --- 步骤 6: 启动主程序 ---
echo "[6/6] 正在启动 Python 主程序: ${PYTHON_SCRIPT_NAME}..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
python "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [  程序日志结束  ] --------------------------"
read -p "程序已执行完毕。按任意键退出..."
