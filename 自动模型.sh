#!/bin/bash

# ==============================================================================
#      AI Ecosystem Core 启动器 v3.3 (目录隔离修复版)
# ==============================================================================
#
# v3.3 更新:
# - 【核心修复】: 使用子shell()来启动模型服务器，防止改变主脚本的工作目录，
#   解决了找不到Python脚本和虚拟环境的严重错误。
#
# v3.2 更新:
# - 【核心功能】: 自动检查并后台启动 llama.cpp 服务器，实现真正的一键启动。
#
# ==============================================================================

# --- 1. 项目配置 ---
PYTHON_SCRIPT_NAME="58.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_REPO_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR="all-MiniLM-L6-v2"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ### --- 本地模型服务器的配置 --- ###
MODEL_SERVER_DIR="$HOME/llama.cpp/build-vulkan-new"
MODEL_PATH="/mnt/data/model/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf"
MODEL_SERVER_EXEC="./bin/llama-server"
MODEL_SERVER_ARGS="-c 56666 -ngl 99 --repeat-penalty 1.1 --presence-penalty 0.5 --top-k 40 --top-p 0.95 --host 0.0.0.0 --port 8087"


# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI Ecosystem Core 启动器 v3.3 (目录隔离修复版)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查并激活虚拟环境 ---
echo "[1/7] 正在检查并准备虚拟环境..."
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ 错误: 未在本目录找到虚拟环境 'venv'！"
    echo "   请先运行: python3 -m venv venv"
    read -p "按任意键退出..."
    exit 1
fi
PYTHON_EXEC="venv/bin/python3"
PIP_EXEC="venv/bin/pip"
echo "✅ 虚拟环境检测通过。"
echo ""

# --- 步骤 2: 安装/更新Python依赖 ---
echo "[2/7] 正在通过 requirements.txt 安装/更新依赖 (使用国内镜像)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

if ! ${PIP_EXEC} install --default-timeout=100 -i ${PIP_MIRROR} -r requirements.txt; then
    echo "❌ 严重错误: Python 依赖安装失败！请检查上面的错误日志。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Python 依赖已是最新状态。"
echo ""

# --- 步骤 3: 检查并确保数据库正在运行 ---
echo "[3/7] 正在检查数据库 (${DB_CONTAINER_NAME}) 状态..."
if [ -n "$(docker ps -q -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo "✅ 数据库正在运行。"
elif [ -n "$(docker ps -aq -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo ">> 检测到已停止的数据库，正在重启..."
    docker start ${DB_CONTAINER_NAME}
    echo "✅ 数据库已启动。"
else
    echo "❌ 严重错误: 未找到Docker容器 '${DB_CONTAINER_NAME}'！"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 4: 检查并下载本地AI模型 (Embedding Model) ---
echo "[4/7] 正在检查本地AI模型 (Embedding)..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ 模型文件夹 '${MODEL_DIR}' 已存在。"
else
    echo ">> 模型不存在，正在从Hugging Face下载..."
    export https_proxy="http://127.0.0.1:2080"
    export http_proxy="http://127.0.0.1:2080"
    git lfs install && git clone ${MODEL_REPO_URL}
    if [ $? -ne 0 ]; then
        echo "❌ 严重错误: 模型下载失败！请检查代理和网络。"
        unset https_proxy http_proxy
        read -p "按任意键退出..."
        exit 1
    fi
    unset https_proxy http_proxy
    echo "✅ 模型下载成功！"
fi
echo ""

# --- 步骤 5: 检查并启动本地模型服务器 ---
echo "[5/7] 正在检查本地AI模型服务器 (llama-server) 状态..."
if pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
    echo "✅ 本地AI模型服务器已在运行。"
else
    echo ">> 服务器未运行，正在后台启动..."
    if [ ! -d "${MODEL_SERVER_DIR}" ]; then
        echo "❌ 严重错误: 模型服务器目录 '${MODEL_SERVER_DIR}' 不存在！"
        read -p "按任意键退出..."
        exit 1
    fi
    
    # 使用 () 将命令包裹在子shell中运行，这样就不会改变主脚本的当前目录
    (
      cd "${MODEL_SERVER_DIR}" && nohup ${MODEL_SERVER_EXEC} -m "${MODEL_PATH}" ${MODEL_SERVER_ARGS} > llama-server.log 2>&1 &
    )
    
    echo ">> 正在等待服务器初始化 (5秒)..."
    sleep 5
    
    if pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
        echo "✅ 服务器成功在后台启动！日志文件: ${MODEL_SERVER_DIR}/llama-server.log"
    else
        echo "❌ 严重错误: 启动本地AI模型服务器失败！"
        echo "   请检查日志文件: ${MODEL_SERVER_DIR}/llama-server.log"
        read -p "按任意键退出..."
        exit 1
    fi
fi
echo ""

# --- 步骤 6: 自动修改Python脚本 (可选) ---
echo "[6/7] 正在检查 Python 脚本模型路径..."
FIXED_LINE="SentenceTransformer('./all-MiniLM-L6-v2')"
if grep -q "${FIXED_LINE}" "${PYTHON_SCRIPT_NAME}"; then
    echo "✅ Python 脚本已配置为从本地加载模型。"
else
    echo ">> 正在自动修改 '${PYTHON_SCRIPT_NAME}'..."
    sed -i.bak "s|SentenceTransformer('all-MiniLM-L6-v2')|${FIXED_LINE} # 由启动器自动修改为本地路径|g" "${PYTHON_SCRIPT_NAME}"
    echo "✅ 脚本修改成功！"
fi
echo ""

# --- 步骤 7: 启动主程序 ---
echo "[7/7] 正在启动 Python 主程序..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已执行完毕。按任意键退出..."
