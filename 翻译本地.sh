#!/bin/bash

# ==============================================================================
#      AI Ecosystem Core 启动器 v4.0 (本地模型增强版)
# ==============================================================================
#
# v4.0 更新:
# - 【本地模型支持】: 完美适配已切换到本地 LLM 模型的 AI Ecosystem Core。
# - 【依赖同步】: 确保在启动前安装 `langchain-openai` 等本地模型所需的核心库。
# - 【健壮性】: 综合了 v3.1 和 MuseBox 启动器的优点，增强了错误检查。
#
# ==============================================================================

# --- 1. 项目配置 ---
# 【重要】请将这里的 PYTHON_SCRIPT_NAME 修改为您最终的 Python 文件名
PYTHON_SCRIPT_NAME="13.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_DIR="all-MiniLM-L6-v2"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"


# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI Ecosystem Core 启动器 v4.0 (本地模型版)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查并准备虚拟环境 ---
echo "[1/6] 正在检查并准备虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未在本目录找到虚拟环境 'venv'，正在自动创建..."
    if ! python3 -m venv venv; then
        echo "❌ 严重错误: 虚拟环境创建失败！请检查您的 Python3 安装。"
        read -p "按任意键退出..."
        exit 1
    fi
fi
PYTHON_EXEC="venv/bin/python3"
PIP_EXEC="venv/bin/pip"
echo "✅ 虚拟环境准备就绪。"
echo ""

# --- 步骤 2: 安装/更新Python依赖 (使用国内镜像) ---
echo "[2/6] 正在通过 requirements.txt 安装/更新依赖..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

# 使用增强的pip命令，并检查其是否成功
if ! ${PIP_EXEC} install --upgrade --default-timeout=200 -i ${PIP_MIRROR} -r requirements.txt; then
    echo "❌ 严重错误: Python 依赖安装失败！请检查上面的错误日志。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Python 依赖已是最新状态。"
echo ""


# --- 步骤 3: 检查并确保 Qdrant 数据库正在运行 ---
echo "[3/6] 正在检查 Qdrant 向量数据库 (${DB_CONTAINER_NAME}) 状态..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ 严重错误: Docker 服务未运行！请先启动 Docker Desktop 或 docker daemon。"
    read -p "按任意键退出..."
    exit 1
fi

if [ -n "$(docker ps -q -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo "✅ Qdrant 数据库正在运行。"
elif [ -n "$(docker ps -aq -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo ">> 检测到已停止的 Qdrant 数据库，正在重启..."
    docker start ${DB_CONTAINER_NAME}
    echo "✅ Qdrant 数据库已启动。"
else
    echo "❌ 严重错误: 未找到名为 '${DB_CONTAINER_NAME}' 的 Docker 容器！"
    echo "   请运行 'docker run -p 6333:6333 qdrant/qdrant' 来启动数据库。"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 4: 检查本地 Embedding 模型 ---
echo "[4/6] 正在检查本地 Embedding 模型..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ Embedding 模型文件夹 '${MODEL_DIR}' 已存在。"
else
    echo "❌ 严重错误: 未找到 Embedding 模型文件夹 '${MODEL_DIR}'！"
    echo "   请从 Hugging Face 下载 'all-MiniLM-L6-v2' 并解压到当前目录。"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 5: 检查系统核心工具 ---
echo "[5/6] 正在检查系统核心工具 (wl-clipboard/xclip)..."
if ! command -v wl-copy &> /dev/null && ! command -v xclip &> /dev/null; then
    echo "❌ 严重错误: 系统中未找到剪贴板工具！"
    echo "   请至少安装一个 (推荐: sudo dnf install wl-clipboard xclip)"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 系统核心工具已找到。"
echo ""

# --- 步骤 6: 启动主程序 ---
echo "[6/6] 正在启动 Python 主程序 (${PYTHON_SCRIPT_NAME})..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已关闭。按任意键退出此启动器窗口..."