#!/bin/bash

# ==============================================================================
#      灵感记忆系统 (MuseBox) v3.0 - 专属启动器
# ==============================================================================
#
# 功能:
# - 完美适配 MuseBox v3.0 的所有依赖和系统需求。
# - 【核心检查】: 检查本地AI服务 (端口 8087) 是否正在运行。
# - 【核心检查】: 检查 Qdrant 数据库 Docker 容器是否正在运行。
# - 【核心检查】: 检查本地 Embedding 模型文件夹是否存在。
# - 自动创建并管理 Python 虚拟环境。
# - 通过国内镜像源快速安装所有必需的 Python 库。
# - 检查系统剪贴板工具 (wl-clipboard, xclip) 是否已安装。
#
# ==============================================================================

# --- 1. 项目配置 ---
# 【重要】请将这里的 PYTHON_SCRIPT_NAME 修改为您最终的 Python 文件名
PYTHON_SCRIPT_NAME="15.py" 
DB_CONTAINER_NAME="ai_database_hub" 
MODEL_DIR="all-MiniLM-L6-v2"
LOCAL_AI_PORT="8087"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    灵感记忆系统 (MuseBox) v3.0 - 正在启动..."
echo "========================================================"
echo ""

# --- 步骤 1: 检查并准备虚拟环境 ---
echo "[1/7] 正在检查并准备虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未在本目录找到虚拟环境 'venv'，正在自动创建..."
    if ! python3 -m venv venv; then
        echo "❌ 严重错误: 虚拟环境创建失败！请检查您的 Python3 安装。"
        read -p "按任意键退出..."
        exit 1
    fi
fi
PYTHON_EXEC="venv/bin/python"
PIP_EXEC="venv/bin/pip"
echo "✅ 虚拟环境准备就绪。"
echo ""

# --- 步骤 2: 检查并安装Python依赖 ---
echo "[2/7] 正在安装/更新Python依赖 (使用国内镜像)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    echo "   -> 请确保该文件存在，并包含 langchain-openai, qdrant-client 等库。"
    read -p "按任意键退出..."
    exit 1
fi
if ! ${PIP_EXEC} install --upgrade --default-timeout=200 -i ${PIP_MIRROR} -r requirements.txt; then
    echo "❌ 严重错误: Python 依赖安装失败！请检查上面的错误日志。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Python 依赖已是最新状态。"
echo ""

# --- 步骤 3: 检查本地AI服务 (Ollama/LiteLLM) ---
echo "[3/7] 正在检查本地AI服务 (端口 ${LOCAL_AI_PORT})..."
if command -v nc &> /dev/null; then
    if ! nc -zvw1 127.0.0.1 ${LOCAL_AI_PORT} &> /dev/null; then
        echo "❌ 严重错误: 无法连接到端口 ${LOCAL_AI_PORT} 上的本地AI服务！"
        echo "   -> 请确保您的 Ollama 或其他模型服务正在运行。"
        read -p "按任意键退出..."
        exit 1
    fi
else
    echo ">> 警告: 未找到 'nc' (netcat) 命令，跳过AI服务端口检查。"
fi
echo "✅ 本地AI服务响应正常。"
echo ""

# --- 步骤 4: 检查 Qdrant 数据库 Docker 容器 ---
echo "[4/7] 正在检查 Qdrant 向量数据库 (${DB_CONTAINER_NAME}) 状态..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ 严重错误: Docker 服务未运行！请先启动 Docker。"
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
    echo "   -> 请运行 'docker run -p 6333:6333 qdrant/qdrant' 启动。"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 5: 检查本地 Embedding 模型 ---
echo "[5/7] 正在检查本地 Embedding 模型..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ Embedding 模型文件夹 '${MODEL_DIR}' 已存在。"
else
    echo "❌ 严重错误: 未找到 Embedding 模型文件夹 '${MODEL_DIR}'！"
    echo "   -> 请从 Hugging Face 下载 'all-MiniLM-L6-v2' 并解压到当前目录。"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 6: 检查系统剪贴板工具 ---
echo "[6/7] 正在检查系统核心工具 (wl-clipboard/xclip)..."
if ! command -v wl-paste &> /dev/null && ! command -v xclip &> /dev/null; then
    echo "❌ 严重错误: 系统中未找到剪贴板工具！"
    echo "   -> 请至少安装一个 (推荐: sudo dnf install wl-clipboard xclip)"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 系统核心工具已找到。"
echo ""

# --- 步骤 7: 启动主程序 ---
echo "[7/7] 所有检查通过！正在启动 MuseBox v3.0 主程序..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已关闭。按任意键退出此启动器窗口..."