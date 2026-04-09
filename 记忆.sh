#!/bin/bash

# ==============================================================================
#      灵感记忆系统 (MuseBox) v1.4 - 启动器 (服务依赖修复版)
# ==============================================================================
#
# v1.4 更新:
# - 【【【 核心修复: 新增 Qdrant 数据库服务检查与启动步骤 】】】
# - 脚本现在会确保 Qdrant Docker 容器在主程序启动前正在运行。
# - 更新了步骤计数和提示信息，使其更加清晰。
#
# ==============================================================================

# --- 1. 项目配置 ---
PYTHON_SCRIPT_NAME="11.py" 

# --- 【【【 已根据您的 `docker ps` 输出为您修正 】】】 ---
DB_CONTAINER_NAME="ai_database_hub" 

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PROXY_URL="http://127.0.0.1:2080"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "        灵感记忆系统 (MuseBox) v1.4 - 正在启动..."
echo "========================================================"
echo ""

# --- 步骤 1: 检查并激活虚拟环境 ---
echo "[1/5] 正在检查并准备虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未在本目录找到虚拟环境 'venv'，正在自动创建..."
    python3 -m venv venv
fi
PYTHON_EXEC="venv/bin/python"
PIP_EXEC="venv/bin/pip"
echo "✅ 虚拟环境准备就绪。"
echo ""

# --- 步骤 2: 安装/更新Python依赖 ---
echo "[2/5] 正在安装/更新Python依赖 (使用国内镜像和强制代理)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

echo ">> 正在检查 langchain, sentence-transformers 等核心库..."
if ! ${PIP_EXEC} install --proxy ${PROXY_URL} --default-timeout=300 -i ${PIP_MIRROR} -r requirements.txt > /dev/null 2>&1; then
    echo "❌ 严重错误: Python 依赖安装失败！"
    echo "   请检查代理是否开启 (${PROXY_URL})，以及 requirements.txt 文件是否正确。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Python 依赖已是最新状态。"
echo ""

# --- 步骤 3: 检查并确保 Qdrant 数据库正在运行 ---
echo "[3/5] 正在检查 Qdrant 向量数据库 (${DB_CONTAINER_NAME}) 状态..."
if [ -n "$(docker ps -q -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo "✅ Qdrant 数据库正在运行。"
elif [ -n "$(docker ps -aq -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo ">> 检测到已停止的 Qdrant 数据库，正在重启..."
    docker start ${DB_CONTAINER_NAME}
    echo "✅ Qdrant 数据库已启动。"
else
    echo "❌ 严重错误: 未找到名为 '${DB_CONTAINER_NAME}' 的 Docker 容器！"
    echo "   请检查脚本顶部的 DB_CONTAINER_NAME 变量是否设置正确。"
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 4: 检查系统核心工具 ---
echo "[4/5] 正在检查系统核心工具 (xclip)..."
if ! command -v xclip &> /dev/null; then
    echo "❌ 严重错误: 系统中未找到 'xclip' 命令！"
    echo "   请确保已安装 (例如: sudo dnf install xclip)"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 系统核心工具已找到。"
echo ""

# --- 步骤 5: 启动主程序 ---
echo "[5/5] 正在启动 Python 主程序..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已关闭。按任意键退出此启动器窗口..."