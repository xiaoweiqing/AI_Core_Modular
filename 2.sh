#!/bin/bash

# ==============================================================================
#      灵感记忆系统 (MuseBox) v1.2 - 启动器 (强制代理版)
# ==============================================================================
#
# v1.2 更新:
# - 【【【 核心修复：强制Pip使用代理 】】】
# - 放弃环境变量，直接在 pip install 命令中使用 --proxy 参数，确保在复杂的
#   网络环境下也能成功安装依赖。
#
# ==============================================================================

# --- 1. 项目配置 ---
PYTHON_SCRIPT_NAME="6.py" # <--- 确保这是您的Python脚本文件名
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
# 【【【 您的代理地址，请按需修改 】】】
PROXY_URL="http://127.0.0.1:2080"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "        灵感记忆系统 (MuseBox) v1.2 - 正在启动..."
echo "========================================================"
echo ""

# --- 步骤 1: 检查并激活虚拟环境 ---
echo "[1/4] 正在检查并准备虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未在本目录找到虚拟环境 'venv'，正在自动创建..."
    python3 -m venv venv
fi
PYTHON_EXEC="venv/bin/python"
PIP_EXEC="venv/bin/pip"
echo "✅ 虚拟环境准备就绪。"
echo ""

# --- 步骤 2: 安装/更新Python依赖 (【【【 核心修复区 】】】) ---
echo "[2/4] 正在安装/更新Python依赖 (使用国内镜像和强制代理)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

echo ">> 正在检查 pynput, sentence-transformers 等核心库..."
# --- 使用 --proxy 参数强制 pip 通过代理 ---
if ! ${PIP_EXEC} install --proxy ${PROXY_URL} --default-timeout=300 -i ${PIP_MIRROR} -r requirements.txt > /dev/null 2>&1; then
    echo "❌ 严重错误: Python 依赖安装失败！"
    echo "   请检查代理是否开启 (${PROXY_URL})，以及 requirements.txt 文件是否正确。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Python 依赖已是最新状态。"
echo ""

# --- 步骤 3: 检查系统核心工具 ---
echo "[3/4] 正在检查系统核心工具 (xclip)..."
if ! command -v xclip &> /dev/null; then
    echo "❌ 严重错误: 系统中未找到 'xclip' 命令！"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 系统核心工具已找到。"
echo ""

# --- 步骤 4: 启动主程序 ---
echo "[4/4] 正在启动 Python 主程序..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已关闭。按任意键退出此启动器窗口..."