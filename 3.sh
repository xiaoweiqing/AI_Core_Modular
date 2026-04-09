#!/bin/bash

# ==============================================================================
#      灵感记忆系统 (MuseBox) v1.3 - 启动器 (信号架构版)
# ==============================================================================
#
# v1.3 更新:
# - 依赖库与代码同步: 移除 pynput，新增 watchdog 依赖检查。
# - 提示信息更新: 启动提示与新的文件信号架构保持一致。
# - 脚本健壮性: 保持强制代理安装，确保网络环境适应性。
#
# ==============================================================================

# --- 1. 项目配置 ---
# <<< 请确保这里的文件名和您保存的Python文件名一致
PYTHON_SCRIPT_NAME="6.py" 
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PROXY_URL="http://127.0.0.1:2080"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "        灵感记忆系统 (MuseBox) v1.3 - 正在启动..."
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

# --- 步骤 2: 安装/更新Python依赖 (【【【 已更新 】】】) ---
echo "[2/4] 正在安装/更新Python依赖 (使用国内镜像和强制代理)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

# <<< MODIFIED: 更新了检查的库名称，使其更有代表性
echo ">> 正在检查 watchdog, sentence-transformers 等核心库..."
# --- 使用 --proxy 参数强制 pip 通过代理 ---
if ! ${PIP_EXEC} install --proxy ${PROXY_URL} --default-timeout=300 -i ${PIP_MIRROR} -r requirements.txt > /dev/null 2>&1; then
    echo "❌ 严重错误: Python 依赖安装失败！"
    echo "   请检查代理是否开启 (${PROXY_URL})，以及 requirements.txt 文件是否正确。"
    echo "   (请确保 requirements.txt 中已包含 watchdog 库)"
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