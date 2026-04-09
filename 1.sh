#!/bin/bash

# ==============================================================================
#      Apple FastVLM 截图分析工具 启动器 v1.1 (缓存优化版)
# ==============================================================================
#
# v1.1 更新:
# - 【缓存优先】: 脚本不再手动下载模型，完全依赖 Transformers 库的全局缓存机制。
# - 如果您已在系统中下载过模型，程序将直接使用缓存，实现秒级启动。
# - 如果是新环境，程序在首次运行时会自动下载并缓存模型。
#
# ==============================================================================

# --- 1. 项目配置 ---
PYTHON_SCRIPT_NAME="2.py"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    Apple FastVLM 截图分析工具 启动器 v1.1 (缓存优化版)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查并激活虚拟环境 ---
echo "[1/4] 正在检查并准备虚拟环境..."
if [ ! -d "venv" ]; then
    echo ">> 未在本目录找到虚拟环境 'venv'，正在自动创建..."
    if ! command -v python3 &> /dev/null; then
        echo "❌ 严重错误: 系统中未找到 'python3' 命令。请先安装 Python 3。"
        read -p "按任意键退出..."
        exit 1
    fi
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ 严重错误: 创建虚拟环境失败！"
        read -p "按任意键退出..."
        exit 1
    fi
    echo "✅ 虚拟环境创建成功。"
fi

PYTHON_EXEC="venv/bin/python3"
PIP_EXEC="venv/bin/pip"
echo "✅ 虚拟环境准备就绪。"
echo ""

# --- 步骤 2: 安装/更新Python依赖 ---
echo "[2/4] 正在安装/更新Python依赖 (使用国内镜像)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

echo ">> 正在检查 PyTorch, Transformers 等核心库..."
if ! ${PIP_EXEC} install --default-timeout=300 -i ${PIP_MIRROR} -r requirements.txt; then
    echo "❌ 严重错误: Python 依赖安装失败！请检查上面的错误日志。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Python 依赖已是最新状态。"
echo ""

# --- 步骤 3: 检查系统截图工具 ---
echo "[3/4] 正在检查系统截图工具 (gnome-screenshot)..."
if ! command -v gnome-screenshot &> /dev/null; then
    echo "❌ 严重错误: 系统中未找到 'gnome-screenshot' 命令！"
    echo "   这是程序截图功能的核心依赖。请根据您的Linux发行版进行安装："
    echo "   - Fedora/RHEL: sudo dnf install gnome-screenshot"
    echo "   - Debian/Ubuntu: sudo apt-get install gnome-screenshot"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 系统截图工具已找到。"
echo ""

# --- 步骤 4: 启动主程序 ---
echo "[4/4] 正在启动 Python 主程序..."
echo ">> 程序将自动使用 Transformers 全局缓存中的模型。"
echo ">> 如果缓存中已有模型，将直接加载；否则，首次运行时会自动下载。"
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已执行完毕。按任意键退出..."