#!/bin/bash

# ==============================================================================
#      AI Ecosystem Core 启动器 v3.1 (网络增强 & 错误处理版)
# ==============================================================================
#
# v3.1 更新:
# - 【网络优化】: 默认使用清华大学 PyPI 镜像源，大幅提升库安装速度。
# - 【网络优化】: 增加 pip 下载超时时间，防止因大文件（如PyTorch）下载失败。
# - 【错误处理】: 增强脚本健壮性，若依赖安装失败，则立即停止并报错。
#
# ==============================================================================

# --- 1. 项目配置 ---
PYTHON_SCRIPT_NAME="29.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_REPO_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR="all-MiniLM-L6-v2"
# 【新增】国内镜像源
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"


# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI Ecosystem Core 启动器 v3.1 (网络增强版)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查并激活虚拟环境 ---
echo "[1/6] 正在检查并准备虚拟环境..."
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

# --- 步骤 2: 安装/更新Python依赖 (【【【 核心修复区 】】】) ---
echo "[2/6] 正在通过 requirements.txt 安装/更新依赖 (使用国内镜像)..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 未找到依赖文件 'requirements.txt'！"
    read -p "按任意键退出..."
    exit 1
fi

# 使用增强的pip命令，并检查其是否成功
if ! ${PIP_EXEC} install --default-timeout=100 -i ${PIP_MIRROR} -r requirements.txt; then
    echo "❌ 严重错误: Python 依赖安装失败！请检查上面的错误日志。"
    read -p "按任意键退出..."
    exit 1
fi

echo "✅ Python 依赖已是最新状态。"
echo ""


# --- 步骤 3: 检查并确保数据库正在运行 ---
echo "[3/6] 正在检查数据库 (${DB_CONTAINER_NAME}) 状态..."
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

# --- 步骤 4: 检查并下载本地AI模型 ---
echo "[4/6] 正在检查本地AI模型..."
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

# --- 步骤 5: 自动修改Python脚本 (可选，但保留兼容) ---
echo "[5/6] 正在检查 Python 脚本模型路径..."
FIXED_LINE="SentenceTransformer('./all-MiniLM-L6-v2')"
if grep -q "${FIXED_LINE}" "${PYTHON_SCRIPT_NAME}"; then
    echo "✅ Python 脚本已配置为从本地加载模型。"
else
    echo ">> 正在自动修改 '${PYTHON_SCRIPT_NAME}'..."
    sed -i.bak "s|SentenceTransformer('all-MiniLM-L6-v2')|${FIXED_LINE} # 由启动器自动修改为本地路径|g" "${PYTHON_SCRIPT_NAME}"
    echo "✅ 脚本修改成功！"
fi
echo ""

# --- 步骤 6: 启动主程序 ---
echo "[6/6] 正在启动 Python 主程序..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "程序已执行完毕。按任意键退出..."