#!/bin/bash

# ==============================================================================
#      AI Core Modular - Launcher v4.0 (Qwen3-8B Q8 Optimized)
# ==============================================================================

# --- 1. Project Configuration ---
CONDA_ENV_NAME="aic-final" 
PYTHON_SCRIPT_NAME="main.py"
CONFIG_FILE_NAME="config.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_DIR="all-MiniLM-L6-v2"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# --- 🚀 Local LLM Server Configuration (已为你修改为 Q8 满血版) ---
MODEL_SERVER_DIR="$HOME/llama.cpp/build-vulkan-new"
# 指向你刚才下载成功的 Q8 模型
MODEL_PATH="/mnt/data/model/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q8_0.gguf"
MODEL_SERVER_EXEC="./bin/llama-server"

# 这里是针对你的 780M 核显优化的“长文本+高画质”参数
MODEL_SERVER_ARGS="-c 65536 \
--parallel 1 \
--flash-attn on \
--cache-type-k q4_0 \
--cache-type-v q4_0 \
-ngl 99 \
-b 1024 \
-ub 512 \
--n-predict -1 \
--temp 1.1 \
--min-p 0.05 \
--repeat-penalty 1.1 \
--host 0.0.0.0 \
--port 8087"

# --- Script Main Logic ---
clear
echo "========================================================"
echo "    AI Core Modular - Launcher v4.0 (Qwen3-8B Q8 Edition)"
echo "========================================================"

# --- Step 1: Activate Conda Environment ---
echo "[1/6] Activating Conda environment: '${CONDA_ENV_NAME}'..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "❌ CRITICAL: Conda initialization script not found."
    exit 1
fi

conda activate "${CONDA_ENV_NAME}"
echo "✅ Conda environment activated."

# --- Step 2: Install/Update Python Dependencies ---
echo "[2/6] Installing/updating dependencies..."
pip install -r requirements.txt -i ${PIP_MIRROR} --quiet
echo "✅ Python dependencies are up to date."

# --- Step 3: Check Database Container ---
echo "[3/6] Checking database container (${DB_CONTAINER_NAME})..."
if [ -n "$(docker ps -q -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo "✅ Database is running."
else
    echo ">> Starting database..."
    docker start ${DB_CONTAINER_NAME} || echo "⚠️ Warning: Failed to start Docker, make sure Docker Desktop is running."
fi

# --- Step 4: Check for Embedding Model ---
echo "[4/6] Verifying local embedding model..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ Embedding model folder found."
else
    echo "❌ Error: Embedding model folder '${MODEL_DIR}' is missing!"
    exit 1
fi

# --- Step 5: Check and Start LLM Server (8087 端口) ---
echo "[5/6] Checking AI provider configuration..."
if grep -q 'ACTIVE_AI_PROVIDER = "local"' "${CONFIG_FILE_NAME}"; then
    echo ">> AI Provider is 'local'. Checking 8087端口服务器..."
    
    # 检查 8087 端口是否已被占用
    if lsof -Pi :8087 -sTCP:LISTEN -t >/dev/null ; then
        echo "✅ Local LLM server (Port 8087) is already running."
    else
        echo ">> Server on 8087 not running, starting Qwen3-8B-Q8 in background..."
        (
          cd "${MODEL_SERVER_DIR}" && nohup ${MODEL_SERVER_EXEC} -m "${MODEL_PATH}" ${MODEL_SERVER_ARGS} > llama-server.log 2>&1 &
        )
        echo ">> Waiting 8 seconds for Q8 model to fill VRAM..."
        sleep 8
        if lsof -Pi :8087 -sTCP:LISTEN -t >/dev/null ; then
            echo "✅ Qwen3-8B Q8 server started successfully on 8087."
        else
            echo "❌ Error: Failed to start the LLM server. Check ${MODEL_SERVER_DIR}/llama-server.log"
            exit 1
        fi
    fi
fi

# --- Step 6: Launch the Main Application ---
echo "[6/6] Launching Python application: ${PYTHON_SCRIPT_NAME}..."
echo "-------------------------- [ Application Log ] --------------------------"
python "${PYTHON_SCRIPT_NAME}"
