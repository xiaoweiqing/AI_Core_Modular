#!/bin/bash

# ==============================================================================
#      AI Core Modular - Launcher v3.0 (Conda Integrated Edition)
# ==============================================================================
# This script has been modified to use a Conda environment instead of venv,
# providing greater stability against system-level changes. It retains the
# smart logic for checking dependencies and conditionally launching the LLM.
# ==============================================================================

# --- 1. Project Configuration ---
CONDA_ENV_NAME="aic-final" # <-- MODIFIED: Specify your Conda environment name here
PYTHON_SCRIPT_NAME="main.py"
CONFIG_FILE_NAME="config.py"
DB_CONTAINER_NAME="ai_database_hub"
MODEL_DIR="all-MiniLM-L6-v2"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# --- Local LLM Server Configuration ---
MODEL_SERVER_DIR="$HOME/llama.cpp/build-vulkan-new"
MODEL_PATH="/mnt/data/model/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf"
MODEL_SERVER_EXEC="./bin/llama-server"
MODEL_SERVER_ARGS="-c 56666 -ngl 99 --repeat-penalty 1.1 --presence-penalty 0.5 --top-k 40 --top-p 0.95 --host 0.0.0.0 --port 8087"

# --- Script Main Logic ---
clear
echo "========================================================"
echo "    AI Core Modular - Launcher v3.0 (Conda Edition)"
echo "========================================================"
echo ""

# --- Step 1: Activate Conda Environment ---
# MODIFIED: Replaced the entire venv block with robust Conda activation.
echo "[1/6] Activating Conda environment: '${CONDA_ENV_NAME}'..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "❌ CRITICAL: Conda initialization script not found."
    echo "   Please ensure Miniconda or Anaconda is installed correctly."
    read -p "Press any key to exit..."; exit 1
fi

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "❌ CRITICAL: Failed to activate Conda environment '${CONDA_ENV_NAME}'."
    echo "   Please ensure the environment exists by running: conda env list"
    read -p "Press any key to exit..."; exit 1
fi
echo "✅ Conda environment activated."
echo ""
# --- END MODIFICATION ---


# --- Step 2: Install/Update Python Dependencies ---
# This step now works correctly because `pip` will be the one from the Conda env.
echo "[2/6] Installing/updating dependencies from requirements.txt..."
if ! pip install -r requirements.txt -i ${PIP_MIRROR}; then
    echo "❌ Critical Error: Python dependency installation failed!"
    read -p "Press any key to exit..."
    exit 1
fi
echo "✅ Python dependencies are up to date."
echo ""

# --- Step 3: Check Database Container ---
echo "[3/6] Checking database container (${DB_CONTAINER_NAME}) status..."
if [ -n "$(docker ps -q -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo "✅ Database is running."
elif [ -n "$(docker ps -aq -f name=^/${DB_CONTAINER_NAME}$)" ]; then
    echo ">> Found stopped database, restarting..."
    docker start ${DB_CONTAINER_NAME}
    echo "✅ Database started."
else
    echo "❌ Critical Error: Docker container '${DB_CONTAINER_NAME}' not found!"
    read -p "Press any key to exit..."
    exit 1
fi
echo ""

# --- Step 4: Check for Embedding Model ---
echo "[4/6] Verifying local embedding model..."
if [ -d "$MODEL_DIR" ]; then
    echo "✅ Embedding model folder '${MODEL_DIR}' found."
else
    echo "❌ Error: Embedding model folder '${MODEL_DIR}' is missing!"
    read -p "Press any key to exit..."
    exit 1
fi
echo ""

# --- Step 5: Check and Start LLM Server (Conditional) ---
echo "[5/6] Checking AI provider configuration in '${CONFIG_FILE_NAME}'..."
if [ ! -f "${CONFIG_FILE_NAME}" ]; then
    echo "❌ Critical Error: Configuration file '${CONFIG_FILE_NAME}' not found!"
    read -p "Press any key to exit..."
    exit 1
fi

if grep -q 'ACTIVE_AI_PROVIDER = "local"' "${CONFIG_FILE_NAME}"; then
    echo ">> AI Provider is 'local'. Checking local LLM server..."
    if pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
        echo "✅ Local LLM server is already running."
    else
        echo ">> Server not running, starting it in the background..."
        (
          cd "${MODEL_SERVER_DIR}" && nohup ${MODEL_SERVER_EXEC} -m "${MODEL_PATH}" ${MODEL_SERVER_ARGS} > llama-server.log 2>&1 &
        )
        echo ">> Waiting 5 seconds for server to initialize..."
        sleep 5
        if pgrep -f "llama-server -m ${MODEL_PATH}" > /dev/null; then
            echo "✅ Server started successfully."
        else
            echo "❌ Critical Error: Failed to start the local LLM server!"
            read -p "Press any key to exit..."
            exit 1
        fi
    fi
else
    echo ">> AI Provider is not 'local'. Skipping local server launch."
fi
echo ""


# --- Step 6: Launch the Main Application ---
# This now correctly uses the Python interpreter from your activated Conda env.
echo "[6/6] Launching Python application: ${PYTHON_SCRIPT_NAME}..."
echo "-------------------------- [ Application Log ] --------------------------"
echo ""
python "${PYTHON_SCRIPT_NAME}"

# --- Script End ---
echo ""
echo "-------------------------- [  Log End  ] --------------------------"
read -p "Application has finished. Press any key to exit..."
