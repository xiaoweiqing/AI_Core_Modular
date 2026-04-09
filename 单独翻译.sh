#!/bin/bash

# ==============================================================================
#      Standalone Local Translator - Launcher v1.0
# ==============================================================================
# This script prepares the environment and reliably starts the local_translator.py script.
# It checks for:
#   1. Python virtual environment
#   2. Required Python libraries
#   3. System-level clipboard tools
#   4. The running status of the local AI service on port 8087
# ==============================================================================

# --- Configuration ---
PYTHON_SCRIPT_NAME="29.py"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
LOCAL_AI_HOST="127.0.0.1"
LOCAL_AI_PORT="8087"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple" # Optional: for faster installs

# --- Main Logic ---
clear
echo "========================================================"
echo "    Standalone Local Translator Launcher v1.0"
echo "========================================================"
echo ""

# --- Step 1: Check for Virtual Environment ---
echo "[1/5] Checking for Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo ">> Virtual environment not found. Creating '$VENV_DIR'..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo "❌ FATAL ERROR: Failed to create virtual environment. Please check your Python3 installation."
        read -p "Press any key to exit..."
        exit 1
    fi
fi
PYTHON_EXEC="$VENV_DIR/bin/python3"
PIP_EXEC="$VENV_DIR/bin/pip"
echo "✅ Virtual environment is ready."
echo ""

# --- Step 2: Install/Update Python Dependencies ---
echo "[2/5] Installing dependencies from '$REQUIREMENTS_FILE'..."
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ FATAL ERROR: '$REQUIREMENTS_FILE' not found in the current directory!"
    read -p "Press any key to exit..."
    exit 1
fi
if ! ${PIP_EXEC} install --upgrade --default-timeout=200 -i ${PIP_MIRROR} -r "$REQUIREMENTS_FILE"; then
    echo "❌ FATAL ERROR: Failed to install Python dependencies. Please check the error messages above."
    read -p "Press any key to exit..."
    exit 1
fi
echo "✅ Python dependencies are up to date."
echo ""

# --- Step 3: Check for System Tools ---
echo "[3/5] Checking for system clipboard tools (wl-clipboard, xclip)..."
if ! command -v wl-copy &> /dev/null && ! command -v xclip &> /dev/null; then
    echo "❌ FATAL ERROR: No clipboard tool found!"
    echo "   Please install at least one: 'sudo dnf install wl-clipboard xclip'"
    read -p "Press any key to exit..."
    exit 1
fi
echo "✅ System tools found."
echo ""

# --- Step 4: Check if Local AI Service is Running ---
echo "[4/5] Checking for Local AI service on ${LOCAL_AI_HOST}:${LOCAL_AI_PORT}..."
if command -v nc &> /dev/null; then
    if ! nc -zvw1 ${LOCAL_AI_HOST} ${LOCAL_AI_PORT} &> /dev/null; then
        echo "❌ FATAL ERROR: Cannot connect to local AI on port ${LOCAL_AI_PORT}."
        echo "   Please make sure your local AI service (Ollama, LiteLLM, etc.) is running."
        read -p "Press any key to exit..."
        exit 1
    fi
else
    echo ">> Warning: 'nc' (netcat) not found. Skipping AI service check. The script might fail if the service isn't running."
fi
echo "✅ Local AI service is responding."
echo ""

# --- Step 5: Launch the Main Application ---
echo "[5/5] All checks passed. Launching the Local Translator..."
echo "-------------------------- [ Application Log ] --------------------------"
echo ""
${PYTHON_EXEC} "${PYTHON_SCRIPT_NAME}"

# --- Script End ---
echo ""
echo "-----------------------------------------------------------------------"
read -p "The translator has stopped. Press any key to close this window..."