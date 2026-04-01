#!/bin/bash

# ==============================================================================
#      Stop LLM Server - Auto Cleanup Script (v1.0)
# ==============================================================================

# --- 配置信息（必须和 launcher.sh 一致） ---
MODEL_PATH="/mnt/data/model/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf"
LLAMA_PID_FILE="/tmp/llama-server-${MODEL_PATH##*/}.pid"
SERVER_LOG="$HOME/logs/llama-server.log"

# --- 检查是否已有进程在运行 ---
check_server_running() {
    pgrep -f "llama-server.*${MODEL_PATH}" > /dev/null
}

# --- 实际停止服务器（kill + 等待）---
stop_llama_server() {
    if [ ! -f "$LLAMA_PID_FILE" ]; then
        echo "⚠️ No PID file found. Maybe server wasn't started by launcher.sh?"
        return 1
    fi

    PID=$(cat "$LLAMA_PID_FILE")
    if kill -0 $PID > /dev/null; then
        echo ">> [Stop] Sending SIGTERM to process (PID: $PID)..."
        kill $PID
        wait $PID
        rm -f "$LLAMA_PID_FILE"
        echo "✅ LLM Server stopped successfully."
    else
        echo "❌ Process not found or already dead."
    fi
}

# --- 主逻辑 ---
echo "========================================================"
echo "     STOP LLM SERVER (Graceful Shutdown) - v1.0"
echo "========================================================"

if check_server_running; then
    stop_llama_server
else
    echo "✅ No active llama-server found. Nothing to do."
fi

read -p "Press any key to exit..."

