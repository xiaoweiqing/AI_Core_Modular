#!/bin/bash

# ==========================================================
#     AI Ecosystem Core - 黄金启动脚本 (v3.0 - 最终版)
# ==========================================================
# 确保在拥有正确“地图”(LD_LIBRARY_PATH)的环境下启动
# ==========================================================

# 【【【 修改点 1: 将“地图”指向我们新的 llama.cpp_new 引擎库！ 】】】
export LD_LIBRARY_PATH=/home/weiyubin/llama.cpp_new/build-vulkan-official/bin:$LD_LIBRARY_PATH

# (可选) 进入 Python 项目所在的目录，这是一个好习惯
# cd /home/weiyubin/projects/AI-Ecosystem-Core/

# 【【【 修改点 2: 启动我们最新的 Python 主程序 35.py！ 】】】
echo ">> [Golden Starter] 正在启动 AI Ecosystem Core (Engine: llama.cpp_new)..."
python3 /home/weiyubin/projects/AI-Ecosystem-Core/35.py