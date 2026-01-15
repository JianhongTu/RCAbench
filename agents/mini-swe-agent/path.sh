#!/bin/bash
# Environment variables for mini-swe-agent
# Source this file: source path.sh

export OPENAI_API_KEY=OPENAI_API_KEY
export WORKSPACE_DIR="./workspace"
export LOG_DIR="./logs"
export TMP_DIR="/tmp/rcabench"
export MODEL="gpt-4o-mini"
export MAX_STEPS=50

# Agent URLs
export GREEN_AGENT_URL="http://127.0.0.1:9009/"
export PURPLE_AGENT_URL="http://127.0.0.1:9019/"

# PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../src:$(pwd)/../../rcabench"

echo "✅ Environment variables set from path.sh"
