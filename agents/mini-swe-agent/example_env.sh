#!/bin/bash
# Example environment variables for mini-swe-agent
# Copy this file to env.sh and fill in your values
# Source the file: source env.sh

export OPENAI_API_KEY="your-openai-api-key-here"
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

echo "✅ Environment variables set from env.sh"
