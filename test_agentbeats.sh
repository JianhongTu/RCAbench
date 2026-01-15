#!/bin/bash
echo "Starting agentbeats-run with --show-logs..."
uv run agentbeats-run agents/mini-swe-agent/scenario.toml --show-logs &
PID=$!
sleep 15
kill $PID 2>/dev/null || true  
wait $PID 2>/dev/null || true
echo "Test complete"
