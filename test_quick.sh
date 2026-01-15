#!/bin/bash
echo "Testing AgentBeats mode..."
uv run agentbeats-run agents/mini-swe-agent/scenario.toml --show-logs 2>&1 &
PID=$!
sleep 20
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "Done"
