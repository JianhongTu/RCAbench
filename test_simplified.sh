#!/bin/bash
echo "Testing simplified mode (always aggregation)..."
uv run agentbeats-run agents/mini-swe-agent/scenario.toml --show-logs 2>&1 &
PID=$!
sleep 25
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "Done"
