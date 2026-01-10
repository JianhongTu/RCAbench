#!/bin/bash
# Simple test script to run both agents

set -e

echo "Starting RCA Judge (Green Agent) on port 9009..."
python scenarios/arvo_rca/rca_judge.py --host 127.0.0.1 --port 9009 &
GREEN_PID=$!

echo "Waiting for green agent to start..."
sleep 3

echo "Starting RCA Finder (Purple Agent) on port 9019..."
python scenarios/arvo_rca/rca_finder.py --host 127.0.0.1 --port 9019 &
PURPLE_PID=$!

echo "Waiting for purple agent to start..."
sleep 3

echo "Both agents are running!"
echo "Green Agent (Judge): http://127.0.0.1:9009"
echo "Purple Agent (Finder): http://127.0.0.1:9019"
echo ""
echo "Press Ctrl+C to stop both agents"

# Wait for user interrupt
trap "kill $GREEN_PID $PURPLE_PID 2>/dev/null; exit" INT TERM
wait
