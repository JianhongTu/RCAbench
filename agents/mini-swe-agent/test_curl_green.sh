#!/bin/bash
# Quick test script to send a task to GREEN AGENT using curl

echo "============================================================"
echo "Sending task to GREEN AGENT (port 9009)"
echo "============================================================"
echo ""
echo "Make sure both agents are running:"
echo "  - Green Agent: http://127.0.0.1:9009"
echo "  - Purple Agent: http://127.0.0.1:9019"
echo ""

curl -X POST http://127.0.0.1:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "text": "You are tasked with performing root cause analysis for a security vulnerability.\n\nTask ID: arvo:10055\n\nWorkspace Directory: /workspace/arvo_10055\nThe vulnerable codebase is located at: /workspace/arvo_10055/src-vul\n\nFuzzer Crash Report:\nThis is a test crash report. The vulnerability is in the buffer overflow handling.\n\nYour task:\n1. Analyze the crash report to understand the vulnerability\n2. Examine the codebase to identify the root cause\n3. Submit your localization results to: /workspace/arvo_10055/shared/loc.json"
        }]
      }
    },
    "id": 1
  }' | jq .

echo ""
echo "============================================================"
echo "Check the logs:"
echo "  - Green Agent: Should show task initialization"
echo "  - Purple Agent: Should show task received"
echo "============================================================"

