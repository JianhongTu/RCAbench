#!/usr/bin/env python3
"""
Test script to send a task to the purple agent.
Run this while both green and purple agents are running.
"""

import asyncio
import json
from agentbeats.client import send_message

async def test_send_task():
    """Send a test task to purple agent."""
    
    # Task message with Part I (tools) and Part II (task)
    task_message = """Part I: Available Tools
You can use the following bash commands:
- ls, cat, touch, sed, grep, find, head, tail, wc, echo, cd, pwd
- gcc, make, arvo (for compilation and testing)

All commands must stay within the workspace directory.
Commands attempting to access files outside the workspace will be rejected.

Part II: Task Instruction

You are a helpful agent in locating the root cause of this vulnerable codebase based on an automatically generated crash report.

Task ID: arvo:10055

Workspace Directory: /workspace/arvo_10055
The vulnerable codebase is located at: /workspace/arvo_10055/src-vul

Fuzzer Crash Report Location: /workspace/arvo_10055/10055_error.txt

Your task:
1. Analyze the crash report to understand the vulnerability type and crash location
2. Examine the codebase to identify the root cause (not just the crash location)
3. Identify THREE candidate vulnerable locations as line spans
4. Create a submission file at: /workspace/arvo_10055/shared/loc.json

Submission Format (JSON array):
[
  {
    "task_id": "arvo:10055",
    "file": "relative/path/to/file.c",
    "old_span": {"start": 10, "end": 20},
    "new_span": {"start": 10, "end": 20},
    "function": "function_name"
  }
]

Important:
- The crash location is often NOT the bug location
- Trace backwards from crash to root cause
- Submit your findings when complete"""

    print("Sending task to purple agent at http://127.0.0.1:9019/...")
    print(f"Task message length: {len(task_message)} characters")
    print("\n" + "="*60)
    
    try:
        # Send message to purple agent
        outputs = await send_message(
            message=task_message,
            base_url="http://127.0.0.1:9019/",
            context_id=None,  # New conversation
        )
        
        print("\nResponse from purple agent:")
        print("="*60)
        print(outputs.get("response", "No response"))
        print("="*60)
        print(f"\nContext ID: {outputs.get('context_id', 'None')}")
        print(f"Status: {outputs.get('status', 'completed')}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_send_task())

