#!/usr/bin/env python3
"""Test script to send EvalRequest to green agent."""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentbeats.client import send_message

async def test_eval_request():
    """Test sending an EvalRequest to the green agent."""
    
    # Create EvalRequest (simulating what AgentBeats would send)
    eval_request = {
        "participants": {
            "purple_agent": "http://localhost:9019/"
        },
        "config": {
            "task_ids": ["10055"],  # Use a simple test task
            "num_tasks": 1,
            "max_task_time": 600,
            "max_turns": 50
        }
    }
    
    print("=" * 60)
    print("Testing EvalRequest Flow")
    print("=" * 60)
    print("\nSending EvalRequest to green agent...")
    print(json.dumps(eval_request, indent=2))
    print("\n" + "=" * 60)
    
    try:
        # Send to green agent
        response = await send_message(
            message=json.dumps(eval_request),
            base_url="http://localhost:9009/",
            context_id=None,
        )
        
        print("\nResponse from green agent:")
        print("-" * 60)
        print(response.get("response", "No response"))
        print("-" * 60)
        
        if response.get("status"):
            print(f"\nStatus: {response['status']}")
        
        if response.get("context_id"):
            print(f"Context ID: {response['context_id']}")
        
        print("\n✅ Test completed!")
        print("\nCheck logs for detailed flow:")
        print("  - Green agent: docker logs green-agent")
        print("  - Purple agent: docker logs purple-agent")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_eval_request())

