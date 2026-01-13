#!/usr/bin/env python3
"""
Test script to send a task to the GREEN AGENT.
Minimal message - just arvo_id. Green agent handles everything else.

Usage:
    python test_send_task_to_green.py <arvo_id>
    python test_send_task_to_green.py 10055
"""

import asyncio
import argparse
import sys
from agentbeats.client import send_message

async def test_send_task_to_green(arvo_id: str):
    """Send a minimal task message to green agent - just arvo_id."""
    
    # Minimal message - green agent only needs arvo_id
    # Green agent will:
    # 1. Extract arvo_id
    # 2. Call prepare_task_assets() to fetch real codebase and error report
    # 3. Create full task description
    # 4. Send to purple agent
    task_message = f"Task ID: arvo:{arvo_id}"
    
    print("="*60)
    print(f"Sending task {arvo_id} to GREEN AGENT")
    print("="*60)
    print(f"Message: {task_message}")
    print("\nGreen Agent will:")
    print("1. Extract arvo_id")
    print("2. Prepare task assets (download codebase, error report)")
    print("3. Initialize ARVO container")
    print("4. Create full task description")
    print("5. Send to Purple Agent")
    print("="*60)
    print()
    
    try:
        outputs = await send_message(
            message=task_message,
            base_url="http://127.0.0.1:9009/",
            context_id=None,
        )
        
        print("\n" + "="*60)
        print("Response from GREEN AGENT:")
        print("="*60)
        print(outputs.get("response", "No response"))
        print("="*60)
        print(f"\nContext ID: {outputs.get('context_id', 'None')}")
        print(f"Status: {outputs.get('status', 'completed')}")
        print("\n✅ Check both agent logs to see the full flow!")
        print("   - Green Agent: Should show task initialization")
        print("   - Purple Agent: Should show task received and command loop starting")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a task to green agent (minimal message)")
    parser.add_argument("arvo_id", type=str, help="ARVO task ID (e.g., 10055)")
    args = parser.parse_args()
    
    asyncio.run(test_send_task_to_green(args.arvo_id))
