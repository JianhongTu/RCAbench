#!/usr/bin/env python3
"""
Send a task to the GREEN AGENT.
Minimal message - just arvo_id. Green agent handles everything else.

Usage:
    python send_task.py [arvo_id]
    python send_task.py 14368  # Send specific task
    python send_task.py  # Uses all task_ids from scenario.toml
    python send_task.py --all  # Explicitly use all task_ids from scenario.toml
"""

import asyncio
import argparse
import random
import sys
from pathlib import Path
from agentbeats.client import send_message
from utility import (
    create_run_log_dir,
    get_base_log_dir,
    CURRENT_RUN_LOG_DIR_MARKER,
    get_or_create_run_log_dir,
    get_task_ids_from_config,
)

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python < 3.11
    except ImportError:
        tomllib = None

async def test_send_task_to_green(arvo_id: str):
    """Send a minimal task message to green agent - just arvo_id."""
    
    # Check if marker file already exists (from start_agents.py)
    # If it exists, use that directory instead of creating a new one
    scenario_file = Path(__file__).parent / "scenario.toml"
    from utility import get_base_log_dir, CURRENT_RUN_LOG_DIR_MARKER, get_or_create_run_log_dir
    
    base_log_dir = get_base_log_dir(scenario_file)
    marker_file = base_log_dir / CURRENT_RUN_LOG_DIR_MARKER
    
    if marker_file.exists():
        # Use existing directory from start_agents.py
        run_log_dir = get_or_create_run_log_dir(scenario_file=scenario_file)
        print(f"Using existing log directory: {run_log_dir}")
    else:
        # Create new directory (fallback if agents weren't started with --arvo-id)
        run_log_dir = create_run_log_dir(arvo_id, scenario_file)
        print(f"Created new log directory: {run_log_dir}")
    
    # Minimal message - green agent only needs arvo_id
    # Green agent will:
    # 1. Extract arvo_id
    # 2. Call prepare_task_assets() to fetch real codebase and error report
    # 3. Create full task description
    # 4. Send to purple agent
    task_message = f"Task ID: arvo:{arvo_id}"
    
    print("="*60)
    print(f"Sending task {arvo_id} to GREEN AGENT")
    print(f"Log directory: {run_log_dir}")
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
        print(f"   - Log file: {run_log_dir / 'agents.log'}")
        print("   - Green Agent: Should show task initialization")
        print("   - Purple Agent: Should show task received and command loop starting")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a task to green agent (minimal message)")
    parser.add_argument(
        "arvo_id",
        type=str,
        nargs="?",
        default=None,
        help="ARVO task ID. If not provided, will use all task_ids from scenario.toml config.task_ids"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Send tasks for all ARVO IDs from scenario.toml (ignores arvo_id argument if provided)"
    )
    args = parser.parse_args()
    
    # Load scenario.toml to get task_ids
    scenario_file = Path(__file__).parent / "scenario.toml"
    task_ids = get_task_ids_from_config(scenario_file)
    
    # Determine which tasks to run
    if args.all:
        # Run all tasks from scenario.toml
        if not task_ids:
            print("❌ Error: --all specified but no task_ids found in scenario.toml")
            sys.exit(1)
        arvo_ids = task_ids
        print(f"Running all tasks from scenario.toml: {arvo_ids}")
    elif args.arvo_id:
        # Use provided ARVO ID
        arvo_ids = [args.arvo_id]
    else:
        # Pick ONE random task ID from config
        if task_ids:
            selected_arvo = random.choice(task_ids)
            arvo_ids = [selected_arvo]
            print(f"🎲 Randomly selected ARVO ID from scenario.toml: {selected_arvo}")
        else:
            print("❌ Error: No ARVO ID provided and none found in scenario.toml")
            print("   Please provide ARVO ID: python send_task.py <arvo_id>")
            print("   Or configure task_ids or task_ids_file in scenario.toml")
            sys.exit(1)
    
    # Send tasks
    for arvo_id in arvo_ids:
        print(f"\n{'='*60}")
        print(f"Processing task {arvo_id} ({arvo_ids.index(arvo_id) + 1}/{len(arvo_ids)})")
        print(f"{'='*60}")
        asyncio.run(test_send_task_to_green(arvo_id))
        if len(arvo_ids) > 1:
            print(f"\n✅ Task {arvo_id} completed. Moving to next task...")
            import time
            time.sleep(2)  # Small delay between tasks
    
    if len(arvo_ids) > 1:
        print(f"\n{'='*60}")
        print(f"✅ All {len(arvo_ids)} tasks completed!")
        print(f"{'='*60}")
