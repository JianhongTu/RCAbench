#!/usr/bin/env python3
"""
Start both green and purple agents from scenario.toml configuration.

Usage:
    python start_agents.py [--scenario scenario.toml] [--arvo-id ARVO_ID]
    
If --arvo-id is provided, creates the log directory marker file before starting agents.
This ensures agents use the correct log directory for the task.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import Optional
from utility import create_run_log_dir, get_random_task_id_from_config

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python < 3.11
    except ImportError:
        print("Error: tomllib (Python 3.11+) or tomli package required")
        print("Install with: pip install tomli")
        sys.exit(1)

def start_agents(scenario_file: Path, arvo_id: Optional[str] = None):
    """Start agents based on scenario.toml configuration."""
    
    if not scenario_file.exists():
        print(f"Error: Scenario file not found: {scenario_file}")
        sys.exit(1)
    
    # Load config to get agent commands
    with open(scenario_file, "rb") as f:
        config = tomllib.load(f)
    
    # If arvo_id not provided, try to get from scenario.toml
    if arvo_id is None:
        arvo_id = get_random_task_id_from_config(scenario_file)
        if arvo_id:
            print(f"🎲 Randomly selected ARVO ID from scenario.toml: {arvo_id}")
        else:
            print("⚠️  Warning: No ARVO ID provided and none found in scenario.toml")
            print("   Agents will create their own log directory")
    
    # If arvo_id is provided, create the log directory marker file BEFORE starting agents
    # This ensures agents use the correct log directory
    # Note: Directory is timestamped, not per-ARVO. Per-ARVO logs are separate files.
    if arvo_id:
        print("="*60)
        print(f"Creating log directory for run (ARVO task: {arvo_id})")
        print("="*60)
        run_log_dir = create_run_log_dir(arvo_id, scenario_file)
        print(f"✅ Log directory created: {run_log_dir}")
        print(f"✅ Marker file created: {scenario_file.parent / 'logs' / '.current_run_log_dir'}")
        print(f"   Per-ARVO logs will be: {run_log_dir / 'arvo_{arvo_id}.log'}")
        print()
    
    green_cmd = config["green_agent"]["cmd"]
    purple_cmd = config["participants"][0]["cmd"]  # First participant is purple
    
    # Change to the directory where the script is located (agents/mini-swe-agent/)
    # This ensures relative paths in scenario.toml work correctly
    script_dir = Path(__file__).parent
    original_cwd = Path.cwd()
    
    print("="*60)
    print("Starting agents from scenario.toml")
    print("="*60)
    if arvo_id:
        print(f"ARVO Task ID: {arvo_id}")
        print(f"Log Directory: {run_log_dir}")
        print()
    print(f"Green Agent: {green_cmd}")
    print(f"Purple Agent: {purple_cmd}")
    print("="*60)
    print("\nStarting agents in background...")
    print("(They will run in the current terminal)")
    print("\nTo stop: Press Ctrl+C or kill the processes")
    print("="*60)
    print()
    
    # Start green agent (run from script directory)
    print("Starting Green Agent...")
    green_process = subprocess.Popen(
        green_cmd.split(),
        cwd=script_dir,  # Run from agents/mini-swe-agent/ directory
    )
    print(f"✅ Green Agent started (PID: {green_process.pid})")
    
    # Small delay to let green agent start
    import time
    time.sleep(1)
    
    # Start purple agent (run from script directory)
    print("Starting Purple Agent...")
    purple_process = subprocess.Popen(
        purple_cmd.split(),
        cwd=script_dir,  # Run from agents/mini-swe-agent/ directory
    )
    print(f"✅ Purple Agent started (PID: {purple_process.pid})")
    
    print("\n" + "="*60)
    print("Both agents are running!")
    print("="*60)
    print(f"Green Agent: http://127.0.0.1:9009/ (PID: {green_process.pid})")
    print(f"Purple Agent: http://127.0.0.1:9019/ (PID: {purple_process.pid})")
    print("\nTo send a task:")
    print(f"  python send_task.py <arvo_id>")
    print("\nPress Ctrl+C to stop both agents")
    print("="*60)
    
    try:
        # Wait for both processes
        green_process.wait()
        purple_process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping agents...")
        green_process.terminate()
        purple_process.terminate()
        green_process.wait()
        purple_process.wait()
        print("✅ Agents stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start agents from scenario.toml")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).parent / "scenario.toml",
        help="Path to scenario.toml file"
    )
    parser.add_argument(
        "--arvo-id",
        type=str,
        default=None,
        help="ARVO task ID. If not provided, will use first task_id from scenario.toml config.task_ids"
    )
    args = parser.parse_args()
    
    start_agents(args.scenario, args.arvo_id)

