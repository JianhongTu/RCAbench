#!/usr/bin/env python3
"""
Start both green and purple agents from scenario.toml configuration.

Usage:
    python start_agents.py [--scenario scenario.toml]
"""

import subprocess
import sys
import argparse
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python < 3.11
    except ImportError:
        print("Error: tomllib (Python 3.11+) or tomli package required")
        print("Install with: pip install tomli")
        sys.exit(1)

def start_agents(scenario_file: Path):
    """Start agents based on scenario.toml configuration."""
    
    if not scenario_file.exists():
        print(f"Error: Scenario file not found: {scenario_file}")
        sys.exit(1)
    
    with open(scenario_file, "rb") as f:
        config = tomllib.load(f)
    
    green_cmd = config["green_agent"]["cmd"]
    purple_cmd = config["participants"][0]["cmd"]  # First participant is purple
    
    # Change to the directory where the script is located (agents/mini-swe-agent/)
    # This ensures relative paths in scenario.toml work correctly
    script_dir = Path(__file__).parent
    original_cwd = Path.cwd()
    
    print("="*60)
    print("Starting agents from scenario.toml")
    print("="*60)
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
    print(f"  python test_send_task_to_green.py <arvo_id>")
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
    args = parser.parse_args()
    
    start_agents(args.scenario)

