#!/usr/bin/env python3
"""
Test script to run both green and purple agents for testing.
"""

import subprocess
import sys
import time
import signal
import os
import argparse
from pathlib import Path

def main():
    """Run both agents in separate processes."""
    parser = argparse.ArgumentParser(description="Test both agents")
    parser.add_argument("--trace", action="store_true", help="Enable conversation trace logging")
    args = parser.parse_args()
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    # Change to project root
    os.chdir(project_root)
    
    green_cmd = [sys.executable, "scenarios/arvo_rca/rca_judge.py", "--host", "127.0.0.1", "--port", "9009"]
    purple_cmd = [sys.executable, "scenarios/arvo_rca/rca_finder.py", "--host", "127.0.0.1", "--port", "9019"]
    
    if args.trace:
        green_cmd.append("--trace")
        purple_cmd.append("--trace")
        print("Trace mode enabled - conversation logs will be displayed")
    
    print("Starting RCA Judge (Green Agent) on port 9009...")
    # If trace is enabled, show output in real-time; otherwise capture it
    if args.trace:
        green_process = subprocess.Popen(
            green_cmd,
        )
    else:
        green_process = subprocess.Popen(
            green_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    
    print("Waiting for green agent to start...")
    time.sleep(3)
    
    print("Starting RCA Finder (Purple Agent) on port 9019...")
    if args.trace:
        purple_process = subprocess.Popen(
            purple_cmd,
        )
    else:
        purple_process = subprocess.Popen(
            purple_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    
    print("Waiting for purple agent to start...")
    time.sleep(3)
    
    print("\n" + "="*60)
    print("Both agents are running!")
    print("Green Agent (Judge): http://127.0.0.1:9009")
    print("Purple Agent (Finder): http://127.0.0.1:9019")
    print("="*60)
    print("\nPress Ctrl+C to stop both agents\n")
    
    def signal_handler(sig, frame):
        print("\nStopping agents...")
        green_process.terminate()
        purple_process.terminate()
        green_process.wait()
        purple_process.wait()
        print("Agents stopped.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for processes
    try:
        green_process.wait()
        purple_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
