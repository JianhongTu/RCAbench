#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append('/home/tovitu/codes/RCAbench/src')

from rcabench.server.server_utils import run_arvo_container

if __name__ == "__main__":
    arvo_id = "25402"
    mode = "vul"
    patch_dir = Path("./tmp/patch")
    try:
        print(f"Running arvo container for {arvo_id} in {mode} mode...")
        exit_code, docker_output = run_arvo_container(arvo_id, mode, patch_dir)
        print(f"Exit code: {exit_code}")
        print("Docker output:")
        print(docker_output.decode('utf-8', errors='ignore'))
    except Exception as e:
        print(f"Error: {e}")
