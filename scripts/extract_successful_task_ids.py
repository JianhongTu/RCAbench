#!/usr/bin/env python3
"""
One-time script to extract ARVO task IDs from the successful_patches folder.

Parses filenames like 'arvo_1065.diff' and extracts the ID (1065).
Writes all IDs to a text file for use in testing.
"""

import os
import re
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PATCHES_DIR = PROJECT_ROOT / "data" / "successful_patches"
OUTPUT_FILE = PROJECT_ROOT / "data" / "successful_task_ids.txt"


def main():
    # Pattern to match arvo_<id>.diff
    pattern = re.compile(r"^arvo_(\d+)\.diff$")
    
    task_ids = []
    
    # Iterate through all .diff files
    for filename in os.listdir(PATCHES_DIR):
        match = pattern.match(filename)
        if match:
            task_id = match.group(1)
            task_ids.append(task_id)
    
    # Sort numerically
    task_ids.sort(key=int)
    
    # Write to output file
    with open(OUTPUT_FILE, "w") as f:
        for task_id in task_ids:
            f.write(f"{task_id}\n")
    
    print(f"Extracted {len(task_ids)} task IDs from {PATCHES_DIR}")
    print(f"Output written to: {OUTPUT_FILE}")
    print(f"\nFirst 10 IDs: {task_ids[:10]}")
    print(f"Last 10 IDs: {task_ids[-10:]}")


if __name__ == "__main__":
    main()
