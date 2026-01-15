#!/usr/bin/env python3
"""
Debug script to investigate ground truth extraction issues.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rcabench.utils import remote_fetch_diff
from rcabench.server.ground_truth_utils import _iter_file_blocks, _is_code_file, HUNK_RE


def debug_ground_truth(arvo_id: str):
    """Debug ground truth extraction for a specific ARVO ID."""
    print(f"\n[DEBUG] Testing ground truth extraction for ARVO {arvo_id}")
    print("=" * 80)
    
    # Step 1: Fetch diff file
    print("\n[STEP 1] Fetching diff file...")
    try:
        diff_path = remote_fetch_diff(arvo_id, use_temp_file=True)
        print(f"[OK] Diff file fetched: {diff_path}")
        
        with open(diff_path, "r", encoding="utf-8", errors="ignore") as f:
            diff_content = f.read()
        
        print(f"[INFO] Diff file size: {len(diff_content)} characters")
        print(f"[INFO] First 500 characters:\n{diff_content[:500]}\n...")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch diff: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Parse file blocks
    print("\n[STEP 2] Parsing file blocks from diff...")
    file_blocks = list(_iter_file_blocks(diff_content))
    print(f"[INFO] Found {len(file_blocks)} file block(s)")
    
    for i, (filepath, block) in enumerate(file_blocks, 1):
        print(f"\n  [{i}] File: {filepath}")
        print(f"      Is code file: {_is_code_file(filepath) if filepath else False}")
        print(f"      Block size: {len(block)} characters")
        
        if filepath and _is_code_file(filepath):
            # Check for hunks
            hunks = list(HUNK_RE.finditer(block))
            print(f"      Hunks found: {len(hunks)}")
            
            for j, h in enumerate(hunks, 1):
                old_start = int(h.group(1))
                old_count = int(h.group(2)) if h.group(2) else 1
                new_start = int(h.group(3))
                new_count = int(h.group(4)) if h.group(4) else 1
                print(f"        Hunk {j}: old={old_start}+{old_count}, new={new_start}+{new_count}")
                # Show a snippet of the hunk
                hunk_start = h.start()
                hunk_end = min(h.end() + 200, len(block))
                print(f"        Hunk content preview:\n{block[hunk_start:hunk_end][:300]}...")
    
    # Step 3: Try to find source files
    print("\n[STEP 3] Checking if source files exist in workspace...")
    from rcabench.task.gen_task import prepare_task_assets
    try:
        task_meta = prepare_task_assets(arvo_id=arvo_id)
        agent_paths = task_meta["agent_paths"]
        workspace_dir = agent_paths.workspace_dir
        
        print(f"[OK] Workspace: {workspace_dir}")
        
        # Check for src-vul directory
        src_vul_dir = workspace_dir / "src-vul"
        if src_vul_dir.exists():
            print(f"[OK] src-vul directory exists: {src_vul_dir}")
            
            # Check if any of the files from the diff exist
            for filepath, _ in file_blocks:
                if filepath and _is_code_file(filepath):
                    possible_paths = [
                        src_vul_dir / filepath,
                        workspace_dir / filepath,
                    ]
                    found = False
                    for path in possible_paths:
                        if path.exists():
                            print(f"[OK] Found source file: {path}")
                            found = True
                            break
                    if not found:
                        print(f"[WARNING] Source file not found: {filepath}")
                        print(f"          Tried: {[str(p) for p in possible_paths]}")
        else:
            print(f"[WARNING] src-vul directory does not exist: {src_vul_dir}")
            
    except Exception as e:
        print(f"[ERROR] Failed to prepare workspace: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/debug_ground_truth.py <arvo_id>")
        sys.exit(1)
    
    arvo_id = sys.argv[1]
    debug_ground_truth(arvo_id)

