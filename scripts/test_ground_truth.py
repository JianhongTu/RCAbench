#!/usr/bin/env python3
"""
Test script to verify ground truth extraction for ARVO tasks.

This script:
1. Picks a random task from the task list
2. Extracts ground truth from patch.diff
3. Finds actual line numbers in the vulnerable codebase
4. Applies ±5 line buffer
5. Derives function names using tree-sitter
6. Prints the results
"""

import sys
import random
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
from rcabench.server.eval_utils import get_ground_truth
from rcabench.server.ground_truth_utils import augment_ground_truth_with_functions


def test_ground_truth(task_id: str = None):
    """
    Test ground truth extraction for a given task ID.
    
    Args:
        task_id: ARVO task ID to test. If None, picks a random one from the list.
    """
    # Read task IDs from the list
    task_ids_file = project_root / "data" / "arvo_hf_task_ids.txt"
    
    if not task_ids_file.exists():
        print(f"[ERROR] Task IDs file not found: {task_ids_file}")
        return
    
    with open(task_ids_file, 'r') as f:
        available_task_ids = [line.strip() for line in f if line.strip()]
    
    # Pick a task ID
    if task_id is None:
        task_id = random.choice(available_task_ids)
        print(f"[RANDOM] Selected task: {task_id}")
    else:
        print(f"[TEST] Testing task: {task_id}")
        if task_id not in available_task_ids:
            print(f"[WARNING] Task {task_id} not in the available task list")
    
    print("=" * 80)
    
    # Prepare task assets
    print(f"\n[SETUP] Preparing task assets for {task_id}...")
    try:
        task_meta = prepare_task_assets(arvo_id=task_id)
        agent_paths = task_meta["agent_paths"]
        workspace_dir = agent_paths.workspace_dir
        
        print(f"[OK] Workspace created at: {agent_paths.agent_dir}")
        print(f"   - Workspace: {workspace_dir}")
        print(f"   - Shared dir: {agent_paths.shared_dir}")
        
    except Exception as e:
        print(f"[ERROR] Failed to prepare task assets: {e}")
        return
    
    try:
        # Step 1: Extract ground truth from patch.diff
        print(f"\n[STEP 1] Extracting ground truth from patch.diff...")
        asset_path = str(agent_paths.agent_dir)
        gts = get_ground_truth(task_id, asset_path=asset_path)
        
        if not gts:
            print(f"[ERROR] No ground truth found for task {task_id}")
            return
        
        print(f"[OK] Found {len(gts)} ground truth localization(s)")
        print("\n[RAW] Ground truth (before function augmentation):")
        for i, gt in enumerate(gts, 1):
            print(f"\n  [{i}] File: {gt.file}")
            print(f"      Line span: {gt.old_span.start}-{gt.old_span.end} (±5 buffer applied)")
            print(f"      Task ID: {gt.task_id}")
            print(f"      Function: {gt.function if gt.function else '(not yet derived)'}")
        
        # Step 2: Augment with function names
        print(f"\n[STEP 2] Deriving function names using tree-sitter...")
        gts_augmented = augment_ground_truth_with_functions(gts, workspace_dir, trace_only=True)
        
        print(f"[OK] Function names derived")
        print("\n[FINAL] Ground truth (with functions):")
        print("=" * 80)
        
        for i, gt in enumerate(gts_augmented, 1):
            print(f"\n[{i}] {gt.file}")
            print(f"    Lines: {gt.old_span.start}-{gt.old_span.end}")
            print(f"    Function: {gt.function if gt.function else '(function not found)'}")
            print(f"    Task: {gt.task_id}")
        
        # Also print as JSON for easy copying
        print("\n" + "=" * 80)
        print("[JSON] Ground truth as JSON:")
        print("=" * 80)
        gt_json = []
        for gt in gts_augmented:
            gt_json.append({
                "file": gt.file,
                "lines": f"{gt.old_span.start}-{gt.old_span.end}",
                "function": gt.function if gt.function else "",
                "task_id": gt.task_id
            })
        print(json.dumps(gt_json, indent=2))
        
        # Summary
        print("\n" + "=" * 80)
        print("[SUMMARY]")
        print(f"   Total ground truths: {len(gts_augmented)}")
        functions_found = sum(1 for gt in gts_augmented if gt.function)
        print(f"   Functions found: {functions_found}/{len(gts_augmented)}")
        
        if functions_found < len(gts_augmented):
            print("\n[WARNING] Some functions could not be derived. This might be because:")
            print("   - The line is not inside a function (e.g., global variable, header file)")
            print("   - The file path doesn't match the workspace structure")
            print("   - The file type is not supported by tree-sitter")
        
    except Exception as e:
        print(f"\n[ERROR] Error during ground truth extraction: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n[CLEANUP] Cleaning up task assets...")
        try:
            cleanup_task_assets(agent_paths)
            print(f"[OK] Cleanup complete")
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test ground truth extraction for ARVO tasks"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Specific task ID to test (e.g., '10400'). If not provided, picks a random one."
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("GROUND TRUTH EXTRACTION TEST")
    print("=" * 80)
    
    test_ground_truth(task_id=args.task_id)
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
