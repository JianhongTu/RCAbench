#!/usr/bin/env python3
"""
Find simpler Arvo tasks for faster testing.

Looks for tasks with:
- Smaller patches (fewer lines changed)
- Fewer files modified
- Simpler codebases (smaller tarballs)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rcabench.db_utils import LiteDatabase
    from rcabench.utils import remote_fetch_diff
except ImportError:
    print("Error: Could not import rcabench modules. Make sure you're in the RCAbench directory.")
    sys.exit(1)


def analyze_task_simplicity(arvo_id: str, db: LiteDatabase) -> dict:
    """Analyze how simple a task is based on patch characteristics."""
    try:
        # Get patch diff
        diff_path = remote_fetch_diff(arvo_id, use_temp_file=True)
        
        with open(diff_path, "r", encoding="utf-8", errors="ignore") as f:
            diff_content = f.read()
        
        # Count metrics
        file_count = diff_content.count("diff --git") or diff_content.count("diff -")
        hunk_count = diff_content.count("@@")
        total_lines = len(diff_content.split("\n"))
        
        # Estimate patch complexity (more hunks/files = more complex)
        complexity = file_count * 10 + hunk_count * 2
        
        return {
            "arvo_id": arvo_id,
            "file_count": file_count,
            "hunk_count": hunk_count,
            "total_lines": total_lines,
            "complexity_score": complexity,
            "status": "ok"
        }
    except Exception as e:
        return {
            "arvo_id": arvo_id,
            "status": "error",
            "error": str(e)
        }


def main():
    # Load task IDs
    task_ids_file = Path(__file__).parent.parent / "data" / "good_arvo_task_ids.json"
    if not task_ids_file.exists():
        print(f"Error: Task IDs file not found: {task_ids_file}")
        sys.exit(1)
    
    with open(task_ids_file, "r") as f:
        task_ids = json.load(f)
    
    print(f"Analyzing {len(task_ids)} tasks to find simplest ones...")
    print("=" * 60)
    
    # Try to connect to database (optional, for more metadata)
    db = None
    try:
        db_path = Path(__file__).parent.parent / "data" / "arvo.db"
        if db_path.exists():
            db = LiteDatabase(db_path)
            print(f"Connected to database: {db_path}")
    except Exception:
        print("Note: Database not available, using patch analysis only")
    
    # Analyze tasks (limit to first 20 for speed)
    results = []
    for i, task_id in enumerate(task_ids[:20], 1):  # Analyze first 20
        print(f"[{i}/{min(20, len(task_ids))}] Analyzing task {task_id}...", end=" ")
        result = analyze_task_simplicity(task_id, db)
        results.append(result)
        if result["status"] == "ok":
            print(f"✓ Files: {result['file_count']}, Hunks: {result['hunk_count']}, Complexity: {result['complexity_score']}")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown')}")
    
    # Sort by complexity (lower = simpler)
    successful_results = [r for r in results if r["status"] == "ok"]
    successful_results.sort(key=lambda x: x["complexity_score"])
    
    print("\n" + "=" * 60)
    print("SIMPLEST TASKS (best for fast testing):")
    print("=" * 60)
    
    for i, result in enumerate(successful_results[:10], 1):
        print(f"{i}. Task {result['arvo_id']:>6} - "
              f"Files: {result['file_count']:>2}, "
              f"Hunks: {result['hunk_count']:>3}, "
              f"Complexity: {result['complexity_score']:>4}")
    
    if successful_results:
        simplest = successful_results[0]
        print(f"\n✅ Recommended for fast testing: {simplest['arvo_id']}")
        print(f"   (Lowest complexity: {simplest['complexity_score']})")


if __name__ == "__main__":
    main()



