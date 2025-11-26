"""
analyze_patch_verification_results.py - Analyze patch verification results

This script analyzes the results from batch patch verification and generates
summary statistics and reports.

Usage:
    python3 scripts/analyze_patch_verification_results.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import sqlite3
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "patch_verification_results"
DB_PATH = PROJECT_ROOT / "data" / "patch_verification.db"


def load_results_from_db() -> List[Dict]:
    """Load all results from the database."""
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patch_verification")
    rows = cursor.fetchall()

    columns = [
        "task_id",
        "status",
        "patch_applied",
        "compiled",
        "fuzzer_passed",
        "error_message",
        "k8s_job_name",
        "start_time",
        "end_time",
        "retry_count",
    ]

    results = [dict(zip(columns, row)) for row in rows]
    conn.close()

    return results


def load_results_from_json() -> List[Dict]:
    """Load results from individual JSON files."""
    results = []

    if not RESULTS_DIR.exists():
        return results

    for json_file in RESULTS_DIR.glob("*_result.json"):
        try:
            with open(json_file, "r") as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze the verification results."""
    if not results:
        return {}

    total_tasks = len(results)
    successful = len([r for r in results if r.get("status") == "success"])
    failed = len([r for r in results if r.get("status") == "failed"])

    # Breakdown by failure type
    failure_types = defaultdict(int)
    for result in results:
        if result.get("status") == "failed":
            error_msg = result.get("error_message", "Unknown error")
            if error_msg and "Patch application failed" in error_msg:
                failure_types["patch_failed"] += 1
            elif error_msg and "Compilation failed" in error_msg:
                failure_types["compile_failed"] += 1
            elif error_msg and "Fuzzer returned non-zero" in error_msg:
                failure_types["fuzzer_failed"] += 1
            else:
                failure_types["other"] += 1

    # Success rate
    success_rate = (successful / total_tasks * 100) if total_tasks > 0 else 0

    # Average execution time
    execution_times = []
    for result in results:
        if result.get("start_time") and result.get("end_time"):
            execution_times.append(result["end_time"] - result["start_time"])

    avg_execution_time = (
        sum(execution_times) / len(execution_times) if execution_times else 0
    )

    return {
        "total_tasks": total_tasks,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "failure_breakdown": dict(failure_types),
        "avg_execution_time": avg_execution_time,
        "results": results,
    }


def generate_report(analysis: Dict):
    """Generate a human-readable report."""
    if not analysis:
        return "No results found to analyze."

    report = []
    report.append("=" * 60)
    report.append("PATCH VERIFICATION RESULTS REPORT")
    report.append("=" * 60)
    report.append("")

    report.append("SUMMARY:")
    report.append(f"  Total tasks processed: {analysis['total_tasks']}")
    report.append(f"  Successful verifications: {analysis['successful']}")
    report.append(f"  Failed verifications: {analysis['failed']}")
    report.append(".1f")
    report.append("")

    if analysis["failure_breakdown"]:
        report.append("FAILURE BREAKDOWN:")
        for failure_type, count in analysis["failure_breakdown"].items():
            report.append(f"  {failure_type}: {count}")
        report.append("")

    report.append("PERFORMANCE:")
    report.append(".2f")
    report.append("")

    # List failed tasks
    failed_tasks = [r for r in analysis["results"] if r.get("status") == "failed"]
    if failed_tasks:
        report.append("FAILED TASKS:")
        for task in failed_tasks[:10]:  # Show first 10
            report.append(
                f"  {task['task_id']}: {task.get('error_message', 'Unknown error')}"
            )
        if len(failed_tasks) > 10:
            report.append(f"  ... and {len(failed_tasks) - 10} more")
        report.append("")

    return "\n".join(report)


def save_detailed_report(
    analysis: Dict, output_file: str = "patch_verification_detailed_report.json"
):
    """Save detailed analysis to JSON file."""
    output_path = PROJECT_ROOT / "data" / output_file

    # Remove the 'results' key to avoid huge JSON files
    summary = {k: v for k, v in analysis.items() if k != "results"}

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Detailed report saved to: {output_path}")


def main():
    print("Analyzing patch verification results...")

    # Load results from both sources
    db_results = load_results_from_db()
    json_results = load_results_from_json()

    # Merge results (prefer DB over JSON for conflicts)
    all_results = {}
    for result in json_results + db_results:
        task_id = result["task_id"]
        if task_id not in all_results or result.get("status") in ["success", "failed"]:
            all_results[task_id] = result

    results = list(all_results.values())

    if not results:
        print("No results found.")
        return

    # Analyze results
    analysis = analyze_results(results)

    # Generate and print report
    report = generate_report(analysis)
    print(report)

    # Save detailed report
    save_detailed_report(analysis)

    print(f"\nAnalysis complete. Processed {len(results)} tasks.")


if __name__ == "__main__":
    main()
