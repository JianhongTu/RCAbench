"""
generate_statistics.py - Generate statistics for verified ARVO tasks

Usage:
    python3 scripts/generate_statistics.py                    # Use verified_jobs.json
    python3 scripts/generate_statistics.py --input verified_jobs.json
    python3 scripts/generate_statistics.py --output stats_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import validation functions
from validate import validate_assets

OUTPUT_DIR = "./data/pipeline_results"


def aggregate_statistics(results: List[Dict]) -> Dict:
    """Aggregate statistics from all task results."""
    stats = {
        'total_tasks': len(results),
        'successful_analyses': 0,
        'failed_analyses': 0,
        'difficulty_distribution': defaultdict(int),
        'patch_analysis': {
            'num_files': {'min': float('inf'), 'max': 0, 'avg': 0, 'total': 0},
            'num_code_files': {'min': float('inf'), 'max': 0, 'avg': 0, 'total': 0},
            'lines_added': {'min': float('inf'), 'max': 0, 'avg': 0, 'total': 0},
            'lines_removed': {'min': float('inf'), 'max': 0, 'avg': 0, 'total': 0},
            'total_changes': {'min': float('inf'), 'max': 0, 'avg': 0, 'total': 0},
        },
        'patch_applies': 0,
        'patch_does_not_apply': 0,
    }
    
    successful_results = []
    
    for result in results:
        if result.get('error'):
            stats['failed_analyses'] += 1
            continue
        
        stats['successful_analyses'] += 1
        successful_results.append(result)
        
        # Difficulty distribution
        difficulty = result.get('difficulty', 'unknown')
        stats['difficulty_distribution'][difficulty] += 1
        
        # Patch applies
        if result.get('patch_applies'):
            stats['patch_applies'] += 1
        else:
            stats['patch_does_not_apply'] += 1
        
        # Patch analysis statistics
        analysis = result.get('patch_analysis')
        if analysis:
            for key in ['num_files', 'num_code_files', 'lines_added', 'lines_removed', 'total_changes']:
                value = analysis.get(key, 0)
                stats['patch_analysis'][key]['min'] = min(stats['patch_analysis'][key]['min'], value)
                stats['patch_analysis'][key]['max'] = max(stats['patch_analysis'][key]['max'], value)
                stats['patch_analysis'][key]['total'] += value
    
    # Calculate averages
    if successful_results:
        for key in stats['patch_analysis']:
            stats['patch_analysis'][key]['avg'] = stats['patch_analysis'][key]['total'] / len(successful_results)
            if stats['patch_analysis'][key]['min'] == float('inf'):
                stats['patch_analysis'][key]['min'] = 0
    
    # Convert defaultdict to regular dict
    stats['difficulty_distribution'] = dict(stats['difficulty_distribution'])
    
    return stats


def print_summary(stats: Dict, results: List[Dict]):
    """Print a human-readable summary of statistics."""
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total tasks analyzed: {stats['total_tasks']}")
    print(f"  Successful analyses: {stats['successful_analyses']}")
    print(f"  Failed analyses: {stats['failed_analyses']}")
    
    print(f"\n📈 Difficulty Distribution:")
    for difficulty, count in sorted(stats['difficulty_distribution'].items()):
        percentage = (count / stats['successful_analyses'] * 100) if stats['successful_analyses'] > 0 else 0
        print(f"  {difficulty.capitalize():8s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n🔧 Patch Application:")
    print(f"  Patches apply: {stats['patch_applies']}")
    print(f"  Patches don't apply: {stats['patch_does_not_apply']}")
    
    print(f"\n📝 Patch Analysis Statistics:")
    for metric, values in stats['patch_analysis'].items():
        metric_name = metric.replace('_', ' ').title()
        print(f"  {metric_name}:")
        print(f"    Min:  {values['min']:6.1f}")
        print(f"    Max:  {values['max']:6.1f}")
        print(f"    Avg:  {values['avg']:6.1f}")
        print(f"    Total: {values['total']:6.0f}")
    
    # Show failed tasks if any
    failed_tasks = [r for r in results if r.get('error')]
    if failed_tasks:
        print(f"\n❌ Failed Tasks ({len(failed_tasks)}):")
        for task in failed_tasks[:10]:  # Show first 10
            print(f"  Task {task['task_id']}: {task.get('error', 'Unknown error')}")
        if len(failed_tasks) > 10:
            print(f"  ... and {len(failed_tasks) - 10} more")


def generate_statistics(task_ids: List[str], output_file: str = None) -> Dict:
    """Generate statistics for a list of task IDs."""
    print(f"Generating statistics for {len(task_ids)} tasks...")
    print("This may take a while as we download and analyze patches...\n")
    
    results = []
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Analyzing task {task_id}...", end=" ", flush=True)
        try:
            result = validate_assets(task_id)
            results.append(result)
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                analysis = result.get('patch_analysis', {})
                difficulty = result.get('difficulty', 'unknown')
                print(f"✓ {difficulty} ({analysis.get('num_code_files', 0)} files, {analysis.get('total_changes', 0)} changes)")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            results.append({'task_id': task_id, 'error': str(e)})
    
    # Aggregate statistics
    stats = aggregate_statistics(results)
    
    # Print summary
    print_summary(stats, results)
    
    # Save results
    output_data = {
        'statistics': stats,
        'detailed_results': results
    }
    
    if output_file:
        output_path = Path(output_file)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = Path(OUTPUT_DIR) / "verified_jobs_statistics.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate statistics for verified ARVO tasks")
    parser.add_argument('--input', default='verified_jobs.json', 
                       help='Input JSON file with task IDs (default: verified_jobs.json)')
    parser.add_argument('--output', default=None,
                       help='Output JSON file for statistics (default: data/pipeline_results/verified_jobs_statistics.json)')
    
    args = parser.parse_args()
    
    # Read task IDs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    with open(input_path, 'r') as f:
        task_ids = json.load(f)
    
    if not isinstance(task_ids, list):
        print(f"Error: Expected a JSON array of task IDs, got {type(task_ids)}")
        sys.exit(1)
    
    # Generate statistics
    generate_statistics(task_ids, args.output)

