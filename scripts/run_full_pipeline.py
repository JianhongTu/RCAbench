"""
run_full_pipeline.py

Master pipeline that runs all validation stages on ARVO tasks.

Stages:
1. Asset validation (download, patch application, analysis)
2. Docker availability check
3. Full Docker validation (optional, slow)

Outputs:
- Comprehensive validation report (JSON)
- Filtered task lists by quality tier
- Summary statistics
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm
import time

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import our validation modules
from validate_tasks import validate_single_task
from check_docker_availability import check_docker_image_exists_api
from docker_validator import test_patch_with_docker


def categorize_difficulty(patch_analysis: Dict) -> str:
    """
    Categorize task difficulty based on patch complexity.
    
    Args:
        patch_analysis: Analysis from analyze_patch
        
    Returns:
        'easy', 'medium', or 'hard'
    """
    if not patch_analysis:
        return 'unknown'
    
    num_code_files = patch_analysis.get('num_code_files', 0)
    total_changes = patch_analysis.get('total_changes', 0)
    
    # Easy: Single file, small change
    if num_code_files == 1 and total_changes <= 20:
        return 'easy'
    
    # Hard: Multiple files or large change
    elif num_code_files >= 3 or total_changes > 100:
        return 'hard'
    
    # Medium: Everything else
    else:
        return 'medium'


def run_pipeline(
    task_ids: List[str],
    cache_dir: str = "./tmp/validation",
    skip_docker: bool = False,
    docker_sample_size: int = None
) -> Dict:
    """
    Run the full validation pipeline on a list of tasks.
    
    Args:
        task_ids: List of task IDs to validate
        cache_dir: Directory for temporary files
        skip_docker: If True, skip Stage 3 (Docker validation)
        docker_sample_size: Only run Docker validation on first N tasks (for testing)
        
    Returns:
        Complete validation report
    """
    
    print("="*80)
    print("ARVO TASK VALIDATION PIPELINE")
    print("="*80)
    print(f"Tasks to validate: {len(task_ids)}")
    print(f"Cache directory: {cache_dir}")
    print(f"Skip Docker validation: {skip_docker}")
    if docker_sample_size:
        print(f"Docker validation sample size: {docker_sample_size}")
    print("="*80)
    
    results = []
    
    for i, task_id in enumerate(tqdm(task_ids, desc="Validating tasks")):
        print(f"\n[{i+1}/{len(task_ids)}] Task {task_id}")
        print("-"*60)
        
        task_result = {
            'task_id': task_id,
            'stage1_passed': False,
            'stage2_passed': False,
            'stage3_passed': False,
            'difficulty': 'unknown',
            'stage1_result': None,
            'stage2_result': None,
            'stage3_result': None
        }
        
        # Stage 1: Asset validation and patch testing
        print("  [Stage 1] Asset validation...")
        try:
            stage1 = validate_single_task(task_id, cache_dir)
            task_result['stage1_result'] = stage1
            
            # Check if Stage 1 passed
            if (stage1.get('patch_available') and 
                stage1.get('error_available') and 
                stage1.get('codebase_available') and
                stage1.get('patch_applies')):
                
                task_result['stage1_passed'] = True
                
                # Categorize difficulty
                if stage1.get('patch_analysis'):
                    task_result['difficulty'] = categorize_difficulty(stage1['patch_analysis'])
                
                print(f"    ✓ Stage 1 PASSED (Difficulty: {task_result['difficulty']})")
            else:
                print(f"    ✗ Stage 1 FAILED")
                results.append(task_result)
                continue  # Skip remaining stages if Stage 1 fails
                
        except Exception as e:
            print(f"    ✗ Stage 1 ERROR: {e}")
            task_result['stage1_result'] = {'error': str(e)}
            results.append(task_result)
            continue
        
        # Stage 2: Docker availability check
        print("  [Stage 2] Docker image check...")
        try:
            docker_exists = check_docker_image_exists_api(task_id)
            task_result['stage2_result'] = {'docker_image_exists': docker_exists}
            task_result['stage2_passed'] = docker_exists
            
            if docker_exists:
                print(f"    ✓ Stage 2 PASSED (Docker image available)")
            else:
                print(f"    ⚠️  Stage 2: No Docker image (can't run Stage 3)")
                
        except Exception as e:
            print(f"    ✗ Stage 2 ERROR: {e}")
            task_result['stage2_result'] = {'error': str(e)}
        
        # Stage 3: Full Docker validation (optional and slow)
        if not skip_docker and task_result['stage2_passed']:
            # Only run on sample if specified
            if docker_sample_size is None or i < docker_sample_size:
                print("  [Stage 3] Docker validation (compile + fuzzer)...")
                try:
                    patch_path = f"{cache_dir}/{task_id}_patch.diff"
                    stage3 = test_patch_with_docker(task_id, patch_path)
                    task_result['stage3_result'] = stage3
                    
                    # Check if all steps passed
                    if (stage3.get('patch_applied') and 
                        stage3.get('code_compiled') and
                        stage3.get('vulnerability_fixed')):
                        task_result['stage3_passed'] = True
                        print(f"    ✓ Stage 3 PASSED (Patch works!)")
                    else:
                        print(f"    ✗ Stage 3 FAILED (See details in report)")
                        
                except Exception as e:
                    print(f"    ✗ Stage 3 ERROR: {e}")
                    task_result['stage3_result'] = {'error': str(e)}
            else:
                print(f"  [Stage 3] Skipped (beyond sample size)")
        
        results.append(task_result)
        
        # Brief pause to avoid rate limiting
        time.sleep(0.2)
    
    return generate_report(results, task_ids)


def generate_report(results: List[Dict], all_task_ids: List[str]) -> Dict:
    """
    Generate comprehensive report from validation results.
    
    Args:
        results: List of validation results
        all_task_ids: Original list of all task IDs
        
    Returns:
        Report dictionary
    """
    
    # Count results by stage
    stage1_pass = [r for r in results if r['stage1_passed']]
    stage2_pass = [r for r in results if r['stage2_passed']]
    stage3_pass = [r for r in results if r['stage3_passed']]
    
    # Categorize by difficulty
    easy = [r for r in stage1_pass if r['difficulty'] == 'easy']
    medium = [r for r in stage1_pass if r['difficulty'] == 'medium']
    hard = [r for r in stage1_pass if r['difficulty'] == 'hard']
    
    # Quality tiers
    tier1 = stage3_pass  # Full validation passed
    tier2 = [r for r in stage2_pass if not r['stage3_passed']]  # Docker available but not tested/failed
    tier3 = [r for r in stage1_pass if not r['stage2_passed']]  # No Docker
    
    report = {
        'summary': {
            'total_tasks': len(all_task_ids),
            'tasks_validated': len(results),
            'stage1_passed': len(stage1_pass),
            'stage2_passed': len(stage2_pass),
            'stage3_passed': len(stage3_pass),
            'by_difficulty': {
                'easy': len(easy),
                'medium': len(medium),
                'hard': len(hard)
            },
            'by_quality_tier': {
                'tier1_gold': len(tier1),
                'tier2_silver': len(tier2),
                'tier3_bronze': len(tier3)
            }
        },
        'tier1_tasks': [r['task_id'] for r in tier1],
        'tier2_tasks': [r['task_id'] for r in tier2],
        'tier3_tasks': [r['task_id'] for r in tier3],
        'easy_tasks': [r['task_id'] for r in easy],
        'medium_tasks': [r['task_id'] for r in medium],
        'hard_tasks': [r['task_id'] for r in hard],
        'detailed_results': results
    }
    
    return report


def save_report(report: Dict, output_dir: Path):
    """
    Save report and filtered task lists.
    
    Args:
        report: Validation report
        output_dir: Where to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full report
    report_file = output_dir / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Full report saved to: {report_file}")
    
    # Save filtered task lists
    for tier in ['tier1', 'tier2', 'tier3']:
        tier_file = output_dir / f"{tier}_tasks.txt"
        tasks = report[f'{tier}_tasks']
        with open(tier_file, 'w') as f:
            for task_id in tasks:
                f.write(task_id + '\n')
        print(f"✓ {tier.upper()} tasks ({len(tasks)}) saved to: {tier_file}")
    
    # Save by difficulty
    for diff in ['easy', 'medium', 'hard']:
        diff_file = output_dir / f"{diff}_tasks.txt"
        tasks = report[f'{diff}_tasks']
        with open(diff_file, 'w') as f:
            for task_id in tasks:
                f.write(task_id + '\n')
        print(f"✓ {diff.upper()} tasks ({len(tasks)}) saved to: {diff_file}")
    
    print(f"\n✓ All outputs saved to: {output_dir}")


def print_summary(report: Dict):
    """Print a nice summary of results."""
    
    summary = report['summary']
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total tasks:        {summary['total_tasks']}")
    print(f"Tasks validated:    {summary['tasks_validated']}")
    print()
    print("Stage Results:")
    print(f"  Stage 1 (Assets):   {summary['stage1_passed']} passed ({summary['stage1_passed']*100//summary['tasks_validated']}%)")
    print(f"  Stage 2 (Docker):   {summary['stage2_passed']} passed ({summary['stage2_passed']*100//summary['tasks_validated'] if summary['tasks_validated'] > 0 else 0}%)")
    print(f"  Stage 3 (Full):     {summary['stage3_passed']} passed")
    print()
    print("Difficulty Distribution:")
    print(f"  Easy:     {summary['by_difficulty']['easy']} tasks")
    print(f"  Medium:   {summary['by_difficulty']['medium']} tasks")
    print(f"  Hard:     {summary['by_difficulty']['hard']} tasks")
    print()
    print("Quality Tiers:")
    print(f"  🥇 Tier 1 (Gold):   {summary['by_quality_tier']['tier1_gold']} - Fully validated with Docker")
    print(f"  🥈 Tier 2 (Silver): {summary['by_quality_tier']['tier2_silver']} - Docker available")
    print(f"  🥉 Tier 3 (Bronze): {summary['by_quality_tier']['tier3_bronze']} - Basic validation only")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full ARVO task validation pipeline")
    parser.add_argument('--task-list', default='data/available_tasks.txt',
                       help='File with task IDs (one per line)')
    parser.add_argument('--output-dir', default='data/pipeline_results',
                       help='Directory for output files')
    parser.add_argument('--cache-dir', default='./tmp/validation',
                       help='Directory for temporary files')
    parser.add_argument('--skip-docker', action='store_true',
                       help='Skip Stage 3 (Docker validation) - much faster')
    parser.add_argument('--docker-sample', type=int,
                       help='Only run Docker validation on first N tasks')
    parser.add_argument('--sample', type=int,
                       help='Only validate first N tasks (for testing)')
    
    args = parser.parse_args()
    
    # Read task IDs
    with open(args.task_list, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    # Sample if requested
    if args.sample:
        task_ids = task_ids[:args.sample]
        print(f"Running on sample of {len(task_ids)} tasks")
    
    # Run pipeline
    report = run_pipeline(
        task_ids,
        cache_dir=args.cache_dir,
        skip_docker=args.skip_docker,
        docker_sample_size=args.docker_sample
    )
    
    # Print summary
    print_summary(report)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_report(report, output_dir)

