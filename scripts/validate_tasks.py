"""
validate_tasks.py

Validates ARVO tasks from the HuggingFace cybergym dataset.
For each task, checks if assets are downloadable and patches are valid.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path so we can import rcabench and scripts
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcabench.utils import remote_fetch_diff, remote_fetch_error, remote_fetch_codebase
from analyze_patch import analyze_patch


def validate_single_task(task_id: str, cache_dir: str = "./tmp/validation") -> Dict:
    """
    Validate a single ARVO task.
    
    This function checks:
    1. Can we download the patch file?
    2. Can we download the error report?
    3. Can we download the vulnerable codebase?
    4. What does the patch look like? (using analyze_patch)
    
    Args:
        task_id: The ARVO task ID (e.g., "10055")
        cache_dir: Where to temporarily store downloaded files
        
    Returns:
        Dictionary with validation results
    """
    
    # Initialize result structure
    result = {
        'task_id': task_id,
        'patch_available': False,
        'error_available': False,
        'codebase_available': False,
        'patch_applies': False,
        'patch_apply_output': None,
        'patch_analysis': None,
        'validation_status': 'unknown',
        'error_message': None
    }
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Step 1: Try to download the patch file
        print(f"  Downloading patch for task {task_id}...")
        patch_path = remote_fetch_diff(task_id, output_dir=cache_dir)
        result['patch_available'] = True
        print(f"    ✓ Patch downloaded")
        
        # Step 2: Try to download the error report
        print(f"  Downloading error report for task {task_id}...")
        error_path = remote_fetch_error(task_id, output_dir=cache_dir)
        result['error_available'] = True
        print(f"    ✓ Error report downloaded")
        
        # Step 3: Download the codebase and test if patch applies
        print(f"  Downloading codebase for task {task_id}...")
        try:
            import tarfile
            import subprocess
            
            # Download the codebase tarball
            codebase_path = remote_fetch_codebase(task_id, output_dir=cache_dir)
            result['codebase_available'] = True
            print(f"    ✓ Codebase downloaded")
            
            # Extract the tarball
            extract_dir = Path(cache_dir) / f"task_{task_id}_extract"
            extract_dir.mkdir(exist_ok=True)
            
            print(f"  Extracting codebase...")
            with tarfile.open(codebase_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            print(f"    ✓ Codebase extracted")
            
            # Find the actual repository directory
            # It's usually "repo-vul" but let's check what actually got extracted
            extracted_items = list(extract_dir.iterdir())
            print(f"    Extracted items: {[item.name for item in extracted_items]}")
            
            # Try to apply the patch
            print(f"  Testing if patch applies...")
            
            # Look for repo-vul directory
            repo_dir = extract_dir / "repo-vul"
            if not repo_dir.exists():
                # Maybe it extracted directly to extract_dir?
                # Or maybe it has a different name?
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    repo_dir = extracted_items[0]
                    print(f"    Using directory: {repo_dir.name}")
                else:
                    repo_dir = extract_dir
                    print(f"    Using extract_dir directly")
            
            if repo_dir.exists() and repo_dir.is_dir():
                # Check if there's a project subdirectory (like graphicsmagick/)
                subdirs = [d for d in repo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if len(subdirs) == 1:
                    # Use the single subdirectory as the patch root
                    patch_root = subdirs[0]
                    print(f"    Using project directory: {patch_root.name}")
                else:
                    patch_root = repo_dir
                
                # Run: patch -p1 --dry-run < patch_file
                # --dry-run tests without actually modifying files
                # We use --force to avoid interactive prompts (which cause hangs)
                patch_cmd = f"cd {patch_root} && patch -p1 --dry-run --force < {Path(patch_path).absolute()}"
                patch_result = subprocess.run(
                    patch_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse the output to see which files succeeded/failed
                output = patch_result.stdout + patch_result.stderr
                
                # Check if any code files were successfully patched
                # Look for lines like "patching file 'path/file.c'" without subsequent "failed" messages
                code_patched = False
                for line in output.split('\n'):
                    if 'patching file' in line.lower():
                        # Extract the filename
                        if any(ext in line for ext in ['.c', '.cpp', '.cc', '.h', '.hpp', '.py', '.rs', '.go']):
                            # This is a code file being patched
                            code_patched = True
                            break
                
                if patch_result.returncode == 0:
                    result['patch_applies'] = True
                    result['patch_apply_output'] = "Patch applies cleanly (all files)"
                    print(f"    ✓ Patch applies cleanly!")
                elif code_patched and 'changelog' in output.lower():
                    # Code files applied but changelog failed - this is acceptable
                    result['patch_applies'] = True
                    result['patch_apply_output'] = "Code files apply cleanly (changelog/docs failed)"
                    print(f"    ✓ Code files apply (changelog failed - OK)")
                else:
                    result['patch_applies'] = False
                    result['patch_apply_output'] = output[:500]  # First 500 chars
                    print(f"    ✗ Patch does NOT apply")
                    print(f"      Output: {output[:200]}")
            else:
                result['patch_applies'] = False
                result['patch_apply_output'] = "repo-vul directory not found in tarball"
                print(f"    ✗ Could not find repo-vul directory")
            
            # Clean up - remove extracted files and tarball
            import shutil
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            if os.path.exists(codebase_path):
                os.remove(codebase_path)
            print(f"    ✓ Cleaned up temporary files")
            
        except Exception as e:
            result['codebase_available'] = False
            result['patch_applies'] = False
            result['patch_apply_output'] = str(e)
            print(f"    ✗ Error testing codebase: {e}")
        
        # Step 4: Analyze the patch to understand its complexity
        print(f"  Analyzing patch structure...")
        patch_analysis = analyze_patch(patch_path)
        result['patch_analysis'] = {
            'num_files': patch_analysis['num_files'],
            'num_code_files': patch_analysis['num_code_files'],
            'lines_added': patch_analysis['lines_added'],
            'lines_removed': patch_analysis['lines_removed'],
            'total_changes': patch_analysis['total_changes'],
            'file_categories': patch_analysis['category_counts']
        }
        print(f"    ✓ Patch analyzed: {patch_analysis['num_code_files']} code files, {patch_analysis['total_changes']} line changes")
        
        # Determine overall status
        if result['patch_available'] and result['error_available'] and result['codebase_available'] and result['patch_applies']:
            result['validation_status'] = 'valid'  # Everything works!
        elif result['patch_available'] and result['error_available'] and result['codebase_available']:
            result['validation_status'] = 'patch_fails'  # Assets exist but patch doesn't apply
        elif result['patch_available'] and result['error_available']:
            result['validation_status'] = 'partial'  # Has patch and error, but no codebase
        else:
            result['validation_status'] = 'invalid'  # Missing critical assets
            
    except Exception as e:
        result['validation_status'] = 'error'
        result['error_message'] = str(e)
        print(f"    ✗ Error validating task {task_id}: {e}")
    
    return result


def validate_all_tasks(task_list_file: str, output_file: str, cache_dir: str = "./tmp/validation") -> List[Dict]:
    """
    Validate all tasks from a list file.
    
    Args:
        task_list_file: Path to file containing task IDs (one per line)
        output_file: Where to save the validation results (JSON)
        cache_dir: Where to store temporary downloaded files
        
    Returns:
        List of validation results for all tasks
    """
    
    # Read task IDs from file
    with open(task_list_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(task_ids)} tasks to validate")
    print("="*60)
    
    results = []
    
    # Validate each task
    for i, task_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}] Validating task {task_id}")
        result = validate_single_task(task_id, cache_dir)
        results.append(result)
        
        # Save results incrementally (so we don't lose progress if it crashes)
        if i % 10 == 0:  # Save every 10 tasks
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved progress: {i}/{len(task_ids)} tasks validated")
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Validation complete! Results saved to {output_file}")
    
    # Print summary
    valid_count = sum(1 for r in results if r['validation_status'] == 'valid')
    patch_fails_count = sum(1 for r in results if r['validation_status'] == 'patch_fails')
    partial_count = sum(1 for r in results if r['validation_status'] == 'partial')
    invalid_count = sum(1 for r in results if r['validation_status'] == 'invalid')
    error_count = sum(1 for r in results if r['validation_status'] == 'error')
    
    print(f"\nSummary:")
    print(f"  ✅ Valid (patch applies cleanly): {valid_count}")
    print(f"  ⚠️  Patch fails to apply: {patch_fails_count}")
    print(f"  ⚠️  Partial (missing codebase): {partial_count}")
    print(f"  ❌ Invalid (missing assets): {invalid_count}")
    print(f"  ❌ Errors: {error_count}")
    
    return results


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ARVO tasks from HuggingFace")
    parser.add_argument('--task-list', default='data/available_tasks.txt',
                       help='File containing task IDs to validate')
    parser.add_argument('--output', default='data/validation_results.json',
                       help='Where to save validation results')
    parser.add_argument('--cache-dir', default='./tmp/validation',
                       help='Directory for temporary files')
    parser.add_argument('--single', type=str,
                       help='Validate a single task ID instead of all tasks')
    
    args = parser.parse_args()
    
    if args.single:
        # Validate just one task
        print(f"Validating single task: {args.single}")
        result = validate_single_task(args.single, args.cache_dir)
        print("\nResult:")
        print(json.dumps(result, indent=2))
    else:
        # Validate all tasks
        validate_all_tasks(args.task_list, args.output, args.cache_dir)
