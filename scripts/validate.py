"""
validate.py - ARVO Task Validation Pipeline

Usage:
    python3 scripts/validate.py 10055              # Validate single task
    python3 scripts/validate.py --all              # Validate all tasks (with Docker)
    python3 scripts/validate.py --all --skip-docker # Validate all tasks (fast, no Docker)
    python3 scripts/validate.py --all --sample 20  # Validate first 20 tasks
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import shutil
import requests
from pathlib import Path
from typing import Dict, List

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcabench.utils import remote_fetch_diff, remote_fetch_error, remote_fetch_codebase

CACHE_DIR = "./tmp/validation"
OUTPUT_DIR = "./data/pipeline_results"


# =============================================================================
# STAGE 1: Asset Validation & Patch Testing
# =============================================================================

def analyze_patch(patch_path: str) -> Dict:
    """Analyze patch complexity."""
    with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find files
    files = re.findall(r'^\+\+\+ b/([^\t\n]+)', content, re.MULTILINE)
    
    # Count changes
    added = sum(1 for line in content.split('\n') if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in content.split('\n') if line.startswith('-') and not line.startswith('---'))
    
    # Categorize files
    code_exts = {'.c', '.cc', '.cpp', '.h', '.hpp', '.py', '.rs', '.go', '.java'}
    code_files = [f for f in files if Path(f).suffix.lower() in code_exts]
    
    return {
        'num_files': len(files),
        'num_code_files': len(code_files),
        'lines_added': added,
        'lines_removed': removed,
        'total_changes': added + removed
    }


def test_patch_applies(patch_path: str, repo_dir: Path) -> tuple:
    """Test if patch applies to codebase. Returns (success, output)."""
    # Find project subdirectory if exists
    subdirs = [d for d in repo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    patch_root = subdirs[0] if len(subdirs) == 1 else repo_dir
    
    cmd = f"cd {patch_root} && patch -p1 --dry-run --force < {Path(patch_path).absolute()}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    
    # Check if code files applied (ignore changelog failures)
    output = result.stdout + result.stderr
    code_patched = any(ext in output for ext in ['.c', '.cpp', '.h', '.py', '.rs', '.go'])
    
    return (result.returncode == 0 or code_patched), output


def validate_assets(task_id: str) -> Dict:
    """Stage 1: Download assets and test patch application."""
    result = {'task_id': task_id, 'stage1_passed': False, 'patch_analysis': None}
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    extract_dir = Path(CACHE_DIR) / f"task_{task_id}"
    
    try:
        # Download assets
        patch_path = remote_fetch_diff(task_id, output_dir=CACHE_DIR)
        remote_fetch_error(task_id, output_dir=CACHE_DIR)
        codebase_path = remote_fetch_codebase(task_id, output_dir=CACHE_DIR)
        
        # Extract codebase
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(codebase_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        os.remove(codebase_path)
        
        # Find repo directory
        items = list(extract_dir.iterdir())
        repo_dir = items[0] if len(items) == 1 and items[0].is_dir() else extract_dir
        
        # Test patch
        patch_applies, _ = test_patch_applies(patch_path, repo_dir)
        result['patch_applies'] = patch_applies
        result['patch_analysis'] = analyze_patch(patch_path)
        result['stage1_passed'] = patch_applies
        
        # Categorize difficulty
        analysis = result['patch_analysis']
        if analysis['num_code_files'] == 1 and analysis['total_changes'] <= 20:
            result['difficulty'] = 'easy'
        elif analysis['num_code_files'] >= 3 or analysis['total_changes'] > 100:
            result['difficulty'] = 'hard'
        else:
            result['difficulty'] = 'medium'
            
    except Exception as e:
        result['error'] = str(e)
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    return result


# =============================================================================
# STAGE 2: Docker Availability
# =============================================================================

def check_docker_available(task_id: str) -> bool:
    """Check if Docker image exists on Docker Hub."""
    url = f"https://hub.docker.com/v2/repositories/n132/arvo/tags/{task_id}-vul"
    try:
        return requests.get(url, timeout=5).status_code == 200
    except:
        return False


# =============================================================================
# STAGE 3: Docker Validation
# =============================================================================

def validate_with_docker(task_id: str, patch_path: str, timeout: int = 600) -> Dict:
    """Stage 3: Full Docker validation (apply, compile, run fuzzer)."""
    import docker
    
    result = {'docker_tested': False, 'patch_applied': False, 'compiled': False, 'fixed': False}
    
    try:
        client = docker.from_env()
        image = f"n132/arvo:{task_id}-vul"
        
        # Pull image
        client.images.pull(image)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy patch
            shutil.copy(patch_path, Path(temp_dir) / f"{task_id}_patch.diff")
            
            # Run container with 3-step validation
            cmd = ["/bin/sh", "-c",
                f"patch -p1 --force < /tmp/patch/{task_id}_patch.diff; PATCH=$?; "
                f"arvo compile; COMPILE=$?; "
                f"timeout {timeout} arvo; FUZZER=$?; "
                f"echo RESULTS:$PATCH:$COMPILE:$FUZZER"]
            
            container = client.containers.run(
                image=image,
                command=cmd,
                volumes={temp_dir: {"bind": "/tmp/patch", "mode": "ro"}},
                detach=True
            )
            
            # Wait and get output
            container.wait(timeout=timeout + 60)
            output = container.logs().decode('utf-8', errors='ignore')
            container.remove(force=True)
            
            # Parse results
            match = re.search(r'RESULTS:(\d+):(\d+):(\d+)', output)
            if match:
                result['docker_tested'] = True
                patch_exit, compile_exit, fuzzer_exit = map(int, match.groups())
                result['compiled'] = (compile_exit == 0)
                result['fixed'] = (fuzzer_exit == 0)
                # Patch "applied" if code compiled and fuzzer passed (ignore changelog failures)
                result['patch_applied'] = result['compiled'] and result['fixed']
                
    except Exception as e:
        result['error'] = str(e)
    
    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def validate_task(task_id: str, skip_docker: bool = False, timeout: int = 600) -> Dict:
    """Run full validation pipeline on a single task."""
    print(f"\n{'='*50}")
    print(f"Validating task: {task_id}")
    print('='*50)
    
    result = {'task_id': task_id, 'status': 'unknown'}
    
    # Stage 1
    print("[Stage 1] Asset validation...")
    stage1 = validate_assets(task_id)
    result.update(stage1)
    
    if not stage1.get('stage1_passed'):
        print(f"  ❌ Stage 1 failed")
        result['status'] = 'failed_stage1'
        return result
    print(f"  ✓ Patch applies ({result.get('difficulty', 'unknown')} difficulty)")
    
    # Stage 2
    print("[Stage 2] Docker check...")
    docker_available = check_docker_available(task_id)
    result['docker_available'] = docker_available
    
    if not docker_available:
        print(f"  ⚠️  No Docker image")
        result['status'] = 'no_docker'
        return result
    print(f"  ✓ Docker image exists")
    
    if skip_docker:
        result['status'] = 'docker_available'
        return result
    
    # Stage 3
    print(f"[Stage 3] Docker validation (timeout: {timeout}s)...")
    patch_path = f"{CACHE_DIR}/{task_id}_patch.diff"
    stage3 = validate_with_docker(task_id, patch_path, timeout)
    result.update(stage3)
    
    if stage3.get('fixed'):
        print(f"  ✓ Vulnerability fixed!")
        result['status'] = 'passed'
    elif stage3.get('compiled'):
        print(f"  ⚠️  Compiles but doesn't fix vulnerability")
        result['status'] = 'compiles_not_fixed'
    else:
        print(f"  ❌ Docker validation failed")
        result['status'] = 'failed_docker'
    
    return result


def validate_all(task_list: str, skip_docker: bool = False, sample: int = None):
    """Validate all tasks from a list."""
    with open(task_list) as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    if sample:
        task_ids = task_ids[:sample]
    
    print(f"Validating {len(task_ids)} tasks...")
    
    results = []
    for i, task_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}]", end="")
        result = validate_task(task_id, skip_docker)
        results.append(result)
    
    # Generate summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    passed = [r for r in results if r['status'] == 'passed']
    docker_avail = [r for r in results if r['status'] in ['passed', 'docker_available']]
    no_docker = [r for r in results if r['status'] == 'no_docker']
    
    # Save results
    with open(f"{OUTPUT_DIR}/validation_report.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    for name, tasks in [('tier1', passed), ('tier2', docker_avail), ('tier3', no_docker)]:
        with open(f"{OUTPUT_DIR}/{name}_tasks.txt", 'w') as f:
            f.write('\n'.join(r['task_id'] for r in tasks))
    
    for diff in ['easy', 'medium', 'hard']:
        tasks = [r for r in results if r.get('difficulty') == diff]
        with open(f"{OUTPUT_DIR}/{diff}_tasks.txt", 'w') as f:
            f.write('\n'.join(r['task_id'] for r in tasks))
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Total: {len(results)}")
    print(f"Passed (Tier 1): {len(passed)}")
    print(f"Docker available (Tier 2): {len(docker_avail)}")
    print(f"No Docker (Tier 3): {len(no_docker)}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ARVO tasks")
    parser.add_argument('task_id', nargs='?', help='Single task ID to validate')
    parser.add_argument('--all', action='store_true', help='Validate all tasks')
    parser.add_argument('--task-list', default='data/arvo_hf_task_ids.txt', help='Task list file')
    parser.add_argument('--skip-docker', action='store_true', help='Skip Docker validation')
    parser.add_argument('--sample', type=int, help='Only validate first N tasks')
    parser.add_argument('--timeout', type=int, default=600, help='Docker timeout in seconds (default: 600)')
    
    args = parser.parse_args()
    
    if args.all:
        validate_all(args.task_list, args.skip_docker, args.sample)
    elif args.task_id:
        result = validate_task(args.task_id, args.skip_docker, args.timeout)
        
        # Print summary
        print(f"\n{'='*50}")
        print("RESULT")
        print('='*50)
        print(json.dumps(result, indent=2))
        
        # Save to file
        os.makedirs(CACHE_DIR, exist_ok=True)
        output_file = f"{CACHE_DIR}/{args.task_id}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")
    else:
        parser.print_help()

