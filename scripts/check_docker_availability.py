"""
check_docker_availability.py

Checks which ARVO Docker images exist for HuggingFace task IDs.
Queries Docker Hub API to avoid pulling thousands of images.
"""

import requests
import json
from pathlib import Path
from typing import List, Dict
import time


def check_docker_image_exists_api(task_id: str) -> bool:
    """
    Check if a Docker image exists using Docker Hub API.
    This is faster than trying to pull the image.
    
    Args:
        task_id: The task ID
        
    Returns:
        True if image exists, False otherwise
    """
    # Docker Hub API endpoint for checking if a tag exists
    url = f"https://hub.docker.com/v2/repositories/n132/arvo/tags/{task_id}-vul"
    
    try:
        response = requests.get(url, timeout=5)
        # If status code is 200, the tag exists
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking {task_id}: {e}")
        return False


def check_docker_image_exists_local(task_id: str) -> bool:
    """
    Check if Docker image exists by trying to pull it.
    Slower but more reliable.
    
    Args:
        task_id: The task ID
        
    Returns:
        True if image can be pulled, False otherwise
    """
    try:
        import docker
        client = docker.from_env()
        image_name = f"n132/arvo:{task_id}-vul"
        
        # Try to pull the image
        client.images.pull(image_name)
        
        # If successful, remove it to save space
        client.images.remove(image_name, force=True)
        
        return True
    except Exception:
        return False


def check_all_tasks(task_list_file: str, output_file: str, method: str = "api") -> Dict:
    """
    Check Docker image availability for all tasks.
    
    Args:
        task_list_file: File with task IDs (one per line)
        output_file: Where to save results
        method: "api" (fast) or "docker" (slow but reliable)
        
    Returns:
        Dictionary with results
    """
    # Read task IDs
    with open(task_list_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Checking Docker availability for {len(task_ids)} tasks...")
    print(f"Method: {method}")
    print(f"This might take a while...")
    print("="*60)
    
    check_func = check_docker_image_exists_api if method == "api" else check_docker_image_exists_local
    
    results = {
        'total_tasks': len(task_ids),
        'images_available': 0,
        'images_missing': 0,
        'available_tasks': [],
        'missing_tasks': []
    }
    
    for i, task_id in enumerate(task_ids, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(task_ids)} ({i*100//len(task_ids)}%)")
        
        exists = check_func(task_id)
        
        if exists:
            results['images_available'] += 1
            results['available_tasks'].append(task_id)
            if i <= 20:  # Show first 20
                print(f"  ✓ {task_id}: Docker image exists")
        else:
            results['images_missing'] += 1
            results['missing_tasks'].append(task_id)
            if i <= 20:  # Show first 20
                print(f"  ✗ {task_id}: No Docker image")
        
        # Rate limiting for API calls
        if method == "api":
            time.sleep(0.1)  # Be nice to Docker Hub
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total tasks: {results['total_tasks']}")
    print(f"  Docker images available: {results['images_available']} ({results['images_available']*100//results['total_tasks']}%)")
    print(f"  Docker images missing: {results['images_missing']} ({results['images_missing']*100//results['total_tasks']}%)")
    print(f"\nResults saved to: {output_file}")
    
    # Save just the available task IDs to a separate file
    available_file = output_file.replace('.json', '_available.txt')
    with open(available_file, 'w') as f:
        for task_id in results['available_tasks']:
            f.write(task_id + '\n')
    print(f"Available task IDs saved to: {available_file}")
    
    return results


def check_sample_tasks(task_ids: List[str]) -> None:
    """
    Quickly check a sample of tasks to see the pattern.
    
    Args:
        task_ids: List of task IDs to check
    """
    print(f"Checking {len(task_ids)} sample tasks...")
    print("="*60)
    
    for task_id in task_ids:
        exists = check_docker_image_exists_api(task_id)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"  {task_id}: {status}")
        time.sleep(0.2)  # Rate limiting


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Docker image availability for ARVO tasks")
    parser.add_argument('--task-list', default='data/available_tasks.txt',
                       help='File containing task IDs')
    parser.add_argument('--output', default='data/docker_availability.json',
                       help='Output file for results')
    parser.add_argument('--method', choices=['api', 'docker'], default='api',
                       help='Method to check: api (fast) or docker (slow but pulls images)')
    parser.add_argument('--sample', action='store_true',
                       help='Just check first 20 tasks as a sample')
    parser.add_argument('--test', nargs='+',
                       help='Test specific task IDs (e.g., --test 10055 10096)')
    
    args = parser.parse_args()
    
    if args.test:
        # Test specific task IDs
        check_sample_tasks(args.test)
    elif args.sample:
        # Check first 20 as a sample
        with open(args.task_list, 'r') as f:
            sample_ids = [line.strip() for line in f if line.strip()][:20]
        check_sample_tasks(sample_ids)
    else:
        # Check all tasks
        check_all_tasks(args.task_list, args.output, args.method)

