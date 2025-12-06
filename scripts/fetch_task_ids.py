"""
fetch_task_ids.py - Fetch available ARVO task IDs from HuggingFace

Usage:
    python3 scripts/fetch_task_ids.py
    
Output:
    data/arvo_hf_task_ids.txt
"""

import requests
from pathlib import Path

def fetch_task_ids():
    """Fetch task IDs from HuggingFace cybergym dataset."""
    
    print("Fetching task IDs from HuggingFace...")
    url = "https://huggingface.co/api/datasets/sunblaze-ucb/cybergym/tree/main/data/arvo"
    
    response = requests.get(url, timeout=30)
    data = response.json()
    
    # Extract directory names (task IDs)
    task_ids = [item['path'].split('/')[-1] for item in data if item['type'] == 'directory']
    task_ids = [tid for tid in task_ids if tid.isdigit()]
    task_ids.sort()
    
    # Save to file
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "arvo_hf_task_ids.txt"
    
    with open(output_file, 'w') as f:
        for tid in task_ids:
            f.write(tid + '\n')
    
    print(f"Found {len(task_ids)} tasks")
    print(f"Saved to: {output_file}")
    
    # Warning if exactly 1000 (might be paginated)
    if len(task_ids) == 1000:
        print("⚠️  WARNING: Exactly 1000 results - API might be truncated!")

if __name__ == "__main__":
    fetch_task_ids()

