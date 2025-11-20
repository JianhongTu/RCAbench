import requests 
import re

print('Querying HuggingFace API for available tasks...')
url = 'https://huggingface.co/api/datasets/sunblaze-ucb/cybergym/tree/main/data/arvo'

try:
    response = requests.get(url, timeout=30)
    data = response.json()

    task_ids = []
    for item in data:
        if item['type'] == 'directory':
            task_id = item['path'].split('/')[-1]
            if task_id.isdigit():
                task_ids.append(task_id)

    task_ids.sort()
    print(f'Found {len(task_ids)} tasks from HuggingFace ARVO (CyberGym)')
    print(f'First 20 IDs: {task_ids[:20]}')
    print(f'Last 20 IDs: {task_ids[-20:]}')

    with open('data/arvo_hf_task_ids.txt', 'w') as f:
        for task_id in task_ids:
            f.write(f'{task_id}\n')
    print(f'Saved task IDs to data/arvo_hf_task_ids.txt')

except Exception as e:
    print(f'Error querying HuggingFace API: {e}')
    exit(1)