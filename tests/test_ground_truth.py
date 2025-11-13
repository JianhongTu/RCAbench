#!/usr/bin/env python3

import sys
import json
from rcabench.server.eval_utils import get_ground_truth

if __name__ == "__main__":
    print("Starting test")
    arvo_id = "10055"
    try:
        gt = get_ground_truth(arvo_id)
        print(f"Got gt: {len(gt)} items")
        if gt:
            task_id = gt[0].task_id
            files = sorted(set(loc.file for loc in gt))
            lines = set()
            for loc in gt:
                lines.update(range(loc.new_span.start, loc.new_span.end + 1))
            lines = sorted(lines)
            functions = sorted(set(loc.function for loc in gt if loc.function))
        else:
            task_id = f"arvo:{arvo_id}"
            files = []
            lines = []
            functions = []
        print(f"Ground truth for {arvo_id}:")
        print(f"Task ID: {task_id}")
        for loc in gt:
            print(f"  File: {loc.file}")
            print(f"    Old span: {loc.old_span.start}-{loc.old_span.end}")
            print(f"    New span: {loc.new_span.start}-{loc.new_span.end}")
            print(f"    Function: {loc.function or 'N/A'}")
        print(f"Files: {files}")
        print(f"Functions: {functions}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Create fake submission
    fake_submission = []
    for loc in gt:
        foke_loc = loc.to_dict()
        fake_submission.append(foke_loc)
    
    with open("./workspace/shared/loc.json", "w") as f:
        json.dump(fake_submission, f, indent=2)
    
    print(f"Fake submission written to ./workspace/shared/loc.json")

