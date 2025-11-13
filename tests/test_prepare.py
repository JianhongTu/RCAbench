#!/usr/bin/env python3
"""
Test script for prepare_task_assets function.
"""

from rcabench.task.gen_task import prepare_task_asssets

def test_prepare_task_assets():
    arvo_id = "10055"
    workspace_path = "./workspace"
    cache_path = "./tmp"
    
    try:
        result = prepare_task_asssets(arvo_id, workspace_path, cache_path)
        print("Test passed!")
        print("Result:", result)
    except Exception as e:
        print("Test failed with error:", e)

if __name__ == "__main__":
    test_prepare_task_assets()
