#!/usr/bin/env python3
"""
Test script for prepare_task_assets function.
"""

import os
from pathlib import Path
from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
from rcabench import DEFAULT_HOST_IP, DEFAULT_HOST_PORT, DEFAULT_WORKSPACE_DIR, CODEBASE_SRC_NAME

def test_prepare_task_assets():
    print("Starting test_prepare_task_assets...")
    arvo_id = "10055"
    workspace_path = "./workspace"
    cache_path = "./tmp"
    
    try:
        result = prepare_task_assets(arvo_id, workspace_path, cache_path, host_ip=DEFAULT_HOST_IP, host_port=DEFAULT_HOST_PORT)
        
        # Check that expected paths exist
        assert os.path.exists(result['diff_path']), f"Diff file not found: {result['diff_path']}"
        assert os.path.exists(result['error_path']), f"Error file not found: {result['error_path']}"
        assert os.path.exists(result['codebase_path']), f"Codebase path not found: {result['codebase_path']}"
        
        # Check submission scripts exist
        submit_loc_path = os.path.join(DEFAULT_WORKSPACE_DIR, "submit_loc.sh")
        submit_patch_path = os.path.join(DEFAULT_WORKSPACE_DIR, "submit_patch.sh")
        assert os.path.exists(submit_loc_path), f"Submit loc script not found: {submit_loc_path}"
        assert os.path.exists(submit_patch_path), f"Submit patch script not found: {submit_patch_path}"
        
        # Check that codebase directory has some content (extracted files)
        codebase_dir = Path(result['codebase_path'])
        assert codebase_dir.is_dir(), f"Codebase path is not a directory: {codebase_dir}"
        files_in_codebase = list(codebase_dir.rglob('*'))
        assert len(files_in_codebase) > 0, f"No files found in codebase directory: {codebase_dir}"
        
        # Check diff file has content and looks like a diff (general check, not Git-specific)
        with open(result['diff_path'], 'r') as f:
            diff_content = f.read()
            assert len(diff_content) > 0, f"Diff file is empty: {result['diff_path']}"
            assert 'diff ' in diff_content, f"Diff file doesn't look like a proper diff: {result['diff_path']}"  # Changed from 'diff --git' to 'diff '
        
        # Check error file has content
        with open(result['error_path'], 'r') as f:
            error_content = f.read()
            assert len(error_content) > 0, f"Error file is empty: {result['error_path']}"
        
        print("All file existence checks passed!")
        print("Result:", result)
        
        # Test cleanup
        print("Testing cleanup...")
        cleanup_task_assets(result)
        
        # Check that files have been removed
        assert not os.path.exists(result['diff_path']), f"Diff file still exists after cleanup: {result['diff_path']}"
        assert not os.path.exists(result['error_path']), f"Error file still exists after cleanup: {result['error_path']}"
        assert not os.path.exists(submit_loc_path), f"Submit loc script still exists after cleanup: {submit_loc_path}"
        assert not os.path.exists(submit_patch_path), f"Submit patch script still exists after cleanup: {submit_patch_path}"
        
        # Check shared directory is removed
        shared_dir = os.path.join(workspace_path, "shared")
        assert not os.path.exists(shared_dir), f"Shared directory still exists after cleanup: {shared_dir}"
        
        # Check src-vul directory is removed
        src_vul_dir = os.path.join(workspace_path, CODEBASE_SRC_NAME)
        assert not os.path.exists(src_vul_dir), f"{CODEBASE_SRC_NAME} directory still exists after cleanup: {src_vul_dir}"
        
        print("Cleanup test passed!")
        
    except Exception as e:
        print("Test failed with error:", e)
        raise

if __name__ == "__main__":
    test_prepare_task_assets()
