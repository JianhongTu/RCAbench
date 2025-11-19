import os
import json
import subprocess
import shutil
from pathlib import Path
import tarfile

from rcabench import CODEBASE_FILE_NAME, DEFAULT_WORKSPACE_DIR, DEFAULT_CACHE_DIR, CODEBASE_SRC_NAME
from rcabench.utils import remote_fetch_diff, remote_fetch_error, remote_fetch_codebase


def prepare_task_assets(
        arvo_id: str,
        workspace_path: str = DEFAULT_WORKSPACE_DIR,
        cache_path: str = DEFAULT_CACHE_DIR,
        host_ip: str = "",
        host_port: int = 0
        ) -> dict:
    """
    Prepares the task assets by fetching the diff, error, and codebase files
    for the given arvo_id from the remote repository.

    Args:
        arvo_id (str): The ARVO task identifier.
        workspace_path (str): Path to the workspace directory containing repo-vul.tar.gz.
        cache_path (str): Path to the cache directory for fetched files.
        host_ip (str): The host IP address of the RCAbench server (optional).
        host_port (int): The host port of the RCAbench server (optional).
    Returns:
        dict: Paths to the diff, error, uncompressed codebase, and optionally submission directory.
    Raises:
        ValueError: If repo-vul.tar.gz is not found.
    """

    diff_path = remote_fetch_diff(arvo_id, output_dir=cache_path)
    error_path = remote_fetch_error(arvo_id, output_dir=workspace_path)
    codebase_tar = remote_fetch_codebase(arvo_id, output_dir=workspace_path)

    # Also prepare the shared directory
    shared_path = f"{workspace_path}/shared"
    os.makedirs(shared_path, exist_ok=True)
    
    if not os.path.exists(codebase_tar):
        raise ValueError(f"{CODEBASE_FILE_NAME} not found in {workspace_path}")
    
    with tarfile.open(codebase_tar, 'r:gz') as tar:
        tar.extractall(workspace_path)
    
    os.remove((codebase_tar))

    # Assuming the uncompressed content is the codebase
    codebase_path = workspace_path

    result = {
        "diff_path": diff_path,
        "error_path": error_path,
        "codebase_path": codebase_path,
    }

    # Prepare submission tools if host details are provided
    if host_ip and host_port:
        submission_dir = prepare_submission_tools(arvo_id, host_ip, host_port, workspace_path)
        result["submission_dir"] = submission_dir

    return result

def prepare_submission_tools(
        arvo_id: str,
        host_ip: str,
        host_port: int,
        submission_dir: str = DEFAULT_WORKSPACE_DIR
    ) -> str:
    """
    Prepares two simple submission scripts (submit_loc.sh and submit_patch.sh)
    which run curl commands to submit localization and patch files to the RCAbench server.

    Args:
        arvo_id (str): The ARVO task identifier.
        host_ip (str): The host IP address of the RCAbench server.
        host_port (int): The host port of the RCAbench server.
    Returns:
        str: The path to the directory containing the submission scripts.
    """
    submission_dir_path = Path(submission_dir)
    submission_dir_path.mkdir(parents=True, exist_ok=True)

    # Script for submitting patch
    patch_script = f'''#!/bin/bash

arvo_id="{arvo_id}"
host_ip="{host_ip}"
host_port={host_port}
patch_dir="{submission_dir}"

url="http://${{host_ip}}:${{host_port}}/patch"
data="{{\\"arvo_id\\": \\"$arvo_id\\", \\"patch_dir\\": \\"$patch_dir\\"}}"

curl -X POST "$url" -H "Content-Type: application/json" -d "$data"
'''

    with open(submission_dir_path / "submit_patch.sh", "w") as f:
        f.write(patch_script)
    os.chmod(submission_dir_path / "submit_patch.sh", 0o755)

    # Script for submitting localization
    loc_script = f'''#!/bin/bash

arvo_id="{arvo_id}"
host_ip="{host_ip}"
host_port={host_port}
patch_dir="{submission_dir}"

url="http://${{host_ip}}:${{host_port}}/evaluate"
data="{{\\"arvo_id\\": \\"$arvo_id\\", \\"patch_dir\\": \\"$patch_dir\\"}}"

curl -X POST "$url" -H "Content-Type: application/json" -d "$data"
'''

    with open(submission_dir_path / "submit_loc.sh", "w") as f:
        f.write(loc_script)
    os.chmod(submission_dir_path / "submit_loc.sh", 0o755)

    return str(submission_dir_path)


def cleanup_task_assets(result: dict) -> None:
    """
    Cleans up the task assets created by prepare_task_assets.
    
    Removes the downloaded diff and error files, submission scripts, shared directory,
    and the extracted source directory.
    
    Args:
        result (dict): The result dictionary returned by prepare_task_assets.
    """
    import shutil
    
    # Remove diff file
    diff_path = result.get("diff_path")
    if diff_path and os.path.exists(diff_path):
        os.remove(diff_path)
        print(f"Removed diff file: {diff_path}")
    
    # Remove error file
    error_path = result.get("error_path")
    if error_path and os.path.exists(error_path):
        os.remove(error_path)
        print(f"Removed error file: {error_path}")
    
    # Remove submission directory if it exists
    submission_dir = result.get("submission_dir")
    if submission_dir and os.path.exists(submission_dir):
        # Remove submission scripts
        for script in ["submit_loc.sh", "submit_patch.sh"]:
            script_path = os.path.join(submission_dir, script)
            if os.path.exists(script_path):
                os.remove(script_path)
                print(f"Removed submission script: {script_path}")
        
        # Remove shared directory
        shared_dir = os.path.join(submission_dir, "shared")
        if os.path.exists(shared_dir):
            shutil.rmtree(shared_dir)
            print(f"Removed shared directory: {shared_dir}")
        
        # Remove extracted src-vul directory
        src_vul_dir = os.path.join(result.get("codebase_path", ""), CODEBASE_SRC_NAME)
        if os.path.exists(src_vul_dir):
            shutil.rmtree(src_vul_dir)
            print(f"Removed {CODEBASE_SRC_NAME} directory: {src_vul_dir}")

