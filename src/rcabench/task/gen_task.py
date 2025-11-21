from pathlib import Path
import os
import json
import subprocess
import shutil
from pathlib import Path
import tarfile
from uuid import uuid4

from rcabench import (
    DEFAULT_TEMP_DIR,
    CODEBASE_FILE_NAME,
    AgentWorldPath,
)
from rcabench.utils import remote_fetch_error, remote_fetch_codebase


def prepare_task_assets(
    arvo_id: str,
    tmp_dir: Path = DEFAULT_TEMP_DIR,
    host_ip: str = "",
    host_port: int = 0,
) -> dict:
    """
    Prepares the task assets by fetching the diff, error, and codebase files
    for the given arvo_id from the remote repository.

    Creates a temporary directory structure with unique agent ID for isolation.

    Args:
        arvo_id (str): The ARVO task identifier.
        host_ip (str): The host IP address of the RCAbench server (optional).
        host_port (int): The host port of the RCAbench server (optional).
    Returns:
        dict: Paths to the diff, error, uncompressed codebase, submission directory,
              and temp_dir containing the created temporary directory path.
    Raises:
        ValueError: If repo-vul.tar.gz is not found.
    """

    # Always create temporary directory with unique agent ID
    agent_id = uuid4().hex
    agent_paths = AgentWorldPath(arvo_id=int(arvo_id), agent_id=agent_id, temp=tmp_dir)

    # Create the directory structure
    agent_paths.agent_dir.mkdir(parents=True, exist_ok=True)
    agent_paths.workspace_dir.mkdir(parents=True, exist_ok=True)
    agent_paths.shared_dir.mkdir(parents=True, exist_ok=True)

    error_path = remote_fetch_error(arvo_id, output_dir=agent_paths.workspace_dir)
    codebase_tar = remote_fetch_codebase(arvo_id, output_dir=agent_paths.workspace_dir)

    if not os.path.exists(agent_paths.codebase_compressed_path):
        raise ValueError(
            f"{CODEBASE_FILE_NAME} not found in {agent_paths.workspace_dir}"
        )

    with tarfile.open(agent_paths.codebase_compressed_path, "r:gz") as tar:
        tar.extractall(agent_paths.workspace_dir)

    os.remove(agent_paths.codebase_compressed_path)

    # Assuming the uncompressed content is the codebase
    codebase_path = str(agent_paths.workspace_dir)

    result = {
        "error_path": str(agent_paths.arvo_id_error_path),
        "codebase_path": codebase_path,
        "temp_dir": str(agent_paths.agent_dir),
        "agent_paths": agent_paths,  # Include the AgentWorldPath object for cleanup
    }

    # Prepare submission tools if host details are provided
    if host_ip and host_port:
        submission_dir = prepare_submission_tools(
            arvo_id, host_ip, host_port, agent_paths.workspace_dir
        )
        result["submission_dir"] = submission_dir

    return result


def prepare_submission_tools(
    arvo_id: str,
    host_ip: str,
    host_port: int,
    submission_dir: Path,
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
    patch_script = f"""#!/bin/bash

arvo_id="{arvo_id}"
host_ip="{host_ip}"
host_port={host_port}
patch_dir="{submission_dir}"

url="http://${{host_ip}}:${{host_port}}/patch"
data="{{\\"arvo_id\\": \\"$arvo_id\\", \\"patch_dir\\": \\"$patch_dir\\"}}"

curl -X POST "$url" -H "Content-Type: application/json" -d "$data"
"""

    with open(submission_dir_path / "submit_patch.sh", "w") as f:
        f.write(patch_script)
    os.chmod(submission_dir_path / "submit_patch.sh", 0o755)

    # Script for submitting localization
    loc_script = f"""#!/bin/bash

arvo_id="{arvo_id}"
host_ip="{host_ip}"
host_port={host_port}
patch_dir="{submission_dir}"

url="http://${{host_ip}}:${{host_port}}/evaluate"
data="{{\\"arvo_id\\": \\"$arvo_id\\", \\"patch_dir\\": \\"$patch_dir\\"}}"

curl -X POST "$url" -H "Content-Type: application/json" -d "$data"
"""

    with open(submission_dir_path / "submit_loc.sh", "w") as f:
        f.write(loc_script)
    os.chmod(submission_dir_path / "submit_loc.sh", 0o755)

    return str(submission_dir_path)


def cleanup_task_assets(agent_paths: AgentWorldPath) -> None:
    """
    Cleans up the task assets created by prepare_task_assets.

    Removes the entire agent-specific directory containing all task assets.

    Args:
        agent_paths (AgentWorldPath): The AgentWorldPath object containing path information.
    """
    import shutil

    # Remove the entire agent directory
    if agent_paths.agent_dir.exists():
        shutil.rmtree(agent_paths.agent_dir)
        print(f"Removed agent directory: {agent_paths.agent_dir}")
