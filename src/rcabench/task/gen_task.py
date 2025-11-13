import os
import tarfile

from rcabench import CODEBASE_FILE_NAME, AGENT_WORKSPACE_DIR, CACHE_DIR, SHARED_DIR
from rcabench.utils import remote_fetch_diff, remote_fetch_error, remote_fetch_codebase


def prepare_task_asssets(
        arvo_id: str,
        workspace_path: str = AGENT_WORKSPACE_DIR,
        cache_path: str = CACHE_DIR
        ) -> dict:
    """
    Prepares the task assets by fetching the diff, error, and codebase files
    for the given arvo_id from the remote repository.

    Args:
        arvo_id (str): The ARVO task identifier.
        workspace_path (str): Path to the workspace directory containing repo-vul.tar.gz.
        cache_path (str): Path to the cache directory for fetched files.
    Returns:
        dict: Paths to the diff, error, and uncompressed codebase.
    Raises:
        ValueError: If repo-vul.tar.gz is not found.
    """

    diff_path = remote_fetch_diff(arvo_id, output_dir=cache_path)
    error_path = remote_fetch_error(arvo_id, output_dir=cache_path)
    codebase_tar = remote_fetch_codebase(arvo_id, output_dir=workspace_path)

    # Also parepare the shared directory
    os.makedirs(SHARED_DIR, exist_ok=True)
    
    if not os.path.exists(codebase_tar):
        raise ValueError(f"{CODEBASE_FILE_NAME} not found in {workspace_path}")
    
    with tarfile.open(codebase_tar, 'r:gz') as tar:
        tar.extractall(workspace_path)
    
    os.remove((codebase_tar))

    # Assuming the uncompressed content is the codebase
    codebase_path = workspace_path

    return {
        "diff_path": diff_path,
        "error_path": error_path,
        "codebase_path": codebase_path,
    }