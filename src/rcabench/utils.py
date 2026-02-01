"""
Utility functions for fetch assets from cybergym HF repo
"""

from pathlib import Path
import tempfile

import httpx

from rcabench import CODEBASE_FILE_NAME, DEFAULT_CACHE_DIR

BASE_REPO_URL = (
    "https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/arvo"
)


def _download_file(
    url: str,
    output_path: Path,
    is_text: bool,
    not_found_msg: str,
    fail_msg: str,
    unexpected_msg: str,
) -> None:
    """
    Helper function to download a file from the given URL and save it to output_path.

    Args:
        url (str): The URL to download from.
        output_path (Path): The path to save the file.
        is_text (bool): Whether the file is text (use write_text) or binary (write_bytes).
        not_found_msg (str): Error message for 404 errors.
        fail_msg (str): Error message for other HTTP errors.
        unexpected_msg (str): Error message for unexpected errors.
    Raises:
        ValueError: With appropriate error messages.
    """
    if output_path.exists():
        return  # File already exists, skip download

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use longer timeouts for large file downloads
        # Connect: 10s, Read: 300s (5 minutes) for large codebase files
        timeout = httpx.Timeout(10.0, read=300.0)
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            # print(f"Status: {response.status_code}, URL: {response.url}")
            response.raise_for_status()
            if is_text:
                output_path.write_text(response.text)
            else:
                output_path.write_bytes(response.content)
    except httpx.ReadTimeout as e:
        raise ValueError(f"{unexpected_msg.format(e=e)}. The file may be large - try again or check your network connection.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise ValueError(not_found_msg)
        else:
            raise ValueError(fail_msg.format(e=e))
    except Exception as e:
        raise ValueError(unexpected_msg.format(e=e))


def _download_tmpfile(
    url: str, is_text: bool, not_found_msg: str, fail_msg: str, unexpected_msg: str
) -> str:
    """
    Helper function to download a file from the given URL and save it to a temporary file.

    Args:
        url (str): The URL to download from.
        is_text (bool): Whether the file is text (use write_text) or binary (write_bytes).
        not_found_msg (str): Error message for 404 errors.
        fail_msg (str): Error message for other HTTP errors.
        unexpected_msg (str): Error message for unexpected errors.
    Returns:
        str: The path to the created temporary file.
    Raises:
        ValueError: With appropriate error messages.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                # Use longer timeouts for large file downloads
                # Connect: 10s, Read: 300s (5 minutes) for large codebase files
                timeout = httpx.Timeout(10.0, read=300.0)
                with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                    response = client.get(url)
                    # print(f"Status: {response.status_code}, URL: {response.url}")
                    response.raise_for_status()
                    if is_text:
                        temp_file.write(response.text.encode("utf-8"))
                    else:
                        temp_file.write(response.content)

                return temp_file.name
            except httpx.ReadTimeout as e:
                raise ValueError(f"{unexpected_msg.format(e=e)}. The file may be large - try again or check your network connection.")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ValueError(not_found_msg)
                else:
                    raise ValueError(fail_msg.format(e=e))
            except Exception as e:
                raise ValueError(unexpected_msg.format(e=e))
    except Exception as e:
        raise ValueError(f"Failed to create temporary file: {e}")


def remote_fetch_diff(
    arvo_id: str, output_dir: Path = DEFAULT_CACHE_DIR, use_temp_file: bool = False
) -> str:
    """
    Fetches the diff file for the given arvo_id from the remote Hugging Face repository.

    Args:
        arvo_id (str): The ARVO task identifier.
        output_dir (str): Directory to save the file (ignored if use_temp_file=True).
        use_temp_file (bool): If True, download to a temporary file instead of output_dir.
    Returns:
        str: The path to the downloaded diff file.
    Raises:
        ValueError: If the file does not exist or download fails.
    """
    SRC_URL = f"{BASE_REPO_URL}/{arvo_id}/patch.diff"

    if use_temp_file:
        return _download_tmpfile(
            SRC_URL,
            False,
            f"Diff file for arvo_id {arvo_id} not found on remote repository.",
            f"Failed to download diff file for arvo_id {arvo_id}: {{e}}",
            f"Unexpected error downloading diff file for arvo_id {arvo_id}: {{e}}",
        )
    else:
        output_path = Path(output_dir) / f"{arvo_id}_patch.diff"

        _download_file(
            SRC_URL,
            output_path,
            False,
            f"Diff file for arvo_id {arvo_id} not found on remote repository.",
            f"Failed to download diff file for arvo_id {arvo_id}: {{e}}",
            f"Unexpected error downloading diff file for arvo_id {arvo_id}: {{e}}",
        )

        return str(output_path)


def remote_fetch_error(arvo_id: str, output_dir: Path = DEFAULT_CACHE_DIR) -> str:
    """
    Fetches the error file for the given arvo_id from the remote Hugging Face repository.

    Args:
        arvo_id (str): The ARVO task identifier.
    Returns:
        str: The path to the downloaded error file.
    Raises:
        ValueError: If the file does not exist or download fails.
    """
    SRC_URL = f"{BASE_REPO_URL}/{arvo_id}/error.txt"
    output_path = Path(output_dir) / f"{arvo_id}_error.txt"

    _download_file(
        SRC_URL,
        output_path,
        True,
        f"Error file for arvo_id {arvo_id} not found on remote repository.",
        f"Failed to download error file for arvo_id {arvo_id}: {{e}}",
        f"Unexpected error downloading error file for arvo_id {arvo_id}: {{e}}",
    )

    return str(output_path)


def remote_fetch_codebase(arvo_id: str, output_dir: Path = DEFAULT_CACHE_DIR) -> str:
    """
    Fetches the codebase file for the given arvo_id from the remote Hugging Face repository.

    Args:
        arvo_id (str): The ARVO task identifier.
    Returns:
        str: The path to the downloaded codebase file.
    Raises:
        ValueError: If the file does not exist or download fails.
    """
    SRC_URL = f"{BASE_REPO_URL}/{arvo_id}/{CODEBASE_FILE_NAME}"
    output_path = Path(output_dir) / f"{arvo_id}_{CODEBASE_FILE_NAME}"

    _download_file(
        SRC_URL,
        output_path,
        False,
        f"Codebase file for arvo_id {arvo_id} not found on remote repository.",
        f"Failed to download codebase file for arvo_id {arvo_id}: {{e}}",
        f"Unexpected error downloading codebase file for arvo_id {arvo_id}: {{e}}",
    )

    return str(output_path)
