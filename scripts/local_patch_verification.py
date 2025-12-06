"""
local_patch_verification.py - Local patch verification for testing

This script runs patch verification locally using Docker containers,
similar to the Kubernetes version but for development/testing.

Usage:
    python3 scripts/local_patch_verification.py --task-id 10055
    python3 scripts/local_patch_verification.py --task-list data/verified_jobs.json --max-parallel 2
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcabench.utils import remote_fetch_diff

RESULTS_DIR = "./data/patch_verification_results"


def verify_patch_locally(task_id: str, timeout: int = 600) -> Dict:
    """Verify a patch locally using Docker."""
    result = {
        "task_id": task_id,
        "status": "unknown",
        "patch_applied": False,
        "compiled": False,
        "fuzzer_passed": False,
        "error_message": None,
        "execution_time": 0,
    }

    start_time = time.time()

    try:
        import docker

        client = docker.from_env()

        image_name = f"n132/arvo:{task_id}-vul"

        # Check if image exists locally
        try:
            client.images.get(image_name)
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"Pulling image: {image_name}")
                client.images.pull(image_name)
            else:
                raise

        # Download patch file
        patch_path = remote_fetch_diff(task_id, output_dir=Path(tempfile.gettempdir()))
        patch_filename = Path(patch_path).name

        # Run verification in container
        cmd = [
            "/bin/sh",
            "-c",
            f"patch -p1 --force < /tmp/{patch_filename}; PATCH_EXIT=$?; "
            "arvo compile; COMPILE_EXIT=$?; "
            f"timeout {timeout} arvo; FUZZER_EXIT=$?; "
            "echo RESULTS:$PATCH_EXIT:$COMPILE_EXIT:$FUZZER_EXIT",
        ]

        container = client.containers.run(
            image=image_name,
            command=cmd,
            volumes={patch_path: {"bind": f"/tmp/{patch_filename}", "mode": "ro"}},
            detach=True,
        )

        # Wait for completion
        exit_code = container.wait(timeout=timeout + 60)["StatusCode"]
        logs = container.logs().decode("utf-8", errors="ignore")
        container.remove(force=True)

        # Parse results
        result["execution_time"] = time.time() - start_time

        if exit_code == 0:
            # Parse the RESULTS line
            for line in logs.split("\n"):
                if line.startswith("RESULTS:"):
                    parts = line.split(":")
                    if len(parts) >= 4:
                        patch_exit, compile_exit, fuzzer_exit = map(int, parts[1:4])
                        result["patch_applied"] = patch_exit == 0
                        result["compiled"] = compile_exit == 0
                        result["fuzzer_passed"] = fuzzer_exit == 0

                        if result["fuzzer_passed"]:
                            result["status"] = "success"
                        elif result["compiled"]:
                            result["status"] = "failed"
                            result["error_message"] = (
                                f"Fuzzer failed with exit code {fuzzer_exit}"
                            )
                        elif result["patch_applied"]:
                            result["status"] = "failed"
                            result["error_message"] = (
                                f"Compilation failed with exit code {compile_exit}"
                            )
                        else:
                            result["status"] = "failed"
                            result["error_message"] = (
                                f"Patch application failed with exit code {patch_exit}"
                            )
                        break
        else:
            result["status"] = "failed"
            result["error_message"] = f"Container exited with code {exit_code}"

        result["logs"] = logs

    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = str(e)
        result["execution_time"] = time.time() - start_time

    return result


def save_result(result: Dict):
    """Save verification result to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = Path(RESULTS_DIR) / f"{result['task_id']}_result.json"

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved: {result_file}")


def load_task_list(task_list_file: str) -> List[str]:
    """Load task IDs from a file."""
    with open(task_list_file, "r") as f:
        if task_list_file.endswith(".json"):
            data = json.load(f)
            if isinstance(data, list):
                return [str(task).replace("arvo:", "") for task in data]
            else:
                return []
        else:
            return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Local patch verification using Docker"
    )
    parser.add_argument("--task-id", help="Single task ID to verify")
    parser.add_argument(
        "--task-list",
        default="data/verified_jobs.json",
        help="File containing task IDs",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Verification timeout in seconds"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=1, help="Maximum parallel verifications"
    )

    args = parser.parse_args()

    if args.task_id:
        # Single task verification
        print(f"Verifying task: {args.task_id}")
        result = verify_patch_locally(args.task_id, args.timeout)
        save_result(result)

        status = result["status"]
        print(f"Result: {status.upper()}")
        if result.get("error_message"):
            print(f"Error: {result['error_message']}")

    else:
        # Batch verification
        task_ids = load_task_list(args.task_list)
        print(f"Loaded {len(task_ids)} tasks for verification")

        if args.max_parallel == 1:
            # Sequential processing
            for i, task_id in enumerate(task_ids, 1):
                print(f"\n[{i}/{len(task_ids)}] Verifying task: {task_id}")
                result = verify_patch_locally(task_id, args.timeout)
                save_result(result)

                status = result["status"]
                print(f"  Result: {status.upper()}")
                if result.get("error_message"):
                    print(f"  Error: {result['error_message']}")
        else:
            # Parallel processing
            print(f"Running {args.max_parallel} parallel verifications...")

            with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
                futures = {
                    executor.submit(
                        verify_patch_locally, task_id, args.timeout
                    ): task_id
                    for task_id in task_ids
                }

                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        result = future.result()
                        save_result(result)

                        status = result["status"]
                        print(f"Task {task_id}: {status.upper()}")
                        if result.get("error_message"):
                            print(f"  Error: {result['error_message']}")
                    except Exception as e:
                        print(f"Task {task_id} failed with exception: {e}")


if __name__ == "__main__":
    main()
