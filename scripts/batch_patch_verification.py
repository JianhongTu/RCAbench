"""
batch_patch_verification.py - Batch verification of patch.diff files using Kubernetes

This script dispatches patch verification jobs to a Kubernetes cluster, where each job:
1. Fetches the vulnerable Docker image
2. Applies the patch.diff file
3. Runs 'arvo compile'
4. Runs 'arvo' (fuzzer) and checks return code (expect 0)
5. Logs errors if any step fails

Results are stored in a database and JSON files.

Usage:
    python3 scripts/batch_patch_verification.py --task-list data/verified_jobs.json --namespace default
    python3 scripts/batch_patch_verification.py --task-id 10055 --namespace default
"""

import argparse
import json
import os
import subprocess
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import yaml
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcabench.utils import remote_fetch_diff, remote_fetch_error, remote_fetch_codebase
from rcabench.db_utils import LiteDatabase

# Configuration
K8S_JOB_TEMPLATE = """
apiVersion: batch/v1
kind: Job
metadata:
  name: patch-verify-{task_id}
  namespace: {namespace}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 300
  template:
    spec:
      containers:
      - name: patch-verifier
        image: n132/arvo:{task_id}-vul
        command:
        - /bin/sh
        - -c
        - |
          echo "Starting patch verification for task {task_id}"
          
          # Install curl if not available (try different package managers)
          echo "Checking for download tools..."
          if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
            echo "Installing curl..."
            if command -v apk >/dev/null 2>&1; then
              apk add --no-cache curl
            elif command -v apt-get >/dev/null 2>&1; then
              apt-get update && apt-get install -y curl
            elif command -v yum >/dev/null 2>&1; then
              yum install -y curl
            elif command -v dnf >/dev/null 2>&1; then
              dnf install -y curl
            else
              echo "PATCH_FAILED: No package manager found and no download tools available"
              exit 1
            fi
          fi
          
          # Download patch file (try curl first, fallback to wget)
          echo "Downloading patch file..."
          if command -v curl >/dev/null 2>&1; then
            if ! curl -s -f -L -o /tmp/patch.diff "https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/arvo/{task_id}/patch.diff"; then
              echo "PATCH_FAILED: Failed to download patch with curl"
              exit 1
            fi
          elif command -v wget >/dev/null 2>&1; then
            if ! wget -q -O /tmp/patch.diff "https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/arvo/{task_id}/patch.diff"; then
              echo "PATCH_FAILED: Failed to download patch with wget"
              exit 1
            fi
          else
            echo "PATCH_FAILED: Neither curl nor wget available after installation attempt"
            exit 1
          fi

          # Apply patch (continue even if it fails)
          echo "Applying patch..."
          if ! patch -p1 --force < /tmp/patch.diff; then
            if [ "{relax}" = "true" ]; then
              echo "RELAX_MODE: Patch application failed but continuing..."
            else
              echo "PATCH_FAILED: Patch application failed"
              exit 1
            fi
          fi
          
          # Compile
          echo "Compiling..."
          if ! arvo compile; then
            echo "COMPILE_FAILED: Compilation failed"
            exit 2
          fi
          
          # Run fuzzer with timeout
          echo "Running fuzzer..."
          if ! timeout {timeout} arvo; then
            echo "FUZZER_FAILED: Fuzzer returned non-zero exit code: $?"
            exit 3
          fi
          
          echo "VERIFICATION_SUCCESS: All steps passed"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: TASK_ID
          value: "{task_id}"
      restartPolicy: Never
"""

RESULTS_DIR = "./data/patch_verification_results"
DB_PATH = "./data/patch_verification.db"


class PatchVerificationDB:
    """Database for storing patch verification results."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_verification (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,  -- 'pending', 'running', 'success', 'failed'
                patch_applied BOOLEAN,
                compiled BOOLEAN,
                fuzzer_passed BOOLEAN,
                error_message TEXT,
                k8s_job_name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                retry_count INTEGER DEFAULT 0
            )
        """
        )

        conn.commit()
        conn.close()

    def insert_or_update_result(self, task_id: str, **kwargs):
        """Insert or update a verification result."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if record exists
        cursor.execute("SELECT 1 FROM patch_verification WHERE task_id = ?", (task_id,))
        exists = cursor.fetchone()

        if exists:
            # Update
            set_parts = [f"{k} = ?" for k in kwargs.keys()]
            values = list(kwargs.values()) + [task_id]
            cursor.execute(
                f"UPDATE patch_verification SET {', '.join(set_parts)} WHERE task_id = ?",
                values,
            )
        else:
            # Insert
            columns = ["task_id"] + list(kwargs.keys())
            placeholders = ["?"] * len(columns)
            values = [task_id] + list(kwargs.values())
            cursor.execute(
                f"INSERT INTO patch_verification ({', '.join(columns)}) VALUES ({', '.join(placeholders)})",
                values,
            )

        conn.commit()
        conn.close()

    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get verification result for a task."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM patch_verification WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            columns = [
                "task_id",
                "status",
                "patch_applied",
                "compiled",
                "fuzzer_passed",
                "error_message",
                "k8s_job_name",
                "start_time",
                "end_time",
                "retry_count",
            ]
            return dict(zip(columns, row))
        return None

    def get_pending_tasks(self) -> List[str]:
        """Get list of tasks that are pending or failed and need retry."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT task_id FROM patch_verification
            WHERE status IN ('pending', 'failed') AND retry_count < 3
        """
        )
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]


def submit_k8s_job(
    task_id: str, namespace: str, timeout: int = 600, relax: bool = False
) -> str:
    """Submit a Kubernetes job for patch verification."""
    job_yaml = K8S_JOB_TEMPLATE.format(
        task_id=task_id, namespace=namespace, timeout=timeout, relax=str(relax).lower()
    )

    job_name = f"patch-verify-{task_id}"

    # Write job spec to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(job_yaml)
        job_file = f.name

    try:
        # Apply the job
        result = subprocess.run(
            ["kubectl", "apply", "-f", job_file],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Submitted job: {job_name}")
        return job_name
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job for {task_id}: {e.stderr}")
        raise
    finally:
        os.unlink(job_file)


def check_job_status(job_name: str, namespace: str) -> Dict:
    """Check the status of a Kubernetes job."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "job", job_name, "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
        )

        job_data = json.loads(result.stdout)
        status = job_data["status"]

        # Check if job completed
        if status.get("succeeded", 0) > 0:
            return {"status": "success", "completed": True}
        elif status.get("failed", 0) > 0:
            return {"status": "failed", "completed": True}
        else:
            return {"status": "running", "completed": False}

    except subprocess.CalledProcessError:
        return {"status": "not_found", "completed": False}


def get_job_logs(job_name: str, namespace: str) -> str:
    """Get logs from a completed job."""
    try:
        # Get pod name from job
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"job-name={job_name}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pod_name = result.stdout.strip()

        # Get logs
        result = subprocess.run(
            ["kubectl", "logs", pod_name, "-n", namespace],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout

    except subprocess.CalledProcessError:
        return "Failed to retrieve logs"


def parse_job_output(logs: str) -> Dict:
    """Parse the job output to determine results."""
    result = {
        "patch_applied": False,
        "compiled": False,
        "fuzzer_passed": False,
        "error_message": None,
    }

    if "VERIFICATION_SUCCESS" in logs:
        result["patch_applied"] = True
        result["compiled"] = True
        result["fuzzer_passed"] = True
    elif "RELAX_MODE: Patch application failed but continuing" in logs:
        result["patch_applied"] = False
        # Continue to check if compilation succeeded
        if "COMPILE_FAILED" not in logs:
            result["compiled"] = True
        if "FUZZER_FAILED" not in logs:
            result["fuzzer_passed"] = True
        result["error_message"] = "Patch application failed (relax mode)"
    elif "PATCH_FAILED: Failed to download patch" in logs:
        result["error_message"] = "Failed to download patch file"
    elif "PATCH_FAILED: No package manager found" in logs:
        result["error_message"] = (
            "No package manager available for installing download tools"
        )
    elif "PATCH_FAILED: Neither curl nor wget available after installation" in logs:
        result["error_message"] = "Failed to install download tools"
    elif "PATCH_FAILED: Patch application failed with both -p1 and -p0" in logs:
        result["error_message"] = "Patch application failed with both -p1 and -p0"
    elif "COMPILE_FAILED" in logs:
        result["patch_applied"] = True
        result["error_message"] = "Compilation failed"
    elif "FUZZER_FAILED" in logs:
        result["patch_applied"] = True
        result["compiled"] = True
        result["error_message"] = "Fuzzer returned non-zero exit code"

    return result


def verify_single_task(
    task_id: str,
    namespace: str,
    db: PatchVerificationDB,
    timeout: int = 600,
    relax: bool = False,
):
    """Verify a single task using Kubernetes."""
    print(f"\n{'='*60}")
    print(f"Verifying task: {task_id}")
    print("=" * 60)

    # Check if already completed
    existing = db.get_result(task_id)
    if existing and existing["status"] == "success":
        print(f"Task {task_id} already verified successfully")
        return

    # Submit job
    try:
        job_name = submit_k8s_job(task_id, namespace, timeout, relax)
        db.insert_or_update_result(
            task_id,
            status="running",
            k8s_job_name=job_name,
            start_time=time.time(),
            retry_count=(existing["retry_count"] + 1) if existing else 0,
        )

        # Wait for completion
        print(f"Waiting for job {job_name} to complete...")
        while True:
            status = check_job_status(job_name, namespace)
            if status["completed"]:
                break
            time.sleep(10)  # Check every 10 seconds

        # Get results
        logs = get_job_logs(job_name, namespace)
        parsed_result = parse_job_output(logs)

        # Update database
        final_status = "success" if parsed_result["fuzzer_passed"] else "failed"
        db.insert_or_update_result(
            task_id, status=final_status, end_time=time.time(), **parsed_result
        )

        print(f"Task {task_id}: {final_status.upper()}")
        if parsed_result["error_message"]:
            print(f"Error: {parsed_result['error_message']}")

        # Save detailed results to JSON
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_file = Path(RESULTS_DIR) / f"{task_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "status": final_status,
                    "job_name": job_name,
                    "logs": logs,
                    **parsed_result,
                },
                f,
                indent=2,
            )

    except Exception as e:
        print(f"Failed to verify task {task_id}: {e}")
        db.insert_or_update_result(
            task_id, status="failed", error_message=str(e), end_time=time.time()
        )


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


async def monitor_job(
    task_id: str, job_name: str, namespace: str, db: PatchVerificationDB, timeout: int
):
    """Monitor a single job asynchronously."""
    try:
        # Wait for completion
        while True:
            status = await asyncio.get_event_loop().run_in_executor(
                None, check_job_status, job_name, namespace
            )
            if status["completed"]:
                break
            await asyncio.sleep(10)  # Check every 10 seconds

        # Get results
        logs = await asyncio.get_event_loop().run_in_executor(
            None, get_job_logs, job_name, namespace
        )
        parsed_result = parse_job_output(logs)

        # Update database
        final_status = "success" if parsed_result["fuzzer_passed"] else "failed"
        db.insert_or_update_result(
            task_id, status=final_status, end_time=time.time(), **parsed_result
        )

        print(f"Task {task_id}: {final_status.upper()}")
        if parsed_result["error_message"]:
            print(f"Error: {parsed_result['error_message']}")

        # Save detailed results to JSON
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_file = Path(RESULTS_DIR) / f"{task_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "status": final_status,
                    "job_name": job_name,
                    "logs": logs,
                    **parsed_result,
                },
                f,
                indent=2,
            )

        return task_id, final_status

    except Exception as e:
        print(f"Failed to monitor job {task_id}: {e}")
        db.insert_or_update_result(
            task_id, status="failed", error_message=str(e), end_time=time.time()
        )
        return task_id, "failed"


async def submit_and_monitor_task(
    task_id: str, namespace: str, db: PatchVerificationDB, timeout: int, relax: bool
) -> tuple:
    """Submit a job and start monitoring it."""
    try:
        # Check if already completed
        existing = db.get_result(task_id)
        if existing and existing["status"] == "success":
            print(f"Task {task_id} already verified successfully")
            return task_id, "skipped"

        # Submit job
        job_name = await asyncio.get_event_loop().run_in_executor(
            None, submit_k8s_job, task_id, namespace, timeout, relax
        )

        db.insert_or_update_result(
            task_id,
            status="running",
            k8s_job_name=job_name,
            start_time=time.time(),
            retry_count=(existing["retry_count"] + 1) if existing else 0,
        )

        # Start monitoring
        return await monitor_job(task_id, job_name, namespace, db, timeout)

    except Exception as e:
        print(f"Failed to submit job for {task_id}: {e}")
        db.insert_or_update_result(
            task_id, status="failed", error_message=str(e), end_time=time.time()
        )
        return task_id, "failed"


async def verify_batch_parallel(
    task_ids: List[str],
    namespace: str,
    db: PatchVerificationDB,
    timeout: int,
    max_parallel: int,
    relax: bool,
):
    """Verify multiple tasks in parallel."""
    print(
        f"Starting parallel verification of {len(task_ids)} tasks (max {max_parallel} concurrent)"
    )

    completed_count = 0
    running_tasks = {}  # task_id -> asyncio.Task
    pending_tasks = task_ids.copy()

    while pending_tasks or running_tasks:
        # Submit new tasks up to max_parallel limit
        while len(running_tasks) < max_parallel and pending_tasks:
            task_id = pending_tasks.pop(0)
            task = asyncio.create_task(
                submit_and_monitor_task(task_id, namespace, db, timeout, relax)
            )
            running_tasks[task_id] = task

        if not running_tasks:
            break

        # Wait for any task to complete
        done, pending = await asyncio.wait(
            running_tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        # Process completed tasks
        for task in done:
            try:
                task_id, status = task.result()
                completed_count += 1
                print(f"Progress: {completed_count}/{len(task_ids)} tasks completed")
                del running_tasks[task_id]
            except Exception as e:
                print(f"Task failed with exception: {e}")
                completed_count += 1

    print(f"Batch verification completed: {completed_count} tasks processed")


def main():
    parser = argparse.ArgumentParser(
        description="Batch patch verification using Kubernetes"
    )
    parser.add_argument("--task-id", help="Single task ID to verify")
    parser.add_argument(
        "--task-list",
        default="data/verified_jobs.json",
        help="File containing task IDs",
    )
    parser.add_argument(
        "--namespace", default="wang-research-lab", help="Kubernetes namespace"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Job timeout in seconds"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=5, help="Maximum parallel jobs"
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Allow partial patch failures and continue with compilation/fuzzing",
    )
    parser.add_argument(
        "--sample", type=int, help="Randomly sample N tasks from the task list"
    )
    parser.add_argument(
        "--check-status", action="store_true", help="Only check status of running jobs"
    )

    args = parser.parse_args()

    # Initialize database
    db = PatchVerificationDB()

    if args.check_status:
        # Check status of running jobs
        print("Checking status of running jobs...")
        # Implementation for status checking
        return

    if args.task_id:
        # Verify single task
        verify_single_task(args.task_id, args.namespace, db, args.timeout, args.relax)
    else:
        # Batch verification
        task_ids = load_task_list(args.task_list)
        print(f"Loaded {len(task_ids)} tasks for verification")

        # Apply sampling if requested
        if args.sample and args.sample < len(task_ids):
            task_ids = random.sample(task_ids, args.sample)
            print(f"Randomly sampled {args.sample} tasks for verification")

        if args.max_parallel == 1:
            # Sequential processing (original logic)
            completed_count = 0
            for task_id in task_ids:
                verify_single_task(
                    task_id, args.namespace, db, args.timeout, args.relax
                )
                completed_count += 1
                print(f"Progress: {completed_count}/{len(task_ids)} tasks processed")
        else:
            # Parallel processing
            asyncio.run(
                verify_batch_parallel(
                    task_ids,
                    args.namespace,
                    db,
                    args.timeout,
                    args.max_parallel,
                    args.relax,
                )
            )


if __name__ == "__main__":
    main()
