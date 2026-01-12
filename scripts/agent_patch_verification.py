"""
agent_patch_verification.py - Patch verification using SweAgent for intelligent patching

This script uses SweAgent to intelligently verify and create patches:
1. Downloads vulnerable Docker image
2. Mounts PVC with pre-installed dependencies
3. Activates sweagent environment from /data
4. Runs SweAgent with the vulnerable code and original patch
5. Agent attempts to apply, debug, and create a working patch
6. Verifies with 'arvo compile' and 'arvo' fuzzer
7. Stores results in database and JSON files

Usage:
    python3 scripts/agent_patch_verification.py --task-list data/verified_jobs.json --max-parallel 2
    python3 scripts/agent_patch_verification.py --task-id 10055
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
import asyncio
import signal

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Global state for cleanup
_running_jobs = []
_cleanup_namespace = "wang-research-lab"


def cleanup_jobs(namespace: str = _cleanup_namespace):
    """Clean up running Kubernetes jobs."""
    if not _running_jobs:
        return
    
    print("\n" + "=" * 60)
    print("Cleaning up running jobs...")
    print("=" * 60)
    
    for job_name in _running_jobs:
        try:
            print(f"Deleting job: {job_name}")
            subprocess.run(
                ["kubectl", "delete", "job", job_name, "-n", namespace],
                capture_output=True,
                check=False,
            )
        except Exception as e:
            print(f"Failed to delete job {job_name}: {e}")
    
    print(f"Cleaned up {len(_running_jobs)} jobs")
    _running_jobs.clear()


def signal_handler(signum, frame):
    """Handle interrupt signals (SIGINT, SIGTERM)."""
    print("\n" + "!" * 60)
    print("Received interrupt signal, cleaning up...")
    print("!" * 60)
    cleanup_jobs()
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configuration - Sidecar pattern with separate agent and dev containers
K8S_AGENT_JOB_TEMPLATE = """
apiVersion: batch/v1
kind: Job
metadata:
  name: agent-patch-verify-{task_id}
  namespace: {namespace}
spec:
  ttlSecondsAfterFinished: 300
  activeDeadlineSeconds: 3600
  template:
    spec:
      containers:
      - name: agent
        image: {agent_image}
        imagePullPolicy: Always
        command: ["/bin/bash", "-c", "{agent_command}"]
        env:
        - name: PYTHONPATH
          value: /data
        - name: MSWEA_GLOBAL_CONFIG_DIR
          value: /data/.config/mini-swe-agent
        - name: COMMAND_PORT
          value: "{command_port}"
        - name: DEV_HOST
          value: "localhost"
        volumeMounts:
        - name: shared-volume
          mountPath: /data
        - name: coordination-volume
          mountPath: /tmp/coordination
        resources:
          requests:
            memory: "{agent_memory_request}"
            cpu: "{agent_cpu_request}"
          limits:
            memory: "{agent_memory_limit}"
            cpu: "{agent_cpu_limit}"
      - name: dev
        image: n132/arvo:{task_id}-vul
        imagePullPolicy: Always
        command: ["/bin/bash", "-c", "{dev_command}"]
        env:
        - name: COMMAND_PORT
          value: "{command_port}"
        volumeMounts:
        - name: shared-volume
          mountPath: /data
        - name: coordination-volume
          mountPath: /tmp/coordination
        resources:
          requests:
            memory: "{dev_memory_request}"
            cpu: "{dev_cpu_request}"
          limits:
            memory: "{dev_memory_limit}"
            cpu: "{dev_cpu_limit}"
      volumes:
      - name: shared-volume
        persistentVolumeClaim:
          claimName: {pvc_name}
      - name: coordination-volume
        emptyDir: {{}}
      restartPolicy: Never
"""

RESULTS_DIR = "./data/agent_patch_verification_results"
DB_PATH = "./data/agent_patch_verification.db"


class AgentPatchVerificationDB:
    """Database for storing agent-based patch verification results."""

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
            CREATE TABLE IF NOT EXISTS agent_patch_verification (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,  -- 'pending', 'running', 'success', 'failed'
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

        cursor.execute("SELECT 1 FROM agent_patch_verification WHERE task_id = ?", (task_id,))
        exists = cursor.fetchone()

        if exists:
            set_parts = [f"{k} = ?" for k in kwargs.keys()]
            values = list(kwargs.values()) + [task_id]
            cursor.execute(
                f"UPDATE agent_patch_verification SET {', '.join(set_parts)} WHERE task_id = ?",
                values,
            )
        else:
            columns = ["task_id"] + list(kwargs.keys())
            placeholders = ["?"] * len(columns)
            values = [task_id] + list(kwargs.values())
            cursor.execute(
                f"INSERT INTO agent_patch_verification ({', '.join(columns)}) VALUES ({', '.join(placeholders)})",
                values,
            )

        conn.commit()
        conn.close()

    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get verification result for a task."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT task_id, status, error_message, k8s_job_name, 
                      start_time, end_time, retry_count 
               FROM agent_patch_verification WHERE task_id = ?""",
            (task_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            columns = [
                "task_id",
                "status",
                "error_message",
                "k8s_job_name",
                "start_time",
                "end_time",
                "retry_count",
            ]
            return dict(zip(columns, row))
        return None

    def get_successful_tasks(self) -> List[str]:
        """Get list of tasks that have been successfully verified."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT task_id FROM agent_patch_verification
            WHERE status = 'success'
        """
        )
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_tasks_exceeding_retry_limit(self, max_retries: int = 2) -> List[str]:
        """Get list of tasks that have exceeded retry limit."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT task_id FROM agent_patch_verification
            WHERE retry_count >= {max_retries}
        """
        )
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def should_skip_task(self, task_id: str, max_retries: int = 2) -> tuple:
        """Check if a task should be skipped. Returns (should_skip, reason)."""
        result = self.get_result(task_id)
        
        if not result:
            return False, ""
        
        if result["status"] == "success":
            return True, "already verified successfully"
        
        retry_count = int(result["retry_count"]) if result["retry_count"] is not None else 0
        if retry_count >= max_retries:
            return True, f"exceeded retry limit (retry_count={retry_count})"
        
        return False, ""


def submit_k8s_job(
    task_id: str,
    namespace: str,
    patch_url: str,
    agent_image: str = "tovitu/sweagent",
    pvc_name: str = "jt-shared",
    command_port: int = 9000,
    dev_memory_request: int = 2,
    dev_cpu_request: int = 2,
) -> str:
    """Submit a Kubernetes Job for agent-based patch verification using sidecar pattern.
    
    Args:
        task_id: Task ID for the vulnerability
        namespace: Kubernetes namespace
        patch_url: URL to download the patch file from
        agent_image: Docker image for agent container
        pvc_name: Persistent volume claim name for shared storage
        command_port: Port for socat communication
        dev_memory_request: Dev container memory request in Gi
        dev_cpu_request: Dev container CPU request in cores
    """
    # Calculate dev limits as 2x requests
    dev_memory_limit = dev_memory_request * 2
    dev_cpu_limit = dev_cpu_request * 2
    
    # Construct agent command - runs mini-swe-agent with sidecar
    agent_command = (
        f"sleep 1 && "
        f"python src/minisweagent/run/sidecar_headless.py "
        f"-c config/sidecar.yaml -s "
        f"-t 'Given a patch file /tmp/patch.diff and a vulnerable source codebase, "
        f"your job is to create a minimal and working patch file in /tmp to fix the vulnerability. "
        f"Todos: 1) identify the source directory via pwd and verify that git has been initialized. "
        f"2) try apply the patch as it is using patch -p1 < /tmp/patch.diff and observe errors. "
        f"3) go through changes in /tmp/patch.diff one by one and determine if it is necessary to fix the vulnerability. "
        f"changes applied to changelog or tests may not be necessary. "
        f"4) since the patch application is likely to fail due to context changes, manually examine the code regions and apply each necessary updates with minimal changes to avoid regression. "
        f"5) run arvo compile && arvo to verify that the program compiles and the vulnerability no longer exists. "
        f"6) create a single new commit and generate a {task_id}.diff file in /data/verified_tasks directory. "
        f"7) create {task_id}.json in /data/verification_results with fields: task_id, list of modified files (List[str]), list of modified functions (List[str]), patch status (bool), compilation status (bool), fuzzer status (bool), and a short summary of what failed (str)."
        f"Note: if adapting the patch requires significant engineering effort, abort the task because the patch maybe unsound.' "
        f"-m glm-4.6 --timeout 300; "
        f"touch /tmp/coordination/agent_done"
    )
    
    # Construct dev command - setup and run socat for communication
    dev_command = (
        f"apt-get update && "
        f"apt-get install -y curl socat && "
        f"curl -L -o /tmp/patch.diff '{patch_url}' && "
        f"socat TCP-LISTEN:{command_port},reuseaddr,fork EXEC:/bin/bash,stderr & "
        f"SOCAT_PID=$!; "
        f"while [ ! -f /tmp/coordination/agent_done ]; do sleep 1; done; "
        f"kill $SOCAT_PID 2>/dev/null || true; "
        f"wait $SOCAT_PID 2>/dev/null || true"
    )
    
    job_yaml = K8S_AGENT_JOB_TEMPLATE.format(
        task_id=task_id,
        namespace=namespace,
        agent_image=agent_image,
        agent_command=agent_command,
        dev_command=dev_command,
        pvc_name=pvc_name,
        command_port=command_port,
        agent_memory_request="1Gi",
        agent_cpu_request="1",
        agent_memory_limit="1Gi",
        agent_cpu_limit="1",
        dev_memory_request=f"{dev_memory_request}Gi",
        dev_cpu_request=str(dev_cpu_request),
        dev_memory_limit=f"{dev_memory_limit}Gi",
        dev_cpu_limit=str(dev_cpu_limit),
    )

    job_name = f"agent-patch-verify-{task_id}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(job_yaml)
        job_file = f.name

    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", job_file],
            capture_output=True,
            text=True,
            check=True,
        )
        _running_jobs.append(job_name)
        print(f"Submitted sidecar Job: {job_name}")
        return job_name
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit Job for {task_id}: {e.stderr}")
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
        status = job_data.get("status", {})
        
        # Check if job has completed
        conditions = status.get("conditions", [])
        for condition in conditions:
            if condition.get("type") == "Complete" and condition.get("status") == "True":
                return {"status": "success", "completed": True}
            if condition.get("type") == "Failed" and condition.get("status") == "True":
                return {"status": "failed", "completed": True}
        
        # If not completed, check if it's active
        if status.get("active", 0) > 0:
            return {"status": "running", "completed": False}
        
        return {"status": "unknown", "completed": False}

    except subprocess.CalledProcessError:
        return {"status": "not_found", "completed": False}


def get_job_logs(job_name: str, namespace: str, container: str = "agent") -> str:
    """Get logs from a job's pod container."""
    try:
        # First, get the pod name for this job
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-l", f"job-name={job_name}", "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True,
            text=True,
            check=True,
        )
        pod_name = result.stdout.strip()
        
        if not pod_name:
            return "No pod found for job"
        
        # Get logs from the pod
        result = subprocess.run(
            ["kubectl", "logs", pod_name, "-n", namespace, "-c", container],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return "Failed to retrieve logs"


def verify_single_task(
    task_id: str,
    namespace: str,
    db: AgentPatchVerificationDB,
    patch_url: str,
    agent_image: str = "tovitu/sweagent",
    pvc_name: str = "jt-shared",
    command_port: int = 9000,
    dev_memory_request: int = 2,
    dev_cpu_request: int = 2,
    max_retries: int = 2,
):
    """Verify a single task using agent-based patching with sidecar pattern."""
    print(f"\n{'='*60}")
    print(f"Agent verifying task: {task_id}")
    print("=" * 60)

    should_skip, reason = db.should_skip_task(task_id, max_retries)
    if should_skip:
        print(f"Task {task_id} skipped: {reason}")
        return

    try:
        job_name = submit_k8s_job(
            task_id,
            namespace,
            patch_url,
            agent_image=agent_image,
            pvc_name=pvc_name,
            command_port=command_port,
            dev_memory_request=dev_memory_request,
            dev_cpu_request=dev_cpu_request,
        )
        existing = db.get_result(task_id)
        db.insert_or_update_result(
            task_id,
            status="running",
            k8s_job_name=job_name,
            start_time=time.time(),
            retry_count=(existing["retry_count"] + 1) if existing else 0,
        )

        print(f"Waiting for sidecar pod {job_name} to complete...")
        while True:
            status = check_job_status(job_name, namespace)
            if status["completed"]:
                break
            time.sleep(10)

        logs = get_job_logs(job_name, namespace, container="agent")
        job_status = status.get("status", "failed")
        success = job_status == "success"

        final_status = "success" if success else "failed"
        db.insert_or_update_result(
            task_id, status=final_status, end_time=time.time()
        )

        print(f"Task {task_id}: {final_status.upper()}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_file = Path(RESULTS_DIR) / f"{task_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "status": final_status,
                    "pod_name": job_name,
                    "logs": logs[:10000],  # Truncate logs for JSON
                },
                f,
                indent=2,
            )

    except Exception as e:
        print(f"Failed to verify task {task_id}: {e}")
        db.insert_or_update_result(
            task_id, status="failed", end_time=time.time()
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
    task_id: str,
    pod_name: str,
    namespace: str,
    db: AgentPatchVerificationDB,
    max_retries: int,
):
    """Monitor a single pod asynchronously."""
    try:
        while True:
            status = await asyncio.get_event_loop().run_in_executor(
                None, check_job_status, pod_name, namespace
            )
            if status["completed"]:
                break
            await asyncio.sleep(10)

        logs = await asyncio.get_event_loop().run_in_executor(
            None, get_job_logs, pod_name, namespace, "agent"
        )
        job_status = status.get("status", "failed")
        success = job_status == "success"

        final_status = "success" if success else "failed"
        db.insert_or_update_result(
            task_id, status=final_status, end_time=time.time()
        )

        print(f"Task {task_id}: {final_status.upper()}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_file = Path(RESULTS_DIR) / f"{task_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "status": final_status,
                    "pod_name": pod_name,
                    "logs": logs[:10000],
                },
                f,
                indent=2,
            )

        # Remove from tracking since it's done
        if pod_name in _running_jobs:
            _running_jobs.remove(pod_name)

        return task_id, final_status

    except Exception as e:
        print(f"Failed to monitor pod {task_id}: {e}")
        db.insert_or_update_result(
            task_id, status="failed", end_time=time.time()
        )
        return task_id, "failed"


async def submit_and_monitor_task(
    task_id: str,
    namespace: str,
    db: AgentPatchVerificationDB,
    patch_url: str,
    agent_image: str,
    pvc_name: str,
    command_port: int,
    dev_memory_request: int,
    dev_cpu_request: int,
    max_retries: int,
) -> tuple:
    """Submit a Job and start monitoring it."""
    try:
        should_skip, reason = db.should_skip_task(task_id, max_retries)
        if should_skip:
            print(f"Task {task_id} skipped: {reason}")
            return task_id, "skipped"

        pod_name = await asyncio.get_event_loop().run_in_executor(
            None,
            submit_k8s_job,
            task_id,
            namespace,
            patch_url,
            agent_image,
            pvc_name,
            command_port,
            dev_memory_request,
            dev_cpu_request,
        )

        existing = db.get_result(task_id)
        db.insert_or_update_result(
            task_id,
            status="running",
            k8s_job_name=pod_name,
            start_time=time.time(),
            retry_count=(existing["retry_count"] + 1) if existing else 0,
        )

        return await monitor_job(task_id, pod_name, namespace, db, max_retries)

    except Exception as e:
        print(f"Failed to submit Job for {task_id}: {e}")
        db.insert_or_update_result(
            task_id, status="failed", end_time=time.time()
        )
        return task_id, "failed"


async def verify_batch_parallel(
    task_ids: List[str],
    namespace: str,
    db: AgentPatchVerificationDB,
    patch_url_template: str,
    agent_image: str,
    pvc_name: str,
    command_port: int,
    dev_memory_request: int,
    dev_cpu_request: int,
    max_parallel: int,
    max_retries: int,
):
    """Verify multiple tasks in parallel using agents."""
    print(
        f"Starting parallel agent verification of {len(task_ids)} tasks (max {max_parallel} concurrent)"
    )

    completed_count = 0
    running_tasks = {}
    pending_tasks = task_ids.copy()

    while pending_tasks or running_tasks:
        while len(running_tasks) < max_parallel and pending_tasks:
            task_id = pending_tasks.pop(0)
            # Format patch URL with task_id (e.g., "http://example.com/patches/{task_id}.diff")
            patch_url = patch_url_template.format(task_id=task_id)
            task = asyncio.create_task(
                submit_and_monitor_task(
                    task_id=task_id,
                    namespace=namespace,
                    db=db,
                    patch_url=patch_url,
                    agent_image=agent_image,
                    pvc_name=pvc_name,
                    command_port=command_port,
                    dev_memory_request=dev_memory_request,
                    dev_cpu_request=dev_cpu_request,
                    max_retries=max_retries,
                )
            )
            running_tasks[task_id] = task

        if not running_tasks:
            break

        done, pending = await asyncio.wait(
            running_tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                task_id, status = task.result()
                completed_count += 1
                print(f"Progress: {completed_count}/{len(task_ids)} tasks completed")
                del running_tasks[task_id]
            except Exception as e:
                print(f"Task failed with exception: {e}")
                completed_count += 1

    print(f"Agent verification completed: {completed_count} tasks processed")


def filter_tasks(task_ids: List[str], db: AgentPatchVerificationDB, max_retries: int) -> List[str]:
    """Filter task list to remove already-successful and over-retry-limit tasks."""
    successful_tasks = set(db.get_successful_tasks())
    over_limit_tasks = set(db.get_tasks_exceeding_retry_limit(max_retries))
    
    filtered = []
    skipped_success = 0
    skipped_retry_limit = 0
    
    for task_id in task_ids:
        if task_id in successful_tasks:
            skipped_success += 1
        elif task_id in over_limit_tasks:
            skipped_retry_limit += 1
        else:
            filtered.append(task_id)
    
    if skipped_success > 0:
        print(f"Filtered out {skipped_success} already-successful tasks")
    if skipped_retry_limit > 0:
        print(f"Filtered out {skipped_retry_limit} tasks exceeding retry limit")
    
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Agent-based batch patch verification using Kubernetes"
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
        "--patch-url-template",
        default="https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/arvo/{task_id}/patch.diff",
        help="Template for patch URL (use {task_id} placeholder)",
    )
    parser.add_argument(
        "--agent-image", default="tovitu/sweagent", help="Agent container image"
    )
    parser.add_argument(
        "--pvc-name", default="jt-shared", help="PVC name for shared storage"
    )
    parser.add_argument(
        "--command-port", type=int, default=9000, help="Port for dev container communication"
    )
    parser.add_argument(
        "--dev-memory-request", type=int, default=2, help="Dev container memory request in Gi (default 2)"
    )
    parser.add_argument(
        "--dev-cpu-request", type=int, default=2, help="Dev container CPU request in cores (default 2)"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=2, help="Maximum parallel pods (default 2 for agent)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Maximum retries per task"
    )
    parser.add_argument(
        "--sample", type=int, help="Randomly sample N tasks from the task list"
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip upfront task filtering (check status individually instead)",
    )

    args = parser.parse_args()

    # Set global namespace for cleanup
    global _cleanup_namespace
    _cleanup_namespace = args.namespace

    db = AgentPatchVerificationDB()

    # Extract sidecar configuration from args
    sidecar_config = {
        "agent_image": args.agent_image,
        "pvc_name": args.pvc_name,
        "command_port": args.command_port,
        "dev_memory_request": args.dev_memory_request,
        "dev_cpu_request": args.dev_cpu_request,
    }

    try:
        if args.task_id:
            # For single task, format patch URL with task_id
            patch_url = args.patch_url_template.format(task_id=args.task_id)
            verify_single_task(
                args.task_id,
                args.namespace,
                db,
                patch_url=patch_url,
                **sidecar_config,
                max_retries=args.max_retries,
            )
        else:
            task_ids = load_task_list(args.task_list)
            print(f"Loaded {len(task_ids)} tasks for agent verification")

            if not args.skip_filter:
                task_ids = filter_tasks(task_ids, db, args.max_retries)
                print(f"After filtering: {len(task_ids)} tasks to process")

            if args.sample and args.sample < len(task_ids):
                task_ids = random.sample(task_ids, args.sample)
                print(f"Randomly sampled {args.sample} tasks for verification")

            if args.max_parallel == 1:
                completed_count = 0
                for task_id in task_ids:
                    # For each task, format patch URL with task_id
                    patch_url = args.patch_url_template.format(task_id=task_id)
                    verify_single_task(
                        task_id,
                        args.namespace,
                        db,
                        patch_url=patch_url,
                        **sidecar_config,
                        max_retries=args.max_retries,
                    )
                    completed_count += 1
                    print(f"Progress: {completed_count}/{len(task_ids)} tasks processed")
            else:
                asyncio.run(
                    verify_batch_parallel(
                        task_ids,
                        args.namespace,
                        db,
                        patch_url_template=args.patch_url_template,
                        **sidecar_config,
                        max_parallel=args.max_parallel,
                        max_retries=args.max_retries,
                    )
                )
    finally:
        # Clean up any remaining jobs
        cleanup_jobs(_cleanup_namespace)


if __name__ == "__main__":
    main()
