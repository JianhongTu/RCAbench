# Running RCAbench on Kubernetes (NRP/Nautilus)

RCAbench can be run on Kubernetes clusters (e.g., NRP/Nautilus) for distributed validation tasks. The system builds from source directly in the cluster, eliminating the need to build and push Docker images.

## Prerequisites

- `kubectl` configured with cluster access
- Kubernetes namespace with appropriate permissions
- **PersistentVolumeClaim (PVC)** for storing results
  - Default: `gaurs-storage` (auto-detected if not specified)
  - **You should specify your PVC name** if it's different from the default
  - The script will attempt to auto-detect a PVC if the default doesn't exist

## Quick Start

1. **Submit a validation job**:
   
   **Script arguments (in order):**
   - `TAG` (optional): Job tag/name (defaults to timestamp)
   - `REPO_URL` (optional): GitHub repository URL (defaults to JianhongTu/RCAbench)
   - `NAMESPACE` (optional): Kubernetes namespace (auto-detected from kubeconfig)
   - `PVC_NAME` (optional): PersistentVolumeClaim name (defaults to `gaurs-storage`)
   
   ```bash
   # Simplest: auto-generated tag, default repo, auto-detected namespace, default PVC
   ./scripts/submit_nrp_job.sh
   
   
   # Full specification: tag, repo, namespace, and PVC (RECOMMENDED if your PVC differs)
   ./scripts/submit_nrp_job.sh my-test-run https://github.com/JianhongTu/RCAbench.git wang-research-lab your-pvc-name
   
   ```
   
   **Note**: To specify your PVC name, you must provide all previous arguments (tag, repo, namespace). The script defaults to `gaurs-storage` and will try to auto-detect a PVC if that doesn't exist, but **explicitly specifying your PVC name is recommended** for reliability.

2. **Monitor the job**:
   ```bash
   # Check job status
   kubectl get jobs -n <namespace>
   
   # Watch logs in real-time
   kubectl logs -f job/rcabench-pipeline-<TAG> -n <namespace>
   
   # Check pod status
   kubectl get pods -n <namespace> | grep rcabench-pipeline
   ```

3. **Results are automatically copied**:
   - The script waits for job completion
   - Results are stored in a PVC during execution (at `/results/<TAG>/validation/`)
   - **Why a temporary pod?** `kubectl cp` only works on Running pods, but job pods complete and enter `Succeeded` state. A temporary pod mounts the same PVC and copies results to your local `./results/<TAG>/validation/` directory
   - Results include: `*_result.json`, `*_patch.diff`, `*_error.txt`

## How It Works

The Kubernetes job uses a two-container approach:

1. **Init Container** (`git-clone`):
   - Clones the RCAbench repository from GitHub
   - Checks out the specified branch (default: `dataset-curation`)
   - Creates the task list file (`data/available_tasks.txt`)

2. **Main Container** (`rcabench-pipeline`):
   - Installs system dependencies (patch utility)
   - Installs Python dependencies from `requirements.txt`
   - Installs RCAbench package in development mode
   - Runs validation for the specified task(s) **with `--skip-docker` flag**
   - Copies results to the mounted PVC at `/results/<TAG>/validation/`

## Validation Limitations in Kubernetes

**Important**: The Kubernetes job runs validation with the `--skip-docker` flag, which means it only performs **Stage 1** and **Stage 2** validation:

- ✅ **Stage 1 (Asset Validation)**: Downloads patch, error report, and codebase. Tests if patch CAN be applied (dry-run only).
- ✅ **Stage 2 (Docker Check)**: Verifies if Docker image exists on Docker Hub.
- ❌ **Stage 3 (Docker Validation)**: **SKIPPED** - This is where actual patch application, compilation, and fuzzer execution happen.

**Why these limitations exist:**

1. **Docker is not available**:
   - The pod uses `python:3.11-slim` base image, which doesn't include Docker daemon
   - Installing Docker-in-Docker (DinD) requires privileged containers
   - Kubernetes cluster policy blocks privileged containers: `Privileged container is not allowed`
   - Docker validation requires the `docker` Python library and a running Docker daemon

2. **Patch is not actually applied**:
   - Stage 1 only performs a `--dry-run` test to check if the patch CAN be applied
   - Actual patch application happens in Stage 3 (Docker validation) inside the Docker container
   - Since Docker validation is skipped, patches are never actually applied to the codebase

3. **Fuzzer-poc is not running**:
   - The fuzzer (`arvo` command) only runs in Stage 3 inside the Docker container
   - The Docker container contains the vulnerable codebase, build environment, and fuzzer setup
   - Since Docker validation is skipped, the fuzzer never executes

**What the Kubernetes job DOES validate:**
- ✅ Assets are downloadable (patch, error report, codebase)
- ✅ Patch format is correct and can theoretically be applied
- ✅ Docker image exists for the task (but not used)
- ✅ Task metadata and difficulty classification

**To run full validation (with Docker, patch application, and fuzzer):**
- Run `validate.py` locally with Docker installed: `python3 scripts/validate.py 10055` (without `--skip-docker`)
- Or use the evaluation server which has Docker access

## Customizing Tasks

Edit `k8s/rcabench-pipeline-job-build.yml` to change which tasks are validated:

```yaml
# Single task (line 28)
echo "10055" > /workspace/rcabench/data/available_tasks.txt

# Multiple tasks
echo -e "10055\n10096\n10123" > /workspace/rcabench/data/available_tasks.txt
```

## Job Manifest

The job manifest (`k8s/rcabench-pipeline-job-build.yml`) includes:

- **Resources**: 2-4 CPU, 4-8Gi memory, 20-50Gi ephemeral storage
- **Tolerations**: `nautilus.io/chase-ci=true` for Nautilus cluster scheduling
- **Volumes**: 
  - EmptyDir for workspace (temporary)
  - PVC for results persistence (`gaurs-storage` by default)
- **TTL**: Jobs are automatically deleted after 24 hours

## Troubleshooting

**Cluster connectivity issues**:
```bash
# Check cluster connectivity
./scripts/check_cluster_connectivity.sh

# Verify namespace access
kubectl get pods -n <namespace>
```

**Job not scheduling**:
- Ensure your namespace has the correct tolerations
- Check resource quotas: `kubectl describe quota -n <namespace>`

**Results not copying**:
- Verify PVC exists: `kubectl get pvc -n <namespace>`
- Check PVC pod logs: `kubectl logs pvc-copy-<TAG>-<ID> -n <namespace>`
- Manually check PVC contents: Create a pod with PVC mount and inspect `/results/`

**Authentication errors**:
- Ensure `kubectl` is properly configured: `kubectl config current-context`
- For OIDC authentication issues, check certificate trust in macOS Keychain Access

