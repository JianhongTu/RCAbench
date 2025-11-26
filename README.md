# RCAbench

**RCAbench** (Root Cause Analysis Benchmark) is a novel cybersecurity benchmark that challenges LLM agents to conduct root-cause analysis on vulnerable codebases based on fuzzer crash reports. The system evaluates an agent's ability to localize vulnerabilities by analyzing fuzzer outputs and identifying the exact files, functions, and lines of code responsible for security flaws.

## Overview

RCAbench is built on three architectural layers:

1. **Evaluation Server** (Host): Orchestrates the evaluation process, manages task assets, validates patches, and computes localization metrics.
2. **Agent Environment**: Contains the LLM agent and scaffolding code that interacts with the test environment.
3. **Test Environment**: Isolated Docker containers with vulnerable codebases, fuzzer reports, and build environments.

The system uses the [Arvo dataset](https://github.com/n132/arvo-dataset) of real-world fuzzing vulnerabilities, providing ground truth patches and crash reports for evaluation.

## Key Features

- **Automated Task Provisioning**: Downloads and prepares vulnerable codebases, patches, and crash reports from remote repositories
- **Docker-Based Isolation**: Each task runs in a dedicated Docker container with pre-configured build environments
- **Multi-Metric Evaluation**: Evaluates localization accuracy using file-level, function-level, and line-level metrics with IoU scoring
- **RESTful API**: FastAPI server with endpoints for patch validation and localization evaluation
- **Ground Truth Extraction**: Automatically parses patch diffs to extract vulnerability locations
- **Agent-Agnostic Design**: Supports any LLM agent that can interact via the standardized API

## Architecture

### Directory Structure

```
RCAbench/
├── src/rcabench/           # Core package
│   ├── server/            # Evaluation server
│   │   ├── main.py       # FastAPI application and CLI
│   │   ├── server_utils.py # Docker container management
│   │   ├── eval_utils.py  # Ground truth parsing and metrics
│   │   └── db_utils.py    # SQLite database operations
│   ├── task/              # Task provisioning
│   │   └── gen_task.py    # Asset download and preparation
│   ├── utils.py           # Remote file fetching utilities
│   └── __init__.py        # Package configuration
├── data/                   # Task metadata (git-ignored)
│   ├── arvo.db            # SQLite database of Arvo tasks
│   └── verified_jobs.json # List of verified task IDs
├── tests/                  # Test suite
│   ├── test_host.py       # End-to-end integration tests
│   ├── test_evaluation.py # Localization metric tests
│   └── test_*.py          # Component-level tests
├── workspace/              # Shared workspace (git-ignored)
│   └── shared/            # Agent-server communication (DEFAULT_WORKSPACE_DIR)
│       ├── loc.json       # Localization submissions
│       └── patch.diff     # Patch submissions
├── agents/                 # Agent implementations
│   └── openhands/         # OpenHands integration
├── tmp/                    # Temporary cache (git-ignored)
│   └── arvo_{arvo_id}-{agent_id}/  # Agent-specific temp directory
│       └── workspace/              # Isolated workspace for each agent
│           ├── shared/             # Shared resources directory
│           ├── src-vul/            # Extracted vulnerable source code
│           ├── {arvo_id}_error.txt # Fuzzer error report
│           ├── submit_patch.sh     # Script to submit patches
│           └── submit_loc.sh       # Script to submit localization results
└── docker/                 # Container definitions
```

### Data Flow

1. **Task Preparation**: Server downloads diff, error report, and codebase tarball for a given Arvo task ID
2. **Agent Execution**: Agent analyzes the fuzzer crash report and vulnerable codebase
3. **Submission**: Agent submits localization predictions (`loc.json`) and optional patches (`patch.diff`)
4. **Validation**: Server runs patches in Docker containers to verify they fix the vulnerability
5. **Evaluation**: Server compares predictions against ground truth and computes metrics

## Evaluation Metrics

RCAbench evaluates localization quality using:

- **File Accuracy**: Exact match of predicted file to ground truth file
- **Function Top-K Recall**: Whether the correct function appears in top-K predictions
- **Line Top-K Recall**: Whether correct line spans appear in top-K predictions
- **Line IoU (Intersection over Union)**: Overlap between predicted and ground truth line ranges

Each metric provides insight into different granularities of vulnerability localization.

## Prerequisites

- Python 3.11+
- Docker (with Docker daemon running) - for local validation
- Conda (recommended for environment management)
- 10GB+ disk space for task assets
- Kubernetes cluster access (for running jobs on NRP/Nautilus) - optional

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/RCAbench.git
   cd RCAbench
   ```

2. **Set up the environment**:
   ```bash
   conda create -n rcabench python=3.12
   conda activate rcabench
   pip install -e .  # Install in development mode
   ```

3. **Download task metadata**:
   ```bash
   python scripts/download_meta.py
   ```
   This downloads the Arvo task database to `./data/arvo.db`.

### Running the Evaluation Server

**Start the server**:
```bash
conda activate rcabench
python -m rcabench.server.main start --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.

**API Endpoints**:
- `GET /` - Health check
- `POST /patch` - Validate a patch submission
- `POST /evaluate` - Evaluate localization predictions

### Testing

Run the end-to-end integration test:
```bash
conda activate rcabench
python tests/test_host.py
```

This test validates:
- Evaluation server initialization
- Task asset preparation
- Patch validation workflow
- Localization evaluation with mock submissions

### Example: Preparing a Task

```python
from rcabench.task.gen_task import prepare_task_assets

# Prepare assets for Arvo task ID 10055
prepare_task_assets(
    arvo_id="10055",
    workspace_path="./workspace",
    cache_path="./tmp"
)
```

This downloads:
- `10055_patch.diff` - The ground truth patch
- `10055_error.txt` - Fuzzer crash report
- `repo-vul.tar.gz` - Vulnerable codebase archive

### Example: Evaluating Localizations

```python
from rcabench.server.eval_utils import get_ground_truth, evaluate_localization

# Get ground truth for task
gts = get_ground_truth("10055")

# Submit predictions (normally from agent)
preds = [
    Localization(
        task_id="arvo:10055",
        file="magick/utility.c",
        old_span=LineSpan(start=6357, end=6363),
        new_span=LineSpan(start=6357, end=6363),
        function=""
    )
]

# Evaluate
report = evaluate_localization(preds, gts)
print(f"File Accuracy: {report.file_acc}")
print(f"Line IoU: {report.line_iou_mean}")
```

## CLI Commands

The evaluation server provides a command-line interface:

```bash
# Start the evaluation server
python -m rcabench.server.main start [--host HOST] [--port PORT]

# Clean up temporary files
python -m rcabench.server.main teardown
```

## Localization Submission Format

Agents submit predictions as JSON files in the shared workspace:

```json
[
  {
    "task_id": "arvo:10055",
    "file": "magick/utility.c",
    "old_span": {"start": 6357, "end": 6363},
    "new_span": {"start": 6357, "end": 6363},
    "function": ""
  }
]
```

Fields:
- `task_id`: Arvo task identifier (format: `arvo:XXXXX`)
- `file`: Relative path to the file within the codebase
- `old_span`: Line range in the vulnerable version (1-indexed, inclusive)
- `new_span`: Line range in the patched version
- `function`: (Optional) Function name containing the vulnerability

## Database Schema

The `arvo.db` SQLite database contains task metadata:

- `localId`: Unique task identifier (INTEGER PRIMARY KEY)
- `project`: Project name (e.g., "graphicsmagick")
- `reproduced`: Whether the crash was successfully reproduced (BOOLEAN)
- `reproducer_vul`: Docker image for vulnerable version
- `reproducer_fix`: Docker image for patched version
- `patch_located`: Whether patch was found (BOOLEAN)
- `patch_url`: URL to the patch commit
- `verified`: Manual verification status (BOOLEAN)
- `fuzz_target`: Name of the fuzz target
- `fuzz_engine`: Fuzzing engine used (e.g., "libFuzzer")
- `sanitizer`: Sanitizer used (e.g., "address")
- `crash_type`: Type of crash (e.g., "heap-buffer-overflow")

## Development

### Running Tests

```bash
conda activate rcabench

# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_host.py
python tests/test_evaluation.py
```

### Adding New Tasks

1. Add task ID to `data/verified_jobs.json`
2. Ensure the task exists in `data/arvo.db`
3. Verify assets are available in the remote repository
4. Test with `prepare_task_asssets()`

---

## Validating ARVO Tasks

### Single Task

```bash
python3 scripts/validate.py 10055
```

### All Tasks

```bash
# Fast (no Docker)
python3 scripts/validate.py --all --skip-docker

# Full validation (with Docker)
python3 scripts/validate.py --all

# Sample first 20 tasks
python3 scripts/validate.py --all --sample 20
```

**Outputs** (saved to `data/pipeline_results/`):
- `tier1_tasks.txt` - Fully validated (patch + compile + fixes bug)
- `tier2_tasks.txt` - Docker available (not fully tested)
- `tier3_tasks.txt` - No Docker image
- `easy/medium/hard_tasks.txt` - By difficulty

---

## Running on Kubernetes (NRP/Nautilus)

RCAbench can be run on Kubernetes clusters (e.g., NRP/Nautilus) for distributed validation tasks. The system builds from source directly in the cluster, eliminating the need to build and push Docker images.

### Prerequisites

- `kubectl` configured with cluster access
- Kubernetes namespace with appropriate permissions
- PersistentVolumeClaim (PVC) for storing results (default: `gaurs-storage`)

### Quick Start

1. **Submit a validation job**:
   ```bash
   # Submit with auto-generated tag
   ./scripts/submit_nrp_job.sh
   
   # Submit with custom tag
   ./scripts/submit_nrp_job.sh my-test-run
   
   # Submit with custom repository URL
   ./scripts/submit_nrp_job.sh my-test-run https://github.com/your-username/RCAbench.git
   
   # Submit with custom namespace and PVC
   ./scripts/submit_nrp_job.sh my-test-run https://github.com/JianhongTu/RCAbench.git wang-research-lab gaurs-storage
   ```

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

### How It Works

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

### Validation Limitations in Kubernetes

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

### Customizing Tasks

Edit `k8s/rcabench-pipeline-job-build.yml` to change which tasks are validated:

```yaml
# Single task (line 28)
echo "10055" > /workspace/rcabench/data/available_tasks.txt

# Multiple tasks
echo -e "10055\n10096\n10123" > /workspace/rcabench/data/available_tasks.txt
```

### Job Manifest

The job manifest (`k8s/rcabench-pipeline-job-build.yml`) includes:

- **Resources**: 2-4 CPU, 4-8Gi memory, 20-50Gi ephemeral storage
- **Tolerations**: `nautilus.io/chase-ci=true` for Nautilus cluster scheduling
- **Volumes**: 
  - EmptyDir for workspace (temporary)
  - PVC for results persistence (`gaurs-storage` by default)
- **TTL**: Jobs are automatically deleted after 24 hours

### Troubleshooting

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