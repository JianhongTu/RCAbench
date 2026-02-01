# RCAbench

**RCAbench** (Root Cause Analysis Benchmark) is a novel cybersecurity benchmark that challenges LLM agents to conduct root-cause analysis on vulnerable codebases based on fuzzer crash reports. The system evaluates an agent's ability to localize vulnerabilities by analyzing fuzzer outputs and identifying the exact files, functions, and lines of code responsible for security flaws.

## Overview

RCAbench uses a **Green-Purple Agent Architecture** based on the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/):

1. **Green Agent**: Orchestrates the evaluation process, manages ARVO Docker containers, executes commands, and computes localization metrics.
2. **Purple Agent**: The LLM-powered agent that performs root cause analysis by exploring the codebase and identifying vulnerability locations.
3. **ARVO Containers**: Isolated Docker containers with vulnerable codebases, fuzzer reports, and build environments.

The system uses the [Arvo dataset](https://github.com/n132/arvo-dataset) of real-world fuzzing vulnerabilities, providing ground truth patches and crash reports for evaluation.

## Key Features

- **A2A Protocol**: Green and purple agents communicate via the standardized Agent-to-Agent protocol
- **Automated Task Provisioning**: Downloads and prepares vulnerable codebases, patches, and crash reports from remote repositories
- **Docker-Based Isolation**: Each task runs in a dedicated ARVO Docker container with pre-configured build environments
- **Multi-Metric Evaluation**: Evaluates localization accuracy using file-level, function-level, and line-level metrics with IoU scoring
- **Ground Truth Extraction**: Automatically parses patch diffs to extract vulnerability locations
- **Leaderboard Integration**: Supports automated evaluation via [AgentBeats](https://agentbeats.dev) leaderboard

## Architecture

### Directory Structure

```
RCAbench/
├── agents/
│   └── mini-swe-agent/        # Main agent implementation
│       ├── green_agent_server.py   # Green agent (orchestrator)
│       ├── purple_agent_server.py  # Purple agent (LLM analyzer)
│       ├── docker_environment.py   # ARVO container management
│       ├── Dockerfile.green        # Green agent Docker image
│       ├── Dockerfile.purple       # Purple agent Docker image
│       ├── docker-compose.yml      # Local testing setup
│       └── scenario.toml           # Local scenario configuration
├── src/
│   ├── agentbeats/            # A2A client/server framework
│   │   ├── client.py          # A2A message sending
│   │   ├── client_cli.py      # CLI for running scenarios
│   │   ├── green_executor.py  # Green agent execution framework
│   │   └── models.py          # Data models
│   └── rcabench/              # Core evaluation package
│       ├── server/
│       │   ├── eval_utils.py      # Ground truth parsing and metrics
│       │   └── ground_truth_utils.py  # Additional ground truth functions
│       ├── task/
│       │   └── gen_task.py        # Asset download and preparation
│       └── utils.py               # Remote file fetching utilities
├── data/
│   ├── successful_patches/    # Verified ground truth patches
│   ├── successful_task_ids.txt # List of verified task IDs
│   └── arvo.db                # SQLite database of Arvo tasks
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
│   ├── A2A_PROTOCOL_EXPLANATION.md
│   └── RCABENCH_AGENTS_EXPLANATION.md
└── .github/workflows/         # CI/CD
    ├── build-green-agent.yml  # Build green agent Docker image
    └── build-purple-agent.yml # Build purple agent Docker image
```

### Data Flow

1. **Task Initialization**: Green agent downloads task assets (diff, error report, codebase) and spins up an ARVO Docker container
2. **A2A Communication**: Green agent sends task info to purple agent via A2A protocol
3. **Analysis**: Purple agent explores the codebase by requesting bash commands from green agent
4. **Submission**: Purple agent submits localization predictions (`loc.json`) to shared workspace
5. **Evaluation**: Green agent compares predictions against ground truth and computes metrics

## Evaluation Metrics

RCAbench evaluates localization quality using:

- **File Accuracy**: Exact match of predicted file to ground truth file
- **Function Top-K Recall**: Whether the correct function appears in top-K predictions
- **Line Top-K Recall**: Whether correct line spans appear in top-K predictions
- **Line IoU (Intersection over Union)**: Overlap between predicted and ground truth line ranges

Each metric provides insight into different granularities of vulnerability localization.

## Prerequisites

- Python 3.11+
- Docker (with Docker daemon running)
- 10GB+ disk space for task assets
- OpenAI API key (or compatible LLM API)

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

3. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Running Locally with Docker Compose

The easiest way to run RCAbench locally:

```bash
cd agents/mini-swe-agent

# Set your API key
export OPENAI_API_KEY="your-api-key"

# Build and run both agents
docker-compose up --build
```

This starts:
- **Green agent** on port 9009
- **Purple agent** on port 9019

### Running with the Leaderboard

RCAbench integrates with [AgentBeats](https://agentbeats.dev) for automated evaluation:

1. Fork the [RCAbench-leaderboard](https://github.com/shubham2345/RCAbench-leaderboard) repository
2. Configure your purple agent in `scenario.toml`
3. Push changes to trigger evaluation
4. Results are automatically submitted and displayed on the leaderboard

### Testing

Run the test suite:
```bash
conda activate rcabench
python -m pytest tests/
```

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
- `patch.diff` - The ground truth patch
- `error.txt` - Fuzzer crash report
- `repo-vul.tar.gz` - Vulnerable codebase archive

### Example: Evaluating Localizations

```python
from rcabench.server.eval_utils import get_ground_truth, evaluate_localization, Localization, LineSpan

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

## Building Docker Images

The green and purple agent Docker images are built automatically via GitHub Actions when pushing to the main branch.

To build locally:

```bash
# Build green agent
docker build -f agents/mini-swe-agent/Dockerfile.green -t rcabench-green-agent .

# Build purple agent
docker build -f agents/mini-swe-agent/Dockerfile.purple -t rcabench-purple-agent .
```

## Localization Submission Format

Purple agents submit predictions as `loc.json` in the shared workspace:

```json
{
  "reasoning": "Description of the vulnerability analysis...",
  "locations": [
    {
      "file": "src/utility.c",
      "function": "parse_input",
      "line_start": 6357,
      "line_end": 6363,
      "description": "Buffer overflow due to unchecked length"
    }
  ]
}
```

Fields:
- `reasoning`: Explanation of the root cause analysis
- `locations`: Array of suspected vulnerability locations
  - `file`: Relative path to the file within the codebase
  - `function`: Function name containing the vulnerability
  - `line_start`/`line_end`: Line range (1-indexed, inclusive)
  - `description`: Explanation of why this location is vulnerable

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

1. Add task ID to `data/successful_task_ids.txt`
2. Add the verified patch to `data/successful_patches/arvo_XXXXX.diff`
3. Ensure the task exists in `data/arvo.db`
4. Verify assets are available in the remote repository
5. Test with `prepare_task_assets()`

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

## Patch Verification

RCAbench includes automated patch verification that tests whether submitted `patch.diff` files actually fix vulnerabilities. The verification process runs in isolated Docker containers and performs three sequential checks:

1. **Apply Patch**: Apply the `patch.diff` to the vulnerable codebase
2. **Compile**: Run `arvo compile` to ensure the code compiles successfully
3. **Run Fuzzer**: Execute `arvo` (the fuzzer) and verify it returns exit code 0 (no crash)

### Local Verification (Development)

For development and testing, use the local verification script:

```bash
# Single task
python3 scripts/local_patch_verification.py --task-id 10055

# Batch verification (sequential)
python3 scripts/local_patch_verification.py --task-list data/verified_jobs.json

# Parallel verification (2 workers)
python3 scripts/local_patch_verification.py --task-list data/verified_jobs.json --max-parallel 2
```

### Kubernetes Batch Verification (Production)

For large-scale verification, use the Kubernetes batch processing:

```bash
# Single task
python3 scripts/batch_patch_verification.py --task-id 10055 --namespace default

# Batch verification (parallel, default 5 concurrent jobs)
python3 scripts/batch_patch_verification.py --task-list data/arvo_hf_task_ids.txt --namespace default

# Custom parallelism and timeout
python3 scripts/batch_patch_verification.py --task-list data/arvo_hf_task_ids.txt --max-parallel 10 --timeout 1200

# Sequential processing (for debugging)
python3 scripts/batch_patch_verification.py --task-list data/verified_jobs.json --max-parallel 1
```

**Features:**
- **Parallel Processing**: Runs up to `--max-parallel` jobs concurrently (default: 5)
- **Async Monitoring**: Efficiently monitors multiple jobs without blocking
- **Automatic Retries**: Failed jobs are retried up to 3 times
- **Progress Tracking**: Real-time progress updates
- **Resource Management**: Respects Kubernetes cluster capacity

**Kubernetes Job Template**: `k8s/patch-verification-job.yaml`

### Results Analysis

Analyze verification results and generate reports:

```bash
python3 scripts/analyze_patch_verification_results.py
```

**Outputs**:
- Console report with success rates and failure breakdown
- `data/patch_verification_detailed_report.json` - Detailed statistics
- `data/patch_verification_results/{task_id}_result.json` - Individual task results

### Database Storage

Results are stored in `data/patch_verification.db` with the following schema:

- `task_id`: Task identifier
- `status`: 'pending', 'running', 'success', 'failed'
- `patch_applied`: Whether patch applied successfully
- `compiled`: Whether code compiled after patching
- `fuzzer_passed`: Whether fuzzer returned exit code 0
- `error_message`: Error details if failed
- `k8s_job_name`: Kubernetes job name (for k8s verification)
- `start_time/end_time`: Execution timestamps
- `retry_count`: Number of retry attempts

---

## Running on Kubernetes

RCAbench can be run on Kubernetes clusters (e.g., NRP/Nautilus) for distributed validation tasks. See [k8s/README.md](k8s/README.md) for detailed documentation on:

- Prerequisites and setup
- Quick start guide
- How the Kubernetes job works
- Validation limitations and explanations
- Customizing tasks
- Troubleshooting

**Quick command**:
```bash
./scripts/submit_nrp_job.sh <tag>
```