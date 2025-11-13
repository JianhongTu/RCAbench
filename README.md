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
│   └── shared/            # Agent-server communication
│       ├── loc.json       # Localization submissions
│       └── patch.diff     # Patch submissions
├── agents/                 # Agent implementations
│   └── openhands/         # OpenHands integration
├── tmp/                    # Temporary cache (git-ignored)
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
- Docker (with Docker daemon running)
- Conda (recommended for environment management)
- 10GB+ disk space for task assets

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/RCAbench.git
   cd RCAbench
   ```

2. **Set up the environment**:
   ```bash
   conda create -n rcabench python=3.11
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
from rcabench.task.gen_task import prepare_task_asssets

# Prepare assets for Arvo task ID 10055
prepare_task_asssets(
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