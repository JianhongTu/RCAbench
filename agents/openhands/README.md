# OpenHands Integration for RCAbench

## Setup

1. **Install OpenHands:**
```bash
cd agents/openhands
git clone https://github.com/OpenHands/OpenHands.git openhands-repo
cd openhands-repo
git checkout c34030b2875da72f752906eec93b379fb7965d0c

# Install dependencies (requires Node.js, Poetry, Python 3.11+)
make build INSTALL_PLAYWRIGHT=false
```

2. **Pull runtime Docker image:**
```bash
docker pull docker.openhands.dev/openhands/runtime:0.62-nikolaik
```

3. **Create template directory:**
```bash
mkdir -p agents/openhands/template
# Copy config.toml and prompt.txt into template/
```

## Usage

### Basic Run
```bash
LLM_API_KEY = "<your-api-key>"
python agents/openhands/run_white_agent.py \
    --model "litellm_proxy/gemma3" \
    --base_url "https://ellm.nrp-nautilus.io/v1" \
    --log_dir ./logs \
    --tmp_dir ./tmp \
    --arvo_id 10055 \
    --workspace_path ./workspace \
    --cache_path ./cache \ 
    --server http://localhost:8000 \
    --max_iter 30 \
    --timeout 1200
```

### Parameters
- `--model`: LLM model (gpt-4, claude-sonnet-4, etc.)
- `--arvo_id`: ARVO task ID from verified_jobs.json
- `--workspace_path`: Where shared submissions go
- `--cache_path`: For task asset caching
- `--server`: Evaluation server URL
- `--max_iter`: Max agent iterations (default: 30)
- `--timeout`: Timeout in seconds (default: 1200)

## Integration with RCAbench Server

The agent automatically:
1. Prepares task assets using `prepare_task_asssets()`
2. Creates workspace with vulnerable code + crash report
3. Runs OpenHands with localization prompt
4. Outputs to `/workspace/shared/loc.json` and optionally `patch.diff`

The server then:
1. Reads submissions from shared workspace
2. Validates patches using Docker
3. Evaluates localizations against ground truth
4. Computes metrics (file accuracy, line IoU, etc.)

## Directory Structure
```
agents/openhands/
├── openhands-repo/          # OpenHands source
├── template/
│   ├── config.toml          # OpenHands config template
│   └── prompt.txt           # Localization task prompt
├── run_white_agent.py                   # Main script
└── README.md               # This file
```

## Troubleshooting

**Poetry not found:**
```bash
pip install poetry
```

**Docker permission denied:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Container cleanup issues:**
The script auto-removes containers, but manual cleanup:
```bash
docker ps -a | grep openhands-runtime | awk '{print $1}' | xargs docker rm -f
```