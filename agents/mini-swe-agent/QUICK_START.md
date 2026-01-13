# Quick Start Guide

## Prerequisites

1. **Docker and Docker Compose** installed
2. **OpenAI API Key** (or other LLM provider)
3. **ARVO Docker images** available (e.g., `n132/arvo:10055-vul`)

## Step 1: Set Environment Variables

```bash
export OPENAI_API_KEY=your-api-key-here
export WORKSPACE_DIR="./workspace"
export LOG_DIR="./logs"
export TMP_DIR="/tmp/rcabench"
export MODEL="gpt-4o-mini"
export MAX_STEPS=50
```

## Step 2: Start Services

### Option A: Using Docker Compose (Recommended for Production)

```bash
cd agents/mini-swe-agent
docker-compose up --build
```

This will start:
- **Green Agent** on `http://localhost:9009`
- **Purple Agent** on `http://localhost:9019`

### Option B: Using scenario.toml (Recommended for Development)

Start both agents from configuration:

```bash
cd agents/mini-swe-agent
source path.sh
python start_agents.py
```

This reads `scenario.toml` and starts both agents automatically.

### Option C: Running Manually (Alternative)

**Terminal 1 - Green Agent:**
```bash
cd agents/mini-swe-agent
source path.sh
python green_agent_server.py --port 9009 --purple-agent-url http://127.0.0.1:9019/
```

**Terminal 2 - Purple Agent:**
```bash
cd agents/mini-swe-agent
source path.sh
python purple_agent_server.py --port 9019 --green-agent-url http://127.0.0.1:9009/
```

## Step 3: Verify Services

Check that both agents are running:

```bash
# Check green agent
curl http://localhost:9009/.well-known/agent-card.json

# Check purple agent
curl http://localhost:9019/.well-known/agent-card.json
```

## Step 4: Send a Task

Use the test script to send a task (just provide arvo_id):

```bash
cd agents/mini-swe-agent
source path.sh
python test_send_task_to_green.py <arvo_id>
python test_send_task_to_green.py 10055
```

**Note:** The script sends a minimal message (`Task ID: arvo:10055`). The Green Agent will:
- Extract the arvo_id
- Call `prepare_task_assets()` to fetch real codebase and error report
- Create the full task description
- Send it to the Purple Agent

### Option A: Using RCAJudge (Full Evaluation Flow)

**What this tests:** The complete end-to-end evaluation pipeline

**Flow:**
```
RCAJudge (Evaluator) 
  → Sends task to Purple Agent
  → Purple Agent decides commands
  → Purple Agent sends to Green Agent  
  → Green Agent executes in ARVO Container
  → Results flow back
  → Purple Agent creates loc.json
  → RCAJudge evaluates results
```

**What it validates:**
- ✅ A2A communication between RCAJudge and Purple Agent
- ✅ Purple Agent can receive and parse tasks
- ✅ Purple Agent can communicate with Green Agent
- ✅ Green Agent can execute commands in ARVO containers
- ✅ Complete task lifecycle (init → execute → submit → evaluate)
- ✅ Metrics calculation (IoU, accuracy, etc.)

**Note:** This requires RCAJudge to be updated to send tasks in the correct format (Part I: tools, Part II: task) OR purple agent needs to handle RCAJudge's current format.

### Option B: Manual Test

Send a test message to purple agent:

```bash
curl -X POST http://localhost:9019/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "text": "Part I: Available Tools\nYou can use: ls, cat, grep, find, gcc, make, arvo\n\nPart II: Task Instruction\nYou are analyzing a vulnerable codebase.\nTask ID: arvo:10055\nWorkspace: /workspace/arvo_10055/\nAnalyze the crash report and find the vulnerability."
        }]
      }
    },
    "id": 1
  }'
```

## Step 5: Monitor Execution

```bash
# Watch green agent logs
docker-compose logs -f green-agent

# Watch purple agent logs
docker-compose logs -f purple-agent

# Check ARVO containers
docker ps | grep arvo
```

## Step 6: Check Results

After task completion, check for submission files:

```bash
# Check workspace for loc.json
ls -la workspace/*/shared/loc.json

# View submission
cat workspace/arvo_*/shared/loc.json
```

## Troubleshooting

### Green agent can't access Docker

```bash
# Check Docker socket permissions
ls -la /var/run/docker.sock

# Ensure Docker daemon is running
docker ps
```

### Purple agent can't reach green agent

```bash
# Check network connectivity
docker-compose exec purple-agent ping green-agent

# Check green agent is running
docker-compose ps green-agent
```

### Import errors

```bash
# Verify source code is mounted
docker-compose exec green-agent ls -la /app/src

# Check PYTHONPATH
docker-compose exec green-agent echo $PYTHONPATH
```

## Stopping Services

```bash
# Stop and remove containers
docker-compose down

# Stop and remove with volumes (cleans workspace)
docker-compose down -v
```

## Next Steps

- See `NEXT_STEPS.md` for detailed implementation checklist
- See `README_A2A.md` for full documentation
- See `ARCHITECTURE.md` for architecture details

