# Mini-SWE-Agent Quick Start

## Prerequisites

1. **Docker and Docker Compose** installed
2. **OpenAI API Key** configured in `path.sh`
3. **ARVO Docker images** available (e.g., `n132/arvo:10055-vul`)

## Getting Started

Choose one option below:

| Option | How it Works | Best For |
|--------|------------|----------|
| **A - AgentBeats** | Starts agents locally + sends tasks automatically | Development (fastest, all-in-one) |
| **B - Manual Scripts** | Starts agents locally with scripts + manual task sending | Debugging individual agents |

---

### Option A: Using AgentBeats (Recommended for Development)

**Quick Start (All-in-one):**
```bash
cd agents/mini-swe-agent
source env.sh (Create a duplicate file from example_env.sh)
uv run agentbeats-run scenario.toml
```

This will:
1. Start both agents (green and purple) automatically
2. Load tasks from `scenario.toml`
3. Send and execute the task(s) automatically
4. Evaluate results and show metrics
5. Shutdown gracefully when complete

### Option B: Using Manual Scripts (For Development)

**Manual Start (Two Steps):**

Start both agents from configuration:
```bash
cd agents/mini-swe-agent
source env.sh (Create a duplicate file from example_env.sh)
python start_agents.py [--arvo-id <arvo_id>]
python start_agents.py  # Uses first task_id from scenario.toml
python start_agents.py --arvo-id 14368  # Override with specific ID
```

Then in another terminal, send the task:
```bash
cd agents/mini-swe-agent
source path.sh
python send_task.py [arvo_id|--all]
python send_task.py  # Uses all task_ids from scenario.toml
python send_task.py 14368  # Override with specific ID
python send_task.py --all  # Explicitly use all task_ids from scenario.toml
```

**Note:** If no ARVO ID is provided, both scripts will use **all** `task_ids` from `scenario.toml`'s `config.task_ids` array. Use `--all` to explicitly run all tasks, or provide a specific ID to run just one.

## Monitoring Execution

Check that both agents are running:

```bash
# Check green agent
curl http://localhost:9009/.well-known/agent-card.json

# Check purple agent
curl http://localhost:9019/.well-known/agent-card.json
```

## Sending Tasks

**With AgentBeats (Option A):** Tasks are sent automatically - no manual step needed.

**With Manual Scripts (Option C):** Use the send script:

```bash
cd agents/mini-swe-agent
source path.sh
python send_task.py [arvo_id]
python send_task.py  # Uses first task_id from scenario.toml
python send_task.py 14368  # Override with specific ID
```

**Note:** The ARVO ID can be provided via command line or will be read from `scenario.toml`'s `config.task_ids` array.

**Note:** The script sends a minimal message (`Task ID: arvo:14368`). The Green Agent will:
- Extract the arvo_id
- Call `prepare_task_assets()` to fetch real codebase and error report
- Create the full task description
- Send it to the Purple Agent

**Log File Location:** 
- Log directory: `./logs/log_{timestamp}/` (timestamped, shared by all tasks in the run)
- General logs: `./logs/log_{timestamp}/agents.log` (shared by both agents)
- Per-ARVO logs: `./logs/log_{timestamp}/arvo_{arvo_id}.log` (one file per ARVO ID, contains logs from both green and purple agents)

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

## Viewing Logs

```bash
# Watch green agent logs
docker-compose logs -f green-agent

# Watch purple agent logs
docker-compose logs -f purple-agent

# Check ARVO containers
docker ps | grep arvo
```

## Checking Results

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

