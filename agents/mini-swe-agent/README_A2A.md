# Mini-SWE-Agent A2A Implementation

This directory contains the A2A-compliant implementation of mini-swe-agent with **3-container architecture**:

1. **Green Agent Container** - Controls ARVO containers, executes commands
2. **Purple Agent Container** - Decides commands, performs root cause analysis
3. **ARVO Container(s)** - Execution environment (created dynamically per task)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GREEN AGENT Container (Port 9009)                     │
│  - Controls ARVO containers                            │
│  - Executes commands in ARVO                           │
│  - Receives: "execute: ls /workspace/src-vul"          │
│  - Returns: Command results                            │
└──────────────────┬──────────────────────────────────────┘
                   │ A2A Protocol
                   │
┌──────────────────▼──────────────────────────────────────┐
│  PURPLE AGENT Container (Port 9019)                    │
│  - Decides which commands to run (LLM)                  │
│  - Sends: "execute: ls /workspace/src-vul"              │
│  - Receives: Command results                            │
│  - Performs root cause analysis                         │
└──────────────────────────────────────────────────────────┘
                   ▲
                   │ docker exec
                   │
┌──────────────────┴──────────────────────────────────────┐
│  ARVO Container (Dynamic, per task)                     │
│  - Executes: ls, cat, gcc, make, arvo, etc.            │
└──────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
# Install A2A SDK and dependencies
pip install a2a-sdk uvicorn openai docker

# Or install from requirements if available
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export WORKSPACE_DIR="./workspace"
export LOG_DIR="./logs"
export TMP_DIR="/tmp/rcabench"
export MODEL="gpt-4o-mini"
export MAX_STEPS=50
```

## Usage

### Option 1: Using Docker Compose (Recommended)

```bash
cd agents/mini-swe-agent

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export WORKSPACE_DIR="./workspace"
export LOG_DIR="./logs"

# Start both containers
docker-compose up --build
```

This will start:
- Green agent on `http://localhost:9009`
- Purple agent on `http://localhost:9019`

### Option 2: Running Individually

**Start Green Agent:**
```bash
python agents/mini-swe-agent/green_agent_server.py \
    --host 127.0.0.1 \
    --port 9009 \
    --tmp-dir /tmp/rcabench
```

**Start Purple Agent:**
```bash
python agents/mini-swe-agent/purple_agent_server.py \
    --host 127.0.0.1 \
    --port 9019 \
    --green-agent-url http://127.0.0.1:9009/ \
    --model gpt-4o-mini \
    --max-steps 50
```

## Communication Flow

### 1. Task Initialization

**Purple Agent receives:**
```
Part I: Available Tools
- ls, cat, grep, gcc, make, arvo, etc.

Part II: Task Instruction
You are analyzing a vulnerable codebase...
Task ID: arvo:10055
Workspace: /workspace/arvo_10055/
```

**Purple Agent sends to Green Agent:**
```
Task received. Initializing analysis...
```

**Green Agent:**
- Prepares task assets
- Creates ARVO container
- Returns: "Task initialized successfully. ARVO container ready."

### 2. Command Execution Loop

**Purple Agent (LLM decides):**
```
"I need to list files first"
→ Sends: "execute: ls /workspace/src-vul"
```

**Green Agent:**
- Validates command
- Executes in ARVO container: `docker exec arvo "ls /workspace/src-vul"`
- Returns: `<returncode>0</returncode><output>file1.c, file2.c...</output>`

**Purple Agent:**
- Receives results
- LLM analyzes: "I see file1.c, let me read it"
- Sends: "execute: cat /workspace/src-vul/file1.c"

**Loop continues until task complete...**

### 3. Task Completion

**Purple Agent:**
- Creates submission: `loc.json`
- Sends: "[TASK FINISHED]"

**Green Agent:**
- Detects completion
- Cleans up ARVO container
- Returns: "[TASK COMPLETED] Commands executed: 15, Failed: 0, Time: 120.5s"

## Multiple Tasks

When running multiple ARVO tasks in parallel:

```
Containers:
├── green-agent (1 container, handles all tasks)
├── purple-agent (1 container, handles all tasks)
├── arvo-10055 (1 container per task)
├── arvo-10056 (1 container per task)
└── arvo-10057 (1 container per task)

Total: 2 persistent + N ARVO containers
```

Each task has its own:
- Workspace directory
- ARVO container
- Context ID (for A2A communication)

## Files

- `green_agent_server.py` - Green agent A2A server
- `purple_agent_server.py` - Purple agent A2A server
- `docker_environment.py` - ARVO container management
- `docker-compose.yml` - Docker Compose configuration
- `Dockerfile` - Green agent container
- `Dockerfile.purple` - Purple agent container

## Testing

### Test Green Agent Directly

```bash
# Start green agent
python green_agent_server.py --port 9009

# In another terminal, send test message
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "execute: ls /workspace"}]
      }
    },
    "id": 1
  }'
```

### Test Purple Agent

```bash
# Start both agents
docker-compose up

# Send task to purple agent
curl -X POST http://localhost:9019/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Part I: Tools: ls, cat\nPart II: Analyze task arvo:10055"}]
      }
    },
    "id": 1
  }'
```

## Troubleshooting

**Green agent can't access Docker:**
- Ensure `/var/run/docker.sock` is mounted
- Check Docker daemon is running
- Verify container has Docker socket access

**Purple agent can't reach green agent:**
- Check network connectivity
- Verify green agent URL: `http://green-agent:9009/` (in Docker) or `http://127.0.0.1:9009/` (local)
- Check firewall settings

**ARVO container not created:**
- Verify ARVO image exists: `docker images | grep arvo`
- Check Docker image name format: `n132/arvo:{arvo_id}-vul`
- Check logs: `docker-compose logs green-agent`

## Notes

- Green agent manages ARVO container lifecycle
- Purple agent makes all command decisions via LLM
- Commands are validated before execution
- Workspace boundaries are enforced
- Ground truth files are hidden from purple agent

