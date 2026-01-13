# 3-Container Architecture Implementation

## Overview

This implementation provides a **3-container architecture** for mini-swe-agent with A2A protocol support:

1. **Green Agent Container** - Controls ARVO containers, executes commands
2. **Purple Agent Container** - Decides commands, performs analysis  
3. **ARVO Container(s)** - Execution environment (dynamic, per task)

## Container Details

### 1. Green Agent Container

**File:** `green_agent_server.py`  
**Port:** 9009  
**Image:** `mini-swe-agent-green:latest`

**Responsibilities:**
- Receives command execution requests from purple agent
- Validates commands (workspace boundaries, allowed tools)
- Manages ARVO container lifecycle (create, execute, cleanup)
- Executes commands in ARVO containers via `docker exec`
- Returns command results to purple agent
- Handles multiple tasks simultaneously (one ARVO container per task)

**Key Features:**
- Task context management (context_id → TaskContext)
- Command validation and security
- ARVO container creation/destruction
- Workspace isolation per task

### 2. Purple Agent Container

**File:** `purple_agent_server.py`  
**Port:** 9019  
**Image:** `mini-swe-agent-purple:latest`

**Responsibilities:**
- Receives task descriptions from external evaluator
- Uses LLM to decide which commands to run
- Sends command requests to green agent via A2A
- Receives command results from green agent
- Performs root cause analysis
- Creates submission files (loc.json)
- Manages conversation state per task

**Key Features:**
- LLM integration (OpenAI API)
- Command extraction from LLM responses
- A2A client communication with green agent
- Exit condition checking (max steps, tokens, timeout)

### 3. ARVO Container(s)

**Type:** Dynamic/ephemeral  
**Image:** `n132/arvo:{arvo_id}-vul`  
**Created by:** Green agent per task

**Responsibilities:**
- Execute bash commands (ls, cat, grep, etc.)
- Compile C/C++ code (gcc, make)
- Run fuzzer tests (arvo)
- Provide isolated execution environment

**Lifecycle:**
- Created when task initializes
- Used for command execution
- Destroyed when task completes

## Communication Flow

```
┌─────────────────────────────────────────────────────────┐
│  External Evaluator (RCAJudge)                          │
│  Sends task to Purple Agent                             │
└──────────────────┬──────────────────────────────────────┘
                   │ A2A
                   │
┌──────────────────▼──────────────────────────────────────┐
│  PURPLE AGENT (Port 9019)                               │
│  1. Receives: "Part I: tools... Part II: task..."      │
│  2. LLM decides: "I need to list files"                │
│  3. Sends: "execute: ls /workspace/src-vul"              │
└──────────────────┬──────────────────────────────────────┘
                   │ A2A
                   │
┌──────────────────▼──────────────────────────────────────┐
│  GREEN AGENT (Port 9009)                                │
│  1. Receives: "execute: ls /workspace/src-vul"         │
│  2. Validates command                                    │
│  3. Executes: docker exec arvo "ls /workspace/src-vul" │
└──────────────────┬──────────────────────────────────────┘
                   │ docker exec
                   │
┌──────────────────▼──────────────────────────────────────┐
│  ARVO CONTAINER                                         │
│  Executes: ls /workspace/src-vul                        │
│  Returns: file1.c, file2.c, file3.c                     │
└──────────────────┬──────────────────────────────────────┘
                   │ Results
                   │
┌──────────────────▼──────────────────────────────────────┐
│  GREEN AGENT                                            │
│  Formats and returns:                                    │
│  <returncode>0</returncode>                             │
│  <output>file1.c, file2.c, file3.c</output>            │
└──────────────────┬──────────────────────────────────────┘
                   │ A2A
                   │
┌──────────────────▼──────────────────────────────────────┐
│  PURPLE AGENT                                           │
│  Receives results, LLM decides next command...          │
│  Loop continues until task complete                      │
└──────────────────────────────────────────────────────────┘
```

## Multiple Tasks Support

The architecture supports running multiple ARVO tasks in parallel:

```
Container State for 3 Parallel Tasks:

┌─────────────────────────────────────────────────────────┐
│  PERSISTENT CONTAINERS                                   │
│  ├── green-agent (1 container, manages all tasks)       │
│  └── purple-agent (1 container, handles all tasks)      │
└──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  DYNAMIC ARVO CONTAINERS                                │
│  ├── arvo-10055-abc123 (Task 10055)                    │
│  ├── arvo-10056-def456 (Task 10056)                    │
│  └── arvo-10057-ghi789 (Task 10057)                    │
└──────────────────────────────────────────────────────────┘

Total: 2 persistent + 3 ARVO = 5 containers
```

**Task Isolation:**
- Each task has unique `context_id` (A2A)
- Each task has its own workspace directory
- Each task has its own ARVO container
- Green agent routes commands to correct ARVO container based on context_id

## File Structure

```
agents/mini-swe-agent/
├── green_agent_server.py      # Green agent A2A server
├── purple_agent_server.py     # Purple agent A2A server
├── docker_environment.py     # ARVO container management
├── docker-compose.yml         # Docker Compose config
├── Dockerfile                 # Green agent container
├── Dockerfile.purple          # Purple agent container
├── README_A2A.md             # Usage documentation
└── ARCHITECTURE.md           # This file
```

## Security Features

1. **Command Validation:**
   - Only allowed tools can be executed
   - Commands with `..` are rejected
   - Absolute paths outside workspace are blocked

2. **Workspace Isolation:**
   - Each task has isolated workspace
   - No cross-task file access
   - Ground truth files hidden

3. **Container Isolation:**
   - ARVO containers are isolated per task
   - No shared state between tasks
   - Cleanup on task completion

## Deployment

### Development
```bash
# Run both agents locally
python green_agent_server.py --port 9009
python purple_agent_server.py --port 9019 --green-agent-url http://127.0.0.1:9009/
```

### Production (Docker Compose)
```bash
docker-compose up --build
```

### Kubernetes (Future)
- Green agent: Deployment with 1 replica
- Purple agent: Deployment with N replicas (horizontal scaling)
- ARVO containers: Created as Jobs per task

## Monitoring

**Green Agent Metrics:**
- Commands executed per task
- Failed commands count
- ARVO container creation/destruction
- Task completion time

**Purple Agent Metrics:**
- LLM API calls
- Steps per task
- Token usage
- Task completion rate

## Future Enhancements

1. **Horizontal Scaling:**
   - Multiple purple agent replicas
   - Load balancing

2. **Caching:**
   - Cache ARVO containers for faster task startup
   - Cache LLM responses

3. **Metrics Collection:**
   - Prometheus integration
   - Grafana dashboards

4. **Error Recovery:**
   - Automatic retry for failed commands
   - Task checkpointing

