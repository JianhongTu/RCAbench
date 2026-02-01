# Mini-SWE-Agent Architecture

## Overview

The mini-swe-agent system consists of two Python processes (agents) that work together to perform root cause analysis on vulnerable codebases. The system uses Docker containers for command execution, but the agents themselves run as Python processes.

## Components

### 1. Green Agent (Python Process)

**What it is:**
- A Python server process (NOT a Docker container)
- Runs on port 9009
- Manages ARVO Docker containers
- Executes bash commands in those containers

**Responsibilities:**
- Receives tasks from RCAJudge or test scripts
- Downloads codebases and error reports
- Creates and manages ARVO Docker containers (one per task)
- Executes commands in ARVO containers
- Sends formatted task descriptions to Purple Agent
- Returns command execution results to Purple Agent

**Key Data Structures:**
- `task_contexts: Dict[str, TaskContext]` - Maps context_id to task state
- Each `TaskContext` contains:
  - `arvo_id`: The ARVO task ID
  - `docker_env`: ArvoDockerEnvironment instance (manages one Docker container)
  - `workspace_dir`: Unique workspace directory for this task
  - `context_id`: A2A context ID for this task

### 2. Purple Agent (Python Process)

**What it is:**
- A Python server process (NOT a Docker container)
- Runs on port 9019
- Uses LLM to decide commands
- Does NOT create or use Docker containers

**Responsibilities:**
- Receives task descriptions from Green Agent
- Uses LLM to decide which commands to run
- Sends command requests to Green Agent
- Receives command results from Green Agent
- Performs root cause analysis
- Creates submission files (`loc.json`)

**Key Data Structures:**
- `task_contexts: Dict[str, Dict[str, Any]]` - Maps context_id to task state
- Each task context contains:
  - `messages`: Conversation history with LLM
  - `command_history`: Executed commands (never summarized)
  - `agent`: DefaultAgent instance for LLM interaction
  - `step_count`: Number of steps executed
  - `total_tokens`: Token usage tracking

### 3. ARVO Docker Containers

**What they are:**
- Docker containers created by Green Agent
- One container per task/context_id
- Based on images like `n132/arvo:{arvo_id}-vul`
- Contain C/C++ compilers and tools

**Lifecycle:**
1. **Creation**: When Green Agent initializes a task
2. **Usage**: Container stays running for the task's lifetime
3. **Cleanup**: Stopped and removed when task finishes

**Isolation:**
- Each task gets its own container, even if they share the same ARVO ID
- Container name: `arvo_{arvo_id}_{unique_object_id}`
- Each container mounts its own unique workspace directory

## Task Flow

### Single Task Flow

```
1. Test Script / RCAJudge
   └─> Sends: "Task ID: arvo:14368"
       │
       ▼
2. Green Agent (Python Process)
   ├─> Extracts arvo_id: 14368
   ├─> Downloads codebase and error report
   ├─> Creates workspace directory
   ├─> Creates ARVO Docker Container (arvo_14368_<id>)
   ├─> Formats task description
   └─> Sends to Purple Agent
       │
       ▼
3. Purple Agent (Python Process)
   ├─> Receives task description
   ├─> Initializes LLM context
   └─> Starts command loop
       │
       ▼
4. Command Loop (Iterative)
   ├─> Purple Agent decides command (using LLM)
   ├─> Sends: "execute: ls /workspace/src-vul"
   │   │
   │   ▼
   ├─> Green Agent receives command
   ├─> Executes in ARVO Container
   ├─> Returns result
   │   │
   │   ▼
   └─> Purple Agent receives result
       └─> Decides next command (repeat)
           │
           ▼
5. Task Completion
   ├─> Purple Agent creates loc.json
   ├─> Green Agent cleans up Docker container
   └─> Both agents clean up task context
```

### Multiple Tasks Flow

When multiple tasks run concurrently:

```
Green Agent Process (Single Instance)
├─> task_contexts[context_id_1]
│   └─> Docker Container: arvo_14368_<id_1>
│       └─> Workspace: /tmp/rcabench/arvo_14368-<uuid_1>/
│
├─> task_contexts[context_id_2]
│   └─> Docker Container: arvo_14368_<id_2>
│       └─> Workspace: /tmp/rcabench/arvo_14368-<uuid_2>/
│
└─> task_contexts[context_id_3]
    └─> Docker Container: arvo_10055_<id_3>
        └─> Workspace: /tmp/rcabench/arvo_10055-<uuid_3>/

Purple Agent Process (Single Instance)
├─> task_contexts[context_id_1]
│   └─> LLM context, messages, command_history
│
├─> task_contexts[context_id_2]
│   └─> LLM context, messages, command_history
│
└─> task_contexts[context_id_3]
    └─> LLM context, messages, command_history
```

**Key Points:**
- One Green Agent process handles all tasks
- One Purple Agent process handles all tasks
- Each task gets its own Docker container
- Each task has isolated workspace and state
- Tasks run concurrently with full isolation

## Logging Architecture

### Log File Structure

```
logs/
  log_2026-01-13_16-14-49/
    agents.log           ← General agent logs (startup, initialization)
    arvo_14368.log       ← All logs for ARVO 14368 (green + purple)
    arvo_10055.log       ← All logs for ARVO 10055 (green + purple)
    arvo_20001.log       ← All logs for ARVO 20001 (green + purple)
```

**Note:** The directory is timestamped (`log_{timestamp}`), not per-ARVO. This allows multiple ARVO tasks from the same run to share the same log directory, with each ARVO having its own log file.

### Log Handler Flow

**Shared Log (`agents.log`):**
- Attached to root logger at startup
- Contains **only non-task-specific logs** (startup, initialization, general errors)
- Task-specific logs are **NOT** written to this file (to prevent duplication)
- Both agents write to this file for non-task logs

**Per-ARVO Logs (`arvo_{arvo_id}.log`):**
- Created dynamically when task is initialized
- Attached to agent-specific logger (green_agent or purple_agent)
- Contains **all logs related to that specific ARVO task**
- Both agents write to the same per-ARVO file (append mode)
- Handler removed when task finishes
- **Propagation is disabled** while task is active to prevent duplication

**Message Flow:**
```
# During task execution:
logger.info("Task message")
  ↓
Agent-specific logger (green_agent or purple_agent)
  ├─> Per-ARVO handler → arvo_{arvo_id}.log  ✅
  └─> (propagate=False) → NOT sent to root logger ❌

# When no tasks are running:
logger.info("Startup message")
  ↓
Agent-specific logger (green_agent or purple_agent)
  └─> (propagate=True) → Root logger
      ├─> Console handler → terminal
      └─> File handler → agents.log  ✅
```

## Isolation Guarantees

1. **Workspace Isolation**: Each task has its own workspace directory
2. **Container Isolation**: Each task has its own Docker container
3. **Context Isolation**: Each task has its own context_id and state
4. **LLM Isolation**: Each task has its own DefaultAgent instance and message history
5. **Token Tracking**: Each task tracks its own token usage independently
6. **Log Isolation**: Each ARVO task has its own log file

## Resource Management

### Docker Containers
- Created: When task is initialized
- Lifecycle: Runs for the entire task duration
- Cleanup: Stopped and removed when task finishes or server shuts down

### Workspace Directories
- Created: When task assets are prepared
- Lifecycle: Exists for the entire task duration
- Cleanup: Removed when task finishes or server shuts down

### Log Files
- Created: Per-ARVO log files created when task is initialized
- Lifecycle: Active while task is running
- Cleanup: Handler removed when task finishes (file remains for history)

## Concurrency Model

- **Agents**: Single process per agent type (green/purple)
- **Tasks**: Multiple tasks can run concurrently
- **Containers**: One container per task
- **Isolation**: Full isolation between tasks via context_id

## Communication

- **Green ↔ Purple**: HTTP/A2A protocol
- **Green ↔ Containers**: Docker exec API
- **Purple ↔ LLM**: HTTP API (OpenAI, etc.)

