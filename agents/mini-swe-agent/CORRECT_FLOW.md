# Correct Flow - Tasks Start from Green Agent ✅

## Updated Flow

```
1. RCAJudge (Evaluator Green Agent)
   ↓ Sends task description (contains arvo_id, workspace, crash report)
   ↓
2. Green Agent (mini-swe-agent) - Port 9009
   ↓ Receives task from RCAJudge
   ↓ Extracts arvo_id
   ↓ Prepares task assets (downloads codebase, crash report)
   ↓ Initializes ARVO container
   ↓ Formats task with "Part I: tools" and "Part II: task instruction"
   ↓ Sends formatted task to Purple Agent
   ↓
3. Purple Agent (mini-swe-agent) - Port 9019
   ↓ Receives formatted task from Green Agent
   ↓ LLM decides first command (e.g., "ls /workspace/src-vul")
   ↓ Sends command to Green Agent: "execute: ls /workspace/src-vul"
   ↓
4. Green Agent
   ↓ Receives command from Purple Agent
   ↓ Validates command
   ↓ Executes in ARVO container
   ↓ Returns results: "<returncode>0</returncode><output>...</output>"
   ↓
5. Purple Agent
   ↓ Receives command results
   ↓ LLM analyzes results
   ↓ Decides next command
   ↓ Loop continues...
   ↓
6. Purple Agent completes analysis
   ↓ Creates loc.json submission file
   ↓ Sends "[TASK FINISHED]" to Green Agent
   ↓
7. Green Agent
   ↓ Cleans up ARVO container
   ↓ Returns "[TASK COMPLETED]" to RCAJudge
   ↓
8. RCAJudge
   ↓ Evaluates results (IoU, accuracy, etc.)
   ↓ Reports metrics
```

## Key Changes

1. ✅ **Tasks start from Green Agent** - RCAJudge sends to Green, not Purple
2. ✅ **Green Agent prepares everything** - Assets, ARVO container, task formatting
3. ✅ **Green Agent sends to Purple** - With "Part I:" and "Part II:" format
4. ✅ **Purple Agent only decides commands** - No task preparation logic
5. ✅ **Green Agent orchestrates** - Controls task lifecycle

## Configuration

**Green Agent:**
```bash
python green_agent_server.py \
  --port 9009 \
  --purple-agent-url http://127.0.0.1:9019/
```

**Purple Agent:**
```bash
python purple_agent_server.py \
  --port 9019 \
  --green-agent-url http://127.0.0.1:9009/
```

## Message Flow

### 1. RCAJudge → Green Agent
```
Message: "You are tasked with performing root cause analysis...
Task ID: arvo:10055
Workspace Directory: /workspace/arvo_10055
Fuzzer Crash Report: [crash details]"
```

### 2. Green Agent → Purple Agent
```
Message: "Part I: Available Tools
You can use: ls, cat, grep, find, gcc, make, arvo...

Part II: Task Instruction
You are analyzing a vulnerable codebase...
Task ID: arvo:10055
Workspace: /workspace/arvo_10055
Crash Report: [crash details]"
```

### 3. Purple Agent → Green Agent
```
Message: "execute: ls /workspace/src-vul"
```

### 4. Green Agent → Purple Agent
```
Message: "<returncode>0</returncode>
<output>
file1.c, file2.c, file3.c
</output>"
```

## Benefits

1. **Clear separation of concerns:**
   - Green Agent: Infrastructure & orchestration
   - Purple Agent: Analysis & decision-making

2. **Green Agent controls lifecycle:**
   - Creates ARVO containers
   - Manages task state
   - Validates commands
   - Cleans up resources

3. **Purple Agent focuses on analysis:**
   - Receives ready-to-use task
   - Decides commands
   - Analyzes results
   - Creates submissions

