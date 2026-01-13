# Current Flow (WRONG - Needs Fix)

## Current Implementation

```
1. RCAJudge (Evaluator)
   ↓ Sends task description
2. Purple Agent (mini-swe-agent)
   ↓ Receives task, sends to green agent
3. Green Agent (mini-swe-agent)
   ↓ Initializes ARVO container
4. Purple Agent
   ↓ Decides commands, sends to green
5. Green Agent
   ↓ Executes in ARVO container
```

**Problem:** Tasks start at Purple Agent, but they should start at Green Agent!

## Correct Flow (What It Should Be)

```
1. RCAJudge (Evaluator)
   ↓ Sends task to Green Agent
2. Green Agent (mini-swe-agent)
   ↓ Receives task, prepares assets, initializes ARVO container
   ↓ Sends task to Purple Agent (with Part I: tools, Part II: task)
3. Purple Agent (mini-swe-agent)
   ↓ Receives task from Green Agent
   ↓ Decides commands using LLM
   ↓ Sends commands to Green
4. Green Agent
   ↓ Executes commands in ARVO container
   ↓ Returns results to Purple
5. Purple Agent
   ↓ Analyzes results, decides next command
   ↓ Loop continues...
```

## Why Green Agent Should Start?

1. **Green Agent controls ARVO containers** - It needs to initialize first
2. **Green Agent prepares task assets** - Downloads codebase, crash reports
3. **Green Agent is the orchestrator** - It manages the task lifecycle
4. **Purple Agent is the participant** - It just decides commands

## What Needs to Change

1. ✅ Green Agent should receive tasks from RCAJudge (not Purple)
2. ✅ Green Agent should prepare assets and initialize ARVO
3. ✅ Green Agent should send formatted task to Purple Agent
4. ✅ Purple Agent should receive from Green (not RCAJudge)
5. ✅ Communication flow: RCAJudge → Green → Purple → Green → Purple → ...

