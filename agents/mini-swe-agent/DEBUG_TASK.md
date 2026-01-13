# Debugging Task Initialization Issue

## Problem

When sending a task, you get:
```
Error: Task not initialized. Please send task description first.
```

## Root Cause Analysis

The error comes from `green_agent_server.py` line 190, which means:
1. Purple agent sent a command to green agent
2. Green agent received it but task context doesn't exist
3. This means green agent didn't receive/process the task initialization

## Flow Check

### Expected Flow:
1. ✅ Purple receives task with "Part I:" and "Part II:"
2. ✅ Purple sends full message to green agent
3. ❌ Green agent should receive and initialize task
4. ❌ Purple waits for green response
5. ❌ Purple starts command loop

### Actual Flow (Based on Error):
1. ✅ Purple receives task
2. ❓ Purple sends to green (did it work?)
3. ❌ Green doesn't have task context
4. ❌ Purple tries to send command
5. ❌ Green says "Task not initialized"

## Debug Steps

### 1. Check Purple Agent Logs

Look for:
```
INFO:purple_agent:Detected task initialization message
INFO:purple_agent:Sending task initialization to green agent...
INFO:purple_agent:Green agent response: ...
```

If you see "Green agent response: Error..." then green agent failed to initialize.

### 2. Check Green Agent Logs

Look for:
```
INFO:green_agent:Received message (context_id=...): Part I: Available Tools...
INFO:green_agent:Initializing task 10055 for context ...
```

If you DON'T see "Initializing task", then green agent didn't recognize the message.

### 3. Check Context IDs

The context_id must be the same for:
- Purple → Green task initialization
- Purple → Green command execution

If they're different, green agent won't find the task context.

## Quick Fix Test

Try sending the task again and watch both terminal logs. You should see:

**Purple Agent:**
```
INFO:purple_agent:Detected task initialization message
INFO:purple_agent:Sending task initialization to green agent...
INFO:purple_agent:Green agent response: Task initialized successfully...
INFO:purple_agent:Task initialized successfully. Starting command decision loop...
```

**Green Agent:**
```
INFO:green_agent:Received message (context_id=...): Part I: Available Tools...
INFO:green_agent:Initializing task 10055 for context ...
INFO:green_agent:Task 10055 initialized successfully
```

If you see different logs, that's where the problem is!

