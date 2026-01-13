# How to Send a Task

## Current Flow (Correct)

**Tasks start from GREEN AGENT** (port 9009), not Purple Agent!

```
RCAJudge/External → Green Agent (9009) → Purple Agent (9019) → Green Agent → ...
```

## Step 1: Start Both Agents

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


## Step 2: Send Task to Green Agent

### Option A: Using Python Script (Recommended)

```bash
cd agents/mini-swe-agent
source path.sh
python test_send_task_to_green.py <arvo_id>
python test_send_task_to_green.py 10055
```

This will:
1. Send minimal message (just `arvo:XXXXX`) to Green Agent (port 9009)
2. Green Agent extracts arvo_id
3. Green Agent calls `prepare_task_assets()` to fetch real codebase and error report
4. Green Agent prepares assets & initializes ARVO
5. Green Agent creates full task description and sends to Purple Agent
6. Purple Agent starts command loop

### Option B: Using curl

```bash
cd agents/mini-swe-agent
./test_curl_green.sh
```

Or manually:
```bash
curl -X POST http://127.0.0.1:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "text": "You are tasked with performing root cause analysis...\n\nTask ID: arvo:10055\n\nWorkspace Directory: /workspace/arvo_10055..."
        }]
      }
    },
    "id": 1
  }'
```

## Task Message Format

The Green Agent expects a **minimal message** with just the arvo_id:

```
Task ID: arvo:10055
```

**Note:** Green Agent will automatically:
- Extract `arvo:10055` from the message
- Call `prepare_task_assets()` to fetch real codebase and error report
- Prepare task assets (download codebase, crash report)
- Initialize ARVO container
- Create full task description with "Part I: tools" and "Part II: task"
- Send formatted version to Purple Agent

**No need to create the full prompt yourself** - Green Agent handles everything!

## What to Expect

### Green Agent Logs:
```
INFO:green_agent:Received message (context_id=...): You are tasked...
INFO:green_agent:Received task 10055 from RCAJudge
INFO:green_agent:Preparing assets for task 10055...
INFO:green_agent:Initializing ARVO container for task 10055...
INFO:green_agent:Sending task 10055 to purple agent...
```

### Purple Agent Logs:
```
INFO:purple_agent:Received message (context_id=...): Part I: Available Tools...
INFO:purple_agent:Detected task initialization message
INFO:purple_agent:Sending task initialization to green agent...
INFO:purple_agent:Task initialized successfully. Starting command decision loop...
INFO:purple_agent:Step 1/50 - Deciding next command...
INFO:purple_agent:Sending command to green agent: ls /workspace/src-vul
```

## Verify It's Working

1. **Check ARVO container was created:**
   ```bash
   docker ps | grep arvo
   ```

2. **Check workspace was created:**
   ```bash
   ls -la /tmp/rcabench/arvo_10055/
   ```

3. **Watch the command loop:**
   - Purple Agent sends commands
   - Green Agent executes them
   - Results flow back
   - Loop continues...

## Troubleshooting

### Green Agent says "Could not find arvo_id"
- Make sure message contains `arvo:XXXXX` or `Task ID: arvo:XXXXX`
- Check the message format matches RCAJudge format

### Purple Agent doesn't receive task
- Check Green Agent logs for "Sending task to purple agent..."
- Verify `PURPLE_AGENT_URL` is correct
- Check network connectivity

### ARVO container not created
- Check if ARVO image exists: `docker images | grep arvo`
- Verify Docker daemon is running
- Check Green Agent logs for errors

