# How to Test - Step by Step

## Current Status

✅ **Both agents are running:**
- Green agent: `http://127.0.0.1:9009` ✅
- Purple agent: `http://127.0.0.1:9019` ✅

❌ **No task has been sent yet** - They're just waiting for messages

## Step 1: Send a Test Task

### Option A: Using Python Script (Recommended)

In a **new terminal** (keep agents running):

```bash
cd agents/mini-swe-agent
python test_send_task.py
```

This will:
1. Send a task to purple agent
2. Purple agent will forward it to green agent
3. Green agent will initialize ARVO container
4. Purple agent will start deciding commands
5. You'll see the command loop in action

### Option B: Using curl

```bash
cd agents/mini-swe-agent
./test_curl.sh
```

Or manually:

```bash
curl -X POST http://127.0.0.1:9019/ \
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

## Step 2: Watch the Logs

After sending the task, you should see activity in both terminals:

**Purple Agent Terminal:**
```
INFO:purple_agent:Received message (context_id=...): Part I: Available Tools...
INFO:purple_agent:Step 1/50 - Deciding next command...
INFO:purple_agent:Sending command to green agent: ls /workspace/src-vul
```

**Green Agent Terminal:**
```
INFO:green_agent:Received message (context_id=...): Part I: Available Tools...
INFO:green_agent:Initializing task 10055 for context ...
INFO:green_agent:Executing command for task 10055: ls /workspace/src-vul
```

## Step 3: What to Expect

### Immediate Response (from curl/Python script):
- You'll get a response like: "Task initialized successfully. ARVO container ready..."

### In the Logs:
1. **Green agent:**
   - Receives task initialization
   - Creates ARVO container
   - Ready to execute commands

2. **Purple agent:**
   - Receives task
   - Sends to green agent
   - LLM decides first command (e.g., "ls /workspace/src-vul")
   - Sends command to green agent
   - Receives results
   - Decides next command
   - Loop continues...

## Step 4: Verify It's Working

Check if ARVO container was created:

```bash
docker ps | grep arvo
```

You should see a container like:
```
CONTAINER ID   IMAGE                    STATUS
abc123def456   n132/arvo:10055-vul      Up 2 minutes
```

## Troubleshooting

### No response from agents?

1. **Check if agents are still running:**
   ```bash
   # Should see both processes
   ps aux | grep -E "(green_agent|purple_agent)"
   ```

2. **Check if ports are listening:**
   ```bash
   lsof -i :9009  # Green agent
   lsof -i :9019  # Purple agent
   ```

3. **Check for errors in logs:**
   - Look for ERROR or Exception messages
   - Check if ARVO image exists: `docker images | grep arvo`

### Purple agent can't reach green agent?

```bash
# Test connectivity
curl http://127.0.0.1:9009/.well-known/agent-card.json
```

### Task initialization fails?

- Check if arvo_id is valid (e.g., 10055)
- Check if ARVO Docker image exists: `n132/arvo:10055-vul`
- Check workspace directory permissions

## Expected Flow After Sending Task

```
1. [You] → Send task to Purple Agent (port 9019)
2. [Purple] → Receives task, parses Part I and Part II
3. [Purple] → Sends full task to Green Agent (port 9009)
4. [Green] → Receives task, extracts arvo_id (10055)
5. [Green] → Prepares task assets (downloads codebase, crash report)
6. [Green] → Creates ARVO container (n132/arvo:10055-vul)
7. [Green] → Returns: "Task initialized successfully..."
8. [Purple] → LLM decides: "I should list files first"
9. [Purple] → Sends: "execute: ls /workspace/src-vul"
10. [Green] → Executes in ARVO container
11. [Green] → Returns: "<returncode>0</returncode><output>file1.c, file2.c...</output>"
12. [Purple] → Receives results, LLM analyzes
13. [Purple] → Decides next command...
14. [Loop continues until task complete]
```

## Quick Test Command

Run this in a new terminal while agents are running:

```bash
cd /Users/shubham.gaur/Documents/RCAbench/agents/mini-swe-agent
python test_send_task.py
```

Then watch both agent terminals for activity!

