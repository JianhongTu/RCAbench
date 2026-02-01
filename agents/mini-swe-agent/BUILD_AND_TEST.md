# Building and Testing Docker Images

## Prerequisites

```bash
# Ensure Docker is running
docker ps

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Building Images

The Dockerfiles expect to be built from the **repository root**, not from `agents/mini-swe-agent/`.

### Option 1: Build from Repository Root (Recommended for Production)

```bash
# Navigate to repository root
cd /Users/rmur/RCAbench

# Build green agent image
docker build -f agents/mini-swe-agent/Dockerfile.green -t mini-swe-agent-green:latest .

# Build purple agent image
docker build -f agents/mini-swe-agent/Dockerfile.purple -t mini-swe-agent-purple:latest .
```

### Option 2: Using Docker Compose (For Development)

The `docker-compose.yml` currently uses `context: .` which builds from `agents/mini-swe-agent/`. 

**Update docker-compose.yml** to build from repo root:

```yaml
services:
  green-agent:
    build:
      context: ../..  # Build from repo root
      dockerfile: agents/mini-swe-agent/Dockerfile.green
    # ... rest of config
```

Then run:
```bash
cd agents/mini-swe-agent
docker-compose up --build
```

## Testing Locally

### Step 1: Start Both Agents

#### Using Docker Compose (Recommended)
```bash
cd agents/mini-swe-agent
export OPENAI_API_KEY="your-api-key"
docker-compose up
```

#### Using Docker Run (Manual)
```bash
# Terminal 1: Start green agent
docker run -it --rm \
  --name green-agent \
  -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/workspace:/workspace \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  mini-swe-agent-green:latest \
  --host 0.0.0.0 --port 9009

# Terminal 2: Start purple agent
docker run -it --rm \
  --name purple-agent \
  -p 9019:9019 \
  --network host \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e GREEN_AGENT_URL="http://localhost:9009/" \
  mini-swe-agent-purple:latest \
  --host 0.0.0.0 --port 9019 --green-agent-url http://localhost:9009/
```

### Step 2: Verify Agents Are Running

```bash
# Check green agent
curl http://localhost:9009/.well-known/agent-card.json

# Check purple agent
curl http://localhost:9019/.well-known/agent-card.json
```

### Step 3: Test with EvalRequest

Create a test script `test_eval_request.py`:

```python
#!/usr/bin/env python3
"""Test script to send EvalRequest to green agent."""
import asyncio
import json
from agentbeats.client import send_message

async def test_eval_request():
    # Create EvalRequest (simulating what AgentBeats would send)
    eval_request = {
        "participants": {
            "purple_agent": "http://localhost:9019/"
        },
        "config": {
            "task_ids": ["10055"],  # Use a simple test task
            "num_tasks": 1,
            "max_task_time": 600
        }
    }
    
    print("Sending EvalRequest to green agent...")
    print(json.dumps(eval_request, indent=2))
    
    # Send to green agent
    response = await send_message(
        message=json.dumps(eval_request),
        base_url="http://localhost:9009/",
        context_id=None,
    )
    
    print("\nResponse:")
    print(response.get("response", "No response"))

if __name__ == "__main__":
    asyncio.run(test_eval_request())
```

Run the test:
```bash
cd agents/mini-swe-agent
python test_eval_request.py
```

### Step 4: Check Logs

```bash
# View green agent logs
docker logs green-agent

# View purple agent logs  
docker logs purple-agent

# Or if using docker-compose
docker-compose logs -f
```

## Expected Flow

1. **Green agent receives EvalRequest**
   - Logs: `[GREEN] Processing X tasks from EvalRequest`
   - Creates task context for each task
   
2. **Green agent sends task to purple agent**
   - Logs: `[GREEN] Sending task description to purple agent`
   
3. **Purple agent receives task and starts analysis**
   - Logs: `[PURPLE] Task initialized. Starting command loop...`
   
4. **Purple agent sends commands to green agent**
   - Logs: `[PURPLE] Command to run: execute: <command>`
   
5. **Green agent executes commands**
   - Logs: `[GREEN] Command executed. ...`
   
6. **Purple agent creates loc.json**
   - Logs: `[PURPLE] Task finished`
   
7. **Green agent evaluates and adds artifact**
   - Logs: `[GREEN] Added A2A artifact with evaluation results`
   - Final artifact with `Result` name should be added

## Troubleshooting

### "Module not found: rcabench" or "agentbeats"
- Ensure Dockerfiles are built from repo root
- Check that `src/` directory was copied correctly in image
- Verify `PYTHONPATH` includes `/tmp/src`

### "Cannot connect to Docker daemon"
- Ensure Docker socket is mounted: `-v /var/run/docker.sock:/var/run/docker.sock`
- On Linux, check socket permissions

### "Purple agent cannot reach green agent"
- Ensure both containers are on same network
- Use `--network host` for manual docker run
- Use `docker-compose` which creates shared network automatically

### "Task context not found" when purple sends commands
- Verify green agent created task context before purple sends commands
- Check that `task_context_id` is consistent between green and purple

## Quick Test Checklist

- [ ] Both images build successfully
- [ ] Both containers start without errors
- [ ] Agent cards are accessible (`/.well-known/agent-card.json`)
- [ ] EvalRequest is accepted by green agent
- [ ] Green agent processes tasks and sends to purple
- [ ] Purple agent receives task and sends commands
- [ ] Green agent executes commands and returns results
- [ ] Purple agent creates loc.json
- [ ] Green agent evaluates results and adds final artifact
- [ ] Artifact contains `Result` name with summary and data

