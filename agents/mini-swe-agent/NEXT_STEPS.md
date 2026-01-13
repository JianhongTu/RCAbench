# Next Steps - Implementation Checklist

## Step 1: Fix Dockerfile Dependencies

The Dockerfiles need to install `rcabench` and `agentbeats` packages. Update both Dockerfiles:

### Update `Dockerfile` (Green Agent):

```dockerfile
# After line 17 (a2a-sdk installation), add:
RUN pip install --no-cache-dir \
    # ... existing packages ...
    # Add these lines to copy and install rcabench/agentbeats:
    && mkdir -p /app/src

# Copy rcabench and agentbeats source code
COPY ../../src /app/src/
COPY ../../rcabench /app/rcabench/

# Or if installed as packages:
# RUN pip install -e /app/src/agentbeats
# RUN pip install -e /app/rcabench
```

### Update `Dockerfile.purple` (Purple Agent):

```dockerfile
# After line 12 (a2a-sdk installation), add:
RUN pip install --no-cache-dir \
    # ... existing packages ...
    && mkdir -p /app/src

# Copy agentbeats source code (purple agent only needs client)
COPY ../../src/agentbeats /app/src/agentbeats/
```

**Alternative:** Mount the source code as volumes in `docker-compose.yml` instead of copying.

## Step 2: Update docker-compose.yml to Mount Source Code

Add volume mounts for rcabench and agentbeats:

```yaml
services:
  green-agent:
    volumes:
      # ... existing volumes ...
      - ../../src:/app/src:ro  # Mount agentbeats
      - ../../rcabench:/app/rcabench:ro  # Mount rcabench
      - ../../src/rcabench:/app/src/rcabench:ro  # If rcabench is in src/

  purple-agent:
    volumes:
      # ... existing volumes ...
      - ../../src/agentbeats:/app/src/agentbeats:ro  # Mount agentbeats client
```

## Step 3: Test Green Agent Locally

```bash
# 1. Install dependencies locally
pip install a2a-sdk uvicorn openai docker

# 2. Set PYTHONPATH to include rcabench and agentbeats
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../src:$(pwd)/../../rcabench"

# 3. Test green agent
cd agents/mini-swe-agent
python green_agent_server.py --host 127.0.0.1 --port 9009 --tmp-dir /tmp/rcabench

# 4. In another terminal, test with curl:
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Part I: Tools: ls, cat\nPart II: Task arvo:10055"}]
      }
    },
    "id": 1
  }'
```

## Step 4: Test Purple Agent Locally

```bash
# 1. Start green agent first (from Step 3)

# 2. In another terminal, start purple agent
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../src"
export OPENAI_API_KEY="your-api-key"
python purple_agent_server.py \
  --host 127.0.0.1 \
  --port 9019 \
  --green-agent-url http://127.0.0.1:9009/ \
  --model gpt-4o-mini

# 3. Test with curl:
curl -X POST http://localhost:9019/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Part I: Tools: ls, cat, grep\nPart II: Analyze task arvo:10055"}]
      }
    },
    "id": 1
  }'
```

## Step 5: Test with Docker Compose

```bash
# 1. Update docker-compose.yml with volume mounts (Step 2)

# 2. Set environment variables
export OPENAI_API_KEY="your-api-key"
export WORKSPACE_DIR="./workspace"
export LOG_DIR="./logs"
export TMP_DIR="/tmp/rcabench"

# 3. Build and start
cd agents/mini-swe-agent
docker-compose up --build

# 4. Check logs
docker-compose logs green-agent
docker-compose logs purple-agent
```

## Step 6: Integrate with RCAJudge

Update `scenarios/arvo_rca/rca_judge.py` to send tasks in the correct format:

The RCAJudge should send messages to purple agent with:
- Part I: List of available tools
- Part II: Task instruction with arvo_id

Example message format:
```
Part I: Available Tools
You can use: ls, cat, grep, find, gcc, make, arvo, etc.

Part II: Task Instruction
You are analyzing a vulnerable codebase...
Task ID: arvo:10055
Workspace: /workspace/arvo_10055/
Crash report: /workspace/arvo_10055/10055_error.txt
```

## Step 7: Fix Import Paths

Ensure all imports work correctly:

### In `green_agent_server.py`:
- ✅ Already has try/except for imports
- ✅ Adds parent directory to sys.path
- ⚠️ May need to adjust path based on deployment

### In `purple_agent_server.py`:
- ✅ Uses `agentbeats.client` which should be available
- ⚠️ May need to add sys.path manipulation if running in container

## Step 8: Test End-to-End with Real Task

```bash
# 1. Start both agents (Step 5)

# 2. Use RCAJudge to send a task
# The RCAJudge should:
#   - Pick an arvo_id (e.g., 10055)
#   - Prepare task assets
#   - Send to purple agent at http://localhost:9019/
#   - Wait for loc.json submission

# 3. Monitor the flow:
#   - Purple agent receives task
#   - Purple agent sends commands to green agent
#   - Green agent executes in ARVO container
#   - Results flow back
#   - Purple agent creates loc.json
```

## Step 9: Handle Edge Cases

### 9.1 Error Handling
- [ ] Handle ARVO container creation failures
- [ ] Handle command execution timeouts
- [ ] Handle LLM API errors
- [ ] Handle network failures between agents

### 9.2 Task Cleanup
- [ ] Ensure ARVO containers are cleaned up on errors
- [ ] Clean up workspace directories
- [ ] Handle partial task completion

### 9.3 Multiple Tasks
- [ ] Test concurrent task execution
- [ ] Verify context isolation
- [ ] Test resource limits

## Step 10: Update Documentation

- [ ] Update main README with A2A setup instructions
- [ ] Add troubleshooting section
- [ ] Document environment variables
- [ ] Add example usage

## Step 11: Integration Testing

Create a test script:

```python
# test_integration.py
import asyncio
from agentbeats.client import send_message

async def test_flow():
    # Send task to purple agent
    response = await send_message(
        message="Part I: Tools: ls, cat\nPart II: Task arvo:10055",
        base_url="http://localhost:9019/",
        context_id=None
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_flow())
```

## Step 12: Production Deployment

- [ ] Set up proper logging
- [ ] Configure resource limits
- [ ] Set up monitoring
- [ ] Configure health checks
- [ ] Set up auto-restart policies

## Current Status

✅ **Completed:**
- Green agent A2A server implementation
- Purple agent A2A server implementation
- Docker Compose configuration
- Dockerfiles for both agents
- Architecture documentation

⚠️ **Needs Work:**
- Dockerfile dependencies (rcabench/agentbeats)
- Import path fixes for containerized deployment
- Integration with RCAJudge
- End-to-end testing
- Error handling improvements

## Quick Start (After Fixes)

```bash
# 1. Fix Dockerfiles (Step 1-2)
# 2. Set environment variables
export OPENAI_API_KEY="your-key"
export WORKSPACE_DIR="./workspace"
export LOG_DIR="./logs"

# 3. Start services
docker-compose up --build

# 4. Test
# Use RCAJudge or send test messages to purple agent
```

