# Testing Guide

## What Each Test Option Validates

### Option A: Using RCAJudge (Full Evaluation)

**Purpose:** Test the complete evaluation pipeline from task selection to metrics calculation.

**What it tests:**
1. **RCAJudge Integration**
   - Can RCAJudge send tasks to purple agent?
   - Does task format match what purple agent expects?
   - Can RCAJudge receive and evaluate results?

2. **Purple Agent Functionality**
   - Can purple agent receive tasks from RCAJudge?
   - Can purple agent parse task description?
   - Can purple agent decide commands using LLM?
   - Can purple agent communicate with green agent?

3. **Green Agent Functionality**
   - Can green agent receive command requests?
   - Can green agent validate commands?
   - Can green agent execute in ARVO containers?
   - Can green agent return results?

4. **ARVO Container Management**
   - Are ARVO containers created correctly?
   - Do commands execute properly?
   - Are containers cleaned up after task?

5. **End-to-End Flow**
   - Does the complete flow work?
   - Are submission files created?
   - Are metrics calculated correctly?

**Current Status:** ⚠️ **Requires Integration Work**
- RCAJudge sends a different message format than purple agent expects
- Need to either:
  - Update RCAJudge to send "Part I: tools, Part II: task" format
  - OR update purple agent to handle RCAJudge's current format

### Option B: Manual Test (Component Testing)

**Purpose:** Test individual components without full integration.

**What it tests:**
1. **Purple Agent A2A Server**
   - Is the A2A server running?
   - Can it receive messages?
   - Can it parse "Part I" and "Part II" format?

2. **Purple → Green Communication**
   - Can purple agent send commands to green agent?
   - Can purple agent receive results?
   - Is the A2A protocol working?

3. **Green Agent Command Execution**
   - Can green agent receive command requests?
   - Can green agent execute in ARVO containers?
   - Are results formatted correctly?

4. **Basic Functionality**
   - Does the command loop work?
   - Can tasks be initialized?
   - Can commands be executed?

**Current Status:** ✅ **Ready to Test**
- This should work immediately
- Tests the core functionality without full integration

## Recommended Testing Order

### Phase 1: Component Testing (Start Here)

1. **Test Green Agent Alone**
   ```bash
   # Start green agent
   python green_agent_server.py --port 9009
   
   # Send test command
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
   
   **What to verify:**
   - Green agent receives message
   - Task initializes
   - ARVO container is created
   - Can send "execute: ls /workspace" and get results

2. **Test Purple Agent Alone**
   ```bash
   # Start both agents
   python green_agent_server.py --port 9009 &
   python purple_agent_server.py --port 9019 --green-agent-url http://127.0.0.1:9009/
   
   # Send task to purple agent
   curl -X POST http://localhost:9019/ ...
   ```
   
   **What to verify:**
   - Purple agent receives task
   - Purple agent sends commands to green agent
   - Results flow back
   - Command loop continues

### Phase 2: Integration Testing

3. **Test with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   
   **What to verify:**
   - Both containers start
   - Network connectivity works
   - Commands execute correctly
   - Multiple tasks can run in parallel

### Phase 3: Full Integration

4. **Test with RCAJudge**
   - Update RCAJudge to send correct format
   - Run full evaluation
   - Verify metrics calculation

## What's Actually Being Tested

### Option A (RCAJudge) Tests:
- **System Integration:** All components working together
- **Evaluation Pipeline:** Complete task lifecycle
- **Metrics Accuracy:** IoU, accuracy calculations
- **Production Readiness:** Real-world usage scenario

### Option B (Manual) Tests:
- **Component Functionality:** Individual pieces work
- **A2A Protocol:** Communication between agents
- **Command Execution:** Green agent can run commands
- **Basic Flow:** Task → Commands → Results

## Current Gaps

1. **Message Format Mismatch**
   - RCAJudge sends: Plain task description
   - Purple agent expects: "Part I: tools... Part II: task..."
   - **Fix needed:** Update one or the other

2. **Task Initialization**
   - Purple agent expects task to include arvo_id
   - Green agent needs to prepare assets
   - **Fix needed:** Ensure green agent prepares assets before purple agent starts

3. **Submission Format**
   - Purple agent creates loc.json
   - RCAJudge expects loc.json + reasoning.json
   - **Fix needed:** Purple agent should create both files

## Quick Test Checklist

- [ ] Green agent starts and responds to agent card request
- [ ] Purple agent starts and responds to agent card request
- [ ] Green agent can initialize a task
- [ ] Green agent can execute a command in ARVO container
- [ ] Purple agent can send command to green agent
- [ ] Purple agent can receive results from green agent
- [ ] Command loop continues (purple → green → results → purple → next command)
- [ ] Task completion is detected
- [ ] Submission files are created
- [ ] ARVO containers are cleaned up

