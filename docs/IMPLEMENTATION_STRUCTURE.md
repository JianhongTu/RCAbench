# Implementation Structure: Green Agent Redesign

## Current Architecture Understanding

### AgentBeats SDK (`src/agentbeats/`)
- **`client.py`**: A2A protocol client - `send_message()` function for communicating with agents
- **`tool_provider.py`**: Wrapper for green agent to talk to purple agent - `talk_to_agent()` method
- **`green_executor.py`**: Wraps `GreenAgent` for A2A protocol - handles incoming requests
- **`models.py`**: `EvalRequest` and `EvalResult` models

### Scenario Code (`scenarios/arvo_rca/`)
- **`rca_judge.py`**: Green agent - implements `GreenAgent` interface
- **`rca_judge_common.py`**: Shared models and utilities for green agent
- **`rca_finder.py`**: Purple agent - receives tasks via A2A, currently runs OpenHands

### Current Communication Flow

```
┌─────────────────────────────────────────────┐
│         Green Agent (rca_judge.py)          │
│  - Uses ToolProvider.talk_to_agent()        │
│  - Sends task description                   │
│  - Polls for files (loc.json, reasoning.json)│
└─────────────────────────────────────────────┘
              ↓ A2A Protocol (one-way send)
┌─────────────────────────────────────────────┐
│        Purple Agent (rca_finder.py)          │
│  - Receives task in handle_task()            │
│  - Runs OpenHands independently             │
│  - Writes files to shared directory          │
└─────────────────────────────────────────────┘
```

## New Communication Flow (Tool Calling)

```
┌─────────────────────────────────────────────┐
│         Green Agent (rca_judge.py)          │
│  - Sends task description (initial)         │
│  - NEW: Receives tool call requests         │
│  - NEW: Executes tools in sandbox            │
│  - NEW: Returns tool results                │
│  - NEW: Manages turn loop                   │
└─────────────────────────────────────────────┘
              ↕ A2A Protocol (bidirectional)
┌─────────────────────────────────────────────┐
│        Purple Agent (rca_finder.py)          │
│  - Receives task description                │
│  - NEW: Sends tool call requests             │
│  - NEW: Receives tool results                │
│  - NEW: Decides next tool to call           │
│  - NEW: Submits final results               │
└─────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Green Agent - Tool Execution Infrastructure

**File: `scenarios/arvo_rca/rca_judge.py`**

#### 1.1 Add Tool Registry
```python
class ToolRegistry:
    """Registry of available tools for purple agent"""
    def __init__(self):
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        # Register all tools (read_file, grep, etc.)
        pass
    
    def get_tool(self, name: str) -> Tool:
        return self.tools.get(name)
    
    def list_tools(self) -> list[dict]:
        # Return tool definitions for purple agent
        pass
```

#### 1.2 Add Sandbox Manager
```python
class DockerSandbox:
    """Manages Docker container for tool execution"""
    def __init__(self, arvo_id: str, workspace_dir: Path):
        self.arvo_id = arvo_id
        self.workspace_dir = workspace_dir
        self.image = f"n132/arvo:{arvo_id}-vul"
        self.container = None
        
    async def start(self):
        """Start long-lived container"""
        # Mount workspace, start container
        
    async def execute_command(self, cmd: list[str], cwd: str = "/workspace") -> dict:
        """Execute command in container"""
        # Run command, return stdout/stderr/exit_code
        
    async def read_file(self, path: str) -> str:
        """Read file from container"""
        # Use docker exec to read file
        
    async def cleanup(self):
        """Remove container"""
```

#### 1.3 Add Turn Manager
```python
class TurnManager:
    """Manages turn counting and limits"""
    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turn_count = 0
        self.history = []
        
    def can_continue(self) -> bool:
        return self.turn_count < self.max_turns
        
    def record_turn(self, tool_call: dict, result: dict):
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "tool": tool_call.get("tool"),
            "success": result.get("success"),
        })
```

#### 1.4 Add Tool Loop Manager
```python
class ToolLoopManager:
    """Manages the Mini-SWE style tool calling loop"""
    def __init__(self, sandbox: DockerSandbox, tool_registry: ToolRegistry, turn_manager: TurnManager):
        self.sandbox = sandbox
        self.tools = tool_registry
        self.turns = turn_manager
        
    async def execute_tool_call(self, tool_call: dict) -> dict:
        """Execute a single tool call"""
        # Validate tool call
        # Check turn limit
        # Execute tool in sandbox
        # Record turn
        # Return result
        
    def check_end_condition(self, shared_dir: Path) -> dict:
        """Check if loop should end"""
        # Check for submission files
        # Check turn limit
        # Check timeout
        # Return end condition status
```

#### 1.5 Modify `RCAJudge._process_task()`

**Current flow:**
```python
async def _process_task(...):
    # 1. Prepare assets
    # 2. Send task description to purple agent
    # 3. Wait for files (polling)
    # 4. Evaluate results
```

**New flow:**
```python
async def _process_task(...):
    # 1. Prepare assets
    # 2. Create sandbox environment
    # 3. Send task description + tool list to purple agent
    # 4. NEW: Enter tool calling loop
    #    - Receive tool call requests from purple agent
    #    - Execute tools in sandbox
    #    - Return results to purple agent
    #    - Check end conditions
    # 5. Evaluate results
    # 6. Cleanup sandbox
```

**Key change:** Instead of one-way `talk_to_agent()` and polling, we need bidirectional communication loop.

### Phase 2: Communication Protocol

#### 2.1 Tool Call Message Format

**Purple Agent → Green Agent:**
```json
{
  "type": "tool_call",
  "tool": "read_file",
  "arguments": {
    "path": "src-vul/main.c"
  },
  "reasoning": "Reading main file to understand entry point"
}
```

**Green Agent → Purple Agent:**
```json
{
  "type": "tool_result",
  "success": true,
  "result": "...",
  "turn": 1,
  "turns_remaining": 49
}
```

#### 2.2 Modify Green Agent Communication

**Option A: Extend `ToolProvider` (Recommended)**
```python
class ToolProvider:
    async def talk_to_agent(self, message: str, url: str, ...):
        # Existing: Send message, get response
        
    async def wait_for_tool_call(self, url: str, context_id: str, timeout: int = 30) -> dict:
        # NEW: Wait for tool call request from purple agent
        # Poll or use A2A event stream
        
    async def send_tool_result(self, result: dict, url: str, context_id: str):
        # NEW: Send tool result back to purple agent
```

**Option B: Use A2A Event Stream**
- Green agent subscribes to events from purple agent
- Purple agent sends tool calls as events
- Green agent responds with tool results

#### 2.3 Initial Task Message

**Green Agent sends to Purple Agent:**
```
Task: arvo:10055

Workspace: /workspace
- Error report: /workspace/10055_error.txt
- Codebase: /workspace/src-vul/

Available Tools:
1. read_file(path: str) - Read file contents
2. grep(pattern: str, path: str) - Search for pattern
3. list_directory(path: str) - List files
...

You can request tools by sending:
{"type": "tool_call", "tool": "read_file", "arguments": {"path": "..."}}

I will execute the tool and return results.
```

### Phase 3: Purple Agent Simplification

**File: `scenarios/arvo_rca/rca_finder.py`**

#### 3.1 Remove OpenHands Dependency
- Remove all OpenHands-related code
- Remove Poetry/dependency management
- Keep only A2A protocol communication

#### 3.2 Add Tool Calling Logic
```python
class RCAFinder:
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.api_key = api_key
        self.llm_client = OpenAI(api_key=api_key)
        self.tool_definitions = []  # Will be provided by green agent
        self.conversation_history = []
        
    async def handle_task(self, context: RequestContext, event_queue: EventQueue):
        # 1. Parse task description and tool list from green agent
        # 2. Initialize conversation with LLM
        # 3. Loop:
        #    - Decide which tool to call (using LLM)
        #    - Send tool call request to green agent
        #    - Receive tool result
        #    - Update conversation history
        #    - Check if should submit results
        # 4. Submit final results
```

#### 3.3 LLM Tool Calling
- Use LLM function calling (OpenAI/Anthropic) to decide which tool to call
- Format tool definitions from green agent
- Parse LLM response to extract tool call
- Send structured tool call to green agent

### Phase 4: Integration Points

#### 4.1 Green Agent Modifications

**In `RCAJudge.__init__()`:**
```python
def __init__(self, ...):
    # Existing
    self._tool_provider = ToolProvider()
    
    # NEW
    self._tool_registry = ToolRegistry()
    self._sandbox_manager = None  # Created per task
    self._turn_manager = None  # Created per task
    self._loop_manager = None  # Created per task
```

**In `RCAJudge._process_task()`:**
```python
async def _process_task(...):
    # ... existing asset preparation ...
    
    # NEW: Create sandbox
    sandbox = DockerSandbox(arvo_id, workspace_dir)
    await sandbox.start()
    
    # NEW: Initialize managers
    turn_manager = TurnManager(max_turns=req.config.get("max_turns", 50))
    loop_manager = ToolLoopManager(sandbox, self._tool_registry, turn_manager)
    
    # Send initial task description with tool list
    task_message = self._create_task_message_with_tools(
        arvo_id, workspace_dir, self._tool_registry.list_tools()
    )
    
    response = await self._tool_provider.talk_to_agent(
        task_message,
        participant_endpoint,
        new_conversation=True,
    )
    
    context_id = self._tool_provider._context_ids[participant_endpoint]
    
    # NEW: Tool calling loop
    while True:
        # Wait for tool call from purple agent
        tool_call = await self._tool_provider.wait_for_tool_call(
            participant_endpoint, context_id, timeout=60
        )
        
        # Execute tool
        result = await loop_manager.execute_tool_call(tool_call)
        
        # Send result back
        await self._tool_provider.send_tool_result(
            result, participant_endpoint, context_id
        )
        
        # Check end conditions
        end_condition = loop_manager.check_end_condition(shared_dir)
        if end_condition["status"] != "continue":
            break
    
    # ... existing evaluation ...
    
    # Cleanup
    await sandbox.cleanup()
```

#### 4.2 Purple Agent Modifications

**In `RCAFinder.handle_task()`:**
```python
async def handle_task(self, context: RequestContext, event_queue: EventQueue):
    # Parse task and tool list from message
    task_info = self._parse_task_message(context.message)
    tool_definitions = task_info["tools"]
    
    # Initialize LLM conversation
    messages = [
        {"role": "system", "content": "You are an RCA analyst. Use tools to analyze vulnerabilities."},
        {"role": "user", "content": task_info["description"]}
    ]
    
    # Tool calling loop
    while True:
        # Get LLM response with tool calling
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[self._format_tool_definition(t) for t in tool_definitions],
            tool_choice="auto"
        )
        
        # Check if LLM wants to call a tool
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            
            # Send tool call to green agent
            # (Need to implement: how purple agent sends to green agent)
            # This is the tricky part - purple agent needs to send back to green agent
            
            # Receive tool result
            # Update conversation with result
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "name": tool_call.function.name,
                "content": tool_result
            })
        else:
            # LLM wants to submit results
            submission = self._extract_submission(response.choices[0].message.content)
            if submission:
                # Submit via tool call
                break
```

**Challenge:** Purple agent needs to send messages back to green agent. Currently, A2A protocol is one-way (green → purple). We need to establish bidirectional communication.

### Phase 5: Bidirectional Communication

#### Option A: Green Agent Polls Purple Agent
- Green agent periodically calls `talk_to_agent()` asking "Do you have a tool call?"
- Purple agent responds with tool call or "waiting"
- Not ideal - polling overhead

#### Option B: Purple Agent Calls Green Agent
- Purple agent uses `ToolProvider` (or similar) to call green agent
- Green agent exposes tool execution endpoint
- More natural but requires green agent to be callable

#### Option C: A2A Event Stream (Best)
- Use A2A's event streaming capabilities
- Purple agent sends tool calls as events
- Green agent subscribes to events
- Most efficient and natural

**Implementation:**
```python
# In green agent
async def _process_task_with_tool_loop(...):
    # Start event stream from purple agent
    async for event in self._tool_provider.stream_events(participant_endpoint, context_id):
        if event.type == "tool_call":
            result = await loop_manager.execute_tool_call(event.data)
            await self._tool_provider.send_tool_result(result, ...)
        elif event.type == "submission":
            break
```

## File Structure

```
scenarios/arvo_rca/
├── rca_judge.py              # Green agent (modify)
├── rca_judge_common.py       # Shared models (extend)
├── rca_finder.py             # Purple agent (simplify)
└── tools/                    # NEW: Tool implementations
    ├── __init__.py
    ├── registry.py           # Tool registry
    ├── sandbox.py            # Docker sandbox manager
    ├── turn_manager.py      # Turn counting
    ├── loop_manager.py      # Tool loop manager
    └── tools/                # Individual tool implementations
        ├── file_ops.py       # read_file, list_directory, etc.
        ├── code_search.py    # grep, find_files
        ├── error_analysis.py # read_error_report, parse_stack_trace
        └── submission.py      # submit_localization, submit_reasoning
```

## Next Steps

1. **Investigate A2A Event Streaming**: Check if A2A SDK supports bidirectional event streams
2. **Design Tool Call Protocol**: Finalize message format for tool calls
3. **Implement Tool Registry**: Start with basic tools (read_file, grep)
4. **Implement Sandbox**: Docker container management
5. **Test Communication**: Verify bidirectional communication works
