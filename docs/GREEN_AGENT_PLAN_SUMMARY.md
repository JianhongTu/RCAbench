# Green Agent Redesign: Implementation Summary

## Key Decisions & Answers

### 1. Mini-SWE Integration ✅
**Understanding**: Mini-SWE is a minimal loop (~100 lines) that:
- Facilitates tool calling for agents
- Provides the loop structure
- **You bring your own environment and tools**

**Our Implementation**:
- Green agent implements the Mini-SWE loop pattern
- We provide our own Docker-based sandbox environment
- We define our own tool registry

### 2. Tool Registry ✅

**Core Tools for RCA Analysis:**

#### File Operations (Essential)
1. `read_file(path: str) -> str` - Read file contents
2. `list_directory(path: str, recursive: bool = False) -> list[str]` - List files
3. `read_file_lines(path: str, start: int, end: int) -> str` - Read specific lines

#### Code Search (Essential)
4. `grep(pattern: str, path: str, recursive: bool = True) -> list[dict]` - Search pattern in files
5. `find_files(pattern: str, path: str) -> list[str]` - Find files by glob pattern

#### Error Analysis (Essential)
6. `read_error_report() -> dict` - Read and parse fuzzer crash report
7. `parse_stack_trace(content: str) -> list[dict]` - Extract stack trace info

#### Code Analysis (Important)
8. `get_function_code(file: str, function: str) -> dict` - Extract function definition
9. `search_code(query: str, path: str) -> list[dict]` - Semantic search (future)

#### Execution (Controlled - Restricted)
10. `run_command(cmd: list[str], cwd: str, timeout: int = 30) -> dict`
    - **WHITELISTED commands only**: `grep`, `find`, `cat`, `head`, `tail`, `wc`, `make`, compile commands
    - **DENIED**: `rm`, `rmdir`, `mv`, network commands, system modifications

#### Submission (Essential)
11. `submit_localization(locations: list[dict]) -> dict` - Submit final results
12. `submit_reasoning_trace(trace: dict) -> dict` - Submit reasoning steps

### 3. Sandboxed Execution Environment ✅

**Decision: Docker-Based Sandbox** (Using Arvo's existing images)

**Why Docker:**
- Arvo provides Docker images: `n132/arvo:{task_id}-vul` (and `-fix`)
- Already have Docker infrastructure (`server_utils.py`)
- Provides isolation and resource limits
- Reproducible environments

**Implementation Approach:**
```python
class DockerSandbox:
    """
    Sandboxed environment using Arvo's vulnerable Docker image.
    Mounts workspace into container for tool execution.
    """
    def __init__(self, arvo_id: str, workspace_dir: Path):
        self.arvo_id = arvo_id
        self.workspace_dir = workspace_dir
        self.image = f"n132/arvo:{arvo_id}-vul"
        self.container = None  # Long-lived container per task
        
    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute tool in Docker container"""
        # Mount workspace: /workspace (read-only for src-vul, writable for shared)
        # Run tool-specific command
        # Return structured result
        
    def cleanup(self):
        """Remove container after task"""
```

**Workspace Structure in Container:**
```
/workspace/
  ├── src-vul/              # Vulnerable codebase (extracted, read-only)
  ├── {arvo_id}_error.txt   # Fuzzer crash report (read-only)
  ├── shared/               # For submissions (writable)
  │   ├── loc.json         # Purple agent writes here
  │   └── reasoning.json   # Purple agent writes here
  └── .sandbox/            # Temporary files (writable)
```

**Security & Resource Limits:**
- Memory: 2GB per container
- CPU: 1 core
- Timeout: 30s per command (configurable)
- Disk: 5GB workspace size
- Network: None (isolated)
- File system: Read-only for source, writable only for `/workspace/shared` and `/workspace/.sandbox`

**Alternative (Future):** Process-based sandbox for lightweight operations (read_file, grep) if Docker overhead becomes an issue.

### 4. Turn Limits ✅

**Default Configuration:**
- **Max Turns**: 50 turns per task (configurable via `EvalRequest.config["max_turns"]`)
- **Warning Threshold**: 80% of max turns (log warning)
- **Turn Counting**: Each tool call = 1 turn (unless marked as "heavy")
  - Heavy operations: `run_command` = 2 turns
  - Failed calls still count as turns

**Turn Management:**
```python
class TurnManager:
    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turn_count = 0
        self.history = []
        
    def can_continue(self) -> bool:
        return self.turn_count < self.max_turns
        
    def record_turn(self, tool_call: ToolCall, result: ToolResult):
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "tool": tool_call.tool_name,
            "success": result.success,
            "timestamp": time.time()
        })
        
    def get_warning_threshold(self) -> int:
        return int(self.max_turns * 0.8)
```

### 5. End Conditions ✅

**Success Conditions (End Loop):**
1. **Submission Received** ✅
   - Both `loc.json` AND `reasoning.json` exist in `/workspace/shared/`
   - Files are valid (parse correctly)
   - `loc.json` contains at least one localization entry
   - Status: `"success"`

2. **Explicit End** ✅
   - Purple agent calls `submit_localization()` with valid data
   - Status: `"success"`

**Failure Conditions (End Loop):**
1. **Max Turns Reached** ⚠️
   - Turn counter reaches maximum
   - Status: `"max_turns_exceeded"`
   - Still evaluate any partial submissions if present

2. **Timeout** ⚠️
   - Total task time exceeds limit (default: 10 minutes, configurable)
   - Status: `"timeout"`

3. **Critical Error** ❌
   - Sandbox crashes or becomes unresponsive
   - Docker daemon unavailable
   - Workspace corruption
   - Status: `"critical_error"`

**Partial Success:**
- Loop ends without complete submission but partial results available
- Status: `"partial_submission"`
- Evaluate what was submitted

**End Condition Priority:**
1. Success (submission received) > Failure (timeout/turns/error)
2. If multiple failure conditions, use first encountered

**Implementation:**
```python
class EndConditionChecker:
    def check_end_condition(self, sandbox: DockerSandbox, turn_manager: TurnManager) -> dict:
        # Check for success conditions first
        if self._has_valid_submission(sandbox):
            return {"status": "success", "reason": "submission_received"}
            
        # Check failure conditions
        if not turn_manager.can_continue():
            return {"status": "max_turns_exceeded", "reason": "turn_limit_reached"}
            
        if self._is_timeout():
            return {"status": "timeout", "reason": "time_limit_exceeded"}
            
        if self._has_critical_error():
            return {"status": "critical_error", "reason": "sandbox_failure"}
            
        return {"status": "continue"}
```

### 6. Maintaining Judge/Finder Structure ✅

**Keeping AgentBeats Structure:**
- ✅ `RCAJudge` still implements `GreenAgent` interface
- ✅ `GreenExecutor` wraps RCAJudge for A2A protocol
- ✅ `EvalRequest` / `EvalResult` models unchanged
- ✅ Purple agent role: `"rca_finder"`

**New Integration Points:**
```python
class RCAJudge(GreenAgent):
    def __init__(self, ...):
        # Existing
        self._tool_provider = ToolProvider()
        
        # New
        self._tool_registry = ToolRegistry()
        self._sandbox_manager = SandboxManager()
        self._loop_manager = MiniSWELoopManager()
        
    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        # 1. Prepare task assets (existing logic)
        # 2. Create sandbox environment (NEW)
        # 3. Initialize tool loop (NEW)
        # 4. Start interaction with purple agent (modified)
        # 5. Execute turns until end condition (NEW)
        # 6. Evaluate results (existing logic)
        # 7. Cleanup sandbox (NEW)
```

**Purple Agent Communication:**
- Initial task description: Use existing `ToolProvider.talk_to_agent()`
- Tool calls: New A2A message protocol (see below)

### 7. Tool Call Protocol (Communication) 

**Option A: Structured JSON Messages (Recommended Initially)**

Since A2A's function calling capabilities aren't clear from the codebase, we'll use structured text messages:

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

**Purple Agent → Green Agent:**
```
Message: {"type": "tool_call", "tool": "grep", "arguments": {"pattern": "buffer", "path": "src-vul"}}
```

**Green Agent → Purple Agent:**
```
Message: {"type": "tool_result", "success": true, "result": [...], "turn": 1}
```

**Option B: A2A Function Calling (If Available)**
- Check A2A SDK for function calling support
- If available, use native function calling
- Otherwise, fall back to structured messages

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Tool registry implementation
- [ ] Docker sandbox wrapper
- [ ] Turn counter and history tracking
- [ ] End condition checking

### Phase 2: Essential Tools (Week 1-2)
- [ ] File operations: read_file, list_directory, read_file_lines
- [ ] Code search: grep, find_files
- [ ] Error report parsing: read_error_report, parse_stack_trace
- [ ] Submission tools: submit_localization, submit_reasoning_trace

### Phase 3: Loop Integration (Week 2)
- [ ] Mini-SWE loop manager
- [ ] Tool call parsing/validation
- [ ] Result formatting
- [ ] History management

### Phase 4: Purple Agent Integration (Week 2-3)
- [ ] Simplify purple agent to lightweight LLM wrapper
- [ ] Tool call message format
- [ ] Handle tool results
- [ ] Reasoning trace collection

### Phase 5: Polish & Testing (Week 3)
- [ ] Error handling and recovery
- [ ] Security hardening
- [ ] Resource management
- [ ] End-to-end testing

## Key Configuration

### EvalRequest Config Extensions:
```python
{
    "task_ids_file": str,          # Existing
    "num_tasks": int,              # Existing
    "max_turns": int = 50,         # New: Max tool call turns
    "max_task_time": int = 600,    # New: Max seconds per task (10 min)
    "tool_timeout": int = 30,      # New: Max seconds per tool
    "sandbox_memory": str = "2Gi", # New: Docker memory limit
}
```

## Next Steps

1. **Confirm Tool List**: Review and approve the tool registry list
2. **A2A Protocol Check**: Investigate A2A SDK for native function calling support
3. **Docker Image Details**: Confirm Arvo Docker image structure and available tools
4. **Start Implementation**: Begin with Phase 1 (Core Infrastructure)

