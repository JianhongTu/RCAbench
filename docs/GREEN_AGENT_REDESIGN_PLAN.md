# Green Agent Redesign Plan: Mini-SWE Integration

## Overview

Redesign the green agent (RCA Judge) to integrate a Mini-SWE-style tool calling loop while maintaining the AgentBeats judge/finder structure. The green agent will orchestrate tool execution in a sandboxed environment, while the purple agent becomes a lightweight LLM wrapper that makes tool call requests.

## Architecture

### Current Structure (Maintained)
```
┌─────────────────────────────────────────────┐
│          AgentBeats Framework                │
│  - GreenAgent interface                      │
│  - GreenExecutor (A2A protocol handler)      │
│  - EvalRequest/Response models               │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│         RCA Judge (Green Agent)              │
│  - Validates requests                        │
│  - Orchestrates evaluation                   │
│  - Manages task lifecycle                    │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│    New: Mini-SWE Tool Execution Loop         │
│  - Tool registry                             │
│  - Sandboxed execution environment           │
│  - Turn management                           │
│  - End condition checking                    │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. Tool Registry

The green agent will provide a set of tools for the purple agent to invoke during RCA analysis.

#### Tool Interface
```python
class Tool(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema
    executor: Callable  # Execution function
```

#### Tool List for RCA Task

**File Operations:**
1. `read_file(path: str) -> str`
   - Read file contents from workspace
   - Returns file content as string
   - Error if file doesn't exist or outside workspace

2. `write_file(path: str, content: str) -> dict`
   - Write content to file in workspace
   - Creates parent directories if needed
   - Returns: `{"success": bool, "path": str, "bytes_written": int}`

3. `list_directory(path: str) -> list[str]`
   - List files and directories at path
   - Returns list of relative paths
   - Recursive option for deep exploration

4. `file_exists(path: str) -> bool`
   - Check if file/directory exists in workspace

**Code Search:**
5. `grep(pattern: str, path: str, recursive: bool = True) -> list[dict]`
   - Search for pattern in files
   - Returns: `[{"file": str, "line": int, "content": str}]`

6. `search_code(query: str, path: str) -> list[dict]`
   - Semantic search for code patterns (if available)
   - Falls back to grep if semantic search unavailable

7. `find_files(pattern: str, path: str) -> list[str]`
   - Find files matching pattern (glob)
   - Returns list of file paths

**Error Analysis:**
8. `read_error_report() -> dict`
   - Read and parse the fuzzer crash report
   - Returns structured error information:
     ```json
     {
       "error_type": str,
       "crash_location": {"file": str, "line": int, "function": str},
       "stack_trace": list[dict],
       "raw_content": str
     }
     ```

9. `parse_stack_trace(content: str) -> list[dict]`
   - Parse stack trace from error report
   - Extracts file, function, line number

**Code Analysis:**
10. `get_function_code(file: str, function: str) -> dict`
    - Extract function definition from file
    - Returns: `{"code": str, "start_line": int, "end_line": int}`

11. `get_call_stack(function: str, file: str) -> list[dict]`
    - Find all callers/callees of function
    - Returns call graph information

**Execution (Controlled):**
12. `run_command(cmd: list[str], cwd: str, timeout: int = 30) -> dict`
    - Run command in sandboxed environment
    - Returns: `{"exit_code": int, "stdout": str, "stderr": str, "success": bool}`
    - **Restricted to safe commands only** (compile, test, grep, etc.)

13. `read_file_lines(path: str, start_line: int, end_line: int) -> dict`
    - Read specific line range from file
    - Useful for examining code regions

**Submission:**
14. `submit_localization(locations: list[dict]) -> dict`
    - Submit final localization results
    - Validates format before accepting
    - Marks task as ready for evaluation

15. `submit_reasoning_trace(trace: dict) -> dict`
    - Submit reasoning trace with steps
    - Validates structure

### 2. Sandboxed Execution Environment

#### Docker-Based Sandbox (Primary)

**Why Docker:**
- Arvo provides Docker images (`n132/arvo:{task_id}-vul` and `-fix`)
- Already have Docker infrastructure in codebase
- Isolated, reproducible environments
- Can control resources (memory, CPU, time)

**Implementation:**
```python
class DockerSandbox:
    def __init__(self, arvo_id: str, workspace_dir: Path):
        self.arvo_id = arvo_id
        self.workspace_dir = workspace_dir
        self.container = None
        self.image = f"n132/arvo:{arvo_id}-vul"
        
    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute tool in Docker container"""
        # Mount workspace into container
        # Run tool-specific command
        # Return results
        
    def cleanup(self):
        """Remove container"""
```

**Workspace Structure in Container:**
```
/workspace/
  ├── src-vul/          # Vulnerable codebase (extracted)
  ├── {arvo_id}_error.txt  # Fuzzer crash report
  ├── shared/           # For submissions (loc.json, reasoning.json)
  └── .sandbox/         # Temporary files created during analysis
```

**Resource Limits:**
- Memory: 2GB per container
- CPU: 1 core
- Timeout: 30s per command (configurable per tool)
- Disk: 5GB (workspace size)

**Security Constraints:**
- No network access (except for tool-specific needs)
- Read-only filesystem for source code (write only to `/workspace/shared` and `/workspace/.sandbox`)
- Restricted command execution (whitelist of safe commands)
- Process limits (max processes, threads)

#### Alternative: Process-Based Sandbox (Future)
- Lighter weight for simple commands
- Less isolation
- Use for non-dangerous operations (read_file, grep)

### 3. Mini-SWE Loop Integration

**Loop Pattern:**
```python
class MiniSWELoop:
    def __init__(self, sandbox: DockerSandbox, tool_registry: ToolRegistry):
        self.sandbox = sandbox
        self.tools = tool_registry
        self.turn_count = 0
        self.max_turns = 50  # Configurable
        self.history = []  # Tool call history
        
    async def execute_turn(self, tool_call: ToolCall) -> ToolResult:
        """Execute one tool call turn"""
        # Validate tool call
        # Check turn limit
        # Execute in sandbox
        # Record in history
        # Check end conditions
        # Return result
        
    def is_finished(self) -> bool:
        """Check if loop should end"""
        # Max turns reached
        # Submission received (loc.json exists)
        # Error condition
        # Agent explicitly ends
```

**Turn Structure:**
```python
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    reasoning: str | None  # Optional explanation from purple agent
    
class ToolResult(BaseModel):
    success: bool
    result: dict | str
    error: str | None
    turn_number: int
    execution_time: float
```

### 4. Turn Limits and Management

#### Maximum Turns
- **Default**: 50 turns per task
- **Configurable**: Via EvalRequest config: `{"max_turns": 30}`
- **Per Tool Type**: Some tools count as "heavy" turns (e.g., run_command = 2 turns)
- **Warning Threshold**: Log warning at 80% of max turns

#### Turn Counting
- Each tool call = 1 turn (unless marked as "heavy")
- Failed tool calls still count as turns
- Resubmission of same tool with different args = new turn

#### Turn History
```python
class TurnHistory:
    turns: list[Turn]
    total_tool_calls: int
    tools_used: set[str]
    submission_count: int
    last_submission_time: datetime | None
```

### 5. End Conditions

The loop should end when any of these conditions are met:

#### Success Conditions:
1. **Submission Received**: 
   - Both `loc.json` and `reasoning.json` exist in `/workspace/shared/`
   - Files are valid (parse correctly)
   - `loc.json` contains at least one localization

2. **Explicit End**:
   - Purple agent sends `submit_final()` or `end_analysis()` tool call
   - Green agent acknowledges and stops loop

#### Failure Conditions:
1. **Max Turns Reached**:
   - Turn counter reaches maximum
   - Status: `"max_turns_exceeded"`
   - Still evaluates any partial submissions

2. **Timeout**:
   - Total task time exceeds limit (e.g., 10 minutes)
   - Status: `"timeout"`

3. **Critical Error**:
   - Sandbox crashes or becomes unresponsive
   - Docker daemon unavailable
   - Workspace corruption
   - Status: `"critical_error"`

#### Partial Success:
- Loop ends without complete submission but partial results available
- Status: `"partial_submission"`
- Evaluate what was submitted

#### End Condition Priority:
1. Success (submission received) > Failure (timeout/turns/error)
2. If multiple failure conditions, use first encountered

### 6. Integration with AgentBeats Structure

#### Maintain Existing Interface:
```python
class RCAJudge(GreenAgent):  # Still implements GreenAgent
    def __init__(self, ...):
        self._tool_registry = ToolRegistry()
        self._sandbox_manager = SandboxManager()
        self._loop_manager = MiniSWELoopManager()
        
    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        # 1. Prepare task assets (existing)
        # 2. Create sandbox environment
        # 3. Initialize tool loop
        # 4. Start interaction with purple agent
        # 5. Execute turns until end condition
        # 6. Evaluate results (existing)
        # 7. Cleanup sandbox
```

#### Purple Agent Communication:
- Use existing `ToolProvider.talk_to_agent()` for initial task description
- **New**: Tool calls come through A2A protocol as messages/tool_calls
- Purple agent sends tool call requests, green agent executes and returns results
- Communication pattern:
  ```
  Purple → Green: "I want to call tool 'read_file' with args {...}"
  Green → Purple: ToolResult with file contents
  Purple → Green: "I want to call tool 'grep' with args {...}"
  Green → Purple: ToolResult with grep results
  ...
  Purple → Green: "I want to call tool 'submit_localization' with args {...}"
  Green → Purple: ToolResult confirming submission
  Green: End loop, evaluate results
  ```

#### Tool Call Protocol (A2A Integration):

Option A: Function Calling (Recommended)
- Use A2A function calling if supported
- Purple agent declares tool calling capability
- Green agent provides tool definitions in agent card

Option B: Structured Messages
- Purple agent sends JSON messages with tool call intent
- Green agent parses and executes
- More flexible but requires custom parsing

Example Message Format:
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

### 7. Implementation Phases

#### Phase 1: Core Infrastructure
- [ ] Tool registry implementation
- [ ] Basic Docker sandbox wrapper
- [ ] Turn counter and history tracking
- [ ] End condition checking

#### Phase 2: Tool Implementation
- [ ] File operations (read_file, write_file, list_directory)
- [ ] Code search (grep, find_files)
- [ ] Error report parsing
- [ ] Safe command execution

#### Phase 3: Loop Integration
- [ ] Mini-SWE loop manager
- [ ] Tool call parsing/validation
- [ ] Result formatting and return
- [ ] History management

#### Phase 4: Purple Agent Integration
- [ ] Update purple agent to make tool calls (lightweight LLM wrapper)
- [ ] Tool call message format
- [ ] Handle tool results in purple agent
- [ ] Reasoning trace collection

#### Phase 5: Evaluation Integration
- [ ] Submission handling
- [ ] Evaluation after loop ends
- [ ] Result aggregation
- [ ] Cleanup and resource management

## Configuration

### EvalRequest Config Extensions:
```python
{
    "task_ids_file": str,          # Existing
    "num_tasks": int,              # Existing
    "max_turns": int = 50,         # New: Max tool call turns
    "max_task_time": int = 600,    # New: Max seconds per task
    "tool_timeout": int = 30,      # New: Max seconds per tool
    "sandbox_memory": str = "2Gi", # New: Docker memory limit
    "enable_semantic_search": bool = False,  # New: Future feature
}
```

## Error Handling

### Tool Execution Errors:
- Invalid tool name → Return error, don't count as turn
- Invalid arguments → Return validation error
- Tool execution failure → Return error with details, count as turn
- Sandbox failure → Critical error, end loop

### Recovery:
- Retry simple operations (file read) on transient errors
- Don't retry failed tool calls automatically (purple agent decides)
- Log all errors for debugging

## Testing Strategy

### Unit Tests:
- Tool registry: Register/execute tools
- Sandbox: Docker container management
- Turn management: Counting, limits, history
- End conditions: All condition types

### Integration Tests:
- Full loop with mock purple agent
- Tool execution in Docker
- Submission workflow
- Error scenarios

### End-to-End Tests:
- Full task with real purple agent (lightweight LLM)
- Multiple tasks with different complexities
- Resource limit enforcement
- Cleanup verification

## Security Considerations

1. **Command Whitelisting**: Only allow safe commands in `run_command`
   - Allowed: `grep`, `find`, `cat`, `head`, `tail`, `wc`, compile commands
   - Denied: `rm`, `rmdir`, `mv`, network commands, system modifications

2. **Path Validation**: All file paths must be within workspace
   - No absolute paths outside workspace
   - No `..` traversal
   - Whitelist of allowed paths

3. **Resource Limits**: Enforce Docker resource constraints
   - Memory limits prevent DoS
   - CPU limits prevent runaway processes
   - Time limits prevent hanging

4. **Input Sanitization**: Validate all tool arguments
   - JSON schema validation
   - Type checking
   - Length limits (file paths, command args)

## Future Enhancements

1. **Semantic Code Search**: Add LLM-based code search tool
2. **Diff Analysis**: Tool to compare vulnerable vs fixed versions
3. **Symbol Resolution**: Find function definitions, call sites
4. **Multi-container Support**: Run multiple tool calls in parallel (with care)
5. **Caching**: Cache tool results for repeated operations
6. **Progress Tracking**: Real-time updates of purple agent progress

## Open Questions

1. **Tool Call Format**: A2A function calling vs custom message format?
   - Need to check A2A SDK capabilities

2. **Parallel Tool Calls**: Allow multiple tools in one turn?
   - Initially: sequential only
   - Future: evaluate parallel execution

3. **Streaming Results**: Stream large tool outputs?
   - Initially: return full results
   - Future: stream for large files/outputs

4. **Workspace Persistence**: Keep workspace between tasks?
   - Currently: isolated per task
   - Future: Consider caching for efficiency

