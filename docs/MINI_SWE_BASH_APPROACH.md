# Mini-SWE Bash-Only Tool Approach

## Key Insight

**Mini-SWE provides only bash as a tool** - the purple agent runs bash commands, and the green agent executes them in a sandboxed environment. This is much simpler than a tool registry!

## Architecture Change

### Instead of Tool Registry:
```python
# ❌ Complex: Multiple tools
tools = {
    "read_file": Tool(...),
    "grep": Tool(...),
    "list_directory": Tool(...),
    ...
}
```

### Use Bash-Only Model:
```python
# ✅ Simple: Just bash
tool = "bash"  # That's it!

# Purple agent runs commands like:
# - cat src-vul/main.c
# - grep -r "pattern" src-vul/
# - ls -la src-vul/
# - find src-vul/ -name "*.c"
```

## Benefits

1. **Simplicity**: No tool registry to maintain
2. **Flexibility**: Purple agent can run any command (within security limits)
3. **Familiar**: Standard Unix commands everyone knows
4. **Extensible**: New capabilities = new commands, no code changes needed
5. **Proven**: Mini-SWE demonstrates this works well

## Implementation

### Green Agent: Bash Executor

```python
class BashExecutor:
    """Executes bash commands in sandboxed Docker environment"""
    
    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox
        self.command_whitelist = [
            "cat", "head", "tail", "grep", "find", "ls", "wc",
            "sed", "awk", "cut", "sort", "uniq", "diff",
            "file", "stat", "readlink", "realpath",
            # Safe read-only operations
        ]
        self.command_blacklist = [
            "rm", "rmdir", "mv", "cp", "chmod", "chown",
            "wget", "curl", "nc", "netcat", "ssh",
            # Dangerous operations
        ]
    
    async def execute(self, command: str, cwd: str = "/workspace") -> dict:
        """Execute bash command in sandbox"""
        # 1. Validate command (whitelist/blacklist)
        # 2. Parse command safely
        # 3. Execute in Docker container
        # 4. Return stdout, stderr, exit_code
        
    def validate_command(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute"""
        # Check against whitelist/blacklist
        # Prevent command injection
        # Check path traversal
        pass
```

### Purple Agent: Bash Command Generator

```python
class RCAFinder:
    """Simplified purple agent that generates bash commands"""
    
    async def handle_task(self, context: RequestContext, event_queue: EventQueue):
        # 1. Parse task description
        # 2. Initialize LLM conversation
        # 3. Loop:
        #    - LLM decides next bash command to run
        #    - Send command to green agent
        #    - Receive command output
        #    - Update conversation with output
        #    - Decide next command or submit results
```

### Communication Protocol

**Purple Agent → Green Agent:**
```json
{
  "type": "bash_command",
  "command": "cat src-vul/main.c | head -50",
  "cwd": "/workspace",
  "reasoning": "Reading first 50 lines of main.c to understand entry point"
}
```

**Green Agent → Purple Agent:**
```json
{
  "type": "command_result",
  "success": true,
  "stdout": "...",
  "stderr": "",
  "exit_code": 0,
  "turn": 1,
  "turns_remaining": 49
}
```

## Security Model

### Command Validation

1. **Whitelist Approach** (Recommended):
   - Only allow safe, read-only commands
   - Whitelist: `cat`, `grep`, `find`, `ls`, `head`, `tail`, `wc`, `file`, etc.
   - Deny: `rm`, `mv`, `wget`, `curl`, `nc`, etc.

2. **Path Restrictions**:
   - All paths must be within `/workspace`
   - Prevent `..` traversal
   - Prevent absolute paths outside workspace

3. **Command Injection Prevention**:
   - Parse commands safely (don't use shell=True with user input)
   - Use `shlex.split()` for safe parsing
   - Validate each argument

4. **Resource Limits**:
   - Timeout per command (30s default)
   - Memory limits in Docker
   - Max output size (prevent DoS)

### Example Validation

```python
def validate_command(self, command: str) -> tuple[bool, str]:
    """Validate bash command before execution"""
    try:
        # Parse command
        parts = shlex.split(command)
        if not parts:
            return False, "Empty command"
        
        cmd = parts[0]
        
        # Check whitelist
        if cmd not in self.command_whitelist:
            return False, f"Command '{cmd}' not in whitelist"
        
        # Check for dangerous patterns
        if any(dangerous in command for dangerous in [';', '&&', '||', '|', '>', '<', '`']):
            # Allow pipes and redirects, but validate carefully
            # Could restrict to simple pipes like: cmd1 | cmd2
            pass
        
        # Check paths
        for part in parts[1:]:
            if part.startswith('/') and not part.startswith('/workspace'):
                return False, f"Path '{part}' outside workspace"
            if '..' in part:
                return False, f"Path traversal detected: '{part}'"
        
        return True, "OK"
    except Exception as e:
        return False, f"Validation error: {e}"
```

## Common Commands for RCA

### File Reading
```bash
# Read entire file
cat src-vul/main.c

# Read first N lines
head -100 src-vul/main.c

# Read last N lines
tail -50 src-vul/main.c

# Read specific lines
sed -n '100,150p' src-vul/main.c
```

### Code Search
```bash
# Search for pattern
grep -r "buffer" src-vul/

# Search with context
grep -r -A 5 -B 5 "malloc" src-vul/

# Find files
find src-vul/ -name "*.c" -type f

# Count occurrences
grep -r "pattern" src-vul/ | wc -l
```

### Directory Exploration
```bash
# List directory
ls -la src-vul/

# List recursively
find src-vul/ -type f

# File information
file src-vul/main.c
stat src-vul/main.c
```

### Error Report Analysis
```bash
# Read error report
cat 10055_error.txt

# Search error report
grep -i "overflow" 10055_error.txt

# Extract stack trace
grep -A 20 "Stack trace" 10055_error.txt
```

### Code Analysis
```bash
# Find function definitions
grep -r "^[a-zA-Z_].*(" src-vul/ | grep -v "^[[:space:]]*//"

# Find function calls
grep -r "function_name(" src-vul/

# Count lines
wc -l src-vul/main.c

# Compare files (if needed)
diff file1.c file2.c
```

## Submission Commands

For final submission, purple agent can write files:

```bash
# Write localization (via echo or here-doc)
cat > /workspace/shared/loc.json << 'EOF'
[{"task_id": "arvo:10055", ...}]
EOF

# Or use printf for simple JSON
printf '{"task_id": "arvo:10055", ...}' > /workspace/shared/loc.json
```

**Note**: Writing to `/workspace/shared/` should be allowed, but writing elsewhere should be restricted.

## Updated Implementation Plan

### Phase 1: Bash Executor (Simplified)

**File: `scenarios/arvo_rca/tools/bash_executor.py`**

```python
class BashExecutor:
    """Executes bash commands in Docker sandbox"""
    
    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox
        self.whitelist = self._load_whitelist()
        self.blacklist = self._load_blacklist()
    
    async def execute(self, command: str, cwd: str = "/workspace") -> dict:
        """Execute bash command"""
        # Validate
        valid, error = self.validate_command(command)
        if not valid:
            return {
                "success": False,
                "error": error,
                "stdout": "",
                "stderr": "",
                "exit_code": 1
            }
        
        # Execute in Docker
        result = await self.sandbox.execute_command(command, cwd)
        return result
    
    def validate_command(self, command: str) -> tuple[bool, str]:
        """Validate command is safe"""
        # Implementation as above
        pass
```

### Phase 2: Green Agent Integration

**In `rca_judge.py`:**

```python
class RCAJudge(GreenAgent):
    def __init__(self, ...):
        # Remove tool registry
        # Add bash executor
        self._bash_executor = None  # Created per task
    
    async def _process_task(...):
        # Create sandbox
        sandbox = DockerSandbox(arvo_id, workspace_dir)
        await sandbox.start()
        
        # Create bash executor
        bash_executor = BashExecutor(sandbox)
        
        # Send task description (with bash instructions)
        task_message = f"""
Task: arvo:{arvo_id}

Workspace: /workspace
- Error report: /workspace/{arvo_id}_error.txt
- Codebase: /workspace/src-vul/

You can run bash commands to analyze the codebase.
Send commands like:
{{"type": "bash_command", "command": "cat src-vul/main.c"}}

I will execute commands in the sandboxed environment.
"""
        
        # Tool calling loop (simplified)
        while True:
            # Receive bash command from purple agent
            tool_call = await self._receive_tool_call(...)
            
            if tool_call["type"] == "bash_command":
                result = await bash_executor.execute(
                    tool_call["command"],
                    tool_call.get("cwd", "/workspace")
                )
                await self._send_tool_result(result, ...)
            
            # Check end conditions
            if self._check_end_condition(shared_dir):
                break
```

### Phase 3: Purple Agent Simplification

**In `rca_finder.py`:**

```python
class RCAFinder:
    async def handle_task(self, context: RequestContext, event_queue: EventQueue):
        # Parse task
        task_info = self._parse_task(context.message)
        
        # Initialize LLM
        messages = [
            {
                "role": "system",
                "content": """You are an RCA analyst. You can run bash commands to analyze code.
Available commands: cat, grep, find, ls, head, tail, wc, etc.
Send commands as: {"type": "bash_command", "command": "..."}"""
            },
            {"role": "user", "content": task_info}
        ]
        
        # Loop: Generate commands, execute, analyze results
        while True:
            # LLM generates next command
            response = self.llm_client.chat.completions.create(...)
            
            # Extract command from LLM response
            command = self._extract_command(response)
            
            # Send to green agent
            result = await self._send_command_to_green_agent(command)
            
            # Update conversation
            messages.append({"role": "assistant", "content": f"Ran: {command}"})
            messages.append({"role": "user", "content": f"Result: {result['stdout']}"})
            
            # Check if should submit
            if self._should_submit(response):
                await self._submit_results(...)
                break
```

## Advantages of Bash-Only Model

1. **No Tool Registry**: Much simpler codebase
2. **Standard Commands**: Everyone knows bash
3. **Flexible**: Can combine commands with pipes
4. **Easy to Extend**: New capabilities = new commands
5. **Proven Model**: Mini-SWE demonstrates effectiveness
6. **Less Code**: Fewer abstractions, easier to maintain

## Security Considerations

1. **Strict Whitelisting**: Only allow safe commands
2. **Path Validation**: Prevent access outside workspace
3. **Command Injection**: Parse commands safely
4. **Resource Limits**: Timeout, memory, output size
5. **Read-Only by Default**: Only allow writes to `/workspace/shared/`

## Comparison

| Aspect | Tool Registry | Bash-Only |
|--------|--------------|-----------|
| Complexity | High | Low |
| Flexibility | Limited | High |
| Security | Easier to secure | Needs careful validation |
| Maintainability | More code | Less code |
| Learning Curve | Need to learn API | Standard bash |
| Extensibility | Add new tools | Use new commands |

## Recommendation

**Use Bash-Only Model** (Mini-SWE style):
- Simpler implementation
- More flexible
- Proven to work
- Easier to maintain
- Standard Unix commands

Just need careful security validation!
