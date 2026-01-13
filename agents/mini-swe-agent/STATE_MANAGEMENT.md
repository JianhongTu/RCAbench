# State Management

## Current State

### In-Memory State (Not Persisted)

**Green Agent:**
- `self.task_contexts: Dict[str, TaskContext]` - Stores task state in memory
- Key: `context_id` (A2A conversation ID)
- Value: `TaskContext` object with:
  - `arvo_id`
  - `workspace_dir` (Path to downloaded codebase)
  - `shared_dir` (Path for submission files)
  - `docker_env` (ARVO container reference)
  - `command_count`, `failed_commands`, etc.

**Purple Agent:**
- `self.task_contexts: Dict[str, Dict]` - Stores conversation state in memory
- Key: `context_id`
- Value: Dict with messages, step_count, tokens, etc.

**Problem:** If agent restarts, all state is lost!

### File-Based State (Persisted)

**Workspace Files:**
- Downloaded codebases: `/tmp/rcabench/arvo_XXXXX-{agent_id}/workspace/`
- Submission files: `/tmp/rcabench/arvo_XXXXX-{agent_id}/workspace/shared/loc.json`
- Error reports: `/tmp/rcabench/arvo_XXXXX-{agent_id}/workspace/{arvo_id}_error.txt`

**Shared Volumes (Docker):**
- `./workspace:/workspace:rw` - Mounted in green agent
- `./workspace:/workspace:ro` - Mounted in purple agent (read-only)
- `./logs:/app/logs:rw` - Logs directory

**Note:** Workspace files persist on disk, but task context (which task is active, command history) is lost on restart.

## Simplified Task Detection

**Before (Regex-based):**
```python
elif (re.search(r"arvo:\d+", user_input) or "Task ID:" in user_input) and \
     not user_input.strip().startswith("Part I:") and \
     context_id not in self.task_contexts:
    # Complex regex matching
```

**After (State-based):**
```python
task_context = self.task_contexts.get(context_id)

if task_context is not None:
    # Task exists - handle commands
    if user_input.strip().startswith("execute:"):
        ...
else:
    # No task - must be new task from RCAJudge
    if "Task ID:" in user_input or "arvo:" in user_input:
        ...
```

**Benefits:**
- ✅ Simpler logic
- ✅ No regex needed
- ✅ State-based (check if context_id exists)
- ✅ Clear separation: task exists vs. new task

## State Persistence (Future Enhancement)

If we want to persist state across restarts:

1. **Save task contexts to file:**
   ```python
   # On task creation
   state_file = shared_dir / "task_state.json"
   with open(state_file, "w") as f:
       json.dump({
           "context_id": context_id,
           "arvo_id": arvo_id,
           "workspace_dir": str(workspace_dir),
           "command_count": command_count,
       }, f)
   ```

2. **Load on startup:**
   ```python
   # On agent startup
   for state_file in shared_dir.glob("*/task_state.json"):
       with open(state_file) as f:
           state = json.load(f)
           # Reconstruct task context
   ```

3. **Use shared volume:**
   - Mount `./state:/app/state:rw` for both agents
   - Store state files there

## Current Architecture

```
In-Memory State (Lost on Restart):
├── Green Agent: task_contexts[context_id] → TaskContext
└── Purple Agent: task_contexts[context_id] → Dict

Persisted Files (Survive Restart):
├── /tmp/rcabench/arvo_XXXXX-{agent_id}/
│   ├── workspace/
│   │   ├── src-vul/ (codebase)
│   │   ├── {arvo_id}_error.txt
│   │   └── shared/
│   │       └── loc.json (submission)
│   └── (workspace files persist)
└── ./workspace/ (mounted volume)
    └── (shared between containers)
```

## Summary

- **State:** In-memory only (lost on restart)
- **Files:** Persisted on disk (workspace, submissions)
- **Detection:** Now uses state check instead of regex
- **Volumes:** Shared workspace for file access between containers

