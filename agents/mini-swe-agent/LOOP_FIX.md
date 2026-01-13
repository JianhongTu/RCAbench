# Loop Fix - Green Agent Re-initializing Tasks

## Problem

Green agent was stuck in a loop:
1. Green receives task from RCAJudge → initializes → sends formatted task to Purple
2. Purple receives formatted task → sends it BACK to Green (wrong!)
3. Green sees "arvo:10055" in message → treats as NEW task → downloads assets again
4. Loop repeats infinitely

## Root Cause

**Purple Agent** was sending the formatted task back to Green Agent:
```python
# WRONG - in purple_agent_server.py _handle_task_init()
green_response = await self._send_to_green_agent(message, context_id)
```

**Green Agent** was treating any message with "arvo:10055" as a new task:
```python
# WRONG - in green_agent_server.py execute()
elif re.search(r"arvo:\d+", user_input) or "Task ID:" in user_input:
    response = await self._handle_task_from_judge(...)  # Re-initializes!
```

## Fix

### 1. Purple Agent - Don't Send Task Back

**Before:**
```python
# Purple agent sends formatted task back to green (WRONG)
green_response = await self._send_to_green_agent(message, context_id)
```

**After:**
```python
# Purple agent just acknowledges receipt, doesn't send back
task_ctx["task_initialized"] = True
# Start command loop immediately
return await self._decide_next_command(...)
```

### 2. Green Agent - Check for Existing Task Context

**Before:**
```python
elif re.search(r"arvo:\d+", user_input) or "Task ID:" in user_input:
    # Always treats as new task
    response = await self._handle_task_from_judge(...)
```

**After:**
```python
elif (re.search(r"arvo:\d+", user_input) or "Task ID:" in user_input) and \
     not user_input.strip().startswith("Part I:") and \
     context_id not in self.task_contexts:
    # Only treat as new task if:
    # - Contains arvo_id
    # - Doesn't start with "Part I:" (formatted version)
    # - Task context doesn't already exist
    response = await self._handle_task_from_judge(...)
```

## Correct Flow Now

```
1. RCAJudge → Green Agent (task with arvo:10055)
2. Green Agent:
   - Extracts arvo_id
   - Downloads assets
   - Initializes ARVO container
   - Formats task (Part I: tools, Part II: task)
   - Sends formatted task to Purple Agent
   - Stores task context
3. Purple Agent:
   - Receives formatted task
   - Marks as initialized
   - Starts command loop
   - Sends commands: "execute: ls /workspace"
4. Green Agent:
   - Receives "execute: ls /workspace"
   - Executes in ARVO container
   - Returns results
5. Loop continues with commands...
```

## Testing

After fix, you should see:
- ✅ Green agent initializes task ONCE
- ✅ Purple agent receives formatted task
- ✅ Purple agent sends commands (not task back)
- ✅ Green agent executes commands
- ✅ No infinite loop of downloading assets

