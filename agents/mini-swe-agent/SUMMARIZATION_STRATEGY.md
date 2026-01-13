# Smart Strategy: Summarize and Store Commands Separately

## Current Implementation

The current summarization strategy in `purple_agent_server.py` works as follows:

### Current Behavior (Lines 541-701)

1. **Trigger**: When token usage reaches 80% of `max_tokens`
2. **Process**: 
   - Keeps last 8 messages (recent context)
   - Summarizes all older messages into a single summary
   - Reconstructs message history: `[system, summary, recent_8_messages]`
3. **Problem**: Commands are embedded in the conversation history and get summarized, losing specific command details

### Current Code Flow

```python
# In _summarize_old_messages():
# 1. Separate messages into: system, old (to summarize), recent (to keep)
# 2. Summarize old_messages into a single summary
# 3. Reconstruct: [system_msg, summary_msg, recent_messages]
```

**Issue**: Commands like `ls /workspace/src-vul` and their results are part of the messages that get summarized, so specific command details are lost.

## Proposed Smart Strategy

### Goal
Store commands separately from conversation history so that:
1. Commands and their results are preserved even after summarization
2. Conversation reasoning/thoughts can be summarized to save tokens
3. Command history remains accessible for the LLM to reference

### Implementation Approach

#### 1. Separate Command Storage

Add to `task_ctx`:
```python
task_ctx = {
    "messages": [...],  # Conversation history (can be summarized)
    "command_history": [  # Commands stored separately (never summarized)
        {
            "step": 1,
            "command": "ls /workspace/src-vul",
            "result": "<returncode>0</returncode><output>...</output>",
            "timestamp": 1234567890
        },
        {
            "step": 2,
            "command": "cat /workspace/src-vul/main.c",
            "result": "<returncode>0</returncode><output>...</output>",
            "timestamp": 1234567891
        },
        # ... more commands
    ],
    # ... other fields
}
```

#### 2. Modified Summarization

When summarizing:
- **Summarize**: Conversation messages (reasoning, thoughts, analysis)
- **Preserve**: Command history (keep all commands intact)
- **Include in context**: Both summary + full command history

#### 3. Context Reconstruction

After summarization, reconstruct context as:
```
[system_message, 
 summary_of_old_conversation,
 recent_conversation_messages,
 command_history_summary]  # Or full command history if small
```

### Benefits

1. **Token Efficiency**: Summarize verbose reasoning while keeping command details
2. **Command Preservation**: Never lose specific commands that were executed
3. **Better Context**: LLM can reference exact commands and results
4. **Debugging**: Easier to see what commands were run

### Implementation Details

#### Extract Commands During Conversation

When processing messages, extract commands:
```python
def _extract_command_from_message(self, message: Dict) -> Optional[Dict]:
    """Extract command and result from message if it contains a command."""
    content = message.get("content", "")
    
    # Look for "execute: <command>" pattern
    if "execute:" in content:
        command_match = re.search(r"execute:\s*(.+?)(?:\n|$)", content)
        if command_match:
            command = command_match.group(1).strip()
            # Extract result from next message or current message
            return {"command": command, "result": ...}
    
    return None
```

#### Store Commands Separately

```python
# In _handle_green_response():
if command_executed:
    task_ctx["command_history"].append({
        "step": task_ctx["step_count"],
        "command": command,
        "result": green_response,
        "timestamp": time.time()
    })
```

#### Modified Summarization

```python
async def _summarize_old_messages(
    self, task_ctx: Dict, context_id: str, threshold: float = 0.8
) -> bool:
    # Separate: system, old_conversation (no commands), recent, command_history
    system_msg = None
    old_conversation = []  # Only non-command messages
    recent_messages = []
    command_history = task_ctx.get("command_history", [])  # Keep all commands
    
    # Filter out command-related messages from old_conversation
    for msg in old_messages:
        if not self._is_command_message(msg):
            old_conversation.append(msg)
    
    # Summarize only conversation (not commands)
    summary = await self._summarize_conversation(old_conversation)
    
    # Reconstruct with command history
    new_messages = [
        system_msg,
        summary_msg,
        *recent_messages,
        self._format_command_history(command_history)  # Include command history
    ]
```

#### Include Command History in Context

```python
def _format_command_history(self, command_history: List[Dict]) -> Dict:
    """Format command history for LLM context."""
    if not command_history:
        return None
    
    # Format as a user message
    history_text = "Command History:\n"
    for cmd in command_history[-20:]:  # Last 20 commands
        history_text += f"Step {cmd['step']}: {cmd['command']}\n"
        history_text += f"Result: {cmd['result'][:200]}...\n\n"
    
    return {
        "role": "user",
        "content": history_text,
        "timestamp": time.time()
    }
```

### Token Management

- **Command History**: Estimate tokens separately
- **Conversation**: Track tokens for summarization
- **Total**: `conversation_tokens + command_history_tokens`

### Example Flow

**Before Summarization:**
```
Messages: [system, msg1, msg2, ..., msg50]  # 50 messages
Command History: [cmd1, cmd2, ..., cmd20]   # 20 commands
Total: ~80k tokens
```

**After Summarization:**
```
Messages: [system, summary, msg43-50]  # 9 messages
Command History: [cmd1, cmd2, ..., cmd20]  # Still 20 commands (unchanged)
Total: ~30k tokens (saved 50k tokens)
```

## Next Steps

1. **Implement command extraction**: Extract commands from messages as they're processed
2. **Add command_history to task_ctx**: Store commands separately
3. **Modify summarization**: Only summarize conversation, preserve commands
4. **Update context building**: Include command history in LLM context
5. **Test**: Verify token savings and command preservation

## Related Code Locations

- **Summarization**: `purple_agent_server.py:541-701` (`_summarize_old_messages`)
- **Command Extraction**: `purple_agent_server.py:509-523` (`_extract_command`)
- **Message Handling**: `purple_agent_server.py:242-281` (`_handle_green_response`)
- **Context Building**: `purple_agent_server.py:335-408` (`_decide_with_default_agent`)

