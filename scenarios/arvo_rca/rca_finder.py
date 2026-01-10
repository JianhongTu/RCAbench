"""
Simplified RCA Finder (Purple Agent) - Lightweight LLM wrapper for bash tool calling.

Receives tasks from green agent and uses LLM to generate bash commands for analysis.
Maintains conversation state to respond to green agent's messages.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

# Set up logging - ensure errors are always visible
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    force=True,  # Override any existing config
)
logger = logging.getLogger("rca_finder")

# Ensure ERROR and WARNING logs are always shown
logger.setLevel(logging.INFO)


class ConversationState:
    """Maintains state for a single conversation."""
    
    def __init__(self, arvo_id: str, task_description: str):
        self.arvo_id = arvo_id
        self.task_description = task_description
        self.messages = [
            {
                "role": "system",
                "content": """You are a security researcher performing root cause analysis.
You can run bash commands to analyze code. When you want to run a command, respond with ONLY a JSON object:
{"type": "bash_command", "command": "your command here"}

When you're done and want to submit results, respond with:
{"type": "done"}

## Recommended Commands (for large files, use targeted commands):
- head -n 100 file.c (read first 100 lines - better than cat for large files)
- tail -n 100 file.c (read last 100 lines)
- sed -n '100,200p' file.c (read lines 100-200)
- grep -n "pattern" file.c (search with line numbers)
- grep -A 5 -B 5 "pattern" file.c (context around matches)
- wc -l file.c (count lines in file)
- cat file.c (only for small files - will be truncated if too large)

## Other useful commands:
- ls -la src-vul/ (list directory)
- find src-vul/ -name "*.c" (find files)
- grep -r "pattern" src-vul/ (recursive search)"""
            },
            {
                "role": "user",
                "content": task_description
            }
        ]
        self.iteration = 0
        self.max_iterations = 50


class RCAFinder:
    """Simplified RCA Finder - uses LLM to generate bash commands."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        trace_conversation: bool = False,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set OPENAI_API_KEY env var or pass --api-key")
        
        self.llm_client = OpenAI(api_key=self.api_key)
        # Store conversation state per context_id
        self.conversations: Dict[str, ConversationState] = {}
        self._trace_conversation = trace_conversation
        
    async def handle_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle message from green agent - respond with tool call or reasoning."""
        # Extract message
        message_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part.root, 'text'):
                    message_text += part.root.text
        
        context_id = context.context_id
        
        # Trace logging
        if self._trace_conversation:
            logger.info(f"[TRACE] PURPLE received from GREEN: {message_text[:200]}...")
        
        logger.info(f"[{context_id[:8]}] Received: {message_text[:100]}...")
        
        # Check message type
        try:
            msg_data = json.loads(message_text)
            
            if "task_description" in msg_data and context_id not in self.conversations:
                task_desc = msg_data["task_description"]
                arvo_id = self._extract_arvo_id(task_desc)
                if arvo_id:
                    self.conversations[context_id] = ConversationState(arvo_id, task_desc)
                    logger.info(f"[{context_id[:8]}] Initialized with task_description for arvo:{arvo_id}")
            
            # Ensure state exists
            if context_id not in self.conversations:
                logger.error(f"[{context_id[:8]}] No conversation state found for this context")
                return
            
            state = self.conversations[context_id]
            
            if msg_data.get("type") == "ready_for_tool_call":
                # Green agent is ready - send tool call
                response = await self._get_next_tool_call(state, event_queue, context_id)
                
                if self._trace_conversation:
                    logger.info(f"[TRACE] PURPLE → GREEN: {response[:200]}...")
                
                await event_queue.enqueue_event(
                    new_agent_text_message(response, context_id=context_id)
                )
            elif msg_data.get("type") == "command_result":
                # Green agent sent command result - process it
                stdout = msg_data.get("stdout", "")
                success = msg_data.get("success", False)
                exit_code = msg_data.get("exit_code", 1)
                
                # Truncate large outputs to prevent token limit issues
                # Rough estimate: ~4 chars per token, so 5000 tokens = ~20k chars
                MAX_OUTPUT_CHARS = 20000
                if len(stdout) > MAX_OUTPUT_CHARS:
                    truncated = stdout[:MAX_OUTPUT_CHARS]
                    stdout = f"{truncated}\n... [Output truncated: {len(stdout)} chars total, showing first {MAX_OUTPUT_CHARS} chars]"
                    logger.warning(f"[{context_id[:8]}] Truncated large command output: {len(msg_data.get('stdout', ''))} chars")
                
                # Add result to conversation
                result_text = f"Command result (success={success}, exit_code={exit_code}):\n{stdout}"
                state.messages.append({
                    "role": "user",
                    "content": result_text
                })
                
                # Smart conversation history management
                # Keep system + task + all assistant messages (reasoning) + recent command results
                MAX_MESSAGES = 20  # Allow more messages
                if len(state.messages) > MAX_MESSAGES:
                    # Separate messages by role
                    system_and_task = state.messages[:2]  # Keep system + task
                    rest = state.messages[2:]
                    
                    # Keep all assistant messages (LLM reasoning is valuable)
                    assistant_msgs = [msg for msg in rest if msg.get("role") == "assistant"]
                    
                    # Keep recent user messages (command results) - last 8
                    user_msgs = [msg for msg in rest if msg.get("role") == "user"]
                    recent_user_msgs = user_msgs[-8:] if len(user_msgs) > 8 else user_msgs
                    
                    # Reconstruct: system + task + all assistant + recent user
                    state.messages = system_and_task + assistant_msgs + recent_user_msgs
                    logger.info(f"[{context_id[:8]}] Trimmed conversation: kept {len(assistant_msgs)} assistant + {len(recent_user_msgs)} recent user messages")
                
                # Get next tool call
                response = await self._get_next_tool_call(state, event_queue, context_id)
                
                if self._trace_conversation:
                    logger.info(f"[TRACE] PURPLE → GREEN: {response[:200]}...")
                
                await event_queue.enqueue_event(
                    new_agent_text_message(response, context_id=context_id)
                )
            else:
                # Unknown message type - treat as regular message
                state.messages.append({"role": "user", "content": message_text})
                response = await self._get_next_tool_call(state, event_queue, context_id)
                
                if self._trace_conversation:
                    logger.info(f"[TRACE] PURPLE → GREEN: {response[:200]}...")
                
                await event_queue.enqueue_event(
                    new_agent_text_message(response, context_id=context_id)
                )
        except json.JSONDecodeError:
            # Not JSON - treat as regular message
            if context_id not in self.conversations:
                logger.error(f"[{context_id[:8]}] Non-JSON message received before initialization")
                return
                
            state = self.conversations[context_id]
            if message_text.strip() and message_text != state.task_description:
                state.messages.append({"role": "user", "content": message_text})
            
            # Get response
            response = await self._get_next_tool_call(state, event_queue, context_id)
            
            if self._trace_conversation:
                logger.info(f"[TRACE] PURPLE → GREEN: {response[:200]}...")
            
            await event_queue.enqueue_event(
                new_agent_text_message(response, context_id=context_id)
            )
    
    async def _get_next_tool_call(
        self,
        state: ConversationState,
        event_queue: EventQueue,
        context_id: str,
    ) -> str:
        """Get next tool call from LLM using a loop for retries."""
        
        for attempt in range(3): # Max 3 attempts per turn to get valid JSON
            if state.iteration >= state.max_iterations:
                return json.dumps({"type": "done", "reason": "Max iterations reached"})
            
            state.iteration += 1
            
            try:
                # Estimate token count and trim if needed
                total_chars = sum(len(msg.get("content", "")) for msg in state.messages)
                if total_chars > 80000: # ~20k tokens
                    logger.warning(f"[{context_id[:8]}] Large conversation, trimming...")
                    # Keep system + task + last 10 messages
                    state.messages = state.messages[:2] + state.messages[-10:]
                
                # LLM call with rate limit retries
                llm_message = await self._call_llm_with_retries(state.messages, context_id)
                state.messages.append({"role": "assistant", "content": llm_message})
                
                # Try to parse as tool call
                try:
                    # Clean up LLM message if it wrapped JSON in code blocks
                    cleaned_message = llm_message.strip()
                    if "```json" in cleaned_message:
                        cleaned_message = cleaned_message.split("```json")[1].split("```")[0].strip()
                    elif "```" in cleaned_message:
                        cleaned_message = cleaned_message.split("```")[1].split("```")[0].strip()
                        
                    tool_call = json.loads(cleaned_message)
                    if tool_call.get("type") in ["bash_command", "done"]:
                        if tool_call.get("type") == "done":
                            logger.info(f"[{context_id[:8]}] Done signal received from LLM")
                        else:
                            logger.info(f"[{context_id[:8]}] Tool call: {tool_call.get('command', '')[:50]}...")
                        return json.dumps(tool_call) # Ensure clean JSON return
                    
                    # If it's JSON but wrong type, ask to retry
                    logger.warning(f"[{context_id[:8]}] LLM returned unknown type: {tool_call.get('type')}")
                    state.messages.append({
                        "role": "user", 
                        "content": "Invalid 'type'. Please use 'bash_command' or 'done'."
                    })
                    
                except json.JSONDecodeError:
                    # Not JSON - try to extract it anyway
                    json_match = re.search(r'\{.*"type".*\}', llm_message, re.DOTALL)
                    if json_match:
                        try:
                            candidate = json.loads(json_match.group(0))
                            if candidate.get("type") in ["bash_command", "done"]:
                                return json.dumps(candidate)
                        except:
                            pass
                            
                    logger.warning(f"[{context_id[:8]}] LLM response was not valid JSON. Retrying...")
                    state.messages.append({
                        "role": "user",
                        "content": "Your last response was not valid JSON. Please respond with ONLY a JSON object."
                    })
                    
            except Exception as e:
                logger.error(f"[{context_id[:8]}] Error in LLM turn: {e}", exc_info=True)
                return json.dumps({"type": "done", "error": str(e)})
        
        # If we exhausted attempts
        return json.dumps({"type": "done", "reason": "Exhausted retries for valid JSON"})

    async def _call_llm_with_retries(self, messages: list[dict], context_id: str) -> str:
        """Call LLM with exponential backoff for rate limits."""
        import time
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"[{context_id[:8]}] Rate limited, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                raise
        raise RuntimeError("Max retries exceeded for LLM call")
    
    def _extract_arvo_id(self, text: str) -> str | None:
        """Extract ARVO ID from task description."""
        # Try multiple patterns
        patterns = [
            r'arvo:(\d+)',
            r'Task ID: arvo:(\d+)',
            r'arvo_id[:\s]+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None


class RCAFinderExecutor(AgentExecutor):
    """Executor for the simplified RCA Finder."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        trace_conversation: bool = False,
    ):
        self.finder = RCAFinder(
            model=model,
            api_key=api_key,
            trace_conversation=trace_conversation,
        )
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute task handling."""
        await self.finder.handle_task(context, event_queue)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution."""
        pass


def main():
    """Main entry point for purple agent server."""
    parser = argparse.ArgumentParser(description="RCA Finder (Purple Agent)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model")
    parser.add_argument("--api-key", type=str, help="API key (or use OPENAI_API_KEY)")
    parser.add_argument("--trace", action="store_true", help="Enable conversation trace logging")
    parser.add_argument("--trace-only", action="store_true", help="Show only trace, suppress other logs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (includes trace)")
    args = parser.parse_args()
    
    # If verbose is enabled, also enable trace
    trace_enabled = args.trace or args.verbose or args.trace_only
    
    # If trace-only mode, suppress all other loggers
    if args.trace_only:
        # Suppress all loggers except trace
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("rca_finder").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("a2a").setLevel(logging.CRITICAL)
    
    # Check if we're running in show-logs mode (stdout/stderr are visible)
    # In that case, also enable trace automatically
    if not trace_enabled:
        import sys
        # If stdout/stderr are not redirected, likely --show-logs is enabled
        if sys.stdout.isatty() or (hasattr(sys.stdout, 'fileno') and sys.stdout.fileno() >= 0):
            trace_enabled = True
            logger.info("Trace mode auto-enabled (logs are visible)")
    
    if trace_enabled and not args.trace_only:
        logger.info("✓ Trace mode enabled - conversation logs will be displayed")
    
    agent_card = AgentCard(
        name="RCAFinder",
        description="Performs root cause analysis using bash commands.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=RCAFinderExecutor(
            model=args.model,
            api_key=args.api_key,
            trace_conversation=trace_enabled,
        ),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    import uvicorn
    # Disable access logging in trace-only mode
    log_level = "critical" if args.trace_only else "info"
    uvicorn.run(
        server.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,  # 5 minutes - long enough for agent interactions
        log_level=log_level,
        access_log=not args.trace_only,
    )


if __name__ == "__main__":
    main()
