"""
Purple Agent A2A Server for mini-swe-agent.

This is the PURPLE AGENT that:
- Decides which commands to run (using LLM)
- Sends command execution requests to green agent
- Receives command results from green agent
- Performs root cause analysis
- Creates submission files
"""

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
)
from a2a.utils import new_agent_text_message

# Configure logging to both console and shared file
# Use shared utility to get run log directory
from utility import get_or_create_run_log_dir

scenario_file = Path(__file__).parent / "scenario.toml"
run_log_dir = get_or_create_run_log_dir(scenario_file=scenario_file)
shared_log_file = run_log_dir / "agents.log"  # Shared log file for both agents

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler (INFO level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Shared file handler (INFO level - same as console)
# Simple FileHandler (no rotation) - thread-safe for async writes
# Ensure the log file is created immediately by touching it
shared_log_file.parent.mkdir(parents=True, exist_ok=True)
try:
    shared_log_file.touch(exist_ok=True)
    # Verify file was created
    if not shared_log_file.exists():
        raise FileNotFoundError(f"Failed to create log file: {shared_log_file}")
except Exception as e:
    print(f"ERROR: Could not create log file {shared_log_file}: {e}", file=sys.stderr)
    raise

file_handler = logging.FileHandler(shared_log_file, encoding='utf-8', mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Configure root logger explicitly (force configuration even if already configured)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Clear any existing handlers to avoid duplicates
root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("purple_agent")
logger.setLevel(logging.INFO)  # Ensure our logger level is set
logger.info(f"Logging to shared file: {shared_log_file}")

# Import mini-swe-agent components
try:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm_model import LitellmModel
    MINI_SWE_AGENT_AVAILABLE = True
except ImportError as e:
    # Fallback if mini-swe-agent not installed
    MINI_SWE_AGENT_AVAILABLE = False
    logger.warning(f"mini-swe-agent not available: {e}. Install with: pip install mini-swe-agent")
    logger.warning("Falling back to manual LLM calls")

from agentbeats.client import send_message

try:
    from a2a_environment import A2AEnvironment
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from a2a_environment import A2AEnvironment


class PurpleAgentExecutor:
    """
    Purple Agent that decides commands and communicates with green agent.
    """
    
    def __init__(
        self,
        green_agent_url: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "",
        max_steps: int = 50,
        max_tokens: int = 100000,
        timeout: int = 1800,
    ):
        self.green_agent_url = green_agent_url
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Task contexts: context_id -> task state (includes DefaultAgent instance)
        self.task_contexts: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Purple Agent initialized. Green agent URL: {green_agent_url}")
        if MINI_SWE_AGENT_AVAILABLE:
            logger.info("Using mini-swe-agent DefaultAgent for LLM interaction")
        else:
            logger.warning("mini-swe-agent not available, falling back to manual LLM calls")
    
    def cleanup_all_tasks(self):
        """Clean up all active task contexts. Called on server shutdown."""
        logger.info(f"[PURPLE] Cleaning up {len(self.task_contexts)} active task context(s)...")
        # Clean up per-ARVO log handlers
        for context_id, task_ctx in list(self.task_contexts.items()):
            if "arvo_log_handler" in task_ctx:
                handler = task_ctx["arvo_log_handler"]
                logger.removeHandler(handler)
                handler.close()
                del task_ctx["arvo_log_handler"]
        # Restore propagation after all handlers are removed (only if no handlers remain)
        # Check if there are any file handlers left (excluding console handler)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        if not file_handlers:
            logger.propagate = True
        # Task contexts are just in-memory, so we just clear them
        # If DefaultAgent instances need cleanup, it would go here
        self.task_contexts.clear()
        logger.info("[PURPLE] All task contexts cleaned up")
    
    async def execute(self, context, event_queue):
        """
        Execute handler for A2A requests.
        Handles task initialization and command decision loop.
        """
        user_input = context.get_user_input()
        context_id = context.context_id
        
        # Initialize task context if not exists
        if context_id not in self.task_contexts:
            
            # Create DefaultAgent with A2AEnvironment if available
            agent = None
            if MINI_SWE_AGENT_AVAILABLE:
                try:
                    # Create A2A environment that sends commands to green agent
                    a2a_env = A2AEnvironment(
                        green_agent_url=self.green_agent_url,
                        context_id=context_id
                    )
                    
                    # Create model using LitellmModel (works with OpenAI and other providers)
                    model_kwargs = {"model_name": self.model}
                    if self.api_key:
                        model_kwargs["api_key"] = self.api_key
                    if self.base_url:
                        model_kwargs["base_url"] = self.base_url
                    model = LitellmModel(**model_kwargs)
                    
                    # Create DefaultAgent
                    agent = DefaultAgent(model=model, environment=a2a_env)
                    logger.info(f"[PURPLE] Created DefaultAgent for context {context_id}")
                except Exception as e:
                    logger.warning(f"[PURPLE] Failed to create DefaultAgent: {e}. Falling back to manual LLM calls.")
                    agent = None
            
            self.task_contexts[context_id] = {
                "messages": [],
                "step_count": 0,
                "total_tokens": 0,
                "start_time": time.time(),
                "task_initialized": False,
                "agent": agent,  # DefaultAgent instance
                "command_history": [],  # Store commands separately (for reference)
            }
        
        task_ctx = self.task_contexts[context_id]
        
        # Check if this is task initialization (contains "Task ID: arvo:" or "arvo:")
        # Also check if task is not yet initialized
        if not task_ctx["task_initialized"] and ("Task ID: arvo:" in user_input or "arvo:" in user_input):
            response = await self._handle_task_init(user_input, context_id, task_ctx, event_queue)
        elif task_ctx["task_initialized"]:
            # Task is initialized, this is a response from green agent
            response = await self._handle_green_response(user_input, context_id, task_ctx, event_queue)
        else:
            # First message but doesn't match expected format
            logger.warning(f"[PURPLE] Invalid task format. Expected 'Task ID: arvo:XXXXX'")
            response = "Error: Invalid task format. Expected task description with 'Task ID: arvo:XXXXX' or 'arvo:XXXXX'"
        
        try:
            await event_queue.enqueue_event(
                new_agent_text_message(response, context_id=context_id)
            )
        except Exception as e:
            logger.warning(f"[PURPLE] Could not enqueue event: {e}")
    
    async def _handle_task_init(
        self, message: str, context_id: str, task_ctx: Dict, event_queue
    ) -> str:
        """Handle task initialization."""
        # Extract arvo_id from message to create per-ARVO log file
        arvo_id_match = re.search(r"arvo:(\d+)", message)
        if arvo_id_match:
            arvo_id = arvo_id_match.group(1)
            
            # Clean up any existing per-ARVO log handlers from previous tasks
            # Remove all FileHandlers except the shared agents.log (which contains "agents.log" in its baseFilename)
            handlers_to_remove = []
            for handler in logger.handlers[:]:  # Copy list to avoid modification during iteration
                if isinstance(handler, logging.FileHandler):
                    # Keep only the shared agents.log handler
                    if "agents.log" not in handler.baseFilename:
                        handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                logger.removeHandler(handler)
                handler.close()
                logger.debug(f"[PURPLE] Removed old per-ARVO log handler: {handler.baseFilename}")

            # Create per-ARVO log file handler
            arvo_log_file = run_log_dir / f"arvo_{arvo_id}.log"
            arvo_log_file.parent.mkdir(parents=True, exist_ok=True)

            # Create file handler for this ARVO
            arvo_handler = logging.FileHandler(arvo_log_file, encoding='utf-8', mode='a')
            arvo_handler.setLevel(logging.INFO)
            # Include context_id in log format for traceability
            arvo_formatter = logging.Formatter(
                f'%(asctime)s - [{context_id[:8]}] - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            arvo_handler.setFormatter(arvo_formatter)

            # Add handler to logger
            # Disable propagation to prevent task logs from going to agents.log
            logger.propagate = False
            logger.addHandler(arvo_handler)
            task_ctx["arvo_log_handler"] = arvo_handler
            
            logger.info(f"[PURPLE] Created per-ARVO log file: {arvo_log_file}")
        
        # The message is already the full task description from green agent
        # Use it directly as the initial user message - no system prompt needed
        # Green agent's message contains all instructions, requirements, and task details
        
        task_ctx["messages"] = [
            {"role": "user", "content": message, "timestamp": time.time()}
        ]
        
        # Mark task as initialized
        # Note: Green agent has already initialized the task and sent it to us
        # We don't need to send anything back to green agent - just acknowledge receipt
        task_ctx["task_initialized"] = True
        
        logger.info("[PURPLE] Task initialized. Starting command loop...")
        
        # Start the command decision loop immediately
        # The green agent has already prepared assets and initialized ARVO container
        # We call _decide_next_command which will send the first command to green agent
        # The green agent's response will come as a NEW request, which will route to _handle_green_response
        try:
            response = await self._decide_next_command(context_id, task_ctx, event_queue)
            
            # IMPORTANT: The response from _decide_next_command is the green agent's command execution result
            # This should be treated as if it came from _handle_green_response, so we process it
            # and decide the next command. But we're still in _handle_task_init, so we need to
            # process this response and continue the loop.
            if response and not response.startswith("[TASK FINISHED]"):
                result = await self._handle_green_response(response, context_id, task_ctx, event_queue)
                # If task is finished, notify green agent so it can run evaluation
                if result and result.startswith("[TASK FINISHED]"):
                    logger.info("[PURPLE] Task finished, notifying green agent to run evaluation...")
                    try:
                        await self._send_to_green_agent("[TASK FINISHED]", context_id)
                    except Exception as e:
                        logger.warning(f"[PURPLE] Failed to notify green agent of task completion: {e}")
                return result
            # If response is already TASK FINISHED, notify green agent
            if response and response.startswith("[TASK FINISHED]"):
                logger.info("[PURPLE] Task finished, notifying green agent to run evaluation...")
                try:
                    await self._send_to_green_agent("[TASK FINISHED]", context_id)
                except Exception as e:
                    logger.warning(f"[PURPLE] Failed to notify green agent of task completion: {e}")
            return response
        except Exception as e:
            logger.error(f"[PURPLE] Exception in _handle_task_init after _decide_next_command: {e}", exc_info=True)
            raise
    
    async def _handle_green_response(
        self, message: str, context_id: str, task_ctx: Dict, event_queue
    ) -> str:
        """Handle response from green agent (command execution result)."""
        try:
            # Truncate large outputs to prevent token limit issues (like rca_finder.py)
            # Rough estimate: ~4 chars per token, so 5000 tokens = ~20k chars
            MAX_OUTPUT_CHARS = 20000
            if len(message) > MAX_OUTPUT_CHARS:
                truncated = message[:MAX_OUTPUT_CHARS]
                message = f"{truncated}\n... [Output truncated: {len(message)} chars total, showing first {MAX_OUTPUT_CHARS} chars]"
                logger.warning(f"[PURPLE] Truncated large command output: {len(message)} chars")
            
            # Add green agent response to context
            task_ctx["messages"].append({
                "role": "user",
                "content": f"Command result:\n{message}",
                "timestamp": time.time()
            })
            
            # Check if task is completed
            if "[TASK COMPLETED]" in message:
                logger.info("[PURPLE] Task completed by green agent")
                return "[TASK FINISHED]"
            
            # Check exit conditions
            should_exit, exit_reason = self._check_exit_conditions(task_ctx)
            if should_exit:
                logger.info(f"[PURPLE] Exit condition: {exit_reason}")
                return f"[TASK FINISHED] Exit reason: {exit_reason}"

            # Check if loc.json exists and is valid - auto-complete
            if await self._check_completion_status(context_id):
                logger.info("[PURPLE] loc.json found and valid, auto-completing task")
                return "[TASK FINISHED]"

            # Decide next command
            result = await self._decide_next_command(context_id, task_ctx, event_queue)
            
            # If task is finished, notify green agent so it can run evaluation
            if result and result.startswith("[TASK FINISHED]"):
                logger.info("[PURPLE] Task finished, notifying green agent to run evaluation...")
                try:
                    # Send TASK FINISHED message to green agent
                    await self._send_to_green_agent("[TASK FINISHED]", context_id)
                except Exception as e:
                    logger.warning(f"[PURPLE] Failed to notify green agent of task completion: {e}")
            
            return result
        except Exception as e:
            logger.error(f"[PURPLE] Exception in _handle_green_response: {e}", exc_info=True)
            raise
    
    async def _decide_next_command(
        self, context_id: str, task_ctx: Dict, event_queue
    ) -> str:
        """Use LLM to decide next command. Uses DefaultAgent if available, otherwise falls back to manual LLM calls."""
        task_ctx["step_count"] += 1
        
        if task_ctx["step_count"] > self.max_steps:
            logger.warning(f"[PURPLE] Max steps reached: {task_ctx['step_count']}")
            return "[TASK FINISHED] Max steps reached"
        
        # Log step progress
        logger.info(f"[PURPLE] Step {task_ctx['step_count']}/{self.max_steps}")
        
        # Simple message trimming (like rca_finder.py) - no summarization
        # Check total character count and trim if needed
        total_chars = sum(len(str(msg.get("content", ""))) for msg in task_ctx["messages"])
        MAX_CHARS = 80000  # ~20k tokens (4 chars per token estimate)
        
        if total_chars > MAX_CHARS:
            logger.warning(f"[PURPLE] Large conversation ({total_chars} chars, ~{total_chars//4} tokens), trimming...")
            self._trim_messages(task_ctx)
        
        try:
            agent = task_ctx.get("agent")
            
            if agent and MINI_SWE_AGENT_AVAILABLE:
                # Use DefaultAgent from mini-swe-agent
                return await self._decide_with_default_agent(context_id, task_ctx, event_queue, agent)
            else:
                # Fallback to manual LLM calls (original implementation)
                return await self._decide_with_manual_llm(context_id, task_ctx, event_queue)
        
        except Exception as e:
            logger.error(f"[PURPLE] Error in _decide_next_command: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def _decide_with_default_agent(
        self, context_id: str, task_ctx: Dict, event_queue, agent
    ) -> str:
        """Use DefaultAgent's model to decide next command."""
        messages = task_ctx["messages"]
        
        # Build messages for the model (remove timestamp)
        model_messages = [
            {k: v for k, v in msg.items() if k != "timestamp"}
            for msg in messages
        ]
        
        # Ensure command history is included if not already in messages
        # (it will be added by summarization, but include it here too for consistency)
        command_history = task_ctx.get("command_history", [])
        if command_history:
            # Check if command history is already in messages
            has_command_history = any("Command History" in str(msg.get("content", "")) for msg in messages)
            if not has_command_history:
                command_history_msg = self._format_command_history(command_history)
                if command_history_msg:
                    model_messages.append({k: v for k, v in command_history_msg.items() if k != "timestamp"})
        
        
        # Use DefaultAgent's model to get response
        # The model handles parameter fallbacks automatically
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        def call_model():
            try:
                # Get the model from the agent
                model = agent.model
                # Call model - mini-swe-agent's model handles parameter fallbacks
                response = model.complete(model_messages)
                return response
            except Exception as e:
                logger.error(f"[PURPLE] Error calling model: {e}", exc_info=True)
                raise
        
        # Run in executor to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, call_model)
            llm_response = await future
        
        # Check for empty or None response from DefaultAgent
        if llm_response is None:
            logger.error(f"[PURPLE] DefaultAgent model returned None response!")
            llm_response = ""
        
        if not llm_response or len(llm_response.strip()) == 0:
            logger.warning(f"[PURPLE] DefaultAgent model returned empty response!")
        
        if not llm_response or len(llm_response.strip()) == 0:
            logger.error(f"[PURPLE] LLM returned empty response!")
        
        # Add LLM response to context
        task_ctx["messages"].append({
            "role": "assistant",
            "content": llm_response or "",  # Ensure it's not None
            "timestamp": time.time()
        })
        
        # Check for task completion first (even if there's also a command)
        task_finished = "[TASK FINISHED]" in llm_response.upper()
        
        # Extract command from LLM response
        command = self._extract_command(llm_response)
        
        if command:
            # Send command to green agent
            command_request = f"execute: {command}"
            logger.info(f"[PURPLE] Command to run: {command}")
            
            # Store command in command history (before execution)
            task_ctx["command_history"].append({
                "step": task_ctx["step_count"],
                "command": command,
                "timestamp": time.time()
            })
            
            try:
                green_response = await self._send_to_green_agent(command_request, context_id)
                
                # Log response from green agent
                response_preview = self._create_response_preview(green_response)
                logger.info(f"[PURPLE] Received from green agent: {response_preview}")
                logger.debug(f"[PURPLE] Full green agent response:\n{green_response}")
                
                # Update command history with result
                if task_ctx["command_history"]:
                    task_ctx["command_history"][-1]["result"] = green_response
                
                # If task is finished, verify submission file was created and then finish
                if task_finished:
                    logger.info("[PURPLE] Task finished signal received")
                    # Check if this was a submission file creation command
                    if "loc.json" in command.lower():
                        # Verify the file was created successfully
                        verify_response = await self._send_to_green_agent("execute: test -f /workspace/shared/loc.json && cat /workspace/shared/loc.json", context_id)
                        if "returncode>0</returncode>" in verify_response or len(verify_response.strip()) < 50:
                            logger.warning("[PURPLE] Submission file verification failed or appears empty")
                            task_ctx["messages"].append({
                                "role": "user",
                                "content": f"Verification result:\n{verify_response}",
                                "timestamp": time.time()
                            })
                            # Verification failed - let agent continue investigating
                            # Process the verification result and decide next command
                            return await self._decide_next_command(context_id, task_ctx, event_queue)
                        else:
                            logger.info("[PURPLE] Submission file verified")
                    return "[TASK FINISHED]"
                
                # Add the command result to message history
                task_ctx["messages"].append({
                    "role": "user",
                    "content": f"Command result:\n{green_response}",
                    "timestamp": time.time()
                })
                
                # Auto-detect completion: if loc.json was created/verified successfully, signal completion
                if "loc.json" in command.lower():
                    # Check if the response indicates successful file creation/verification
                    if "<returncode>0</returncode>" in green_response:
                        # If this was an echo command creating the file, verify it by reading it
                        if command.strip().startswith("echo") and ">" in command and "/workspace/shared/loc.json" in command:
                            # Automatically verify the file was created correctly
                            logger.info("[PURPLE] Echo command created loc.json, verifying file content...")
                            verify_response = await self._send_to_green_agent("execute: cat /workspace/shared/loc.json", context_id)
                            if "<returncode>0</returncode>" in verify_response:
                                output_match = re.search(r'<output>(.*?)</output>', verify_response, re.DOTALL)
                                if output_match:
                                    output_content = output_match.group(1).strip()
                                    if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                                        try:
                                            json.loads(output_content)
                                            logger.info("[PURPLE] Auto-detected completion: loc.json successfully created and verified")
                                            return "[TASK FINISHED]"
                                        except json.JSONDecodeError as e:
                                            if '"reasoning"' in output_content and '"locations"' in output_content:
                                                logger.info("[PURPLE] Auto-detected completion: loc.json created (JSON may be incomplete but has required fields)")
                                                return "[TASK FINISHED]"
                        # For cat commands or other commands that output JSON directly
                        elif "cat /workspace/shared/loc.json" in command.lower():
                            # Try to extract JSON from response to validate it
                            output_match = re.search(r'<output>(.*?)</output>', green_response, re.DOTALL)
                            if output_match:
                                output_content = output_match.group(1).strip()
                                # Check if it looks like valid JSON (starts with { and contains "reasoning" or "locations")
                                if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                                    try:
                                        # Validate JSON structure
                                        json.loads(output_content)
                                        logger.info("[PURPLE] Auto-detected completion: loc.json successfully created and verified")
                                        return "[TASK FINISHED]"
                                    except json.JSONDecodeError as e:
                                        # JSON might be incomplete or malformed, but if it has the right structure, still complete
                                        # Check if it has both reasoning and locations keys (even if incomplete)
                                        if '"reasoning"' in output_content and '"locations"' in output_content:
                                            logger.info("[PURPLE] Auto-detected completion: loc.json created (JSON may be incomplete but has required fields)")
                                            return "[TASK FINISHED]"
                                        # Not valid JSON yet, continue
                                        pass
                
                # Process the command result immediately and decide next command
                # This processes one more turn (result + next command) but keeps queue open
                # by returning promptly with the next command
                return await self._decide_next_command(context_id, task_ctx, event_queue)
            except Exception as e:
                logger.error(f"[PURPLE] Error sending command to green agent: {e}", exc_info=True)
                return f"Error sending command to green agent: {str(e)}"
        elif task_finished:
            logger.info("[PURPLE] Task finished (no command)")
            return "[TASK FINISHED]"
        else:
            # No command extracted - check if LLM says task is complete and verify loc.json
            # Check if LLM response indicates completion (mentions "successfully", "created", "completed", etc.)
            completion_indicators = [
                "successfully created", "successfully written", "successfully saved",
                "has been successfully", "is successfully created", "is successfully written",
                "json object has been", "analysis is complete", "task is complete"
            ]
            llm_lower = llm_response.lower()
            if any(indicator in llm_lower for indicator in completion_indicators) and "loc.json" in llm_lower:
                # LLM says it's done - verify loc.json exists and is valid
                logger.info("[PURPLE] LLM indicated completion, verifying loc.json...")
                verify_response = await self._send_to_green_agent("execute: cat /workspace/shared/loc.json", context_id)
                if "<returncode>0</returncode>" in verify_response:
                    output_match = re.search(r'<output>(.*?)</output>', verify_response, re.DOTALL)
                    if output_match:
                        output_content = output_match.group(1).strip()
                        if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                            try:
                                json.loads(output_content)
                                logger.info("[PURPLE] Auto-detected completion: loc.json verified after LLM indicated completion")
                                return "[TASK FINISHED]"
                            except json.JSONDecodeError as e:
                                if '"reasoning"' in output_content and '"locations"' in output_content:
                                    logger.info("[PURPLE] Auto-detected completion: loc.json verified (JSON may be incomplete but has required fields)")
                                    return "[TASK FINISHED]"
            
            # Log what the LLM actually returned for debugging
            logger.warning(f"[PURPLE] No command extracted from LLM response (DefaultAgent). Response preview: {llm_response[:500]}")
            logger.debug(f"[PURPLE] Full LLM response: {llm_response}")
            task_ctx["messages"].append({
                "role": "user",
                "content": "Please provide a command in the format: execute: <command>",
                "timestamp": time.time()
            })
            return "Please provide a command to execute."
    
    async def _decide_with_manual_llm(
        self, context_id: str, task_ctx: Dict, event_queue
    ) -> str:
        """Fallback: Manual LLM calls (original implementation)."""
        from openai import OpenAI
        
        # Initialize OpenAI client if not already done
        if not hasattr(self, '_openai_client'):
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._openai_client = OpenAI(**client_kwargs)
        
        
        messages = [
            {k: v for k, v in msg.items() if k != "timestamp"}
            for msg in task_ctx["messages"]
        ]
        
        # Ensure command history is included if not already in messages
        command_history = task_ctx.get("command_history", [])
        if command_history:
            # Check if command history is already in messages
            has_command_history = any("Command History" in str(msg.get("content", "")) for msg in task_ctx["messages"])
            if not has_command_history:
                command_history_msg = self._format_command_history(command_history)
                if command_history_msg:
                    messages.append({k: v for k, v in command_history_msg.items() if k != "timestamp"})
        
        # Try with max_completion_tokens first, fallback based on errors
        # Increased to 4096 to handle longer responses when context is large
        max_tokens_limit = 4096
        try:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens_limit,
            )
        except Exception as e:
            error_str = str(e)
            if "max_completion_tokens" in error_str:
                try:
                    response = self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens_limit,
                    )
                except Exception as e2:
                    if "temperature" in str(e2):
                        response = self._openai_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=max_tokens_limit,
                        )
                    else:
                        raise
            elif "temperature" in error_str:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens_limit,
                )
            else:
                raise
        
        # Extract response content
        choice = response.choices[0]
        llm_response = choice.message.content
        
        # Check for empty or None response
        if llm_response is None:
            finish_reason = getattr(choice, 'finish_reason', 'unknown')
            logger.error(f"[PURPLE] LLM returned None content! finish_reason={finish_reason}, response_id={response.id}")
            logger.error(f"[PURPLE] Full response object: {response}")
            llm_response = ""  # Set to empty string to avoid None errors
        
        if not llm_response or len(llm_response.strip()) == 0:
            finish_reason = getattr(choice, 'finish_reason', 'unknown')
            logger.warning(f"[PURPLE] LLM returned empty response! finish_reason={finish_reason}")
            logger.warning(f"[PURPLE] Response object: choices={len(response.choices)}, model={response.model}")
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                logger.warning(f"[PURPLE] Finish reason: {choice.finish_reason}")
            
            # If response was truncated due to length, log warning
            if finish_reason == "length":
                logger.warning(f"[PURPLE] Response truncated due to max_tokens limit. Messages will be trimmed on next iteration.")
        
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        task_ctx["total_tokens"] += prompt_tokens + completion_tokens
        
        # Log token usage (tracking only, no limit enforcement)
        total_tokens = task_ctx["total_tokens"]
        logger.info(f"[PURPLE] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens} tokens")
        
        if not llm_response or len(llm_response.strip()) == 0:
            logger.error(f"[PURPLE] LLM returned empty response (finish_reason={getattr(choice, 'finish_reason', 'unknown')})")
        
        # Add LLM response to context
        task_ctx["messages"].append({
            "role": "assistant",
            "content": llm_response or "",  # Ensure it's not None
            "timestamp": time.time()
        })
        
        # Check for task completion first (even if there's also a command)
        task_finished = "[TASK FINISHED]" in llm_response.upper()
        
        # Extract command from LLM response
        # If response is empty, we can't extract a command
        if not llm_response or len(llm_response.strip()) == 0:
            logger.error("[PURPLE] Cannot extract command from empty LLM response (manual LLM)")
            task_ctx["messages"].append({
                "role": "user",
                "content": "The LLM returned an empty response. Please try again with a command.",
                "timestamp": time.time()
            })
            return "Error: LLM returned empty response. Please try again."
        
        command = self._extract_command(llm_response)
        if command:
            command_request = f"execute: {command}"
            logger.info(f"[PURPLE] Command to run: {command}")
            
            # Store command in command history (before execution)
            task_ctx["command_history"].append({
                "step": task_ctx["step_count"],
                "command": command,
                "timestamp": time.time()
            })
            
            try:
                green_response = await self._send_to_green_agent(command_request, context_id)
                
                # Log response from green agent
                response_preview = self._create_response_preview(green_response)
                logger.info(f"[PURPLE] Received from green agent: {response_preview}")
                logger.debug(f"[PURPLE] Full green agent response:\n{green_response}")
                
                # Update command history with result
                if task_ctx["command_history"]:
                    task_ctx["command_history"][-1]["result"] = green_response
                
                # Add the command result to message history
                task_ctx["messages"].append({
                    "role": "user",
                    "content": f"Command result:\n{green_response}",
                    "timestamp": time.time()
                })
                
                # If task is finished, verify submission file was created and then finish
                if task_finished:
                    logger.info("[PURPLE] Task finished signal received")
                    # Check if this was a submission file creation command
                    if "loc.json" in command.lower():
                        # Verify the file was created successfully
                        verify_response = await self._send_to_green_agent("execute: test -f /workspace/shared/loc.json && cat /workspace/shared/loc.json", context_id)
                        if "returncode>0</returncode>" in verify_response or len(verify_response.strip()) < 50:
                            logger.warning("[PURPLE] Submission file verification failed or appears empty")
                            task_ctx["messages"].append({
                                "role": "user",
                                "content": f"Verification result:\n{verify_response}",
                                "timestamp": time.time()
                            })
                            # Verification failed - let agent continue investigating
                            # Process the verification result and decide next command
                            return await self._decide_next_command(context_id, task_ctx, event_queue)
                        else:
                            logger.info("[PURPLE] Submission file verified")
                    return "[TASK FINISHED]"

                # Check if loc.json was just successfully created - auto-complete even if LLM didn't signal finish
                if "loc.json" in command.lower() and "<returncode>0</returncode>" in green_response:
                    logger.info("[PURPLE] loc.json successfully created, auto-completing task")
                    return "[TASK FINISHED]"

                # Auto-detect completion: if loc.json was created/verified successfully, signal completion
                if "loc.json" in command.lower():
                    # Check if the response indicates successful file creation/verification
                    if "<returncode>0</returncode>" in green_response:
                        # If this was an echo command creating the file, verify it by reading it
                        if command.strip().startswith("echo") and ">" in command and "/workspace/shared/loc.json" in command:
                            # Automatically verify the file was created correctly
                            logger.info("[PURPLE] Echo command created loc.json, verifying file content...")
                            verify_response = await self._send_to_green_agent("execute: cat /workspace/shared/loc.json", context_id)
                            if "<returncode>0</returncode>" in verify_response:
                                output_match = re.search(r'<output>(.*?)</output>', verify_response, re.DOTALL)
                                if output_match:
                                    output_content = output_match.group(1).strip()
                                    if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                                        try:
                                            json.loads(output_content)
                                            logger.info("[PURPLE] Auto-detected completion: loc.json successfully created and verified")
                                            return "[TASK FINISHED]"
                                        except json.JSONDecodeError as e:
                                            if '"reasoning"' in output_content and '"locations"' in output_content:
                                                logger.info("[PURPLE] Auto-detected completion: loc.json created (JSON may be incomplete but has required fields)")
                                                return "[TASK FINISHED]"
                        # For cat commands or other commands that output JSON directly
                        elif "cat /workspace/shared/loc.json" in command.lower():
                            # Try to extract JSON from response to validate it
                            output_match = re.search(r'<output>(.*?)</output>', green_response, re.DOTALL)
                            if output_match:
                                output_content = output_match.group(1).strip()
                                # Check if it looks like valid JSON (starts with { and contains "reasoning" or "locations")
                                if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                                    try:
                                        # Validate JSON structure
                                        json.loads(output_content)
                                        logger.info("[PURPLE] Auto-detected completion: loc.json successfully created and verified")
                                        return "[TASK FINISHED]"
                                    except json.JSONDecodeError as e:
                                        # JSON might be incomplete or malformed, but if it has the right structure, still complete
                                        # Check if it has both reasoning and locations keys (even if incomplete)
                                        if '"reasoning"' in output_content and '"locations"' in output_content:
                                            logger.info("[PURPLE] Auto-detected completion: loc.json created (JSON may be incomplete but has required fields)")
                                            return "[TASK FINISHED]"
                                        # Not valid JSON yet, continue
                                        pass
                
                # Process the command result immediately and decide next command
                # This processes one more turn (result + next command) but keeps queue open
                # by returning promptly with the next command
                return await self._decide_next_command(context_id, task_ctx, event_queue)
            except Exception as e:
                logger.error(f"[PURPLE] Error sending command to green agent: {e}", exc_info=True)
                return f"Error sending command to green agent: {str(e)}"
        elif task_finished:
            logger.info("[PURPLE] Task finished (no command)")
            return "[TASK FINISHED]"
        else:
            # No command extracted - check if LLM says task is complete and verify loc.json
            # Check if LLM response indicates completion (mentions "successfully", "created", "completed", etc.)
            llm_lower = llm_response.lower()
            if "loc.json" in llm_lower:
                # LLM says it's done - verify loc.json exists and is valid
                logger.info("[PURPLE] LLM indicated completion, verifying loc.json...")
                verify_response = await self._send_to_green_agent("execute: cat /workspace/shared/loc.json", context_id)
                if "<returncode>0</returncode>" in verify_response:
                    output_match = re.search(r'<output>(.*?)</output>', verify_response, re.DOTALL)
                    if output_match:
                        output_content = output_match.group(1).strip()
                        if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                            try:
                                json.loads(output_content)
                                logger.info("[PURPLE] Auto-detected completion: loc.json verified after LLM indicated completion")
                                return "[TASK FINISHED]"
                            except json.JSONDecodeError as e:
                                if '"reasoning"' in output_content and '"locations"' in output_content:
                                    logger.info("[PURPLE] Auto-detected completion: loc.json verified (JSON may be incomplete but has required fields)")
                                    return "[TASK FINISHED]"
            
            # Log what the LLM actually returned for debugging
            logger.warning(f"[PURPLE] No command extracted from LLM response (manual LLM). Response preview: {llm_response[:500]}")
            logger.debug(f"[PURPLE] Full LLM response: {llm_response}")
            task_ctx["messages"].append({
                "role": "user",
                "content": "Please provide a command in the format: execute: <command>",
                "timestamp": time.time()
            })
            return "Please provide a command to execute."
    
    def _extract_command(self, text: str) -> Optional[str]:
        """Extract command from LLM response."""
        if not text or not text.strip():
            return None
        
        # Look for "execute: <command>" pattern (most common)
        # Handle both single-line and multi-line commands
        # Capture everything until next "execute:" or double newline or end of text
        pattern = r"execute:\s*(.+?)(?=\n\s*execute:|\n\n|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            cmd = match.group(1).strip()
            # Remove backticks if present (LLM sometimes wraps commands in backticks)
            cmd = cmd.strip('`').strip()
            
            # Check if command starts with "bash" or "sh" followed by newline (multi-line command)
            # Pattern: bash\n<rest> or sh\n<rest>
            bash_match = re.match(r'^(bash|sh)\s*\n(.+)$', cmd, re.DOTALL | re.IGNORECASE)
            if bash_match:
                shell_cmd = bash_match.group(1)
                rest_cmd = bash_match.group(2).strip()
                
                # If this looks like JSON creation with echo, convert to cat with heredoc
                if "loc.json" in rest_cmd and ("{" in rest_cmd or '"reasoning"' in rest_cmd):
                    # Try to extract JSON content from echo command
                    # Pattern: echo '...' > loc.json or echo "..." > loc.json
                    json_match = re.search(r"echo\s+['\"](.*?)['\"]\s*>\s*/workspace/shared/loc\.json", rest_cmd, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        # Convert to cat with heredoc format (more reliable for multi-line JSON)
                        cmd = f"cat > /workspace/shared/loc.json << 'EOF'\n{json_content}\nEOF"
                    else:
                        # If we can't extract JSON, try to use the rest_cmd directly
                        # Remove the echo part and just use what comes after
                        # But this is a fallback - ideally we should extract the JSON
                        cmd = rest_cmd
                else:
                    # Not JSON creation, use the rest_cmd directly (skip bash/sh)
                    cmd = rest_cmd
            
            # Remove any trailing punctuation or extra formatting
            cmd = re.sub(r'[\.;]$', '', cmd).strip()
            if cmd and self._is_valid_command(cmd):
                return cmd
        
        # Look for backtick-wrapped commands: `command` or ```bash\ncommand\n```
        pattern = r"`([^`]+)`"
        matches = re.findall(pattern, text)
        for cmd in matches:
            cmd = cmd.strip()
            # Skip if it's a code block marker (bash, sh, shell, or just backticks)
            if cmd.lower() in ['bash', 'sh', 'shell', '```bash', '```', '```sh', '```shell']:
                continue
            # Skip if it starts with triple backticks
            if cmd.startswith('```'):
                continue
            # Remove any remaining backticks (safety measure)
            cmd = cmd.strip('`').strip()
            # Check if it looks like a command
            if self._is_valid_command(cmd):
                return cmd
        
        # Look for code blocks
        pattern = r"```(?:bash|sh|shell)?\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            cmd = match.group(1).strip()
            # Remove any backticks that might be in the code block content
            cmd = cmd.strip('`').strip()
            if self._is_valid_command(cmd):
                return cmd
        
        # Look for "Please run:" or "run:" followed by command
        pattern = r"(?:Please\s+)?run:?\s*(?:-\s*)?(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            cmd = match.group(1).strip()
            # Remove backticks if present
            cmd = cmd.strip('`').strip()
            cmd = re.sub(r'[\.;]$', '', cmd).strip()
            if self._is_valid_command(cmd):
                return cmd
        
        # Look for standalone commands (common bash commands)
        # This is a fallback for when LLM provides reasoning with a command suggestion
        common_commands = r"(sed|grep|cat|find|ls|head|tail|wc|gcc|make|arvo|cd|pwd|echo|grep|awk|cut|sort|uniq|diff|patch|git|svn|hg|bzr)"
        pattern = rf"(?:^|\n)\s*(?:-\s*)?`?{common_commands}\s+[^\n`]+`?"
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            cmd = match.group(0).strip().lstrip('-').strip().strip('`').strip()
            if self._is_valid_command(cmd):
                return cmd
        
        return None
    
    def _is_valid_command(self, cmd: str) -> bool:
        """Check if extracted text looks like a valid bash command."""
        if not cmd or len(cmd.strip()) < 2:
            return False
        
        cmd_stripped = cmd.strip()
        
        # Reject code block markers
        if cmd_stripped.startswith('```') or cmd_stripped.lower() in ['bash', 'sh', 'shell', '```bash', '```sh', '```shell']:
            return False
        
        # Must contain at least one non-whitespace character that's not just punctuation
        if not re.search(r'[a-zA-Z0-9/]', cmd_stripped):
            return False
        
        # Reject filenames (common patterns: .json, .c, .h, .py, etc.)
        if re.match(r'^[a-zA-Z0-9_/-]+\.(json|c|h|cpp|hpp|py|js|ts|java|go|rs|sh|bash)$', cmd_stripped, re.IGNORECASE):
            return False
        
        # Reject function names (C-style: starts with capital letter, contains underscores, no spaces)
        # Pattern: CapitalLetter followed by alphanumeric/underscores, no spaces, no special chars
        if re.match(r'^[A-Z][a-zA-Z0-9_]+$', cmd_stripped) and '_' in cmd_stripped and ' ' not in cmd_stripped:
            return False
        
        # Reject obviously invalid commands (just plain text, no command structure)
        # Check if it's just lowercase words without any command-like structure
        # This catches things like "resort in corresponding breakdown"
        if re.match(r'^[a-z\s]+$', cmd_stripped) and not re.match(
            r'^(sed|grep|cat|find|ls|head|tail|wc|gcc|make|arvo|cd|pwd|echo|grep|awk|cut|sort|uniq|diff|patch|git|svn|hg|bzr|test|mkdir|rm|cp|mv|chmod|chown|tar|zip|unzip|curl|wget|python|node|npm|pip)',
            cmd_stripped,
            re.IGNORECASE
        ):
            # If it's just words and doesn't start with a known command, reject it
            # But allow if it contains paths, quotes, or other command-like characters
            if not re.search(r'[/"\'`$<>|&;()]', cmd_stripped):
                return False
        
        # Reject sentence-like text that starts with capital letter (unless it's a command)
        if re.match(r'^[A-Z][a-z\s]+$', cmd_stripped) and not re.match(
            r'^(Sed|Grep|Cat|Find|Ls|Head|Tail|Wc|Gcc|Make|Arvo|Cd|Pwd|Echo|Awk|Cut|Sort|Uniq|Diff|Patch|Git|Svn|Hg|Bzr)',
            cmd_stripped
        ):
            # Sentence-like but not a command - reject unless it has command-like characters
            if not re.search(r'[/"\'`$<>|&;()]', cmd_stripped):
                return False
        
        return True
    
    def _create_response_preview(self, output: str, max_length: int = 200) -> str:
        """Create a preview of the command response for terminal display."""
        if not output or len(output.strip()) == 0:
            return "Empty output"
        
        # Extract returncode if present
        returncode_match = re.search(r'<returncode>(\d+)</returncode>', output)
        returncode = returncode_match.group(1) if returncode_match else "?"
        
        # Extract output content
        output_match = re.search(r'<output>(.*?)</output>', output, re.DOTALL)
        if output_match:
            content = output_match.group(1).strip()
        else:
            # No XML tags, use raw output
            content = output.strip()
        
        # Create preview
        if len(content) <= max_length:
            preview = content
        else:
            preview = content[:max_length] + "..."
        
        # Add returncode info
        if returncode != "?":
            return f"returncode={returncode}, {preview}"
        return preview
    
    def _check_exit_conditions(self, task_ctx: Dict) -> tuple[bool, str]:
        """Check if we should exit."""
        # Check max steps
        if task_ctx["step_count"] >= self.max_steps:
            return True, "MAX_STEPS_REACHED"

        # Check timeout
        if time.time() - task_ctx["start_time"] > self.timeout:
            return True, "TIMEOUT"

        return False, ""

    async def _check_completion_status(self, context_id: str) -> bool:
        """Check if loc.json exists and contains valid submission."""
        try:
            verify_response = await self._send_to_green_agent("execute: cat /workspace/shared/loc.json", context_id)
            if "<returncode>0</returncode>" in verify_response:
                output_match = re.search(r'<output>(.*?)</output>', verify_response, re.DOTALL)
                if output_match:
                    output_content = output_match.group(1).strip()
                    if output_content.startswith('{') and ('"reasoning"' in output_content or '"locations"' in output_content):
                        try:
                            json.loads(output_content)
                            return True
                        except json.JSONDecodeError:
                            if '"reasoning"' in output_content and '"locations"' in output_content:
                                return True
        except Exception as e:
            logger.debug(f"[PURPLE] Completion check failed: {e}")
        return False
    
    def _trim_messages(self, task_ctx: Dict) -> None:
        """
        Trim messages using simple approach (like rca_finder.py):
        - Keep system + task (first 2 messages)
        - Keep all assistant messages (reasoning is valuable)
        - Keep last 8 user messages (recent command results)
        - Drop everything else
        """
        messages = task_ctx["messages"]
        if len(messages) <= 10:
            return  # Not enough messages to trim
        
        # Separate messages by role
        system_and_task = messages[:2]  # Keep system + task
        rest = messages[2:]
        
        # Keep all assistant messages (LLM reasoning is valuable)
        assistant_msgs = [msg for msg in rest if msg.get("role") == "assistant"]
        
        # Keep recent user messages (command results) - last 8
        user_msgs = [msg for msg in rest if msg.get("role") == "user"]
        recent_user_msgs = user_msgs[-8:] if len(user_msgs) > 8 else user_msgs
        
        # Reconstruct: system + task + all assistant + recent user
        task_ctx["messages"] = system_and_task + assistant_msgs + recent_user_msgs
        logger.info(f"[PURPLE] Trimmed conversation: kept {len(assistant_msgs)} assistant + {len(recent_user_msgs)} recent user messages")
    
    def _is_command_message(self, msg: Dict) -> bool:
        """
        Check if a message is ONLY a command (no reasoning/analysis).
        Messages with reasoning + command should NOT be filtered (they contain valuable analysis).
        """
        content = msg.get("content", "")
        role = msg.get("role", "")
        
        # "Command result:" messages are always command-related (from green agent)
        if "Command result:" in content:
            return True
        
        # For assistant messages, check if it's ONLY a command (no reasoning)
        if role == "assistant":
            # If it contains "execute:" but also has substantial reasoning (more than just the command)
            # then it's NOT just a command message - it contains analysis
            if "execute:" in content:
                # Extract the part before "execute:"
                before_command = content.split("execute:")[0].strip()
                # If there's substantial text before the command, it's reasoning
                if len(before_command) > 50:  # Has reasoning before command
                    return False
                # Check if there's reasoning after the command
                after_match = re.search(r"execute:.*?\n(.+)", content, re.DOTALL)
                if after_match and len(after_match.group(1).strip()) > 50:
                    return False
                # Otherwise, it's just a command
                return True
            return False
        
        # For user messages with "execute:", they're command requests
        if role == "user" and "execute:" in content:
            return True
        
        return False
    
    async def _summarize_command_results(
        self, command_history: List[Dict], keep_recent: int = 5
    ) -> Optional[str]:
        """
        Summarize command results to extract insights and patterns.
        Only summarizes older commands, keeps recent ones in full detail.
        
        Args:
            command_history: List of command dictionaries
            keep_recent: Number of recent commands to keep in full detail (not summarized)
            
        Returns:
            Summary string of what was achieved, patterns discovered, etc., or None if no commands to summarize
        """
        if not command_history or len(command_history) <= keep_recent:
            # Not enough commands to summarize, or all are recent
            return None
        
        # Separate old commands (to summarize) from recent ones (to keep)
        old_commands = command_history[:-keep_recent] if len(command_history) > keep_recent else []
        recent_commands = command_history[-keep_recent:] if len(command_history) > keep_recent else command_history
        
        if not old_commands:
            return None
        
        logger.info(f"[PURPLE] Summarizing {len(old_commands)} old commands (keeping {len(recent_commands)} recent commands in full)")
        
        # Build command summary prompt with old commands
        commands_text = ""
        for cmd in old_commands:
            step = cmd.get("step", "?")
            command = cmd.get("command", "unknown")
            result = cmd.get("result", "")
            
            commands_text += f"Step {step}: {command}\n"
            # Show more context for summarization (up to 1000 chars per result)
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            commands_text += f"Result: {result_preview}\n\n"
        
        summary_prompt = f"""Analyze the following command execution history from a root cause analysis task.

For the commands below, identify:
1. What was achieved or discovered with each command
2. What patterns, files, or structures were revealed
3. What insights can be extracted from the results
4. What key information is relevant for continuing the analysis

Command History (to summarize):
{commands_text}

Provide a concise summary focusing on:
- What files/directories were examined
- What patterns or structures were discovered
- What key findings emerged from the command outputs
- What this tells us about the codebase or the bug location
- What approaches were tried and what was learned

Keep the summary focused and actionable for continuing the root cause analysis."""

        try:
            from openai import OpenAI
            if not hasattr(self, '_openai_client'):
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self._openai_client = OpenAI(**client_kwargs)
            
            # Call LLM for command summarization
            # Try max_completion_tokens first, fallback to max_tokens for older models
            try:
                summary_response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes command execution history to extract insights for root cause analysis."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_completion_tokens=1500,  # More tokens for command summary
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                    # Fallback for older models
                    summary_response = self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes command execution history to extract insights for root cause analysis."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        max_tokens=1500,  # More tokens for command summary
                    )
                else:
                    raise
            
            command_summary = summary_response.choices[0].message.content
            summary_tokens = summary_response.usage.prompt_tokens + summary_response.usage.completion_tokens
            
            logger.info(f"[PURPLE] Generated command summary ({len(command_summary)} chars, {summary_tokens} tokens)")
            logger.debug(f"[PURPLE] Command summary: {command_summary[:200]}...")
            
            return command_summary
            
        except Exception as e:
            logger.error(f"[PURPLE] Error summarizing command results: {e}", exc_info=True)
            return None
    
    def _format_command_history(self, command_history: List[Dict], max_commands: int = 20) -> Optional[Dict]:
        """Format command history for LLM context."""
        if not command_history:
            return None
        
        # Get last N commands
        recent_commands = command_history[-max_commands:] if len(command_history) > max_commands else command_history
        
        history_text = "Command History (preserved from previous steps):\n"
        for cmd in recent_commands:
            step = cmd.get("step", "?")
            command = cmd.get("command", "unknown")
            result = cmd.get("result", "")
            
            # Truncate result if too long
            result_preview = result[:300] + "..." if len(result) > 300 else result
            
            history_text += f"Step {step}: {command}\n"
            if result:
                history_text += f"  Result: {result_preview}\n"
            history_text += "\n"
        
        if len(command_history) > max_commands:
            history_text += f"\n(Total {len(command_history)} commands executed, showing last {max_commands})\n"
        
        return {
            "role": "user",
            "content": history_text,
            "timestamp": time.time()
        }
    
    async def _summarize_old_messages(
        self, task_ctx: Dict, context_id: str, threshold: float = 0.8
    ) -> bool:
        """
        Summarize old messages when token usage approaches limit.
        Preserves command history separately (never summarized).
        
        Args:
            task_ctx: Task context dictionary
            context_id: Context ID for logging
            threshold: Token usage threshold (0.8 = 80%) to trigger summarization
            
        Returns:
            True if summarization was performed, False otherwise
        """
        current_usage = task_ctx["total_tokens"] / self.max_tokens
        
        # Only summarize if we're above threshold and have enough messages
        if current_usage < threshold or len(task_ctx["messages"]) <= 10:
            return False
        
        logger.info(f"[PURPLE] Token usage at {current_usage*100:.1f}% - summarizing old messages")
        logger.info(f"[PURPLE] Command history has {len(task_ctx.get('command_history', []))} commands (will be preserved)")
        
        # Separate messages: system, old (to summarize), recent (to keep)
        # Command history is stored separately and never summarized
        system_msg = None
        old_conversation = []  # Only non-command messages
        recent_messages = []
        command_history = task_ctx.get("command_history", [])  # Preserve all commands
        
        # Keep last 8 messages (recent context)
        # Summarize everything before that (excluding command messages)
        keep_recent = 8
        
        for i, msg in enumerate(task_ctx["messages"]):
            if msg.get("role") == "system":
                system_msg = msg
            elif i < len(task_ctx["messages"]) - keep_recent:
                # Filter out command-related messages from old conversation
                if not self._is_command_message(msg):
                    old_conversation.append(msg)
            else:
                recent_messages.append(msg)
        
        if not old_conversation:
            logger.info("[PURPLE] No old conversation messages to summarize")
            return False
        
        # Create summarization prompt
        # Focus on: reasoning, insights, findings (commands are preserved separately)
        old_conversation_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:500]}"
            for msg in old_conversation
        ])
        
        summary_prompt = f"""Summarize the following conversation history from a root cause analysis task. 
Focus on:
1. What findings or insights were discovered
2. What approaches were tried but didn't lead to the root cause
3. What the agent learned about the codebase structure
4. What still needs to be investigated
5. The agent's reasoning and thought process

Note: Commands and their results are stored separately and don't need to be included in this summary.

Conversation history:
{old_conversation_text}

Provide a concise summary that will help the agent continue the analysis with this context."""

        try:
            # Use LLM to summarize (using manual LLM call for now)
            from openai import OpenAI
            if not hasattr(self, '_openai_client'):
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self._openai_client = OpenAI(**client_kwargs)
            
            # Call LLM for summarization
            # Try max_completion_tokens first, fallback to max_tokens for older models
            try:
                summary_response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes conversation history for root cause analysis tasks."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_completion_tokens=1000,  # Keep summary concise
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                    # Fallback for older models
                        summary_response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversation history for root cause analysis tasks."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=1000,  # Keep summary concise
            )
                else:
                    raise
            
            summary = summary_response.choices[0].message.content
            summary_tokens = summary_response.usage.prompt_tokens + summary_response.usage.completion_tokens
            
            logger.info(f"[PURPLE] Generated summary ({len(summary)} chars, {summary_tokens} tokens)")
            logger.debug(f"[PURPLE] Summary: {summary[:200]}...")
            
            # Reconstruct messages: system + summary + recent messages + command history
            new_messages = []
            if system_msg:
                new_messages.append(system_msg)
            
            # Add summary as a user message
            new_messages.append({
                "role": "user",
                "content": f"Previous conversation summary:\n{summary}\n\nContinue the analysis with this context.",
                "timestamp": time.time()
            })
            
            # Summarize command results intelligently (only old commands, keep recent ones)
            command_summary = await self._summarize_command_results(command_history, keep_recent=5)
            command_summary_tokens = 0
            if command_summary:
                new_messages.append({
                    "role": "user",
                    "content": f"Command Execution Summary (from previous steps):\n{command_summary}",
                    "timestamp": time.time()
                })
                command_summary_tokens = len(command_summary) // 4
            
            # Add recent messages
            new_messages.extend(recent_messages)
            
            # Add recent command history in full detail (last 5 commands)
            # This gives immediate context for recent commands
            recent_command_history = command_history[-5:] if len(command_history) > 5 else command_history
            if recent_command_history:
                recent_commands_msg = self._format_command_history(recent_command_history, max_commands=5)
                command_history_tokens = 0
                if recent_commands_msg:
                    new_messages.append(recent_commands_msg)
                    command_history_tokens = len(str(recent_commands_msg.get("content", ""))) // 4
            
            # Update task context
            task_ctx["messages"] = new_messages
            
            # Recalculate tokens: estimate from new messages + add summary tokens
            # Subtract old message tokens, add summary tokens
            # Note: Command history tokens are preserved and counted separately
            old_message_tokens = sum(
                len(str(msg.get("content", ""))) // 4 
                for msg in old_conversation
            )
            new_message_tokens = sum(
                len(str(msg.get("content", ""))) // 4 
                for msg in new_messages
            )
            
            # Update total tokens (rough estimate)
            # Subtract old conversation tokens, add new message tokens (which includes command summary and recent commands)
            # Note: new_message_tokens already includes command_summary_tokens and command_history_tokens
            task_ctx["total_tokens"] = task_ctx["total_tokens"] - old_message_tokens + new_message_tokens
            
            logger.info(f"[PURPLE] Summarization complete. New token count: {task_ctx['total_tokens']} (reduced by ~{old_message_tokens - new_message_tokens} tokens)")
            logger.info(f"[PURPLE] Command history: {len(command_history)} total commands")
            if command_summary:
                old_command_count = len(command_history) - len(recent_command_history) if recent_command_history else len(command_history)
                logger.info(f"[PURPLE]   - Summarized {old_command_count} old commands (~{command_summary_tokens} tokens)")
            if recent_command_history:
                logger.info(f"[PURPLE]   - Kept {len(recent_command_history)} recent commands in full detail (~{command_history_tokens} tokens)")
            
            return True
            
        except Exception as e:
            logger.error(f"[PURPLE] Error during summarization: {e}", exc_info=True)
            # Fallback: just truncate if summarization fails
            logger.warning("[PURPLE] Summarization failed, falling back to truncation")
            return self._truncate_messages_fallback(task_ctx, keep_recent=keep_recent)
    
    def _truncate_messages_fallback(self, task_ctx: Dict, keep_recent: int = 8) -> bool:
        """Fallback: Simple truncation if summarization fails."""
        messages = task_ctx["messages"]
        
        # Keep system message and last N messages
        system_msg = None
        other_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                other_messages.append(msg)
        
        # Keep only last N
        truncated = other_messages[-keep_recent:]
        
        # Reconstruct
        if system_msg:
            task_ctx["messages"] = [system_msg] + truncated
        else:
            task_ctx["messages"] = truncated
        
        # Recalculate tokens
        task_ctx["total_tokens"] = sum(
            len(str(msg.get("content", ""))) // 4 
            for msg in task_ctx["messages"]
        )
        
        logger.info(f"[PURPLE] Truncated to {len(task_ctx['messages'])} messages")
        return True
    
    async def _send_to_green_agent(self, message: str, context_id: str) -> str:
        """Send message to green agent via A2A."""
        try:
            logger.info(f"[PURPLE] Sending to green agent (url={self.green_agent_url}, context_id={context_id})")
            logger.debug(f"[PURPLE] Message: {message[:200]}...")
            outputs = await send_message(
                message=message,
                base_url=self.green_agent_url,
                context_id=context_id,
            )
            response = outputs.get("response", "")
            return response
        except Exception as e:
            logger.error(f"[PURPLE] Error sending to green agent: {e}", exc_info=True)
            return f"Error communicating with green agent: {str(e)}"


def create_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Create agent card for purple agent."""
    skill = AgentSkill(
        id='root_cause_analysis',
        name='Root Cause Analysis',
        description='Purple agent that performs root cause analysis on vulnerable codebases by deciding which commands to execute.',
        tags=['security', 'vulnerability', 'root-cause-analysis'],
    )
    
    return AgentCard(
        name=agent_name,
        description='Purple agent that decides which commands to run and communicates with green agent to execute them.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    """Main entry point for purple agent A2A server."""
    parser = argparse.ArgumentParser(description="Run purple agent as A2A server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--green-agent-url", type=str, default="http://127.0.0.1:9009/", help="Green agent URL")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model (e.g., gpt-4o, gpt-4-turbo, gpt-4o-mini)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
    args = parser.parse_args()
    
    # Create purple agent
    executor_instance = PurpleAgentExecutor(
        green_agent_url=args.green_agent_url,
        model=args.model,
        api_key=args.api_key,
        max_steps=args.max_steps,
    )
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"[PURPLE] Received signal {signum}, cleaning up...")
        executor_instance.cleanup_all_tasks()
        # Exit after cleanup
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create executor wrapper
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    
    class PurpleAgentExecutorWrapper(AgentExecutor):
        def __init__(self, agent_instance):
            self.agent = agent_instance
        
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            await self.agent.execute(context, event_queue)
        
        async def cancel(self, request: RequestContext, event_queue: EventQueue):
            # Cancel not supported
            return None
    
    executor = PurpleAgentExecutorWrapper(executor_instance)
    
    # Create agent card
    card_url = args.card_url or f"http://{args.host}:{args.port}/"
    agent_card = create_agent_card("PurpleAgent", card_url)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Run server
    logger.info(f"Starting purple agent server on {args.host}:{args.port}")
    logger.info(f"Green agent URL: {args.green_agent_url}")
    uvicorn_config = uvicorn.Config(
        server.build(), 
        host=args.host, 
        port=args.port,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())

