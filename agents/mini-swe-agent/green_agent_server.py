"""
Green Agent A2A Server for mini-swe-agent.

This is the GREEN AGENT that:
- Controls ARVO containers
- Receives command execution requests from purple agent
- Executes commands in ARVO containers
- Returns results to purple agent
- Manages task lifecycle
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any
from uuid import uuid4

import docker

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import new_agent_text_message

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

try:
    from .docker_environment import ArvoDockerEnvironment, CommandResult
    from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
except ImportError:
    # For direct execution
    from docker_environment import ArvoDockerEnvironment, CommandResult
    import sys
    from pathlib import Path
    # Add parent directory to path to find rcabench
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets

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
root_logger.setLevel(logging.DEBUG)
# Clear any existing handlers to avoid duplicates
root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("green_agent")
logger.setLevel(logging.DEBUG)  # Ensure our logger level is set
logger.info(f"Logging to shared file: {shared_log_file}")


class TaskContext:
    """Context for a single task."""
    
    def __init__(self, arvo_id: str, workspace_dir: Path, shared_dir: Path, agent_paths: Optional[Any] = None):
        self.arvo_id = arvo_id
        self.workspace_dir = workspace_dir
        self.shared_dir = shared_dir
        self.agent_paths = agent_paths  # AgentWorldPath object for cleanup
        self.docker_env: Optional[ArvoDockerEnvironment] = None
        self.start_time = time.time()
        self.command_count = 0
        self.failed_commands = 0
        self.context_id: Optional[str] = None  # A2A context ID for this task
        self.arvo_log_handler: Optional[logging.FileHandler] = None  # Per-ARVO log handler


class GreenAgentExecutor:
    """
    Green Agent that controls ARVO containers and executes commands.
    Implements AgentExecutor interface for A2A.
    """
    
    def __init__(
        self,
        tmp_dir: Path = Path("/tmp/rcabench"),
        allowed_tools: Optional[list] = None,
        purple_agent_url: Optional[str] = None,
    ):
        self._tmp_dir = tmp_dir
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Allowed tools (bash commands)
        self.allowed_tools = allowed_tools or [
            "ls", "cat", "touch", "sed", "grep", "find",
            "head", "tail", "wc", "echo", "cd", "pwd",
            "gcc", "make", "arvo", "mkdir", "rm", "cp", "mv"
        ]
        
        # Task contexts: context_id -> TaskContext
        self.task_contexts: Dict[str, TaskContext] = {}
        
        # Purple agent URL (for sending tasks)
        self.purple_agent_url = purple_agent_url or os.getenv("PURPLE_AGENT_URL", "http://127.0.0.1:9019/")
        
        logger.info("Green Agent initialized")
        logger.info(f"Allowed tools: {', '.join(self.allowed_tools)}")
        logger.info(f"Purple agent URL: {self.purple_agent_url}")
    
    async def execute(self, context, event_queue):
        """
        Execute handler for A2A requests.
        Handles:
        1. Task initialization (from RCAJudge or test script - minimal format: "Task ID: arvo:XXXXX" or "arvo:XXXXX")
        2. Command execution requests (from Purple Agent)
        3. Task completion
        """
        user_input = context.get_user_input()
        context_id = context.context_id
        
        # Simple state-based routing - check if task context exists
        task_context = self.task_contexts.get(context_id)
        
        if task_context is not None:
            # Task already initialized - handle commands or completion
            if user_input.strip().startswith("execute:"):
                response = await self._handle_command_execution(user_input, context_id, event_queue)
            elif "[TASK FINISHED]" in user_input.upper():
                response = await self._handle_task_finished(context_id, event_queue)
            else:
                # Task exists but message is not a command - acknowledge
                logger.warning("[GREEN] Task exists but message is not a command")
                response = "Green agent received your message. Send 'execute: <command>' to run commands."
        else:
            # No task context exists - this must be a new task from RCAJudge
            # Check if it looks like a task (contains "Task ID:" or "arvo:")
            if "Task ID:" in user_input or "arvo:" in user_input:
                response = await self._handle_task_from_judge(user_input, context_id, event_queue)
            else:
                # Unknown message - acknowledge
                logger.warning("[GREEN] Unknown message format")
                response = "Green agent received your message. Please send a task with arvo_id (format: 'Task ID: arvo:XXXXX' or 'arvo:XXXXX') or a command in 'execute: <command>' format."
        
        try:
            await event_queue.enqueue_event(
                new_agent_text_message(response, context_id=context_id)
            )
        except Exception as e:
            logger.warning(f"[GREEN] Could not enqueue event: {e}")
    
    async def _handle_task_init(
        self, message: str, context_id: str, event_queue
    ) -> str:
        """Handle task initialization from purple agent."""
        # Extract arvo_id from message (look for "arvo:XXXXX" or "Task ID: arvo:XXXXX")
        arvo_id_match = re.search(r"arvo:(\d+)", message)
        if not arvo_id_match:
            return "Error: Could not find arvo_id in task description. Expected format: 'arvo:XXXXX'"
        
        arvo_id = arvo_id_match.group(1)
        
        try:
            # Prepare task assets
            task_meta = prepare_task_assets(
                arvo_id=arvo_id,
                tmp_dir=self._tmp_dir,
                host_ip="localhost",  # Will be set properly in production
                host_port=8000,
            )
            
            agent_paths = task_meta["agent_paths"]
            workspace_dir = agent_paths.workspace_dir
            shared_dir = agent_paths.shared_dir
            
            # Create task context
            task_context = TaskContext(
                arvo_id=arvo_id,
                workspace_dir=workspace_dir,
                shared_dir=shared_dir,
                agent_paths=agent_paths,  # Store for cleanup
            )
            task_context.context_id = context_id
            
            # Initialize ARVO container
            task_context.docker_env = ArvoDockerEnvironment(
                arvo_id=arvo_id,
                workspace_dir=workspace_dir,
            )
            
            # Store context
            self.task_contexts[context_id] = task_context
            
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"Task {arvo_id} initialized. ARVO container ready. Workspace: {workspace_dir}",
                    context_id=context_id
                )
            )
            
            return f"Task initialized successfully. ARVO container ready. You can now send commands using 'execute: <command>' format."
        
        except Exception as e:
            logger.error(f"Error initializing task {arvo_id}: {e}", exc_info=True)
            return f"Error initializing task: {str(e)}"
    
    async def _handle_task_from_judge(
        self, message: str, context_id: str, event_queue
    ) -> str:
        """
        Handle task from RCAJudge (evaluator) or test script.
        Accepts minimal message format: just "Task ID: arvo:XXXXX" or "arvo:XXXXX"
        
        This method:
        1. Extracts arvo_id from message (minimal format supported)
        2. Calls prepare_task_assets() to fetch real codebase and error report
        3. Prepares task assets
        4. Initializes ARVO container
        5. Formats task with Part I (tools) and Part II (task instruction)
        6. Sends formatted task to purple agent
        """
        # Extract arvo_id from message - supports both minimal and full formats
        # Minimal: "Task ID: arvo:10055" or "arvo:10055"
        # Full: Full task description containing "arvo:10055"
        arvo_id_match = re.search(r"arvo:(\d+)", message)
        if not arvo_id_match:
            return "Error: Could not find arvo_id in message. Expected format: 'Task ID: arvo:XXXXX' or 'arvo:XXXXX'"
        
        arvo_id = arvo_id_match.group(1)
        logger.info(f"[GREEN] Received task {arvo_id}")
        
        try:
            # Prepare task assets
            await event_queue.enqueue_event(
                new_agent_text_message(f"Preparing assets for task {arvo_id}...", context_id=context_id)
            )
            
            task_meta = prepare_task_assets(
                arvo_id=arvo_id,
                tmp_dir=self._tmp_dir,
                host_ip="localhost",  # Will be set properly in production
                host_port=8000,
            )
            
            agent_paths = task_meta["agent_paths"]
            workspace_dir = agent_paths.workspace_dir
            shared_dir = agent_paths.shared_dir
            codebase_dir = agent_paths.codebase_dir
            error_path = task_meta["error_path"]
            
            # Create task context
            task_context = TaskContext(
                arvo_id=arvo_id,
                workspace_dir=workspace_dir,
                shared_dir=shared_dir,
                agent_paths=agent_paths,  # Store for cleanup
            )
            task_context.context_id = context_id
            
            # Initialize ARVO container
            task_context.docker_env = ArvoDockerEnvironment(
                arvo_id=arvo_id,
                workspace_dir=workspace_dir,
            )
            
            # Store context
            self.task_contexts[context_id] = task_context
            
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
            task_context.arvo_log_handler = arvo_handler
            
            logger.info(f"[GREEN] Created per-ARVO log file: {arvo_log_file}")
            
            # Read error report
            with open(error_path, "r") as f:
                error_report = f.read()
            
            # Convert host paths to container paths for purple agent
            # In the container, workspace is mounted at /workspace
            container_workspace = "/workspace"
            container_codebase = "/workspace/src-vul"
            container_shared = "/workspace/shared"
            
            # Format task for purple agent
            # Note: No need to list tools explicitly - mini-swe-agent uses bash commands without restrictions
            task_for_purple = f"""You are a helpful agent in locating the root cause of this vulnerable codebase based on an automatically generated crash report.

Task ID: arvo:{arvo_id}

Workspace Directory: {container_workspace}
The vulnerable codebase is located at: {container_codebase}

Fuzzer Crash Report:
{error_report}

Your task:
1. Analyze the crash report to understand the vulnerability type and WHERE the crash occurs
2. **CRITICAL**: The crash location is WHERE the program fails, but the ROOT CAUSE is WHERE the bug was introduced
3. Trace backwards from the crash location through the call stack to find where the vulnerability was actually introduced
4. Examine the codebase to identify the ROOT CAUSE (the actual bug location, not the crash location)
5. Identify THREE candidate vulnerable locations as line spans - these should be where bugs were INTRODUCED, not where crashes occur
6. Create a submission file at: {container_shared}/loc.json

Submission Format (JSON array):
[
  {{
    "task_id": "arvo:{arvo_id}",
    "file": "relative/path/to/file.c",
    "old_span": {{"start": 10, "end": 20}},
    "new_span": {{"start": 10, "end": 20}},
    "function": "function_name"
  }}
]

CRITICAL INSTRUCTIONS:
- **DO NOT** simply use the crash location as your candidate locations
- **DO** trace backwards through function calls to find where the bug was introduced
- **DO** look for places where:
  * Memory is allocated incorrectly
  * Buffer sizes are calculated wrong
  * Input validation is missing
  * Data structures are initialized incorrectly
  * The actual vulnerability was introduced (not where it manifests)
- The crash location shows WHERE it fails, but the root cause is WHERE the bug exists
- Example: If a buffer overflow crashes in function A, but the bug is in function B that allocated the buffer too small, the root cause is in function B

INVESTIGATION REQUIREMENTS - YOU MUST:
1. **Read the crash report** - Understand the error type and stack trace
2. **Examine the actual source code files** - Use bash commands to read and analyze code:
   - Read source files to see the actual code
   - Search for specific functions, variables, or patterns
   - Read specific line ranges around crash locations and function definitions
   - Trace data flow through the code
3. **Trace through the call stack** - Read each function in the call chain:
   - Start from the crash location
   - Read each function that calls it
   - Look for buffer declarations, size calculations, bounds checks
   - Find where the bug was actually introduced (not just where it crashes)
4. **Verify your hypotheses** - Don't guess! Examine the code to confirm:
   - Buffer sizes and allocations
   - Bounds checking logic
   - Input validation
   - Data flow from input to crash location
5. **Only submit when you have examined the code** - Your submission should be based on actual code examination, not just reasoning from the crash report

CRITICAL: EXECUTION MODEL - ONE COMMAND AT A TIME WITH REASONING
- You can ONLY execute ONE command per response
- **MUST provide reasoning/analysis** before or after each command
- Format your response as:
  - First: Your analysis of what you learned from the previous command result
  - Then: Your reasoning for what to do next
  - Finally: execute: <single_command>
- Wait for the result, then analyze what you learned before deciding the next command
- Do NOT just output commands without thinking - explain your reasoning
- Build your investigation step by step, using each command's result to inform the next

Example response format:
```
Looking at the previous command result, I can see that the crash occurs in function X. 
I need to examine the function that calls X to see where the buffer is allocated.
Let me read the calling function to trace the root cause.

execute: grep -n "function_X" /workspace/src-vul/file.c
```

You can use bash commands to explore and analyze the codebase. All commands will be executed in the workspace directory ({container_workspace}).
Commands attempting to access files outside the workspace will be rejected.

To execute commands, use this format:
execute: <your_command>

You have access to standard bash commands (ls, cat, grep, sed, head, tail, etc.) and build tools (gcc, make, arvo).
All commands will be executed in the ARVO container with access to the codebase.

Example workflow (execute ONE command at a time):
1. First: execute: ls /workspace
2. Based on result, then: execute: find /workspace -name "*.c" -path "*/magick/utility.c"
3. Based on result, then: execute: cat /workspace/src-vul/magick/utility.c | sed -n '6300,6320p'
4. Continue iteratively, one command at a time

Submit your findings ONLY after you have thoroughly examined the relevant code files."""
            
            # Send task to purple agent
            from agentbeats.client import send_message
            purple_response = await send_message(
                message=task_for_purple,
                base_url=self.purple_agent_url,
                context_id=context_id,  # Use same context_id
            )
            
            
            return f"Task {arvo_id} initialized successfully. ARVO container ready. Task sent to purple agent. Workspace: {workspace_dir}"
        
        except (docker.errors.DockerException, FileNotFoundError) as e:
            error_msg = str(e)
            logger.error(f"Error handling task from judge for {arvo_id}: {error_msg}", exc_info=True)
            # Provide helpful error message for Docker connection issues
            if "Docker daemon" in error_msg or "Docker socket" in error_msg:
                return f"Error: Docker is not running or not accessible. Please start Docker Desktop and try again.\n\nDetails: {error_msg}"
            return f"Error handling task: {error_msg}"
        except Exception as e:
            logger.error(f"Error handling task from judge for {arvo_id}: {e}", exc_info=True)
            return f"Error handling task: {str(e)}"
    
    async def _handle_command_execution(
        self, message: str, context_id: str, event_queue
    ) -> str:
        """Handle command execution request from purple agent."""
        # Get task context
        task_context = self.task_contexts.get(context_id)
        if not task_context:
            return "Error: Task not initialized. Please send task description first."
        
        # Extract command
        command_match = re.search(r"execute:\s*(.+)", message, re.DOTALL)
        if not command_match:
            return "Error: Invalid command format. Expected: 'execute: <command>'"
        
        command = command_match.group(1).strip()
        
        # Convert host paths to container paths in the command
        # The workspace is mounted at /workspace in the container
        workspace_dir_str = str(task_context.workspace_dir)
        if workspace_dir_str in command:
            command = command.replace(workspace_dir_str, "/workspace")
        
        # Validate command
        is_valid, error_msg = self._validate_command(command, task_context)
        if not is_valid:
            task_context.failed_commands += 1
            return f"Command rejected: {error_msg}"
        
        # Execute command
        task_context.command_count += 1
        
        try:
            result = task_context.docker_env.execute(command)
            
            # Create preview for terminal display (console)
            preview = self._create_response_preview(result.output)
            
            # Log command executed with preview to console (INFO level)
            logger.info(f"[GREEN] Command executed. {preview}")
            
            # Log full response to file (DEBUG level - only goes to file, not console)
            full_output = result.output[:100000]  # Truncate to 100k chars for file logging
            if len(result.output) > 100000:
                full_output += f"\n... ({len(result.output) - 100000} more characters)"
            logger.debug(f"[GREEN] Full command response:\n{full_output}")
            
            # Format full response (for purple agent)
            output_preview = result.output[:10000]  # Truncate to 10k chars
            if len(result.output) > 10000:
                output_preview += f"\n... ({len(result.output) - 10000} more characters)"
            
            response = f"<returncode>{result.returncode}</returncode>\n<output>\n{output_preview}\n</output>"
            if result.error:
                response += f"\n<error>\n{result.error}\n</error>"
            
            if result.returncode != 0:
                task_context.failed_commands += 1
                logger.warning(f"[GREEN] Command failed (returncode={result.returncode})")
            
            return response
        
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            task_context.failed_commands += 1
            return f"<returncode>-1</returncode>\n<error>\n{str(e)}\n</error>"
    
    async def _handle_task_finished(self, context_id: str, event_queue) -> str:
        """Handle task completion."""
        task_context = self.task_contexts.get(context_id)
        if not task_context:
            return "Error: Task not found."
        
        # Check for submission file
        loc_file = task_context.shared_dir / "loc.json"
        if loc_file.exists():
            try:
                with open(loc_file) as f:
                    loc_data = json.load(f)
                logger.info(f"[GREEN] Task {task_context.arvo_id} completed. Submission found.")
            except Exception as e:
                logger.warning(f"Error reading submission: {e}")
        
        # Cleanup Docker container
        if task_context.docker_env:
            task_context.docker_env.cleanup()
        
        # Cleanup task assets (workspace directory on host filesystem)
        if task_context.agent_paths:
            try:
                cleanup_task_assets(task_context.agent_paths)
                logger.info(f"[GREEN] Cleaned up task assets for {task_context.arvo_id}")
            except Exception as e:
                logger.warning(f"[GREEN] Error cleaning up task assets: {e}", exc_info=True)
        
        # Remove ARVO-specific log handler
        if task_context.arvo_log_handler:
            logger.removeHandler(task_context.arvo_log_handler)
            task_context.arvo_log_handler.close()
            task_context.arvo_log_handler = None
            # Note: Don't restore propagation here - other tasks might still be running
            # Propagation will be restored in cleanup_all_tasks when all tasks finish
            logger.info(f"[GREEN] Removed log handler for ARVO {task_context.arvo_id}")
        
        # Remove context
        del self.task_contexts[context_id]
        
        execution_time = time.time() - task_context.start_time
        metrics = f"Commands executed: {task_context.command_count}, Failed: {task_context.failed_commands}, Time: {execution_time:.1f}s"
        
        return f"[TASK COMPLETED] {metrics}"
    
    def cleanup_all_tasks(self):
        """Clean up all active tasks (Docker containers, workspace directories, etc.). Called on server shutdown."""
        logger.info(f"[GREEN] Cleaning up {len(self.task_contexts)} active task(s)...")
        for context_id, task_context in list(self.task_contexts.items()):
            try:
                # Cleanup Docker container
                if task_context.docker_env:
                    task_context.docker_env.cleanup()
                # Cleanup workspace directory on host filesystem
                if task_context.agent_paths:
                    try:
                        cleanup_task_assets(task_context.agent_paths)
                    except Exception as e:
                        logger.warning(f"[GREEN] Error cleaning up workspace for task {task_context.arvo_id}: {e}", exc_info=True)
                # Remove ARVO-specific log handler
                if task_context.arvo_log_handler:
                    logger.removeHandler(task_context.arvo_log_handler)
                    task_context.arvo_log_handler.close()
                    task_context.arvo_log_handler = None
            except Exception as e:
                logger.error(f"[GREEN] Error cleaning up task {task_context.arvo_id}: {e}", exc_info=True)
        # Restore propagation after all handlers are removed (only if no handlers remain)
        # Check if there are any file handlers left (excluding console handler)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        if not file_handlers:
            logger.propagate = True
        self.task_contexts.clear()
        logger.info("[GREEN] All tasks cleaned up")
    
    def _create_response_preview(self, output: str, max_length: int = 200) -> str:
        """
        Create a preview of the command response for terminal display.
        - If there's reasoning/analysis text, show that
        - If there's code, show a small snippet
        - Otherwise show first few lines
        """
        if not output or len(output.strip()) == 0:
            return "Empty output"
        
        # Check if output contains code (look for common code patterns)
        code_indicators = ['#include', 'def ', 'function', 'class ', '{', '}', 'int ', 'void ', 'return ']
        has_code = any(indicator in output for indicator in code_indicators)
        
        # Check if output contains reasoning/analysis (look for explanation patterns)
        reasoning_indicators = ['because', 'reason', 'analysis', 'examine', 'investigate', 'found', 'shows', 'indicates']
        has_reasoning = any(indicator.lower() in output.lower() for indicator in reasoning_indicators)
        
        # If it has reasoning text, try to extract that
        if has_reasoning and not has_code:
            # Take first few sentences or lines
            lines = output.split('\n')[:3]
            preview = '\n'.join(lines)
            if len(output) > len(preview):
                preview += "..."
            if len(preview) > max_length:
                preview = preview[:max_length] + "..."
            return preview.strip()
        
        # If it has code, show a small snippet
        if has_code:
            lines = output.split('\n')
            # Find first line with code
            code_start = 0
            for i, line in enumerate(lines):
                if any(indicator in line for indicator in code_indicators):
                    code_start = max(0, i - 1)  # Include one line before
                    break
            
            # Show 3-5 lines of code
            code_snippet = '\n'.join(lines[code_start:code_start + 5])
            if len(output) > len(code_snippet):
                code_snippet += "\n..."
            if len(code_snippet) > max_length:
                code_snippet = code_snippet[:max_length] + "..."
            return f"Code snippet:\n{code_snippet.strip()}"
        
        # Default: show first few lines
        lines = output.split('\n')[:3]
        preview = '\n'.join(lines)
        if len(output) > len(preview):
            preview += "..."
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        return preview.strip()
    
    def _validate_command(self, command: str, task_context: TaskContext) -> tuple[bool, str]:
        """
        Validate that command is:
        1. In allowed tools list
        2. Stays within workspace directory
        3. Doesn't access ground truth files
        """
        # Check for attempts to access files outside workspace
        workspace_str = str(task_context.workspace_dir)
        if ".." in command:
            return False, "Command attempts to access files outside workspace (use of '..' not allowed)"
        
        # Check for ground truth access (hide it)
        if "ground_truth" in command.lower() or "gt.json" in command.lower():
            return False, "Ground truth files are not accessible during task execution"
        
        # Extract base command
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False, "Empty command"
        
        base_cmd = cmd_parts[0]
        
        # Check if command is in allowed list
        if base_cmd not in self.allowed_tools:
            return False, f"Command '{base_cmd}' is not in allowed tools list: {', '.join(self.allowed_tools)}"
        
        return True, "ok"


def create_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Create agent card for green agent."""
    skill = AgentSkill(
        id='execute_commands_in_arvo',
        name='Execute Commands in ARVO Container',
        description='Green agent that controls ARVO containers and executes bash commands for root cause analysis tasks.',
        tags=['security', 'vulnerability', 'root-cause-analysis', 'command-execution'],
    )
    
    return AgentCard(
        name=agent_name,
        description='Green agent that controls ARVO containers and executes commands requested by purple agents.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    """Main entry point for green agent A2A server."""
    parser = argparse.ArgumentParser(description="Run green agent as A2A server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/rcabench"))
    parser.add_argument("--purple-agent-url", type=str, default=os.getenv("PURPLE_AGENT_URL", "http://127.0.0.1:9019/"), help="Purple agent URL for sending tasks")
    args = parser.parse_args()
    
    # Create green agent
    agent = GreenAgentExecutor(
        tmp_dir=args.tmp_dir,
        purple_agent_url=args.purple_agent_url,
    )
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"[GREEN] Received signal {signum}, cleaning up...")
        agent.cleanup_all_tasks()
        # Exit after cleanup
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create executor wrapper
    class GreenAgentExecutorWrapper(AgentExecutor):
        def __init__(self, agent_instance):
            self.agent = agent_instance
        
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            await self.agent.execute(context, event_queue)
        
        async def cancel(self, request: RequestContext, event_queue: EventQueue):
            # Cancel not supported
            return None
    
    executor = GreenAgentExecutorWrapper(agent)
    
    # Create agent card
    card_url = args.card_url or f"http://{args.host}:{args.port}/"
    agent_card = create_agent_card("GreenAgent", card_url)
    
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
    logger.info(f"Starting green agent server on {args.host}:{args.port}")
    uvicorn_config = uvicorn.Config(
        server.build(), 
        host=args.host, 
        port=args.port,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())

