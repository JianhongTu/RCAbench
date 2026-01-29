"""
Mock RCA Finder (Purple Agent) for fast testing.

This is a lightweight alternative to OpenHands that quickly generates
test submissions (loc.json and reasoning.json) for testing the green agent.

Usage:
    python scenarios/arvo_rca/rca_finder_mock.py --host 127.0.0.1 --port 9019
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rca_finder_mock")


class MockRCAFinder:
    """Mock RCA Finder that quickly generates test submissions."""
    
    def __init__(self, delay: float = 2.0):
        """
        Args:
            delay: Delay in seconds before generating files (simulates processing time)
        """
        self.delay = delay
    
    async def handle_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle a task request from the green agent."""
        # Get the message text from the context
        message_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part.root, 'text'):
                    message_text += part.root.text
        
        logger.info(f"Received task: {message_text[:200]}...")
        
        # Parse task information from the message
        arvo_id = self._extract_arvo_id(message_text)
        workspace_dir = self._extract_workspace_dir(message_text)
        
        if not arvo_id or not workspace_dir:
            error_msg = f"Could not parse task information. arvo_id={arvo_id}, workspace_dir={workspace_dir}"
            logger.error(error_msg)
            try:
                await event_queue.enqueue_event(
                    new_agent_text_message(f"Error: {error_msg}", context_id=context.context_id)
                )
            except Exception:
                pass  # Queue might be closed
            return
        
        logger.info(f"Parsed task: arvo_id={arvo_id}, workspace_dir={workspace_dir}")
        
        try:
            await event_queue.enqueue_event(
                new_agent_text_message(f"Mock agent: Generating test submissions for arvo:{arvo_id}...", context_id=context.context_id)
            )
        except Exception:
            pass  # Queue might be closed
        
        # Simulate some processing time
        await asyncio.sleep(self.delay)
        
        # Generate test submissions
        await self._generate_test_submissions(arvo_id, Path(workspace_dir))
        
        logger.info(f"Generated test submissions for arvo:{arvo_id}")
    
    async def _generate_test_submissions(self, arvo_id: str, workspace_dir: Path) -> None:
        """Generate mock loc.json and reasoning.json files."""
        shared_dir = workspace_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to read the error file to get some context (if it exists)
        error_file = workspace_dir / f"{arvo_id}_error.txt"
        error_context = ""
        if error_file.exists():
            try:
                with open(error_file, "r") as f:
                    error_content = f.read()[:500]  # First 500 chars
                    error_context = f"Error report indicates: {error_content[:100]}..."
            except Exception:
                pass
        
        # Generate mock localization (with some realistic-looking values)
        loc_data = [
            {
                "task_id": f"arvo:{arvo_id}",
                "file": "magick/render.c",  # Common file name in graphicsmagick
                "old_span": {"start": 100, "end": 105},
                "new_span": {"start": 100, "end": 105},
                "function": "render_function"
            }
        ]
        
        # Generate mock reasoning trace
        reasoning_data = {
            "task_id": f"arvo:{arvo_id}",
            "reasoning_steps": [
                {
                    "step": 1,
                    "type": "observation",
                    "content": f"Crash report shows buffer overflow at line 1234 in render.c. {error_context}",
                    "evidence": [f"{arvo_id}_error.txt:45-50"]
                },
                {
                    "step": 2,
                    "type": "hypothesis",
                    "content": "The buffer is allocated in allocate_buffer() but size calculation is wrong",
                    "evidence": ["render.c:1200-1210"]
                },
                {
                    "step": 3,
                    "type": "analysis",
                    "content": "Traced data flow: input -> parse_size() -> allocate_buffer() -> write_buffer()",
                    "evidence": ["render.c:1150-1234"]
                },
                {
                    "step": 4,
                    "type": "verification",
                    "content": "Verified by checking: if parse_size() validates input, the overflow would not occur",
                    "evidence": ["render.c:1180-1185"]
                },
                {
                    "step": 5,
                    "type": "conclusion",
                    "content": "Root cause: parse_size() doesn't validate input, leading to integer overflow when calculating buffer size. The bug is at lines 100-105 where input validation is missing.",
                    "evidence": ["render.c:100-105"],
                    "prediction_id": 0
                }
            ],
            "rejected_hypotheses": [
                {
                    "hypothesis": "Crash is in write_buffer() function",
                    "why_rejected": "write_buffer() is correct - it writes what it's told. The issue is that it receives a buffer that's too small because size calculation was wrong."
                },
                {
                    "hypothesis": "Buffer allocation is the problem",
                    "why_rejected": "allocate_buffer() correctly allocates the size it's given. The problem is earlier in the chain - the size calculation in parse_size() is wrong."
                }
            ]
        }
        
        # Write files
        loc_file = shared_dir / "loc.json"
        with open(loc_file, "w") as f:
            json.dump(loc_data, f, indent=2)
        
        reasoning_file = shared_dir / "reasoning.json"
        with open(reasoning_file, "w") as f:
            json.dump(reasoning_data, f, indent=2)
        
        logger.info(f"Created {loc_file} and {reasoning_file}")
    
    def _extract_arvo_id(self, text: str) -> str | None:
        """Extract ARVO ID from task description."""
        match = re.search(r'arvo:(\d+)', text)
        if match:
            return match.group(1)
        return None
    
    def _extract_workspace_dir(self, text: str) -> str | None:
        """Extract workspace directory from task description."""
        match = re.search(r'Workspace Directory:\s*([^\n]+)', text)
        if match:
            return match.group(1).strip()
        return None


class MockRCAFinderExecutor(AgentExecutor):
    """Executor for the Mock RCA Finder purple agent."""
    
    def __init__(self, delay: float = 2.0):
        self.finder = MockRCAFinder(delay=delay)
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the task handling."""
        await self.finder.handle_task(context, event_queue)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution."""
        pass


def main():
    parser = argparse.ArgumentParser(description="Run the Mock RCA finder agent (fast testing).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay before generating files (seconds, default: 2.0)")
    args = parser.parse_args()
    
    agent_card = AgentCard(
        name="RCAFinder",
        description="Mock RCA finder for fast testing - quickly generates test submissions.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=MockRCAFinderExecutor(delay=args.delay),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    import uvicorn
    print(f"Starting Mock RCA Finder (fast testing) on {args.host}:{args.port}")
    print(f"Delay: {args.delay}s (adjust with --delay)")
    uvicorn.run(
        server.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()



