"""
A2A Environment for mini-swe-agent.

This environment intercepts command execution and sends commands to the green agent
via A2A protocol, instead of executing them locally.
"""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass

from agentbeats.client import send_message

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of executing a command via A2A."""
    returncode: int
    output: str
    error: str = ""


class A2AEnvironment:
    """
    Environment that sends commands to green agent via A2A protocol.
    
    This implements the same interface as mini-swe-agent's LocalEnvironment,
    but instead of executing commands locally, it sends them to the green agent.
    """
    
    def __init__(self, green_agent_url: str, context_id: str):
        """
        Initialize A2A environment.
        
        Args:
            green_agent_url: URL of the green agent A2A server
            context_id: A2A context ID for this task
        """
        self.green_agent_url = green_agent_url
        self.context_id = context_id
        logger.info(f"A2AEnvironment initialized (green_agent_url={green_agent_url}, context_id={context_id})")
    
    def execute(self, command: str, cwd: Optional[str] = None) -> CommandResult:
        """
        Execute a command by sending it to the green agent via A2A.
        
        Args:
            command: The bash command to execute
            cwd: Working directory (ignored, green agent handles this)
            
        Returns:
            CommandResult with returncode, output, and error
        """
        # Format command for green agent
        command_request = f"execute: {command}"
        
        logger.info(f"[A2A_ENV] Sending command to green agent: {command[:100]}...")
        
        try:
            # Send to green agent via A2A (synchronous call)
            # Note: mini-swe-agent's DefaultAgent expects synchronous execute()
            # but send_message is async, so we need to run it in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                # For now, create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._send_sync, command_request)
                    response = future.result()
            else:
                response = loop.run_until_complete(self._send_async(command_request))
            
            # Parse response from green agent
            # Green agent returns: "<returncode>0</returncode><output>...</output>"
            returncode = 0
            output = response
            error = ""
            
            # Try to parse structured response
            if "<returncode>" in response:
                import re
                returncode_match = re.search(r"<returncode>(\d+)</returncode>", response)
                if returncode_match:
                    returncode = int(returncode_match.group(1))
                
                output_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
                if output_match:
                    output = output_match.group(1)
                else:
                    # If no <output> tag, take everything after </returncode>
                    output = response.split("</returncode>", 1)[-1].strip()
            
            logger.info(f"[A2A_ENV] Command executed (returncode={returncode}, output_length={len(output)})")
            
            return CommandResult(
                returncode=returncode,
                output=output,
                error=error,
            )
        except Exception as e:
            logger.error(f"[A2A_ENV] Error executing command via A2A: {e}", exc_info=True)
            return CommandResult(
                returncode=-1,
                output="",
                error=str(e),
            )
    
    def _send_sync(self, message: str) -> str:
        """Synchronous wrapper for async send_message."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._send_async(message))
        finally:
            loop.close()
    
    async def _send_async(self, message: str) -> str:
        """Send message to green agent asynchronously."""
        outputs = await send_message(
            message=message,
            base_url=self.green_agent_url,
            context_id=self.context_id,
        )
        return outputs.get("response", "")

