"""
RCA Finder (Purple Agent) that uses OpenAI to analyze vulnerabilities.

This implementation:
1. Receives task descriptions from the green agent
2. Parses the task information (arvo_id, workspace, codebase, error report)
3. Uses OpenAI LLM to analyze the crash report and codebase
4. Generates localization results
5. Writes results to the shared directory
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rca_finder")


class RCAFinder:
    """RCA Finder that uses OpenAI to analyze vulnerabilities."""
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    async def handle_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Handle a task request from the green agent.
        
        Analyzes the vulnerability using OpenAI and generates localization results.
        """
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
        codebase_dir = self._extract_codebase_dir(message_text)
        
        if not arvo_id or not workspace_dir or not codebase_dir:
            error_msg = f"Could not parse task information. arvo_id={arvo_id}, workspace_dir={workspace_dir}, codebase_dir={codebase_dir}"
            logger.error(error_msg)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {error_msg}", context_id=context.context_id)
            )
            return
        
        logger.info(f"Parsed task: arvo_id={arvo_id}, workspace_dir={workspace_dir}, codebase_dir={codebase_dir}")
        
        # Send status update
        await event_queue.enqueue_event(
            new_agent_text_message(f"Starting analysis for arvo:{arvo_id}...", context_id=context.context_id)
        )
        
        try:
            # Read error report from workspace (it's in the message, but also try to read from file)
            error_report = self._extract_error_report(message_text, workspace_dir, arvo_id)
            
            # Sample code files from codebase
            code_samples = self._sample_codebase(codebase_dir)
            
            # Analyze with OpenAI
            await event_queue.enqueue_event(
                new_agent_text_message("Analyzing vulnerability with LLM...", context_id=context.context_id)
            )
            
            localization = await self._analyze_vulnerability(
                arvo_id=arvo_id,
                error_report=error_report,
                code_samples=code_samples,
            )
            
            # Write localization to file
            shared_dir = Path(workspace_dir) / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)
            loc_file = shared_dir / "loc.json"
            
            with open(loc_file, "w") as f:
                json.dump(localization, f, indent=2)
            
            logger.info(f"Created localization at {loc_file}")
            
            # Send confirmation
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"Analysis complete for arvo:{arvo_id}. Localization saved to {loc_file}",
                    context_id=context.context_id
                )
            )
            
        except Exception as e:
            logger.error(f"Error analyzing task {arvo_id}: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"Error analyzing vulnerability: {str(e)}",
                    context_id=context.context_id
                )
            )
    
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
    
    def _extract_codebase_dir(self, text: str) -> str | None:
        """Extract codebase directory from task description."""
        match = re.search(r'The vulnerable codebase is located at:\s*([^\n]+)', text)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_error_report(self, text: str, workspace_dir: str | None, arvo_id: str) -> str:
        """Extract error report from task description or read from file."""
        # Try to extract from the message (between "Fuzzer Crash Report:" and next section)
        match = re.search(r'Fuzzer Crash Report:\s*\n(.*?)(?:\n\nYour task:|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: try to read from workspace directory
        if workspace_dir:
            error_path = Path(workspace_dir) / f"{arvo_id}_error.txt"
            if error_path.exists():
                with open(error_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        
        return "Error report not found in message or file system."
    
    def _sample_codebase(self, codebase_dir: str, max_files: int = 10, max_lines_per_file: int = 200) -> dict[str, str]:
        """
        Sample code files from the codebase.
        
        Returns a dict mapping file paths to their content (truncated).
        """
        codebase_path = Path(codebase_dir)
        if not codebase_path.exists():
            logger.warning(f"Codebase directory not found: {codebase_dir}")
            return {}
        
        code_samples = {}
        code_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".rs", ".go", ".py"}
        
        # Find code files
        code_files = []
        for ext in code_extensions:
            code_files.extend(codebase_path.rglob(f"*{ext}"))
        
        # Limit number of files
        code_files = code_files[:max_files]
        
        for file_path in code_files:
            try:
                relative_path = str(file_path.relative_to(codebase_path))
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    # Truncate if too long
                    lines = content.split("\n")
                    if len(lines) > max_lines_per_file:
                        content = "\n".join(lines[:max_lines_per_file]) + f"\n... (truncated, {len(lines)} total lines)"
                    code_samples[relative_path] = content
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return code_samples
    
    async def _analyze_vulnerability(
        self,
        arvo_id: str,
        error_report: str,
        code_samples: dict[str, str],
    ) -> list[dict]:
        """
        Use OpenAI to analyze the vulnerability and generate localization.
        
        Returns a list of localization dictionaries.
        """
        # Build prompt
        code_context = ""
        for file_path, content in list(code_samples.items())[:5]:  # Limit to 5 files for context
            code_context += f"\n\n=== File: {file_path} ===\n{content}\n"
        
        prompt = f"""You are a security researcher performing root cause analysis on a vulnerable codebase.

Task ID: arvo:{arvo_id}

Fuzzer Crash Report:
{error_report}

Codebase Samples:
{code_context}

Your task:
1. Analyze the crash report to understand the vulnerability type (buffer overflow, use-after-free, etc.)
2. Examine the code samples to identify the root cause location
3. Identify the exact file, function, and line numbers where the vulnerability exists

Provide your analysis in the following JSON format:
{{
  "file": "relative/path/to/file.c",
  "old_span": {{"start": <line_number>, "end": <line_number>}},
  "new_span": {{"start": <line_number>, "end": <line_number>}},
  "function": "function_name",
  "reasoning": "Brief explanation of why this is the root cause"
}}

Important:
- File paths should be relative to the codebase root
- Line numbers are 1-indexed and inclusive
- Focus on the ROOT CAUSE, not just where the crash occurs
- The crash stack trace shows the symptom, not necessarily the bug location
- old_span and new_span are typically the same unless you're proposing a fix

Return ONLY valid JSON, no markdown formatting."""

        # Call OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert security researcher specializing in root cause analysis of software vulnerabilities. You analyze fuzzer crash reports and identify the exact location of security bugs in source code."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for more deterministic results
            response_format={"type": "json_object"},
        )
        
        result_text = response.choices[0].message.content
        logger.info(f"LLM response: {result_text[:500]}...")
        
        # Parse JSON response
        try:
            result = json.loads(result_text)
            
            # Convert to localization format
            localization = [{
                "task_id": f"arvo:{arvo_id}",
                "file": result.get("file", "unknown.c"),
                "old_span": result.get("old_span", {"start": 1, "end": 1}),
                "new_span": result.get("new_span", result.get("old_span", {"start": 1, "end": 1})),
                "function": result.get("function", ""),
            }]
            
            logger.info(f"Generated localization: {localization}")
            return localization
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {result_text}")
            # Fallback: try to extract information using regex
            return self._fallback_parse(arvo_id, result_text)
    
    def _fallback_parse(self, arvo_id: str, text: str) -> list[dict]:
        """Fallback parser if JSON parsing fails."""
        # Try to extract file and line numbers using regex
        file_match = re.search(r'"file"\s*:\s*"([^"]+)"', text)
        start_match = re.search(r'"start"\s*:\s*(\d+)', text)
        end_match = re.search(r'"end"\s*:\s*(\d+)', text)
        func_match = re.search(r'"function"\s*:\s*"([^"]*)"', text)
        
        return [{
            "task_id": f"arvo:{arvo_id}",
            "file": file_match.group(1) if file_match else "unknown.c",
            "old_span": {
                "start": int(start_match.group(1)) if start_match else 1,
                "end": int(end_match.group(1)) if end_match else 1,
            },
            "new_span": {
                "start": int(start_match.group(1)) if start_match else 1,
                "end": int(end_match.group(1)) if end_match else 1,
            },
            "function": func_match.group(1) if func_match else "",
        }]


class RCAFinderExecutor(AgentExecutor):
    """Executor for the RCA Finder purple agent."""
    
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.finder = RCAFinder(model=model, api_key=api_key)
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the task handling."""
        await self.finder.handle_task(context, event_queue)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution."""
        pass


def main():
    parser = argparse.ArgumentParser(description="Run the A2A RCA finder agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    args = parser.parse_args()
    
    agent_card = AgentCard(
        name="RCAFinder",
        description="Performs root cause analysis on vulnerable codebases using OpenAI.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=RCAFinderExecutor(model=args.model, api_key=args.api_key),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    import uvicorn
    uvicorn.run(
        server.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    asyncio.run(main())
