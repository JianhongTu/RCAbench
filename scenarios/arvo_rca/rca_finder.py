"""
RCA Finder (Purple Agent) that uses OpenHands to analyze vulnerabilities.

This implementation:
1. Receives task descriptions from the green agent
2. Parses the task information (arvo_id, workspace, codebase, error report)
3. Runs OpenHands agentic framework to analyze the vulnerability
4. Reads localization results from the shared directory
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
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

import tomllib
import tomli_w

# Helper functions (copied from run_white_agent.py to avoid importing openhands)
OPENAI_PREFIXES = ["gpt-", "o3", "o4"]
ANTHROPIC_PREFIXES = ["claude-"]


def get_api_key(model: str):
    """Get API key from environment based on model name."""
    if any(model.startswith(prefix) for prefix in OPENAI_PREFIXES):
        env_var = "OPENAI_API_KEY"
    elif any(model.startswith(prefix) for prefix in ANTHROPIC_PREFIXES):
        env_var = "ANTHROPIC_API_KEY"
    else:
        env_var = "LLM_API_KEY"
    
    api_key = os.getenv(env_var)
    return api_key if api_key else "EMPTY"


def model_map(model: str):
    """Map model name to OpenHands format."""
    if model.startswith("claude-"):
        return model
    elif len(model.split("/")) >= 2:
        return model
    return f"openai/{model}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rca_finder")

# OpenHands repo path (adjust if needed)
OPENHANDS_REPO = Path(__file__).parent.parent.parent / "agents" / "openhands" / "openhands-repo"
OPENHANDS_TEMPLATE = Path(__file__).parent.parent.parent / "agents" / "openhands" / "template"


class RCAFinder:
    """RCA Finder that uses OpenHands to analyze vulnerabilities."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_iter: int = 30,
        timeout: int | None = None,  # None = no timeout, allows indefinite execution
        openhands_repo: Path | None = None,
    ):
        self.model = model
        # Default to OpenAI API key from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY env var or pass --api-key")
        self.max_iter = max_iter
        self.timeout = timeout
        self.openhands_repo = openhands_repo or OPENHANDS_REPO
        self.template_dir = OPENHANDS_TEMPLATE
        
        # Check if OpenHands is set up
        if not self.openhands_repo.exists():
            logger.warning(f"OpenHands repo not found at {self.openhands_repo}")
            logger.warning("To set up OpenHands, run:")
            logger.warning("  cd agents/openhands")
            logger.warning("  git clone https://github.com/OpenHands/OpenHands.git openhands-repo")
            logger.warning("  cd openhands-repo")
            logger.warning("  git checkout c34030b2875da72f752906eec93b379fb7965d0c")
            logger.warning("  make build INSTALL_PLAYWRIGHT=false")
        
        # Verify browsergym is available (will be checked/installed at runtime if missing)
        # This is a required dependency that may be missing if dependencies weren't fully installed
    
    async def handle_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Handle a task request from the green agent.
        
        Runs OpenHands to analyze the vulnerability and generate localization results.
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
        
        if not arvo_id or not workspace_dir:
            error_msg = f"Could not parse task information. arvo_id={arvo_id}, workspace_dir={workspace_dir}"
            logger.error(error_msg)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {error_msg}", context_id=context.context_id)
            )
            return
        
        logger.info(f"Parsed task: arvo_id={arvo_id}, workspace_dir={workspace_dir}")
        
        # Send status update
        await event_queue.enqueue_event(
            new_agent_text_message(f"Starting OpenHands analysis for arvo:{arvo_id}...", context_id=context.context_id)
        )
        
        try:
            # Run OpenHands on the existing workspace
            await self._run_openhands_on_workspace(
                arvo_id=arvo_id,
                workspace_dir=Path(workspace_dir),
                event_queue=event_queue,
                context_id=context.context_id,
            )
            
            # Check if results were created
            shared_dir = Path(workspace_dir) / "shared"
            loc_file = shared_dir / "loc.json"
            
            if loc_file.exists():
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Analysis complete for arvo:{arvo_id}. Localization saved to {loc_file}",
                        context_id=context.context_id
                    )
                )
            else:
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Warning: OpenHands completed but no loc.json found at {loc_file}",
                        context_id=context.context_id
                    )
                )
                
        except Exception as e:
            logger.error(f"Error analyzing task {arvo_id}: {e}", exc_info=True)
            error_msg = f"Error analyzing task {arvo_id}: {str(e)}"
            try:
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        error_msg,
                        context_id=context.context_id
                    )
                )
            except Exception as queue_error:
                # If event queue is closed, log the error
                logger.warning(f"Could not send error message to event queue: {queue_error}")
            # Re-raise to ensure the error is propagated
            raise
    
    async def _run_openhands_on_workspace(
        self,
        arvo_id: str,
        workspace_dir: Path,
        event_queue: EventQueue,
        context_id: str,
    ) -> None:
        """
        Run OpenHands on an existing workspace directory.
        
        The workspace is already prepared by the green agent, so we just need to:
        1. Set up OpenHands config pointing to this workspace
        2. Run OpenHands
        3. Wait for completion
        """
        # Create temporary directory for OpenHands config and logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Copy template files
            template_dest = tmp_path / "template"
            if self.template_dir.exists():
                shutil.copytree(self.template_dir, template_dest)
            else:
                logger.error(f"OpenHands template directory not found: {self.template_dir}")
                raise FileNotFoundError(f"OpenHands template not found: {self.template_dir}")
            
            # Update prompt.txt with task ID and correct paths
            prompt_file = template_dest / "prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, "r") as f:
                    prompt_content = f.read()
                # Replace placeholder with actual task ID
                prompt_content = prompt_content.replace("arvo:XXXXX", f"arvo:{arvo_id}")
                # Update paths to match actual workspace structure
                # The workspace has: {arvo_id}_error.txt and src-vul/ directory
                error_file = f"{arvo_id}_error.txt"
                prompt_content = prompt_content.replace("/workspace/error.txt", f"/workspace/{error_file}")
                prompt_content = prompt_content.replace("/workspace/repo-vul/", "/workspace/src-vul/")
                with open(prompt_file, "w") as f:
                    f.write(prompt_content)
            
            # Configure OpenHands config.toml
            config_path = template_dest / "config.toml"
            with open(config_path, "r") as f:
                config = tomllib.loads(f.read())
            
            # Set workspace to the existing workspace directory
            config["core"]["workspace_base"] = str(workspace_dir.absolute()).replace("\\", "/")
            
            # Set cache and log directories
            log_dir = tmp_path / "logs"
            log_dir.mkdir()
            config["core"]["cache_dir"] = str((log_dir / "cache").absolute()).replace("\\", "/")
            config["core"]["file_store_path"] = str((log_dir / "file").absolute()).replace("\\", "/")
            config["core"]["save_trajectory_path"] = str((log_dir / "trajectory").absolute()).replace("\\", "/")
            
            # Configure LLM
            config["llm"]["model"] = model_map(self.model)
            config["llm"]["base_url"] = ""  # Use default OpenAI endpoint
            config["llm"]["max_output_tokens"] = 2048
            config["llm"]["top_p"] = 1.0
            config["llm"]["temperature"] = 0.0
            
            # Write updated config
            with open(config_path, "w") as f:
                f.write(tomli_w.dumps(config))
            
            # Update environment for OpenHands
            # Start with a copy of the current environment to preserve Poetry's settings
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = self.api_key or get_api_key(self.model)
            
            # Poetry needs these environment variables to work correctly
            # Don't override them if they're already set
            if "POETRY_VENV" not in env:
                # Poetry will find its venv automatically, but we can help it
                pass
            
            # Check and install OpenHands dependencies if needed (before running OpenHands)
            await event_queue.enqueue_event(
                new_agent_text_message("Checking OpenHands dependencies...", context_id=context_id)
            )
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    self._ensure_openhands_dependencies_installed,
                    env,
                )
            except Exception as deps_error:
                logger.error(f"Failed to ensure OpenHands dependencies are installed: {deps_error}", exc_info=True)
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Error: Failed to install required OpenHands dependencies: {str(deps_error)}",
                        context_id=context_id
                    )
                )
                raise
            
            # Run OpenHands in a thread pool (since it's blocking)
            await event_queue.enqueue_event(
                new_agent_text_message("Running OpenHands agentic analysis...", context_id=context_id)
            )
            
            # Run OpenHands synchronously in executor
            await loop.run_in_executor(
                None,
                self._run_openhands_sync,
                config_path,
                prompt_file,
                log_dir / "logs",
                env,
            )
    
    def _find_poetry(self) -> str:
        """Find poetry executable."""
        import shlex
        
        # Try multiple possible venv locations
        possible_venv_paths = [
            Path(__file__).parent.parent.parent / ".venv" / "bin" / "poetry",
            Path.cwd() / ".venv" / "bin" / "poetry",
            Path.home() / ".local" / "bin" / "poetry",
        ]
        
        poetry_path = None
        for venv_poetry in possible_venv_paths:
            if venv_poetry.exists() and venv_poetry.is_file():
                poetry_path = str(venv_poetry)
                break
        
        if not poetry_path:
            poetry_path = shutil.which("poetry")
        
        if not poetry_path:
            raise FileNotFoundError(
                f"Poetry not found. Tried: {[str(p) for p in possible_venv_paths]}. "
                "Install with: uv pip install poetry"
            )
        
        return poetry_path
    
    def _ensure_openhands_dependencies_installed(self, env: dict) -> None:
        """Ensure critical OpenHands dependencies are installed in the Poetry environment."""
        if not self.openhands_repo.exists():
            raise FileNotFoundError(f"OpenHands repo not found: {self.openhands_repo}")
        
        poetry_path = self._find_poetry()
        
        # List of critical dependencies to check and install if missing
        # These are packages that are commonly missing and cause import errors
        critical_deps = [
            ("termcolor", None),  # Required by logger.py
            ("browsergym", "0.13.3"),  # Required by browsing agents
        ]
        
        missing_deps = []
        
        # Check each dependency individually
        for dep_name, dep_version in critical_deps:
            check_cmd = [
                poetry_path,
                "run",
                "python",
                "-c",
                f"import {dep_name}; print('{dep_name} OK')",
            ]
            try:
                check_result = subprocess.run(
                    check_cmd,
                    cwd=self.openhands_repo,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout for import check
                )
                
                if check_result.returncode != 0:
                    missing_deps.append((dep_name, dep_version))
                    logger.warning(f"{dep_name} not found in Poetry environment")
                else:
                    logger.debug(f"{dep_name} is already installed")
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout checking {dep_name}, assuming it's missing")
                missing_deps.append((dep_name, dep_version))
        
        # Install missing dependencies
        if missing_deps:
            logger.info(f"Installing {len(missing_deps)} missing dependencies...")
            for dep_name, dep_version in missing_deps:
                # Use pip install within poetry environment to avoid dependency resolution issues
                install_spec = f"{dep_name}=={dep_version}" if dep_version else dep_name
                install_cmd = [
                    poetry_path,
                    "run",
                    "pip",
                    "install",
                    install_spec,
                ]
                logger.info(f"Installing {dep_name}...")
                try:
                    install_result = subprocess.run(
                        install_cmd,
                        cwd=self.openhands_repo,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout for installation
                    )
                except subprocess.TimeoutExpired:
                    error_msg = (
                        f"Timeout installing {dep_name} (exceeded 5 minutes).\n"
                        f"This may indicate network issues or Poetry environment problems.\n"
                        f"To fix manually, run:\n"
                        f"  cd {self.openhands_repo}\n"
                        f"  poetry run pip install {install_spec}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                if install_result.returncode != 0:
                    error_msg = (
                        f"Failed to install {dep_name}.\n"
                        f"Installation error: {install_result.stderr}\n"
                        f"Installation stdout: {install_result.stdout}\n"
                        f"To fix manually, run:\n"
                        f"  cd {self.openhands_repo}\n"
                        f"  poetry run pip install {install_spec}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                logger.info(f"{dep_name} installed successfully")
            
            # Verify all dependencies are now installed
            logger.info("Verifying all dependencies are installed...")
            verify_cmd = [
                poetry_path,
                "run",
                "python",
                "-c",
                "; ".join([f"import {name}" for name, _ in critical_deps]) + "; print('All dependencies OK')",
            ]
            try:
                verify_result = subprocess.run(
                    verify_cmd,
                    cwd=self.openhands_repo,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout for verification
                )
            except subprocess.TimeoutExpired:
                error_msg = (
                    f"Timeout verifying dependencies.\n"
                    f"Please verify manually:\n"
                    f"  cd {self.openhands_repo}\n"
                    f"  poetry run python -c \"import termcolor; import browsergym\""
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if verify_result.returncode != 0:
                error_msg = (
                    f"Dependencies were installed but verification failed.\n"
                    f"Verify stderr: {verify_result.stderr}\n"
                    f"Verify stdout: {verify_result.stdout}\n"
                    f"Please install manually:\n"
                    f"  cd {self.openhands_repo}\n"
                    f"  poetry run pip install {' '.join([f'{name}=={ver}' if ver else name for name, ver in critical_deps])}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info("All critical dependencies installed and verified successfully")
        else:
            logger.info("All critical dependencies are already installed")
    
    def _run_openhands_sync(
        self,
        config_path: Path,
        prompt_path: Path,
        log_dir: Path,
        env: dict,
    ) -> None:
        """Run OpenHands synchronously (called from executor)."""
        import shlex
        
        if not self.openhands_repo.exists():
            raise FileNotFoundError(f"OpenHands repo not found: {self.openhands_repo}")
        
        poetry_path = self._find_poetry()
        
        # Use poetry run - it handles the environment correctly
        # Ensure we're in the OpenHands repo directory so Poetry finds the right pyproject.toml
        cmd = [
            poetry_path,
            "run",
            "python",
            "-m",
            "openhands.core.main",
            "--config-file",
            str(config_path.absolute()),
            "--file",
            str(prompt_path.absolute()),
            "--max-iterations",
            str(self.max_iter),
        ]
        
        env["LOG_TO_FILE"] = "1"
        env["LOG_DIR"] = str(log_dir.absolute())
        env["LOG_ALL_EVENTS"] = "1"
        
        logger.info(f"Running OpenHands: {shlex.join(cmd)}")
        logger.info(f"Working directory: {self.openhands_repo}")
        
        try:
            # Only pass timeout if it's not None (None means no timeout)
            run_kwargs = {
                "cwd": self.openhands_repo,  # Critical: Poetry needs to be in the repo directory
                "env": env,
                "capture_output": True,
                "text": True,
            }
            if self.timeout is not None:
                run_kwargs["timeout"] = self.timeout
            
            result = subprocess.run(cmd, **run_kwargs)
            
            if result.returncode != 0:
                logger.error(f"OpenHands failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"OpenHands failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("OpenHands timed out")
            raise TimeoutError("OpenHands analysis timed out")
    
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


class RCAFinderExecutor(AgentExecutor):
    """Executor for the RCA Finder purple agent."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_iter: int = 30,
        timeout: int | None = None,  # None = no timeout, allows indefinite execution
        openhands_repo: Path | None = None,
    ):
        self.finder = RCAFinder(
            model=model,
            api_key=api_key,
            max_iter=max_iter,
            timeout=timeout,
            openhands_repo=openhands_repo,
        )
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the task handling."""
        await self.finder.handle_task(context, event_queue)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution."""
        pass


def main():
    parser = argparse.ArgumentParser(description="Run the A2A RCA finder agent with OpenHands.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--max-iter", type=int, default=30, help="Max OpenHands iterations (default: 30)")
    parser.add_argument("--timeout", type=int, default=None, help="OpenHands timeout in seconds (default: None = no timeout)")
    parser.add_argument("--openhands-repo", type=str, help="Path to OpenHands repository")
    args = parser.parse_args()
    
    openhands_repo = Path(args.openhands_repo) if args.openhands_repo else None
    
    agent_card = AgentCard(
        name="RCAFinder",
        description="Performs root cause analysis on vulnerable codebases using OpenHands agentic framework.",
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
            max_iter=args.max_iter,
            timeout=args.timeout,
            openhands_repo=openhands_repo,
        ),
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
        timeout_keep_alive=None,  # No timeout - allows long-running RCA analysis
    )


if __name__ == "__main__":
    main()
