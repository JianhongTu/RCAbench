"""
Docker sandbox manager for executing commands in isolated containers.

Uses Arvo's Docker images to provide a sandboxed environment for bash command execution.
"""

import docker
from pathlib import Path
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class DockerSandbox:
    """
    Manages a long-lived Docker container for executing bash commands.
    
    Uses Arvo's vulnerable Docker image and mounts the workspace directory.
    """
    
    def __init__(
        self,
        arvo_id: str,
        workspace_dir: Path,
        mode: Literal["vul", "fix"] = "vul",
    ):
        """
        Initialize sandbox.
        
        Args:
            arvo_id: Arvo task ID
            workspace_dir: Path to workspace directory (will be mounted)
            mode: "vul" for vulnerable version, "fix" for fixed version
        """
        self.arvo_id = arvo_id
        self.workspace_dir = Path(workspace_dir).resolve()
        self.mode = mode
        self.image = f"n132/arvo:{arvo_id}-{mode}"
        self.container = None
        self.client = docker.from_env()
        
    async def start(self):
        """Start the Docker container."""
        try:
            # Check if image exists, pull if needed
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling Docker image: {self.image}")
                self.client.images.pull(self.image)
            
            # Create container with workspace mounted
            # Use a long-running command to keep container alive
            self.container = self.client.containers.create(
                image=self.image,
                command=["tail", "-f", "/dev/null"],  # Keep container running
                volumes={
                    str(self.workspace_dir): {"bind": "/workspace", "mode": "rw"}
                },
                working_dir="/workspace",
                detach=True,
                # Resource limits
                mem_limit="2g",
                cpu_count=1,
            )
            
            # Start the container
            self.container.start()
            logger.info(f"Started Docker container: {self.container.id[:12]}")
            
        except docker.errors.DockerException as e:
            logger.error(f"Failed to start Docker container: {e}")
            raise RuntimeError(f"Failed to start Docker container: {e}") from e
        
    async def execute_command(
        self,
        command: str,
        cwd: str = "/workspace",
        timeout: int = 30,
    ) -> dict:
        """
        Execute a bash command in the container.
        
        Args:
            command: Bash command to execute
            cwd: Working directory in container
            timeout: Command timeout in seconds
            
        Returns:
            dict with keys: stdout, stderr, exit_code
        """
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")
        
        try:
            # Execute command in the running container
            # Use sh -c to run the command as a shell command
            # exec_run expects a list for the command
            # Run in executor to avoid blocking event loop
            # Note: exec_run doesn't support timeout parameter, use asyncio timeout instead
            import asyncio
            loop = asyncio.get_event_loop()
            
            async def run_with_timeout():
                return await loop.run_in_executor(
                    None,
                    lambda: self.container.exec_run(
                        ["sh", "-c", command],
                        workdir=cwd,
                    )
                )
            
            exec_result = await asyncio.wait_for(run_with_timeout(), timeout=timeout)
            
            # exec_run returns (exit_code, output)
            exit_code = exec_result.exit_code
            output = exec_result.output
            
            # Decode output (it's bytes)
            if isinstance(output, bytes):
                output_str = output.decode("utf-8", errors="replace")
            else:
                output_str = str(output)
            
            # For now, we don't separate stdout/stderr in exec_run
            # Both go to stdout. This is a limitation of docker exec_run.
            return {
                "stdout": output_str,
                "stderr": "",  # exec_run doesn't separate stderr
                "exit_code": exit_code,
            }
            
        except docker.errors.ContainerError as e:
            logger.error(f"Container error executing command: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
            }
        
    async def cleanup(self):
        """Stop and remove the container."""
        if self.container is None:
            return
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Stop the container (run in executor to avoid blocking)
            await loop.run_in_executor(None, lambda: self.container.stop(timeout=10))
            logger.info(f"Stopped Docker container: {self.container.id[:12]}")
            
            # Remove the container
            await loop.run_in_executor(None, lambda: self.container.remove(force=True))
            logger.info(f"Removed Docker container: {self.container.id[:12]}")
            
        except docker.errors.NotFound:
            # Container already removed
            logger.debug("Container already removed")
        except Exception as e:
            logger.warning(f"Error cleaning up container: {e}")
            # Try to force remove
            try:
                self.container.remove(force=True)
            except Exception:
                pass
        finally:
            self.container = None
