"""
Custom Docker environment for mini-swe-agent that executes commands in arvo containers.

This environment executes commands in the arvo Docker container (C/C++ compilers),
while files remain on the local filesystem (mounted as volumes).
"""

import docker
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of executing a command in Docker."""
    returncode: int
    output: str
    error: str = ""
    execution_time: float = 0.0


class ArvoDockerEnvironment:
    """
    Docker environment that executes commands in arvo containers.
    
    Commands are executed in the arvo container (C/C++ compilers),
    while files remain on the local filesystem (mounted as volumes).
    """
    
    def __init__(
        self,
        arvo_id: str,
        workspace_dir: Path,
        docker_image: Optional[str] = None,
        container_timeout: int = 30,
        command_timeout: int = 120,
    ):
        """
        Initialize Docker environment.
        
        Args:
            arvo_id: ARVO task ID
            workspace_dir: Local workspace directory (will be mounted in container)
            docker_image: Docker image name (defaults to n132/arvo:{arvo_id}-vul)
            container_timeout: Timeout for container operations
            command_timeout: Timeout for individual commands
        """
        self.arvo_id = arvo_id
        self.workspace_dir = Path(workspace_dir).resolve()
        self.docker_image = docker_image or f"n132/arvo:{arvo_id}-vul"
        self.container_timeout = container_timeout
        self.command_timeout = command_timeout
        
        # Initialize Docker client with better error handling
        try:
            self.client = docker.from_env()
            # Test connection by checking if Docker daemon is accessible
            self.client.ping()
        except docker.errors.DockerException as e:
            error_msg = (
                f"Docker daemon is not accessible. Please ensure Docker is running.\n"
                f"Error: {str(e)}\n\n"
                f"To fix:\n"
                f"  1. Start Docker Desktop (macOS: 'open -a Docker')\n"
                f"  2. Wait for Docker to fully start\n"
                f"  3. Verify with: docker ps\n"
                f"  4. Check Docker socket: ls -la /var/run/docker.sock"
            )
            raise docker.errors.DockerException(error_msg) from e
        except FileNotFoundError as e:
            error_msg = (
                f"Docker socket not found. Docker daemon is not running.\n"
                f"Error: {str(e)}\n\n"
                f"To fix:\n"
                f"  1. Start Docker Desktop (macOS: 'open -a Docker')\n"
                f"  2. Wait for Docker to fully start\n"
                f"  3. Verify with: docker ps"
            )
            raise FileNotFoundError(error_msg) from e
        
        self.container = None
        
        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ArvoDockerEnvironment for {arvo_id}")
        logger.info(f"Workspace: {self.workspace_dir}")
        logger.info(f"Docker image: {self.docker_image}")
    
    def _ensure_container_running(self):
        """Ensure the arvo container is running. Creates a new container for this task."""
        if self.container is None:
            try:
                # Always create a new container for this task to ensure isolation
                # Each task gets its own container, even if they share the same image
                self.container = self.client.containers.run(
                    image=self.docker_image,
                    command="/bin/sh",  # Keep container running
                    volumes={
                        str(self.workspace_dir): {
                            "bind": "/workspace",
                            "mode": "rw"
                        }
                    },
                    detach=True,
                    tty=True,
                    stdin_open=True,
                    # Use a unique name to avoid conflicts
                    name=f"arvo_{self.arvo_id}_{id(self)}",
                    remove=False,  # Don't auto-remove, we'll clean it up explicitly
                )
                logger.info(f"Created new container: {self.container.id[:12]} for task {self.arvo_id}")
            except docker.errors.ImageNotFound:
                logger.warning(f"Image {self.docker_image} not found, pulling...")
                self.client.images.pull(self.docker_image)
                # Retry container creation (always create new, no reuse)
                self.container = self.client.containers.run(
                    image=self.docker_image,
                    command="/bin/sh",
                    volumes={
                        str(self.workspace_dir): {
                            "bind": "/workspace",
                            "mode": "rw"
                        }
                    },
                    detach=True,
                    tty=True,
                    stdin_open=True,
                    name=f"arvo_{self.arvo_id}_{id(self)}",
                    remove=False,
                )
                logger.info(f"Created container after pull: {self.container.id[:12]} for task {self.arvo_id}")
    
    def execute(self, command: str, cwd: Optional[str] = None) -> CommandResult:
        """
        Execute a command in the arvo Docker container.
        
        Args:
            command: Bash command to execute
            cwd: Working directory (relative to /workspace in container)
        
        Returns:
            CommandResult with returncode, output, and error
        """
        self._ensure_container_running()
        
        start_time = time.time()
        
        # Prepare command with working directory
        if cwd:
            # Convert local path to container path
            if str(cwd).startswith(str(self.workspace_dir)):
                container_cwd = str(cwd).replace(str(self.workspace_dir), "/workspace")
            else:
                container_cwd = f"/workspace/{cwd}"
            full_command = f"cd {container_cwd} && {command}"
        else:
            full_command = f"cd /workspace && {command}"
        
        try:
            # Execute command in container
            # Note: exec_run() doesn't support timeout parameter directly
            # Timeout is handled at the Docker client level if needed
            exec_result = self.container.exec_run(
                cmd=["/bin/sh", "-c", full_command],
                workdir="/workspace",
            )
            
            execution_time = time.time() - start_time
            
            output = exec_result.output.decode("utf-8", errors="replace") if exec_result.output else ""
            
            return CommandResult(
                returncode=exec_result.exit_code,
                output=output,
                error="",  # Docker exec_run combines stdout/stderr
                execution_time=execution_time,
            )
        except docker.errors.APIError as e:
            execution_time = time.time() - start_time
            logger.error(f"Docker API error executing command: {e}")
            return CommandResult(
                returncode=-1,
                output="",
                error=str(e),
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing command in Docker: {e}")
            return CommandResult(
                returncode=-1,
                output="",
                error=str(e),
                execution_time=execution_time,
            )
    
    def cleanup(self):
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.stop(timeout=10)
                self.container.remove()
                logger.info(f"Cleaned up container: {self.container.id[:12]}")
            except Exception as e:
                logger.warning(f"Error cleaning up container: {e}")
            finally:
                self.container = None

