"""
Bash command executor with security validation.

Validates and executes bash commands in a sandboxed environment.
"""

import shlex
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BashExecutor:
    """
    Executes bash commands with security validation.
    
    Uses whitelist/blacklist approach to ensure only safe commands are executed.
    """
    
    # Whitelist of safe commands (read-only operations)
    COMMAND_WHITELIST = {
        "cat", "head", "tail", "grep", "find", "ls", "wc",
        "sed", "awk", "cut", "sort", "uniq", "diff",
        "file", "stat", "readlink", "realpath",
        "echo", "printf",  # For writing to shared directory
    }
    
    # Blacklist of dangerous commands
    COMMAND_BLACKLIST = {
        "rm", "rmdir", "mv", "cp", "chmod", "chown", "chgrp",
        "wget", "curl", "nc", "netcat", "ssh", "scp",
        "python", "python3", "node", "npm",  # Prevent code execution
        "sh", "bash",  # Prevent nested shells (unless explicitly allowed)
    }
    
    def __init__(self, sandbox):
        """
        Initialize bash executor.
        
        Args:
            sandbox: DockerSandbox instance for command execution
        """
        self.sandbox = sandbox
        
    def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a command is safe to execute.
        
        Args:
            command: Bash command to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
        """
        if not command or not command.strip():
            return False, "Empty command"
        
        try:
            # Parse command safely
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command after parsing"
            
            cmd = parts[0]
            
            # Check blacklist first
            if cmd in self.COMMAND_BLACKLIST:
                return False, f"Command '{cmd}' is blacklisted"
            
            # Check whitelist
            if cmd not in self.COMMAND_WHITELIST:
                return False, f"Command '{cmd}' not in whitelist"
            
            # Check for path traversal
            for part in parts[1:]:
                if '..' in part:
                    return False, f"Path traversal detected: '{part}'"
                # Check for absolute paths outside workspace
                if part.startswith('/') and not part.startswith('/workspace'):
                    return False, f"Path outside workspace: '{part}'"
            
            return True, None
            
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    async def execute(self, command: str, cwd: str = "/workspace") -> dict:
        """
        Execute a bash command.
        
        Args:
            command: Bash command to execute
            cwd: Working directory in container
            
        Returns:
            dict with keys: success, stdout, stderr, exit_code, error
        """
        # Validate command
        is_valid, error = self.validate_command(command)
        if not is_valid:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": 1,
                "error": error,
            }
        
        # Execute in sandbox
        try:
            result = await self.sandbox.execute_command(command, cwd=cwd)
            return {
                "success": result.get("exit_code", 1) == 0,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result.get("exit_code", 1),
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exit_code": 1,
                "error": str(e),
            }
