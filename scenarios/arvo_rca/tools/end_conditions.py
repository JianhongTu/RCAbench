"""
End condition checking for the tool calling loop.

Determines when the loop should end (success, failure, timeout, etc.)
"""

import json
from pathlib import Path
from typing import Literal
import logging

from .turn_manager import TurnManager

logger = logging.getLogger(__name__)


class EndConditionChecker:
    """
    Checks various end conditions for the tool calling loop.
    """
    
    def __init__(
        self,
        turn_manager: TurnManager,
        shared_dir: Path,
        max_task_time: int = 600,  # 10 minutes default
    ):
        """
        Initialize end condition checker.
        
        Args:
            turn_manager: TurnManager instance
            shared_dir: Path to shared directory where submissions are written
            max_task_time: Maximum time for task in seconds
        """
        self.turn_manager = turn_manager
        self.shared_dir = Path(shared_dir)
        self.max_task_time = max_task_time
        self.start_time = turn_manager.start_time
        
    def check(self) -> dict:
        """
        Check all end conditions.
        
        Returns:
            dict with keys:
            - status: "continue" | "success" | "max_turns_exceeded" | "timeout" | "critical_error"
            - reason: Description of why loop ended
            - has_submission: bool (if status is success or partial)
        """
        # Check for success first (submission received)
        submission_check = self._check_submission()
        if submission_check["has_submission"]:
            return {
                "status": "success",
                "reason": "submission_received",
                "has_submission": True,
                "loc_file": submission_check.get("loc_file"),
                "reasoning_file": submission_check.get("reasoning_file"),
            }
        
        # Check failure conditions
        if not self.turn_manager.can_continue():
            return {
                "status": "max_turns_exceeded",
                "reason": f"Reached maximum turns ({self.turn_manager.max_turns})",
                "has_submission": False,
            }
        
        if self._is_timeout():
            return {
                "status": "timeout",
                "reason": f"Task exceeded maximum time ({self.max_task_time}s)",
                "has_submission": False,
            }
        
        # Continue
        return {
            "status": "continue",
            "reason": None,
            "has_submission": False,
        }
    
    def _check_submission(self) -> dict:
        """
        Check if valid submission files exist.
        
        Returns:
            dict with has_submission bool and file paths if found
        """
        loc_file = self.shared_dir / "loc.json"
        reasoning_file = self.shared_dir / "reasoning.json"
        
        has_loc = loc_file.exists()
        has_reasoning = reasoning_file.exists()
        
        if has_loc and has_reasoning:
            # Validate files are parseable
            try:
                with open(loc_file, "r") as f:
                    loc_data = json.load(f)
                # Check it's a list with at least one entry
                if isinstance(loc_data, list) and len(loc_data) > 0:
                    with open(reasoning_file, "r") as f:
                        reasoning_data = json.load(f)
                    # Basic validation - has task_id and reasoning_steps
                    if isinstance(reasoning_data, dict) and "reasoning_steps" in reasoning_data:
                        return {
                            "has_submission": True,
                            "loc_file": str(loc_file),
                            "reasoning_file": str(reasoning_file),
                        }
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Invalid submission files: {e}")
                return {"has_submission": False}
        
        return {"has_submission": False}
    
    def _is_timeout(self) -> bool:
        """Check if task has exceeded maximum time."""
        elapsed = self.turn_manager.get_elapsed_time()
        return elapsed >= self.max_task_time
