"""
Turn management for the tool calling loop.

Tracks turn counts, enforces limits, and manages turn history.
"""

import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TurnRecord:
    """Record of a single turn."""
    turn_number: int
    tool: str  # "bash" in our case
    command: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class TurnManager:
    """
    Manages turn counting and limits for the tool calling loop.
    """
    
    def __init__(self, max_turns: int = 150):
        """
        Initialize turn manager.
        
        Args:
            max_turns: Maximum number of turns allowed (default: 150)
        """
        self.max_turns = max_turns
        self.turn_count = 0
        self.history: list[TurnRecord] = []
        self.start_time = time.time()
        
    def can_continue(self) -> bool:
        """Check if more turns are allowed."""
        return self.turn_count < self.max_turns
        
    def record_turn(self, command: str, success: bool):
        """
        Record a turn.
        
        Args:
            command: The command that was executed
            success: Whether the command succeeded
        """
        self.turn_count += 1
        self.history.append(
            TurnRecord(
                turn_number=self.turn_count,
                tool="bash",
                command=command,
                success=success,
            )
        )
        
    def get_turns_remaining(self) -> int:
        """Get number of turns remaining."""
        return max(0, self.max_turns - self.turn_count)
        
    def get_warning_threshold(self) -> int:
        """Get turn count at which to warn (80% of max)."""
        return int(self.max_turns * 0.8)
        
    def should_warn(self) -> bool:
        """Check if we should warn about approaching turn limit."""
        return self.turn_count >= self.get_warning_threshold()
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
