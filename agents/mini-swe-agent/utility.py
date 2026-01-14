"""
Shared utility functions for mini-swe-agent.

Provides consistent log directory creation, timestamp formatting, and task ID management
across test scripts and agent servers.
"""

import os
import random
import time
from pathlib import Path
from typing import List, Optional

# Constants
CURRENT_RUN_LOG_DIR_MARKER = ".current_run_log_dir"
RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def get_run_timestamp() -> str:
    """Get the current run timestamp in a consistent format."""
    return time.strftime(RUN_TIMESTAMP_FORMAT)


def get_base_log_dir(scenario_file: Optional[Path] = None) -> Path:
    """
    Get the base log directory from scenario.toml or environment variable.
    
    Args:
        scenario_file: Path to scenario.toml file. If None, looks for it in the same directory as this module.
    
    Returns:
        Path to the base log directory
    """
    # Check environment variable first (but only if it's a base directory, not a run directory)
    # If LOG_DIR points to a run directory (contains 'log_' or 'run_'), use its parent
    log_dir_env = os.getenv("LOG_DIR")
    if log_dir_env:
        log_dir_path = Path(log_dir_env).resolve()
        # If it's a run directory (not base), use parent
        if log_dir_path.name.startswith('log_') or log_dir_path.name.startswith('run_'):
            return log_dir_path.parent
        return log_dir_path
    
    # Try to read from scenario.toml
    if scenario_file is None:
        scenario_file = Path(__file__).parent / "scenario.toml"
    
    if scenario_file.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # Fallback for Python < 3.11
            except ImportError:
                tomllib = None
        
        if tomllib:
            with open(scenario_file, "rb") as f:
                config = tomllib.load(f)
            log_dir_from_config = config.get("config", {}).get("log_dir", "./logs")
            return (scenario_file.parent / log_dir_from_config).resolve()
    
    # Default fallback
    return Path("./logs").resolve()


def get_or_create_run_log_dir(
    base_log_dir: Optional[Path] = None,
    arvo_id: Optional[str] = None,
    scenario_file: Optional[Path] = None
) -> Path:
    """
    Get or create the run log directory.
    
    Priority:
    1. Check for marker file (created by test script)
    2. Create new directory with timestamp (log_{timestamp})
    
    Note: arvo_id parameter is ignored - all tasks in a run share the same log directory.
    Per-ARVO logs are stored as separate files (arvo_{arvo_id}.log) within the directory.
    
    Args:
        base_log_dir: Base log directory. If None, will be determined from scenario.toml or env.
        arvo_id: ARVO task ID (ignored, kept for backward compatibility)
        scenario_file: Path to scenario.toml file (optional)
    
    Returns:
        Path to the run log directory
    """
    if base_log_dir is None:
        base_log_dir = get_base_log_dir(scenario_file)
    
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for marker file created by test script
    marker_file = base_log_dir / CURRENT_RUN_LOG_DIR_MARKER
    if marker_file.exists():
        with open(marker_file, "r") as f:
            marker_path = f.read().strip()
        # Resolve to absolute path to avoid nested directory issues
        # If marker_path is relative, resolve it relative to base_log_dir
        # If it's absolute, resolve() will keep it absolute
        if not Path(marker_path).is_absolute():
            run_log_dir = (base_log_dir / marker_path).resolve()
        else:
            run_log_dir = Path(marker_path).resolve()
        run_log_dir.mkdir(parents=True, exist_ok=True)
        return run_log_dir
    
    # Create new directory with timestamp (not per-ARVO)
    timestamp = get_run_timestamp()
    run_log_dir = base_log_dir / f"log_{timestamp}"
    
    run_log_dir.mkdir(parents=True, exist_ok=True)
    return run_log_dir


def create_run_log_dir(
    arvo_id: str,
    scenario_file: Optional[Path] = None
) -> Path:
    """
    Create a run log directory and write marker file.
    
    This should be called by scripts (test or production) to create the log directory
    and marker file that agents will use. The marker file allows agents to find
    the correct log directory for the current run.
    
    Note: arvo_id is used for reference but the directory is timestamped only.
    Per-ARVO logs are stored as separate files (arvo_{arvo_id}.log) within the directory.
    
    Args:
        arvo_id: ARVO task ID (used for reference, not in directory name)
        scenario_file: Path to scenario.toml file (optional)
    
    Returns:
        Path to the created run log directory
    """
    base_log_dir = get_base_log_dir(scenario_file)
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique log directory for this run (timestamp only, not per-ARVO)
    timestamp = get_run_timestamp()
    run_log_dir = base_log_dir / f"log_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Write run log directory path to marker file that agents can read
    # Always write absolute path to avoid resolution issues
    marker_file = base_log_dir / CURRENT_RUN_LOG_DIR_MARKER
    with open(marker_file, "w") as f:
        f.write(str(run_log_dir.resolve()))
    
    # Also set environment variable for agents (if they check it)
    os.environ["LOG_DIR"] = str(run_log_dir)
    
    return run_log_dir


def get_task_ids_from_config(scenario_file: Optional[Path] = None) -> List[str]:
    """
    Get task IDs from scenario.toml.
    
    Priority:
    1. task_ids_file (if uncommented) -> reads file, returns all IDs
    2. task_ids (if task_ids_file not found) -> returns list from TOML
    
    Args:
        scenario_file: Path to scenario.toml file. If None, looks for it in the same directory as this module.
    
    Returns:
        List of task ID strings (empty list if none found)
    """
    if scenario_file is None:
        scenario_file = Path(__file__).parent / "scenario.toml"
    
    if not scenario_file.exists():
        return []
    
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Fallback for Python < 3.11
        except ImportError:
            return []
    
    try:
        with open(scenario_file, "rb") as f:
            config = tomllib.load(f)
        config_section = config.get("config", {})
        
        # Check for task_ids_file first
        task_ids_file_rel = config_section.get("task_ids_file")
        if task_ids_file_rel:
            # Resolve path relative to scenario.toml
            task_ids_file_path = (scenario_file.parent / task_ids_file_rel).resolve()
            if task_ids_file_path.exists():
                # Read task IDs from file (one per line)
                with open(task_ids_file_path, "r") as f:
                    lines = f.readlines()
                # Parse task IDs (strip whitespace, skip empty lines)
                task_ids = [line.strip() for line in lines if line.strip()]
                return task_ids
            else:
                # File not found - fall through to task_ids
                pass
        
        # Fall back to task_ids list
        task_ids = config_section.get("task_ids", [])
        return [str(tid) for tid in task_ids]
    except Exception:
        return []


def get_random_task_id_from_config(scenario_file: Optional[Path] = None) -> Optional[str]:
    """
    Get a single random task ID from config (for log directory).
    
    This is useful for start_agents.py which only needs one ARVO ID
    to create the log directory marker file.
    
    Args:
        scenario_file: Path to scenario.toml file. If None, looks for it in the same directory as this module.
    
    Returns:
        Randomly selected task ID string, or None if none found
    """
    task_ids = get_task_ids_from_config(scenario_file)
    if task_ids:
        return random.choice(task_ids)
    return None

