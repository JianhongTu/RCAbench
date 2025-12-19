#!/usr/bin/env python3
"""
Purple Agent for RCAbench - AgentBeats Compatible

This purple agent implements the A2A (Agent-to-Agent) protocol to participate
in AgentBeats assessments. It performs root cause analysis on vulnerable codebases.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Add parent directory to path to import rcabench modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
    from rcabench.server.eval_utils import Localization, LineSpan
except ImportError:
    # Fallback: try installing rcabench as a package
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(project_root)])
    from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
    from rcabench.server.eval_utils import Localization, LineSpan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RCAbench Purple Agent")

# Global state
current_task_id: Optional[str] = None
agent_card_url: Optional[str] = None


class AssessmentRequest(BaseModel):
    """A2A Assessment Request Model"""
    participants: Dict[str, str]
    config: Dict[str, Any]


class TaskUpdate(BaseModel):
    """A2A Task Update Model"""
    content: str
    level: str = "info"


class Artifact(BaseModel):
    """A2A Artifact Model"""
    name: str
    content_type: str
    content: Dict[str, Any]


class A2AMessage(BaseModel):
    """A2A Message Model"""
    task_id: str
    role: str
    content: Dict[str, Any]


def analyze_vulnerability(arvo_id: str, workspace_dir: Path) -> Dict[str, Any]:
    """
    Perform root cause analysis on the vulnerable codebase.
    
    This is a simplified implementation. In a real scenario, you would:
    1. Use an LLM to analyze the crash report
    2. Examine the codebase
    3. Identify the vulnerability location
    4. Optionally generate a patch
    
    Args:
        arvo_id: The ARVO task identifier
        workspace_dir: Path to the workspace containing the codebase and error report
        
    Returns:
        Dictionary containing localization results and optional patch
    """
    logger.info(f"Analyzing vulnerability for task {arvo_id}")
    
    # Read the error report
    error_file = workspace_dir / f"{arvo_id}_error.txt"
    if not error_file.exists():
        # Try alternative naming
        error_file = workspace_dir / "error.txt"
    
    error_content = ""
    if error_file.exists():
        with open(error_file, "r") as f:
            error_content = f.read()
        logger.info(f"Read error report: {len(error_content)} characters")
    else:
        logger.warning(f"Error report not found at {error_file}")
    
    # Find the codebase directory
    codebase_dir = None
    for item in workspace_dir.iterdir():
        if item.is_dir() and item.name not in ["shared", ".git"]:
            codebase_dir = item
            break
    
    if not codebase_dir:
        logger.warning("Codebase directory not found")
        codebase_dir = workspace_dir
    
    # Simplified analysis - in a real implementation, you would use an LLM here
    # For now, we'll create a placeholder localization
    localizations = []
    
    # Try to find C/C++ source files
    source_files = list(codebase_dir.rglob("*.c")) + list(codebase_dir.rglob("*.cpp"))
    if source_files:
        # Use the first source file as a placeholder
        first_file = source_files[0]
        rel_path = first_file.relative_to(codebase_dir)
        
        # Read a few lines to estimate vulnerability location
        try:
            with open(first_file, "r") as f:
                lines = f.readlines()
                if lines:
                    # Placeholder: assume vulnerability is in first 100 lines
                    start_line = min(1, len(lines))
                    end_line = min(100, len(lines))
                    
                    localizations.append({
                        "task_id": f"arvo:{arvo_id}",
                        "file": str(rel_path),
                        "old_span": {"start": start_line, "end": end_line},
                        "new_span": {"start": start_line, "end": end_line},
                        "function": ""
                    })
        except Exception as e:
            logger.error(f"Error reading source file: {e}")
    
    result = {
        "arvo_id": arvo_id,
        "localizations": localizations,
        "analysis": {
            "error_report_length": len(error_content),
            "codebase_files_found": len(source_files),
            "status": "completed"
        }
    }
    
    logger.info(f"Analysis complete: {len(localizations)} localizations found")
    return result


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "agent_type": "purple",
        "capabilities": ["root_cause_analysis", "vulnerability_localization"]
    }


@app.get("/card")
async def get_card():
    """Return agent card (A2A protocol)"""
    if not agent_card_url:
        raise HTTPException(status_code=500, detail="Agent card URL not configured")
    
    return {
        "name": "RCAbench Purple Agent",
        "description": "A purple agent for root cause analysis of vulnerable codebases",
        "url": agent_card_url,
        "capabilities": ["root_cause_analysis", "vulnerability_localization"]
    }


@app.post("/assessments")
async def handle_assessment_request(request: AssessmentRequest):
    """
    Handle A2A assessment request from green agent.
    
    This endpoint receives an assessment request, performs the analysis,
    and returns results as A2A artifacts.
    """
    global current_task_id
    
    logger.info(f"Received assessment request: {request.config}")
    
    # Extract task configuration
    config = request.config
    arvo_id = config.get("arvo_id")
    
    if not arvo_id:
        raise HTTPException(
            status_code=400,
            detail="Missing 'arvo_id' in assessment config"
        )
    
    current_task_id = f"arvo:{arvo_id}"
    
    # Create temporary workspace
    temp_dir = Path(tempfile.mkdtemp(prefix=f"rca_{arvo_id}_"))
    agent_paths = None
    
    try:
        # Prepare task assets
        logger.info(f"Preparing task assets for {arvo_id}")
        task_meta = prepare_task_assets(
            arvo_id=arvo_id,
            tmp_dir=temp_dir,
            host_ip="",  # Not needed for purple agent
            host_port=0,
        )
        
        agent_paths = task_meta.get("agent_paths")
        workspace_dir = agent_paths.workspace_dir if agent_paths else Path(task_meta["codebase_path"])
        
        # Perform analysis
        logger.info("Starting vulnerability analysis")
        analysis_result = analyze_vulnerability(arvo_id, workspace_dir)
        
        # Create A2A artifacts
        artifacts = []
        
        # Artifact 1: Localization results
        loc_artifact = {
            "name": "localization_results",
            "content_type": "application/json",
            "content": {
                "task_id": current_task_id,
                "localizations": analysis_result["localizations"],
                "analysis_metadata": analysis_result["analysis"]
            }
        }
        artifacts.append(loc_artifact)
        
        # Artifact 2: Summary
        summary_artifact = {
            "name": "analysis_summary",
            "content_type": "application/json",
            "content": {
                "task_id": current_task_id,
                "status": "completed",
                "localizations_count": len(analysis_result["localizations"]),
                "analysis": analysis_result["analysis"]
            }
        }
        artifacts.append(summary_artifact)
        
        logger.info(f"Assessment complete: {len(artifacts)} artifacts created")
        
        return {
            "task_id": current_task_id,
            "status": "completed",
            "artifacts": artifacts,
            "updates": [
                {
                    "content": f"Analysis complete for task {arvo_id}",
                    "level": "info"
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error during assessment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Assessment failed: {str(e)}"
        )
    finally:
        # Cleanup
        if agent_paths:
            try:
                cleanup_task_assets(agent_paths)
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
        
        # Remove temp directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error removing temp directory: {e}")


@app.post("/tasks/{task_id}/updates")
async def send_task_update(task_id: str, update: TaskUpdate):
    """Handle task updates (A2A protocol)"""
    logger.info(f"Task update for {task_id}: {update.content}")
    return {"status": "received"}


def main():
    """Main entry point for the purple agent server"""
    parser = argparse.ArgumentParser(description="RCAbench Purple Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument(
        "--card-url",
        help="URL to advertise in agent card (e.g., http://your-ip:8001)"
    )
    
    args = parser.parse_args()
    
    global agent_card_url
    if args.card_url:
        agent_card_url = args.card_url
    else:
        # Auto-detect card URL
        agent_card_url = f"http://{args.host}:{args.port}"
    
    logger.info(f"Starting RCAbench Purple Agent on {args.host}:{args.port}")
    logger.info(f"Agent card URL: {agent_card_url}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

