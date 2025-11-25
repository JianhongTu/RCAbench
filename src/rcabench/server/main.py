#!/usr/bin/env python3
"""
main.py

CLI application for the RCAbench evaluation server.
Provides commands to start the server and validate patches using Docker containers.
"""

import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Literal
import shutil
import json

from .server_utils import run_arvo_container
from .eval_utils import (
    EvalReport,
    Localization,
    get_ground_truth,
    evaluate_localization,
)

app = FastAPI(
    title="RCAbench Evaluation Server",
    description="Server for evaluating patches in cybersecurity benchmarks",
)


@app.get("/")
def read_root():
    return {"message": "RCAbench Evaluation Server is running"}


class EvalRequest(BaseModel):
    arvo_id: str
    patch_dir: str = "./workspace/shared"


class PatchTestResponse(BaseModel):
    exit_code: int
    output: str
    success: bool
    message: str


@app.post("/patch", response_model=PatchTestResponse)
def evaluate_patch(request: EvalRequest):
    """
    Endpoint to evaluate a patch by running it in a Docker container.
    """
    try:
        patch_dir_path = Path(request.patch_dir)
        exit_code, docker_output = run_arvo_container(
            request.arvo_id, "vul", patch_dir_path
        )
        output_str = docker_output.decode("utf-8", errors="ignore")
        success = exit_code == 0
        message = "Validation successful" if success else "Validation failed"
        if exit_code == 300:
            message = "Validation timed out"
        return PatchTestResponse(
            exit_code=exit_code, output=output_str, success=success, message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")


def evaluate_patch_cli(arvo_id: str, patch_dir: str = "./tmp/patch"):
    """
    CLI function to evaluate a patch by running it in a Docker container.
    """
    try:
        patch_dir_path = Path(patch_dir)
        exit_code, docker_output = run_arvo_container(arvo_id, "vul", patch_dir_path)
        output_str = docker_output.decode("utf-8", errors="ignore")
        success = exit_code == 0
        message = "Validation successful" if success else "Validation failed"
        if exit_code == 300:
            message = "Validation timed out"
        print(f"Exit code: {exit_code}")
        print(f"Success: {success}")
        print(f"Message: {message}")
        print(f"Output:\n{output_str}")
    except Exception as e:
        print(f"Evaluation error: {e}")


@app.post("/evaluate", response_model=EvalReport)
def evaluate_final(request: EvalRequest):
    """
    Endpoint to evaluate localization by running it in a Docker container.
    """
    # Read localization submission
    try:
        loc_submission_path = request.patch_dir + "/loc.json"
        # Check if file exists
        if not Path(loc_submission_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Localization submission file not found at {loc_submission_path}",
            )

        # Load the submitted localizations
        with open(loc_submission_path, "r") as f:
            loc_data = f.read()
        loc_candidates = [Localization.from_dict(loc) for loc in json.loads(loc_data)]

        # Load and parse the ground truth
        loc_gts = get_ground_truth(request.arvo_id)

        # Evaluate localization
        report = evaluate_localization(loc_candidates, loc_gts)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCAbench Evaluation Server CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: start
    start_parser = subparsers.add_parser(
        "start", help="Start the FastAPI evaluation server"
    )
    start_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server to"
    )
    start_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    args = parser.parse_args()

    if args.command == "start":
        print(f"Starting RCAbench Evaluation Server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()
