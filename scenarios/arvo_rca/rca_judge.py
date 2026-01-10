import json
import logging
import asyncio
import os
import argparse
import uvicorn
import shlex
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# A2A and AgentBeats imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

from agentbeats.models import EvalRequest, EvalResult
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.tool_provider import ToolProvider

# Local imports
try:
    from rca_judge_common import TaskResult, ReasoningTrace
except ImportError:
    from scenarios.arvo_rca.rca_judge_common import TaskResult, ReasoningTrace

try:
    from tools.bash_executor import BashExecutor
    from tools.docker_sandbox import DockerSandbox
    from tools.turn_manager import TurnManager
    from tools.end_conditions import EndConditionChecker
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from tools.bash_executor import BashExecutor
    from tools.docker_sandbox import DockerSandbox
    from tools.turn_manager import TurnManager
    from tools.end_conditions import EndConditionChecker

# Import EvalReport from rcabench
try:
    from rcabench.server.eval_utils import EvalReport, evaluate_localization, get_ground_truth, Localization, LineSpan
except ImportError:
    from src.rcabench.server.eval_utils import EvalReport, evaluate_localization, get_ground_truth, Localization, LineSpan

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger("rca_judge")
trace_logger = logging.getLogger("rca_judge.trace")

class RCAJudge(GreenAgent):
    """Green Agent that orchestrates RCA tasks and provides bash tools."""
    
    def __init__(self, trace_conversation: bool = False, trace_only: bool = False):
        self._tool_provider = ToolProvider()
        self._trace_conversation = trace_conversation
        self._trace_only = trace_only
        self._conversation_trace = []
        
        if trace_conversation:
            trace_logger.setLevel(logging.INFO)
            if not trace_logger.handlers:
                handler = logging.StreamHandler()
                # In trace-only mode, don't include timestamp/logger name
                if trace_only:
                    handler.setFormatter(logging.Formatter('%(message)s'))
                trace_logger.addHandler(handler)
                trace_logger.propagate = False  # Don't pass to parent loggers

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        if "task_ids" not in request.config and "task_ids_file" not in request.config:
            return False, "No task_ids or task_ids_file provided in config"
        return True, ""

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """Run evaluation for the requested tasks."""
        results = []
        task_ids = request.config.get("task_ids", [])
        if not task_ids:
            task_ids_file = request.config.get("task_ids_file")
            if task_ids_file and os.path.exists(task_ids_file):
                with open(task_ids_file, "r") as f:
                    if task_ids_file.endswith(".json"):
                        task_ids = json.load(f)
                    else:
                        task_ids = [line.strip() for line in f if line.strip()]
        
        # Filter tasks to only those with ground truth available
        logger.info(f"Loaded {len(task_ids)} task IDs, filtering for ground truth availability...")
        valid_task_ids = []
        for arvo_id in task_ids:
            try:
                gts = get_ground_truth(arvo_id)
                if gts and len(gts) > 0:
                    valid_task_ids.append(arvo_id)
            except Exception as e:
                logger.debug(f"Task {arvo_id} has no ground truth: {e}")
                continue
        
        logger.info(f"Found {len(valid_task_ids)} tasks with ground truth")
        
        # Respect num_tasks limit from config (with random sampling)
        num_tasks = request.config.get("num_tasks")
        if num_tasks is not None and num_tasks > 0:
            if num_tasks < len(valid_task_ids):
                valid_task_ids = random.sample(valid_task_ids, num_tasks)
                logger.info(f"Randomly sampled {num_tasks} task(s) from {len(valid_task_ids)} valid tasks")
            else:
                logger.info(f"Using all {len(valid_task_ids)} valid tasks (requested {num_tasks})")

        for arvo_id in valid_task_ids:
            logger.info(f"Processing task {arvo_id}")
            result = await self._process_task(arvo_id, request.config, updater)
            results.append(result)
            
            # Send result as a text message (artifacts not supported in update_status)
            result_text = f"Task {arvo_id} complete:\n{json.dumps(result.model_dump(), indent=2)}"
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(result_text, context_id=updater.context_id)
            )

        summary = f"Evaluation complete. Processed {len(results)} tasks."
        logger.info(summary)

    async def _process_task(self, arvo_id: str, config: Dict[str, Any], updater: TaskUpdater) -> TaskResult:
        """Process a single ARVO task."""
        from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
        from rcabench.utils import remote_fetch_error
        
        agent_paths = None
        try:
            logger.info(f"[{arvo_id}] Preparing task assets...")
            task_meta = prepare_task_assets(arvo_id=arvo_id)
            agent_paths = task_meta["agent_paths"]
            workspace_dir = agent_paths.workspace_dir
            shared_dir = agent_paths.shared_dir
            
            error_path = str(workspace_dir / f"{arvo_id}_error.txt")
            if not os.path.exists(error_path):
                remote_fetch_error(arvo_id, output_dir=workspace_dir)
            
            participant_endpoint = config.get("participant_endpoint", "http://127.0.0.1:9019")
            task_description = self._create_task_description(arvo_id, workspace_dir, error_path)
            
            if self._trace_conversation:
                self._conversation_trace = []
                # Don't add to trace yet, _run_tool_calling_loop will handle it
            
            max_turns = config.get("max_turns", 50)
            max_task_time = config.get("max_task_time", 1200)
            
            # Start the loop directly with the task description
            loop_result = await self._run_tool_calling_loop(
                arvo_id=arvo_id,
                workspace_dir=workspace_dir,
                shared_dir=shared_dir,
                participant_endpoint=participant_endpoint,
                updater=updater,
                max_turns=max_turns,
                max_task_time=max_task_time,
                task_description=task_description,
            )
            
            if loop_result.get("status") == "success" or os.path.exists(shared_dir / "loc.json"):
                eval_report = await self._evaluate_task_results(arvo_id, shared_dir)
                
                reasoning_trace = None
                reasoning_file = shared_dir / "reasoning.json"
                if reasoning_file.exists():
                    try:
                        with open(reasoning_file, "r") as f:
                            reasoning_trace = ReasoningTrace.model_validate(json.load(f))
                    except Exception as e:
                        logger.warning(f"Could not load reasoning trace: {e}")
                
                return TaskResult(
                    arvo_id=arvo_id,
                    file_acc=eval_report.file_acc,
                    func_topk_recall=eval_report.func_topk_recall,
                    line_topk_recall=eval_report.line_topk_recall,
                    line_iou_mean=eval_report.line_iou_mean,
                    line_proximity_mean=eval_report.line_proximity_mean,
                    n_gt=eval_report.n_gt,
                    n_pred=eval_report.n_pred,
                    success=True,
                    error=None,
                    reasoning_trace=reasoning_trace,
                )
            else:
                return TaskResult(
                    arvo_id=arvo_id,
                    file_acc=0.0,
                    func_topk_recall={},
                    line_topk_recall={},
                    line_iou_mean=0.0,
                    line_proximity_mean=0.0,
                    n_gt=0,
                    n_pred=0,
                    success=False,
                    error=loop_result.get("reason", "No submission found"),
                )
                
        except Exception as e:
            logger.error(f"Error processing task {arvo_id}: {e}", exc_info=True)
            return TaskResult(
                arvo_id=arvo_id,
                file_acc=0.0,
                func_topk_recall={},
                line_topk_recall={},
                line_iou_mean=0.0,
                line_proximity_mean=0.0,
                n_gt=0,
                n_pred=0,
                success=False,
                error=str(e),
            )
        finally:
            if agent_paths:
                try:
                    cleanup_task_assets(agent_paths)
                except:
                    pass

    async def _run_tool_calling_loop(
        self,
        arvo_id: str,
        workspace_dir: Path,
        shared_dir: Path,
        participant_endpoint: str,
        updater: TaskUpdater,
        max_turns: int,
        max_task_time: int,
        task_description: str,
    ) -> dict:
        """Sequential tool calling loop."""
        sandbox = DockerSandbox(arvo_id, workspace_dir)
        await sandbox.start()
        
        bash_executor = BashExecutor(sandbox)
        turn_manager = TurnManager(max_turns=max_turns)
        end_checker = EndConditionChecker(turn_manager, shared_dir, max_task_time=max_task_time)
        
        try:
            # Initial request: includes the task description
            green_msg = json.dumps({
                "type": "ready_for_tool_call",
                "task_description": task_description
            })
            
            if self._trace_conversation:
                trace_logger.info(f"[TRACE] GREEN → PURPLE (START): {green_msg[:200]}...")
            
            logger.info(f"[{arvo_id}] Starting tool calling loop...")
            response = await self._tool_provider.talk_to_agent(green_msg, participant_endpoint, new_conversation=True)
            
            while True:
                end_condition = end_checker.check()
                if end_condition["status"] != "continue":
                    logger.info(f"Loop ending: {end_condition['reason']}")
                    return end_condition
                
                if self._trace_conversation:
                    trace_logger.info(f"[TRACE] PURPLE → GREEN: {response[:200]}...")
                
                try:
                    tool_call = json.loads(response)
                    if tool_call.get("type") == "bash_command":
                        command = tool_call.get("command", "")
                        result = await bash_executor.execute(command)
                        turn_manager.record_turn(command, result["success"])
                        
                        stdout = result["stdout"]
                        stderr = result["stderr"]
                        MAX_CHARS = 15000
                        
                        if len(stdout) > MAX_CHARS:
                            stdout = stdout[:MAX_CHARS] + f"\n... [Output truncated, {len(result['stdout'])} total chars]"
                        if len(stderr) > MAX_CHARS:
                            stderr = stderr[:MAX_CHARS] + f"\n... [Output truncated, {len(result['stderr'])} total chars]"
                            
                        result_msg = json.dumps({
                            "type": "command_result",
                            "success": result["success"],
                            "stdout": stdout,
                            "stderr": stderr,
                            "exit_code": result["exit_code"],
                            "turn": turn_manager.turn_count,
                            "turns_remaining": turn_manager.get_turns_remaining(),
                        })
                        
                        if self._trace_conversation:
                            trace_logger.info(f"[TRACE] GREEN → PURPLE: {result_msg[:200]}...")
                            
                        response = await self._tool_provider.talk_to_agent(result_msg, participant_endpoint, new_conversation=False)
                        
                    elif tool_call.get("type") == "done":
                        logger.info("Received 'done' signal")
                        break
                    else:
                        logger.warning(f"Unknown type: {tool_call.get('type')}")
                        green_msg = '{"type": "ready_for_tool_call", "error": "Unknown message type"}'
                        response = await self._tool_provider.talk_to_agent(green_msg, participant_endpoint, new_conversation=False)
                
                except json.JSONDecodeError:
                    logger.warning("Non-JSON response received, asking for retry")
                    green_msg = '{"type": "ready_for_tool_call", "error": "Please provide your next action in JSON format."}'
                    response = await self._tool_provider.talk_to_agent(green_msg, participant_endpoint, new_conversation=False)
                
                await asyncio.sleep(0.1)
                
        finally:
            await sandbox.cleanup()
        
        return end_checker.check()

    def _create_task_description(self, arvo_id: str, workspace_dir: Path, error_path: str) -> str:
        with open(error_path, "r") as f:
            error_report = f.read()
            
        return f"""Task: arvo:{arvo_id}
Workspace: /workspace
- Error report: /workspace/{arvo_id}_error.txt
- Codebase: /workspace/src-vul/

Goal: Find the root cause of the crash described in the error report.

Available Tool: Bash (send a JSON object with type="bash_command" and command="your command").
Submission: When finished, create '/workspace/shared/loc.json' with your findings, then send type="done".

loc.json format:
{{
  "reasoning": "your detailed analysis",
  "locations": [
    {{"file": "path/to/file.c", "function": "func_name", "line": 123, "description": "why this line is relevant"}}
  ]
}}
"""

    async def _evaluate_task_results(self, arvo_id: str, shared_dir: Path) -> Any:
        """Evaluate the localization results against ground truth."""
        loc_file = shared_dir / "loc.json"
        if not loc_file.exists():
            if self._trace_only:
                trace_logger.info(f"\n[PREDICTIONS] No loc.json file found")
            return EvalReport(task_id=arvo_id, file_acc=0.0, func_topk_recall={}, line_topk_recall={}, line_iou_mean=0.0, line_proximity_mean=0.0, n_gt=0, n_pred=0, per_gt=[])
            
        with open(loc_file, "r") as f:
            try:
                data = json.load(f)
                
                # Show predictions in trace-only mode
                if self._trace_only:
                    trace_logger.info(f"\n[PREDICTIONS] Submitted localizations:")
                    trace_logger.info(json.dumps(data, indent=2))
                
                locations_data = data.get("locations", [])
                preds = []
                for loc in locations_data:
                    line = loc.get("line", 0)
                    preds.append(Localization(
                        task_id=arvo_id,
                        file=loc.get("file", ""),
                        old_span=LineSpan(start=line, end=line),
                        new_span=LineSpan(start=line, end=line),
                        function=loc.get("function", "")
                    ))
                gts = get_ground_truth(arvo_id)
                return evaluate_localization(preds, gts)
            except Exception as e:
                logger.error(f"Error evaluating localizations: {e}")
                return EvalReport(task_id=arvo_id, file_acc=0.0, func_topk_recall={}, line_topk_recall={}, line_iou_mean=0.0, line_proximity_mean=0.0, n_gt=0, n_pred=0, per_gt=[])

async def main():
    parser = argparse.ArgumentParser(description="RCA Judge (Green Agent)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--trace", action="store_true", help="Enable conversation trace logging")
    parser.add_argument("--trace-only", action="store_true", help="Show only trace, suppress other logs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    trace_enabled = args.trace or args.verbose or args.trace_only
    
    # If trace-only mode, suppress all other loggers
    if args.trace_only:
        # Suppress all loggers except trace
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("rca_judge").setLevel(logging.CRITICAL)
        logging.getLogger("tools.docker_sandbox").setLevel(logging.CRITICAL)
        logging.getLogger("tools.bash_executor").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("a2a").setLevel(logging.CRITICAL)
        # Enable only trace logger
        trace_logger.setLevel(logging.INFO)
    
    agent = RCAJudge(trace_conversation=trace_enabled, trace_only=args.trace_only)
    executor = GreenExecutor(agent)
    
    agent_card = AgentCard(
        name="RCAJudge",
        description="Orchestrates RCA evaluation.",
        url=f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[]
    )
    
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    
    # Disable access logging in trace-only mode
    log_level = "critical" if args.trace_only else "info"
    config = uvicorn.Config(app.build(), host=args.host, port=args.port, log_level=log_level, access_log=not args.trace_only)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
