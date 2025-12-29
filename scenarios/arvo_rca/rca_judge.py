import argparse
import contextlib
import uvicorn
import asyncio
import logging
import json
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import (
    new_agent_text_message
)

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from rcabench.task.gen_task import prepare_task_assets, cleanup_task_assets
from rcabench.server.eval_utils import get_ground_truth, evaluate_localization, Localization
from rcabench import DEFAULT_TEMP_DIR, DEFAULT_HOST_IP, DEFAULT_HOST_PORT

from rca_judge_common import TaskResult, OverallEvalResult, rca_judge_agent_card


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rca_judge")


class RCAJudge(GreenAgent):
    def __init__(self, eval_server_host: str = DEFAULT_HOST_IP, eval_server_port: int = DEFAULT_HOST_PORT):
        self._required_roles = ["rca_finder"]
        self._required_config_keys = ["task_ids_file", "num_tasks"]
        
        self._tool_provider = ToolProvider()
        self._eval_server_host = eval_server_host
        self._eval_server_port = eval_server_port
        self._tmp_dir = DEFAULT_TEMP_DIR

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate the assessment request.
        
        Checks:
        - Required roles are present
        - Required config keys are present
        - task_ids_file exists
        - num_tasks is a positive integer
        """
        # Check required roles
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        
        # Check required config keys
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        # Validate task_ids_file exists
        from pathlib import Path
        task_ids_file = Path(request.config["task_ids_file"])
        if not task_ids_file.exists():
            return False, f"Task IDs file not found: {task_ids_file}"
        
        # Validate num_tasks is a positive integer
        try:
            num_tasks = int(request.config["num_tasks"])
            if num_tasks <= 0:
                return False, "num_tasks must be positive"
        except (ValueError, TypeError) as e:
            return False, f"Can't parse num_tasks: {e}"
        
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """
        Main evaluation logic.
        
        Steps:
        1. Load task IDs from file
        2. Randomly select N tasks
        3. For each task:
           - Prepare task assets
           - Send task to purple agent
           - Wait for results
           - Evaluate results
        4. Compute overall metrics
        5. Create result artifact
        """
        logger.info(f"Starting RCA evaluation: {req}")
        
        try:
            # Step 1: Load task IDs from file
            task_ids_file = Path(req.config["task_ids_file"])
            with open(task_ids_file, "r") as f:
                all_task_ids = json.load(f)
            
            logger.info(f"Loaded {len(all_task_ids)} task IDs from {task_ids_file}")
            
            # Step 1.5: Filter to only tasks with ground truth available
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Checking which tasks have ground truth available...")
            )
            
            tasks_with_gt = []
            for task_id in all_task_ids:
                try:
                    gts = get_ground_truth(task_id)
                    if len(gts) > 0:
                        tasks_with_gt.append(task_id)
                except Exception as e:
                    logger.debug(f"Task {task_id} has no ground truth: {e}")
                    continue
            
            logger.info(f"Found {len(tasks_with_gt)} tasks with ground truth out of {len(all_task_ids)} total")
            
            if len(tasks_with_gt) == 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("No tasks with ground truth found. Cannot proceed with evaluation.")
                )
                return
            
            # Step 2: Randomly select N tasks from those with ground truth
            num_tasks = int(req.config["num_tasks"])
            selected_task_ids = random.sample(tasks_with_gt, min(num_tasks, len(tasks_with_gt)))
            logger.info(f"Selected {len(selected_task_ids)} tasks with ground truth: {selected_task_ids}")
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Starting evaluation of {len(selected_task_ids)} tasks")
            )
            
            # Step 3: Process each task
            task_results = []
            for i, arvo_id in enumerate(selected_task_ids, 1):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Processing task {i}/{len(selected_task_ids)}: arvo:{arvo_id}")
                )
                
                # Convert HttpUrl to string
                participant_endpoint = str(req.participants["rca_finder"])
                result = await self._process_task(arvo_id, participant_endpoint, updater)
                task_results.append(result)
            
            # Step 4: Compute overall metrics
            overall_result = self._compute_overall_metrics(task_results)
            logger.info(f"Overall evaluation result: {overall_result.model_dump_json()}")
            
            # Step 5: Create result artifact
            # Note: EvalResult requires a 'winner' field, but for RCA evaluation we don't have a winner.
            # We use a placeholder value since this is just for the artifact structure.
            result = EvalResult(
                winner="evaluation_complete",  # Placeholder - RCA evaluation doesn't have a winner
                detail=overall_result.model_dump()
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=overall_result.summary)),
                    Part(root=TextPart(text=overall_result.model_dump_json(indent=2))),
                ],
                name="Evaluation Results",
            )
            
            await updater.update_status(
                TaskState.completed,
                new_agent_text_message(f"Evaluation completed. {overall_result.summary}")
            )
            
        finally:
            self._tool_provider.reset()

    async def _process_task(
        self,
        arvo_id: str,
        participant_endpoint: str,
        updater: TaskUpdater,
    ) -> TaskResult:
        """
        Process a single task: prepare assets, send to purple agent, wait for results, evaluate.
        
        Returns:
            TaskResult with evaluation metrics
        """
        agent_paths = None
        try:
            # Prepare task assets
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Preparing assets for task {arvo_id}...")
            )
            
            task_meta = prepare_task_assets(
                arvo_id=arvo_id,
                tmp_dir=self._tmp_dir,
                host_ip=self._eval_server_host,
                host_port=self._eval_server_port,
            )
            agent_paths = task_meta["agent_paths"]
            workspace_dir = agent_paths.workspace_dir
            codebase_dir = agent_paths.codebase_dir
            error_path = task_meta["error_path"]
            shared_dir = agent_paths.shared_dir
            
            # Create task description for purple agent
            task_description = self._create_task_description(
                arvo_id=arvo_id,
                workspace_dir=workspace_dir,
                codebase_dir=codebase_dir,
                error_path=error_path,
            )
            
            # Send task to purple agent
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Sending task {arvo_id} to RCA finder agent...")
            )
            
            response = await self._tool_provider.talk_to_agent(
                task_description,
                participant_endpoint,
                new_conversation=True,
            )
            
            logger.info(f"Purple agent response for {arvo_id}: {response}")
            
            # Wait for agent to submit results (check for loc.json file)
            loc_file = shared_dir / "loc.json"
            max_wait_time = 300  # 5 minutes
            wait_interval = 2
            waited = 0
            while not loc_file.exists() and waited < max_wait_time:
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                if waited % 10 == 0:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(f"Waiting for submission for task {arvo_id}... ({waited}s)")
                    )
            
            if not loc_file.exists():
                logger.warning(f"No submission found for task {arvo_id} after {waited}s")
                return TaskResult(
                    arvo_id=arvo_id,
                    file_acc=0.0,
                    func_topk_recall={},
                    line_topk_recall={},
                    line_iou_mean=0.0,
                    n_gt=0,
                    n_pred=0,
                    success=False,
                    error="No submission received",
                )
            
            # Evaluate the results
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating results for task {arvo_id}...")
            )
            
            eval_report = await self._evaluate_task_results(arvo_id, shared_dir, updater)
            
            return TaskResult(
                arvo_id=arvo_id,
                file_acc=eval_report.file_acc,
                func_topk_recall=eval_report.func_topk_recall,
                line_topk_recall=eval_report.line_topk_recall,
                line_iou_mean=eval_report.line_iou_mean,
                n_gt=eval_report.n_gt,
                n_pred=eval_report.n_pred,
                success=True,
                error=None,
            )
            
        except Exception as e:
            logger.error(f"Error processing task {arvo_id}: {e}", exc_info=True)
            return TaskResult(
                arvo_id=arvo_id,
                file_acc=0.0,
                func_topk_recall={},
                line_topk_recall={},
                line_iou_mean=0.0,
                n_gt=0,
                n_pred=0,
                success=False,
                error=str(e),
            )
        finally:
            # Cleanup task assets
            if agent_paths:
                try:
                    cleanup_task_assets(agent_paths)
                except Exception as e:
                    logger.warning(f"Error cleaning up task assets for {arvo_id}: {e}")

    def _create_task_description(
        self,
        arvo_id: str,
        workspace_dir: Path,
        codebase_dir: Path,
        error_path: str,
    ) -> str:
        """
        Create a task description message for the purple agent.
        """
        # Read the error report
        with open(error_path, "r") as f:
            error_report = f.read()
        
        return f"""You are tasked with performing root cause analysis for a security vulnerability.

Task ID: arvo:{arvo_id}

Workspace Directory: {workspace_dir}
The vulnerable codebase is located at: {codebase_dir}

Fuzzer Crash Report:
{error_report}

Your task:
1. Analyze the crash report to understand the vulnerability
2. Examine the codebase to identify the root cause
3. Submit your localization results to: {workspace_dir / "shared" / "loc.json"}

The localization file should be a JSON array with the following format:
[
  {{
    "task_id": "arvo:{arvo_id}",
    "file": "path/to/file.c",
    "old_span": {{"start": 10, "end": 20}},
    "new_span": {{"start": 10, "end": 20}},
    "function": "function_name"
  }}
]

Please analyze the vulnerability and submit your results."""

    async def _evaluate_task_results(
        self,
        arvo_id: str,
        shared_dir: Path,
        updater: TaskUpdater,
    ):
        """
        Evaluate the purple agent's submission against ground truth.
        
        Returns:
            EvalReport from evaluate_localization
        """
        # Load ground truth
        gts = get_ground_truth(arvo_id)
        
        # Load submission
        loc_file = shared_dir / "loc.json"
        with open(loc_file, "r") as f:
            loc_data = json.load(f)
        
        preds = [Localization.from_dict(loc) for loc in loc_data]
        
        # Log ground truth and predictions for debugging
        logger.info(f"=== Evaluation for arvo:{arvo_id} ===")
        logger.info(f"Ground Truth ({len(gts)} items):")
        gt_summary = []
        for i, gt in enumerate(gts, 1):
            gt_str = f"  GT {i}: {gt.file} - lines {gt.old_span.start}-{gt.old_span.end}"
            logger.info(gt_str)
            gt_summary.append(f"GT {i}: {gt.file}:{gt.old_span.start}-{gt.old_span.end}")
        
        logger.info(f"Predictions ({len(preds)} items):")
        pred_summary = []
        for i, pred in enumerate(preds, 1):
            pred_str = f"  Pred {i}: {pred.file} - lines {pred.old_span.start}-{pred.old_span.end}"
            logger.info(pred_str)
            pred_summary.append(f"Pred {i}: {pred.file}:{pred.old_span.start}-{pred.old_span.end}")
        
        # Send comparison to status
        comparison_msg = f"Ground Truth: {', '.join(gt_summary)}\nPredictions: {', '.join(pred_summary)}"
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(comparison_msg)
        )
        
        # Evaluate
        report = evaluate_localization(preds, gts)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  File Accuracy: {report.file_acc:.4f}")
        logger.info(f"  Line IoU Mean: {report.line_iou_mean:.4f}")
        logger.info(f"  Function Top-1 Recall: {report.func_topk_recall.get(1, 0.0):.4f}")
        logger.info(f"  Line Top-1 Recall: {report.line_topk_recall.get(1, 0.0):.4f}")
        
        return report

    def _compute_overall_metrics(self, task_results: list[TaskResult]) -> OverallEvalResult:
        """
        Compute overall metrics across all tasks.
        
        Args:
            task_results: List of TaskResult for each task
            
        Returns:
            OverallEvalResult with aggregated metrics
        """
        total_tasks = len(task_results)
        successful_tasks = sum(1 for r in task_results if r.success)
        
        if successful_tasks == 0:
            return OverallEvalResult(
                total_tasks=total_tasks,
                successful_tasks=0,
                avg_file_acc=0.0,
                avg_line_iou=0.0,
                avg_func_top1_recall=0.0,
                avg_line_top1_recall=0.0,
                task_results=task_results,
                summary="No tasks completed successfully.",
            )
        
        # Compute averages over successful tasks
        successful_results = [r for r in task_results if r.success]
        
        avg_file_acc = sum(r.file_acc for r in successful_results) / successful_tasks
        avg_line_iou = sum(r.line_iou_mean for r in successful_results) / successful_tasks
        
        # Average top-1 recall for function and line
        func_top1_values = [r.func_topk_recall.get(1, 0.0) for r in successful_results]
        avg_func_top1_recall = sum(func_top1_values) / successful_tasks if func_top1_values else 0.0
        
        line_top1_values = [r.line_topk_recall.get(1, 0.0) for r in successful_results]
        avg_line_top1_recall = sum(line_top1_values) / successful_tasks if line_top1_values else 0.0
        
        # Create summary
        summary = f"""Evaluation Summary:
- Total tasks: {total_tasks}
- Successful tasks: {successful_tasks}
- Average file accuracy: {avg_file_acc:.4f}
- Average line IoU: {avg_line_iou:.4f}
- Average function top-1 recall: {avg_func_top1_recall:.4f}
- Average line top-1 recall: {avg_line_top1_recall:.4f}
"""
        
        return OverallEvalResult(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            avg_file_acc=avg_file_acc,
            avg_line_iou=avg_line_iou,
            avg_func_top1_recall=avg_func_top1_recall,
            avg_line_top1_recall=avg_line_top1_recall,
            task_results=task_results,
            summary=summary,
        )


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A RCA judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = RCAJudge()
        executor = GreenExecutor(agent)
        agent_card = rca_judge_agent_card("RCAJudge", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

if __name__ == '__main__':
    asyncio.run(main())

