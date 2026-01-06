import argparse
import contextlib
import uvicorn
import asyncio
import logging
import json
import random
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
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

from rca_judge_common import TaskResult, OverallEvalResult, LLMJudgment, ReasoningTrace, ReasoningJudgment, rca_judge_agent_card


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rca_judge")


class RCAJudge(GreenAgent):
    def __init__(
        self,
        eval_server_host: str = DEFAULT_HOST_IP,
        eval_server_port: int = DEFAULT_HOST_PORT,
        llm_model: str = "gpt-4o",
        llm_api_key: str | None = None,
        use_llm_judge: bool = True,
    ):
        self._required_roles = ["rca_finder"]
        self._required_config_keys = ["task_ids_file", "num_tasks"]
        
        self._tool_provider = ToolProvider()
        self._eval_server_host = eval_server_host
        self._eval_server_port = eval_server_port
        self._tmp_dir = DEFAULT_TEMP_DIR
        
        # LLM-as-a-judge setup (defaults to using OPENAI_API_KEY from env)
        self._use_llm_judge = use_llm_judge
        if self._use_llm_judge:
            api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("LLM judge enabled but no API key found. Set OPENAI_API_KEY env var or pass --llm-api-key")
                self._use_llm_judge = False
                self._llm_client = None
                self._llm_model = None
            else:
                self._llm_client = OpenAI(api_key=api_key)
                self._llm_model = llm_model
        else:
            self._llm_client = None
            self._llm_model = None

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
            
            # Build artifact content with metrics and LLM judgment
            artifact_parts = [Part(root=TextPart(text=overall_result.summary))]
            
            # Add LLM assessment if available
            if overall_result.llm_overall_assessment:
                artifact_parts.append(Part(root=TextPart(text=f"\nLLM Assessment: {overall_result.llm_overall_assessment}")))
            
            # Add detailed results
            artifact_parts.append(Part(root=TextPart(text=overall_result.model_dump_json(indent=2))))
            
            await updater.add_artifact(
                parts=artifact_parts,
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
            
            # Wait for agent to submit results (check for loc.json and reasoning.json files)
            loc_file = shared_dir / "loc.json"
            reasoning_file = shared_dir / "reasoning.json"
            max_wait_time = 300  # 5 minutes
            wait_interval = 2
            waited = 0
            while (not loc_file.exists() or not reasoning_file.exists()) and waited < max_wait_time:
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                if waited % 10 == 0:
                    missing = []
                    if not loc_file.exists():
                        missing.append("loc.json")
                    if not reasoning_file.exists():
                        missing.append("reasoning.json")
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(f"Waiting for submission for task {arvo_id}... ({waited}s) Missing: {', '.join(missing)}")
                    )
            
            if not loc_file.exists():
                logger.warning(f"No loc.json submission found for task {arvo_id} after {waited}s")
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
                    error="No loc.json submission received",
                    reasoning_trace=None,
                )
            
            if not reasoning_file.exists():
                logger.warning(f"No reasoning.json submission found for task {arvo_id} after {waited}s")
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
                    error="No reasoning.json submission received (reasoning trace is required)",
                    reasoning_trace=None,
                )
            
            # Load and validate reasoning trace
            reasoning_trace = None
            try:
                with open(reasoning_file, "r") as f:
                    reasoning_data = json.load(f)
                reasoning_trace = ReasoningTrace.model_validate(reasoning_data)
                
                # Validate that task_id matches
                if reasoning_trace.task_id != f"arvo:{arvo_id}" and reasoning_trace.task_id != arvo_id:
                    logger.warning(f"Reasoning trace task_id mismatch: expected arvo:{arvo_id}, got {reasoning_trace.task_id}")
                
                # Validate that reasoning steps exist
                if not reasoning_trace.reasoning_steps:
                    logger.warning(f"Reasoning trace has no steps for task {arvo_id}")
                
                logger.info(f"Loaded reasoning trace with {len(reasoning_trace.reasoning_steps)} steps for task {arvo_id}")
            except Exception as e:
                logger.error(f"Error loading/validating reasoning trace for task {arvo_id}: {e}", exc_info=True)
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
                    error=f"Invalid reasoning trace: {str(e)}",
                    reasoning_trace=None,
                )
            
            # Evaluate the results
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating results for task {arvo_id}...")
            )
            
            eval_report = await self._evaluate_task_results(arvo_id, shared_dir, updater)
            
            # Get LLM judgment (if enabled)
            llm_judgment = None
            if self._use_llm_judge:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Getting LLM judgment for task {arvo_id}...")
                )
                # Load predictions and ground truth for LLM context
                loc_file = shared_dir / "loc.json"
                with open(loc_file, "r") as f:
                    loc_data = json.load(f)
                preds = [Localization.from_dict(loc) for loc in loc_data]
                gts = get_ground_truth(arvo_id)
                
                llm_judgment = await self._get_llm_judgment(
                    arvo_id, eval_report, gts, preds, reasoning_trace
                )
            
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
                llm_judgment=llm_judgment,
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
                reasoning_trace=None,
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
        logger.info(f"  Line Proximity Mean: {report.line_proximity_mean:.4f} (for same-file matches)")
        logger.info(f"  Function Top-1 Recall: {report.func_topk_recall.get(1, 0.0):.4f}")
        logger.info(f"  Line Top-1 Recall: {report.line_topk_recall.get(1, 0.0):.4f}")
        
        return report

    async def _evaluate_reasoning_quality(
        self,
        arvo_id: str,
        reasoning_trace: ReasoningTrace,
        eval_report,
        gts: list | None,
        preds: list | None,
    ) -> ReasoningJudgment:
        """
        Use LLM to evaluate the quality of the reasoning process separately from correctness.
        
        This evaluates HOW the agent reasoned, not whether the answer was correct.
        Like a teacher grading the work shown, not just the final answer.
        
        Returns:
            ReasoningJudgment with reasoning quality scores
        """
        # Format reasoning steps
        reasoning_steps_text = ""
        for step in reasoning_trace.reasoning_steps:
            evidence_str = ", ".join(step.evidence) if step.evidence else "none"
            pred_link = f" (links to prediction {step.prediction_id})" if step.prediction_id is not None else ""
            reasoning_steps_text += f"\nStep {step.step} ({step.type}): {step.content}\n"
            reasoning_steps_text += f"  Evidence: {evidence_str}{pred_link}\n"
        
        rejected_text = ""
        if reasoning_trace.rejected_hypotheses:
            rejected_text = "\nRejected Hypotheses:\n"
            for rej in reasoning_trace.rejected_hypotheses:
                rejected_text += f"  - {rej.hypothesis}\n"
                rejected_text += f"    Why rejected: {rej.why_rejected}\n"
        
        # Build correctness context (for reference, but reasoning is evaluated separately)
        correctness_context = f"""
Reference: Correctness Metrics (for context only - evaluate reasoning separately):
- File Accuracy: {eval_report.file_acc:.4f}
- Line IoU Mean: {eval_report.line_iou_mean:.4f}
- Function Top-1 Recall: {eval_report.func_topk_recall.get(1, 0.0):.4f}
- Line Top-1 Recall: {eval_report.line_topk_recall.get(1, 0.0):.4f}
"""
        
        prompt = f"""You are an expert security researcher evaluating the REASONING PROCESS of a root cause analysis.

Task ID: arvo:{arvo_id}

{correctness_context}

Reasoning Trace:
{reasoning_steps_text}
{rejected_text}

**IMPORTANT**: Evaluate the QUALITY OF THE REASONING PROCESS, not just whether the answer was correct.
Give partial credit for good reasoning even if the final answer was wrong.

Evaluate these dimensions separately:

1. **Logical Flow** (0.0-1.0): Are the steps logical, sequential, and coherent? Do they build on each other?
2. **Evidence Quality** (0.0-1.0): Are evidence citations appropriate? Do they support the reasoning?
3. **Root Cause Focus** (0.0-1.0): Does the reasoning trace from symptoms to root cause? Or does it stop at the crash location?
4. **Overall Reasoning Quality** (0.0-1.0): How well-structured and thorough is the reasoning process?

Provide your judgment in JSON format with:
- "reasoning_score": Overall reasoning quality score (0.0 to 1.0)
- "logical_flow_score": Logical flow score (0.0 to 1.0)
- "evidence_quality_score": Evidence quality score (0.0 to 1.0)
- "root_cause_focus_score": Root cause focus score (0.0 to 1.0)
- "reasoning_strengths": List of what was done well in the reasoning
- "reasoning_weaknesses": List of what could be improved
- "reasoning_assessment": One of "excellent", "good", "fair", or "poor"

Return ONLY valid JSON, no markdown formatting."""

        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert security researcher specializing in evaluating reasoning processes in root cause analysis. You assess the quality of the thinking process separately from answer correctness, like a teacher grading the work shown on an exam."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return ReasoningJudgment(
            reasoning_score=float(result.get("reasoning_score", 0.0)),
            logical_flow_score=float(result.get("logical_flow_score", 0.0)),
            evidence_quality_score=float(result.get("evidence_quality_score", 0.0)),
            root_cause_focus_score=float(result.get("root_cause_focus_score", 0.0)),
            reasoning_strengths=result.get("reasoning_strengths", []),
            reasoning_weaknesses=result.get("reasoning_weaknesses", []),
            reasoning_assessment=result.get("reasoning_assessment", "fair"),
        )

    async def _get_llm_judgment(
        self,
        arvo_id: str,
        eval_report,
        gts: list | None,
        preds: list | None,
        reasoning_trace: ReasoningTrace | None = None,
    ) -> LLMJudgment:
        """
        Use LLM to judge the quality of the root cause analysis.
        
        Now evaluates both correctness AND reasoning quality separately.
        
        Returns:
            LLMJudgment with combined score, correctness score, and reasoning judgment
        """
        # Compute correctness score from metrics (strict, no inflation)
        correctness_score = (
            eval_report.file_acc * 0.4 +  # File accuracy is important
            eval_report.line_iou_mean * 0.3 +  # Line precision matters
            eval_report.line_topk_recall.get(1, 0.0) * 0.3  # Top-1 recall
        )
        
        # Evaluate reasoning quality separately if reasoning trace is available
        reasoning_judgment = None
        if reasoning_trace and self._use_llm_judge:
            reasoning_judgment = await self._evaluate_reasoning_quality(
                arvo_id, reasoning_trace, eval_report, gts, preds
            )
        
        # Build context for overall assessment
        metrics_summary = f"""
Quantitative Metrics (Correctness):
- File Accuracy: {eval_report.file_acc:.4f}
- Line IoU Mean: {eval_report.line_iou_mean:.4f}
- Function Top-1 Recall: {eval_report.func_topk_recall.get(1, 0.0):.4f}
- Line Top-1 Recall: {eval_report.line_topk_recall.get(1, 0.0):.4f}
- Ground Truth Locations: {eval_report.n_gt}
- Predicted Locations: {eval_report.n_pred}
- Computed Correctness Score: {correctness_score:.4f}
"""
        
        if reasoning_judgment:
            metrics_summary += f"""
Reasoning Quality Scores:
- Overall Reasoning Score: {reasoning_judgment.reasoning_score:.4f}
- Logical Flow: {reasoning_judgment.logical_flow_score:.4f}
- Evidence Quality: {reasoning_judgment.evidence_quality_score:.4f}
- Root Cause Focus: {reasoning_judgment.root_cause_focus_score:.4f}
"""
        
        gt_summary = ""
        if gts:
            gt_summary = "\nGround Truth Locations:\n"
            for i, gt in enumerate(gts[:5], 1):  # Show up to 5
                gt_summary += f"  {i}. {gt.file}:{gt.old_span.start}-{gt.old_span.end} (function: {gt.function or 'N/A'})\n"
        
        pred_summary = ""
        if preds:
            pred_summary = "\nPredicted Locations:\n"
            for i, pred in enumerate(preds[:5], 1):  # Show up to 5
                pred_summary += f"  {i}. {pred.file}:{pred.old_span.start}-{pred.old_span.end} (function: {pred.function or 'N/A'})\n"
        
        prompt = f"""You are an expert security researcher evaluating root cause analysis (RCA) performance on a vulnerability localization task.

Task ID: arvo:{arvo_id}

{metrics_summary}
{gt_summary}
{pred_summary}

Evaluate the overall quality considering both correctness and reasoning quality.

Provide your judgment in JSON format with:
- "score": Combined overall score (0.0 to 1.0) - weighted combination of correctness and reasoning
- "reasoning": Detailed explanation of your assessment
- "strengths": List of what was done well
- "weaknesses": List of areas for improvement
- "quality_assessment": One of "excellent", "good", "fair", or "poor"

Return ONLY valid JSON, no markdown formatting."""

        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert security researcher specializing in evaluating root cause analysis of software vulnerabilities. You assess both quantitative metrics (correctness) and qualitative aspects (reasoning quality) of vulnerability localization."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return LLMJudgment(
            score=float(result.get("score", correctness_score)),  # Combined score
            correctness_score=correctness_score,  # Strict correctness (no inflation)
            reasoning=result.get("reasoning", ""),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            quality_assessment=result.get("quality_assessment", "fair"),
            reasoning_judgment=reasoning_judgment,  # Separate reasoning evaluation
        )

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
                avg_line_proximity=0.0,
                avg_func_top1_recall=0.0,
                avg_line_top1_recall=0.0,
                avg_llm_score=0.0,
                avg_reasoning_score=0.0,
                task_results=task_results,
                summary="No tasks completed successfully.",
                llm_overall_assessment=None,
            )
        
        # Compute averages over successful tasks
        successful_results = [r for r in task_results if r.success]
        
        avg_file_acc = sum(r.file_acc for r in successful_results) / successful_tasks
        avg_line_iou = sum(r.line_iou_mean for r in successful_results) / successful_tasks
        avg_line_proximity = sum(r.line_proximity_mean for r in successful_results) / successful_tasks
        
        # Average top-1 recall for function and line
        func_top1_values = [r.func_topk_recall.get(1, 0.0) for r in successful_results]
        avg_func_top1_recall = sum(func_top1_values) / successful_tasks if func_top1_values else 0.0
        
        line_top1_values = [r.line_topk_recall.get(1, 0.0) for r in successful_results]
        avg_line_top1_recall = sum(line_top1_values) / successful_tasks if line_top1_values else 0.0
        
        # Average LLM judgment score
        llm_scores = [r.llm_judgment.score for r in successful_results if r.llm_judgment]
        avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
        
        # Average reasoning quality score (separate from correctness)
        reasoning_scores = [
            r.llm_judgment.reasoning_judgment.reasoning_score
            for r in successful_results
            if r.llm_judgment and r.llm_judgment.reasoning_judgment
        ]
        avg_reasoning_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
        
        # Get overall LLM assessment if available
        llm_overall_assessment = None
        if llm_scores:
            # Get reasoning from one of the judgments or create summary
            judgments = [r.llm_judgment for r in successful_results if r.llm_judgment]
            if judgments:
                # Use the average quality assessment or create one
                quality_counts = {}
                for j in judgments:
                    quality_counts[j.quality_assessment] = quality_counts.get(j.quality_assessment, 0) + 1
                most_common = max(quality_counts.items(), key=lambda x: x[1])[0] if quality_counts else "fair"
                llm_overall_assessment = f"Overall quality: {most_common} (avg LLM score: {avg_llm_score:.2f})"
                if avg_reasoning_score > 0:
                    llm_overall_assessment += f", avg reasoning score: {avg_reasoning_score:.2f}"
        
        # Create summary
        summary = f"""Evaluation Summary:
- Total tasks: {total_tasks}
- Successful tasks: {successful_tasks}
- Average file accuracy: {avg_file_acc:.4f}
- Average line IoU: {avg_line_iou:.4f}
- Average line proximity: {avg_line_proximity:.4f} (for same-file matches)
- Average function top-1 recall: {avg_func_top1_recall:.4f}
- Average line top-1 recall: {avg_line_top1_recall:.4f}
"""
        if avg_llm_score > 0:
            summary += f"- Average LLM judgment score: {avg_llm_score:.4f}\n"
        if avg_reasoning_score > 0:
            summary += f"- Average reasoning quality score: {avg_reasoning_score:.4f}\n"
        
        return OverallEvalResult(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            avg_file_acc=avg_file_acc,
            avg_line_iou=avg_line_iou,
            avg_line_proximity=avg_line_proximity,
            avg_func_top1_recall=avg_func_top1_recall,
            avg_line_top1_recall=avg_line_top1_recall,
            avg_llm_score=avg_llm_score,
            avg_reasoning_score=avg_reasoning_score,
            task_results=task_results,
            summary=summary,
            llm_overall_assessment=llm_overall_assessment,
        )


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A RCA judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM model for judgment (default: gpt-4o)")
    parser.add_argument("--llm-api-key", type=str, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-a-judge (use only metrics)")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = RCAJudge(
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            use_llm_judge=not args.no_llm_judge,
        )
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

