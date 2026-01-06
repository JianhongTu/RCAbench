"""
Common utilities and models for the RCAbench green agent (RCA Judge).
"""

from pydantic import BaseModel
from typing import Dict, List, Optional
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""
    step: int  # Step number (1-indexed)
    type: str  # "observation", "hypothesis", "analysis", "conclusion", "verification"
    content: str  # Description of this step
    evidence: List[str] = []  # Citations (e.g., ["error.txt:45-50", "render.c:1200-1210"])
    prediction_id: Optional[int] = None  # Links to loc.json[prediction_id] if this step leads to a prediction


class RejectedHypothesis(BaseModel):
    """A hypothesis that was considered but rejected."""
    hypothesis: str  # What was considered
    why_rejected: str  # Why it was rejected


class ReasoningTrace(BaseModel):
    """Step-by-step reasoning trace for root cause analysis."""
    task_id: str
    reasoning_steps: List[ReasoningStep]
    rejected_hypotheses: List[RejectedHypothesis] = []


class ReasoningJudgment(BaseModel):
    """LLM judgment of the reasoning quality (separate from answer correctness)."""
    reasoning_score: float  # 0.0 to 1.0 - quality of reasoning process
    reasoning_strengths: List[str]  # What was done well in the reasoning
    reasoning_weaknesses: List[str]  # What could be improved in the reasoning
    logical_flow_score: float  # 0.0 to 1.0 - how logical and coherent the steps are
    evidence_quality_score: float  # 0.0 to 1.0 - quality of evidence cited
    root_cause_focus_score: float  # 0.0 to 1.0 - how well it focuses on root cause vs symptoms
    reasoning_assessment: str  # "excellent", "good", "fair", "poor"


class LLMJudgment(BaseModel):
    """LLM judgment of the analysis quality."""
    score: float  # 0.0 to 1.0 - overall score (combines correctness + reasoning)
    correctness_score: float  # 0.0 to 1.0 - how correct the answer is (based on metrics)
    reasoning: str  # Overall reasoning explanation
    strengths: List[str]
    weaknesses: List[str]
    quality_assessment: str  # "excellent", "good", "fair", "poor"
    reasoning_judgment: Optional[ReasoningJudgment] = None  # Separate reasoning quality assessment


class TaskResult(BaseModel):
    """Result for a single task evaluation."""
    arvo_id: str
    file_acc: float
    func_topk_recall: Dict[int, float]
    line_topk_recall: Dict[int, float]
    line_iou_mean: float
    line_proximity_mean: float  # Proximity score for same-file matches (even if IoU=0)
    n_gt: int
    n_pred: int
    success: bool
    error: str | None = None
    reasoning_trace: Optional[ReasoningTrace] = None  # Reasoning trace from purple agent
    llm_judgment: LLMJudgment | None = None  # LLM-as-a-judge assessment


class OverallEvalResult(BaseModel):
    """Overall evaluation result across all tasks."""
    total_tasks: int
    successful_tasks: int
    avg_file_acc: float
    avg_line_iou: float
    avg_line_proximity: float  # Average proximity score
    avg_func_top1_recall: float
    avg_line_top1_recall: float
    avg_llm_score: float  # Average LLM judgment score
    avg_reasoning_score: float  # Average reasoning quality score (separate from correctness)
    task_results: List[TaskResult]
    summary: str
    llm_overall_assessment: str | None = None  # Overall LLM assessment


def rca_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Create an agent card for the RCA Judge green agent."""
    skill = AgentSkill(
        id='evaluate_root_cause_analysis',
        name='Evaluate Root Cause Analysis',
        description='Evaluates LLM agents on root cause analysis of security vulnerabilities from fuzzer crash reports.',
        tags=['security', 'vulnerability', 'root-cause-analysis'],
        examples=["""
{
  "participants": {
    "rca_finder": "http://127.0.0.1:9019"
  },
  "config": {
    "task_ids_file": "data/good_arvo_task_ids.json",
    "num_tasks": 1
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Evaluates LLM agents on root cause analysis tasks from the Arvo dataset. Provides vulnerable codebases and fuzzer crash reports, then evaluates localization accuracy.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
