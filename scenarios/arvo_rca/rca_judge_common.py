"""
Common utilities and models for the RCAbench green agent (RCA Judge).
"""

from pydantic import BaseModel
from typing import Dict, List
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


class LLMJudgment(BaseModel):
    """LLM judgment of the analysis quality."""
    score: float  # 0.0 to 1.0
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]
    quality_assessment: str  # "excellent", "good", "fair", "poor"


class TaskResult(BaseModel):
    """Result for a single task evaluation."""
    arvo_id: str
    file_acc: float
    func_topk_recall: Dict[int, float]
    line_topk_recall: Dict[int, float]
    line_iou_mean: float
    n_gt: int
    n_pred: int
    success: bool
    error: str | None = None
    llm_judgment: LLMJudgment | None = None  # LLM-as-a-judge assessment


class OverallEvalResult(BaseModel):
    """Overall evaluation result across all tasks."""
    total_tasks: int
    successful_tasks: int
    avg_file_acc: float
    avg_line_iou: float
    avg_func_top1_recall: float
    avg_line_top1_recall: float
    avg_llm_score: float  # Average LLM judgment score
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
