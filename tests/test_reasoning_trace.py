#!/usr/bin/env python3
"""
Test reasoning trace validation and evaluation.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scenarios" / "arvo_rca"))

from rca_judge_common import ReasoningTrace, ReasoningStep, RejectedHypothesis


def test_reasoning_trace_validation():
    """Test that reasoning trace can be validated."""
    print("Testing reasoning trace validation...")
    
    # Create a valid reasoning trace
    reasoning_trace = ReasoningTrace(
        task_id="arvo:10055",
        reasoning_steps=[
            ReasoningStep(
                step=1,
                type="observation",
                content="Crash report shows buffer overflow at line 1234 in render.c",
                evidence=["10055_error.txt:45-50"]
            ),
            ReasoningStep(
                step=2,
                type="hypothesis",
                content="The buffer is allocated in allocate_buffer() but size calculation is wrong",
                evidence=["render.c:1200-1210"]
            ),
            ReasoningStep(
                step=3,
                type="analysis",
                content="Traced data flow: input -> parse_size() -> allocate_buffer() -> write_buffer()",
                evidence=["render.c:1150-1234"]
            ),
            ReasoningStep(
                step=4,
                type="conclusion",
                content="Root cause: parse_size() doesn't validate input, leading to integer overflow",
                evidence=["render.c:1180-1185"],
                prediction_id=0
            )
        ],
        rejected_hypotheses=[
            RejectedHypothesis(
                hypothesis="Crash is in write_buffer()",
                why_rejected="write_buffer() is correct, the issue is in size calculation"
            )
        ]
    )
    
    # Test serialization
    trace_dict = reasoning_trace.model_dump()
    print(f"✓ Reasoning trace created with {len(reasoning_trace.reasoning_steps)} steps")
    print(f"✓ Serialization successful")
    
    # Test deserialization
    reconstructed = ReasoningTrace.model_validate(trace_dict)
    assert reconstructed.task_id == reasoning_trace.task_id
    assert len(reconstructed.reasoning_steps) == len(reasoning_trace.reasoning_steps)
    print(f"✓ Deserialization successful")
    
    # Test JSON round-trip
    json_str = reasoning_trace.model_dump_json(indent=2)
    parsed = json.loads(json_str)
    reconstructed2 = ReasoningTrace.model_validate(parsed)
    assert reconstructed2.task_id == reasoning_trace.task_id
    print(f"✓ JSON round-trip successful")
    
    print()


def test_reasoning_trace_file_format():
    """Test that reasoning trace matches expected file format."""
    print("Testing reasoning trace file format...")
    
    # Create example reasoning trace as it would appear in reasoning.json
    example_trace = {
        "task_id": "arvo:10055",
        "reasoning_steps": [
            {
                "step": 1,
                "type": "observation",
                "content": "Crash report shows buffer overflow at line 1234 in render.c",
                "evidence": ["10055_error.txt:45-50"]
            },
            {
                "step": 2,
                "type": "hypothesis",
                "content": "The buffer is allocated in allocate_buffer() but size calculation is wrong",
                "evidence": ["render.c:1200-1210"]
            },
            {
                "step": 3,
                "type": "analysis",
                "content": "Traced data flow: input -> parse_size() -> allocate_buffer() -> write_buffer()",
                "evidence": ["render.c:1150-1234"]
            },
            {
                "step": 4,
                "type": "conclusion",
                "content": "Root cause: parse_size() doesn't validate input, leading to integer overflow",
                "evidence": ["render.c:1180-1185"],
                "prediction_id": 0
            }
        ],
        "rejected_hypotheses": [
            {
                "hypothesis": "Crash is in write_buffer()",
                "why_rejected": "write_buffer() is correct, the issue is in size calculation"
            }
        ]
    }
    
    # Validate it can be parsed
    trace = ReasoningTrace.model_validate(example_trace)
    assert trace.task_id == "arvo:10055"
    assert len(trace.reasoning_steps) == 4
    assert trace.reasoning_steps[0].type == "observation"
    assert trace.reasoning_steps[3].prediction_id == 0
    assert len(trace.rejected_hypotheses) == 1
    print(f"✓ Example file format validated")
    print(f"  - Task ID: {trace.task_id}")
    print(f"  - Steps: {len(trace.reasoning_steps)}")
    print(f"  - Rejected hypotheses: {len(trace.rejected_hypotheses)}")
    print()


def test_minimal_reasoning_trace():
    """Test minimal valid reasoning trace (no rejected hypotheses)."""
    print("Testing minimal reasoning trace...")
    
    minimal_trace = {
        "task_id": "arvo:10055",
        "reasoning_steps": [
            {
                "step": 1,
                "type": "observation",
                "content": "Crash at line 100",
                "evidence": []
            },
            {
                "step": 2,
                "type": "conclusion",
                "content": "Bug is at line 100",
                "evidence": ["file.c:100"],
                "prediction_id": 0
            }
        ]
    }
    
    trace = ReasoningTrace.model_validate(minimal_trace)
    assert trace.task_id == "arvo:10055"
    assert len(trace.reasoning_steps) == 2
    assert len(trace.rejected_hypotheses) == 0  # Should default to empty list
    print(f"✓ Minimal trace validated")
    print()


def test_invalid_reasoning_trace():
    """Test that invalid reasoning traces are rejected."""
    print("Testing invalid reasoning trace rejection...")
    
    # Missing required fields
    invalid_traces = [
        {},  # Missing task_id and reasoning_steps
        {"task_id": "arvo:10055"},  # Missing reasoning_steps
        {"reasoning_steps": []},  # Missing task_id
        {
            "task_id": "arvo:10055",
            "reasoning_steps": [
                {"step": 1}  # Missing type and content
            ]
        },
    ]
    
    for i, invalid in enumerate(invalid_traces):
        try:
            ReasoningTrace.model_validate(invalid)
            print(f"✗ Invalid trace {i+1} was incorrectly accepted")
            assert False, f"Should have rejected invalid trace {i+1}"
        except Exception as e:
            print(f"✓ Invalid trace {i+1} correctly rejected: {type(e).__name__}")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Reasoning Trace Tests")
    print("=" * 60)
    print()
    
    test_reasoning_trace_validation()
    test_reasoning_trace_file_format()
    test_minimal_reasoning_trace()
    test_invalid_reasoning_trace()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


