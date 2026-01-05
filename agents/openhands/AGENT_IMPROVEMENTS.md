# Agent Improvement Plan

## Current Status
- ✅ File accuracy: Good (agent finds correct files)
- ❌ Line accuracy: Poor (agent finds wrong line numbers)
- ✅ System working: OpenHands runs successfully, dependencies installed

## Question 1: Evaluation Metrics ✅ DONE
**Answer: Added proximity metric**

We've added a `line_proximity_mean` metric that gives partial credit for being in the right file but wrong lines. This uses exponential decay based on distance:
- Same location: 1.0
- 1x average range distance: ~0.37
- 2x average range distance: ~0.14

This rewards agents that find the right file but are close to the right lines, even if they don't overlap.

## Question 2: Agent Improvements

### Option A: Multi-Phase Approach (Recommended)
Break the task into explicit phases:

**Phase 1: File Discovery**
- Read crash report
- Identify files mentioned in stack trace
- List candidate files

**Phase 2: Function Identification**
- For each candidate file, identify functions in call stack
- Narrow down to most likely functions

**Phase 3: Line Narrowing**
- Read the identified functions
- Trace data flow
- Identify exact bug location

**Phase 4: Verification**
- Verify the identified location makes sense
- Check if fixing it would prevent the crash
- Submit results

### Option B: Enhanced Prompt with Examples
Add concrete examples showing:
- How to read a stack trace
- How to trace from crash to root cause
- Example: "If crash is at line 50 in function_a, but buffer was allocated in function_b at line 20, the bug is at line 20"

### Option C: Structured Output Format
Instead of free-form analysis, require structured intermediate outputs:
1. Parse crash report → extract error type, crash location
2. Identify relevant files → list files from stack trace
3. Trace backwards → list functions in call chain
4. Identify bug → exact location

### Option D: Increase Iterations + Better Guidance
- Increase max_iter from 30 to 50-100
- Add checkpoints: "After 10 iterations, you should have identified the file"
- Add self-verification steps

## Recommended Approach: Hybrid

1. **Enhanced Prompt** (already done) ✅
2. **Add Structured Checkpoints** - Require agent to output intermediate results
3. **Increase Max Iterations** - Allow more exploration
4. **Add Self-Verification** - Agent should verify its findings before submitting

## Implementation Priority

1. **High Priority**: Add structured checkpoints to prompt
2. **Medium Priority**: Increase max_iter default to 50
3. **Low Priority**: Add example cases to prompt

