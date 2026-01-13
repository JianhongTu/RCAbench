# Important Notes

## ⚠️ DO NOT REFERENCE SCENARIOS FOLDER

**The `scenarios/` folder contains OLD CODE and is WRONG.**

- Do NOT look at `scenarios/arvo_rca/rca_judge.py` for reference
- Do NOT look at `scenarios/arvo_rca/rca_finder.py` for reference
- Do NOT use any code from the scenarios folder as a guide

The scenarios folder represents the old implementation and does not reflect the current architecture.

## Current Architecture

The correct implementation is in:
- `agents/mini-swe-agent/green_agent_server.py` - Green Agent (orchestrator)
- `agents/mini-swe-agent/purple_agent_server.py` - Purple Agent (command decider)

## Flow

1. **RCAJudge** (external evaluator) → sends task to **Green Agent**
2. **Green Agent** → prepares assets, initializes ARVO, sends formatted task to **Purple Agent**
3. **Purple Agent** → decides commands, sends to **Green Agent**
4. **Green Agent** → executes in ARVO container, returns results
5. Loop continues...

## Key Points

- Tasks START from Green Agent (not Purple Agent)
- Green Agent controls ARVO containers
- Green Agent orchestrates the task lifecycle
- Purple Agent only decides commands (no task preparation)

