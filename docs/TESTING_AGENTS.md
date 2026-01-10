# Testing the Simplified Agents

## Overview

The system now has two simplified agents:
- **Green Agent (RCA Judge)**: Orchestrates tool execution in Docker sandbox
- **Purple Agent (RCA Finder)**: Lightweight LLM wrapper that generates bash commands

## Quick Start

### 1. Start Both Agents

**Option A: Use the test script (Windows)**
```bash
python scripts/test_agents.py
```

**Option B: Start manually in separate terminals**

Terminal 1 (Green Agent):
```bash
python scenarios/arvo_rca/rca_judge.py --host 127.0.0.1 --port 9009
```

Terminal 2 (Purple Agent):
```bash
python scenarios/arvo_rca/rca_finder.py --host 127.0.0.1 --port 9019 --model gpt-4o
```

### 2. Run an Evaluation

The green agent expects an evaluation request with:
- `task_ids_file`: Path to file with task IDs (one per line)
- `num_tasks`: Number of tasks to evaluate
- `participants`: Must include `rca_finder` role pointing to purple agent URL

Example request (via AgentBeats client or direct API call):
```json
{
  "participants": {
    "rca_finder": "http://127.0.0.1:9019/"
  },
  "config": {
    "task_ids_file": "data/arvo_hf_task_ids.txt",
    "num_tasks": "1"
  }
}
```

## How It Works

1. **Green Agent** receives evaluation request
2. **Green Agent** prepares task assets (workspace, error report, codebase)
3. **Green Agent** sends task description to **Purple Agent**
4. **Purple Agent** uses LLM to generate bash commands
5. **Purple Agent** sends commands back to **Green Agent** as JSON:
   ```json
   {"type": "bash_command", "command": "cat src-vul/main.c"}
   ```
6. **Green Agent** executes command in Docker sandbox
7. **Green Agent** sends result back to **Purple Agent**:
   ```json
   {
     "type": "command_result",
     "success": true,
     "stdout": "...",
     "exit_code": 0,
     "turn": 1,
     "turns_remaining": 49
   }
   ```
8. Loop continues until:
   - Purple agent sends `{"type": "done"}`
   - Max turns reached
   - Timeout
   - Submission files created (`loc.json` and `reasoning.json`)

## Requirements

- Docker installed and running
- OpenAI API key in `OPENAI_API_KEY` environment variable
- Python dependencies installed (`pip install -e .`)

## Troubleshooting

- **Docker errors**: Make sure Docker is running and you have permission to create containers
- **API key errors**: Set `OPENAI_API_KEY` environment variable
- **Port conflicts**: Change ports with `--port` argument
- **Communication errors**: Check that both agents are running and URLs are correct
