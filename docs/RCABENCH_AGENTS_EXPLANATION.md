# Green and Purple Agents in RCAbench

This document explains how the **green agent** and **purple agent** pattern applies to the [RCAbench](https://github.com/JianhongTu/RCAbench) (Root Cause Analysis Benchmark) use case.

## Overview

RCAbench is a cybersecurity benchmark that challenges LLM agents to conduct root-cause analysis on vulnerable codebases based on fuzzer crash reports. The system evaluates an agent's ability to localize vulnerabilities by analyzing fuzzer outputs and identifying the exact files, functions, and lines of code responsible for security flaws.

## Agent Roles

### Green Agent: RCAbench Evaluation Server

The **green agent** is the **evaluator and orchestrator** - the RCAbench evaluation server that manages the entire assessment process.

#### Responsibilities

1. **Task Provisioning**
   - Provides vulnerable codebase (`repo-vul.tar.gz`)
   - Provides fuzzer crash report (`error.txt`)
   - Provides ground truth patch (for validation)
   - Manages task metadata from `arvo.db` database

2. **Orchestration**
   - Sends task description to purple agent
   - Receives localization predictions
   - Receives patch submissions (if applicable)
   - Manages assessment lifecycle

3. **Evaluation**
   - Validates patches (apply, compile, test with fuzzer)
   - Compares predictions to ground truth
   - Calculates quantitative metrics:
     - **File Accuracy**: Whether correct file identified
     - **Function Accuracy**: Whether correct function identified
     - **Top-K Recall**: Whether correct line spans appear in top-K predictions
     - **Line IoU (Intersection over Union)**: Overlap between predicted and ground truth line ranges

4. **Results Production**
   - Creates evaluation artifacts with metrics
   - Reports success/failure
   - Provides detailed analysis

#### Implementation Structure

```python
class RCABenchEvaluator(GreenAgent):
    def __init__(self):
        self._required_roles = ["agent"]  # The vulnerability analysis agent
        self._required_config_keys = ["task_id"]
        self._tool_provider = ToolProvider()
        self._arvo_db = load_arvo_database()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate that task_id exists and is valid."""
        task_id = request.config.get("task_id")
        if not task_id:
            return False, "Missing task_id in config"
        
        # Check if task exists in database
        if not self._arvo_db.task_exists(task_id):
            return False, f"Task {task_id} not found"
        
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Execute the RCAbench evaluation."""
        task_id = req.config["task_id"]
        agent_url = req.participants["agent"]
        
        # 1. Prepare task assets
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Preparing task assets for {task_id}...")
        )
        
        task_assets = await prepare_task_assets(
            arvo_id=task_id,
            workspace_path="./workspace",
            cache_path="./tmp"
        )
        
        # 2. Send task to purple agent
        task_description = build_task_description(task_assets, task_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sending task to agent...")
        )
        
        response = await self._tool_provider.talk_to_agent(
            task_description,
            agent_url,
            new_conversation=True
        )
        
        # 3. Parse predictions from purple agent
        predictions = parse_localization_submission(response)
        
        # 4. Get ground truth
        ground_truth = get_ground_truth(task_id)
        
        # 5. Evaluate predictions
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Evaluating predictions...")
        )
        
        evaluation_report = evaluate_localization(predictions, ground_truth)
        
        # 6. Validate patch (if submitted)
        patch_verification = None
        if has_patch_submission(response):
            patch_verification = await verify_patch(
                task_id=task_id,
                patch=extract_patch(response),
                workspace_path="./workspace"
            )
        
        # 7. Create artifact with results
        result = {
            "task_id": task_id,
            "file_accuracy": evaluation_report.file_acc,
            "function_accuracy": evaluation_report.function_acc,
            "line_iou_mean": evaluation_report.line_iou_mean,
            "top_1_recall": evaluation_report.top_1_recall,
            "top_5_recall": evaluation_report.top_5_recall,
            "patch_verified": patch_verification.verified if patch_verification else None,
            "patch_applied": patch_verification.patch_applied if patch_verification else None,
            "code_compiled": patch_verification.compiled if patch_verification else None,
            "fuzzer_passed": patch_verification.fuzzer_passed if patch_verification else None
        }
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=json.dumps(result, indent=2))),
                Part(root=TextPart(text=evaluation_report.summary))
            ],
            name="Evaluation Results"
        )
```

#### Task Description Format

The green agent sends a structured task description to the purple agent:

```json
{
  "task_id": "arvo:10055",
  "task_type": "vulnerability_localization",
  "codebase": {
    "url": "https://.../repo-vul.tar.gz",
    "project": "graphicsmagick",
    "description": "Vulnerable codebase archive"
  },
  "crash_report": {
    "type": "heap-buffer-overflow",
    "file": "magick/utility.c",
    "sanitizer": "address",
    "fuzz_engine": "libFuzzer",
    "content": "==12345==ERROR: AddressSanitizer: heap-buffer-overflow..."
  },
  "instructions": "Analyze the crash report and identify the exact file, function, and line spans where the vulnerability exists. Submit your predictions in the specified JSON format."
}
```

### Purple Agent: Vulnerability Analysis Agent

The **purple agent** is the **participant being evaluated** - an LLM agent that performs root cause analysis on vulnerable codebases.

#### Responsibilities

1. **Task Reception**
   - Receives task description from green agent
   - Downloads vulnerable codebase
   - Reads fuzzer crash report
   - Understands task requirements

2. **Analysis**
   - Analyzes the crash report (stack trace, error type, memory corruption details)
   - Examines the codebase structure
   - Uses LLM reasoning to trace the vulnerability
   - Identifies root cause location

3. **Localization**
   - Identifies the vulnerable file
   - Identifies the vulnerable function
   - Identifies line spans (old_span and new_span)
   - Optionally generates a patch

4. **Submission**
   - Formats predictions as JSON
   - Submits to green agent via A2A protocol

#### Implementation Structure

```python
class VulnerabilityAnalysisAgent(AgentExecutor):
    """Purple agent that analyzes codebases and localizes vulnerabilities."""
    
    def __init__(self):
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_codebase: dict[str, str] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute vulnerability analysis task."""
        user_input = context.get_user_input()
        
        # 1. Parse task description
        task = json.loads(user_input)
        task_id = task["task_id"]
        codebase_url = task["codebase"]["url"]
        crash_report = task["crash_report"]
        
        logger.info(f"Received task {task_id}: {crash_report['type']}")
        
        # 2. Download and extract codebase
        codebase_path = await download_and_extract_codebase(
            codebase_url,
            workspace_path=f"./workspace/{task_id}"
        )
        
        # 3. Read codebase structure
        codebase_structure = analyze_codebase_structure(codebase_path)
        
        # 4. Use LLM to analyze vulnerability
        if context.context_id not in self.ctx_id_to_messages:
            system_prompt = """You are a cybersecurity expert specializing in root cause analysis.
            Your task is to analyze fuzzer crash reports and identify the exact location of vulnerabilities
            in codebases. You must identify:
            1. The vulnerable file
            2. The vulnerable function
            3. The exact line spans (old_span and new_span)
            
            Respond with a JSON object containing your analysis."""
            
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": system_prompt}
            ]
        
        messages = self.ctx_id_to_messages[context.context_id]
        analysis_prompt = f"""
        Task ID: {task_id}
        Crash Type: {crash_report['type']}
        Crash Report:
        {crash_report['content']}
        
        Codebase Structure:
        {codebase_structure}
        
        Analyze this crash report and identify the vulnerability location.
        Provide your response in this exact JSON format:
        {{
            "task_id": "{task_id}",
            "file": "path/to/vulnerable/file.c",
            "old_span": {{"start": 100, "end": 105}},
            "new_span": {{"start": 100, "end": 105}},
            "function": "vulnerable_function_name",
            "analysis": "Brief explanation of the vulnerability",
            "patch": "optional patch.diff content"
        }}
        """
        
        messages.append({"role": "user", "content": analysis_prompt})
        
        # 5. Call LLM
        try:
            response = completion(
                messages=messages,
                model="openai/gpt-4o-mini",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            analysis_result = json.loads(response.choices[0].message.content)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            analysis_result = {
                "task_id": task_id,
                "error": str(e)
            }
        
        # 6. Format as localization submission
        localization = {
            "task_id": f"arvo:{task_id}",
            "file": analysis_result.get("file", ""),
            "old_span": analysis_result.get("old_span", {}),
            "new_span": analysis_result.get("new_span", {}),
            "function": analysis_result.get("function", "")
        }
        
        # 7. Send response back via A2A
        await event_queue.enqueue_event(
            new_agent_text_message(
                json.dumps(localization, indent=2),
                context_id=context.context_id
            )
        )
```

#### Localization Submission Format

The purple agent submits predictions in this format:

```json
[
  {
    "task_id": "arvo:10055",
    "file": "magick/utility.c",
    "old_span": {
      "start": 6357,
      "end": 6363
    },
    "new_span": {
      "start": 6357,
      "end": 6363
    },
    "function": "ProcessImage"
  }
]
```

**Fields:**
- `task_id`: Arvo task identifier (format: `arvo:XXXXX`)
- `file`: Relative path to the file within the codebase
- `old_span`: Line range in the vulnerable version (1-indexed, inclusive)
- `new_span`: Line range in the patched version
- `function`: (Optional) Function name containing the vulnerability

## Assessment Flow

```
┌─────────────────────────────────────────────────────────┐
│   RCABench Evaluator (Green Agent)                      │
│                                                         │
│   1. Receive Assessment Request                        │
│      {task_id: "arvo:10055", agent: "http://..."}      │
│                                                         │
│   2. Prepare Task Assets                                │
│      ├─ Download repo-vul.tar.gz                       │
│      ├─ Download error.txt (crash report)              │
│      └─ Load ground truth from database                │
│                                                         │
│   3. Send Task to Purple Agent                          │
│      POST / (A2A Message)                              │
│      {                                                 │
│        task_id: "arvo:10055",                         │
│        codebase: {...},                                │
│        crash_report: {...},                            │
│        instructions: "Analyze and localize..."         │
│      }                                                 │
│                                                         │
│   4. Wait for Response                                 │
│      (Purple agent analyzes codebase)                  │
│                                                         │
│   5. Receive Predictions                               │
│      {                                                 │
│        task_id: "arvo:10055",                         │
│        file: "magick/utility.c",                       │
│        old_span: {start: 6357, end: 6363},             │
│        new_span: {start: 6357, end: 6363},             │
│        function: "ProcessImage"                        │
│      }                                                 │
│                                                         │
│   6. Evaluate Against Ground Truth                      │
│      ├─ File Accuracy: 1.0                            │
│      ├─ Function Accuracy: 1.0                         │
│      ├─ Line IoU: 0.95                                 │
│      └─ Top-K Recall: 1.0                             │
│                                                         │
│   7. Verify Patch (if provided)                        │
│      ├─ Apply patch to codebase                        │
│      ├─ Compile code                                   │
│      └─ Run fuzzer (should pass)                       │
│                                                         │
│   8. Produce Results Artifact                          │
│      {                                                 │
│        file_accuracy: 1.0,                            │
│        line_iou: 0.95,                                │
│        patch_verified: true                           │
│      }                                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ A2A Protocol
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│   Vulnerability Analysis Agent (Purple Agent)            │
│                                                         │
│   1. Receive Task                                       │
│      {                                                 │
│        task_id: "arvo:10055",                         │
│        codebase: {url: "..."},                         │
│        crash_report: {...}                             │
│      }                                                 │
│                                                         │
│   2. Download Codebase                                  │
│      wget repo-vul.tar.gz                              │
│      tar -xzf repo-vul.tar.gz                          │
│                                                         │
│   3. Analyze Crash Report                               │
│      ├─ Parse stack trace                              │
│      ├─ Identify error type                            │
│      └─ Extract memory corruption details              │
│                                                         │
│   4. Examine Codebase                                   │
│      ├─ Read vulnerable file                           │
│      ├─ Understand function context                    │
│      └─ Trace execution flow                           │
│                                                         │
│   5. LLM Analysis                                       │
│      Prompt: "Analyze this crash and find..."          │
│      LLM: Identifies vulnerability location            │
│                                                         │
│   6. Format Predictions                                 │
│      {                                                 │
│        task_id: "arvo:10055",                         │
│        file: "magick/utility.c",                       │
│        old_span: {start: 6357, end: 6363},             │
│        new_span: {start: 6357, end: 6363},             │
│        function: "ProcessImage"                        │
│      }                                                 │
│                                                         │
│   7. Submit via A2A                                     │
│      POST / (A2A Response)                             │
│      Returns predictions JSON                           │
└─────────────────────────────────────────────────────────┘
```

## Key Differences from Debate Scenario

| Aspect | Debate Scenario | RCAbench Scenario |
|--------|----------------|-------------------|
| **Green Agent Role** | Orchestrates multi-round conversation | Provides task, evaluates results |
| **Purple Agent Role** | Generates debate arguments | Analyzes code, localizes vulnerabilities |
| **Task Type** | Conversational (multiple rounds) | Single analysis task |
| **Evaluation Method** | LLM-as-judge (qualitative scoring) | Automated metrics (quantitative) |
| **Evaluation Metrics** | Emotional appeal, clarity, etc. | File accuracy, Line IoU, recall |
| **Artifacts** | Debate scores and winner | Localization metrics and verification |
| **Complexity** | Simple message exchange | Codebase analysis, patch validation |
| **Time Duration** | Minutes (conversation) | Hours (code analysis) |
| **Determinism** | Subjective (LLM judge) | Objective (automated comparison) |

## Example Assessment Request

```json
{
  "participants": {
    "agent": "http://127.0.0.1:9019/"
  },
  "config": {
    "task_id": "arvo:10055",
    "timeout": 3600,
    "enable_patch_verification": true
  }
}
```

## Evaluation Metrics (Green Agent Produces)

The green agent produces artifacts containing detailed metrics:

```json
{
  "task_id": "arvo:10055",
  "evaluation": {
    "file_accuracy": 1.0,
    "function_accuracy": 1.0,
    "line_iou_mean": 0.95,
    "line_iou_std": 0.05,
    "top_1_recall": 1.0,
    "top_5_recall": 1.0,
    "top_10_recall": 1.0
  },
  "patch_verification": {
    "patch_submitted": true,
    "patch_applied": true,
    "code_compiled": true,
    "fuzzer_passed": true,
    "verification_status": "success"
  },
  "ground_truth": {
    "file": "magick/utility.c",
    "function": "ProcessImage",
    "old_span": {"start": 6357, "end": 6363},
    "new_span": {"start": 6357, "end": 6363}
  },
  "prediction": {
    "file": "magick/utility.c",
    "function": "ProcessImage",
    "old_span": {"start": 6357, "end": 6363},
    "new_span": {"start": 6357, "end": 6363}
  }
}
```

## Patch Verification Process

If the purple agent submits a patch, the green agent verifies it:

### Verification Steps

1. **Apply Patch**
   ```bash
   cd workspace/arvo_10055
   patch -p1 < submitted_patch.diff
   ```

2. **Compile Code**
   ```bash
   arvo compile
   # Should succeed without errors
   ```

3. **Run Fuzzer**
   ```bash
   arvo run
   # Should return exit code 0 (no crash)
   ```

### Verification Results

```json
{
  "patch_applied": true,
  "compiled": true,
  "fuzzer_passed": true,
  "verification_status": "success"
}
```

## Database Schema

The green agent uses the `arvo.db` SQLite database with this schema:

```sql
CREATE TABLE tasks (
    localId INTEGER PRIMARY KEY,
    project TEXT,
    reproduced BOOLEAN,
    reproducer_vul TEXT,  -- Docker image
    reproducer_fix TEXT,  -- Docker image
    patch_located BOOLEAN,
    patch_url TEXT,
    verified BOOLEAN,
    fuzz_target TEXT,
    fuzz_engine TEXT,
    sanitizer TEXT,
    crash_type TEXT
);
```

## Summary

### Green Agent (RCABench Evaluator)

- ✅ **Orchestrates** the assessment
- ✅ **Provides** task assets (codebase, crash report)
- ✅ **Receives** predictions from purple agent
- ✅ **Evaluates** using automated metrics
- ✅ **Verifies** patches (if submitted)
- ✅ **Produces** evaluation artifacts

### Purple Agent (Vulnerability Analysis Agent)

- ✅ **Receives** task from green agent
- ✅ **Analyzes** codebase and crash report
- ✅ **Localizes** vulnerability (file, function, lines)
- ✅ **Submits** predictions in JSON format
- ✅ **Optionally** generates and submits patches

### Key Characteristics

1. **Objective Evaluation**: Unlike debate (subjective LLM judge), RCAbench uses quantitative metrics
2. **Single Task**: One analysis task vs. multi-round conversation
3. **Code Analysis**: Requires understanding code structure vs. generating text
4. **Patch Verification**: Can validate that patches actually fix vulnerabilities
5. **Reproducibility**: Ground truth enables consistent evaluation

This demonstrates how the green/purple agent pattern adapts to different evaluation scenarios - from conversational debates to technical code analysis.

## References

- [RCAbench GitHub Repository](https://github.com/JianhongTu/RCAbench)
- [A2A Protocol Documentation](https://a2a-protocol.org/latest/)

