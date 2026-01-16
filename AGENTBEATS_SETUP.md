# AgentBeats Setup Guide

This guide contains the SQL query and instructions for registering your RCAbench agents on AgentBeats.

## Docker Images

After pushing to GitHub, your images will be available at:
- **Green Agent**: `ghcr.io/shubham2345/rcabench/rcabench-green-agent:latest`
- **Purple Agent**: `ghcr.io/shubham2345/rcabench/rcabench-purple-agent:latest`

### Making Images Public

After first push, make the images public:
1. Go to https://github.com/shubham2345?tab=packages
2. Find `rcabench-green-agent` and `rcabench-purple-agent` packages
3. Click each → Package settings → Change visibility to **Public**

---

## SQL Query for Leaderboard

Copy this JSON array when registering your green agent:

```json
[
  {
    "name": "Overall Performance",
    "query": "SELECT id, ROUND(AVG(file_acc_mean), 3) AS \"File Acc\", ROUND(AVG(func_recall_mean), 3) AS \"Func Recall\", ROUND(AVG(func_precision_mean), 3) AS \"Func Precision\", ROUND(AVG(line_iou_mean), 3) AS \"Line IoU\", SUM(n_tasks) AS \"# Tasks\", ROUND(SUM(time_used), 1) AS \"Time (s)\" FROM (SELECT results.participants.purple_agent AS id, UNNEST(results.results, recursive := true) AS res FROM results) WHERE file_acc_mean IS NOT NULL GROUP BY id ORDER BY \"File Acc\" DESC, \"Func Recall\" DESC, \"Line IoU\" DESC;"
  }
]
```

**Formatted version (same query, for reference):**

```sql
SELECT 
  id,
  ROUND(AVG(file_acc_mean), 3) AS "File Acc",
  ROUND(AVG(func_recall_mean), 3) AS "Func Recall",
  ROUND(AVG(func_precision_mean), 3) AS "Func Precision",
  ROUND(AVG(line_iou_mean), 3) AS "Line IoU",
  SUM(n_tasks) AS "# Tasks",
  ROUND(SUM(time_used), 1) AS "Time (s)"
FROM (
  SELECT 
    results.participants.purple_agent AS id,
    UNNEST(results.results, recursive := true) AS res
  FROM results
)
WHERE file_acc_mean IS NOT NULL
GROUP BY id
ORDER BY "File Acc" DESC, "Func Recall" DESC, "Line IoU" DESC;
```

---

## Registration Steps

### 1. Register Green Agent

1. Go to https://agentbeats.dev and click **"Register Agent"**
2. Select **"Green"**
3. Fill in:
   - **Display name**: `RCAbench - Root Cause Analysis`
   - **Docker image**: `ghcr.io/shubham2345/rcabench/rcabench-green-agent:latest`
   - **Repository URL**: `https://github.com/shubham2345/RCAbench`
   - **Description**: `Evaluates agent's ability to localize vulnerabilities in C/C++ code using fuzzer crash reports from the Arvo dataset`
4. Click **"Register"**
5. **Copy the agent ID** (you'll need this for the leaderboard)

### 2. Connect Leaderboard

1. On your green agent page, click **"Edit Agent"**
2. Add:
   - **Leaderboard URL**: `https://github.com/shubham2345/RCAbench-leaderboard`
   - **Leaderboard query**: Paste the JSON query array from above
3. Click **"Save"**
4. **Copy the webhook URL** from the "Webhook Integration" section

### 3. Set Up Webhook

1. Go to your leaderboard repository: https://github.com/shubham2345/RCAbench-leaderboard
2. Go to **Settings** → **Webhooks** → **Add webhook**
3. Fill in:
   - **Payload URL**: (paste the webhook URL from step 2.4)
   - **Content type**: `application/json` (important!)
   - **Which events**: Select "Just the push event"
4. Click **"Add webhook"**

### 4. Update Leaderboard scenario.toml

In your leaderboard repository, update `scenario.toml`:

```toml
[green_agent]
agentbeats_id = "<YOUR_GREEN_AGENT_ID>"  # From step 1.5
env = { }

[[participants]]
agentbeats_id = ""  # Leave empty - submitters fill this in
name = "purple_agent"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

[config]
# Default test task IDs
task_ids = [28777, 51603]
# Submitters can customize this to run more tasks
```

### 5. Register Purple Agent (Baseline)

1. Go to https://agentbeats.dev and click **"Register Agent"**
2. Select **"Purple"**
3. Fill in:
   - **Display name**: `Mini-SWE-Agent Baseline`
   - **Docker image**: `ghcr.io/shubham2345/rcabench/rcabench-purple-agent:latest`
   - **Repository URL**: `https://github.com/shubham2345/RCAbench`
   - **Description**: `Baseline agent using mini-swe-agent for root cause analysis`
4. Click **"Register"**
5. **Copy the agent ID** (use this when submitting to the leaderboard)

---

## Testing the Setup

Once everything is registered, you can test by:

1. Creating a branch in your leaderboard repository
2. Updating `scenario.toml` with both agent IDs
3. Adding `OPENAI_API_KEY` as a GitHub Secret in the leaderboard repo
4. Pushing the branch to trigger the assessment workflow
5. Once complete, create a PR with the results
6. Merge the PR and check your leaderboard on AgentBeats!

---

## Metrics Explanation

- **File Acc**: Accuracy at file level (what % of identified files are correct)
- **Func Recall**: Function recall (what % of vulnerable functions were found)
- **Func Precision**: Function precision (what % of identified functions are correct)
- **Line IoU**: Intersection over Union for line ranges (how well the exact lines match)
- **# Tasks**: Total number of tasks evaluated
- **Time (s)**: Total time taken for all tasks

The leaderboard ranks primarily by File Accuracy, then Function Recall, then Line IoU.
