# RCAbench Purple Agent

A **purple agent** implementation for AgentBeats that performs root cause analysis on vulnerable codebases. This agent follows the [AgentBeats tutorial](https://docs.agentbeats.dev/tutorial/) and implements the A2A (Agent-to-Agent) protocol.

## Overview

This purple agent:
- Implements the A2A protocol for AgentBeats compatibility
- Performs root cause analysis on vulnerable codebases
- Localizes vulnerabilities based on fuzzer crash reports
- Returns results as A2A artifacts

## Prerequisites

- Python 3.11+
- Docker (for containerization)
- Access to RCAbench task assets (Arvo dataset)

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   cd agents/purple_agent
   pip install -r requirements.txt
   ```

2. **Install RCAbench package:**
   ```bash
   # From project root
   pip install -e .
   ```

3. **Run the agent:**
   ```bash
   python purple_agent.py --host 0.0.0.0 --port 8001 --card-url http://localhost:8001
   ```

   The agent will be available at `http://localhost:8001`

### Testing the Agent

Test the health check endpoint:
```bash
curl http://localhost:8001/
```

Test the agent card:
```bash
curl http://localhost:8001/card
```

## Docker Setup

### Building the Image

```bash
cd agents/purple_agent
docker build --platform linux/amd64 -t ghcr.io/YOUR_USERNAME/rcabench-purple-agent:v1.0 .
```

**Important:** Always build for `linux/amd64` architecture as that is used by GitHub Actions.

### Running the Container

```bash
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=your-key-here \
  ghcr.io/YOUR_USERNAME/rcabench-purple-agent:v1.0 \
  --card-url http://YOUR_PUBLIC_IP:8001
```

### Publishing to GitHub Container Registry

1. **Login to GitHub Container Registry:**
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
   ```

2. **Push the image:**
   ```bash
   docker push ghcr.io/YOUR_USERNAME/rcabench-purple-agent:v1.0
   ```

## AgentBeats Registration

### 1. Update Agent Card

Edit `purple_agent_card.toml` and set your public URL:
```toml
url = "http://YOUR_PUBLIC_IP:YOUR_AGENT_PORT"
```

### 2. Register on AgentBeats

1. Go to [AgentBeats Register Agent](https://agentbeats.dev/register-agent)
2. Select "Purple" agent type
3. Fill in:
   - **Agent Name**: RCAbench Purple Agent
   - **Docker Image**: `ghcr.io/YOUR_USERNAME/rcabench-purple-agent:v1.0`
   - **Repository URL**: Your GitHub repo URL
   - **Agent Card**: Copy contents from `purple_agent_card.toml`

### 3. Get Agent ID

After registration, copy your agent ID from the AgentBeats page. You'll need this for assessments.

## Assessment Flow

The agent follows the A2A assessment flow:

1. **Green agent sends assessment request** to `/assessments` endpoint
2. **Purple agent receives**:
   ```json
   {
     "participants": {
       "agent": "http://purple-agent:8001"
     },
     "config": {
       "arvo_id": "10055"
     }
   }
   ```

3. **Purple agent performs analysis**:
   - Downloads task assets (codebase, error report)
   - Analyzes the vulnerability
   - Localizes the root cause

4. **Purple agent returns artifacts**:
   ```json
   {
     "task_id": "arvo:10055",
     "status": "completed",
     "artifacts": [
       {
         "name": "localization_results",
         "content_type": "application/json",
         "content": {
           "localizations": [...]
         }
       }
     ]
   }
   ```

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "agent_type": "purple",
  "capabilities": ["root_cause_analysis", "vulnerability_localization"]
}
```

### `GET /card`
Returns agent card information (A2A protocol).

**Response:**
```json
{
  "name": "RCAbench Purple Agent",
  "description": "...",
  "url": "http://...",
  "capabilities": [...]
}
```

### `POST /assessments`
Handles assessment requests from green agents.

**Request:**
```json
{
  "participants": {
    "agent": "http://..."
  },
  "config": {
    "arvo_id": "10055"
  }
}
```

**Response:**
```json
{
  "task_id": "arvo:10055",
  "status": "completed",
  "artifacts": [...],
  "updates": [...]
}
```

## Integration with RCAbench

This purple agent uses RCAbench's task preparation system:
- Downloads Arvo task assets (codebase, error report, patch)
- Uses RCAbench's localization format
- Compatible with RCAbench evaluation metrics

## Development

### Adding LLM Integration

The current implementation is a placeholder. To add real LLM analysis:

1. Add LLM client (OpenAI, Anthropic, etc.) to `requirements.txt`
2. Modify `analyze_vulnerability()` in `purple_agent.py`
3. Use the LLM to:
   - Analyze the crash report
   - Examine the codebase
   - Identify vulnerability locations
   - Generate patches (optional)

### Example LLM Integration

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_vulnerability(arvo_id: str, workspace_dir: Path) -> Dict[str, Any]:
    # Read error report
    error_content = read_error_report(workspace_dir)
    
    # Use LLM to analyze
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a security researcher..."},
            {"role": "user", "content": f"Analyze this crash report:\n{error_content}"}
        ]
    )
    
    # Parse LLM response and extract localizations
    # ...
```

## GitHub Actions Workflow

See `.github/workflows/publish.yml` for automated build and publish workflow.

## Troubleshooting

**Agent not responding:**
- Check that the port is accessible
- Verify Docker container is running
- Check logs: `docker logs <container-id>`

**Assessment failures:**
- Ensure `arvo_id` is provided in config
- Verify task assets can be downloaded
- Check network connectivity

**Docker build issues:**
- Ensure you're building for `linux/amd64`
- Check that all dependencies are in `requirements.txt`

## Next Steps

1. **Enhance Analysis**: Add real LLM-based vulnerability analysis
2. **Add Patch Generation**: Generate patches for identified vulnerabilities
3. **Improve Localization**: Better line-level and function-level accuracy
4. **Add Caching**: Cache analysis results for faster responses
5. **Add Metrics**: Track performance metrics

## References

- [AgentBeats Tutorial](https://docs.agentbeats.dev/tutorial/)
- [A2A Protocol](https://a2a-protocol.org/latest/)
- [RCAbench Documentation](../README.md)

