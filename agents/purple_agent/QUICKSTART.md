# Quick Start Guide: Purple Agent

This guide will help you quickly get your purple agent running and registered on AgentBeats.

## Step 1: Build the Docker Image

From the project root:

```bash
cd agents/purple_agent
docker build --platform linux/amd64 -t ghcr.io/YOURUSERNAME/rcabench-purple-agent:v1.0 -f Dockerfile ../..
```

**Important:** The build context is the project root (two levels up), not the agent directory.

## Step 2: Test Locally

Run the container locally:

```bash
docker run -p 8001:8001 \
  ghcr.io/YOURUSERNAME/rcabench-purple-agent:v1.0 \
  --host 0.0.0.0 \
  --port 8001 \
  --card-url http://localhost:8001
```

Test the endpoints:

```bash
# Health check
curl http://localhost:8001/

# Agent card
curl http://localhost:8001/card
```

## Step 3: Push to GitHub Container Registry

1. **Login:**
   ```bash
   export GITHUB_TOKEN=""
   echo $GITHUB_TOKEN | docker login ghcr.io -u YOURUSERNAME --password-stdin
   ```

2. **Push:**
   ```bash
   docker push ghcr.io/YOURUSERNAME/rcabench-purple-agent:v1.0
   ```

## Step 4: Register on AgentBeats

1. Go to https://agentbeats.dev/register-agent
2. Select "Purple" agent type
3. Fill in:
   - **Name**: RCAbench Purple Agent
   - **Docker Image**: `ghcr.io/YOURUSERNAME/rcabench-purple-agent:v1.0`
   - **Repository**: Your GitHub repo URL
   - **Agent Card**: Copy from `purple_agent_card.toml` (update the URL first!)

4. **Copy your Agent ID** - you'll need this for assessments

## Step 5: Run an Assessment

1. Create a `scenario.toml` file (see `example_scenario.toml`)
2. Update with your agent ID and image
3. Run using AgentBeats CLI or GitHub Actions workflow

## Troubleshooting

**Build fails:**
- Ensure you're building from project root with correct context
- Check that all dependencies are in `requirements.txt`

**Agent not responding:**
- Verify port is accessible
- Check Docker logs: `docker logs <container-id>`

**Import errors:**
- Ensure RCAbench package is installed in Docker image
- Check that `src/rcabench` is copied correctly

## Next Steps

- Add LLM integration for real vulnerability analysis
- Enhance localization accuracy
- Add patch generation capabilities
- Set up automated testing

