# RCAbench

A cloud-based evaluation server for AI agents on cybersecurity tasks. This project provides an orchestrator (gateway) that dispatches tasks to workers, manages progress, and creates evaluation containers for root-cause localization on codebases.

## Roadmap

- [ ] Integrate the AgentBeats SDK
- [ ] Create a dummy red agent and test on AgentBeats battle
- [ ] Locally integrate Arvo docker-based evaluation workflow
- [ ] Repurpose Arvo evaluation for root-cause localization task

## Features

- **Orchestrator Gateway**: FastAPI-based API with health checks, HTTP/WS queries.
- **Kubernetes Deployment**: Scalable deployment on k8s with ingress.
- **Client CLI**: Python client for testing API endpoints.
- **Security**: TLS-enabled, secrets management.

## Prerequisites

- Python 3.11+
- Docker
- Kubernetes cluster (for deployment)
- `kubectl` configured
- Conda (optional, for environment management)

## Quick Start

### Local Development

1. **Set up environment**:
   ```bash
   conda activate rcabench  # or create with conda env create -f environment.yml
   pip install -r requirements.txt
   ```

2. **Configure secrets**:
   - Copy `src/rcabench/.env.example` to `src/rcabench/.env`
   - Set `OPENAI_API_KEY` and other variables.

3. **Run locally**:
   ```bash
   make run-local
   ```
   This builds the Docker image and runs the container on `localhost:8080`.

4. **Test with client**:
   ```bash
   python client.py health
   python client.py query "Hello, world!"
   ```

### Kubernetes Deployment

1. **Build and push image**:
   ```bash
   make build
   make push
   ```

2. **Deploy to k8s**:
   ```bash
   make secret  # Creates k8s secrets
   make deploy  # Deploys ingress, service, deployment
   ```

3. **Monitor deployment**:
   ```bash
   make logs
   kubectl get pods -l app=rcabench
   ```

4. **Access the service**:
   - External: `https://rcabench.nrp-nautilus.io` (with self-signed cert)
   - Test: `python client.py health`

5. **Tear down**:
   ```bash
   make teardown
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /query` - HTTP query (JSON: `{"prompt": "your prompt"}`)
- `WS /ws` - WebSocket query
- `POST /evaluate` - Trigger evaluation job (JSON: `{"codebase": "url", "task": "root-cause"}`)

## Client Usage

The `client.py` script provides a CLI for testing:

```bash
# Health check
python client.py health

# HTTP query
python client.py query "Explain AI"

# WebSocket query
python client.py ws "What is cybersecurity?"

# Trigger evaluation
python client.py evaluate "https://github.com/user/repo" --task root-cause
```

Set `RCA_HOST` to change the target (default: deployment URL).

## Configuration

Environment variables (in `src/rcabench/.env`):

- `OPENAI_API_KEY` - API key for downstream LLM
- `DOWNSTREAM_URL` - Downstream API URL (default: `https://ellm.nrp-nautilus.io/v1`)
- `PORT` - Container port (default: `8080`)
- `LOG_LEVEL` - Logging level (default: `info`)

## Development

- **Linting**: Run tests with `python -m pytest`
- **Docker**: `make build` to build image
- **K8s**: Manifests in `k8s/` directory

## Security Notes

- TLS is enabled with self-signed cert for testing.
- Secrets are managed via k8s secrets.
- For production, use valid certificates and proper secret management.
