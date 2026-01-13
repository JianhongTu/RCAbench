# Dockerfile for green agent container (mini-swe-agent)
# This container runs the green agent that controls ARVO containers

FROM python:3.12-slim

# Install Docker CLI (to execute commands in arvo containers)
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    openai>=1.0.0 \
    docker>=7.1.0 \
    uvicorn \
    a2a-sdk

# Set working directory
WORKDIR /app

# Copy agent code
COPY docker_environment.py green_agent_server.py ./

# Note: rcabench and agentbeats are mounted as volumes in docker-compose.yml
# This allows live code updates without rebuilding the image

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOCKER_HOST=unix:///var/run/docker.sock
ENV PYTHONPATH=/app:/workspace

# Default command
CMD ["python", "/app/green_agent_server.py", "--host", "0.0.0.0", "--port", "9009"]

