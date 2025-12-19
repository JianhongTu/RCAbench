# Dockerfile for RCAbench Purple Agent
# This follows the AgentBeats tutorial requirements

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project root files needed for installation
# Note: When building, context should be the project root
COPY pyproject.toml setup.py README.md ./
COPY src /app/src

# Copy agent requirements (from agents/purple_agent/ relative to project root)
COPY agents/purple_agent/requirements.txt .

# Install RCAbench package and agent dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt

# Copy agent code (from agents/purple_agent/ relative to project root)
COPY agents/purple_agent/purple_agent.py .

# Make the script executable
RUN chmod +x purple_agent.py

# Set entrypoint (required by AgentBeats)
# AgentBeats will pass --host, --port, and --card-url arguments
ENTRYPOINT ["python", "purple_agent.py"]

