# Docker Connection Error Fix

## Error

```
docker.errors.DockerException: Error while fetching server API version: 
('Connection aborted.', ConnectionRefusedError(61, 'Connection refused'))
```

## Cause

The Docker daemon is not running or not accessible.

## Solution

### 1. Start Docker Desktop (macOS)

```bash
# Open Docker Desktop application
open -a Docker
```

Wait for Docker to fully start (check the Docker icon in the menu bar).

### 2. Verify Docker is Running

```bash
# Check Docker daemon
docker ps

# Should show something like:
# CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

If you get an error, Docker is not running.

### 3. Check Docker Socket

```bash
# Check if Docker socket exists
ls -la /var/run/docker.sock

# Should show:
# srw-rw-rw-  1 root  daemon  0 Jan 10 04:34 /var/run/docker.sock
```

### 4. Test Docker Connection

```bash
# Test Docker API
docker version

# Should show both Client and Server versions
```

## After Starting Docker

Once Docker is running, try sending the task again:

```bash
cd agents/mini-swe-agent
source path.sh
uv run python test_send_task_to_green.py
```

## What Was Working

✅ Green Agent received the task  
✅ Green Agent extracted arvo_id (10055)  
✅ Green Agent downloaded assets (error.txt, repo-vul.tar.gz)  
❌ Green Agent failed to create ARVO container (Docker not running)

## Next Steps

1. Start Docker Desktop
2. Wait for it to fully initialize
3. Run `docker ps` to verify
4. Send the task again

