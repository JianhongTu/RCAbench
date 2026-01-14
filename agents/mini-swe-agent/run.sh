#!/bin/bash
# Run agents and send a task
# Usage: ./run.sh [arvo_id|--all]
# Example: ./run.sh 14368  # Run specific task
# Example: ./run.sh  # Uses all task_ids from scenario.toml
# Example: ./run.sh --all  # Explicitly run all tasks from scenario.toml

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for --all flag
RUN_ALL=false
if [ "$1" = "--all" ]; then
    RUN_ALL=true
    shift
fi

# Get arvo_id from command line or scenario.toml
if [ -z "$1" ] && [ "$RUN_ALL" = false ]; then
    # Try to get all from scenario.toml using Python
    TASK_IDS=$(python3 -c "
import sys
from pathlib import Path
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit(1)

scenario_file = Path('scenario.toml')
if scenario_file.exists():
    with open(scenario_file, 'rb') as f:
        config = tomllib.load(f)
    task_ids = config.get('config', {}).get('task_ids', [])
    if task_ids:
        print(' '.join(str(tid) for tid in task_ids))
        sys.exit(0)
sys.exit(1)
" 2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$TASK_IDS" ]; then
        echo "❌ Error: No ARVO ID provided and none found in scenario.toml"
        echo "Usage: $0 [arvo_id|--all]"
        echo "Example: $0 14368"
        echo "Example: $0 --all  # Run all tasks from scenario.toml"
        exit 1
    fi
    # Use all task IDs
    ARVO_IDS=($TASK_IDS)
    echo "Using all ARVO IDs from scenario.toml: ${ARVO_IDS[@]}"
elif [ "$RUN_ALL" = true ]; then
    # Get all from scenario.toml
    TASK_IDS=$(python3 -c "
import sys
from pathlib import Path
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit(1)

scenario_file = Path('scenario.toml')
if scenario_file.exists():
    with open(scenario_file, 'rb') as f:
        config = tomllib.load(f)
    task_ids = config.get('config', {}).get('task_ids', [])
    if task_ids:
        print(' '.join(str(tid) for tid in task_ids))
        sys.exit(0)
sys.exit(1)
" 2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$TASK_IDS" ]; then
        echo "❌ Error: --all specified but no task_ids found in scenario.toml"
        exit 1
    fi
    ARVO_IDS=($TASK_IDS)
    echo "Running all ARVO IDs from scenario.toml: ${ARVO_IDS[@]}"
else
    # Use provided ARVO ID
    ARVO_IDS=("$1")
fi

# Source path.sh to set environment variables
if [ -f "path.sh" ]; then
    source path.sh
fi

echo "============================================================"
echo "Starting agents and sending tasks"
echo "============================================================"
echo ""

# Use first ARVO ID for log directory (agents need one log dir)
FIRST_ARVO_ID="${ARVO_IDS[0]}"

# Start agents in background (use first ARVO ID for log directory)
echo "Starting agents with ARVO ID: $FIRST_ARVO_ID (for log directory)..."
python start_agents.py --arvo-id "$FIRST_ARVO_ID" > /dev/null 2>&1 &
AGENTS_PID=$!

# Wait for agents to start
echo "Waiting for agents to start..."
sleep 5

# Check if agents are running by checking if ports are listening
if ! nc -z localhost 9009 2>/dev/null || ! nc -z localhost 9019 2>/dev/null; then
    echo "⚠️  Warning: Agents may not be fully started yet, but continuing..."
fi

# Send tasks
echo ""
if [ ${#ARVO_IDS[@]} -eq 1 ]; then
    echo "Sending task ${ARVO_IDS[0]}..."
    python send_task.py "${ARVO_IDS[0]}"
else
    echo "Sending ${#ARVO_IDS[@]} tasks: ${ARVO_IDS[@]}"
    python send_task.py --all
fi

# Wait for agents (they will keep running)
echo ""
echo "============================================================"
echo "Agents are running in background"
echo "Green Agent: http://127.0.0.1:9009"
echo "Purple Agent: http://127.0.0.1:9019"
echo ""
echo "To stop agents:"
echo "  pkill -f 'green_agent_server.py'"
echo "  pkill -f 'purple_agent_server.py'"
echo "Or press Ctrl+C to stop this script"
echo "============================================================"

# Wait for user interrupt
trap "echo ''; echo 'Stopping agents...'; pkill -f 'green_agent_server.py' 2>/dev/null || true; pkill -f 'purple_agent_server.py' 2>/dev/null || true; echo '✅ Agents stopped'; exit 0" INT TERM

# Keep script running until interrupted
while true; do
    sleep 1
done

