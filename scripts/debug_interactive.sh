#!/bin/bash
# Interactive debugging script for RCAbench tasks
# Usage: ./debug_interactive.sh <task_id>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <task_id>"
    echo "Example: $0 10055"
    exit 1
fi

TASK_ID=$1
PATCH_FILE="/tmp/patch_${TASK_ID}.diff"
IMAGE_NAME="n132/arvo:${TASK_ID}-vul"

echo "=== INTERACTIVE DEBUGGING FOR TASK ${TASK_ID} ==="
echo ""

# Download patch file
echo "1. Downloading patch file..."
if ! curl -s -f -L -o "${PATCH_FILE}" "https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/arvo/${TASK_ID}/patch.diff"; then
    echo "❌ Failed to download patch file"
    exit 1
fi
echo "✅ Patch file downloaded to ${PATCH_FILE}"

# Pull Docker image
echo ""
echo "2. Pulling Docker image..."
if ! docker pull "${IMAGE_NAME}"; then
    echo "❌ Failed to pull Docker image"
    exit 1
fi
echo "✅ Docker image pulled"

echo ""
echo "3. Starting interactive container..."
echo "You'll be dropped into a bash shell inside the container."
echo "The container will automatically detect the source directory."
echo "Run these commands step by step:"
echo ""
echo "  # The container starts in the detected source directory"
echo "  pwd  # Check current directory"
echo "  ls -la | head -10"
echo "  cat /tmp/patch.diff | head -20  # Check patch content"
echo "  patch -p1 --force < /tmp/patch.diff"
echo "  arvo compile"
echo "  timeout 600 arvo"
echo ""
echo "Type 'exit' when done to stop the container."
echo ""

# Run interactive container
docker run --rm -it -v "${PATCH_FILE}:/tmp/patch.diff" "${IMAGE_NAME}" /bin/bash