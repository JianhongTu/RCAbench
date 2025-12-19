#!/bin/bash
# Local verification script for RCAbench tasks
# Usage: ./local_verify.sh <task_id>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <task_id>"
    echo "Example: $0 10055"
    exit 1
fi

TASK_ID=$1
PATCH_FILE="/tmp/patch_${TASK_ID}.diff"
IMAGE_NAME="n132/arvo:${TASK_ID}-vul"

echo "=== LOCAL VERIFICATION FOR TASK ${TASK_ID} ==="
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

# Run verification
echo ""
echo "3. Running verification in container..."
docker run --rm -v "${PATCH_FILE}:/tmp/patch.diff" "${IMAGE_NAME}" /bin/bash