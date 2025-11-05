#!/usr/bin/env bash
set -euo pipefail

# Create or update the Kubernetes secret containing the OPENAI API key.
# Requires OPENAI_API_KEY to be set in the environment.

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY environment variable is not set."
  echo "Set it in your shell or put it in gateway/.env and run 'export \\$(cat gateway/.env | xargs)'"
  exit 1
fi

kubectl create secret generic openai-api \
  --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secret openai-api created/updated"
