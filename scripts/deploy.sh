#!/usr/bin/env bash
set -euo pipefail

# Deploy gateway-related Kubernetes manifests.
# This script will create/update the secret (requires OPENAI_API_KEY env var)
# and apply the Deployment, Service, and Ingress manifests.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Creating/updating secret..."
"${ROOT_DIR}/scripts/create-secret.sh"

echo "Applying Kubernetes manifests..."
kubectl apply -f "${ROOT_DIR}/k8s/rcabench-deployment.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/rcabench-service.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/rcabench-ingress.yaml"

echo "Deployment applied. Run 'kubectl get pods -l app=rcabench' to check status."
