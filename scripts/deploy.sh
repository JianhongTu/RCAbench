#!/usr/bin/env bash
set -euo pipefail

# Deploy gateway-related Kubernetes manifests.
# This script will create/update the secret (requires OPENAI_API_KEY env var)
# and apply the Deployment, Service, and Ingress manifests.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Creating/updating secret..."
"${ROOT_DIR}/scripts/create-secret.sh"

echo "Applying gateway secret manifest (if present)"
kubectl apply -f "${ROOT_DIR}/k8s/gateway-secret.yaml" --ignore-not-found

echo "Applying Kubernetes manifests..."
kubectl apply -f "${ROOT_DIR}/k8s/gateway-deployment.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/gateway-service.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/gateway-ingress.yaml"

echo "Deployment applied. Run 'kubectl get pods -l app=rcabench-gateway' to check status."
