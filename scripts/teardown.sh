#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Deleting gateway Ingress, Service, Deployment..."
kubectl delete -f "${ROOT_DIR}/k8s/rcabench-ingress.yaml" --ignore-not-found
kubectl delete -f "${ROOT_DIR}/k8s/rcabench-service.yaml" --ignore-not-found
kubectl delete -f "${ROOT_DIR}/k8s/rcabench-deployment.yaml" --ignore-not-found

echo "Deleting secrets..."
kubectl delete secret openai-api --ignore-not-found
kubectl delete secret rcabench-tls --ignore-not-found

echo "Teardown complete."
