#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Deleting gateway Ingress, Service, Deployment..."
kubectl delete -f "${ROOT_DIR}/k8s/gateway-ingress.yaml" --ignore-not-found
kubectl delete -f "${ROOT_DIR}/k8s/gateway-service.yaml" --ignore-not-found
kubectl delete -f "${ROOT_DIR}/k8s/gateway-deployment.yaml" --ignore-not-found

echo "Deleting secret..."
kubectl delete secret rcabench-gateway-secret --ignore-not-found

echo "Teardown complete."
