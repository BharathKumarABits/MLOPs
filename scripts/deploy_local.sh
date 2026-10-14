#!/usr/bin/env bash
# One-command local deployment: Minikube cluster + image build + Kubernetes manifests.
set -euo pipefail

IMAGE="heart-risk-api:1.0"

cd "$(dirname "$0")/.."

if ! minikube status >/dev/null 2>&1; then
  minikube start --driver=docker --cpus=2 --memory=4096
fi

docker build -t "$IMAGE" .
minikube image load "$IMAGE"

kubectl apply -f k8s/
kubectl rollout status deployment/heart-risk-api --timeout=180s
kubectl get nodes,deploy,pods,svc
