#!/usr/bin/env bash
# Deploy Prometheus and Grafana (with a provisioned dashboard) into the Minikube cluster.
set -euo pipefail

cd "$(dirname "$0")/.."

apply_cm() {  # apply_cm <name> <file | key=file>
  kubectl create configmap "$1" --from-file="$2" --dry-run=client -o yaml | kubectl apply -f -
}

apply_cm prometheus-config prometheus.yml=monitoring/prometheus-config.yml
apply_cm grafana-datasources monitoring/grafana/datasource.yaml
apply_cm grafana-dashboard-provider monitoring/grafana/dashboard-provider.yaml
apply_cm grafana-dashboards monitoring/grafana/heart-risk-api.json

kubectl apply -f k8s/
kubectl apply -f monitoring/prometheus-deploy.yaml -f monitoring/grafana-deploy.yaml
kubectl rollout status deployment/prometheus --timeout=180s
kubectl rollout status deployment/grafana --timeout=180s
kubectl get pods,svc
