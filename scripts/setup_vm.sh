#!/usr/bin/env bash
# Rebuild the lab-VM toolchain after a reset (system installs are lost on Terminate).
# Safe to re-run. Docker and Minikube ship with the lab image.
set -u

KUBECTL_VERSION="v1.32.0"   # keep in step with the Minikube Kubernetes version
ENV_PY="$HOME/workspace/envs/heart-mlops/bin/python"

if ! command -v kubectl >/dev/null 2>&1 \
   || ! kubectl version --client 2>/dev/null | grep -q "${KUBECTL_VERSION}"; then
  echo ">> installing kubectl ${KUBECTL_VERSION}"
  curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl && rm -f /tmp/kubectl
fi

if ! command -v gh >/dev/null 2>&1; then
  echo ">> installing GitHub CLI"
  sudo dnf install -y 'dnf-command(config-manager)' \
    && sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo \
    && sudo dnf install -y gh
fi

echo ">> VS Code extensions"
for ext in ms-python.python ms-toolsai.jupyter ms-azuretools.vscode-docker \
           ms-kubernetes-tools.vscode-kubernetes-tools redhat.vscode-yaml; do
  code --install-extension "$ext" >/dev/null 2>&1 || echo "   could not install $ext"
done

if [ -x "$ENV_PY" ]; then
  echo ">> registering Jupyter kernel"
  "$ENV_PY" -m ipykernel install --user --name heart-mlops \
    --display-name "Python 3.11 (heart-mlops)" >/dev/null
fi

echo ">> done. Versions:"
docker --version; minikube version --short; kubectl version --client 2>/dev/null | head -1
echo "If git pushes fail: gh auth login -h github.com -p https -s workflow -w && gh auth setup-git"
