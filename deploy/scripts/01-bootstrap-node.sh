#!/usr/bin/env bash
# Готовит ноду 192.168.31.241 под k3s + локальный registry.
# Предполагается: NVIDIA driver и nvidia-container-toolkit уже установлены.
set -euo pipefail

NODE_IP="${NODE_IP:-192.168.31.241}"
REGISTRY_PORT="${REGISTRY_PORT:-5000}"

echo "==> 1. Установка k3s (без traefik можно отключить флагом)"
curl -sfL https://get.k3s.io | sh -s - \
    --write-kubeconfig-mode=644 \
    --node-ip="${NODE_IP}" \
    --tls-san="${NODE_IP}"

echo "==> 2. Настройка containerd на nvidia runtime"
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd
sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl > /dev/null <<'EOF'
version = 2

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  privileged_without_host_devices = false
  runtime_engine = ""
  runtime_root = ""
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
EOF
sudo systemctl restart k3s

echo "==> 3. RuntimeClass nvidia + NVIDIA device plugin"
kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml

echo "==> 4. Локальный registry на ноде"
sudo mkdir -p /var/lib/registry
sudo docker run -d --restart=always --name registry \
    -p "${REGISTRY_PORT}:5000" \
    -v /var/lib/registry:/var/lib/registry \
    registry:2 || echo "(registry уже запущен)"

echo "==> 5. Доверяем insecure registry в k3s/containerd"
sudo tee /etc/rancher/k3s/registries.yaml > /dev/null <<EOF
mirrors:
  "${NODE_IP}:${REGISTRY_PORT}":
    endpoint:
      - "http://${NODE_IP}:${REGISTRY_PORT}"
configs:
  "${NODE_IP}:${REGISTRY_PORT}":
    tls:
      insecure_skip_verify: true
EOF
sudo systemctl restart k3s

echo "==> 6. Хост-директории для shared cache (hostPath PV)"
sudo mkdir -p /var/lib/llm/{hf-cache,paddlex,data,workspace}
sudo chmod -R 777 /var/lib/llm

echo "==> 7. Проверка"
kubectl get nodes -o wide
kubectl get runtimeclass
kubectl describe node | grep -A2 "nvidia.com/gpu" || true

echo "Готово. kubeconfig: /etc/rancher/k3s/k3s.yaml"
