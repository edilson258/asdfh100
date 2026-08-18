#!/usr/bin/env bash
set -euo pipefail

# Lightweight installer for h100_massive_inference
# - creates a Python venv
# - installs pure-Python deps from requirements.txt
# - attempts to install a compatible torch wheel based on detected GPU / arch

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

echo "Preparing virtualenv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip & wheel"
pip install -U pip wheel setuptools

echo "Installing core Python packages (explicit list; no requirements.txt used)"
# Install core Python packages directly to avoid relying on requirements.txt
pip install --upgrade \
  ultralytics \
  opencv-python-headless \
  google-cloud-storage \
  loguru \
  rich \
  "psycopg[binary]" \
  pydantic-settings \
  Pillow \
  numpy \
  typing_extensions

# Detect architecture and GPU vendor
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

HAS_NVIDIA=0
HAS_ROCM=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi
if [ -d "/opt/rocm" ] || [ -n "${ROCM_PATH:-}" ]; then
  HAS_ROCM=1
fi

echo "Detecting best torch wheel to install..."
if [ "$ARCH" = "x86_64" ] && [ "$HAS_NVIDIA" -eq 1 ]; then
  echo "NVIDIA GPU detected on x86_64: installing CUDA-enabled PyTorch (stable cu118)."
  pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio --upgrade
elif [ "$HAS_ROCM" -eq 1 ]; then
  echo "ROCm/AMD detected: attempting ROCm PyTorch wheel."
  pip install --index-url https://download.pytorch.org/whl/rocm5.4.2 torch torchvision --upgrade || \
    { echo "ROCm wheel install failed; falling back to CPU wheel."; pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade; }
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  echo "ARM64 detected. Attempting to install a compatible PyTorch wheel."
  # ARM builds vary by distro; attempt official CPU wheel first
  pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade || \
    { echo "ARM wheel install failed. Please install a platform-specific PyTorch wheel manually."; exit 1; }
else
  echo "No GPU detected; installing CPU PyTorch wheel."
  pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade
fi

echo "Installing 'uv' CLI and helpers (uv/uvicorn)"
# "uv" entrypoint is used (uv run ...) — install it and uvicorn for local runs
pip install uv || true
pip install uvicorn[standard] || true

# On Ubuntu systems, suggest installing common native build deps if not present.
if [ -f "/etc/lsb-release" ] || [ -f "/etc/os-release" ]; then
  echo "Detected an Ubuntu-like OS. You may need system packages for wheels/builds."
  echo "If you see build errors, run as root:"
  echo "  sudo apt update && sudo apt install -y build-essential python3-dev libffi-dev libssl-dev"
fi

echo "Installation complete. Activate with: source $VENV_DIR/bin/activate"
echo "Run the pipeline with: uv run python main.py <command>"
