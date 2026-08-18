# h100_massive_inference — Installation Notes

This project uses a Python virtual environment and requires a platform-appropriate
build of PyTorch for GPU acceleration. Use the provided `install.sh` to create
and populate a virtualenv; the installer deliberately avoids `requirements.txt`
and installs the essential Python packages directly so it can be tuned per-host.

Quick start (Ubuntu x86_64 / ARM64)

```bash
cd h100_massive_inference
./install.sh
source .venv/bin/activate
uv run python main.py start /path/to/images --dry-run
```

Ubuntu-specific notes
- The installer will not run `apt` automatically. If you encounter build or
  wheel errors, install common build deps first:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev libffi-dev libssl-dev
```

ARM64 and AMD/ROCm
- ARM64: official PyTorch ARM64 wheels are distribution-specific. If the
  automatic installer fails, follow the PyTorch ARM instructions at
  https://pytorch.org/get-started/locally/ and choose the wheel that matches
  your OS and Python version.
- AMD/ROCm: ensure ROCm is installed and `/opt/rocm` exists. Installing ROCm
  is a system-level operation; refer to the ROCm docs for your Ubuntu version.

Avoiding `requirements.txt`
- The install script installs required packages directly to facilitate multi-arch
  customizations. If you prefer a lockfile for deployments, generate one from
  your environment after a successful install (e.g., `pip freeze > requirements.txt`).

