#!/usr/bin/env bash
# One-shot environment setup on a fresh vast.ai Ubuntu/CUDA instance.
# Installs uv, a Python 3.12 venv, lerobot[training] with CUDA torch, ffmpeg.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq ffmpeg rsync git curl > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/so101 && cd /workspace/so101
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'lerobot[training]'
.venv/bin/python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('lerobot', lerobot.__version__)"
echo SETUP_DONE
