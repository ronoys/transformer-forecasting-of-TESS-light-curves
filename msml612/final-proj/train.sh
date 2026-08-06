#!/bin/bash
# GPU training job. tess.sh builds data/tess_windows.npz on a CPU node; this trains
# on it and then runs the full metric/figure pass.
#
#   sbatch train.sh
#   sbatch --export=ALL,D_MODEL=128,N_LAYERS=6,MAX_EPOCHS=80 train.sh   # sweep
#
#SBATCH --job-name=msml612-train
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# The a100_1g.5gb MIG slice above is the one this project has already run on. It is
# 5 GB — fine for the default 100k-parameter models, tight if you scale d_model or
# batch size much past 128. For a full card check what the cluster advertises:
#   sinfo -o "%P %G" | sort -u
# then swap the --gres line for the full-A100 name it reports.

set -euo pipefail

mkdir -p logs outputs

DATA=${DATA:-data/tess_windows.npz}
if [ ! -f "$DATA" ]; then
  # Compute nodes have no outbound internet, so the build cannot be rescued here.
  echo "ERROR: $DATA not found. Run the data build first:  sbatch tess.sh" >&2
  exit 1
fi
echo "training on $DATA ($(du -h "$DATA" | cut -f1))"

module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

# --system-site-packages so the module's torch/CUDA build is used as-is; the venv
# only supplies what the module is missing (matplotlib for the figures).
if [ ! -d .venv-train ]; then
  python -m venv --system-site-packages .venv-train
fi
source .venv-train/bin/activate
python -c "import matplotlib" 2>/dev/null || python -m pip install -q matplotlib

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU visible — check --partition and --gres"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PY

echo "=== training ==="
TESS_RUN=1 python real_data_msml612_demo.py

echo "=== evaluation ==="
python evaluate.py --outdir outputs

echo "done. figures and metrics in outputs/"
