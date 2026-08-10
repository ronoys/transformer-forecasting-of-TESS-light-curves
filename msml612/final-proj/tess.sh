#!/bin/bash
#SBATCH --job-name=msml612-handoff
#SBATCH --account=msml612-class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-249%20
#SBATCH --output=logs/handoff-%A_%a.out
#SBATCH --error=logs/handoff-%A_%a.err

set -euo pipefail

module load python || true

mkdir -p logs data/parts

if [ ! -d .venv ] || [ ! -f data/tic_ids.txt ]; then
  echo "ERROR: run the setup commands before submitting this array job:" >&2
  echo "  module load python || true" >&2
  echo "  python -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  python -m pip install --upgrade pip" >&2
  echo "  python -m pip install -r requirements-zaratan.txt" >&2
  echo "  python zaratan_handoff.py --n-targets 250 --write-tic-file data/tic_ids.txt" >&2
  exit 1
fi

source .venv/bin/activate

python zaratan_handoff.py \
  --tic-file data/tic_ids.txt \
  --out "data/parts/tess_windows_part_${SLURM_ARRAY_TASK_ID}.npz" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --partial \
  --max-products 1 \
  --max-windows-per-star 200
