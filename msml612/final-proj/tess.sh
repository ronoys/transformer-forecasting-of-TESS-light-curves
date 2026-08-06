#!/bin/bash
#SBATCH --job-name=msml612-handoff
#SBATCH --account=msml612-class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/handoff-%j.out
#SBATCH --error=logs/handoff-%j.err

set -euo pipefail

module load python || true

mkdir -p logs data

if [ ! -d .venv ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-zaratan.txt

python zaratan_handoff.py \
  --out data/tess_windows.npz \
  --n-targets 250 \
  --max-products 4 \
  --max-windows-per-star 500
