#!/bin/bash
# Hyperparameter sweep. One array task per config; each writes its own
# sweep/<tag>/outputs/run_summary.json, which collect_sweep.py then ranks.
#
#   DATA=run_two/data/tess_windows.npz sbatch sweep.sh
#   python collect_sweep.py                       # after the array finishes
#
# Only the model named in ONLY is trained per task, so the transformer sweep does
# not pay to refit the LSTM 24 times. Persistence rides along free as the floor.
#
#SBATCH --job-name=msml612-sweep
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-39%8
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --error=logs/sweep-%A_%a.err

set -euo pipefail

# --- the grid ----------------------------------------------------------------
# 40 configs: 28 transformer, 12 LSTM.
#
# USE_HUBER is swept on every axis and is the most consequential knob here. On the
# run_two window set, 18.5% of windows (142 of 776 stars are intrinsically variable
# or corrupted) carry ~100% of the total squared error, so a plain MSE run tunes
# almost entirely on those stars while the transit-forecasting question lives in the
# quiet 81.5%. Huber is the designed-in defence and had never been switched on.
#
# ANCHOR and POOL are the two axes run one's post-mortem implicates: its transformer
# was ANCHOR=0 POOL=mean and finished within 0.02% of persistence. Both of those
# settings stay in the grid as the control, so the report can show the fix mattered
# rather than asserting it.
GRID=(
  "ONLY=persistence,transformer ANCHOR=0 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=0 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=0 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=0 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=0 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=0 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=0 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=0 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=0 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=0 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=0 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=0 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=mean     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=last     D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=64 N_HEADS=4 N_LAYERS=3 FF_DIM=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=128 N_HEADS=8 N_LAYERS=4 FF_DIM=256 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=128 N_HEADS=8 N_LAYERS=4 FF_DIM=256 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=128 N_HEADS=8 N_LAYERS=4 FF_DIM=256 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,transformer ANCHOR=1 POOL=meanlast D_MODEL=128 N_HEADS=8 N_LAYERS=4 FF_DIM=256 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,lstm ANCHOR=0 LSTM_HIDDEN=64 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=0 LSTM_HIDDEN=64 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=0 LSTM_HIDDEN=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=0 LSTM_HIDDEN=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=1 LSTM_HIDDEN=64 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=1 LSTM_HIDDEN=64 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=1 LSTM_HIDDEN=128 USE_HUBER=0 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=1 LSTM_HIDDEN=128 USE_HUBER=1 LR=1e-3"
  "ONLY=persistence,lstm ANCHOR=0 LSTM_HIDDEN=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,lstm ANCHOR=0 LSTM_HIDDEN=128 USE_HUBER=1 LR=3e-4"
  "ONLY=persistence,lstm ANCHOR=1 LSTM_HIDDEN=128 USE_HUBER=0 LR=3e-4"
  "ONLY=persistence,lstm ANCHOR=1 LSTM_HIDDEN=128 USE_HUBER=1 LR=3e-4"
)

CFG="${GRID[$SLURM_ARRAY_TASK_ID]}"
TAG="cfg$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"

mkdir -p logs "sweep/$TAG"

DATA=${DATA:-run_two/data/tess_windows.npz}
if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA not found. Build it first (see RUN_TWO.md)." >&2
  exit 1
fi

module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

if [ ! -d .venv-train ]; then
  python -m venv --system-site-packages .venv-train
fi
source .venv-train/bin/activate
python -c "import matplotlib" 2>/dev/null || python -m pip install -q matplotlib

echo "=== $TAG: $CFG ==="

# The sweep ranks on val loss only, so it skips the tiny-subset gate and the
# figure pass; both are re-run once for the winning config in final.sh.
env $CFG \
  TAG="$TAG" \
  OUT="sweep/$TAG" \
  DATA="$DATA" \
  SKIP_TINY=1 \
  MAX_EPOCHS=30 \
  PATIENCE=5 \
  BATCH_SIZE=128 \
  TESS_RUN=1 \
  python real_data_msml612_demo.py

echo "done: sweep/$TAG/outputs/run_summary.json"
