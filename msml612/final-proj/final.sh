#!/bin/bash
# Final three-way comparison at the sweep's winning hyperparameters, followed by
# the full metric/figure pass. This is the run the report is written from.
#
#   DATA=run_two/data/tess_windows.npz sbatch final.sh
#
# Fill in WINNERS below from `python collect_sweep.py` first — it prints the exact
# env line to paste for each model. All three models train in one job so they share
# one split and land in one test_predictions.npz.
#
#SBATCH --job-name=msml612-final
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/final-%j.out
#SBATCH --error=logs/final-%j.err

set -euo pipefail

# --- paste the sweep winners here -------------------------------------------
# PLACEHOLDERS until the sweep runs — they are only a guess at the winner, so
# replace both lines with what `python collect_sweep.py` prints. It emits them
# already model-prefixed, which is what lets the two models disagree on shared
# settings like LR inside a single job. Copy the line verbatim.
LSTM_WINNER="LSTM_ANCHOR=1 LSTM_LSTM_HIDDEN=128 LSTM_LR=1e-3"
TRANSFORMER_WINNER="TRANSFORMER_ANCHOR=1 TRANSFORMER_POOL=meanlast TRANSFORMER_D_MODEL=128 TRANSFORMER_N_HEADS=8 TRANSFORMER_N_LAYERS=4 TRANSFORMER_FF_DIM=256 TRANSFORMER_LR=1e-3"

OUT_DIR=${OUT_DIR:-run_two}

mkdir -p logs "$OUT_DIR/outputs"

DATA=${DATA:-data/tess_windows.npz}
if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA not found. Build it first (see RUN_TWO.md)." >&2
  exit 1
fi
echo "training on $DATA ($(du -h "$DATA" | cut -f1))"

module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

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

echo "=== training (final) ==="
# Longer budget and more patience than the sweep: the sweep only had to rank
# configs, this run has to squeeze the winner. Tiny-subset gate left ON here as the
# pipeline check it was written to be.
env $LSTM_WINNER $TRANSFORMER_WINNER \
  TAG=final \
  OUT="$OUT_DIR" \
  DATA="$DATA" \
  MAX_EPOCHS=80 \
  PATIENCE=10 \
  BATCH_SIZE=128 \
  TESS_RUN=1 \
  python real_data_msml612_demo.py

echo "=== evaluation ==="
python evaluate.py \
  --preds "$OUT_DIR/test_predictions.npz" \
  --history "$OUT_DIR/outputs/history.json" \
  --outdir "$OUT_DIR/outputs"

echo "done. figures and metrics in $OUT_DIR/outputs/"
