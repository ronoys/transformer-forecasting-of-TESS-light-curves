# Transformer Forecasting of TESS Light Curves

This repository contains an end-to-end forecasting pipeline for TESS light curves. It builds supervised windows from real TESS time-series data, trains comparable forecasting models, and generates shared evaluation outputs for the final report.

## Repository Layout

```text
msml612/final-proj/
  run_two_handoff.py              Build the sector-coverage TESS handoff dataset
  zaratan_handoff.py              Shared handoff helpers: download, clean, window, split, sanity check
  merge_handoff_parts.py          Merge handoff outputs
  real_data_msml612_demo.py       Main training script
  evaluate.py                     Generate metrics, plots, and report artifacts
  eval_metrics.py                 Metric helpers
  eval_charts.py                  Figure helpers
  filter_windows.py               Optional quality filter for outlier-heavy windows
  sweep.sh                        Zaratan hyperparameter sweep job
  collect_sweep.py                Rank sweep results and print winning configs
  final.sh                        Final Zaratan three-way comparison job
  train.sh                        Training job wrapper
  run_live_demo.sh                Guided mini live demo
  demo_running_model.py           Local live inference graph server
  demo_handoff_from_existing.py   Demo handoff builder from existing real data
  demo_make_predictions.py        Demo prediction artifact creator
  requirements-zaratan.txt        Handoff dependencies for Zaratan/local setup
  RUN_TWO.md                      Notes for the larger sector-coverage dataset
  TUNING.md                       Notes for tuned three-way comparison
  ZARATAN_HANDOFF.md              Notes for Zaratan handoff jobs
  run_two/                       Real-data handoff and outputs, when present
  outputs/                        Default evaluation figures and metrics
```

## Data Contract

The model handoff file is a compressed NumPy file:

```text
tess_windows.npz
```

Expected keys:

```text
X              float32 [N, 256]   past flux context window
y              float32 [N, 32]    future flux target
tic_id         int64   [N]        source TIC ID
split          str     [N]        train / val / test split by TIC
transit_depth  float32 [N]        simple dip-depth proxy
sector         int16   [N]        TESS sector, included in run_two
```

Splitting happens by `tic_id`, so windows from the same star do not appear in more than one split.

## Local Setup

From the repo root:

```bash
cd msml612/final-proj
python3 -m venv .venv-local
source .venv-local/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas matplotlib torch lightkurve astroquery
```

## Run the Guided Mini Demo

```bash
cd msml612/final-proj
./run_live_demo.sh
```

The script does three things:

```text
1. Builds a tiny demo handoff and prints window-creation logs
2. Starts a local browser demo that runs the saved Transformer on that tiny handoff
3. Generates tiny prediction/evaluation outputs
```

When Step 2 starts, open:

```text
http://127.0.0.1:8767/
```

After Step 3 finishes, click **Open figures** in the demo page or visit:

```text
http://127.0.0.1:8767/figures
```

Generated demo artifacts are written to:

```text
demo_handoff/
demo_outputs/
```

## Build the Larger Run Two Dataset

`run_two_handoff.py` builds the sector-coverage dataset used for the larger real-data run.

Default intent:

```text
sectors: 25-107
TICs per sector: 10
max windows per light curve: 100
output: run_two/data/tess_windows.npz
manifest: run_two/lightcurve_manifest.csv
```

Run locally or on the Zaratan login node:

```bash
cd msml612/final-proj
python3 run_two_handoff.py \
  --out-dir run_two \
  --start-sector 25 \
  --end-sector 107 \
  --tics-per-sector 10 \
  --max-windows-per-lightcurve 100 \
  --per-tic-timeout 60
```

Run smoke test:

```bash
python3 run_two_handoff.py \
  --out-dir run_two_smoke \
  --start-sector 25 \
  --end-sector 25 \
  --tics-per-sector 3 \
  --max-windows-per-lightcurve 20 \
  --per-tic-timeout 60
```

## Train Locally

Use the real-data training script with a handoff file:

```bash
cd msml612/final-proj
DATA=run_two/data/tess_windows.npz \
TESS_RUN=1 \
python3 real_data_msml612_demo.py
```

Faster local run:

```bash
SKIP_TINY=1 \
MAX_EPOCHS=3 \
PATIENCE=1 \
BATCH_SIZE=4096 \
LSTM_HIDDEN=16 \
D_MODEL=16 \
N_HEADS=1 \
N_LAYERS=1 \
FF_DIM=32 \
DATA=run_two/data/tess_windows.npz \
TESS_RUN=1 \
python3 real_data_msml612_demo.py
```

Main training artifacts:

```text
results_table.csv
transformer_best.pt
test_predictions.npz
outputs/history.json
```

## Evaluation

After training creates `test_predictions.npz`, run:

```bash
cd msml612/final-proj
python3 evaluate.py \
  --preds test_predictions.npz \
  --outdir outputs
```

Viewing Outputs for a Specific Run:

```bash
python3 evaluate.py \
  --preds run_two/test_predictions.npz \
  --history run_two/outputs/history.json \
  --outdir run_two/outputs
```

List of evaluation output files:

```text
metrics_summary.csv
metrics_by_horizon.csv
metrics_by_depth.csv
metrics_report.md
metrics_headline.png
error_by_horizon.png
error_distribution.png
residual_heatmap.png
dip_recovery.png
detection_roc.png
examples_forecast.png
```

## Using Zaratan for Larger datasets


Basic setup:

```bash
cd msml612/final-proj
mkdir -p logs data
module load python || true
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-zaratan.txt
```

Small handoff:

```bash
python zaratan_handoff.py --n-targets 25
```

Shard-based handoff jobs:

```bash
sbatch tess.sh
```

Merge outputs:

```bash
source .venv/bin/activate
python merge_handoff_parts.py
```

The merged handoff is written to:

```text
data/tess_windows.npz
```

## Zaratan Training

Train on the run two handoff:

```bash
cd msml612/final-proj
DATA=run_two/data/tess_windows.npz sbatch train.sh
```

Check job status:

```bash
squeue -u "$USER"
tail -f logs/train-<jobid>.out
```

## Hyperparameter Sweep and Final Comparison

Run the sweep:

```bash
cd msml612/final-proj
DATA=run_two/data/tess_windows.npz sbatch sweep.sh
```

Collect winners:

```bash
python collect_sweep.py
```

Paste the printed winner lines into `final.sh`, then run:

```bash
DATA=run_two/data/tess_windows.npz sbatch final.sh
```

Final outputs are written under:

```text
run_two/
run_two/outputs/
```
