# Run Three: tuned three-way comparison

Run one's transformer landed within 0.02% of persistence on every metric (skill
+0.0%, R² -0.137 against persistence's -0.137). That is not a hyperparameter
result — the head never learned a delta worth making. Two suspects, both now
switchable and both swept rather than assumed:

- **`ANCHOR`** — feed the encoder `x - x[-1]` instead of raw flux. The level of a
  detrended window says nothing about its next 32 points, so subtracting it stops
  the model spending capacity on per-star offsets.
- **`POOL`** — `mean` averages the 32 steps that matter with 224 that do not.
  `last` uses the step the forecast continues from; `meanlast` concatenates both.

Run one's setting was `ANCHOR=0 POOL=mean`, kept in the grid as the control, so
the report can show the fix mattered instead of asserting it.

## The run_two window set is outlier-dominated — read this first

A 3-epoch pass over `run_two/data/tess_windows.npz` reports MAE **180,312 ppm**.
Run one reported 6,589 ppm on the same metric. The data, not the model, moved:

| | run one | run two |
|---|---|---|
| star selection | 8 hand-picked planet hosts | 10 random TICs per sector, 25–107 |
| stars | 8 | 776 |
| windows | ~2,000 | 77,700 |
| median window range | — | 0.0075 (sane) |
| p99 window range | — | 10.3 (1000% flux swing) |

`run_two_handoff.py` takes whatever `Observations.query_criteria` returns per
sector, so the sample includes intrinsically variable stars (pulsators, eclipsing
binaries, flares) and some corrupted targets — one has 1st-percentile flux of
-3.65, which is unphysical. The damage is concentrated, not diffuse: **142 of 776
stars have >50% bad windows, while 614 have <5%.**

The consequence for tuning is the important part:

> 18.5% of windows carry **~100.0%** of the total squared error, and 98.1% of the
> total absolute error.

So a plain-MSE sweep ranks configs almost entirely on how well they fit variable
stars, while the transit-forecasting question this project asks lives in the quiet
81.5%. `USE_HUBER` is therefore swept on every axis of the grid. It was available
from the start (`use_huber` in `Config`) and had never been switched on.

**This is a mitigation, not a fix.** Huber reweights the outliers; it does not
remove them, and the headline MAE will stay large and hard to compare against run
one. If the report needs numbers comparable to run one, the window set needs a
quality cut at build time — see "Optional quality filter" below.

## Sequence

Everything below runs from `msml612/final-proj/` on Zaratan.

```bash
# 0. one-time, if .venv-train does not exist yet — sweep.sh creates it otherwise
sinfo -o "%P %G" | sort -u        # confirm the a100_1g.5gb MIG slice still exists

# 1. sweep: 40 configs (28 transformer, 12 LSTM), 8 at a time
DATA=run_two/data/tess_windows.npz sbatch sweep.sh
squeue -u "$USER"

# 2. rank them once the array drains
python collect_sweep.py            # writes sweep/results.csv, prints winners

# 3. paste the two printed env lines into final.sh (LSTM_WINNER / TRANSFORMER_WINNER)
#    prefixing each knob with the model name, then:
DATA=run_two/data/tess_windows.npz sbatch final.sh

# 4. the report
cat run_two/outputs/metrics_report.md
```

## What each piece does

| file | role |
|---|---|
| `sweep.sh` | 40-task array; each task trains one model at one config into `sweep/cfgNN/` |
| `collect_sweep.py` | ranks every run on **validation** loss, prints the env line to paste |
| `final.sh` | trains all three models at their winning configs in one job, then runs `evaluate.py` |

Ranking is on validation loss only. The test split picks nothing — it is read once,
in step 3, to produce the numbers that go in the report.

## Knobs added to `Config`

| env | default | meaning |
|---|---|---|
| `ANCHOR` | `1` | feed the encoder/LSTM `x - x[-1]` |
| `POOL` | `meanlast` | transformer pooling: `mean` \| `last` \| `meanlast` |
| `USE_HUBER` | `0` | Huber instead of MSE — swept, see the outlier section above |
| `WEIGHT_DECAY` | `1e-2` | AdamW weight decay |
| `ONLY` | all three | comma list of models to train, e.g. `ONLY=persistence,transformer` |
| `OUT` | `.` | artifact root, so parallel array tasks do not collide |
| `TAG` | `""` | label recorded in `run_summary.json` |

Any `Config` field can also be set per model with a `<MODEL>_<FIELD>` prefix —
`LSTM_LR=3e-4 TRANSFORMER_LR=1e-3` — which is how `final.sh` gives each model its
own sweep winner inside a single job, on one shared split.

## New artifact

Each run now writes `<OUT>/outputs/run_summary.json`: config (global and per
model), parameter counts, best validation loss, epochs actually run, and the test
table. `collect_sweep.py` reads these; the report cites them for the tuning
section.

## Optional quality filter

`filter_windows.py` applies a quality cut and writes a **new** file, leaving the
original untouched so both runs stay reproducible:

```bash
python filter_windows.py --in  run_two/data/tess_windows.npz \
                         --out run_two/data/tess_windows_clean.npz
# 77,700 windows / 776 stars  ->  56,018 windows / 567 stars (72.1% kept)
# window range p99: 10.3  ->  0.045
```

It drops windows whose flux range exceeds `--max-range` (default 5%), then drops
any star that is more than `--max-star-badfrac` bad windows — a variable star's few
quiet windows are still that star, and with split-by-star, keeping them leaks its
behaviour across the split. Splits stay populated (39,078 / 8,626 / 8,314).

Whether to use it is a call for the writeup, not a default:

- **Unfiltered** keeps every star, so the headline MAE stays ~180,000 ppm and is
  not comparable to run one. Huber is doing the heavy lifting.
- **Filtered** is comparable to run one's regime and isolates the transit question,
  but discards real astrophysical variability — which some readers will call
  cherry-picking unless the cut is stated up front.

The most defensible option for a report is to run `final.sh` on both and present
them side by side, with the cut criterion stated. That is one extra job:

```bash
DATA=run_two/data/tess_windows_clean.npz OUT_DIR=run_two_clean sbatch final.sh
```

## If the 5 GB MIG slice is too small

`D_MODEL=128 N_LAYERS=4` at `BATCH_SIZE=128` is the largest config in the grid and
is the one that will OOM first. Either drop to `BATCH_SIZE=64` for those tasks or
swap the `--gres` line in `sweep.sh` for a full-card name from `sinfo -o "%P %G"`.
