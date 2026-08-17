# -*- coding: utf-8 -*-
"""Rank the sweep runs and print the winning config per model.

    python collect_sweep.py                       # ranks sweep/*/outputs/run_summary.json
    python collect_sweep.py --csv sweep/results.csv

Ranking is on validation loss, never on the test metrics that also live in each
summary — the test split picks nothing here, it only reports the final numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Config fields the sweep actually varies; everything else is constant across runs
# and would only pad the table.
SWEPT = ["anchor", "pool", "d_model", "n_heads", "n_layers", "ff_dim",
         "lstm_hidden", "lr", "weight_decay", "batch_size", "use_huber"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweepdir", default="sweep", help="directory of per-config run dirs")
    p.add_argument("--csv", default="sweep/results.csv", help="where to write the full table")
    p.add_argument("--top", type=int, default=5, help="rows to print per model")
    return p.parse_args()


def load_runs(sweepdir: Path):
    """One row per (run, trained model). Persistence rows are dropped: it has no
    hyperparameters, so ranking it across configs is meaningless."""
    rows, missing = [], []
    for summary in sorted(sweepdir.glob("*/outputs/run_summary.json")):
        try:
            with open(summary) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            missing.append(f"{summary}: {e}")
            continue
        base, per_model = s.get("config", {}), s.get("config_per_model", {})
        test = s.get("test", {})
        for model, val in s.get("val_loss", {}).items():
            if model == "persistence":
                continue
            cfg = per_model.get(model, base)   # per-model overrides win when present
            row = {"tag": s.get("tag") or summary.parts[-3], "model": model,
                   "val_loss": val,
                   "epochs": s.get("epochs_run", {}).get(model),
                   "n_params": s.get("n_params", {}).get(model),
                   "persistence_val": s.get("val_loss", {}).get("persistence")}
            row.update({k: cfg.get(k) for k in SWEPT})
            row.update({k: test.get(model, {}).get(k)
                        for k in ("MAE_all", "MAE_transit", "MAE_quiet")})
            rows.append(row)
    return rows, missing


def val_skill(row):
    """Fraction of the persistence val loss removed. Comparable across models,
    and the sign immediately says whether a config beat doing nothing at all."""
    floor = row.get("persistence_val")
    if not floor or row["val_loss"] is None:
        return float("nan")
    return 1.0 - row["val_loss"] / floor


def main():
    args = parse_args()
    sweepdir = Path(args.sweepdir)
    if not sweepdir.is_dir():
        raise SystemExit(f"no sweep directory at {sweepdir}/ — has the array job run?")

    rows, missing = load_runs(sweepdir)
    for m in missing:
        print(f"WARNING unreadable summary  {m}")
    if not rows:
        raise SystemExit(f"no run_summary.json found under {sweepdir}/*/outputs/")

    for r in rows:
        r["val_skill"] = val_skill(r)

    expected = len(list(sweepdir.glob("*/")))
    print(f"{len(rows)} trained models across {expected} run dirs under {sweepdir}/\n")

    fields = (["tag", "model", "val_loss", "val_skill", "persistence_val", "epochs", "n_params"]
              + SWEPT + ["MAE_all", "MAE_transit", "MAE_quiet"])
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["model"], r["val_loss"])))
    print(f"full table -> {out}\n")

    for model in sorted({r["model"] for r in rows}):
        sub = sorted((r for r in rows if r["model"] == model), key=lambda r: r["val_loss"])
        print(f"=== {model}: {len(sub)} configs, best first ===")
        for r in sub[: args.top]:
            knobs = " ".join(f"{k}={r[k]}" for k in SWEPT
                             if r[k] is not None and _relevant(k, model))
            print(f"  {r['tag']}  val={r['val_loss']:.5e}  skill={r['val_skill']:+.2%}  "
                  f"ep={r['epochs']}  {knobs}")
        best = sub[0]
        print(f"  -> winner {best['tag']}, export for final.sh:")
        print(f"     {export_line(best)}\n")


def _relevant(key, model):
    if key == "lstm_hidden":
        return model == "lstm"
    if key in ("pool", "d_model", "n_heads", "n_layers", "ff_dim"):
        return model == "transformer"
    return True


def export_line(row):
    """The env assignments to paste into final.sh for this model.

    Emitted already prefixed with the model name (`LSTM_LR=...`), because final.sh
    trains all three models in one process and the prefix is what keeps each
    model's winning config from overwriting the others'.
    """
    model = row["model"]
    keys = [k for k in SWEPT if row.get(k) is not None and _relevant(k, model)]
    parts = []
    for k in keys:
        v = row[k]
        if isinstance(v, bool):
            v = int(v)
        parts.append(f"{model.upper()}_{k.upper()}={v}")
    return " ".join(parts)


if __name__ == "__main__":
    main()
