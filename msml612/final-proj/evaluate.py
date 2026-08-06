# -*- coding: utf-8 -*-
"""Evaluation + visualization pass for the TESS forecasting comparison.

Runs off the saved artifacts, so it is cheap to iterate on metrics without
retraining:

    python evaluate.py                                  # defaults, light theme
    python evaluate.py --theme dark --outdir outputs/dark
    python evaluate.py --preds test_predictions.npz --history outputs/history.json

Writes to --outdir:
    metrics_summary.csv     one row per model, headline metrics
    metrics_by_horizon.csv  MAE/RMSE/skill per forecast step
    metrics_by_depth.csv    MAE and detection AUC per transit-SNR bin
    metrics_report.md       the same numbers with the caveats attached
    *.png                   the figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import eval_charts as charts
import eval_metrics as M


def parse_args():
    p = argparse.ArgumentParser(description="Metrics and figures for the TESS forecasting models.")
    p.add_argument("--preds", default="test_predictions.npz", help="output of the training run")
    p.add_argument("--history", default="outputs/history.json",
                   help="per-epoch losses; skipped if absent")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--theme", default="light", choices=sorted(charts.THEMES))
    p.add_argument("--transit-snr", type=float, default=3.0,
                   help="dip SNR at or above which a window counts as a transit")
    p.add_argument("--quiet-snr", type=float, default=1.5,
                   help="dip SNR at or below which a window counts as quiet")
    p.add_argument("--n-bins", type=int, default=5, help="transit-SNR quantile bins")
    p.add_argument("--n-examples", type=int, default=3, help="qualitative example panels")
    return p.parse_args()


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_history(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt(value, spec=".1f"):
    return "n/a" if value is None or not np.isfinite(value) else format(value, spec)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = M.load_predictions(args.preds)
    models, true = data["models"], data["true"]
    theme = charts.Theme(args.theme, list(models))
    written = []

    # --- window labelling -------------------------------------------------
    sigma = M.window_noise(true)
    snr = M.transit_snr(data["transit_depth"], sigma)
    labels = M.label_windows(snr, args.transit_snr, args.quiet_snr)
    is_transit, is_quiet = labels == "transit", labels == "quiet"

    print(f"{true.shape[0]} test windows, {true.shape[1]}-step horizon, "
          f"models: {', '.join(models)}")
    print(f"transit SNR >= {args.transit_snr}: {is_transit.sum()} | "
          f"quiet SNR <= {args.quiet_snr}: {is_quiet.sum()} | "
          f"ambiguous: {(labels == 'ambiguous').sum()}")
    if is_transit.sum() < 10:
        print("WARNING: too few real transits for the detection metrics to mean much. "
              "This is expected on the 8-star pilot set; rerun after the Zaratan build.")

    # --- headline table ---------------------------------------------------
    overall = M.global_table(models, true)
    transit_tbl = M.global_table(models, true, is_transit) if is_transit.any() else {}
    quiet_tbl = M.global_table(models, true, is_quiet) if is_quiet.any() else {}

    summary_fields = ["model", "N", "MAE", "RMSE", "MedAE", "P90AE", "MaxAE", "Bias",
                      "R2", "ShapeCorr", "Skill_MAE", "Skill_RMSE", "MASE",
                      "MAE_transit", "RMSE_transit", "MAE_quiet",
                      "dip_recovery_slope", "dip_floor_MAE", "detect_AUC", "detect_AP"]

    # --- transit-conditioned analysis ------------------------------------
    recovery, rocs, detect = {}, {}, {}
    if is_transit.any():
        for name, pred in models.items():
            recovery[name] = M.dip_recovery(pred[is_transit], true[is_transit])
    if is_transit.any() and is_quiet.any():
        both = is_transit | is_quiet
        for name, pred in models.items():
            scores = M.detection_scores(pred[both], true[both])
            positive = is_transit[both]
            fpr, tpr, auc = M.roc_curve(scores, positive)
            rocs[name] = (fpr, tpr, auc)
            detect[name] = {"AUC": auc, "AP": M.average_precision(scores, positive)}

    rows = []
    for name in models:
        row = {"model": name}
        row.update({k: overall[name].get(k) for k in
                    ("N", "MAE", "RMSE", "MedAE", "P90AE", "MaxAE", "Bias", "R2",
                     "ShapeCorr", "Skill_MAE", "Skill_RMSE", "MASE")})
        row["MAE_transit"] = transit_tbl.get(name, {}).get("MAE")
        row["RMSE_transit"] = transit_tbl.get(name, {}).get("RMSE")
        row["MAE_quiet"] = quiet_tbl.get(name, {}).get("MAE")
        row["dip_recovery_slope"] = recovery.get(name, {}).get("recovery_slope")
        row["dip_floor_MAE"] = recovery.get(name, {}).get("floor_MAE")
        row["detect_AUC"] = detect.get(name, {}).get("AUC")
        row["detect_AP"] = detect.get(name, {}).get("AP")
        rows.append(row)
    write_csv(outdir / "metrics_summary.csv", rows, summary_fields)
    written.append(outdir / "metrics_summary.csv")

    # --- horizon ----------------------------------------------------------
    curves = M.horizon_curves(models, true)
    horizon_rows = []
    for name, c in curves.items():
        for step in range(len(c["MAE"])):
            horizon_rows.append({
                "model": name, "step": step + 1,
                "MAE": float(c["MAE"][step]), "RMSE": float(c["RMSE"][step]),
                "Skill_RMSE": float(c.get("Skill_RMSE", np.full(len(c["MAE"]), np.nan))[step]),
            })
    write_csv(outdir / "metrics_by_horizon.csv", horizon_rows,
              ["model", "step", "MAE", "RMSE", "Skill_RMSE"])
    written.append(outdir / "metrics_by_horizon.csv")

    # --- depth bins -------------------------------------------------------
    bins = M.snr_bins(snr, np.ones(len(snr), dtype=bool), args.n_bins)
    bin_labels = [label for label, _ in bins]
    bin_masks = [mask for _, mask in bins]
    bin_counts = [int(m.sum()) for m in bin_masks]

    mae_by_bin = {name: [M.global_table({name: models[name]}, true, m)[name]["MAE"]
                         for m in bin_masks] for name in models}

    depth_rows = []
    for i, label in enumerate(bin_labels):
        for name in models:
            depth_rows.append({"snr_bin": label, "n_windows": bin_counts[i],
                               "model": name, "MAE": mae_by_bin[name][i]})
    write_csv(outdir / "metrics_by_depth.csv", depth_rows,
              ["snr_bin", "n_windows", "model", "MAE"])
    written.append(outdir / "metrics_by_depth.csv")

    # Detection is binned over transit windows only. Binning it over all windows
    # puts every transit in the top bin and leaves the rest of the chart empty.
    auc_bin_labels, auc_by_bin, auc_bin_counts = [], {}, []
    if is_transit.sum() >= 6 and is_quiet.sum() >= 3:
        n_auc_bins = max(2, min(args.n_bins, int(is_transit.sum()) // 8))
        auc_bins = M.snr_bins(snr, is_transit, n_auc_bins)
        auc_bin_labels = [label for label, _ in auc_bins]
        auc_bin_counts = [int(m.sum()) for m in (m for _, m in auc_bins)]
        for name, pred in models.items():
            per_bin = []
            for _, pos in auc_bins:
                sel = pos | is_quiet
                if pos.sum() < 3:
                    per_bin.append(np.nan)
                    continue
                _, _, auc = M.roc_curve(M.detection_scores(pred[sel], true[sel]), pos[sel])
                per_bin.append(auc)
            if np.isfinite(per_bin).any():
                auc_by_bin[name] = per_bin

        auc_rows = [{"snr_bin": auc_bin_labels[i], "n_transit": auc_bin_counts[i],
                     "n_quiet": int(is_quiet.sum()), "model": name, "detect_AUC": aucs[i]}
                    for name, aucs in auc_by_bin.items() for i in range(len(auc_bin_labels))]
        write_csv(outdir / "detection_by_depth.csv", auc_rows,
                  ["snr_bin", "n_transit", "n_quiet", "model", "detect_AUC"])
        written.append(outdir / "detection_by_depth.csv")

    # --- figures ----------------------------------------------------------
    written.append(charts.fig_headline(theme, overall, outdir / "metrics_headline.png"))
    written.append(charts.fig_horizon(theme, curves, outdir / "error_by_horizon.png"))
    written.append(charts.fig_error_ecdf(theme, models, true, outdir / "error_distribution.png"))
    written.append(charts.fig_depth_bins(theme, bin_labels, mae_by_bin,
                                         outdir / "error_by_depth.png", bin_counts))
    written.append(charts.fig_residual_heatmap(theme, models, true, bin_labels, bin_masks,
                                               outdir / "residual_heatmap.png"))
    if recovery:
        written.append(charts.fig_dip_recovery(theme, recovery, outdir / "dip_recovery.png"))
    if rocs:
        written.append(charts.fig_detection(theme, rocs, outdir / "detection_roc.png"))
    if auc_by_bin:
        written.append(charts.fig_auc_by_depth(theme, auc_bin_labels, auc_by_bin,
                                               outdir / "detection_auc_by_depth.png",
                                               auc_bin_counts))

    per_star = None
    if data["tic_id"] is not None:
        per_star = M.per_star_mae(models, true, data["tic_id"])
        written.append(charts.fig_per_star(theme, per_star, outdir / "error_by_star.png"))

    deepest = np.argsort(-snr)[: args.n_examples]
    titles = [f"SNR {snr[i]:.1f} · depth {data['transit_depth'][i] * M.FLUX_TO_PPM:.0f} ppm"
              for i in deepest]
    written.append(charts.fig_examples(theme, models, true, data["X"], deepest,
                                       outdir / "examples_forecast.png", titles))

    history = load_history(args.history)
    if history:
        path = charts.fig_training_curves(theme, history, outdir / "training_curves.png")
        if path:
            written.append(path)
    else:
        print(f"no training history at {args.history} — skipping training_curves.png")

    # --- report -----------------------------------------------------------
    report = build_report(args, data, models, overall, transit_tbl, quiet_tbl,
                          recovery, detect, curves, labels, snr, per_star)
    (outdir / "metrics_report.md").write_text(report)
    written.append(outdir / "metrics_report.md")

    print("\nwrote:")
    for path in written:
        print(f"  {path}")


def build_report(args, data, models, overall, transit_tbl, quiet_tbl,
                 recovery, detect, curves, labels, snr, per_star):
    n, horizon = data["true"].shape
    lines = [
        "# Forecasting evaluation",
        "",
        f"Source: `{args.preds}` — {n} test windows, {horizon}-step horizon, "
        f"{len(models)} models.",
        "All errors in ppm of relative flux (flux sits at ~1.0, so 1e-6 = 1 ppm).",
        "",
        "## Window labels",
        "",
        f"`transit_depth` is `median(window) - min(window)`, which is positive for pure "
        f"noise — on this set it flags {int((data['transit_depth'] > 0).sum())}/{n} windows. "
        f"Windows are instead labelled by dip SNR (depth / robust window noise):",
        "",
        f"- transit (SNR >= {args.transit_snr}): **{int((labels == 'transit').sum())}**",
        f"- quiet (SNR <= {args.quiet_snr}): **{int((labels == 'quiet').sum())}**",
        f"- ambiguous (excluded from binary comparisons): "
        f"**{int((labels == 'ambiguous').sum())}**",
        f"- SNR range: {snr.min():.2f} to {snr.max():.2f}",
        "",
        "## Headline",
        "",
        "| model | MAE | RMSE | P90 AE | Bias | R² | shape r | skill vs persistence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in overall.items():
        skill = "—" if name == M.BASELINE else f"{100 * s.get('Skill_RMSE', np.nan):+.1f}%"
        lines.append(
            f"| {name} | {fmt(s['MAE'])} | {fmt(s['RMSE'])} | {fmt(s['P90AE'])} | "
            f"{fmt(s['Bias'], '+.1f')} | {fmt(s['R2'], '.3f')} | "
            f"{fmt(s['ShapeCorr'], '.3f')} | {skill} |"
        )
    lines += ["",
              "`shape r` is the mean per-window correlation between forecast and truth — "
              "whether the model tracks the shape of the next segment at all, which the "
              "error columns do not measure. It is `n/a` for persistence by construction: "
              "a flat forecast has no variance to correlate."]

    if transit_tbl:
        lines += ["", "## Transit vs quiet windows", "",
                  "| model | MAE (transit) | RMSE (transit) | MAE (quiet) | dip depth recovered | dip floor MAE |",
                  "|---|---|---|---|---|---|"]
        for name in models:
            rec = recovery.get(name, {})
            lines.append(
                f"| {name} | {fmt(transit_tbl[name]['MAE'])} | {fmt(transit_tbl[name]['RMSE'])} | "
                f"{fmt(quiet_tbl.get(name, {}).get('MAE', np.nan))} | "
                f"{fmt(100 * rec.get('recovery_slope', np.nan))}% | "
                f"{fmt(rec.get('floor_MAE', np.nan))} |"
            )

    if detect:
        lines += ["", "## Transit detection from forecast residuals", "",
                  "Score is `max(pred - true)` over the window: the largest amount the truth "
                  "fell below the forecast. Positives are transit windows, negatives are quiet "
                  "windows; ambiguous windows are excluded.", "",
                  "| model | ROC AUC | avg precision |", "|---|---|---|"]
        for name, d in detect.items():
            lines.append(f"| {name} | {fmt(d['AUC'], '.3f')} | {fmt(d['AP'], '.3f')} |")

    lines += ["", "## Useful forecast horizon", ""]
    for name, c in curves.items():
        if name == M.BASELINE or "Skill_RMSE" not in c:
            continue
        skill = c["Skill_RMSE"]
        wins = int((skill > 0).sum())
        # count of winning steps, not first-to-last: skill that oscillates around
        # zero would otherwise read as one contiguous winning stretch
        lines.append(
            f"- **{name}** beats persistence at {wins}/{horizon} steps "
            f"(best {100 * np.nanmax(skill):+.1f}% at step {int(np.nanargmax(skill)) + 1}, "
            f"worst {100 * np.nanmin(skill):+.1f}%); "
            f"step 1 MAE {c['MAE'][0]:.1f} ppm, step {horizon} MAE {c['MAE'][-1]:.1f} ppm"
        )

    if per_star:
        lines += ["", "## Per-star spread", ""]
        for name, per_tic in per_star.items():
            values = np.array(list(per_tic.values()))
            worst = max(per_tic, key=per_tic.get)
            lines.append(f"- **{name}**: median TIC MAE {np.median(values):.1f} ppm, "
                         f"worst TIC {worst} at {per_tic[worst]:.1f} ppm "
                         f"({len(per_tic)} test stars)")

    lines += ["", "---", "",
              f"Regenerate with `python evaluate.py --preds {args.preds} "
              f"--outdir {args.outdir} --theme {args.theme}`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
