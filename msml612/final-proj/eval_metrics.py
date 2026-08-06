# -*- coding: utf-8 -*-
"""Metric engine for the TESS forecasting comparison.

Everything here is pure numpy on top of the `test_predictions.npz` handoff, so
metrics can be recomputed without a GPU or a retrain. `evaluate.py` is the CLI.

Why SNR and not `transit_depth > 0`: the builder defines depth as
`median(window) - min(window)`, which is positive for pure photon noise. On the
8-star pilot set that flagged 398/400 test windows as "transit". Every
transit-conditioned number here instead thresholds depth against the window's own
robust noise level, so "transit" means a dip that is actually detectable.
"""

from __future__ import annotations

import numpy as np


FLUX_TO_PPM = 1e6  # detrended flux sits at ~1.0, so a delta of 1e-6 is 1 ppm

MODEL_ORDER = ["persistence", "lstm", "transformer"]
BASELINE = "persistence"


# ---------------------------------------------------------------- loading


def load_predictions(path: str) -> dict:
    """Read test_predictions.npz into {models, true, transit_depth, ...}.

    Tolerates the pilot-run file, which has no tic_id/X arrays.
    """
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)

    models = {}
    for key in sorted(k for k in keys if k.startswith("pred_")):
        models[key[len("pred_") :]] = np.asarray(z[key], dtype=np.float64)

    if not models:
        raise ValueError(f"{path} has no pred_* arrays; keys were {sorted(keys)}")

    out = {
        "models": order_models(models),
        "true": np.asarray(z["true"], dtype=np.float64),
        "transit_depth": np.asarray(z["transit_depth"], dtype=np.float64),
        "tic_id": np.asarray(z["tic_id"]) if "tic_id" in keys else None,
        "X": np.asarray(z["X"], dtype=np.float64) if "X" in keys else None,
    }

    n, horizon = out["true"].shape
    for name, pred in out["models"].items():
        if pred.shape != (n, horizon):
            raise ValueError(f"pred_{name} is {pred.shape}, expected {(n, horizon)}")
    return out


def order_models(models: dict) -> dict:
    """Fixed series order so a model keeps its color across every figure."""
    known = [m for m in MODEL_ORDER if m in models]
    return {m: models[m] for m in known + sorted(set(models) - set(MODEL_ORDER))}


# ---------------------------------------------------------------- labelling


def window_noise(true: np.ndarray) -> np.ndarray:
    """Robust per-window sigma from successive differences.

    MAD of diff() is immune to the transit itself (a box dip contributes only at
    its two edges), which a plain std over the window is not.
    """
    diffs = np.abs(np.diff(true, axis=1))
    return 1.4826 * np.median(diffs, axis=1) / np.sqrt(2.0)


def transit_snr(depth: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return depth / np.maximum(sigma, 1e-12)


def label_windows(snr: np.ndarray, transit_snr_min=3.0, quiet_snr_max=1.5) -> np.ndarray:
    """transit / quiet / ambiguous. The gap keeps marginal windows out of the
    binary comparisons instead of arbitrarily assigning them to one side."""
    labels = np.full(len(snr), "ambiguous", dtype="<U9")
    labels[snr >= transit_snr_min] = "transit"
    labels[snr <= quiet_snr_max] = "quiet"
    return labels


# ---------------------------------------------------------------- metrics


def error_stats(pred: np.ndarray, true: np.ndarray) -> dict:
    """Scale-dependent error summary, in ppm of relative flux."""
    if pred.size == 0:
        return {k: float("nan") for k in
                ("MAE", "RMSE", "MedAE", "P90AE", "MaxAE", "Bias", "R2", "ShapeCorr")}

    err = (pred - true) * FLUX_TO_PPM
    abs_err = np.abs(err)
    true_ppm = true * FLUX_TO_PPM

    ss_res = float((err ** 2).sum())
    ss_tot = float(((true_ppm - true_ppm.mean()) ** 2).sum())

    return {
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MedAE": float(np.median(abs_err)),
        "P90AE": float(np.percentile(abs_err, 90)),
        "MaxAE": float(abs_err.max()),
        "Bias": float(err.mean()),
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "ShapeCorr": mean_window_corr(pred, true),
    }


def mean_window_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-window Pearson r between forecast and truth.

    Error metrics reward predicting the flat mean; this asks whether the
    forecast tracks the *shape* of the next 32 points at all.
    """
    p = pred - pred.mean(axis=1, keepdims=True)
    t = true - true.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    good = denom > 0
    if not good.any():
        return float("nan")
    return float(((p[good] * t[good]).sum(axis=1) / denom[good]).mean())


def skill_scores(stats: dict, baseline_stats: dict) -> dict:
    """Fraction of baseline error removed. 0 = no better than persistence,
    1 = perfect, negative = worse than doing nothing."""
    out = {}
    for key in ("MAE", "RMSE"):
        base = baseline_stats.get(key, float("nan"))
        out[f"Skill_{key}"] = 1.0 - stats[key] / base if base and np.isfinite(base) else float("nan")
    out["MASE"] = stats["MAE"] / baseline_stats["MAE"] if baseline_stats["MAE"] else float("nan")
    return out


def global_table(models: dict, true: np.ndarray, mask=None) -> dict:
    """{model: {metric: value}} over the selected windows, skill vs persistence."""
    sel = slice(None) if mask is None else mask
    stats = {name: error_stats(pred[sel], true[sel]) for name, pred in models.items()}
    base = stats.get(BASELINE)
    if base is not None:
        for name in stats:
            stats[name].update(skill_scores(stats[name], base))
    for name in stats:
        stats[name]["N"] = int(true[sel].shape[0])
    return stats


def horizon_curves(models: dict, true: np.ndarray, mask=None) -> dict:
    """Per-forecast-step MAE/RMSE/skill. Answers 'how far ahead is it useful?'"""
    sel = slice(None) if mask is None else mask
    t = true[sel]
    curves = {}
    for name, pred in models.items():
        err = (pred[sel] - t) * FLUX_TO_PPM
        curves[name] = {
            "MAE": np.abs(err).mean(axis=0),
            "RMSE": np.sqrt((err ** 2).mean(axis=0)),
        }
    if BASELINE in curves:
        base = curves[BASELINE]
        for name, c in curves.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                c["Skill_RMSE"] = 1.0 - c["RMSE"] / base["RMSE"]
    return curves


# ---------------------------------------------------------------- transits


def dip_recovery(pred: np.ndarray, true: np.ndarray) -> dict:
    """How much of each real dip the forecast reproduces.

    Depth is measured self-referentially (median minus min of the same series)
    so predicted and true depths are on the same footing.
    """
    depth_true = (np.median(true, axis=1) - true.min(axis=1)) * FLUX_TO_PPM
    depth_pred = (np.median(pred, axis=1) - pred.min(axis=1)) * FLUX_TO_PPM
    floor_err = np.abs(true.min(axis=1) - pred.min(axis=1)) * FLUX_TO_PPM

    denom = float((depth_true ** 2).sum())
    slope = float((depth_true * depth_pred).sum() / denom) if denom > 0 else float("nan")

    return {
        "depth_true": depth_true,
        "depth_pred": depth_pred,
        "floor_err": floor_err,
        "recovery_slope": slope,          # 1.0 = dips reproduced at full depth
        "floor_MAE": float(floor_err.mean()),
    }


def detection_scores(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Anomaly score per window: the largest downward surprise, in ppm.

    A transit the model did not anticipate makes the truth fall below the
    forecast, so max(pred - true) is a detection statistic derived purely from
    forecasting. This is the framing behind the shallow-dip claim.
    """
    return (pred - true).max(axis=1) * FLUX_TO_PPM


def roc_curve(scores: np.ndarray, positive: np.ndarray) -> tuple:
    """(fpr, tpr, auc) via rank statistics; no sklearn dependency on the cluster."""
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / n_neg])
    return fpr, tpr, float(np.trapz(tpr, fpr))


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under precision-recall; the honest number when transits are rare."""
    positive = positive.astype(bool)
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = positive[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)


def snr_bins(snr: np.ndarray, mask: np.ndarray, n_bins: int = 5) -> list:
    """Quantile bins over the selected windows -> [(label, boolean mask), ...].

    Quantiles rather than fixed cuts because depth distributions shift a lot
    between the pilot set and a full Zaratan build.
    """
    values = snr[mask]
    if len(values) < n_bins * 2:
        return [("all", mask)]

    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return [("all", mask)]

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = mask & (snr >= lo) & ((snr < hi) if hi != edges[-1] else (snr <= hi))
        if in_bin.sum() > 0:
            bins.append((f"{lo:.1f}–{hi:.1f}", in_bin))
    return bins


def per_star_mae(models: dict, true: np.ndarray, tic_id: np.ndarray) -> dict:
    """{model: {tic: MAE}} — exposes whether one noisy star drives the average."""
    out = {}
    for name, pred in models.items():
        per_tic = {}
        for tic in np.unique(tic_id):
            m = tic_id == tic
            per_tic[int(tic)] = float(np.abs((pred[m] - true[m]) * FLUX_TO_PPM).mean())
        out[name] = per_tic
    return out
