# -*- coding: utf-8 -*-
"""Figures for the TESS forecasting comparison.

One theme object drives every figure so the report reads as a single system:
a model keeps its hue everywhere, grids stay recessive, and every panel with
two or more series carries both a legend and direct labels (identity is never
carried by color alone).

Palette: slots 1-3 of the validated categorical set, checked with the dataviz
validator at all-pairs strictness in both modes (worst CVD deltaE 9.2 light /
9.4 dark, worst normal-vision deltaE 24.0 / 20.9). Aqua sits below 3:1 on the
light surface, so direct labels are mandatory rather than optional.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt
import numpy as np

from eval_metrics import BASELINE, FLUX_TO_PPM


THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2dd",
        "series": {"persistence": "#2a78d6", "lstm": "#eb6834", "transformer": "#1baf7a"},
        "diverging": ("#2a78d6", "#eae9e2", "#e34948"),
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a37",
        "series": {"persistence": "#3987e5", "lstm": "#d95926", "transformer": "#199e70"},
        "diverging": ("#3987e5", "#2e2e2b", "#e66767"),
    },
}

FALLBACK_HUES = ["#eda100", "#e87ba4", "#008300", "#4a3aa7"]


class Theme:
    """Colors + the small amount of chart chrome every figure repeats."""

    def __init__(self, name: str, model_names):
        self.name = name
        t = THEMES[name]
        self.surface, self.text, self.muted, self.grid = t["surface"], t["text"], t["muted"], t["grid"]
        self.diverging = t["diverging"]
        self.colors = dict(t["series"])
        # any model beyond the known three takes the next fixed slot, never a
        # generated hue
        spare = list(FALLBACK_HUES)
        for m in model_names:
            if m not in self.colors:
                self.colors[m] = spare.pop(0) if spare else self.muted

    def color(self, model: str) -> str:
        return self.colors.get(model, self.muted)

    def figure(self, *args, **kwargs):
        fig, axes = plt.subplots(*args, **kwargs)
        fig.patch.set_facecolor(self.surface)
        for ax in np.atleast_1d(np.asarray(axes)).ravel():
            self.style_axes(ax)
        return fig, axes

    def style_axes(self, ax):
        ax.set_facecolor(self.surface)
        ax.grid(True, color=self.grid, lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self.grid)
        ax.tick_params(colors=self.muted, labelsize=8, length=0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(self.muted)

    def title(self, ax, text, subtitle=None):
        ax.set_title(text, color=self.text, fontsize=11, loc="left", pad=14 if subtitle else 8)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=self.muted,
                    fontsize=8.5, va="bottom")

    def labels(self, ax, x=None, y=None):
        if x:
            ax.set_xlabel(x, color=self.muted, fontsize=9)
        if y:
            ax.set_ylabel(y, color=self.muted, fontsize=9)

    def legend(self, ax, **kwargs):
        leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=self.muted, **kwargs)
        return leg

    def save(self, fig, path):
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=self.surface)
        plt.close(fig)
        return path


def end_label(ax, theme, x, y, text, color):
    """Direct label at the end of a line. Text stays in ink; the line carries hue."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8.5, va="center", fontweight="medium")


def end_labels(ax, theme, x, entries):
    """Direct-label several lines at once, nudged apart so they stay readable.

    Models that agree closely — persistence and a model that learned nothing,
    for instance — would otherwise print on top of each other and read as one
    label. Draws a leader tick when a label had to move off its line.
    """
    entries = sorted(entries, key=lambda e: e[0])
    lo, hi = ax.get_ylim()
    gap = abs(hi - lo) * 0.055

    placed = []
    for y, text, color in entries:
        if placed and y - placed[-1][0] < gap:
            y = placed[-1][0] + gap
        placed.append((y, text, color))

    # re-center the stack so nudging does not push labels out of the axes
    shift = max(0.0, placed[-1][0] - (hi - gap * 0.4))
    for (y, text, color), (y0, _, _) in zip(placed, entries):
        y -= shift
        ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=8.5, va="center", fontweight="medium")
        if abs(y - y0) > gap * 0.25:
            ax.plot([x, x], [y0, y], lw=0.8, color=color, alpha=0.5, zorder=1)


def bar_labels(ax, theme, bars, values, fmt="{:.0f}"):
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        ax.annotate(fmt.format(value), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=theme.text)


# ---------------------------------------------------------------- figures


def fig_headline(theme, table, out):
    """Two panels: absolute test error, and error removed vs persistence."""
    names = list(table)
    fig, (ax1, ax2) = theme.figure(1, 2, figsize=(9.5, 3.6))

    rmse = [table[n]["RMSE"] for n in names]
    bars = ax1.bar(names, rmse, color=[theme.color(n) for n in names], width=0.6)
    bar_labels(ax1, theme, bars, rmse)
    theme.title(ax1, "Test RMSE", "ppm of relative flux — lower is better")
    theme.labels(ax1, y="RMSE (ppm)")
    ax1.set_ylim(0, max(rmse) * 1.18)

    skill = [100 * table[n].get("Skill_RMSE", np.nan) for n in names]
    skill = [0.0 if not np.isfinite(s) else s for s in skill]
    bars = ax2.bar(names, skill, color=[theme.color(n) for n in names], width=0.6)
    bar_labels(ax2, theme, bars, skill, fmt="{:+.1f}%")
    ax2.axhline(0, color=theme.muted, lw=1)
    theme.title(ax2, "Forecast skill vs persistence", "% of baseline RMSE removed — 0 = no gain")
    theme.labels(ax2, y="skill (%)")
    pad = max(abs(min(skill)), abs(max(skill)), 1.0) * 0.35
    ax2.set_ylim(min(0, min(skill)) - pad, max(0, max(skill)) + pad)

    fig.tight_layout()
    return theme.save(fig, out)


def fig_horizon(theme, curves, out):
    """Error growth with lead time, and where skill survives."""
    names = list(curves)
    steps = np.arange(1, len(curves[names[0]]["MAE"]) + 1)
    fig, (ax1, ax2) = theme.figure(1, 2, figsize=(10, 3.8))

    for name in names:
        ax1.plot(steps, curves[name]["MAE"], lw=2, color=theme.color(name), label=name)
    theme.title(ax1, "Error by forecast step", f"MAE at each of the {len(steps)} predicted points")
    theme.labels(ax1, x="steps ahead", y="MAE (ppm)")
    ax1.set_xlim(1, steps[-1] * 1.22)
    end_labels(ax1, theme, steps[-1],
               [(curves[n]["MAE"][-1], n, theme.color(n)) for n in names])
    theme.legend(ax1, loc="upper left")

    learned = [n for n in names if n != BASELINE and "Skill_RMSE" in curves[n]]
    for name in learned:
        skill = 100 * curves[name]["Skill_RMSE"]
        ax2.plot(steps, skill, lw=2, color=theme.color(name), label=name)
    ax2.axhline(0, color=theme.muted, lw=1, ls="--", dashes=(4, 3))
    ax2.annotate("persistence", xy=(1, 0), xytext=(2, 4), textcoords="offset points",
                 color=theme.muted, fontsize=8)
    theme.title(ax2, "Skill by forecast step", "above 0 = beating persistence at that lead time")
    theme.labels(ax2, x="steps ahead", y="skill (%)")
    ax2.set_xlim(1, steps[-1] * 1.22)
    end_labels(ax2, theme, steps[-1],
               [(100 * curves[n]["Skill_RMSE"][-1], n, theme.color(n)) for n in learned])
    theme.legend(ax2, loc="upper right")

    fig.tight_layout()
    return theme.save(fig, out)


def fig_error_ecdf(theme, models, true, out):
    """Full error distribution, not just its mean.

    ECDF rather than overlaid histograms: no bin choice, and the tail — where a
    missed transit lives — is readable.
    """
    fig, ax = theme.figure(figsize=(6.4, 3.8))
    for name, pred in models.items():
        err = np.sort(np.abs((pred - true) * FLUX_TO_PPM).ravel())
        y = 100 * np.arange(1, len(err) + 1) / len(err)
        c = theme.color(name)
        p90 = float(np.percentile(err, 90))
        # the p90 value rides in the legend: all three curves cross y=90 within a
        # few hundred ppm of each other, so on-plot labels would overprint
        ax.plot(err, y, lw=2, color=c, label=f"{name} · p90 {p90:,.0f} ppm")
        ax.plot([p90], [90], "o", ms=7, color=c, mec=theme.surface, mew=2, zorder=3)
    ax.axhline(90, color=theme.grid, lw=1, ls="--", dashes=(4, 3), zorder=1)
    ax.annotate("p90", xy=(1, 90), xytext=(-3, 4), textcoords="offset points",
                xycoords=("axes fraction", "data"), color=theme.muted, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_ylim(0, 103)
    theme.title(ax, "Absolute error distribution",
                "every predicted point, test split — left and steep is better")
    theme.labels(ax, x="absolute error (ppm, log scale)", y="% of points below")
    theme.legend(ax, loc="upper left")
    fig.tight_layout()
    return theme.save(fig, out)


def fig_depth_bins(theme, bin_labels, per_bin, out, counts=None):
    """MAE by transit-SNR bin — does accuracy hold up where the dips are?"""
    names = list(per_bin)
    x = np.arange(len(bin_labels))
    width = 0.8 / max(len(names), 1)
    fig, ax = theme.figure(figsize=(max(6.5, 1.5 * len(bin_labels)), 3.8))

    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        # 2px surface gap between adjacent bars
        bars = ax.bar(x + offset, per_bin[name], width=width * 0.9, color=theme.color(name),
                      label=name, edgecolor=theme.surface, linewidth=1)
        if len(bin_labels) <= 6:
            bar_labels(ax, theme, bars, per_bin[name])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nn={counts[i]}" if counts else lab
                        for i, lab in enumerate(bin_labels)], fontsize=8, color=theme.muted)
    theme.title(ax, "Error by transit depth", "test windows binned by dip SNR (depth / window noise)")
    theme.labels(ax, x="transit SNR", y="MAE (ppm)")
    theme.legend(ax, loc="upper left", ncol=len(names))
    fig.tight_layout()
    return theme.save(fig, out)


def fig_dip_recovery(theme, recovery, out):
    """Predicted vs true dip depth, one panel per model, shared scales.

    Small multiples instead of one overplotted axes: with three overlapping
    point clouds the 1:1 comparison is the thing you cannot read.
    """
    names = list(recovery)
    fig, axes = theme.figure(1, len(names), figsize=(3.5 * len(names), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    all_true = np.concatenate([recovery[n]["depth_true"] for n in names])
    all_pred = np.concatenate([recovery[n]["depth_pred"] for n in names])
    x_hi = float(np.percentile(all_true, 99.5)) or 1.0
    # y is scaled to the predictions, not to the 1:1 line. Forecasts recover only a
    # few percent of each dip, so a shared square scale flattens the whole cloud onto
    # the axis and hides what structure there is; the 1:1 line still anchors the
    # comparison by leaving the panel almost immediately.
    y_hi = float(np.percentile(all_pred, 99.5)) or 1.0
    y_hi = max(y_hi * 1.15, x_hi * 0.02)

    for ax, name in zip(axes, names):
        r = recovery[name]
        ax.plot([0, x_hi], [0, x_hi], lw=1.5, ls="--", dashes=(4, 3), color=theme.muted, zorder=1)
        ax.scatter(r["depth_true"], r["depth_pred"], s=14, alpha=0.45,
                   color=theme.color(name), linewidths=0, zorder=2)
        slope = r["recovery_slope"]
        ax.plot([0, x_hi], [0, slope * x_hi], lw=2, color=theme.color(name), zorder=3)
        theme.title(ax, name, f"depth recovered: {100 * slope:.1f}% of true")
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, y_hi)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        theme.labels(ax, x="true dip depth (ppm)")

    axes[0].annotate("1:1 (leaves panel here)", xy=(y_hi, y_hi), xytext=(6, -2),
                     textcoords="offset points", color=theme.muted, fontsize=8)
    theme.labels(axes[0], y="predicted dip depth (ppm)")
    fig.suptitle("Dip depth recovery on transit windows", color=theme.text, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return theme.save(fig, out)


def fig_detection(theme, rocs, out):
    """Transit detection from forecast residuals: the novelty claim, as a curve."""
    fig, ax = theme.figure(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], lw=1.5, ls="--", dashes=(4, 3), color=theme.muted)
    ax.annotate("chance", xy=(0.62, 0.58), color=theme.muted, fontsize=8.5, rotation=38)

    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, lw=2, color=theme.color(name), label=f"{name}  AUC {auc:.3f}")
    theme.title(ax, "Transit detection from forecast residuals",
                "score = largest downward surprise, max(pred - true)")
    theme.labels(ax, x="false positive rate", y="true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    theme.legend(ax, loc="lower right")
    fig.tight_layout()
    return theme.save(fig, out)


def fig_auc_by_depth(theme, bin_labels, per_bin, out, counts=None):
    """The shallow-dip question: does detection hold as depth shrinks?"""
    x = np.arange(len(bin_labels))
    fig, ax = theme.figure(figsize=(max(6, 1.4 * len(bin_labels)), 3.8))

    for name, aucs in per_bin.items():
        ax.plot(x, aucs, lw=2, color=theme.color(name), marker="o", ms=6,
                mec=theme.surface, mew=1.5, label=name)
    ax.set_ylim(0, 1.05)
    end_labels(ax, theme, x[-1],
               [(per_bin[n][-1], n, theme.color(n)) for n in per_bin
                if np.isfinite(per_bin[n][-1])])
    ax.axhline(0.5, color=theme.muted, lw=1, ls="--", dashes=(4, 3))
    ax.annotate("chance", xy=(0, 0.5), xytext=(2, 3), textcoords="offset points",
                color=theme.muted, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nn={counts[i]}" if counts else lab
                        for i, lab in enumerate(bin_labels)], fontsize=8, color=theme.muted)
    ax.set_xlim(-0.3, len(bin_labels) - 0.15)
    theme.title(ax, "Detection AUC by transit depth",
                "transit windows binned by their own SNR — shallowest on the left")
    theme.labels(ax, x="transit SNR bin", y="ROC AUC")
    theme.legend(ax, loc="lower right")
    fig.tight_layout()
    return theme.save(fig, out)


def fig_per_star(theme, per_star, out, top_n=25):
    """Per-TIC MAE — shows whether the headline average is one bad star."""
    names = list(per_star)
    tics = sorted(per_star[names[0]], key=lambda t: per_star[names[0]][t], reverse=True)[:top_n]
    x = np.arange(len(tics))
    width = 0.8 / len(names)
    fig, ax = theme.figure(figsize=(max(6.5, 0.42 * len(tics) + 2), 3.9))

    # with only a handful of test stars, full-width bars read as blocks
    group = min(0.8, 0.22 * len(names) * max(1, min(len(tics), 4)) / 2)
    width = group / len(names)
    peak = 0.0
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        values = [per_star[name][t] for t in tics]
        peak = max(peak, max(values))
        ax.bar(x + offset, values, width=width * 0.9, color=theme.color(name),
               label=name, edgecolor=theme.surface, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tics], rotation=90 if len(tics) > 8 else 0,
                       fontsize=7, color=theme.muted)
    ax.set_ylim(0, peak * 1.24)  # headroom so the legend never sits on a bar
    theme.title(ax, "Error by star", f"{len(tics)} noisiest test TICs, worst first")
    theme.labels(ax, x="TIC", y="MAE (ppm)")
    theme.legend(ax, loc="upper right", ncol=len(names))
    fig.tight_layout()
    return theme.save(fig, out)


def fig_residual_heatmap(theme, models, true, bin_labels, bin_masks, out):
    """Signed residual by (depth bin x forecast step).

    Diverging ramp with a neutral midpoint: the sign is the point — a model that
    consistently over-predicts flux during transits is missing dips, and that
    reads as one-sided color.
    """
    names = list(models)
    steps = true.shape[1]
    lo, mid, hi = theme.diverging
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("resid", [lo, mid, hi])

    grids = {}
    for name, pred in models.items():
        grid = np.full((len(bin_masks), steps), np.nan)
        for i, m in enumerate(bin_masks):
            if m.sum():
                grid[i] = ((pred[m] - true[m]) * FLUX_TO_PPM).mean(axis=0)
        grids[name] = grid

    vmax = float(np.nanmax(np.abs(np.concatenate(list(grids.values()))))) or 1.0
    fig, axes = theme.figure(len(names), 1, figsize=(7.5, 1.5 * len(names) + 1.4), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, names):
        im = ax.imshow(grids[name], aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                       extent=(0.5, steps + 0.5, len(bin_masks) - 0.5, -0.5))
        ax.grid(False)
        ax.set_yticks(range(len(bin_labels)))
        ax.set_yticklabels(bin_labels, fontsize=7.5, color=theme.muted)
        theme.title(ax, name)
        theme.labels(ax, y="SNR bin")

    theme.labels(axes[-1], x="steps ahead")
    cbar = fig.colorbar(im, ax=axes.tolist(), pad=0.02, fraction=0.03)
    cbar.set_label("mean signed residual, pred - true (ppm)", color=theme.muted, fontsize=8.5)
    cbar.ax.tick_params(colors=theme.muted, labelsize=7.5, length=0)
    cbar.outline.set_visible(False)
    fig.suptitle("Where each model's forecast is biased", color=theme.text, fontsize=11,
                 x=0.01, ha="left")
    return theme.save(fig, out)


def fig_examples(theme, models, true, X, indices, out, row_titles=None):
    """Qualitative panel: context, truth, and every model on the same axes."""
    fig, axes = theme.figure(1, len(indices), figsize=(3.6 * len(indices), 3.2))
    axes = np.atleast_1d(axes)
    L = X.shape[1] if X is not None else 0
    H = true.shape[1]
    future = np.arange(L, L + H)

    for k, (ax, idx) in enumerate(zip(axes, indices)):
        if X is not None:
            ax.plot(np.arange(L), X[idx], lw=1, color=theme.muted, alpha=0.55,
                    label="context", zorder=2)
            ax.axvline(L - 0.5, color=theme.grid, lw=1.5)
            ax.set_xlim(L - min(L, 64), L + H)
        else:
            # no context saved (pilot-run npz): show the forecast window alone
            # rather than two thirds of empty axes
            ax.set_xlim(L, L + H - 1)
        ax.plot(future, true[idx], lw=2.4, color=theme.text, label="truth", zorder=4)
        for name, pred in models.items():
            ax.plot(future, pred[idx], lw=2, ls="--", dashes=(5, 3),
                    color=theme.color(name), label=name, zorder=3)
        theme.title(ax, row_titles[k] if row_titles else f"window {idx}")
        theme.labels(ax, x="steps ahead" if X is None else "time step")

    theme.labels(axes[0], y="relative flux")
    # figure-level legend: an in-axes box would sit on top of the light curve in at
    # least one panel whatever corner it takes
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8.5, labelcolor=theme.muted,
               ncol=len(labels_), loc="upper right", bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Forecasts on the deepest test dips", color=theme.text, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return theme.save(fig, out)


def fig_training_curves(theme, history, out):
    """Train/val loss per epoch, one panel per trained model."""
    names = [n for n in history if history[n].get("val")]
    if not names:
        return None
    fig, axes = theme.figure(1, len(names), figsize=(4.2 * len(names), 3.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, names):
        c = theme.color(name)
        h = history[name]
        epochs = np.arange(1, len(h["val"]) + 1)
        if h.get("train"):
            ax.plot(epochs, h["train"], lw=2, color=c, alpha=0.45, label="train")
        ax.plot(epochs, h["val"], lw=2, color=c, label="val")
        best = int(np.argmin(h["val"]))
        ax.plot([epochs[best]], [h["val"][best]], "o", ms=7, color=c,
                mec=theme.surface, mew=2, zorder=3)
        ax.annotate(f"best e{epochs[best]}", xy=(epochs[best], h["val"][best]),
                    xytext=(6, 8), textcoords="offset points", color=theme.text, fontsize=8.5)
        # log only when the loss actually spans a decade; on a narrow range it just
        # produces unreadable 7.8x10^0 tick labels
        values = np.concatenate([h["val"], h.get("train", h["val"])])
        decades = np.log10(max(values.max(), 1e-12) / max(values.min(), 1e-12))
        ax.set_yscale("log" if decades > 1 else "linear")
        theme.title(ax, name, f"loss in model units{', log scale' if decades > 1 else ''}")
        theme.labels(ax, x="epoch")
        theme.legend(ax, loc="upper right")

    theme.labels(axes[0], y="loss")
    fig.tight_layout()
    return theme.save(fig, out)
