# -*- coding: utf-8 -*-
"""Training panel: how the sweep actually behaved, not how it was configured.

Four cuts, all from sweep/results.csv and the per-config history/summary files:
convergence, the epoch budget, what the search was and was not sensitive to,
and the target scaling that makes the gradients usable in the first place.

Skill comparisons stay inside the Huber subset. val_skill divides by the
persistence loss under the same objective, so Huber (floor 200.8) and MSE
(floor 1.17e6) rows are not on one scale.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from eval_charts import Theme, bar_labels

PROJ = Path(__file__).resolve().parent
BEST = {"transformer": "cfg15", "lstm": "cfg35"}
MODELS = ["lstm", "transformer"]
CAP = 30  # MAX_EPOCHS in sweep.sh


def load_rows():
    return [r for r in csv.DictReader(open(PROJ / "sweep/results.csv"))
            if r["tag"].startswith("cfg")]


def load_runs():
    out = {}
    for model, tag in BEST.items():
        d = PROJ / "sweep" / tag / "outputs"
        out[model] = {
            "summary": json.loads((d / "run_summary.json").read_text()),
            "history": json.loads((d / "history.json").read_text())[model],
        }
    return out


# ------------------------------------------------------------------ panels

def p_curves(ax, theme, runs, rows):
    """Both winners are Huber, so they share an axis and a persistence floor."""
    floor = runs["transformer"]["summary"]["val_loss"]["persistence"]

    for model in MODELS:
        h = runs[model]["history"]
        ep = np.arange(1, len(h["val"]) + 1)
        c = theme.color(model)
        ax.plot(ep, h["train"], lw=1.4, ls="--", dashes=(3, 2), color=c, alpha=0.75)
        ax.plot(ep, h["val"], lw=2.2, color=c, label=model)
        b = int(np.argmin(h["val"]))
        ax.plot([b + 1], [h["val"][b]], "o", ms=6.5, color=c,
                mec=theme.surface, mew=1.5, zorder=3)

    ax.axhline(floor, color=theme.color("persistence"), lw=1.5, ls="--", dashes=(4, 3))
    ax.annotate("do-nothing baseline", xy=(0.99, floor), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=theme.color("persistence"))
    ax.annotate("dashed = practice data\nsolid = unseen stars\ndot = best epoch", xy=(0.03, 0.40),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=8, color=theme.muted)
    theme.title(ax, "Both models learn, neither memorizes",
                "error on unseen stars, per epoch — lower is better")
    theme.labels(ax, x="epoch (pass over the data)", y="error")
    theme.legend(ax, loc="lower right")


def p_budget(ax, theme, rows):
    """Every LSTM run is budget-limited; its number is a floor, not a ceiling."""
    for i, model in enumerate(MODELS):
        ep = sorted(int(r["epochs"]) for r in rows if r["model"] == model)
        jitter = np.linspace(-0.17, 0.17, len(ep))
        ax.scatter(np.full(len(ep), i) + jitter, ep, s=48, color=theme.color(model),
                   edgecolor=theme.surface, linewidth=1, zorder=3, alpha=0.95)
        med = int(np.median(ep))
        if med < CAP:  # the lstm median is the cap itself; the cap note says so
            ax.plot([i - 0.3, i + 0.3], [med, med], lw=2, color=theme.color(model), zorder=2)
            ax.annotate(f"median {med}", xy=(i + 0.32, med), fontsize=8.5,
                        va="center", color=theme.color(model))

    ax.axhline(CAP, color=theme.diverging[2], lw=1.5, ls="--", dashes=(4, 3))
    ax.annotate(f"our {CAP}-epoch limit", xy=(0.02, CAP), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points", ha="left", va="bottom",
                fontsize=8.5, color=theme.diverging[2])
    ax.annotate("every lstm run\nran out of time", xy=(0, CAP), xytext=(0, -34),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.5, color=theme.diverging[2])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=9.5, color=theme.text)
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylim(0, 34)
    theme.title(ax, "The transformer finishes early",
                "training stops automatically once a model stops improving")
    theme.labels(ax, y="epochs before stopping")


def p_sensitivity(ax, theme, rows):
    """Longer training never rescued a bad representation, and LR barely mattered."""
    sub = [r for r in rows if r["model"] == "transformer" and r["use_huber"] == "True"]
    pts = [(int(r["epochs"]), 100 * float(r["val_skill"])) for r in sub]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=62,
               color=theme.color("transformer"), edgecolor=theme.surface,
               linewidth=1, zorder=3)

    dead = [(int(r["epochs"]), 100 * float(r["val_skill"]))
            for r in sub if float(r["val_skill"]) < 0.1]
    ax.scatter([p[0] for p in dead], [p[1] for p in dead], s=170, facecolors="none",
               edgecolors=theme.diverging[2], linewidth=1.8, zorder=4)
    ax.annotate("these two trained the longest\nand still learned nothing",
                xy=(24, 0.5), xytext=(-6, 26), textcoords="offset points",
                ha="right", fontsize=8.5, color=theme.diverging[2])

    ax.set_ylim(-4, 40)
    theme.title(ax, "Training longer did not help",
                "each dot is one transformer run")
    theme.labels(ax, x="epochs before stopping", y="% better than doing nothing")


def p_scaling(ax, theme):
    """Why training happens in (flux - 1) x 1000 units."""
    z = np.load(PROJ / "run_two/data/tess_windows.npz", allow_pickle=True)
    m = z["split"] == "train"
    X, y = z["X"][m], z["y"][m]
    d = np.abs(y - X[:, -1:]).ravel()
    d = d[np.isfinite(d) & (d > 0)]

    bins = np.logspace(-6, 1, 70)
    ax.hist(d, bins=bins, color=theme.color("transformer"), alpha=0.85,
            edgecolor="none")
    ax.set_xscale("log")

    med = float(np.median(d))
    ax.axvline(med, color=theme.diverging[2], lw=1.6, ls="--", dashes=(4, 3))
    ax.annotate("typical move is ~0.2%\nso we scale it up to train",
                xy=(med, 0.82), xycoords=("data", "axes fraction"),
                xytext=(9, 0), textcoords="offset points",
                fontsize=8.5, color=theme.diverging[2])
    theme.title(ax, "The signal is tiny",
                "how far the star's brightness moves in one step")
    theme.labels(ax, x="change in brightness (log scale)", y="number of points")


# ------------------------------------------------------------------ figure

def build(theme_name, out, runs, rows):
    theme = Theme(theme_name, MODELS + ["persistence"])
    fig, axes = theme.figure(2, 2, figsize=(12.5, 8.2))
    p_curves(axes[0][0], theme, runs, rows)
    p_budget(axes[0][1], theme, rows)
    p_sensitivity(axes[1][0], theme, rows)
    p_scaling(axes[1][1], theme)
    fig.tight_layout(pad=2.2, w_pad=3.0, h_pad=3.4)
    return theme.save(fig, PROJ / out)


if __name__ == "__main__":
    runs, rows = load_runs(), load_rows()
    for tn, out in [("light", "outputs/training_panel.png"),
                    ("dark", "outputs/training_panel_dark.png")]:
        print(build(tn, out, runs, rows))
