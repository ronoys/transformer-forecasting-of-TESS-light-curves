# -*- coding: utf-8 -*-
"""Evaluation panel for the tuned run: four cuts the headline skill bar hides.

Everything here is the *test* split of the two winning sweep configs (cfg15
transformer, cfg35 lstm), read from their run_summary/history so the figure
cannot drift from what those jobs actually reported. The headline chart lives
in chart_tuned.py and plots validation skill; nothing is repeated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval_charts import Theme, bar_labels

PROJ = Path(__file__).resolve().parent
BEST = {"transformer": "cfg15", "lstm": "cfg35"}
MODELS = ["lstm", "transformer"]


def load():
    out = {}
    for model, tag in BEST.items():
        d = PROJ / "sweep" / tag / "outputs"
        out[model] = {
            "summary": json.loads((d / "run_summary.json").read_text()),
            "history": json.loads((d / "history.json").read_text())[model],
        }
    return out


def skill(base, got):
    """Fraction of the persistence error removed. Negative = worse than doing nothing."""
    return 100.0 * (1.0 - got / base)


# ------------------------------------------------------------------ panels

def p_skill_by_measure(ax, theme, runs):
    """The LSTM's gain does not survive squaring; the transformer's does.

    Deliberately not split by transit/quiet: run_summary's split is
    `transit_depth > 0`, which flags 97.4% of test windows, so "quiet" is 307
    degenerate windows rather than calm stars. eval_metrics.py's SNR labels are
    the sound version but need saved predictions the sweep did not write.
    """
    measures = [("MAE_all", "typical miss"), ("RMSE_transit", "big misses")]
    x = np.arange(len(measures))
    width = 0.38

    for i, model in enumerate(MODELS):
        t = runs[model]["summary"]["test"]
        vals = [skill(t["persistence"][k], t[model][k]) for k, _ in measures]
        bars = ax.bar(x + (i - 0.5) * width, vals, width * 0.92,
                      color=theme.color(model), label=model,
                      edgecolor=theme.surface, linewidth=1)
        bar_labels(ax, theme, bars, vals, fmt="{:+.1f}%")

    ax.axhline(0, color=theme.muted, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in measures], fontsize=9, color=theme.text)
    ax.set_ylim(0, 42)
    theme.title(ax, "Does the win hold up?",
                "how much better than doing nothing, on unseen stars")
    theme.labels(ax, y="% better than doing nothing")
    theme.legend(ax, loc="upper right", ncol=2)


def p_error_concentration(ax, theme, runs):
    """Almost all of the error sits in a sliver of windows -- hence Huber.

    Persistence only, because its prediction is x[-1] and needs no checkpoint;
    the point is a property of the window set, not of a trained model.
    """
    z = np.load(PROJ / "run_two/data/tess_windows.npz", allow_pickle=True)
    m = z["split"] == "test"
    X, y = z["X"][m], z["y"][m]
    pred = np.repeat(X[:, -1:], y.shape[1], axis=1)

    frac = np.linspace(0, 1, 400)
    for err, label, color in [
        (((pred - y) ** 2).mean(1), "counting big misses", theme.diverging[2]),
        (np.abs(pred - y).mean(1), "counting every miss", theme.color("persistence")),
    ]:
        cum = np.cumsum(np.sort(err)[::-1])
        cum = cum / cum[-1]
        idx = np.clip((frac * (len(cum) - 1)).astype(int), 0, len(cum) - 1)
        ax.plot(100 * frac, 100 * cum[idx], lw=2, color=color, label=label)

    ax.plot([1], [70.5], "o", ms=7, color=theme.diverging[2],
            mec=theme.surface, mew=1.5, zorder=3)
    ax.annotate("the worst 1% of stretches\ncause 70% of the trouble", xy=(1, 70.5),
                xytext=(14, -6), textcoords="offset points", fontsize=8.5,
                color=theme.diverging[2])
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 105)
    theme.title(ax, "A few bad stretches cause most of the error",
                "sorted worst-first — the curve shoots up immediately")
    theme.labels(ax, x="% of the data (worst first)", y="% of all the error")
    theme.legend(ax, loc="lower right")


def p_training(ax, theme, runs):
    """Convergence: the transformer early-stops; the lstm never stops improving."""
    floor = runs["transformer"]["summary"]["val_loss"]["persistence"]

    for model in MODELS:
        val = runs[model]["history"]["val"]
        epochs = np.arange(1, len(val) + 1)
        ax.plot(epochs, val, lw=2, color=theme.color(model), label=model)
        best = int(np.argmin(val))
        ax.plot([best + 1], [val[best]], "o", ms=6, color=theme.color(model),
                mec=theme.surface, mew=1.5, zorder=3)

    ax.axhline(floor, color=theme.color("persistence"), lw=1.5, ls="--", dashes=(4, 3))
    ax.annotate("do-nothing baseline", xy=(0.98, floor), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=theme.color("persistence"))
    theme.title(ax, "The transformer settles quickly",
                "error on unseen stars — the lstm was still improving when time ran out")
    theme.labels(ax, x="epoch (pass over the data)", y="error")
    theme.legend(ax, loc="center right")


def p_efficiency(ax, theme, runs):
    """Accuracy per parameter — the transformer wins on both axes at once."""
    for model in MODELS:
        s = runs[model]["summary"]
        t = s["test"]
        n = s["n_params"][model] / 1000.0
        sk = skill(t["persistence"]["MAE_all"], t[model]["MAE_all"])
        ax.scatter([n], [sk], s=150, color=theme.color(model), zorder=3,
                   edgecolor=theme.surface, linewidth=1.5)
        ax.annotate(f"{model}\n{n:.0f}k settings · {sk:+.1f}%",
                    xy=(n, sk), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=theme.color(model))

    ax.scatter([0], [0], s=110, color=theme.color("persistence"), zorder=3,
               edgecolor=theme.surface, linewidth=1.5)
    ax.annotate("do nothing\nno model at all", xy=(0, 0), xytext=(6, 10),
                textcoords="offset points", fontsize=8.5,
                color=theme.color("persistence"))

    ax.axhline(0, color=theme.muted, lw=1)
    ax.set_xlim(-18, 250)
    ax.set_ylim(-8, 42)
    theme.title(ax, "Smaller model, bigger gain",
                "the transformer wins with half as many moving parts")
    theme.labels(ax, x="model size (thousands of settings)",
                 y="% better than doing nothing")


# ------------------------------------------------------------------ figure

def build(theme_name, out, runs):
    theme = Theme(theme_name, MODELS + ["persistence"])
    fig, axes = theme.figure(2, 2, figsize=(12.5, 8.2))
    p_skill_by_measure(axes[0][0], theme, runs)
    p_error_concentration(axes[0][1], theme, runs)
    p_training(axes[1][0], theme, runs)
    p_efficiency(axes[1][1], theme, runs)
    fig.tight_layout(pad=2.2, w_pad=3.0, h_pad=3.4)
    return theme.save(fig, PROJ / out)


if __name__ == "__main__":
    runs = load()
    for tn, out in [("light", "outputs/evaluation_panel.png"),
                    ("dark", "outputs/evaluation_panel_dark.png")]:
        print(build(tn, out, runs))
