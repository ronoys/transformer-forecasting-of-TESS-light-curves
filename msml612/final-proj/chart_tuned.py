"""Headline chart for run three: best swept config per model vs persistence.

Reuses eval_charts' Theme so the tuned figure sits in the same visual system as
the run-one figures. Numbers come straight from sweep/results.csv, so the slide
and the sweep table cannot drift apart.

Skill here is collect_sweep's val_skill — fraction of the persistence
*validation loss* removed, not the RMSE skill the run-one headline plots. Both
winning configs train with Huber, so they share a denominator and each other's
scale; the subtitle says so rather than letting the slide imply RMSE.
"""
import csv
from pathlib import Path

from eval_charts import Theme, bar_labels

PROJ = Path(__file__).resolve().parent

# Huber only. val_skill divides by the persistence loss under the *same*
# objective, so an MSE row (denominator 1.17e6) and a Huber row (denominator
# 200.8) are not on one scale -- cfg21's "+52.4%" is not 1.8x cfg15's +29.2%,
# it is a different ruler. Both winning configs are Huber, so fixing the loss
# keeps every bar comparable.
rows = [r for r in csv.DictReader(open(PROJ / "sweep/results.csv"))
        if r["tag"].startswith("cfg") and r["use_huber"] == "True"]

best = {}
for r in rows:
    m = r["model"]
    if m not in best or float(r["val_skill"]) > float(best[m]["val_skill"]):
        best[m] = r

names = ["persistence", "lstm", "transformer"]
skill = [0.0] + [100 * float(best[m]["val_skill"]) for m in ("lstm", "transformer")]
tags = ["baseline", best["lstm"]["tag"], best["transformer"]["tag"]]


def build(theme_name, out):
    theme = Theme(theme_name, names)
    fig, ax = theme.figure(figsize=(7.2, 4.0))
    bars = ax.bar(names, skill, color=[theme.color(n) for n in names], width=0.6)
    bar_labels(ax, theme, bars, skill, fmt="{:+.1f}%")
    ax.axhline(0, color=theme.muted, lw=1)
    theme.title(ax, "Forecast skill vs persistence",
                "best swept config per model — % of baseline validation loss removed")
    theme.labels(ax, y="skill (%)")
    ax.set_ylim(-2, max(skill) * 1.28)
    # name the config each bar came from on a second tick line, so the slide
    # stays traceable to the sweep without overprinting the model names
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n{t}" for n, t in zip(names, tags)],
                       fontsize=9, color=theme.text)
    fig.tight_layout()
    return theme.save(fig, PROJ / out)


def build_ablation(theme_name, out):
    """Why the transformer went from +0% to +29%.

    Transformer x Huber only, so every cell shares a denominator. Each cell
    takes its best config -- the question is what a setting makes reachable,
    not how it does averaged over learning rates.
    """
    cells = {}
    for r in rows:
        if r["model"] != "transformer":
            continue
        key = (r["anchor"] == "True", r["pool"])
        v = 100 * float(r["val_skill"])
        if key not in cells or v > cells[key]:
            cells[key] = v

    order = [(False, "mean"), (True, "mean"), (False, "last"), (True, "last"),
             (False, "meanlast"), (True, "meanlast")]
    vals = [cells[k] for k in order]
    labels = [f"{p}\n{'anchor' if a else 'raw'}" for a, p in order]

    theme = Theme(theme_name, names)
    fig, ax = theme.figure(figsize=(7.2, 4.0))
    # the control is the one bar that failed; red says so without a legend
    colors = [theme.diverging[2] if v < 10 else theme.color("transformer") for v in vals]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.66)
    bar_labels(ax, theme, bars, vals, fmt="{:+.1f}%")
    ax.axhline(0, color=theme.muted, lw=1)
    theme.title(ax, "What the transformer's skill hinged on",
                "best Huber config per pooling x input setting")
    theme.labels(ax, y="skill (%)")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5, color=theme.text)
    ax.set_ylim(0, max(vals) * 1.3)

    lo = vals.index(min(vals))
    ax.annotate("run one's setting", xy=(lo, vals[lo]), xytext=(0, 26),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=theme.diverging[2],
                arrowprops=dict(arrowstyle="-", color=theme.diverging[2], lw=0.9))
    fig.tight_layout()
    return theme.save(fig, PROJ / out)


for tn, out in [("light", "outputs/skill_tuned.png"), ("dark", "outputs/skill_tuned_dark.png")]:
    print(build(tn, out))
for tn, out in [("light", "outputs/ablation_tuned.png"), ("dark", "outputs/ablation_tuned_dark.png")]:
    print(build_ablation(tn, out))
for m in ("lstm", "transformer"):
    b = best[m]
    print(f"{m:12s} {b['tag']}  skill={100*float(b['val_skill']):+.1f}%  "
          f"anchor={b['anchor']} pool={b['pool']} huber={b['use_huber']} lr={b['lr']}")
