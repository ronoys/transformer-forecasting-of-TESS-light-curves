# -*- coding: utf-8 -*-
"""Optional quality cut on a handoff window set.

    python filter_windows.py --in run_two/data/tess_windows.npz \
                             --out run_two/data/tess_windows_clean.npz

Writes a NEW file and never modifies the input, so the unfiltered run stays
reproducible and the report can quote both.

Why this exists: on run_two, 18.5% of windows carry ~100% of the total squared
error, because `run_two_handoff.py` samples random TICs per sector and so picks up
intrinsically variable stars (pulsators, eclipsing binaries, flares) alongside a
few corrupted targets. Detrended flux is supposed to sit at ~1.0 with deltas of
~1e-3; these windows swing by 100% or more.

This is a scientific choice, not a bug fix. Cutting variable stars makes the
numbers comparable to run one's hand-picked planet hosts and focuses the model on
transit-shaped structure — but it also removes real astrophysical signal. Report
which set each number came from.
"""

from __future__ import annotations

import argparse

import numpy as np

KEYS = ["X", "y", "tic_id", "split", "transit_depth"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--max-range", type=float, default=0.05,
                   help="drop windows whose full flux range exceeds this (default 0.05 = 5%%)")
    p.add_argument("--max-star-badfrac", type=float, default=0.5,
                   help="drop a star entirely once this fraction of its windows are bad")
    p.add_argument("--dry-run", action="store_true", help="report the cut, write nothing")
    return p.parse_args()


def main():
    args = parse_args()
    z = np.load(args.src, allow_pickle=True)
    d = {k: z[k] for k in KEYS}
    extra = {k: z[k] for k in z.files if k not in KEYS}  # e.g. run_two's `sector`

    X, y, tic = d["X"], d["y"], d["tic_id"]

    # A window is bad if either half swings more than max_range, or is non-finite.
    span = np.maximum(X.max(1) - X.min(1), y.max(1) - y.min(1))
    finite = np.isfinite(X).all(1) & np.isfinite(y).all(1)
    bad = (span > args.max_range) | ~finite

    # Drop whole stars that are mostly bad: a variable star's few quiet windows are
    # still that star, and split-by-star means keeping them leaks its behaviour.
    stars = np.unique(tic)
    badfrac = {s: bad[tic == s].mean() for s in stars}
    dead = {s for s in stars if badfrac[s] > args.max_star_badfrac}
    keep = ~bad & ~np.isin(tic, list(dead)) if dead else ~bad

    print(f"input : {len(X):,} windows, {len(stars):,} stars")
    print(f"  windows failing range>{args.max_range} or non-finite: {bad.sum():,} "
          f"({bad.mean():.1%})")
    print(f"  stars dropped entirely (>{args.max_star_badfrac:.0%} bad): {len(dead):,}")
    print(f"output: {keep.sum():,} windows, {len(np.unique(tic[keep])):,} stars "
          f"({keep.mean():.1%} kept)")

    out = {k: v[keep] for k, v in d.items()}
    out.update({k: v[keep] for k, v in extra.items() if len(v) == len(X)})

    counts = {s: int((out["split"] == s).sum()) for s in ("train", "val", "test")}
    print(f"  splits: {counts}")
    empty = [s for s, n in counts.items() if n == 0]
    if empty:
        raise SystemExit(f"ERROR: the cut emptied the {empty} split(s); loosen --max-range")

    span_out = np.maximum(out["X"].max(1) - out["X"].min(1),
                          out["y"].max(1) - out["y"].min(1))
    print(f"  window range after cut: median={np.median(span_out):.4g} "
          f"p99={np.percentile(span_out, 99):.4g} max={span_out.max():.4g}")

    if args.dry_run:
        print("dry run: nothing written")
        return
    np.savez_compressed(args.dst, **out)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
