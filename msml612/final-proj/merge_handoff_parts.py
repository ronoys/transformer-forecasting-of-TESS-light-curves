"""Merge Slurm-array TESS handoff shards into data/tess_windows.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zaratan_handoff import sanity_check, split_by_tic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", default="data/parts")
    parser.add_argument("--out", default="data/tess_windows.npz")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--target-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=612)
    return parser.parse_args()


def main():
    args = parse_args()
    parts = sorted(Path(args.parts_dir).glob("tess_windows_part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"No part files found in {args.parts_dir}")

    arrays = {"X": [], "y": [], "tic_id": [], "transit_depth": []}
    for part in parts:
        z = np.load(part, allow_pickle=True)
        if z["X"].shape[0] == 0:
            continue
        for key in arrays:
            arrays[key].append(z[key])

    if not arrays["X"]:
        raise RuntimeError("Part files existed, but none contained windows.")

    data = {
        "X": np.concatenate(arrays["X"]).astype(np.float32),
        "y": np.concatenate(arrays["y"]).astype(np.float32),
        "tic_id": np.concatenate(arrays["tic_id"]).astype(np.int64),
        "transit_depth": np.concatenate(arrays["transit_depth"]).astype(np.float32),
    }
    data["split"] = split_by_tic(data["tic_id"], args.seed)
    sanity_check(data, args.input_len, args.target_len)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"Merged {len(parts)} part files into {out}")


if __name__ == "__main__":
    main()
