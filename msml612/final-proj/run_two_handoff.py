"""Build run_two: 10 TICs for each TESS sector 25 through 107.

Run this on the Zaratan login node, not as an sbatch compute-node job. The
download step needs outbound access to MAST, which was unreliable from compute
nodes during run_one.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from zaratan_handoff import (
    Timeout,
    clean_flux,
    load_lightcurve,
    make_windows,
    sanity_check,
    split_by_tic,
)


def sector_tics_from_mast(sector: int, n_tics: int, seed: int) -> list[int]:
    """Return up to n_tics TIC IDs with TESS time-series products in a sector."""
    from astroquery.mast import Observations

    table = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,
    )
    names = []
    for row in table:
        value = str(row.get("target_name", "")).strip()
        if value.upper().startswith("TIC"):
            value = value.split()[-1]
        try:
            tic_id = int(value)
        except ValueError:
            continue
        names.append(tic_id)

    unique = np.array(sorted(set(names)), dtype=np.int64)
    if unique.size == 0:
        return []

    rng = np.random.default_rng(seed + sector)
    rng.shuffle(unique)
    return [int(x) for x in unique[:n_tics]]


def build_run_two(args):
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "lightcurve_manifest.csv"

    X_all, y_all, tic_all, depth_all, sector_all = [], [], [], [], []
    manifest_rows = []
    rng = np.random.default_rng(args.seed)

    for sector in range(args.start_sector, args.end_sector + 1):
        print(f"\n=== Sector {sector} ===", flush=True)
        try:
            tic_ids = sector_tics_from_mast(sector, args.tics_per_sector, args.seed)
        except Exception as exc:
            print(f"  skipped sector lookup: {exc}", flush=True)
            continue

        print(f"  selected {len(tic_ids)} TICs: {tic_ids}", flush=True)
        for tic_id in tic_ids:
            try:
                with Timeout(args.per_tic_timeout):
                    lcc = load_lightcurve(
                        tic_id,
                        max_products=1,
                        author=args.author,
                        sector=sector,
                    )
                    if lcc is None or len(lcc) == 0:
                        print(f"  TIC {tic_id}: no light curve", flush=True)
                        continue
                    flux = clean_flux(lcc)
                    X, y, tic, depth = make_windows(
                        flux,
                        tic_id=tic_id,
                        input_len=args.input_len,
                        target_len=args.target_len,
                        stride=args.stride,
                    )

                if len(X) > args.max_windows_per_lightcurve:
                    keep = np.sort(rng.choice(len(X), args.max_windows_per_lightcurve, replace=False))
                    X = [X[i] for i in keep]
                    y = [y[i] for i in keep]
                    tic = [tic[i] for i in keep]
                    depth = [depth[i] for i in keep]

                print(f"  TIC {tic_id}: kept {len(X)} windows", flush=True)
                X_all.extend(X)
                y_all.extend(y)
                tic_all.extend(tic)
                depth_all.extend(depth)
                sector_all.extend([sector] * len(X))
                manifest_rows.append(
                    {
                        "sector": sector,
                        "tic_id": tic_id,
                        "windows": len(X),
                        "status": "kept",
                    }
                )
            except Exception as exc:
                print(f"  TIC {tic_id}: skipped {exc}", flush=True)
                manifest_rows.append(
                    {
                        "sector": sector,
                        "tic_id": tic_id,
                        "windows": 0,
                        "status": f"skipped: {exc}",
                    }
                )

    if not X_all:
        raise RuntimeError("No windows were created for run_two.")

    tic_id = np.asarray(tic_all, dtype=np.int64)
    data = {
        "X": np.asarray(X_all, dtype=np.float32),
        "y": np.asarray(y_all, dtype=np.float32),
        "tic_id": tic_id,
        "split": split_by_tic(tic_id, args.seed),
        "transit_depth": np.asarray(depth_all, dtype=np.float32),
        "sector": np.asarray(sector_all, dtype=np.int16),
    }

    sanity_check(data, args.input_len, args.target_len)
    out_path = data_dir / "tess_windows.npz"
    np.savez_compressed(out_path, **data)

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sector", "tic_id", "windows", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved {out_path}")
    print(f"Saved {manifest_path}")
    print(f"Total windows: {data['X'].shape[0]}")
    print(f"Unique TICs: {len(np.unique(data['tic_id']))}")
    print(f"Sectors represented: {len(np.unique(data['sector']))}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="run_two")
    parser.add_argument("--start-sector", type=int, default=25)
    parser.add_argument("--end-sector", type=int, default=107)
    parser.add_argument("--tics-per-sector", type=int, default=10)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--target-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max-windows-per-lightcurve", type=int, default=100)
    parser.add_argument("--per-tic-timeout", type=int, default=45)
    parser.add_argument("--author", default="SPOC")
    parser.add_argument("--seed", type=int, default=612)
    return parser.parse_args()


if __name__ == "__main__":
    build_run_two(parse_args())
