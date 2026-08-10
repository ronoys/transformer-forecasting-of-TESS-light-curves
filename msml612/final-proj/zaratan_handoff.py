"""Generate the MSML612 TESS data handoff file on Zaratan.

This script only does Ronoy's data handoff:
  1. get TIC IDs,
  2. download TESS light curves with Lightkurve,
  3. convert them into fixed X/y forecasting windows,
  4. split by TIC to prevent leakage,
  5. save and validate tess_windows.npz.

It intentionally does not train the LSTM/Transformer model.
"""

from __future__ import annotations

import argparse
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd


HANDOFF_KEYS = ["X", "y", "tic_id", "split", "transit_depth"]

FALLBACK_TIC_IDS = [
    261136679,
    307210830,
    410153553,
    150428135,
    231663901,
    25155310,
    260128333,
    425997655,
]


def fetch_toi_tic_ids(n_targets: int) -> list[int]:
    """Fetch TOI TIC IDs from NASA Exoplanet Archive."""
    query = f"""
    select top {n_targets} tid, toi, tfopwg_disp
    from toi
    where tid is not null
    order by toi
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(
        {"query": query, "format": "csv"}
    )
    table = pd.read_csv(url)
    return table["tid"].dropna().astype(int).drop_duplicates().tolist()


def read_tic_file(path: str | None) -> list[int] | None:
    if path is None:
        return None
    values = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(int(line.split(",")[0]))
    return values


def write_tic_file(path: str, tic_ids: list[int]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(tic_id) for tic_id in tic_ids) + "\n")
    print(f"Wrote {len(tic_ids)} TIC IDs to {out}")


def resolve_tic_ids(args) -> list[int]:
    tic_ids = read_tic_file(args.tic_file)
    if tic_ids is not None:
        return tic_ids
    if args.fallback_only:
        return FALLBACK_TIC_IDS[: args.n_targets]
    try:
        return fetch_toi_tic_ids(args.n_targets)
    except Exception as exc:
        print(f"TOI fetch failed, using fallback TICs: {exc}")
        return FALLBACK_TIC_IDS[: args.n_targets]


def apply_shard(tic_ids: list[int], shard_index: int | None, num_shards: int | None) -> list[int]:
    if shard_index is None and num_shards is None:
        return tic_ids
    if shard_index is None or num_shards is None:
        raise ValueError("--shard-index and --num-shards must be used together.")
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    return tic_ids[shard_index::num_shards]


def load_lightcurve(tic_id: int, max_products: int | None, author: str | None):
    import lightkurve as lk

    target = f"TIC {tic_id}"
    authors = [author] if author else ["SPOC", None]
    for author_name in authors:
        search = lk.search_lightcurve(target, mission="TESS", author=author_name)
        if len(search) == 0:
            continue
        if max_products is not None:
            search = search[:max_products]
        try:
            return search.download_all(quality_bitmask="default", flux_column="pdcsap_flux")
        except Exception:
            return search.download_all(quality_bitmask="default")
    return None


def clean_flux(lcc) -> np.ndarray:
    lc = lcc.stitch().remove_nans().remove_outliers(sigma=8).normalize()
    flux = np.asarray(lc.flux.value, dtype=np.float32)
    flux = flux[np.isfinite(flux)]
    median = np.nanmedian(flux)
    if np.isfinite(median) and median != 0:
        flux = flux / median
    return flux.astype(np.float32)


def make_windows(
    flux: np.ndarray,
    tic_id: int,
    input_len: int,
    target_len: int,
    stride: int,
):
    X_rows, y_rows, tic_rows, depth_rows = [], [], [], []
    total_len = input_len + target_len
    for start in range(0, len(flux) - total_len + 1, stride):
        X = flux[start : start + input_len]
        y = flux[start + input_len : start + total_len]
        if X.shape[0] != input_len or y.shape[0] != target_len:
            continue
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            continue

        baseline = float(np.nanmedian(np.concatenate([X, y])))
        transit_depth = max(0.0, baseline - float(np.nanmin(y)))
        X_rows.append(X.astype(np.float32))
        y_rows.append(y.astype(np.float32))
        tic_rows.append(int(tic_id))
        depth_rows.append(np.float32(transit_depth))
    return X_rows, y_rows, tic_rows, depth_rows


def split_by_tic(tic_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stars = np.unique(tic_id)
    rng.shuffle(stars)
    if len(stars) < 3:
        raise ValueError("Need at least 3 TICs with valid windows for train/val/test splits.")

    n_train = max(1, int(round(0.70 * len(stars))))
    n_val = max(1, int(round(0.15 * len(stars))))
    if n_train + n_val >= len(stars):
        n_train = max(1, len(stars) - 2)
        n_val = 1

    train_stars = set(stars[:n_train])
    val_stars = set(stars[n_train : n_train + n_val])

    split = np.empty(len(tic_id), dtype="<U5")
    for i, star in enumerate(tic_id):
        if star in train_stars:
            split[i] = "train"
        elif star in val_stars:
            split[i] = "val"
        else:
            split[i] = "test"
    return split


def sanity_check(data: dict[str, np.ndarray], input_len: int, target_len: int) -> None:
    for key in HANDOFF_KEYS:
        assert key in data, f"missing required key: {key}"

    X, y = data["X"], data["y"]
    tic_id, split, transit_depth = data["tic_id"], data["split"], data["transit_depth"]
    n = X.shape[0]

    assert X.dtype == np.float32, f"X must be float32, got {X.dtype}"
    assert y.dtype == np.float32, f"y must be float32, got {y.dtype}"
    assert tic_id.dtype == np.int64, f"tic_id must be int64, got {tic_id.dtype}"
    assert transit_depth.dtype == np.float32, f"transit_depth must be float32, got {transit_depth.dtype}"
    assert X.shape == (n, input_len), f"bad X shape: {X.shape}"
    assert y.shape == (n, target_len), f"bad y shape: {y.shape}"
    assert tic_id.shape == (n,), f"bad tic_id shape: {tic_id.shape}"
    assert split.shape == (n,), f"bad split shape: {split.shape}"
    assert transit_depth.shape == (n,), f"bad transit_depth shape: {transit_depth.shape}"
    assert np.isfinite(X).all() and np.isfinite(y).all(), "X/y contain NaN or inf"
    assert np.isfinite(transit_depth).all(), "transit_depth contains NaN or inf"
    assert set(np.unique(split)) <= {"train", "val", "test"}, f"bad split labels: {np.unique(split)}"

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(tic_id[split == a]) & set(tic_id[split == b])
        assert not overlap, f"star leakage between {a}/{b}: {overlap}"

    print("Sanity checks passed.")
    print("Window counts:", {s: int((split == s).sum()) for s in ["train", "val", "test"]})
    print("TIC counts:", {s: int(len(np.unique(tic_id[split == s]))) for s in ["train", "val", "test"]})
    print(f"X range: [{X.min():.6f}, {X.max():.6f}]")
    print(f"y range: [{y.min():.6f}, {y.max():.6f}]")


def build_dataset(args) -> dict[str, np.ndarray]:
    tic_ids = resolve_tic_ids(args)
    tic_ids = apply_shard(tic_ids, args.shard_index, args.num_shards)

    print(f"Attempting {len(tic_ids)} TICs.")
    if not tic_ids and args.partial:
        return {
            "X": np.empty((0, args.input_len), dtype=np.float32),
            "y": np.empty((0, args.target_len), dtype=np.float32),
            "tic_id": np.empty((0,), dtype=np.int64),
            "transit_depth": np.empty((0,), dtype=np.float32),
        }

    rng = np.random.default_rng(args.seed)
    X_all, y_all, tic_all, depth_all = [], [], [], []

    for idx, tic_id in enumerate(tic_ids, 1):
        print(f"[{idx}/{len(tic_ids)}] TIC {tic_id}", flush=True)
        try:
            lcc = load_lightcurve(tic_id, max_products=args.max_products, author=args.author)
            if lcc is None or len(lcc) == 0:
                print("  skipped: no light curve found")
                continue
            flux = clean_flux(lcc)
            X, y, tic, depth = make_windows(
                flux,
                tic_id=tic_id,
                input_len=args.input_len,
                target_len=args.target_len,
                stride=args.stride,
            )
            if args.max_windows_per_star is not None and len(X) > args.max_windows_per_star:
                keep = np.sort(rng.choice(len(X), args.max_windows_per_star, replace=False))
                X = [X[i] for i in keep]
                y = [y[i] for i in keep]
                tic = [tic[i] for i in keep]
                depth = [depth[i] for i in keep]
            print(f"  kept {len(X)} windows")
            X_all.extend(X)
            y_all.extend(y)
            tic_all.extend(tic)
            depth_all.extend(depth)
        except Exception as exc:
            print(f"  skipped: {exc}")

    if not X_all and args.partial:
        return {
            "X": np.empty((0, args.input_len), dtype=np.float32),
            "y": np.empty((0, args.target_len), dtype=np.float32),
            "tic_id": np.empty((0,), dtype=np.int64),
            "transit_depth": np.empty((0,), dtype=np.float32),
        }
    if not X_all:
        raise RuntimeError("No valid windows created.")

    data = {
        "X": np.asarray(X_all, dtype=np.float32),
        "y": np.asarray(y_all, dtype=np.float32),
        "tic_id": np.asarray(tic_all, dtype=np.int64),
        "transit_depth": np.asarray(depth_all, dtype=np.float32),
    }
    if args.partial:
        print(f"Partial shard complete: {data['X'].shape[0]} windows from {len(np.unique(data['tic_id']))} TICs.")
        return data

    data["split"] = split_by_tic(data["tic_id"], args.seed)
    sanity_check(data, args.input_len, args.target_len)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Build tess_windows.npz for the MSML612 data handoff.")
    parser.add_argument("--out", default="data/tess_windows.npz")
    parser.add_argument("--n-targets", type=int, default=250)
    parser.add_argument("--tic-file", default=None, help="Optional file with one TIC ID per line.")
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--target-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max-products", type=int, default=4)
    parser.add_argument("--max-windows-per-star", type=int, default=500)
    parser.add_argument("--author", default="SPOC")
    parser.add_argument("--seed", type=int, default=612)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--partial", action="store_true", help="Write a shard file without train/val/test split.")
    parser.add_argument("--write-tic-file", default=None, help="Write resolved TIC IDs to this path and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.write_tic_file:
        write_tic_file(args.write_tic_file, resolve_tic_ids(args))
        return

    data = build_dataset(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"Saved {out}")
    print("Handoff complete. Give this file to the model notebook as data/tess_windows.npz.")


if __name__ == "__main__":
    main()
