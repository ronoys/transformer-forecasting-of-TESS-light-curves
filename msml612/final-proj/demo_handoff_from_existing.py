from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "run_two" / "data" / "tess_windows.npz"
OUT_DIR = ROOT / "demo_handoff"
OUT_DATA = OUT_DIR / "data" / "tess_windows.npz"
MANIFEST = OUT_DIR / "lightcurve_manifest.csv"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"missing source handoff file: {SOURCE}")

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)

    with np.load(SOURCE, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}

    sectors = data.get("sector")
    if sectors is None:
        sector_mask = np.ones(len(data["X"]), dtype=bool)
        sector_label = "existing"
    else:
        preferred = 25
        sector_mask = sectors == preferred
        if not sector_mask.any():
            preferred = int(sectors[0])
            sector_mask = sectors == preferred
        sector_label = str(preferred)

    tic_ids = []
    for split_name in ("train", "val", "test"):
        split_mask = data["split"] == split_name
        candidates = np.unique(data["tic_id"][sector_mask & split_mask])
        if len(candidates) == 0:
            candidates = np.unique(data["tic_id"][split_mask])
        if len(candidates):
            tic_ids.append(candidates[0])
    tic_ids = np.array(tic_ids, dtype=data["tic_id"].dtype)
    if len(tic_ids) < 3:
        tic_ids = np.unique(data["tic_id"])[:3]

    print(f"=== Sector {sector_label} ===")
    print(f"  selected {len(tic_ids)} TICs: {[int(t) for t in tic_ids]}")

    keep_rows = []
    manifest_rows = ["tic_id,sector,windows_kept"]
    for tic in tic_ids:
        rows = np.flatnonzero(sector_mask & (data["tic_id"] == tic))[:5]
        if len(rows) == 0:
            rows = np.flatnonzero(data["tic_id"] == tic)[:5]
        if len(rows) == 0:
            continue
        keep_rows.extend(rows.tolist())
        sector_value = data["sector"][rows[0]] if "sector" in data else sector_label
        print(f"  TIC {int(tic)}: kept {len(rows)} windows")
        manifest_rows.append(f"{int(tic)},{sector_value},{len(rows)}")

    if not keep_rows:
        raise RuntimeError("No demo windows could be sampled from the existing handoff.")

    keep = np.array(keep_rows, dtype=int)
    subset = {k: v[keep] for k, v in data.items()}

    assert not np.isnan(subset["X"]).any()
    assert not np.isnan(subset["y"]).any()
    assert subset["X"].shape[1] == 256
    assert subset["y"].shape[1] == 32

    splits = subset["split"]
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(subset["tic_id"][splits == a]) & set(subset["tic_id"][splits == b])
        assert not overlap, f"star leakage between {a}/{b}: {overlap}"

    np.savez_compressed(OUT_DATA, **subset)
    MANIFEST.write_text("\n".join(manifest_rows) + "\n")

    print("Sanity checks passed.")
    print(f"Saved {OUT_DATA.relative_to(ROOT)}")
    print(f"Saved {MANIFEST.relative_to(ROOT)}")
    print(f"Total windows: {len(keep)}")
    print(f"Unique TICs: {len(np.unique(subset['tic_id']))}")
    print("Split counts:", {s: int((subset["split"] == s).sum()) for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
