# Run Two: Sector Coverage Dataset

`run_two_handoff.py` builds a larger sector-coverage handoff dataset:

```text
sectors: 25-107
TICs per sector: 10
max light curve products per TIC-sector: 1
max windows per light curve: 100
output: run_two/data/tess_windows.npz
manifest: run_two/lightcurve_manifest.csv
```

Run the handoff script on the Zaratan login node because it downloads from MAST. After the `.npz` is created, submit training with:

```bash
DATA=run_two/data/tess_windows.npz sbatch train.sh
```

The saved `.npz` includes the standard model keys:

```text
X, y, tic_id, split, transit_depth
```

It also includes an extra `sector` array for traceability. The existing model loader ignores extra keys, so this is safe.
