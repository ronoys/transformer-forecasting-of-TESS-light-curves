# Zaratan Data Handoff

This folder contains the data handoff portion of the project. It generates `data/tess_windows.npz` and does not train the model.

## Run

```bash
cd msml612/final-proj
mkdir -p logs data
sbatch tess.sh
```

Check progress:

```bash
squeue -u $USER
tail -f logs/handoff-<jobid>.out
```

## Output

The job writes:

```text
data/tess_windows.npz
```

The file contains the model handoff contract:

```text
X              float32 [N, 256]
y              float32 [N, 32]
tic_id         int64   [N]
split          str     [N], train/val/test split by TIC
transit_depth  float32 [N]
```

Default scale is `250` TICs, up to `4` light curve products per TIC, and up to `500` windows per TIC. Increase those values in `tess.sh` for larger Zaratan runs.
