# Zaratan Data Handoff

This folder contains the data handoff portion of the project. It generates `data/tess_windows.npz` and does not train the model.

## Run

```bash
cd msml612/final-proj
mkdir -p logs data
module load python || true
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-zaratan.txt
python zaratan_handoff.py --n-targets 250 --write-tic-file data/tic_ids.txt
sbatch tess.sh
```

Check progress:

```bash
squeue -u $USER
tail -f logs/handoff-<arrayjobid>_<taskid>.out
```

After all array tasks finish, merge the part files:

```bash
source .venv/bin/activate
python merge_handoff_parts.py
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

Default scale is `250` TICs. `tess.sh` runs them as a Slurm array with up to 20 tasks at once. Each task processes a shard with `1` light curve product per TIC and up to `200` windows per TIC, so individual tasks finish much faster than the old serial job.
