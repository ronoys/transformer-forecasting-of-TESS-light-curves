from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DATA_PATH", str(ROOT / "demo_handoff" / "data" / "tess_windows.npz"))

from demo_running_model import DATA, TEST_ROWS, predict_row


def main():
    out = ROOT / "demo_handoff" / "test_predictions.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = TEST_ROWS[: min(20, len(TEST_ROWS))]
    if len(rows) == 0:
        raise RuntimeError("demo handoff has no held-out rows to predict.")

    preds_transformer = []
    preds_persistence = []
    true = []
    x_context = []
    for row in rows:
        x, y, pred, persistence = predict_row(int(row))
        x_context.append(x)
        true.append(y)
        preds_transformer.append(pred)
        preds_persistence.append(persistence)

    np.savez(
        out,
        pred_persistence=np.asarray(preds_persistence, dtype=np.float32),
        pred_transformer=np.asarray(preds_transformer, dtype=np.float32),
        true=np.asarray(true, dtype=np.float32),
        X=np.asarray(x_context, dtype=np.float32),
        tic_id=DATA["tic_id"][rows],
        transit_depth=DATA["transit_depth"][rows],
    )

    print(f"Saved {out.relative_to(ROOT)}")
    print(f"Predicted windows: {len(rows)}")
    print("Models: persistence, transformer")


if __name__ == "__main__":
    main()
