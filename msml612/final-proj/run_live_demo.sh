#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8767}"
DEMO_HANDOFF_DIR="${DEMO_HANDOFF_DIR:-demo_handoff}"
DEMO_OUTPUT_DIR="${DEMO_OUTPUT_DIR:-demo_outputs}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo
    echo "Stopping live model server..."
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

pause() {
  echo
  read -r -p "Press Enter to continue..." _
}

echo "MSML612 TESS Forecasting Live Demo"
echo "Working directory: $(pwd)"
echo
echo "This script runs:"
echo "  1. a tiny TESS handoff build with visible window logs"
echo "  2. a live saved-model inference demo in the browser"
echo "  3. the evaluation/output script"
echo
pause

echo "Step 1/3: Building a tiny demo handoff"
echo "This is intentionally small and writes to ${DEMO_HANDOFF_DIR}/."
echo
if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("astroquery") else 1)
PY
then
  python3 run_two_handoff.py \
    --out-dir "${DEMO_HANDOFF_DIR}" \
    --start-sector 25 \
    --end-sector 25 \
    --tics-per-sector 3 \
    --max-windows-per-lightcurve 5 \
    --per-tic-timeout 60
else
  python3 demo_handoff_from_existing.py
fi

echo
echo "Handoff demo complete."
echo "Look for: selected TICs, kept windows, sanity checks passed, and saved dataset."
pause

echo "Step 2/3: Starting live model inference demo"
echo
echo "Opening server at: http://127.0.0.1:${PORT}/"
echo "This graph uses the tiny handoff from Step 1."
echo "In your browser, click 'Run next prediction' to show a fresh model pass."
echo

if python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(0.5)
    raise SystemExit(0 if s.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
then
  echo "A server is already running on port ${PORT}; reusing it."
else
  DATA_PATH="${DEMO_HANDOFF_DIR}/data/tess_windows.npz" DEMO_OUTPUT_DIR="${DEMO_OUTPUT_DIR}" PORT="${PORT}" python3 demo_running_model.py &
  SERVER_PID="$!"
fi

sleep 4
if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "Live model server failed to start."
  exit 1
fi

echo "Live demo server is running."
echo "Open a browser tab to:"
echo "  http://127.0.0.1:${PORT}/"
pause

echo "Step 3/3: Running evaluation/output script"
echo "This runs predictions on the tiny handoff, then writes fresh demo outputs to ${DEMO_OUTPUT_DIR}/."
echo
DATA_PATH="${DEMO_HANDOFF_DIR}/data/tess_windows.npz" python3 demo_make_predictions.py

python3 evaluate.py \
  --preds "${DEMO_HANDOFF_DIR}/test_predictions.npz" \
  --outdir "${DEMO_OUTPUT_DIR}"

echo
echo "Evaluation complete. Generated files:"
ls -lh "${DEMO_OUTPUT_DIR}"

echo
echo "Optional report preview:"
echo "  cat ${DEMO_OUTPUT_DIR}/metrics_report.md"
echo
echo "Demo finished."
