from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "run_two" / "data" / "tess_windows.npz"))
CKPT_PATH = ROOT / "transformer_best.pt"
OUTPUT_DIR = Path(os.environ.get("DEMO_OUTPUT_DIR", ROOT / "demo_outputs"))


@dataclass
class Config:
    input_len: int = 256
    target_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 128
    dropout: float = 0.1
    flux_scale: float = 1000.0
    anchor: bool = False
    pool: str = "mean"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(1, cfg.d_model)
        self.pe = PositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            cfg.ff_dim,
            cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        feat_dim = cfg.d_model * (2 if cfg.pool == "meanlast" else 1)
        self.head = nn.Linear(feat_dim, cfg.target_len)

    def pool_features(self, h):
        if self.cfg.pool == "mean":
            return h.mean(dim=1)
        if self.cfg.pool == "last":
            return h[:, -1]
        if self.cfg.pool == "meanlast":
            return torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)
        raise ValueError(f"unknown pool={self.cfg.pool}")

    def forward(self, x):
        x_in = x - x[:, -1:] if self.cfg.anchor else x
        h = self.encoder(self.pe(self.embed(x_in.unsqueeze(-1))))
        return x[:, -1:] + self.head(self.pool_features(h))


def to_model_units(a, cfg: Config):
    return (a - 1.0) * cfg.flux_scale


def from_model_units(a, cfg: Config):
    return a / cfg.flux_scale + 1.0


def load_demo_state():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"missing handoff file: {DATA_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"missing checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    raw_cfg = ckpt.get("config", {})
    state_dict = ckpt["state_dict"]
    cfg = Config(**{k: v for k, v in raw_cfg.items() if k in Config.__annotations__})

    head_width = state_dict["head.weight"].shape[1]
    if head_width == cfg.d_model * 2:
        cfg.pool = raw_cfg.get("pool", "meanlast")
    else:
        cfg.pool = raw_cfg.get("pool", "mean")
    cfg.anchor = bool(raw_cfg.get("anchor", False))

    model = TransformerForecaster(cfg)
    model.load_state_dict(state_dict)
    model.eval()

    with np.load(DATA_PATH, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    split = data["split"]
    test_rows = np.flatnonzero(split == "test")
    if len(test_rows) == 0:
        raise ValueError("handoff file has no test split rows")

    depths = data["transit_depth"][test_rows]
    test_series = np.concatenate([data["X"][test_rows], data["y"][test_rows]], axis=1)
    spread = np.ptp(test_series, axis=1)
    stable = spread <= 0.10
    ranked_rows = test_rows[stable] if stable.any() else test_rows
    ranked_depths = data["transit_depth"][ranked_rows]
    featured = ranked_rows[np.argsort(-ranked_depths)[: min(60, len(ranked_rows))]]
    return cfg, model, data, test_rows, featured


CFG, MODEL, DATA, TEST_ROWS, FEATURED_ROWS = load_demo_state()


def predict_row(row: int):
    x = DATA["X"][row].astype(np.float32)
    y = DATA["y"][row].astype(np.float32)
    with torch.no_grad():
        xb = torch.from_numpy(to_model_units(x[None, :], CFG))
        pred = from_model_units(MODEL(xb).numpy()[0], CFG)
    persistence = np.repeat(x[-1], CFG.target_len)
    return x, y, pred, persistence


def plot_prediction(row: int):
    x, y, pred, persistence = predict_row(row)
    horizon_x = np.arange(CFG.input_len, CFG.input_len + CFG.target_len)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(np.arange(CFG.input_len), x, color="#6b7280", lw=1.2, label="context")
    ax.plot(horizon_x, y, color="#111827", lw=2.2, label="truth")
    ax.plot(horizon_x, pred, color="#2563eb", lw=2.0, ls="--", label="Transformer live prediction")
    ax.plot(horizon_x, persistence, color="#dc2626", lw=1.5, alpha=0.75, label="persistence baseline")
    ax.axvline(CFG.input_len - 0.5, color="#9ca3af", lw=1)
    ax.set_xlim(CFG.input_len - 96, CFG.input_len + CFG.target_len)
    ax.set_xlabel("cadence step")
    ax.set_ylabel("relative flux")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=9)

    tic = int(DATA["tic_id"][row])
    sector = DATA["sector"][row] if "sector" in DATA else "n/a"
    depth_ppm = float(DATA["transit_depth"][row]) * 1e6
    mae_model = float(np.abs(pred - y).mean()) * 1e6
    mae_persist = float(np.abs(persistence - y).mean()) * 1e6
    ax.set_title(
        f"TIC {tic} | sector {sector} | depth {depth_ppm:,.0f} ppm | "
        f"MAE: Transformer {mae_model:,.0f} ppm vs persistence {mae_persist:,.0f} ppm"
    )

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii"), mae_model, mae_persist


def html_page(row: int):
    img64, mae_model, mae_persist = plot_prediction(row)
    tic = int(DATA["tic_id"][row])
    sector = DATA["sector"][row] if "sector" in DATA else "n/a"
    depth_ppm = float(DATA["transit_depth"][row]) * 1e6
    pos = int(np.where(FEATURED_ROWS == row)[0][0]) if row in set(FEATURED_ROWS.tolist()) else 0
    prev_row = int(FEATURED_ROWS[(pos - 1) % len(FEATURED_ROWS)])
    next_row = int(FEATURED_ROWS[(pos + 1) % len(FEATURED_ROWS)])
    data_label = DATA_PATH.relative_to(ROOT) if DATA_PATH.is_relative_to(ROOT) else DATA_PATH

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live TESS Model Demo</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #1f2937;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 22px 30px;
      background: white;
      border-bottom: 1px solid #d8dee8;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
    }}
    p {{
      margin: 0;
      color: #667085;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 22px 30px 34px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }}
    .stat {{
      background: white;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 13px 14px;
    }}
    .stat b {{
      display: block;
      font-size: 22px;
      line-height: 1.1;
    }}
    .stat span {{
      display: block;
      margin-top: 4px;
      color: #667085;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .panel {{
      background: white;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      background: white;
    }}
    .actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px;
      border-top: 1px solid #d8dee8;
    }}
    a, button {{
      appearance: none;
      border: 1px solid #b8c3d3;
      background: #f9fafb;
      color: #1f2937;
      border-radius: 7px;
      padding: 9px 12px;
      font: inherit;
      text-decoration: none;
      cursor: pointer;
    }}
    a.primary {{
      background: #235f9f;
      border-color: #235f9f;
      color: white;
    }}
    .note {{
      padding: 14px 2px 0;
      color: #667085;
      font-size: 14px;
      line-height: 1.45;
    }}
    code {{
      background: #eef2f6;
      border: 1px solid #dce3eb;
      border-radius: 5px;
      padding: 2px 5px;
    }}
    @media (max-width: 850px) {{
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .actions {{ flex-wrap: wrap; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Live TESS Transformer Forecast</h1>
    <p>Each click runs the saved Transformer checkpoint on a real held-out TESS window from the handoff file.</p>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><b>{len(DATA["X"]):,}</b><span>handoff windows</span></div>
      <div class="stat"><b>{len(TEST_ROWS):,}</b><span>test windows</span></div>
      <div class="stat"><b>{tic}</b><span>TIC ID</span></div>
      <div class="stat"><b>{sector}</b><span>sector</span></div>
      <div class="stat"><b>{depth_ppm:,.0f}</b><span>depth ppm</span></div>
    </div>
    <div class="panel">
      <img alt="Live Transformer forecast" src="data:image/png;base64,{img64}">
      <div class="actions">
        <a href="/?row={prev_row}">Previous sample</a>
        <span>Transformer MAE {mae_model:,.0f} ppm | Persistence MAE {mae_persist:,.0f} ppm</span>
        <a href="/figures" target="_blank" rel="noopener">Open figures</a>
        <a class="primary" href="/?row={next_row}">Run next prediction</a>
      </div>
    </div>
    <p class="note">
      This is a live inference demo: <code>demo_running_model.py</code> loads
      <code>transformer_best.pt</code>, reads <code>{data_label}</code>,
      converts flux into model units, runs <code>TransformerForecaster.forward()</code>,
      converts the forecast back to relative flux, and plots it against truth and persistence.
    </p>
  </main>
</body>
</html>"""


def figures_page():
    images = sorted(OUTPUT_DIR.glob("*.png"))
    if images:
        cards = "\n".join(
            f"""<figure>
  <a href="/figure/{img.name}" target="_blank" rel="noopener">
    <img src="/figure/{img.name}" alt="{img.stem.replace('_', ' ')}">
  </a>
  <figcaption>{img.stem.replace('_', ' ').title()}</figcaption>
</figure>"""
            for img in images
        )
    else:
        cards = """<div class="empty">
  No figures found yet. Run Step 3 in <code>run_live_demo.sh</code>, then refresh this tab.
</div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TESS Evaluation Figures</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #1f2937;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 22px 30px;
      background: white;
      border-bottom: 1px solid #d8dee8;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
    }}
    p {{
      margin: 0;
      color: #667085;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 22px 30px 34px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      background: white;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      background: white;
    }}
    figcaption {{
      padding: 10px 12px;
      border-top: 1px solid #d8dee8;
      color: #667085;
      font-size: 14px;
    }}
    .empty {{
      background: white;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 18px;
      color: #667085;
    }}
    code {{
      background: #eef2f6;
      border: 1px solid #dce3eb;
      border-radius: 5px;
      padding: 2px 5px;
    }}
    @media (max-width: 850px) {{
      .grid {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TESS Evaluation Figures</h1>
    <p>Generated output images from the tiny demo prediction/evaluation pass.</p>
  </header>
  <main>
    <div class="grid">
      {cards}
    </div>
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/figures":
            body = figures_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path.startswith("/figure/"):
            name = unquote(parsed.path.removeprefix("/figure/"))
            path = (OUTPUT_DIR / name).resolve()
            output_root = OUTPUT_DIR.resolve()
            if path.parent != output_root or path.suffix.lower() != ".png" or not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        qs = parse_qs(parsed.query)
        try:
            row = int(qs.get("row", [int(FEATURED_ROWS[0])])[0])
        except ValueError:
            row = int(FEATURED_ROWS[0])
        if row not in set(TEST_ROWS.tolist()):
            row = int(FEATURED_ROWS[0])

        body = html_page(row).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args)


def main():
    host, port = "127.0.0.1", int(os.environ.get("PORT", "8766"))
    data_label = DATA_PATH.relative_to(ROOT) if DATA_PATH.is_relative_to(ROOT) else DATA_PATH
    print(f"Loaded {CKPT_PATH.name} and {data_label}")
    print(f"Serving live model demo at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
