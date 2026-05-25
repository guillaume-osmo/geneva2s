"""Six-framework forward-pass benchmark + log-batch plot.

Frameworks compared (same model, same weights, Apple Silicon):
  - TF 2.15 + Metal             (requires tf 2.15.1 env)
  - PyTorch + MPS  (eager)
  - MLX baseline   + mx.compile
  - MLX fused      + mx.compile  (Python grouped LSTM)
  - MLX metal      + mx.compile  (custom Metal cell kernel via mlx-addons)
  - MLX metal-grp  + mx.compile  (custom Metal grouped cell kernel)

Outputs: bench/perf.png + bench/perf_data.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PT_MODEL = REPO_ROOT / "models" / "geneva2s.pt"
KERAS_MODEL = REPO_ROOT / "models" / "geneva2s.keras"


def _time(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    return (time.time() - t0) / repeat


def bench_pytorch(batch_sizes, vocab_size, maxlen):
    import torch
    from geneva2s.torch.model import GenevaBiLSTM
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GenevaBiLSTM(vocab_size, init_keras=False).to(device)
    model.load_state_dict(torch.load(str(PT_MODEL), map_location=device, weights_only=True))
    model.eval()
    out = []
    for B in batch_sizes:
        x = torch.randint(0, vocab_size, (B, maxlen), dtype=torch.long, device=device)
        def step():
            with torch.no_grad():
                _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
        out.append(_time(step))
    return out


def bench_mlx_variant(cls, batch_sizes, vocab_size, maxlen):
    import mlx.core as mx
    model = cls(vocab_size)
    model.load_pt_checkpoint(str(PT_MODEL))
    fwd = mx.compile(model)
    out = []
    for B in batch_sizes:
        x = mx.array(np.random.randint(0, vocab_size, (B, maxlen)).astype(np.int32))
        def step():
            mx.eval(fwd(x))
        out.append(_time(step))
    return out


def bench_tf(batch_sizes, vocab_size, maxlen):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        return None
    try:
        model = keras.models.load_model(str(KERAS_MODEL))
    except (TypeError, ImportError):
        return None
    out = []
    for B in batch_sizes:
        x = np.random.randint(0, vocab_size, (B, maxlen)).astype(np.int64)
        x_oh = np.eye(vocab_size, dtype=np.float32)[x]
        x_tf = tf.constant(x_oh)
        def step():
            _ = model(x_tf, training=False)
        out.append(_time(step))
    return out


def make_plot(data, batch_sizes, out_path):
    import matplotlib.pyplot as plt
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)

    series = [
        ("TF 2.15 + Metal",       data.get("tf"),            "#888888", "o", "-",  2.0),
        ("PyTorch + MPS eager",   data.get("pt"),            "#E74C3C", "o", "-",  2.0),
        ("MLX baseline + mx.compile",       data.get("mlx_base"),  "#7FB3D5", "s", "--", 1.6),
        ("MLX Python-fused + mx.compile",   data.get("mlx_fused"), "#2980B9", "s", "-",  2.0),
        ("MLX Metal-cell + mx.compile",     data.get("mlx_metal"), "#52BE80", "^", "-",  2.0),
        ("MLX Metal-grouped + mx.compile",  data.get("mlx_mg"),    "#1E8449", "*", "-",  2.6),
    ]
    for label, ys, color, marker, ls, lw in series:
        if ys is None:
            continue
        ax.plot(batch_sizes, [1000 * y for y in ys], color=color, marker=marker,
                linestyle=ls, linewidth=lw, markersize=9, label=label)

    # Use categorical x positions (evenly spaced) so small batches don't get squashed
    xpos = list(range(len(batch_sizes)))
    fig.clf()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    for label, ys, color, marker, ls, lw in series:
        if ys is None:
            continue
        ax.plot(xpos, [1000 * y for y in ys], color=color, marker=marker,
                linestyle=ls, linewidth=lw, markersize=9, label=label)

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=11)
    ax.set_xlabel("Batch size  (molecules per forward pass)", fontsize=12)
    ax.set_ylabel("Forward-pass latency  (ms)", fontsize=12)
    ax.set_title(
        "GENEVA²S forward-pass speed on Apple M4 Pro\n"
        "biLSTM 128 → 4 × LSTM 64 → Dense (≈495k params), one-hot input maxlen=42",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_ylim(0, max([1000 * y for ys in data.values() if ys for y in ys]) * 1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256, 512])
    args = parser.parse_args()
    batch_sizes = args.batches
    vocab_size, maxlen = 27, 42

    from geneva2s.mlx.model import (
        GenevaBiLSTMMLX, GenevaBiLSTMMLXFused,
        GenevaBiLSTMMLXMetal, GenevaBiLSTMMLXMetalGrouped,
    )

    print("Running benchmarks (each: 3 warmup + 10 timed forward passes)...")
    data = {}
    print("  PyTorch + MPS eager...", flush=True)
    data["pt"] = bench_pytorch(batch_sizes, vocab_size, maxlen)
    print("  MLX baseline...", flush=True)
    data["mlx_base"] = bench_mlx_variant(GenevaBiLSTMMLX, batch_sizes, vocab_size, maxlen)
    print("  MLX Python-fused...", flush=True)
    data["mlx_fused"] = bench_mlx_variant(GenevaBiLSTMMLXFused, batch_sizes, vocab_size, maxlen)
    print("  MLX Metal-cell...", flush=True)
    data["mlx_metal"] = bench_mlx_variant(GenevaBiLSTMMLXMetal, batch_sizes, vocab_size, maxlen)
    print("  MLX Metal-grouped...", flush=True)
    data["mlx_mg"] = bench_mlx_variant(GenevaBiLSTMMLXMetalGrouped, batch_sizes, vocab_size, maxlen)
    print("  TensorFlow 2.15 + Metal...", flush=True)
    data["tf"] = bench_tf(batch_sizes, vocab_size, maxlen)

    out_data = REPO_ROOT / "bench" / "perf_data.json"
    out_png = REPO_ROOT / "bench" / "perf.png"
    with open(out_data, "w") as f:
        json.dump({"batch_sizes": batch_sizes, **{k: v for k, v in data.items() if v is not None}},
                  f, indent=2)
    print(f"saved → {out_data}")
    make_plot(data, batch_sizes, out_png)


if __name__ == "__main__":
    main()
