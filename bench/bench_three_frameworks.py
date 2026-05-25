"""Three-framework forward-pass benchmark on Apple Silicon.

Same trained weights loaded into PyTorch / MLX / TF (Keras). Times single-batch
forward passes across batch sizes. Includes baseline + best-tuned variants for
each framework so you can see what each runtime can actually achieve.

For TF, install in a separate env (Python 3.11):
  pip install tensorflow==2.15.1 tensorflow-metal==1.1.0 h5py
TF 2.16+ silently drops LSTM ops to CPU on Apple Silicon — do not use.

Run:
  python bench/bench_three_frameworks.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PT_MODEL = REPO_ROOT / "models" / "geneva2s.pt"
KERAS_MODEL = REPO_ROOT / "models" / "geneva2s.keras"


def _time(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    return (time.time() - t0) / repeat


def bench_pytorch(vocab_size, maxlen, batch_sizes, compile_mode=False, warmup=3, repeat=5):
    import torch
    from geneva2s.torch.model import GenevaBiLSTM

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model = GenevaBiLSTM(vocab_size, init_keras=False).to(device)
    model.load_state_dict(torch.load(str(PT_MODEL), map_location=device, weights_only=True))
    model.eval()
    fwd = torch.compile(model, mode="reduce-overhead", dynamic=False) if compile_mode else model

    results = []
    for B in batch_sizes:
        x = torch.randint(0, vocab_size, (B, maxlen), dtype=torch.long, device=device)
        def step():
            with torch.no_grad():
                _ = fwd(x)
            if device.type == "mps":
                torch.mps.synchronize()
        try:
            dt = _time(step, warmup, repeat)
        except Exception as e:
            dt = float("nan")
        results.append((B, dt))
    return results, device.type


def bench_mlx(vocab_size, maxlen, batch_sizes, fused=False, compile=True, warmup=3, repeat=5):
    import mlx.core as mx
    from geneva2s.mlx.model import GenevaBiLSTMMLX, GenevaBiLSTMMLXFused

    cls = GenevaBiLSTMMLXFused if fused else GenevaBiLSTMMLX
    model = cls(vocab_size)
    model.load_pt_checkpoint(str(PT_MODEL))
    fwd = mx.compile(model) if compile else model

    results = []
    for B in batch_sizes:
        x = mx.array(np.random.randint(0, vocab_size, (B, maxlen)).astype(np.int32))
        def step():
            out = fwd(x); mx.eval(out)
        dt = _time(step, warmup, repeat)
        results.append((B, dt))
    return results


def bench_tf(vocab_size, maxlen, batch_sizes, warmup=3, repeat=5):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        return None
    try:
        model = keras.models.load_model(str(KERAS_MODEL))
    except (TypeError, ImportError):
        return None

    results = []
    for B in batch_sizes:
        x = np.random.randint(0, vocab_size, (B, maxlen)).astype(np.int64)
        x_oh = np.eye(vocab_size, dtype=np.float32)[x]
        x_tf = tf.constant(x_oh)
        def step():
            _ = model(x_tf, training=False)
        dt = _time(step, warmup, repeat)
        results.append((B, dt))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256, 512])
    args = parser.parse_args()
    batch_sizes = args.batches

    vocab_size, maxlen = 27, 42

    print("=== Forward pass latency (ms per call) — Apple Silicon ===\n")

    print("  Running PyTorch (eager) …", flush=True)
    pt_eager, pt_device = bench_pytorch(vocab_size, maxlen, batch_sizes, compile_mode=False)
    print("  Running PyTorch (torch.compile) …", flush=True)
    pt_compile, _ = bench_pytorch(vocab_size, maxlen, batch_sizes, compile_mode=True)
    print("  Running MLX (mx.compile) …", flush=True)
    mlx_base = bench_mlx(vocab_size, maxlen, batch_sizes, fused=False, compile=True)
    print("  Running MLX (fused + mx.compile) …", flush=True)
    mlx_fused = bench_mlx(vocab_size, maxlen, batch_sizes, fused=True, compile=True)
    print("  Running TensorFlow + Metal …", flush=True)
    tf_results = bench_tf(vocab_size, maxlen, batch_sizes)

    print()
    have_tf = tf_results is not None
    cols = ["TF 2.15"] if have_tf else []
    cols += ["MLX +compile", "MLX-fused +cmp", f"PT-{pt_device}", f"PT +cmp"]
    print("  " + f"{'batch':>6}  " + "  ".join(f"{c:>14}" for c in cols))
    print("  " + f"{'─'*6}  " + "  ".join("─" * 14 for _ in cols))
    for i, B in enumerate(batch_sizes):
        row = []
        if have_tf:
            row.append(tf_results[i][1])
        row += [mlx_base[i][1], mlx_fused[i][1], pt_eager[i][1], pt_compile[i][1]]
        cells = "  ".join(f"{1000*t:>12.2f}ms" for t in row)
        print(f"  {B:>6d}  {cells}")
    if not have_tf:
        print("\n  (TF skipped — install tensorflow==2.15.1 + tensorflow-metal==1.1.0 in a separate env)")


if __name__ == "__main__":
    main()
