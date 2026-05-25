"""End-to-end GENEVA²S training + generation orchestrator.

Run with the PyTorch backend (default):
    python -m geneva2s.main --epochs 80 --n-generate 1000

Run with the MLX backend (Apple Silicon, custom Metal LSTM kernel):
    python -m geneva2s.main --use-mlx --epochs 80 --n-generate 1000

Run with the MLX backend + the grouped Metal kernel (fastest training path):
    python -m geneva2s.main --use-mlx --grouped --epochs 80 --n-generate 1000

Required deps:
    PyTorch path:                 pip install -e .[torch]
    MLX baseline path:            pip install -e .[mlx]
    MLX + custom Metal kernels:   pip install -e .[mlx-metal]   (requires mlx-addons)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from .tokenizer import CharTokenizer
from .utils import canonicalize, sanity_check


def _load_corpus(path: str):
    with open(path) as f:
        raw = [line.strip() for line in f if line.strip()]
    train_canonical = set(c for c in (canonicalize(s) for s in raw) if c)
    return raw, train_canonical


def _score(generated, train_canonical):
    rdkit_valid = [c for c in (canonicalize(s) for s in generated) if c]
    sc_pass = [s for s in generated if sanity_check(s)]
    unique = set(rdkit_valid)
    novel = unique - train_canonical
    return {
        "total": len(generated),
        "sc": len(sc_pass) / max(1, len(generated)),
        "rdkit": len(rdkit_valid) / max(1, len(generated)),
        "unique": len(unique),
        "novel": len(novel),
    }


def _print_score(s, label=""):
    print(f"--- {label} ---")
    print(f"  generated:   {s['total']}")
    print(f"  SanityCheck: {100*s['sc']:.2f}%   (text-balance check)")
    print(f"  rdkit:       {100*s['rdkit']:.2f}%   (chemical validity)")
    print(f"  unique:      {s['unique']}")
    print(f"  novel:       {s['novel']}  ({100*s['novel']/max(1,s['total']):.2f}% of generated)")


# ----------------------------------------------------------------------------
# PyTorch backend
# ----------------------------------------------------------------------------

def _run_pytorch(args, raw, train_canonical):
    import torch
    from .torch.generate import predict_batch_seeds
    from .torch.model import GenevaBiLSTM
    from .torch.train import fit_best

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"backend: PyTorch on {device}")

    tok = CharTokenizer(maxlen=42, step=3)
    text = tok.prepare_corpus(raw)
    print(f"vocab={tok.vocab_size}, corpus_chars={len(text):,}")

    model = GenevaBiLSTM(tok.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GenevaBiLSTM params: {n_params:,}")

    if not args.skip_train:
        X, y = tok.sliding_window(text)
        print(f"training {args.epochs} epochs on X={X.shape}...")
        t0 = time.time()
        fit_best(model, X, y, device, tok, text,
                 num_epochs=args.epochs, batch_size=args.batch_size,
                 check_every=args.check_every, save_path=args.model_path)
        print(f"  trained in {time.time()-t0:.1f}s -> {args.model_path}")
    else:
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"--skip-train but no checkpoint at {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device,
                                          weights_only=True))
        model.to(device).eval()
        print(f"loaded {args.model_path}")

    print(f"\ngenerating {args.n_generate} molecules (ncopies={args.ncopies})...")
    t0 = time.time()
    generated = predict_batch_seeds(
        model, tok, text,
        ncollect=args.n_generate, ncopies=args.ncopies, device=device,
    )
    gen_time = time.time() - t0
    print(f"  gen time: {gen_time:.2f}s ({args.n_generate/gen_time:.0f} mol/s)\n")
    _print_score(_score(generated, train_canonical), "PyTorch result")
    return generated


# ----------------------------------------------------------------------------
# MLX backend
# ----------------------------------------------------------------------------

def _run_mlx(args, raw, train_canonical):
    from .mlx.generate import predict_batch_seeds
    from .mlx.model import (
        GenevaBiLSTMMLX, GenevaBiLSTMMLXFused,
        GenevaBiLSTMMLXMetal, GenevaBiLSTMMLXMetalGrouped,
    )
    from .mlx.train import fit_best, load_state

    # Pick model variant
    if args.grouped:
        cls = GenevaBiLSTMMLXMetalGrouped
        variant = "GenevaBiLSTMMLXMetalGrouped (Metal kernels + grouped branches)"
    elif args.metal:
        cls = GenevaBiLSTMMLXMetal
        variant = "GenevaBiLSTMMLXMetal (Metal LSTM cell kernel)"
    elif args.fused:
        cls = GenevaBiLSTMMLXFused
        variant = "GenevaBiLSTMMLXFused (Python-grouped LSTM)"
    else:
        cls = GenevaBiLSTMMLX
        variant = "GenevaBiLSTMMLX (mlx.nn.LSTM baseline)"
    print(f"backend: MLX on Apple Silicon — {variant}")

    tok = CharTokenizer(maxlen=42, step=3)
    text = tok.prepare_corpus(raw)
    print(f"vocab={tok.vocab_size}, corpus_chars={len(text):,}")

    model = cls(tok.vocab_size)

    if not args.skip_train:
        X, y = tok.sliding_window(text)
        print(f"training {args.epochs} epochs on X={X.shape}...")
        t0 = time.time()
        fit_best(model, X, y, tok, text,
                 num_epochs=args.epochs, batch_size=args.batch_size,
                 check_every=args.check_every, save_path=args.model_path)
        print(f"  trained in {time.time()-t0:.1f}s -> {args.model_path}")
    else:
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"--skip-train but no checkpoint at {args.model_path}")
        load_state(model, args.model_path)
        print(f"loaded {args.model_path}")

    print(f"\ngenerating {args.n_generate} molecules (ncopies={args.ncopies})...")
    t0 = time.time()
    generated = predict_batch_seeds(
        model, tok, text,
        ncollect=args.n_generate, ncopies=args.ncopies,
    )
    gen_time = time.time() - t0
    print(f"  gen time: {gen_time:.2f}s ({args.n_generate/gen_time:.0f} mol/s)\n")
    _print_score(_score(generated, train_canonical), "MLX result")
    return generated


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--smi", default=str(repo_root / "data" / "chembl_9k_organic.smi"),
                   help="Path to training SMILES file (one per line)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--check-every", type=int, default=5,
                   help="Validity check + best-weight tracking interval (epochs)")
    p.add_argument("--n-generate", type=int, default=1000,
                   help="Number of molecules to generate at the end")
    p.add_argument("--ncopies", type=int, default=20,
                   help="Parallel completions per seed during generation")
    p.add_argument("--use-mlx", action="store_true",
                   help="Use MLX backend instead of PyTorch (Apple Silicon only)")
    p.add_argument("--metal", action="store_true",
                   help="MLX only: use custom Metal LSTM cell kernel (requires mlx-addons)")
    p.add_argument("--grouped", action="store_true",
                   help="MLX only: use Metal kernel + grouped 4-branch architecture (fastest)")
    p.add_argument("--fused", action="store_true",
                   help="MLX only: use Python-fused grouped LSTM (no Metal kernel)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training and load --model-path instead")
    p.add_argument("--model-path", default=None,
                   help="Where to save/load model weights (auto-named by backend if not given)")
    args = p.parse_args()

    if args.grouped and not args.use_mlx:
        p.error("--grouped requires --use-mlx")
    if args.metal and not args.use_mlx:
        p.error("--metal requires --use-mlx")
    if args.fused and not args.use_mlx:
        p.error("--fused requires --use-mlx")
    # --grouped implies --metal
    if args.grouped:
        args.metal = True

    if args.model_path is None:
        if args.use_mlx:
            ext = ".safetensors"
            tag = "mlx_metal_grouped" if args.grouped else (
                "mlx_metal" if args.metal else "mlx_fused" if args.fused else "mlx")
            args.model_path = str(repo_root / "models" / f"geneva2s_{tag}{ext}")
        else:
            args.model_path = str(repo_root / "models" / "geneva2s_torch.pt")

    raw, train_canonical = _load_corpus(args.smi)
    print(f"corpus: {len(raw)} input SMILES, {len(train_canonical)} canonical unique")

    if args.use_mlx:
        _run_mlx(args, raw, train_canonical)
    else:
        _run_pytorch(args, raw, train_canonical)


if __name__ == "__main__":
    main()
