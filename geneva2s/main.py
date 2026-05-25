"""End-to-end GENEVA²S training + (adaptive) generation orchestrator.

Three backends, three adaptive-inference modes:

    --backend torch         (default; PyTorch + MPS)
    --backend tf            (TensorFlow 2.15 + Metal)
    --backend mlx           (MLX, Apple Silicon native)

    --adaptive              run the autodidactic round-based loop
    --mode default          InChIKey-3-prefix clustering (CPU, fast)
    --mode discovery        Morgan/Tanimoto clustering (GPU via mlxmolkit)
    --mode erg              ERG fingerprint + cosine (pharmacophore-coherent;
                            scaffold-hop aware; GPU via mlxmolkit)

    --optimizer adam        Keras-parity default
              adamuonn      MuonN matrix + AdamN scalar (recommended on M3 Max)
    --augment 5             reproduce the 2019 naug_5x training corpus

Quick start:

    # one-shot generation, PyTorch
    python -m geneva2s.main --epochs 80 --n-generate 1000

    # adaptive discovery mode, MLX + grouped Metal kernel
    python -m geneva2s.main --backend mlx --grouped --adaptive \\
        --rounds 5 --n-generate 1000 --mode discovery

    # adaptive ERG mode (scaffold-hop aware), MLX
    python -m geneva2s.main --backend mlx --grouped --adaptive \\
        --rounds 5 --n-generate 1000 --mode erg

    # TF backend, one-shot
    python -m geneva2s.main --backend tf --skip-train --n-generate 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .tokenizer import CharTokenizer
from .utils import augment_corpus, canonicalize, sanity_check


def _load_corpus(path: str, augment: int = 1, verbose: bool = True):
    """Load a SMILES corpus and optionally expand it via random-walk augmentation.

    augment=1 keeps the canonical form only (= the original geneva2s behavior).
    augment=5 reproduces Chembl24_9k_organic_naug_5x.smi (5× variants per mol),
    which is what the 2019 Smiles-GEN paper trained on and what the Keras/DPO
    checkpoints used. n_aug > 1 trades startup cost for higher validity.
    """
    with open(path) as f:
        raw = [line.strip() for line in f if line.strip()]
    if augment > 1:
        if verbose:
            print(f"  augmenting corpus {augment}× via random-walk SMILES variants...")
        raw = augment_corpus(raw, n_aug=augment)
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


def _generation_phase(args, generator_func, train_canonical, label: str):
    """Run either one-shot or adaptive generation given a backend's gen function."""
    if args.adaptive:
        from .adaptive import run_adaptive
        print(f"\nadaptive generation — mode={args.mode}, rounds={args.rounds}, "
              f"{args.n_generate}/round")
        t0 = time.time()
        # Map CLI --mode to adaptive.run_adaptive's mode strings.
        if args.mode == "erg":
            mode = "erg"
        elif args.mode in ("morgan", "discovery"):
            mode = "discovery"
        else:
            mode = "default"
        explorer_kwargs = {}
        if mode in ("discovery", "erg"):
            explorer_kwargs.update(
                cluster_threshold=args.cluster_threshold,
                max_per_cluster=args.max_per_cluster,
                novelty_threshold=args.novelty_threshold,
            )
        else:
            explorer_kwargs.update(
                use_cluster=True,
                max_cluster=args.max_per_cluster,
            )
        log_path = (Path(args.log_dir) / f"adaptive_{label.lower()}.json"
                    if args.log_dir else None)
        explorer = run_adaptive(
            generator_func=generator_func,
            n_rounds=args.rounds,
            n_samples_per_round=args.n_generate,
            mode=mode,
            save_log_path=str(log_path) if log_path else None,
            **explorer_kwargs,
        )
        gen_time = time.time() - t0
        accepted = explorer.get_dataset()
        print(f"  total wall time: {gen_time:.1f}s")
        print(f"  rounds: {explorer.round} | accepted (deduped): {len(accepted)}")
        _print_score(_score(accepted, train_canonical), f"{label} + adaptive ({args.mode})")
        return accepted

    # One-shot batch generation (existing behaviour)
    print(f"\ngenerating {args.n_generate} molecules (ncopies={args.ncopies})...")
    t0 = time.time()
    generated = generator_func(args.n_generate, 1.0)  # T=1 default for one-shot
    gen_time = time.time() - t0
    print(f"  gen time: {gen_time:.2f}s ({args.n_generate/gen_time:.0f} mol/s)\n")
    _print_score(_score(generated, train_canonical), label)
    return generated


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
    print(f"GenevaBiLSTM params: {sum(p.numel() for p in model.parameters()):,}")

    if not args.skip_train:
        X, y = tok.sliding_window(text)
        print(f"training {args.epochs} epochs on X={X.shape}...")
        t0 = time.time()
        fit_best(model, X, y, device, tok, text,
                 num_epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, optimizer=args.optimizer,
                 weight_decay=args.weight_decay,
                 check_every=args.check_every, save_path=args.model_path)
        print(f"  trained in {time.time()-t0:.1f}s -> {args.model_path}")
    else:
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"--skip-train but no checkpoint at {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device,
                                          weights_only=True))
        model.to(device).eval()
        print(f"loaded {args.model_path}")

    def gen_fn(n, _temp):
        return predict_batch_seeds(
            model, tok, text, ncollect=n, ncopies=args.ncopies, device=device,
        )

    return _generation_phase(args, gen_fn, train_canonical, "PyTorch")


# ----------------------------------------------------------------------------
# TensorFlow backend
# ----------------------------------------------------------------------------

def _run_tf(args, raw, train_canonical):
    from .tf.generate import predict_batch_seeds
    from .tf.model import load_keras_model

    print(f"backend: TensorFlow + Metal")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"TF backend requires --skip-train with --model-path pointing to a "
            f"saved Keras model. No checkpoint at {args.model_path}."
        )

    tok = CharTokenizer(maxlen=42, step=3)
    text = tok.prepare_corpus(raw)
    print(f"vocab={tok.vocab_size}, corpus_chars={len(text):,}")

    model = load_keras_model(args.model_path)
    print(f"loaded {args.model_path}")

    def gen_fn(n, _temp):
        return predict_batch_seeds(model, tok, text, ncollect=n, ncopies=args.ncopies)

    return _generation_phase(args, gen_fn, train_canonical, "TF")


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

    if args.grouped:
        cls, variant = GenevaBiLSTMMLXMetalGrouped, "MetalGrouped"
    elif args.metal:
        cls, variant = GenevaBiLSTMMLXMetal, "Metal"
    elif args.fused:
        cls, variant = GenevaBiLSTMMLXFused, "Python-fused"
    else:
        cls, variant = GenevaBiLSTMMLX, "baseline"
    print(f"backend: MLX — GenevaBiLSTMMLX{variant if variant != 'baseline' else ''}")

    tok = CharTokenizer(maxlen=42, step=3)
    text = tok.prepare_corpus(raw)
    print(f"vocab={tok.vocab_size}, corpus_chars={len(text):,}")

    model = cls(tok.vocab_size)
    if not args.skip_train:
        X, y = tok.sliding_window(text)
        if args.optimizer != "adam":
            print(f"  note: --optimizer {args.optimizer} is torch-only for now; "
                  f"MLX backend uses mlx.optimizers.Adam.")
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

    def gen_fn(n, _temp):
        return predict_batch_seeds(model, tok, text, ncollect=n, ncopies=args.ncopies)

    return _generation_phase(args, gen_fn, train_canonical, "MLX")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--smi", default=str(repo_root / "data" / "chembl_9k_organic.smi"),
                   help="Path to training SMILES file (one per line)")
    p.add_argument("--backend", choices=["torch", "tf", "mlx"], default=None,
                   help="Inference backend (default: torch; --use-mlx kept for back-compat)")
    # Legacy back-compat (single-flag selection)
    p.add_argument("--use-mlx", action="store_true",
                   help="(legacy) shorthand for --backend mlx")
    p.add_argument("--use-tf", action="store_true",
                   help="(legacy) shorthand for --backend tf")

    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--check-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-3,
                   help="Learning rate (Keras-parity default 3e-3)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Weight decay (passed to AdamW / Muon family; 0 for plain Adam)")
    p.add_argument("--optimizer",
                   choices=["adam", "adamw", "adamn",
                            "muon", "adamuon", "adamuonn",
                            "adamuon_official", "muon_vx"],
                   default="adam",
                   help="Optimizer for PyTorch training. 'adam' = Keras-parity default; "
                        "'adamuonn' = MuonN matrix + AdamN scalar (recommended on M3 Max)")
    p.add_argument("--augment", type=int, default=1,
                   help="Per-molecule random-walk SMILES variants. "
                        "1 = canonical only (default); 5 = reproduce the 2019 naug_5x corpus")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--model-path", default=None)

    # Generation (one-shot)
    p.add_argument("--n-generate", type=int, default=1000,
                   help="Number of molecules per round (or one-shot total)")
    p.add_argument("--ncopies", type=int, default=20)

    # Adaptive
    p.add_argument("--adaptive", action="store_true",
                   help="Run adaptive (round-based) inference instead of one-shot")
    p.add_argument("--rounds", type=int, default=5,
                   help="Number of adaptive rounds (default 5)")
    p.add_argument("--mode", choices=["default", "inchikey", "morgan", "discovery", "erg"],
                   default="default",
                   help="Adaptive clustering: 'default'/'inchikey' (CPU, fast), "
                        "'morgan'/'discovery' (GPU Tanimoto via mlxmolkit), or "
                        "'erg' (ERG FP + GPU cosine, scaffold-hop aware; mlxmolkit>=0.5.0)")
    p.add_argument("--cluster-threshold", type=float, default=None,
                   help="Cluster-membership threshold (discovery/erg). "
                        "Default 0.6 for Tanimoto (discovery), 0.75 for cosine (erg).")
    p.add_argument("--max-per-cluster", type=int, default=20,
                   help="Cap on accepted molecules per cluster")
    p.add_argument("--novelty-threshold", type=float, default=None,
                   help="Max-similarity reject threshold against accepted bank "
                        "(discovery/erg). Default 0.85 (Tanimoto) / 0.95 (cosine).")
    p.add_argument("--log-dir", default=None,
                   help="Directory to save adaptive logs (JSON)")

    # MLX-specific model variant flags
    p.add_argument("--metal", action="store_true",
                   help="MLX only: use Metal LSTM cell kernel (requires mlx-addons)")
    p.add_argument("--grouped", action="store_true",
                   help="MLX only: Metal + grouped 4-branch (fastest)")
    p.add_argument("--fused", action="store_true",
                   help="MLX only: Python-fused grouped LSTM")
    args = p.parse_args()

    # Resolve --backend from legacy flags
    if args.backend is None:
        if args.use_mlx:
            args.backend = "mlx"
        elif args.use_tf:
            args.backend = "tf"
        else:
            args.backend = "torch"

    # Validation
    if (args.metal or args.grouped or args.fused) and args.backend != "mlx":
        p.error("--metal/--grouped/--fused require --backend mlx")
    if args.grouped:
        args.metal = True
    if args.adaptive and args.mode in ("morgan", "discovery", "erg"):
        try:
            import mlxmolkit  # noqa
        except ImportError:
            p.error(f"--mode {args.mode} requires mlxmolkit: pip install mlxmolkit-rdkit")
        if args.mode == "erg":
            try:
                from mlxmolkit.erg_features import erg_fp_from_smiles  # noqa
                from mlxmolkit.cosine_dense import cosine_matrix_dense  # noqa
            except ImportError:
                p.error("--mode erg requires mlxmolkit>=0.5.0 (erg_features + cosine_dense). "
                        "Upgrade with: pip install -U mlxmolkit-rdkit")

    # Threshold defaults: Tanimoto for morgan/discovery, cosine for erg.
    if args.cluster_threshold is None:
        args.cluster_threshold = 0.75 if args.mode == "erg" else 0.6
    if args.novelty_threshold is None:
        args.novelty_threshold = 0.95 if args.mode == "erg" else 0.85

    # Default model path per backend
    if args.model_path is None:
        if args.backend == "mlx":
            tag = "mlx_metal_grouped" if args.grouped else (
                "mlx_metal" if args.metal else "mlx_fused" if args.fused else "mlx")
            args.model_path = str(repo_root / "models" / f"geneva2s_{tag}.safetensors")
        elif args.backend == "tf":
            args.model_path = str(repo_root / "models" / "geneva2s.keras")
        else:
            args.model_path = str(repo_root / "models" / "geneva2s_torch.pt")

    raw, train_canonical = _load_corpus(args.smi, augment=args.augment)
    print(f"corpus: {len(raw)} input SMILES, {len(train_canonical)} canonical unique")

    if args.backend == "mlx":
        _run_mlx(args, raw, train_canonical)
    elif args.backend == "tf":
        _run_tf(args, raw, train_canonical)
    else:
        _run_pytorch(args, raw, train_canonical)


if __name__ == "__main__":
    main()
