"""Per-round chemical-space animation in a FIXED training-set 2D reference.

The 2D layout is fit ONCE on the model's training corpus (the "learning data
space", round -1). Each adaptive round's generated molecules are projected
into that fixed reference via k-NN out-of-sample projection (UMAP and
similar manifold methods have no native `.transform()` — we use the standard
weighted-kNN reconstruction onto the reference embedding).

The training corpus appears as a muted grey background; each round's
discoveries are layered on top, coloured by discovery round. The result
shows *where* the adaptive loop is exploring relative to what the model
was actually trained on.

Pipeline (default `--fp erg --method umap`, 9k training ref, 866k generated):
- training FPs (mlxmolkit GPU)   : <1s
- training UMAP fit (mlx-vis GPU): ~1s
- per-round kNN project (chunked): ~10-30s total for ~100 rounds × ~8k mols
- rendering (matplotlib + ffmpeg): ~10s

CLI:
    python -m geneva2s.viz logs/adaptive_<backend>.json \\
        --fp erg --method umap \\
        --ref-smiles data/chembl_9k_organic.smi

Outputs: <log_stem>_<fp>_<method>_ref.{png,mp4} alongside the input log.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================================
# Log parsing
# ============================================================================

def load_log(path: str) -> Dict[int, List[str]]:
    """Return {round_idx: [smiles, ...]} from an adaptive-generation log.

    Auto-detects format via extension: `.parquet` → parquet + meta sidecar
    via `adaptive.load_log_full`; otherwise JSON.
    """
    from .adaptive import load_log_full
    data = load_log_full(path)
    src = data.get("iterations") or data.get("generated") or {}
    out: Dict[int, List[str]] = {}
    for k, rows in src.items():
        round_idx = int(k)
        smis = [row["smiles"] for row in rows if row.get("smiles")]
        if smis:
            out[round_idx] = smis
    return out


def _load_ref_smiles(path: str, cap: int = 0) -> List[str]:
    """Read training corpus, dedupe canonical, optionally cap."""
    from .utils import canonicalize
    with open(path) as f:
        raw = [line.strip() for line in f if line.strip()]
    canon = {c for c in (canonicalize(s) for s in raw) if c}
    smis = sorted(canon)
    if cap and len(smis) > cap:
        rng = random.Random(0)
        smis = rng.sample(smis, cap)
    return smis


def stratified_subsample(
    rounds_to_smis: Dict[int, List[str]],
    per_round: int,
    rng: random.Random,
) -> Tuple[List[str], np.ndarray]:
    """Return (smiles_list, round_per_mol) with at most `per_round` from each round."""
    smis: List[str] = []
    round_per_mol: List[int] = []
    for r in sorted(rounds_to_smis):
        bucket = rounds_to_smis[r]
        if per_round and len(bucket) > per_round:
            bucket = rng.sample(bucket, per_round)
        smis.extend(bucket)
        round_per_mol.extend([r] * len(bucket))
    return smis, np.asarray(round_per_mol, dtype=np.int32)


# ============================================================================
# Fingerprints (GPU via mlxmolkit / rdkit MorganGenerator)
# ============================================================================

def compute_fps(smiles: List[str], kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fp_array, kept_idx) where fp_array is float32 (N_valid, D)."""
    if kind == "erg":
        from mlxmolkit.erg_features import erg_fp_from_smiles
        fp, idx_map = erg_fp_from_smiles(smiles)
        if fp is None or fp.shape[0] == 0:
            return np.zeros((0, 315), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.array(fp).astype(np.float32), np.asarray(idx_map, dtype=np.int64)
    elif kind == "morgan":
        from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        rows, kept = [], []
        for i, smi in enumerate(smiles):
            mol = MolFromSmiles(smi)
            if mol is None:
                continue
            rows.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
            kept.append(i)
        if not rows:
            return np.zeros((0, 2048), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.stack(rows), np.asarray(kept, dtype=np.int64)
    else:
        raise ValueError(f"Unknown fp kind: {kind!r}. Use 'morgan' or 'erg'.")


# ============================================================================
# Reference 2D embedding (fit once on training corpus)
# ============================================================================

def fit_reference(
    ref_fps: np.ndarray,
    method: str,
    verbose: bool = True,
) -> np.ndarray:
    """Fit a 2D manifold on the training-corpus fingerprints. Returns (N, 2)."""
    if ref_fps.shape[0] < 5:
        return np.column_stack(
            [np.arange(ref_fps.shape[0]), np.zeros(ref_fps.shape[0])]
        ).astype(np.float32)

    import mlx_vis as mv
    method = method.lower()
    if method == "umap":
        reducer = mv.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, verbose=verbose)
    elif method == "tsne":
        # pca_dim=50 also works around mlx-vis 0.7.0 UnboundLocalError on `X_mx`.
        reducer = mv.TSNE(n_components=2, verbose=verbose, pca_dim=50)
    elif method == "pacmap":
        reducer = mv.PaCMAP(n_components=2, verbose=verbose)
    elif method == "localmap":
        reducer = mv.LocalMAP(n_components=2, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'umap', 'tsne', 'pacmap', or 'localmap'."
        )

    t0 = time.time()
    Y = reducer.fit_transform(ref_fps.astype(np.float32))
    if verbose:
        print(f"  {method} fit on {ref_fps.shape[0]:,} ref pts: {time.time()-t0:.1f}s")
    return np.asarray(Y, dtype=np.float32)


# ============================================================================
# Out-of-sample kNN projection (GPU cosine via mlxmolkit)
# ============================================================================

def knn_project(
    new_fps: np.ndarray,
    ref_fps: np.ndarray,
    ref_coords: np.ndarray,
    k: int = 5,
    chunk: int = 8192,
) -> np.ndarray:
    """Project (N_new, D) FPs onto a 2D reference embedding via weighted kNN.

    For each new point: cosine similarity to all ref points → top-k → 2D coord
    is the similarity-weighted average of those k ref coords. Standard hack
    for UMAP out-of-sample projection (UMAP has no native transform method).

    Chunked to keep `(chunk, N_ref)` cosine matmul memory bounded.
    """
    import mlx.core as mx
    from mlxmolkit.cosine_dense import cosine_matrix_dense

    n_new = new_fps.shape[0]
    n_ref = ref_fps.shape[0]
    if n_new == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if k > n_ref:
        k = n_ref

    ref_mx = mx.array(ref_fps.astype(np.float32))
    out = np.empty((n_new, 2), dtype=np.float32)

    for start in range(0, n_new, chunk):
        end = min(start + chunk, n_new)
        batch = mx.array(new_fps[start:end].astype(np.float32))
        sims = np.array(cosine_matrix_dense(batch, ref_mx))  # (chunk, N_ref)

        topk_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]   # (chunk, k)
        topk_sims = np.take_along_axis(sims, topk_idx, axis=1)    # (chunk, k)
        # Floor negative similarities to 0 so we don't get repulsive weights.
        topk_sims = np.clip(topk_sims, 0.0, None)
        weight_sums = topk_sims.sum(axis=1, keepdims=True)
        # Fall back to uniform weights if all top-k sims are 0 (degenerate cand).
        uniform_fallback = (weight_sums == 0)
        weight_sums = np.where(uniform_fallback, 1.0, weight_sums)
        weights = topk_sims / weight_sums                          # (chunk, k)
        weights = np.where(uniform_fallback, 1.0 / k, weights)

        gathered = ref_coords[topk_idx]                            # (chunk, k, 2)
        out[start:end] = np.einsum("ij,ijk->ik", weights, gathered)

    return out


# ============================================================================
# Rendering — training as muted background, rounds layered on top
# ============================================================================

def _axis_limits(ref_coords: np.ndarray, *extra: np.ndarray, pad: float = 0.05):
    all_pts = np.concatenate([ref_coords, *extra], axis=0) if extra else ref_coords
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    return (xmin - pad * dx, xmax + pad * dx, ymin - pad * dy, ymax + pad * dy)


def render_static(
    coords: np.ndarray,
    round_per_mol: np.ndarray,
    ref_coords: np.ndarray,
    out_path: str,
    title: str,
    point_size: float = 1.5,
    ref_point_size: float = 1.0,
    alpha: float = 0.55,
    ref_alpha: float = 0.20,
    dpi: int = 150,
) -> None:
    """Final-state scatter: training in grey, all rounds coloured by index."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        ref_coords[:, 0], ref_coords[:, 1],
        c="0.55", s=ref_point_size, alpha=ref_alpha, linewidths=0, zorder=1,
    )
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=round_per_mol, cmap="Spectral",
        s=point_size, alpha=alpha, linewidths=0, zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Discovery round")
    ax.set_title(title)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_aspect("equal")
    xmin, xmax, ymin, ymax = _axis_limits(ref_coords, coords)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out_path}")


def render_movie(
    coords: np.ndarray,
    round_per_mol: np.ndarray,
    ref_coords: np.ndarray,
    out_path: str,
    title_prefix: str,
    fps: int = 8,
    point_size: float = 2.5,
    ref_point_size: float = 1.0,
    alpha_new: float = 0.85,
    alpha_old: float = 0.30,
    ref_alpha: float = 0.20,
    bitrate: int = 4000,
) -> None:
    """Cumulative reveal in the FIXED training reference.

    Training set is a static grey background; round r's molecules appear
    bright at frame r, then dim into the Spectral colormap as later rounds
    are added.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    rounds = sorted(set(int(r) for r in round_per_mol.tolist()))
    if not rounds:
        print("  no rounds to render; skipping movie")
        return
    rmax = max(rounds)
    cmap = plt.get_cmap("Spectral")

    fig, ax = plt.subplots(figsize=(10, 10))
    xmin, xmax, ymin, ymax = _axis_limits(ref_coords, coords)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")

    # Static training background (drawn once).
    ax.scatter(
        ref_coords[:, 0], ref_coords[:, 1],
        c="0.55", s=ref_point_size, alpha=ref_alpha, linewidths=0, zorder=1,
    )
    title_obj = ax.set_title("")

    old_sc = ax.scatter(
        [], [], s=point_size * 0.7, alpha=alpha_old, linewidths=0, zorder=2,
    )
    new_sc = ax.scatter(
        [], [], s=point_size * 1.6, alpha=alpha_new, linewidths=0.3,
        edgecolors="white", zorder=3,
    )

    def _frame(frame_round: int):
        old_mask = round_per_mol < frame_round
        new_mask = round_per_mol == frame_round
        old_pts = coords[old_mask]
        new_pts = coords[new_mask]

        old_sc.set_offsets(old_pts if old_pts.shape[0] else np.empty((0, 2)))
        new_sc.set_offsets(new_pts if new_pts.shape[0] else np.empty((0, 2)))

        if old_pts.shape[0]:
            old_sc.set_array(round_per_mol[old_mask].astype(np.float32))
            old_sc.set_cmap(cmap)
            old_sc.set_clim(0, rmax)
        if new_pts.shape[0]:
            new_sc.set_array(np.full(new_pts.shape[0], frame_round, dtype=np.float32))
            new_sc.set_cmap(cmap)
            new_sc.set_clim(0, rmax)

        title_obj.set_text(
            f"{title_prefix} — round {frame_round}/{rmax} "
            f"(+{new_pts.shape[0]:,}, total {old_pts.shape[0]+new_pts.shape[0]:,})"
        )
        return old_sc, new_sc, title_obj

    anim = FuncAnimation(fig, _frame, frames=rounds, blit=False, interval=1000/fps)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=bitrate, codec="libx264")
    print(f"  rendering {len(rounds)} frames → {out_path} ...")
    t0 = time.time()
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path} ({time.time()-t0:.1f}s)")


# ============================================================================
# CLI
# ============================================================================

DEFAULT_REF = "data/chembl_9k_organic.smi"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Animate per-round chemical-space extension in the training-corpus "
                    "2D reference (UMAP fit on training, OOS kNN projection per round).",
    )
    p.add_argument("log_path", help="Path to adaptive_<backend>.json")
    p.add_argument("--fp", choices=["morgan", "erg"], default="erg",
                   help="Fingerprint kind (default: erg)")
    p.add_argument("--method", choices=["umap", "tsne", "pacmap", "localmap"],
                   default="umap",
                   help="Reference 2D projection (default: umap)")
    p.add_argument("--ref-smiles", default=DEFAULT_REF,
                   help=f"Training corpus path used to fit the 2D reference "
                        f"(default: {DEFAULT_REF})")
    p.add_argument("--ref-cap", type=int, default=0,
                   help="Random sub-cap on unique canonical training SMILES "
                        "for the UMAP fit (0 = no cap; recommended: leave at 0 for ~9k corpora)")
    p.add_argument("--knn-k", type=int, default=5,
                   help="k for the OOS kNN projection (default: 5)")
    p.add_argument("--subsample-per-round", type=int, default=0,
                   help="Random cap on generated SMILES per round "
                        "(default: 0 = use all; raise to skip if any round is huge)")
    p.add_argument("--out-png", default=None,
                   help="Output static scatter PNG (default: <log_stem>_<fp>_<method>_ref.png)")
    p.add_argument("--out-movie", default=None,
                   help="Output animation MP4 (default: <log_stem>_<fp>_<method>_ref.mp4)")
    p.add_argument("--fps", type=int, default=8, help="Animation frames per second")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-png", action="store_true")
    p.add_argument("--skip-movie", action="store_true")
    args = p.parse_args(argv)

    log_path = Path(args.log_path)
    stem = log_path.stem
    default_png = log_path.with_name(f"{stem}_{args.fp}_{args.method}_ref.png")
    default_mp4 = log_path.with_name(f"{stem}_{args.fp}_{args.method}_ref.mp4")
    out_png = Path(args.out_png) if args.out_png else default_png
    out_movie = Path(args.out_movie) if args.out_movie else default_mp4

    # --- 1. Reference (training corpus) → 2D layout ---
    ref_path = Path(args.ref_smiles)
    if not ref_path.is_file():
        print(f"ERROR: --ref-smiles {ref_path} not found", file=sys.stderr)
        return 1
    print(f"loading training corpus from {ref_path} ...")
    ref_smis = _load_ref_smiles(str(ref_path), cap=args.ref_cap)
    print(f"  {len(ref_smis):,} unique canonical SMILES")
    print(f"computing reference {args.fp} fingerprints ...")
    t0 = time.time()
    ref_fps, ref_kept = compute_fps(ref_smis, args.fp)
    print(f"  {ref_fps.shape[0]:,} valid FPs of dim {ref_fps.shape[1]} in {time.time()-t0:.1f}s")
    if ref_fps.shape[0] == 0:
        print("ERROR: no valid training FPs; aborting", file=sys.stderr)
        return 1
    print(f"fitting reference 2D ({args.method}) ...")
    ref_coords = fit_reference(ref_fps, args.method, verbose=True)

    # --- 2. Load adaptive log + collect generated SMILES ---
    print(f"\nloading {log_path} ...")
    rounds = load_log(str(log_path))
    n_rounds = len(rounds)
    n_total = sum(len(v) for v in rounds.values())
    print(f"  {n_rounds} rounds, {n_total:,} molecules total")

    rng = random.Random(args.seed)
    per_round = args.subsample_per_round if args.subsample_per_round > 0 else 0
    smis, rpm = stratified_subsample(rounds, per_round=per_round, rng=rng)
    cap_note = f"{per_round}/round cap" if per_round else "no cap, full"
    print(f"  collected {len(smis):,} generated SMILES ({cap_note})")

    # --- 3. Per-round projection into the fixed reference ---
    print(f"computing generated {args.fp} fingerprints ...")
    t0 = time.time()
    gen_fps, gen_kept = compute_fps(smis, args.fp)
    print(f"  {gen_fps.shape[0]:,} valid FPs in {time.time()-t0:.1f}s")
    if gen_fps.shape[0] == 0:
        print("ERROR: no valid generated FPs; aborting", file=sys.stderr)
        return 1
    rpm = rpm[gen_kept]

    print(f"kNN-projecting (k={args.knn_k}) {gen_fps.shape[0]:,} generated "
          f"onto the {ref_fps.shape[0]:,}-point training reference ...")
    t0 = time.time()
    gen_coords = knn_project(gen_fps, ref_fps, ref_coords, k=args.knn_k)
    print(f"  projection done in {time.time()-t0:.1f}s")

    # --- 4. Render ---
    title_prefix = f"{args.fp.upper()} · {args.method.upper()} (training-fixed ref)"
    if not args.skip_png:
        render_static(
            gen_coords, rpm, ref_coords, str(out_png),
            title=f"{title_prefix} — {gen_coords.shape[0]:,} mols across {n_rounds} rounds "
                  f"({ref_fps.shape[0]:,} training ref)",
        )
    if not args.skip_movie:
        render_movie(
            gen_coords, rpm, ref_coords, str(out_movie),
            title_prefix=title_prefix, fps=args.fps,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
