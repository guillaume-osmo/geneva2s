"""Per-round chemical-space animation for adaptive-generation logs.

Reads an `adaptive_<backend>.json` log produced by
`AdaptiveSmilesExplorer.save_log()` and renders:

- `--out-png`   final-state scatter (PNG) coloured by discovery round
- `--out-movie` MP4 animation showing molecules appearing round by round

Fingerprint variants:
- `morgan`  — 2048-bit, Jaccard space (`mlxmolkit.morgan_cpu`, GPU pack)
- `erg`     — 315-dim dense float, cosine space (`mlxmolkit.erg_features`)

Projection:
- `umap`    — `mlx_vis.UMAP` (Metal GPU, 30-46× faster than `umap-learn`)
- `tsne`    — `mlx_vis.TSNE`
- `pacmap`  — `mlx_vis.PaCMAP` (preserves global structure)

CLI:
    python -m geneva2s.viz logs/adaptive_mlx.json \
        --fp erg --method umap \
        --out-png logs/space.png --out-movie logs/discovery.mp4 \
        --subsample-per-round 500
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

    Prefers `iterations` (accepted only). Falls back to `generated` if present.
    Round keys come back as Python ints.
    """
    with open(path) as f:
        data = json.load(f)
    src = data.get("iterations") or data.get("generated") or {}
    out: Dict[int, List[str]] = {}
    for k, rows in src.items():
        round_idx = int(k)
        smis = [row["smiles"] for row in rows if row.get("smiles")]
        if smis:
            out[round_idx] = smis
    return out


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
        if len(bucket) > per_round:
            bucket = rng.sample(bucket, per_round)
        smis.extend(bucket)
        round_per_mol.extend([r] * len(bucket))
    return smis, np.asarray(round_per_mol, dtype=np.int32)


# ============================================================================
# Fingerprints (GPU via mlxmolkit, dropping invalids)
# ============================================================================

def compute_fps(smiles: List[str], kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fp_array, kept_idx) where fp_array is float32 (N_valid, D).

    Morgan FPs are unpacked from bits → 2048 float dims (UMAP-friendly).
    ERG FPs are already (N, 315) float32.
    """
    if kind == "erg":
        from mlxmolkit.erg_features import erg_fp_from_smiles
        fp, idx_map = erg_fp_from_smiles(smiles)
        if fp is None or fp.shape[0] == 0:
            return np.zeros((0, 315), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.array(fp).astype(np.float32), np.asarray(idx_map, dtype=np.int64)
    elif kind == "morgan":
        from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
        # Modern MorganGenerator path (the legacy GetMorganFingerprintAsBitVect
        # emits a per-call DEPRECATION WARNING in 2025+ rdkit).
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
# 2D projection (GPU via mlx-vis)
# ============================================================================

def reduce_2d(fps: np.ndarray, method: str, verbose: bool = True) -> np.ndarray:
    """Project (N, D) → (N, 2) using mlx-vis. Returns float32 numpy array."""
    if fps.shape[0] < 5:
        # Degenerate corpus — emit a trivial linear layout so the pipeline doesn't crash.
        return np.column_stack([np.arange(fps.shape[0]), np.zeros(fps.shape[0])]).astype(np.float32)

    import mlx_vis as mv
    method = method.lower()
    if method == "umap":
        reducer = mv.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, verbose=verbose)
    elif method == "tsne":
        reducer = mv.TSNE(n_components=2, verbose=verbose) if "verbose" in mv.TSNE.__init__.__doc__ else mv.TSNE(n_components=2)
    elif method == "pacmap":
        reducer = mv.PaCMAP(n_components=2, verbose=verbose) if "verbose" in mv.PaCMAP.__init__.__doc__ else mv.PaCMAP(n_components=2)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'umap', 'tsne', or 'pacmap'.")

    t0 = time.time()
    Y = reducer.fit_transform(fps.astype(np.float32))
    if verbose:
        print(f"  {method} on {fps.shape[0]:,} pts: {time.time()-t0:.1f}s")
    return np.asarray(Y, dtype=np.float32)


# ============================================================================
# Rendering
# ============================================================================

def render_static(
    coords: np.ndarray,
    round_per_mol: np.ndarray,
    out_path: str,
    title: str,
    point_size: float = 1.5,
    alpha: float = 0.55,
    dpi: int = 150,
) -> None:
    """Final-state scatter colored by discovery round."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=round_per_mol, cmap="Spectral",
        s=point_size, alpha=alpha, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Discovery round")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out_path}")


def render_movie(
    coords: np.ndarray,
    round_per_mol: np.ndarray,
    out_path: str,
    title_prefix: str,
    fps: int = 8,
    point_size: float = 2.5,
    alpha_new: float = 0.85,
    alpha_old: float = 0.25,
    bitrate: int = 4000,
) -> None:
    """Cumulative reveal: frame r shows rounds 0..r.

    Older rounds fade to muted colour; freshly-added round is highlighted.
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
    ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    title_obj = ax.set_title("")

    # Pre-mask each round's indices for fast frame composition.
    idx_per_round = {r: np.where(round_per_mol == r)[0] for r in rounds}

    # Two scatter layers: "old" (rounds < r, dimmed) and "new" (round == r, bright).
    old_sc = ax.scatter([], [], s=point_size * 0.7, alpha=alpha_old, linewidths=0)
    new_sc = ax.scatter([], [], s=point_size * 1.6, alpha=alpha_new, linewidths=0,
                         edgecolors="white")

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

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Animate per-round chemical-space extension from an adaptive log.",
    )
    p.add_argument("log_path", help="Path to adaptive_<backend>.json")
    p.add_argument("--fp", choices=["morgan", "erg"], default="erg",
                   help="Fingerprint kind (default: erg)")
    p.add_argument("--method", choices=["umap", "tsne", "pacmap"], default="umap",
                   help="2D projection method (default: umap)")
    p.add_argument("--subsample-per-round", type=int, default=500,
                   help="Max molecules to keep per round (UMAP cost is ~O(n^2) on GPU). "
                        "0 = keep everything (slow for >50k total points).")
    p.add_argument("--out-png", default=None,
                   help="Output static scatter PNG (default: <log_stem>_<fp>_<method>.png)")
    p.add_argument("--out-movie", default=None,
                   help="Output animation MP4 (default: <log_stem>_<fp>_<method>.mp4)")
    p.add_argument("--fps", type=int, default=8, help="Animation frames per second (default: 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-png", action="store_true")
    p.add_argument("--skip-movie", action="store_true")
    args = p.parse_args(argv)

    log_path = Path(args.log_path)
    stem = log_path.stem
    default_png = log_path.with_name(f"{stem}_{args.fp}_{args.method}.png")
    default_mp4 = log_path.with_name(f"{stem}_{args.fp}_{args.method}.mp4")
    out_png = Path(args.out_png) if args.out_png else default_png
    out_movie = Path(args.out_movie) if args.out_movie else default_mp4

    print(f"loading {log_path} ...")
    rounds = load_log(str(log_path))
    n_rounds = len(rounds)
    n_total = sum(len(v) for v in rounds.values())
    print(f"  {n_rounds} rounds, {n_total:,} molecules total")

    rng = random.Random(args.seed)
    per_round = args.subsample_per_round if args.subsample_per_round > 0 else 10**9
    smis, rpm = stratified_subsample(rounds, per_round=per_round, rng=rng)
    print(f"  subsampled to {len(smis):,} ({per_round}/round cap)")

    print(f"computing {args.fp} fingerprints ...")
    t0 = time.time()
    fps, kept = compute_fps(smis, args.fp)
    print(f"  {fps.shape[0]:,} valid FPs of dim {fps.shape[1]} in {time.time()-t0:.1f}s")
    if fps.shape[0] == 0:
        print("ERROR: no valid molecules; aborting", file=sys.stderr)
        return 1
    rpm = rpm[kept]

    print(f"projecting to 2D with {args.method} ...")
    coords = reduce_2d(fps, args.method, verbose=True)

    title_prefix = f"{args.fp.upper()} · {args.method.upper()}"
    if not args.skip_png:
        render_static(coords, rpm, str(out_png),
                      title=f"{title_prefix} — final state ({coords.shape[0]:,} mols, {n_rounds} rounds)")
    if not args.skip_movie:
        render_movie(coords, rpm, str(out_movie),
                     title_prefix=title_prefix, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
