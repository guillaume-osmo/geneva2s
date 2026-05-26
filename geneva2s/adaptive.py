"""Adaptive inference / autodidactic exploration loop.

Faithful port of the original GENEVA²S adaptive sampler, plus two GPU-batched
discovery-mode variants. All routed through mlxmolkit (chemistry primitives
on the Metal GPU).

Three explorer classes:
- AdaptiveSmilesExplorer:       InChIKey-3-prefix clustering, CPU Morgan-Tanimoto.
                                Standard library deps only (rdkit). No GPU needed.
- MorganAdaptiveExplorer:       Morgan fingerprint + online single-link clustering
                                in Tanimoto space + GPU-batched Tanimoto novelty
                                filter (binary fingerprints, scaffold-coarse).
                                Requires `mlxmolkit` for the GPU path.
- ErgAdaptiveExplorer:          ERG (Extended Reduced Graph) fingerprint + online
                                single-link clustering in cosine space + GPU-batched
                                cosine novelty filter (dense float vectors,
                                pharmacophore-coherent). Captures scaffold hops
                                (same pharmacophore on different skeleton) as same
                                cluster. Requires `mlxmolkit>=0.5.0` for the
                                erg_features + cosine_dense modules.

Plus a cyclic temperature schedule that matches the original.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np
from rdkit.Chem import AllChem, DataStructs, MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold

from .utils import canonicalize, get_inchikey_prefix, is_valid_molecule


# ============================================================================
# Per-round memory snapshot (process RSS + MLX active/peak)
# ============================================================================

def _reset_round_peak() -> None:
    try:
        import mlx.core as mx
        mx.reset_peak_memory()
    except Exception:
        pass


def _mem_snapshot() -> str:
    """Compact per-round memory line: process RSS + MLX active + MLX peak-this-round."""
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        rss = f"{rss_mb:,.0f}MB"
    except Exception:
        rss = "?"
    try:
        import mlx.core as mx
        active = mx.get_active_memory() / (1024 * 1024)
        peak = mx.get_peak_memory() / (1024 * 1024)
        return f"rss={rss} | mlx_active={active:,.0f}MB | mlx_peak={peak:,.0f}MB"
    except Exception:
        return f"rss={rss}"


# ============================================================================
# Cyclic temperature schedule (matches original GEN code)
# ============================================================================

def adaptive_temperature(round_idx: int, freq_counter=None, cluster_counter=None) -> float:
    """5-step cycle: 1.5 → 1.2 → 1.0 → 0.8 → 0.6 → repeat.

    High T explores; low T exploits. Cycling through both per epoch keeps
    the adaptive loop from getting stuck.
    """
    return [1.5, 1.2, 1.0, 0.8, 0.6][round_idx % 5]


def should_switch_to_erg(history, drop_threshold: float = 0.01, patience: int = 2) -> bool:
    """Return True when accepted and novelty both fall for `patience` rounds."""
    if patience < 1 or len(history) < patience + 1:
        return False

    recent = history[-(patience + 1):]
    for prev, curr in zip(recent, recent[1:]):
        accepted_drop = prev["accepted_rate"] - curr["accepted_rate"]
        novelty_drop = prev["novelty_rate"] - curr["novelty_rate"]
        if accepted_drop <= drop_threshold or novelty_drop <= drop_threshold:
            return False
    return True


def _round_metrics(explorer, round_idx: int, n_samples: int, reference_canonical=None):
    accepted = explorer.iteration_data.get(round_idx, [])
    accepted_rate = len(accepted) / max(1, n_samples)

    if reference_canonical is None:
        novel_count = len(accepted)
    else:
        novel_count = sum(
            1
            for row in accepted
            if canonicalize(row["smiles"]) not in reference_canonical
        )

    novelty_rate = novel_count / max(1, n_samples)
    return {
        "round": round_idx,
        "accepted": len(accepted),
        "accepted_rate": accepted_rate,
        "novel": novel_count,
        "novelty_rate": novelty_rate,
    }


# ============================================================================
# AdaptiveSmilesExplorer — InChIKey-prefix clustering (CPU only)
# ============================================================================

class AdaptiveSmilesExplorer:
    """100%-faithful port of adaptive_smiles_generator.AdaptiveSmilesExplorer.

    Per round: generate → validate → check freq/dup/cluster/tanimoto → accept.
    Cluster is the 3-char InChIKey prefix (coarse skeleton hash).
    Optional Morgan-Tanimoto novelty filter (CPU; slow at large dataset sizes).
    """

    def __init__(
        self,
        generator_func: Callable,
        temperature_func: Callable = None,
        use_tanimoto: bool = False,
        tanimoto_threshold: float = 0.7,
        use_cluster: bool = True,
        max_cluster: int = 20,
        max_freq: int = 2,
    ):
        self.generator_func = generator_func
        self.temperature_func = temperature_func or (lambda r, f, c: 1.0)
        self.use_tanimoto = use_tanimoto
        self.tanimoto_threshold = tanimoto_threshold
        self.use_cluster = use_cluster
        self.max_cluster = max_cluster
        self.max_freq = max_freq

        self.round = 0
        self.dataset: set = set()
        self.freq: Counter = Counter()
        self.cluster_counts: Counter = Counter()
        self.fp_cache: dict = {}
        self.fp_dataset: list = []
        self.iteration_data: dict = defaultdict(list)  # accepted
        self.generated_data: dict = defaultdict(list)  # all valid (with accepted flag)

    def _get_fp(self, smi: str):
        if smi not in self.fp_cache:
            mol = MolFromSmiles(smi)
            self.fp_cache[smi] = (
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None
            )
        return self.fp_cache[smi]

    def _max_tanimoto_to_dataset(self, smi: str) -> float:
        fp = self._get_fp(smi)
        if not fp or not self.fp_dataset:
            return 0.0
        sims = DataStructs.BulkTanimotoSimilarity(fp, [f for (_, f) in self.fp_dataset])
        return max(sims) if sims else 0.0

    @staticmethod
    def _scaffold(smi: str):
        try:
            return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
        except Exception:
            return None

    def run_round(self, n_samples: int = 1000, verbose: bool = False):
        start = time.time()
        _reset_round_peak()
        temperature = self.temperature_func(self.round, self.freq, self.cluster_counts)
        generated = self.generator_func(n_samples, temperature)
        added = 0

        for smi in generated:
            if not is_valid_molecule(smi):
                continue
            self.freq[smi] += 1
            prefix = get_inchikey_prefix(smi)
            scaff = self._scaffold(smi)

            accept = True
            if smi in self.dataset:
                accept = False
            if self.freq[smi] > self.max_freq:
                accept = False
            if self.use_cluster and prefix and self.cluster_counts[prefix] >= self.max_cluster:
                accept = False
            if self.use_tanimoto and self._max_tanimoto_to_dataset(smi) >= self.tanimoto_threshold:
                accept = False

            self.generated_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": prefix,
                "scaffold": scaff,
                "accepted": accept,
            })

            if not accept:
                continue

            self.dataset.add(smi)
            if self.use_cluster and prefix:
                self.cluster_counts[prefix] += 1
            if self.use_tanimoto:
                fp = self._get_fp(smi)
                if fp:
                    self.fp_dataset.append((smi, fp))
            self.iteration_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": prefix,
                "scaffold": scaff,
            })
            added += 1

        if verbose:
            print(
                f"[Round {self.round}] Temp: {temperature:.2f} | "
                f"Added: {added}/{n_samples} | Time: {time.time()-start:.2f}s | "
                f"{_mem_snapshot()}"
            )
        self.round += 1

    def save_log(self, path: str, save_all: bool = False, only_round=None):
        to_save = {
            "iterations": (
                dict(self.iteration_data) if only_round is None
                else {only_round: self.iteration_data[only_round]}
            ),
            "frequency": dict(self.freq),
            "cluster_counts": dict(self.cluster_counts),
        }
        if save_all:
            to_save["generated"] = dict(self.generated_data)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(to_save, f, indent=2)

    def get_dataset(self) -> list:
        return list(self.dataset)


# ============================================================================
# Top-level run_adaptive — backend-agnostic adaptive sampling loop
# ============================================================================

def run_adaptive(
    generator_func: Callable,
    n_rounds: int = 5,
    n_samples_per_round: int = 1000,
    mode: str = "default",
    temperature_func: Callable = None,
    verbose: bool = True,
    save_log_path: str = None,
    reference_canonical=None,
    auto_erg_switch: bool = False,
    auto_erg_drop: float = 0.01,
    auto_erg_patience: int = 2,
    after_round: Callable = None,
    **explorer_kwargs,
):
    """Run an adaptive multi-round sampling loop.

    Backend-agnostic: `generator_func(n, temp) -> list[str]` is the only
    backend hook. Wraps PyTorch / TF / MLX `predict_batch_seeds` cleanly.

    mode:
        "default" / "inchikey"   — AdaptiveSmilesExplorer (InChIKey-3-prefix
                                   clustering, CPU; faithful to original GEN).
        "morgan" / "discovery"   — MorganAdaptiveExplorer (Morgan FP + GPU
                                   Tanimoto, scaffold-coarse; requires mlxmolkit).
        "erg"                    — ErgAdaptiveExplorer (ERG fingerprint + GPU
                                   cosine, pharmacophore-coherent / scaffold-hop
                                   aware; requires mlxmolkit>=0.5.0).

    Returns the explorer instance — call `.get_dataset()` for accepted SMILES,
    `.save_log(path)` for the full audit trail.
    """
    if temperature_func is None:
        temperature_func = adaptive_temperature

    if mode in ("default", "inchikey"):
        explorer = AdaptiveSmilesExplorer(
            generator_func=generator_func,
            temperature_func=temperature_func,
            **explorer_kwargs,
        )
    elif mode in ("morgan", "discovery"):
        explorer = MorganAdaptiveExplorer(
            generator_func=generator_func,
            temperature_func=temperature_func,
            **explorer_kwargs,
        )
    elif mode == "erg":
        explorer = ErgAdaptiveExplorer(
            generator_func=generator_func,
            temperature_func=temperature_func,
            **explorer_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. "
            f"Expected one of: default, inchikey, morgan, discovery, erg"
        )

    if verbose:
        print(f"adaptive: mode={mode}, rounds={n_rounds}, "
              f"n_per_round={n_samples_per_round}, explorer={type(explorer).__name__}")

    round_history = []
    for _ in range(n_rounds):
        explorer.run_round(n_samples=n_samples_per_round, verbose=verbose)
        round_idx = explorer.round - 1
        round_history.append(
            _round_metrics(
                explorer,
                round_idx=round_idx,
                n_samples=n_samples_per_round,
                reference_canonical=reference_canonical,
            )
        )
        if after_round is not None:
            try:
                after_round(explorer, round_idx)
            except Exception as e:
                if verbose:
                    print(f"  [after_round callback raised {type(e).__name__}: {e}]")

        if (
            auto_erg_switch
            and mode in ("default", "inchikey")
            and should_switch_to_erg(
                round_history,
                drop_threshold=auto_erg_drop,
                patience=auto_erg_patience,
            )
        ):
            erg_kwargs = {
                "cluster_threshold": explorer_kwargs.get("cluster_threshold", 0.75),
                "max_per_cluster": explorer_kwargs.get(
                    "max_per_cluster", explorer_kwargs.get("max_cluster", 20)
                ),
                "max_freq": explorer_kwargs.get("max_freq", 2),
                "novelty_threshold": explorer_kwargs.get("novelty_threshold", 0.95),
            }
            next_explorer = ErgAdaptiveExplorer(
                generator_func=generator_func,
                temperature_func=temperature_func,
                **erg_kwargs,
            )
            next_explorer.seed_dataset(explorer.get_dataset())
            next_explorer.round = explorer.round
            next_explorer.freq = Counter(explorer.freq)
            next_explorer.iteration_data.update(explorer.iteration_data)
            next_explorer.generated_data.update(explorer.generated_data)
            explorer = next_explorer
            mode = "erg"
            if verbose:
                print(
                    f"  auto-switch → erg after round {round_idx} "
                    f"(accepted and novelty dropped > {100 * auto_erg_drop:.1f}% "
                    f"for {auto_erg_patience} rounds)"
                )

    if save_log_path:
        explorer.save_log(save_log_path, save_all=True)
        if verbose:
            print(f"  log saved → {save_log_path}")

    return explorer


def run_adaptive_iterations(
    explorer,
    n_rounds: int,
    n_samples_per_round: int,
    verbose: bool = True,
):
    """Run the explicit round loop used by the CLI.

    This helper exists so library users can drive the exact same iterative
    process as `python -m geneva2s.main --adaptive ...` without going through
    argparse.
    """
    for _ in range(n_rounds):
        explorer.run_round(n_samples=n_samples_per_round, verbose=verbose)


# ============================================================================
# Morgan fingerprint utilities (uint32-packed, GPU-ready via mlxmolkit)
# ============================================================================

def compute_morgan_batch(smiles_list, radius: int = 2, nbits: int = 2048):
    """Batch Morgan FP. Returns (fp_u32, idx_map) where:
      fp_u32  : mx.array (N_valid, nwords) uint32 — Metal-ready packed bits
      idx_map : list[int] mapping each row back to its index in smiles_list

    Invalid SMILES are dropped. Empty input → (None, []).

    Requires mlxmolkit (and mlx) for the uint32 packing kernel. CPU-only
    callers should fall back to AdaptiveSmilesExplorer (InChIKey mode).
    """
    from mlxmolkit import morgan_fp_bytes_from_smiles, fp_uint8_to_uint32

    valid_smis, idx_map = [], []
    for i, smi in enumerate(smiles_list):
        if smi and MolFromSmiles(smi) is not None:
            valid_smis.append(smi)
            idx_map.append(i)
    if not valid_smis:
        return None, []
    fp_bytes = morgan_fp_bytes_from_smiles(
        valid_smis, radius=radius, nbits=nbits, use_chirality=False
    )
    fp_u32 = fp_uint8_to_uint32(fp_bytes)
    return fp_u32, idx_map


# ============================================================================
# MorganAdaptiveExplorer — online single-link Tanimoto cluster + GPU novelty
#                          (powered by mlxmolkit's Metal Tanimoto pipeline)
# ============================================================================

class MorganAdaptiveExplorer:
    """Adaptive explorer with Morgan/Tanimoto clustering + GPU-batched novelty
    filtering, all routed through mlxmolkit's Metal Tanimoto pipeline.

    Differences from AdaptiveSmilesExplorer:
    - Cluster key = online single-link Butina-style cluster in Tanimoto space
      (not InChIKey-3-prefix). Each accepted molecule joins the closest
      centroid if Tanimoto ≥ cluster_threshold, else starts a new cluster.
    - Novelty filter = mlxmolkit's GPU Tanimoto kernel against the accepted
      bank. Sub-millisecond at 10k+ accepted on Apple Silicon.

    Requires `mlxmolkit` from GitHub:
    `uv pip install --force-reinstall --no-deps git+https://github.com/guillaume-osmo/mlxmolkit.git@main`.
    """

    def __init__(
        self,
        generator_func: Callable,
        temperature_func: Callable = None,
        cluster_threshold: float = 0.6,
        max_per_cluster: int = 20,
        max_freq: int = 2,
        novelty_threshold: float = 0.85,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
    ):
        try:
            import mlx.core as mx  # noqa
            from mlxmolkit import tanimoto_matrix_metal_u32  # noqa
        except ImportError as e:
            raise ImportError(
                "MorganAdaptiveExplorer requires mlx and mlxmolkit. "
                "Install with: pip install 'geneva2s[discovery]'"
            ) from e
        import mlx.core as mx
        from mlxmolkit import tanimoto_matrix_metal_u32
        self._mx = mx
        self._tanimoto = tanimoto_matrix_metal_u32

        self.generator_func = generator_func
        self.temperature_func = temperature_func or (lambda r, f, c: 1.0)
        self.cluster_threshold = cluster_threshold
        self.max_per_cluster = max_per_cluster
        self.max_freq = max_freq
        self.novelty_threshold = novelty_threshold
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits

        self.round = 0
        self.dataset: set = set()
        self.freq: Counter = Counter()
        self.iteration_data: dict = defaultdict(list)
        self.generated_data: dict = defaultdict(list)

        # mx.array (N_accepted, nwords) of uint32-packed Morgan FPs (the "bank").
        # Same for centroids. None until first accept.
        self._bank: Optional["mx.array"] = None
        self._centroids: Optional["mx.array"] = None
        self._centroid_counts: list = []  # parallel to _centroids rows

    @staticmethod
    def _scaffold(smi: str):
        try:
            return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
        except Exception:
            return None

    def _append_rows(self, current, new_rows):
        """Concat an mx.array (N, K) with another (M, K) along axis 0. None-safe."""
        mx = self._mx
        if current is None:
            return new_rows
        return mx.concatenate([current, new_rows], axis=0)

    def run_round(self, n_samples: int = 1000, verbose: bool = False):
        start = time.time()
        _reset_round_peak()
        mx = self._mx
        cluster_summary = {i: c for i, c in enumerate(self._centroid_counts)}
        temperature = self.temperature_func(self.round, self.freq, cluster_summary)
        generated = self.generator_func(n_samples, temperature)

        # Batched Morgan FP (also a validity filter)
        cand_fps_u32, idx_map = compute_morgan_batch(
            generated, radius=self.morgan_radius, nbits=self.morgan_nbits
        )
        if cand_fps_u32 is None:
            if verbose:
                print(f"[Round {self.round}] Temp: {temperature:.2f} | No valid mols")
            self.round += 1
            return
        cand_smiles = [generated[i] for i in idx_map]

        # Batched novelty: max Tanimoto to bank (GPU, sub-ms)
        if self._bank is not None:
            sim_to_bank = self._tanimoto(cand_fps_u32, self._bank)
            max_sims = np.array(sim_to_bank.max(axis=-1))
        else:
            max_sims = np.zeros(len(cand_smiles), dtype=np.float32)

        # Batched cluster assignment: cand × centroids → argmax per row
        if self._centroids is not None:
            sim_to_cent = np.array(self._tanimoto(cand_fps_u32, self._centroids))
        else:
            sim_to_cent = np.zeros((len(cand_smiles), 0), dtype=np.float32)

        added = 0
        accepted_idx = []
        for i, smi in enumerate(cand_smiles):
            self.freq[smi] += 1
            scaff = self._scaffold(smi)

            # Decide cluster (uses centroids snapshotted at start of batch;
            # new clusters created within this batch aren't considered until
            # next round — acceptable for batched online sampling).
            if sim_to_cent.shape[1] > 0:
                best_cid = int(np.argmax(sim_to_cent[i]))
                best_sim = float(sim_to_cent[i, best_cid])
            else:
                best_cid, best_sim = -1, 0.0

            joins_existing = best_sim >= self.cluster_threshold

            accept = True
            if smi in self.dataset:
                accept = False
            elif self.freq[smi] > self.max_freq:
                accept = False
            elif max_sims[i] >= self.novelty_threshold:
                accept = False
            elif joins_existing and self._centroid_counts[best_cid] >= self.max_per_cluster:
                accept = False

            cluster_id = best_cid if joins_existing else -1  # -1 = new cluster pending
            self.generated_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": cluster_id,
                "scaffold": scaff,
                "tanimoto_max": float(max_sims[i]),
                "accepted": accept,
            })

            if not accept:
                continue

            self.dataset.add(smi)
            if joins_existing:
                self._centroid_counts[best_cid] += 1
            else:
                # New cluster: this fp becomes a centroid; count starts at 1.
                self._centroid_counts.append(1)
            accepted_idx.append(i)
            self.iteration_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": cluster_id if joins_existing else len(self._centroid_counts) - 1,
                "scaffold": scaff,
                "tanimoto_max": float(max_sims[i]),
            })
            added += 1

        # Single GPU-side concat: bank gets all accepted, centroids only the
        # new-cluster ones (i.e. those whose best_sim < cluster_threshold).
        if accepted_idx:
            accepted_rows = cand_fps_u32[mx.array(accepted_idx)]
            self._bank = self._append_rows(self._bank, accepted_rows)

            new_centroid_local_idx = []
            for local_pos, i in enumerate(accepted_idx):
                if sim_to_cent.shape[1] == 0 or sim_to_cent[i].max() < self.cluster_threshold:
                    new_centroid_local_idx.append(local_pos)
            if new_centroid_local_idx:
                new_centroid_rows = accepted_rows[mx.array(new_centroid_local_idx)]
                self._centroids = self._append_rows(self._centroids, new_centroid_rows)

        if verbose:
            n_clusters = len(self._centroid_counts)
            bank_size = 0 if self._bank is None else int(self._bank.shape[0])
            print(
                f"[Round {self.round}] T={temperature:.2f} | "
                f"valid={len(cand_smiles)}/{len(generated)} | "
                f"accepted={added} | clusters={n_clusters} | "
                f"bank={bank_size} | time={time.time()-start:.2f}s | "
                f"{_mem_snapshot()}"
            )
        self.round += 1

    def save_log(self, path: str, save_all: bool = False, only_round=None):
        cluster_counts = {i: c for i, c in enumerate(self._centroid_counts)}
        to_save = {
            "iterations": (
                dict(self.iteration_data) if only_round is None
                else {only_round: self.iteration_data.get(only_round, [])}
            ),
            "frequency": dict(self.freq),
            "cluster_counts": cluster_counts,
        }
        if save_all:
            to_save["generated"] = dict(self.generated_data)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(to_save, f, indent=2)

    def get_dataset(self) -> list:
        return list(self.dataset)


# ============================================================================
# ERG fingerprint utilities — wrappers around mlxmolkit's GPU-ready ERG
# ============================================================================

def compute_erg_batch(smiles_list):
    """Batch ERG FP via mlxmolkit. Returns (fp, idx_map) where:
      fp      : mx.array (N_valid, 315) float32 — GPU-resident, cosine-ready
      idx_map : list[int] mapping each row back to its index in smiles_list

    Invalid SMILES are dropped. Empty input → (None, []).

    Requires `mlxmolkit>=0.5.0` for `erg_fp_from_smiles`.
    """
    from mlxmolkit.erg_features import erg_fp_from_smiles

    if not smiles_list:
        return None, []
    fp, idx_map = erg_fp_from_smiles(smiles_list)
    if fp.shape[0] == 0:
        return None, []
    return fp, idx_map


# ============================================================================
# ErgAdaptiveExplorer — ERG cluster + GPU cosine novelty (via mlxmolkit)
# ============================================================================

class ErgAdaptiveExplorer:
    """Adaptive explorer with pharmacophore-coherent (ERG) clustering and
    GPU-batched cosine novelty filtering, both routed through mlxmolkit.

    Differences from `MorganAdaptiveExplorer`:
    - Fingerprint = ERG (Extended Reduced Graph, Stiefl 2006, 315-dim dense
      float) rather than Morgan binary bits. Captures "same pharmacophore
      arrangement on a different skeleton" (scaffold hops) as same cluster —
      something Morgan/Tanimoto under-rates because the bit patterns differ.
    - Similarity space = cosine on dense vectors (`cosine_matrix_dense`,
      `max_cosine_to_set`) rather than Tanimoto on bits. Cosine thresholds
      typically run higher than Tanimoto for the same "perceived similarity"
      (defaults: cluster≥0.75, novelty≥0.95).
    - Bank + centroids are `mx.array (N, 315) float32` — concatenated on
      accept and then queried via a single batched matmul per round.

    Requires `mlxmolkit>=0.5.0` from GitHub:
    `uv pip install --force-reinstall --no-deps git+https://github.com/guillaume-osmo/mlxmolkit.git@main`.
    """

    def __init__(
        self,
        generator_func: Callable,
        temperature_func: Callable = None,
        cluster_threshold: float = 0.75,
        max_per_cluster: int = 20,
        max_freq: int = 2,
        novelty_threshold: float = 0.95,
        max_centroids: int = 2_000,
    ):
        try:
            import mlx.core as mx  # noqa
            from mlxmolkit.erg_features import erg_fp_from_smiles  # noqa
            from mlxmolkit.cosine_dense import (
                cosine_matrix_dense,
                max_cosine_to_set,
            )
        except ImportError as e:
            raise ImportError(
                "ErgAdaptiveExplorer requires mlx and mlxmolkit>=0.5.0. "
                "Install with: pip install 'geneva2s[discovery]'"
            ) from e
        import mlx.core as mx
        from mlxmolkit.cosine_dense import (
            cosine_matrix_dense,
            max_cosine_to_set,
        )
        self._mx = mx
        self._cosine_matrix = cosine_matrix_dense
        self._max_cosine_to_set = max_cosine_to_set

        self.generator_func = generator_func
        self.temperature_func = temperature_func or (lambda r, f, c: 1.0)
        self.cluster_threshold = cluster_threshold
        self.max_per_cluster = max_per_cluster
        self.max_freq = max_freq
        self.novelty_threshold = novelty_threshold
        self.max_centroids = max_centroids

        self.round = 0
        self.dataset: set = set()
        self.freq: Counter = Counter()
        self.iteration_data: dict = defaultdict(list)
        self.generated_data: dict = defaultdict(list)

        # Novelty + cluster assignment both run against centroids only — bounded
        # by the number of distinct clusters, not total accepted molecules.
        self._centroids: Optional["mx.array"] = None
        self._centroid_counts: list = []

    @staticmethod
    def _scaffold(smi: str):
        try:
            return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
        except Exception:
            return None

    def _append_rows(self, current, new_rows):
        """Concat an mx.array (N, D) with another (M, D) along axis 0. None-safe."""
        mx = self._mx
        if current is None:
            return new_rows
        return mx.concatenate([current, new_rows], axis=0)

    def seed_dataset(self, smiles_list, seed_batch_size: int = 5_000):
        """Seed accepted molecules when switching from default/inchikey to ERG.

        All input SMILES are added to ``self.dataset`` so the ERG explorer never
        re-accepts molecules from the previous phase.  ERG FPs are computed in
        chunks of ``seed_batch_size`` to bound peak memory (avoids allocating
        a huge RDKit-object graph for the entire list at once), then merged and
        sampled down to ``max_centroids`` representatives.
        """
        import random as _random

        smiles_list = list(smiles_list)
        if not smiles_list:
            return

        mx = self._mx

        # Always add ALL molecules to the dataset so they cannot be re-accepted.
        self.dataset.update(smiles_list)

        # Compute ERG FPs in chunks to keep peak memory bounded.
        chunk_fps_list = []
        for b0 in range(0, len(smiles_list), seed_batch_size):
            batch = smiles_list[b0: b0 + seed_batch_size]
            fps_chunk, _ = compute_erg_batch(batch)
            if fps_chunk is not None:
                chunk_fps_list.append(np.array(fps_chunk))
            del fps_chunk

        if not chunk_fps_list:
            return

        all_fps = np.concatenate(chunk_fps_list, axis=0)  # shape (N_valid, 315)
        del chunk_fps_list
        n = all_fps.shape[0]

        if n <= self.max_centroids:
            self._centroids = self._append_rows(self._centroids, mx.array(all_fps))
            self._centroid_counts.extend([1] * n)
        else:
            # Random subsample to respect the cap
            chosen = sorted(_random.sample(range(n), self.max_centroids))
            self._centroids = mx.array(all_fps[chosen])
            self._centroid_counts = [1] * self.max_centroids
        del all_fps

    def run_round(self, n_samples: int = 1000, verbose: bool = False):
        start = time.time()
        _reset_round_peak()
        mx = self._mx
        cluster_summary = {i: c for i, c in enumerate(self._centroid_counts)}
        temperature = self.temperature_func(self.round, self.freq, cluster_summary)
        generated = self.generator_func(n_samples, temperature)

        # Batched ERG computation (also a validity filter — invalid SMILES → dropped)
        cand_fps, idx_map = compute_erg_batch(generated)
        if cand_fps is None:
            if verbose:
                print(f"[Round {self.round}] Temp: {temperature:.2f} | No valid mols")
            self.round += 1
            return
        cand_smiles = [generated[i] for i in idx_map]

        # Chunked centroid similarity — avoids creating a full (n_cand × n_cent)
        # matrix in one shot (which was ~1.8 GB per round with 50k centroids).
        # Each chunk is centroid_chunk_size centroids → ~180 MB intermediate.
        n_cand = len(cand_smiles)
        if self._centroids is not None:
            n_cent = int(self._centroids.shape[0])
            best_sims = np.full(n_cand, -1.0, dtype=np.float32)
            best_cids = np.full(n_cand, -1, dtype=np.int32)
            _chunk = getattr(self, "centroid_chunk_size", 5_000)
            for c0 in range(0, n_cent, _chunk):
                chunk = self._centroids[c0: c0 + _chunk]
                chunk_sim = np.array(self._cosine_matrix(cand_fps, chunk))
                chunk_max = chunk_sim.max(axis=1)
                chunk_arg = chunk_sim.argmax(axis=1)
                better = chunk_max > best_sims
                best_sims = np.where(better, chunk_max, best_sims)
                best_cids = np.where(better, c0 + chunk_arg, best_cids)
                del chunk_sim
            max_sims = best_sims  # alias for novelty filter
        else:
            n_cent = 0
            best_sims = np.zeros(n_cand, dtype=np.float32)
            best_cids = np.full(n_cand, -1, dtype=np.int32)
            max_sims = best_sims

        added = 0
        accepted_idx = []
        n_new_centroids_this_round = 0  # tracks how many _centroid_counts.append(1) happen
        for i, smi in enumerate(cand_smiles):
            self.freq[smi] += 1
            scaff = self._scaffold(smi)

            # Cluster assignment from pre-computed chunked best_sims/best_cids.
            if n_cent > 0:
                best_cid = int(best_cids[i])
                best_sim = float(best_sims[i])
            else:
                best_cid, best_sim = -1, 0.0
            joins_existing = best_sim >= self.cluster_threshold

            accept = True
            if smi in self.dataset:
                accept = False
            elif self.freq[smi] > self.max_freq:
                accept = False
            elif max_sims[i] >= self.novelty_threshold:
                accept = False
            elif joins_existing and self._centroid_counts[best_cid] >= self.max_per_cluster:
                accept = False

            cluster_id = best_cid if joins_existing else -1  # -1 = new cluster pending
            self.generated_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": cluster_id,
                "scaffold": scaff,
                "erg_max_cosine": float(max_sims[i]),
                "accepted": accept,
            })

            if not accept:
                continue

            self.dataset.add(smi)
            if joins_existing:
                self._centroid_counts[best_cid] += 1
            else:
                # Respect max_centroids cap: only create a new centroid if room remains.
                under_cap = (
                    len(self._centroid_counts) + n_new_centroids_this_round
                    < self.max_centroids
                )
                if under_cap:
                    self._centroid_counts.append(1)
                    n_new_centroids_this_round += 1
                    cluster_id = len(self._centroid_counts) - 1
                else:
                    cluster_id = -1  # accepted into dataset but no new centroid slot
            accepted_idx.append(i)
            self.iteration_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": cluster_id,
                "scaffold": scaff,
                "erg_max_cosine": float(max_sims[i]),
            })
            added += 1

        # Only new-cluster accepts (best_sim < cluster_threshold) become centroids,
        # and only up to n_new_centroids_this_round of them (those that got a slot).
        if accepted_idx and n_new_centroids_this_round > 0:
            new_centroid_local_idx = []
            for local_pos, i in enumerate(accepted_idx):
                if len(new_centroid_local_idx) >= n_new_centroids_this_round:
                    break
                if n_cent == 0 or float(best_sims[i]) < self.cluster_threshold:
                    new_centroid_local_idx.append(local_pos)
            if new_centroid_local_idx:
                accepted_rows = cand_fps[mx.array(accepted_idx)]
                new_centroid_rows = accepted_rows[mx.array(new_centroid_local_idx)]
                self._centroids = self._append_rows(self._centroids, new_centroid_rows)

        if verbose:
            n_clusters = len(self._centroid_counts)
            print(
                f"[Round {self.round}] T={temperature:.2f} | "
                f"valid={len(cand_smiles)}/{len(generated)} | "
                f"accepted={added} | clusters={n_clusters} | "
                f"accepted_total={len(self.dataset)} | time={time.time()-start:.2f}s | "
                f"{_mem_snapshot()}"
            )
        self.round += 1

    def save_log(self, path: str, save_all: bool = False, only_round=None):
        cluster_counts = {i: c for i, c in enumerate(self._centroid_counts)}
        to_save = {
            "iterations": (
                dict(self.iteration_data) if only_round is None
                else {only_round: self.iteration_data.get(only_round, [])}
            ),
            "frequency": dict(self.freq),
            "cluster_counts": cluster_counts,
        }
        if save_all:
            to_save["generated"] = dict(self.generated_data)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(to_save, f, indent=2)

    def get_dataset(self) -> list:
        return list(self.dataset)
