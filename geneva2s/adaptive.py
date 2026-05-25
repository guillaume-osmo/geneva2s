"""Adaptive inference / autodidactic exploration loop.

Faithful port of the original GENEVA²S adaptive sampler, plus an ERG-based
variant that uses pharmacophore-coherent clustering and GPU-batched novelty
filtering for much better scaffold diversity.

Two explorer classes:
- AdaptiveSmilesExplorer:     InChIKey-3-prefix clustering, CPU Morgan-Tanimoto.
                              Standard library deps only (rdkit). No GPU needed.
- ErgAdaptiveExplorer:        ERG fingerprint + online pharmacophore clustering
                              + GPU-batched cosine novelty filter via mlx-addons.
                              Requires `mlx-addons` for the GPU path.

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
# Cyclic temperature schedule (matches original GEN code)
# ============================================================================

def adaptive_temperature(round_idx: int, freq_counter=None, cluster_counter=None) -> float:
    """5-step cycle: 1.5 → 1.2 → 1.0 → 0.8 → 0.6 → repeat.

    High T explores; low T exploits. Cycling through both per epoch keeps
    the adaptive loop from getting stuck.
    """
    return [1.5, 1.2, 1.0, 0.8, 0.6][round_idx % 5]


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
                f"Added: {added}/{n_samples} | Time: {time.time()-start:.2f}s"
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
    **explorer_kwargs,
):
    """Run an adaptive multi-round sampling loop.

    Backend-agnostic: `generator_func(n, temp) -> list[str]` is the only
    backend hook. Wraps PyTorch / TF / MLX `predict_batch_seeds` cleanly.

    mode:
        "default" / "inchikey"   — AdaptiveSmilesExplorer (InChIKey-3-prefix
                                   clustering, CPU; faithful to original GEN).
        "erg" / "discovery"      — ErgAdaptiveExplorer (pharmacophore-coherent
                                   ERG clustering + GPU-batched novelty filter;
                                   requires `mlx-addons`).

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
    elif mode in ("erg", "discovery"):
        explorer = ErgAdaptiveExplorer(
            generator_func=generator_func,
            temperature_func=temperature_func,
            **explorer_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of: default, inchikey, erg, discovery"
        )

    if verbose:
        print(f"adaptive: mode={mode}, rounds={n_rounds}, "
              f"n_per_round={n_samples_per_round}, explorer={type(explorer).__name__}")

    for _ in range(n_rounds):
        explorer.run_round(n_samples=n_samples_per_round, verbose=verbose)

    if save_log_path:
        explorer.save_log(save_log_path, save_all=True)
        if verbose:
            print(f"  log saved → {save_log_path}")

    return explorer


# ============================================================================
# ERG fingerprint utilities (rdkit, CPU; cheap per-mol)
# ============================================================================

def compute_erg(smi: str) -> Optional[np.ndarray]:
    """Compute the 315-dim ERG fingerprint for a SMILES. Returns None if invalid."""
    from rdkit.Chem import rdReducedGraphs
    mol = MolFromSmiles(smi)
    if mol is None:
        return None
    return np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)


def compute_erg_batch(smiles_list) -> tuple[np.ndarray, list]:
    """Returns (fp_matrix (N_valid, 315), index_map list[int])."""
    fps, idx_map = [], []
    for i, smi in enumerate(smiles_list):
        fp = compute_erg(smi)
        if fp is not None:
            fps.append(fp)
            idx_map.append(i)
    if not fps:
        return np.zeros((0, 315), dtype=np.float32), []
    return np.stack(fps, axis=0), idx_map


# ============================================================================
# ErgAdaptiveExplorer — ERG cluster + GPU novelty (via mlx-addons)
# ============================================================================

class ErgAdaptiveExplorer:
    """Adaptive explorer with pharmacophore-coherent (ERG) clustering and
    GPU-batched cosine novelty filtering.

    Differences from AdaptiveSmilesExplorer:
    - Cluster key = ERG online single-link cluster (in cosine space), not
      InChIKey-3-prefix. Captures "same pharmacophore arrangement on a
      different skeleton" (scaffold hops) as same cluster.
    - Novelty filter = MLX-batched cosine sim against all-accepted bank.
      Sub-millisecond at 10k+ accepted; matches BulkTanimotoSimilarity
      semantics but ~100× faster on Apple Silicon.

    Requires `mlx-addons` (`pip install mlx-addons`).
    """

    def __init__(
        self,
        generator_func: Callable,
        temperature_func: Callable = None,
        cluster_threshold: float = 0.75,
        max_per_cluster: int = 20,
        max_freq: int = 2,
        novelty_threshold: float = 0.90,
    ):
        try:
            import mlx.core as mx  # noqa
            from mlx_addons.similarity import (
                OnlineSingleLinkCluster,
                StreamingFingerprintBank,
                max_cosine_to_set,
            )
        except ImportError as e:
            raise ImportError(
                "ErgAdaptiveExplorer requires mlx and mlx-addons. "
                "Install with: pip install 'geneva2s[mlx-metal]'"
            ) from e
        import mlx.core as mx
        self._mx = mx
        self._max_cosine_to_set = max_cosine_to_set

        self.generator_func = generator_func
        self.temperature_func = temperature_func or (lambda r, f, c: 1.0)
        self.max_freq = max_freq
        self.novelty_threshold = novelty_threshold

        self.round = 0
        self.dataset: set = set()
        self.freq: Counter = Counter()
        self.iteration_data: dict = defaultdict(list)
        self.generated_data: dict = defaultdict(list)

        self.cluster = OnlineSingleLinkCluster(
            threshold=cluster_threshold, max_per_cluster=max_per_cluster
        )
        self.fp_bank = StreamingFingerprintBank(dim=315)

    @staticmethod
    def _scaffold(smi: str):
        try:
            return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
        except Exception:
            return None

    def run_round(self, n_samples: int = 1000, verbose: bool = False):
        start = time.time()
        mx = self._mx
        temperature = self.temperature_func(self.round, self.freq, self.cluster.summary())
        generated = self.generator_func(n_samples, temperature)

        # Batched ERG computation (also a validity filter — invalid SMILES → None)
        cand_smiles, cand_fps_np = [], []
        for smi in generated:
            if not smi:
                continue
            fp = compute_erg(smi)
            if fp is None:
                continue
            cand_smiles.append(smi)
            cand_fps_np.append(fp)

        if not cand_fps_np:
            if verbose:
                print(f"[Round {self.round}] Temp: {temperature:.2f} | No valid mols")
            self.round += 1
            return

        # Batched novelty: max cosine to all-accepted bank (sub-ms on GPU)
        cand_fps = mx.array(np.stack(cand_fps_np))
        if len(self.fp_bank) > 0:
            max_sims = np.array(self._max_cosine_to_set(cand_fps, self.fp_bank.matrix))
        else:
            max_sims = np.zeros(len(cand_smiles), dtype=np.float32)

        added = 0
        accepted_fps_np = []
        for i, smi in enumerate(cand_smiles):
            self.freq[smi] += 1
            cluster_id, _ = self.cluster.assign(cand_fps_np[i])
            scaff = self._scaffold(smi)

            accept = True
            if smi in self.dataset:
                accept = False
            if self.freq[smi] > self.max_freq:
                accept = False
            if self.cluster.at_capacity(cluster_id):
                accept = False
            if max_sims[i] >= self.novelty_threshold:
                accept = False

            self.generated_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": cluster_id,
                "scaffold": scaff,
                "erg_max_sim": float(max_sims[i]),
                "accepted": accept,
            })

            if not accept:
                continue

            self.dataset.add(smi)
            self.cluster.increment(cluster_id)
            self.iteration_data[self.round].append({
                "smiles": smi,
                "round": self.round,
                "freq": self.freq[smi],
                "cluster": cluster_id,
                "scaffold": scaff,
                "erg_max_sim": float(max_sims[i]),
            })
            accepted_fps_np.append(cand_fps_np[i])
            added += 1

        if accepted_fps_np:
            self.fp_bank.add_batch(mx.array(np.stack(accepted_fps_np)))

        if verbose:
            print(
                f"[Round {self.round}] T={temperature:.2f} | "
                f"valid={len(cand_smiles)}/{len(generated)} | "
                f"accepted={added} | clusters={self.cluster.n_clusters} | "
                f"bank={len(self.fp_bank)} | time={time.time()-start:.2f}s"
            )
        self.round += 1

    def save_log(self, path: str, save_all: bool = False, only_round=None):
        to_save = {
            "iterations": (
                dict(self.iteration_data) if only_round is None
                else {only_round: self.iteration_data.get(only_round, [])}
            ),
            "frequency": dict(self.freq),
            "cluster_counts": self.cluster.summary(),
        }
        if save_all:
            to_save["generated"] = dict(self.generated_data)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(to_save, f, indent=2)

    def get_dataset(self) -> list:
        return list(self.dataset)
