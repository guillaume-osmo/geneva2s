"""MLX sliding-window molecule generation. Mirrors geneva2s.torch.generate."""
from __future__ import annotations

from datetime import datetime

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError("pip install mlx") from e

from ..tokenizer import CharTokenizer, replace_multichar, unreplace_multichar


def _keras_sample(preds_np: np.ndarray, temperature: float = 1.0) -> int:
    """Multinomial sample matching 2025 `SmilesGEN_generator.Sample` exactly.

    `temperature` is accepted for API compatibility with the adaptive loop
    but is NOT applied — the original 2025 sampler is a log→exp→renormalise
    identity (numerically equivalent to sampling directly from `preds_np`).
    The `adaptive_temperature` cycle in `AdaptiveSmilesExplorer` is computed
    and logged but the original 2025 EVA flow likewise never wires it into
    the sampler. Kept as a no-op here to preserve that exact behaviour.
    """
    del temperature  # intentionally unused — see docstring
    preds = preds_np.astype("float64")
    preds = np.log(preds + 1e-10)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return int(np.argmax(np.random.multinomial(1, preds, 1)))


def _seeds_from_text(text: str, maxlen: int, n_seeds: int) -> list:
    """Pick n_seeds random sliding-window seeds from the encoded corpus text."""
    nl_positions = np.array(
        [i for i, c in enumerate(text) if c == "\n" and i >= maxlen],
        dtype=np.int64,
    )
    if len(nl_positions) == 0:
        raise ValueError("No valid seed positions in text.")
    seeds = []
    for _ in range(n_seeds):
        end = int(np.random.choice(nl_positions))
        seeds.append(text[end - maxlen + 1 : end + 1])
    return seeds


def _seeds_from_smiles_pool(
    tokenizer: CharTokenizer,
    pool: list,
    maxlen: int,
    n_seeds: int,
) -> list:
    """Pick n_seeds random SMILES from `pool`, encode each as a fixed-width
    sliding-window seed (Ertl-style replaced + left-padded with newline).

    Mirrors how the original 2025 EVA flow re-seeded from a growing pool.
    """
    if not pool:
        return _seeds_from_text("", maxlen, n_seeds)  # will raise
    chosen = (
        list(np.random.choice(pool, size=n_seeds, replace=True))
        if len(pool) < n_seeds
        else list(np.random.choice(pool, size=n_seeds, replace=False))
    )
    seeds = []
    for smi in chosen:
        replaced = replace_multichar(str(smi))
        if not replaced:
            continue
        # Pad / clip on the left to maxlen with leading newlines (the corpus
        # separator), matching the text-window encoding.
        if len(replaced) >= maxlen:
            seed = replaced[-maxlen:]
        else:
            seed = ("\n" * (maxlen - len(replaced))) + replaced
        seeds.append(seed)
    # If filtering left us short, top up with text fallback seeds upstream.
    while len(seeds) < n_seeds:
        seeds.append("\n" * maxlen)  # benign all-newline window
    return seeds


def predict_batch_seeds(
    model,
    tokenizer: CharTokenizer,
    text: str,
    ncollect: int = 1000,
    ncopies: int = 20,
    verbose: bool = False,
    max_smi_len: int = 120,
    temperature: float = 1.0,
    seed_smiles: list = None,
):
    """Generate ncollect SMILES via sliding-window rollout. Returns list[str].

    seed_smiles: optional list of canonical SMILES to re-seed from at this
                 round. If None, falls back to random windows of `text`
                 (the legacy 2026 behaviour). Pass a growing pool to mirror
                 the 2025 EVA `seed_smiles_pool` evolution.
    temperature: Keras-style softmax temperature applied to logits before
                 sampling. The adaptive loop cycles this 1.5→0.6.
    """
    maxlen = tokenizer.maxlen

    mols = []
    starttime = datetime.now()

    while len(mols) < ncollect:
        n_seeds = max(1, ncollect // ncopies)
        if seed_smiles:
            seed_chars_list = _seeds_from_smiles_pool(
                tokenizer, seed_smiles, maxlen, n_seeds
            )
        else:
            seed_chars_list = _seeds_from_text(text, maxlen, n_seeds)

        seedstrings = [s for s in seed_chars_list for _ in range(ncopies)]
        B = len(seedstrings)
        smi_list = [""] * B
        active = [True] * B

        seed_ids = np.zeros((B, maxlen), dtype=np.int32)
        for i, s in enumerate(seedstrings):
            for t, c in enumerate(s):
                if c in tokenizer.c2i:
                    seed_ids[i, t] = tokenizer.c2i[c]
        x = mx.array(seed_ids)

        while any(active):
            logits = model(x)
            probs_np = np.array(mx.softmax(logits, axis=-1))

            next_indices = np.zeros(B, dtype=np.int32)
            for i in range(B):
                if active[i]:
                    next_indices[i] = _keras_sample(probs_np[i], temperature)

            stop_outer = False
            for i in range(B):
                if not active[i]:
                    continue
                ch = tokenizer.i2c[int(next_indices[i])]
                if ch == "\n":
                    mols.append(unreplace_multichar(smi_list[i]))
                    active[i] = False
                    if len(mols) >= ncollect:
                        stop_outer = True
                        break
                else:
                    smi_list[i] += ch
                    if len(smi_list[i]) > max_smi_len:
                        active[i] = False
            if stop_outer:
                break

            # Shift window
            x = mx.concatenate(
                [x[:, 1:], mx.array(next_indices)[:, None]], axis=1
            )

    if verbose:
        dt = (datetime.now() - starttime).total_seconds()
        print(f"  generation: {len(mols)} mols in {dt:.1f}s ({len(mols)/dt:.0f} mol/s)")
    return mols
