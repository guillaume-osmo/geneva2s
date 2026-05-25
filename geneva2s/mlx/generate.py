"""MLX sliding-window molecule generation. Mirrors geneva2s.torch.generate."""
from __future__ import annotations

from datetime import datetime

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError("pip install mlx") from e

from ..tokenizer import CharTokenizer, unreplace_multichar


def _keras_sample(preds_np: np.ndarray) -> int:
    preds = preds_np.astype("float64")
    preds = np.log(preds + 1e-10)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return int(np.argmax(np.random.multinomial(1, preds, 1)))


def predict_batch_seeds(
    model,
    tokenizer: CharTokenizer,
    text: str,
    ncollect: int = 1000,
    ncopies: int = 20,
    verbose: bool = False,
    max_smi_len: int = 120,
):
    """Generate ncollect SMILES via sliding-window rollout. Returns list[str]."""
    maxlen = tokenizer.maxlen
    nl_positions = np.array(
        [i for i, c in enumerate(text) if c == "\n" and i >= maxlen],
        dtype=np.int64,
    )
    if len(nl_positions) == 0:
        raise ValueError("No valid seed positions in text.")

    mols = []
    starttime = datetime.now()

    while len(mols) < ncollect:
        n_seeds = max(1, ncollect // ncopies)
        seed_chars_list = []
        for _ in range(n_seeds):
            end = int(np.random.choice(nl_positions))
            seed = text[end - maxlen + 1 : end + 1]
            seed_chars_list.append(seed)

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
                    next_indices[i] = _keras_sample(probs_np[i])

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
