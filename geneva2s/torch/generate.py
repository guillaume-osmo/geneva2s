"""Sliding-window molecule generation matching the original Keras Generator.

Seeds are random maxlen-char windows from training text ending at \\n.
Each step does a full forward on (B, maxlen, vocab), samples next char via
np.random.multinomial(1, softmax(log(p)+ε), 1) — the exact original Sample().
Then shifts the window: drop first, append sampled.

biLSTM cannot be KV-cached (backward direction depends on future tokens), so
this is the appropriate generation mode for the faithful architecture.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import torch

from ..tokenizer import CharTokenizer, unreplace_multichar


def _keras_sample(preds_np: np.ndarray) -> int:
    """Exact port of Generator.Sample."""
    preds = preds_np.astype("float64")
    preds = np.log(preds + 1e-10)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return int(np.argmax(np.random.multinomial(1, preds, 1)))


def _build_seed_id_tensor(seedstrings, tokenizer: CharTokenizer, device):
    B = len(seedstrings)
    maxlen = tokenizer.maxlen
    arr = np.zeros((B, maxlen), dtype=np.int64)
    for i, s in enumerate(seedstrings):
        for t, c in enumerate(s):
            if c in tokenizer.c2i:
                arr[i, t] = tokenizer.c2i[c]
    return torch.from_numpy(arr).to(device)


@torch.no_grad()
def predict_batch_seeds(
    model,
    tokenizer: CharTokenizer,
    text: str,
    ncollect: int = 1000,
    ncopies: int = 20,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
    max_smi_len: int = 120,
):
    """Generate ncollect SMILES via sliding-window rollout. Returns list[str]."""
    model.eval()
    maxlen = tokenizer.maxlen

    nl_positions = np.array(
        [i for i, c in enumerate(text) if c == "\n" and i >= maxlen],
        dtype=np.int64,
    )
    if len(nl_positions) == 0:
        raise ValueError(
            "No valid seed positions in text (no \\n at position >= maxlen)."
        )

    mols = []
    good = bad = 0
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
        seed_ids = _build_seed_id_tensor(seedstrings, tokenizer, device)

        while any(active):
            logits = model(seed_ids)
            probs = torch.softmax(logits, dim=-1).cpu().float().numpy()
            next_indices = np.zeros(B, dtype=np.int64)
            for i in range(B):
                if active[i]:
                    next_indices[i] = _keras_sample(probs[i])

            stop_outer = False
            for i in range(B):
                if not active[i]:
                    continue
                ch = tokenizer.i2c[int(next_indices[i])]
                if ch == "\n":
                    smi = unreplace_multichar(smi_list[i])
                    mols.append(smi)
                    good += 1
                    if verbose and len(mols) % 100 == 0:
                        print(f"  {len(mols)}/{ncollect} G/B={good}/{bad} {smi}")
                    active[i] = False
                    if len(mols) >= ncollect:
                        stop_outer = True
                        break
                else:
                    smi_list[i] += ch
                    if len(smi_list[i]) > max_smi_len:
                        active[i] = False
                        bad += 1
            if stop_outer:
                break

            next_id_t = torch.from_numpy(next_indices).to(device).unsqueeze(1)
            seed_ids = torch.cat([seed_ids[:, 1:], next_id_t], dim=1)

    if verbose:
        dt = (datetime.now() - starttime).total_seconds()
        print(f"  generation: {len(mols)} mols in {dt:.1f}s ({len(mols)/dt:.0f} mol/s)")
    return mols
