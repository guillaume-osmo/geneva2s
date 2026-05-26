"""Training + sampling for the GPT-MLX SMILES decoder.

Mirrors the role of `mlx/train.py` + `mlx/generate.py` for the BiLSTM, but
the data shape is fundamentally different (causal LM at every position vs
BiLSTM single-next-char) so we keep a separate module rather than overload
the existing ones.

Shapes:
    X : (N, T) int32  — input token ids (BOS-prefixed, EOS-terminated, PAD-right)
    y : (N, T) int32  — same shifted left by 1 (predict next token)
    pad_mask : (N, T) bool — True where target is PAD; loss is masked there

Both BiLSTM-and-GPT paths converge on the adaptive-loop level: `gen_fn`
returns `list[str]` regardless of architecture.
"""
from __future__ import annotations

import os
import time
from typing import List, Sequence

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as mlxnn
    import mlx.optimizers as optim
except ImportError as e:
    raise ImportError("pip install mlx") from e

from ..smiles_tokenizer import SmilesBPE


# ============================================================================
# Data preparation
# ============================================================================

def tokens_from_corpus(
    bpe: SmilesBPE,
    smiles_list: Sequence[str],
    max_len: int,
) -> np.ndarray:
    """BPE-encode each SMILES with BOS/EOS, pad-right to `max_len`, drop overflow.

    Returns (N, max_len) int32 — directly consumable by GPT training.
    """
    encs = bpe.encode_batch(list(smiles_list), add_special_tokens=True)
    # Truncate sequences too long, drop those whose useful content overflows
    rows = []
    for ids in encs:
        if len(ids) > max_len:
            continue  # silently drop molecules too long for the context window
        rows.append(bpe.pad_to_length(ids, max_len, pad_left=False))
    if not rows:
        raise ValueError(
            f"No SMILES fit in max_len={max_len} BPE tokens. "
            f"Either raise --context-length or shorten the corpus."
        )
    return np.asarray(rows, dtype=np.int32)


def prepare_xy(tokens: np.ndarray, pad_id: int):
    """Causal-LM shift: (N, T) -> (X=(N,T-1), y=(N,T-1), pad_mask=(N,T-1))."""
    X = tokens[:, :-1]
    y = tokens[:, 1:]
    pad_mask = y == pad_id
    return X, y, pad_mask


# ============================================================================
# Training
# ============================================================================

def _loss_fn(model, x, y, pad_mask):
    logits = model(x, training=True)                          # (B, T, vocab)
    # Per-position CE, masked over PAD targets
    loss = mlxnn.losses.cross_entropy(logits, y, reduction="none")   # (B, T)
    # Where pad_mask is True, zero out the loss; divide by # of non-pad tokens
    valid = mx.logical_not(pad_mask).astype(mx.float32)
    loss = loss * valid
    n_valid = mx.maximum(valid.sum(), mx.array(1.0, dtype=mx.float32))
    return loss.sum() / n_valid


def _make_compiled_step(model, optimizer):
    loss_and_grad_fn = mlxnn.value_and_grad(model, _loss_fn)
    state = [model.state, optimizer.state]

    def step(x, y, m):
        loss, grads = loss_and_grad_fn(model, x, y, m)
        optimizer.update(model, grads)
        return loss

    return mx.compile(step, inputs=state, outputs=state), state


def fit_gpt(
    model,
    X: np.ndarray,
    y: np.ndarray,
    pad_mask: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    save_path: str = None,
    verbose: bool = True,
    use_compile: bool = True,
):
    """Causal-LM training loop for the GPT-MLX decoder.

    lr default is 3e-4 (small-Transformer regime) vs 3e-3 used for BiLSTM —
    transformers diverge with the BiLSTM lr.
    """
    optimizer = optim.Adam(learning_rate=lr, eps=1e-7)
    if use_compile:
        step, state = _make_compiled_step(model, optimizer)
    else:
        loss_and_grad_fn = mlxnn.value_and_grad(model, _loss_fn)
        state = [model.state, optimizer.state]

        def step(x, y, m):
            loss, grads = loss_and_grad_fn(model, x, y, m)
            optimizer.update(model, grads)
            return loss

    n = X.shape[0]
    history = []
    for ep in range(num_epochs):
        t0 = time.time()
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = mx.array(X[idx])
            yb = mx.array(y[idx])
            mb = mx.array(pad_mask[idx])
            loss = step(xb, yb, mb)
            mx.eval(loss)
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        if verbose:
            print(f"  epoch {ep+1:>3}/{num_epochs}  loss={mean_loss:.4f}  "
                  f"({time.time()-t0:.1f}s, {n//batch_size} steps)")

    if save_path:
        save_state(model, save_path)
        if verbose:
            print(f"  saved → {save_path}")
    return model, history


def save_state(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = dict(_flatten_params("", model.parameters()))
    mx.save_safetensors(path, weights)


def load_state(model, path: str) -> None:
    weights = mx.load(path)
    model.update(_unflatten_params(weights))


def _flatten_params(prefix, p):
    if hasattr(p, "shape"):
        yield (prefix.lstrip("."), p)
    elif isinstance(p, dict):
        for k, v in p.items():
            yield from _flatten_params(f"{prefix}.{k}", v)
    elif isinstance(p, list):
        for i, v in enumerate(p):
            yield from _flatten_params(f"{prefix}.{i}", v)


def _unflatten_params(flat: dict):
    root: dict = {}
    for k, v in flat.items():
        parts = k.split(".")
        node = root
        for p in parts[:-1]:
            if p.isdigit():
                p = int(p)
            if isinstance(node, list):
                while len(node) <= p:
                    node.append({})
                if not isinstance(node[p], (dict, list)):
                    node[p] = {}
                node = node[p]
            else:
                if p not in node:
                    node[p] = {}
                node = node[p]
        last = parts[-1]
        if last.isdigit():
            last = int(last)
        if isinstance(node, list):
            while len(node) <= last:
                node.append(None)
            node[last] = v
        else:
            node[last] = v
    return root


# ============================================================================
# Sampling (autoregressive, no KV-cache yet — single forward per token)
# ============================================================================

def _keras_sample(probs: np.ndarray, temperature: float = 1.0) -> int:
    """Match the 2025 `SmilesGEN_generator.Sample` semantics (T no-op by default)."""
    p = probs.astype(np.float64)
    p = np.log(p + 1e-10)
    if temperature != 1.0:
        p = p / max(temperature, 1e-6)
    ep = np.exp(p)
    p = ep / np.sum(ep)
    return int(np.argmax(np.random.multinomial(1, p, 1)))


def predict_gpt_batch(
    model,
    bpe: SmilesBPE,
    ncollect: int = 1000,
    batch_size: int = 64,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    seed_smiles: list = None,
) -> List[str]:
    """Generate `ncollect` SMILES by greedy/temperature-sampled rollout.

    Each rollout starts from BOS (or a BPE-encoded seed SMILES) and proceeds
    one token at a time until EOS or `max_new_tokens` reached. Returns a list
    of decoded SMILES (specials stripped).

    No KV-cache: each step re-runs the full prefix (~T tokens). For SMILES at
    T=30-60 this is fine — sub-second per molecule at typical model sizes.
    """
    context = model.cfg["context_length"]

    mols: List[str] = []
    while len(mols) < ncollect:
        b = min(batch_size, ncollect - len(mols))

        # Seed: BOS-only by default; if seed_smiles provided, sample from them.
        if seed_smiles:
            import random as _random
            seeds = _random.choices(seed_smiles, k=b)
            seed_ids = [
                [bpe.bos_id] + bpe.encode(s, add_special_tokens=False)
                for s in seeds
            ]
        else:
            seed_ids = [[bpe.bos_id] for _ in range(b)]

        # Pad-right to common starting length so we can batch
        max_seed_len = max(len(ids) for ids in seed_ids)
        x_np = np.full((b, max_seed_len), bpe.pad_id, dtype=np.int32)
        cur_len = np.zeros(b, dtype=np.int32)
        for i, ids in enumerate(seed_ids):
            x_np[i, :len(ids)] = ids
            cur_len[i] = len(ids)

        finished = np.zeros(b, dtype=bool)
        out_ids: List[List[int]] = [list(ids) for ids in seed_ids]

        for _ in range(max_new_tokens):
            # Stop if all done or context full
            if finished.all():
                break
            curT = int(cur_len.max())
            if curT >= context:
                break

            x_in = mx.array(x_np[:, :curT])
            logits = model(x_in, training=False)             # (b, curT, vocab)
            mx.eval(logits)
            # Per-row: take the logit at its current position-1
            for i in range(b):
                if finished[i]:
                    continue
                t_i = cur_len[i] - 1
                p_i = np.array(mx.softmax(logits[i, t_i], axis=-1))
                next_id = _keras_sample(p_i, temperature=temperature)
                # Grow x_np if needed
                if cur_len[i] >= x_np.shape[1]:
                    pad = np.full((b, 1), bpe.pad_id, dtype=np.int32)
                    x_np = np.concatenate([x_np, pad], axis=1)
                x_np[i, cur_len[i]] = next_id
                out_ids[i].append(next_id)
                cur_len[i] += 1
                if next_id == bpe.eos_id:
                    finished[i] = True

        # Decode each sequence (strips specials)
        for ids in out_ids:
            mols.append(bpe.decode(ids, skip_special_tokens=True))
            if len(mols) >= ncollect:
                break

    return mols
