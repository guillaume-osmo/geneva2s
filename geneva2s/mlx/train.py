"""MLX training loop matching the PyTorch and Keras Trainer setup.

Adam(lr=3e-3, eps=1e-7), categorical cross-entropy on next-char prediction.
Works with any GenevaBiLSTMMLX{,Fused,Metal} model — they're all differentiable
end-to-end (Metal variants via the VJP kernel + mx.custom_function wiring in
mlx-addons).

`fit_best` adds validity-driven best-checkpoint tracking (equivalent to the
Keras OnlineGenerator + restore_best_weights=True callback).
"""
from __future__ import annotations

import copy
import os
import time

try:
    import mlx.core as mx
    import mlx.nn as mlxnn
    import mlx.optimizers as optim
except ImportError as e:
    raise ImportError("pip install mlx") from e
import numpy as np

from ..tokenizer import CharTokenizer
from ..utils import sanity_check
from .generate import predict_batch_seeds


def _loss_fn(model, x, y):
    logits = model(x)  # (B, vocab)
    return mlxnn.losses.cross_entropy(logits, y, reduction="mean")


def _make_compiled_step(model, optimizer):
    """Build an mx.compile'd train step. Captures model + optimizer state
    explicitly so MLX can fuse the whole forward+backward+update path."""
    loss_and_grad_fn = mlxnn.value_and_grad(model, _loss_fn)
    state = [model.state, optimizer.state]

    def step_inner(x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    return mx.compile(step_inner, inputs=state, outputs=state), state


def fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-3,
    save_path: str = None,
    verbose: bool = True,
    use_compile: bool = True,
):
    """Adam training matching Keras Trainer.compile(Adam(lr=3e-3, eps=1e-7)).

    use_compile=True (default) wraps the forward+backward+update step in
    `mx.compile` for ~1.3-1.4× speedup. Disable for debugging.
    """
    optimizer = optim.Adam(learning_rate=lr, eps=1e-7)
    if use_compile:
        step, state = _make_compiled_step(model, optimizer)
    else:
        loss_and_grad_fn = mlxnn.value_and_grad(model, _loss_fn)
        state = [model.state, optimizer.state]
        def step(xb, yb):
            loss, grads = loss_and_grad_fn(model, xb, yb)
            optimizer.update(model, grads)
            return loss

    n = X.shape[0]
    history = []
    for ep in range(num_epochs):
        t0 = time.time()
        perm = np.random.permutation(n)
        total, n_seen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = mx.array(X[idx])
            yb = mx.array(y[idx])
            loss = step(xb, yb)
            mx.eval(state)
            total += float(loss) * xb.shape[0]
            n_seen += xb.shape[0]
        avg = total / n_seen
        history.append(avg)
        if verbose:
            print(f"epoch {ep:3d}: loss={avg:.4f} time={time.time() - t0:.1f}s")
    if save_path:
        save_state(model, save_path)
    return history


def fit_best(
    model,
    X: np.ndarray,
    y: np.ndarray,
    tokenizer: CharTokenizer,
    text_for_seeds: str,
    num_epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-3,
    check_every: int = 5,
    ncollect_check: int = 500,
    ncopies_check: int = 20,
    save_path: str = None,
    verbose: bool = True,
    use_compile: bool = True,
):
    """Like fit() but restores best-validity weights at the end (Keras
    OnlineGenerator + restore_best_weights=True equivalent)."""
    optimizer = optim.Adam(learning_rate=lr, eps=1e-7)
    if use_compile:
        step, state = _make_compiled_step(model, optimizer)
    else:
        loss_and_grad_fn = mlxnn.value_and_grad(model, _loss_fn)
        state = [model.state, optimizer.state]
        def step(xb, yb):
            loss, grads = loss_and_grad_fn(model, xb, yb)
            optimizer.update(model, grads)
            return loss

    n = X.shape[0]
    history = []
    best_val, best_params, best_epoch = -1.0, None, -1

    for ep in range(num_epochs):
        t0 = time.time()
        perm = np.random.permutation(n)
        total, n_seen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = mx.array(X[idx])
            yb = mx.array(y[idx])
            loss = step(xb, yb)
            mx.eval(state)
            total += float(loss) * xb.shape[0]
            n_seen += xb.shape[0]
        avg = total / n_seen

        msg_extra = ""
        if (ep + 1) % check_every == 0 or ep == num_epochs - 1:
            mols = predict_batch_seeds(
                model, tokenizer, text_for_seeds,
                ncollect=ncollect_check, ncopies=ncopies_check,
            )
            val = sum(1 for s in mols if sanity_check(s)) / max(1, len(mols))
            history.append((ep, avg, val))
            if val > best_val:
                best_val, best_epoch = val, ep
                best_params = copy.deepcopy(model.parameters())
                msg_extra = f" valid_sc={100 * val:.1f}% NEW BEST"
            else:
                msg_extra = f" valid_sc={100 * val:.1f}%"

        if verbose:
            print(f"epoch {ep:3d}: loss={avg:.4f} time={time.time() - t0:.1f}s{msg_extra}")

    if best_params is not None:
        model.update(best_params)
        if verbose:
            print(
                f"=== restored best weights from epoch {best_epoch}: "
                f"sanitycheck={100 * best_val:.2f}% ==="
            )

    if save_path:
        save_state(model, save_path)
    return history, best_epoch, best_val


# ============================================================================
# State I/O — save/load model weights cross-framework
# ============================================================================

def save_state(model, path: str) -> None:
    """Save model parameters. Format depends on extension:
      .safetensors → MLX-native (preferred for MLX-only round-trips)
      .npz         → numpy archive (loadable from any framework)
      .pt          → PyTorch state_dict format (loadable by GenevaBiLSTM.load_state_dict)
    """
    from mlx.utils import tree_flatten
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = dict(tree_flatten(model.parameters()))

    if path.endswith(".safetensors"):
        mx.save_safetensors(path, flat)
    elif path.endswith(".npz"):
        np.savez(path, **{k: np.array(v) for k, v in flat.items()})
    elif path.endswith(".pt"):
        import torch
        sd = _mlx_to_pt_state_dict(flat, model)
        torch.save(sd, path)
    else:
        raise ValueError(f"Unknown save extension for path: {path}")


def _mlx_to_pt_state_dict(flat_params: dict, model) -> dict:
    """Map MLX param names to PyTorch GenevaBiLSTM state_dict keys.

    MLX:                                    PyTorch:
      emb_fwd.Wx                             embedding.weight_ih_l0
      emb_fwd.Wh                             embedding.weight_hh_l0
      emb_fwd.bias                           embedding.bias_ih_l0 + bias_hh_l0  (split: bias_ih = full, bias_hh = 0)
      emb_bwd.{Wx,Wh,bias}                   embedding.weight_*_reverse
      branches.N.{Wx,Wh,bias}                branches.N.weight_*  (N=0..3)
      head.weight                            head.weight
      head.bias                              head.bias
    """
    import torch

    def to_torch(arr):
        return torch.from_numpy(np.array(arr))

    sd = {}
    H1 = model.hidden[0] if hasattr(model, "hidden") else 128
    H2 = model.hidden[1] if hasattr(model, "hidden") else 64
    n_branches = model.n_branches if hasattr(model, "n_branches") else 4

    # biLSTM forward
    if "emb_fwd.Wx" in flat_params:
        sd["embedding.weight_ih_l0"] = to_torch(flat_params["emb_fwd.Wx"])
        sd["embedding.weight_hh_l0"] = to_torch(flat_params["emb_fwd.Wh"])
        sd["embedding.bias_ih_l0"] = to_torch(flat_params["emb_fwd.bias"])
        sd["embedding.bias_hh_l0"] = torch.zeros(4 * H1)
        sd["embedding.weight_ih_l0_reverse"] = to_torch(flat_params["emb_bwd.Wx"])
        sd["embedding.weight_hh_l0_reverse"] = to_torch(flat_params["emb_bwd.Wh"])
        sd["embedding.bias_ih_l0_reverse"] = to_torch(flat_params["emb_bwd.bias"])
        sd["embedding.bias_hh_l0_reverse"] = torch.zeros(4 * H1)
    # 4 branches (Module list)
    for i in range(n_branches):
        for src, dst in [("Wx", "weight_ih_l0"), ("Wh", "weight_hh_l0"), ("bias", "bias_ih_l0")]:
            key = f"branches.{i}.{src}"
            if key in flat_params:
                sd[f"branches.{i}.{dst}"] = to_torch(flat_params[key])
        sd[f"branches.{i}.bias_hh_l0"] = torch.zeros(4 * H2)
    # Head
    sd["head.weight"] = to_torch(flat_params["head.weight"])
    sd["head.bias"] = to_torch(flat_params["head.bias"])
    return sd


def load_state(model, path: str) -> None:
    """Load model parameters from .safetensors / .npz / .pt path."""
    if path.endswith(".safetensors"):
        weights = mx.load(path)
        # Group by '.' to nested dict for tree_unflatten
        from mlx.utils import tree_unflatten
        model.update(tree_unflatten(list(weights.items())))
    elif path.endswith(".npz"):
        d = np.load(path)
        from mlx.utils import tree_unflatten
        items = [(k, mx.array(d[k])) for k in d.files]
        model.update(tree_unflatten(items))
    elif path.endswith(".pt"):
        # Use the existing PT-state-dict loader on the model
        model.load_pt_checkpoint(path)
    else:
        raise ValueError(f"Unknown load extension: {path}")
