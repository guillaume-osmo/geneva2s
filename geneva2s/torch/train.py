"""Training loop matching Keras: Adam(lr=3e-3, eps=1e-7), categorical CE.

fit_best optionally tracks the highest-validity epoch and restores those weights
(equivalent to the OnlineGenerator + restore_best_weights=True callback).

The optimizer can be swapped via the `optimizer` arg (default "adam" for
Keras-parity; pass "adamuonn" for the Muon-N + AdamN combo benchmarked on M3 Max).
"""
from __future__ import annotations

import copy
import os
import time
import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..tokenizer import CharTokenizer
from ..utils import sanity_check
from .generate import predict_batch_seeds
from .optimizers import build_optimizer


def _make_optimizer(model, *, optimizer: str, lr: float, weight_decay: float = 0.0,
                    extra: dict | None = None):
    """Wrap build_optimizer with a SimpleNamespace shim (args-style API)."""
    cfg = {"optimizer": optimizer, "lr": lr, "weight_decay": weight_decay}
    if extra:
        cfg.update(extra)
    return build_optimizer(model, types.SimpleNamespace(**cfg))


def fit(
    model: nn.Module,
    X,
    y,
    device: torch.device,
    num_epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    save_path: str = None,
    verbose: bool = True,
):
    x_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    loader = DataLoader(
        TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    model.to(device)
    opt = _make_optimizer(model, optimizer=optimizer, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for ep in range(num_epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        avg = total / n
        history.append(avg)
        if verbose:
            print(f"epoch {ep:3d}: loss={avg:.4f} time={time.time()-t0:.1f}s")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return history


def fit_best(
    model: nn.Module,
    X,
    y,
    device: torch.device,
    tokenizer: CharTokenizer,
    text_for_seeds: str,
    num_epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    check_every: int = 5,
    ncollect_check: int = 500,
    ncopies_check: int = 20,
    save_path: str = None,
    verbose: bool = True,
):
    """Like fit() but every check_every epochs runs a SanityCheck-validity
    eval and restores best-validity weights at the end."""
    x_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    loader = DataLoader(
        TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    model.to(device)
    opt = _make_optimizer(model, optimizer=optimizer, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = -1.0, None, -1
    history = []

    for ep in range(num_epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        avg = total / n

        msg_extra = ""
        if (ep + 1) % check_every == 0 or ep == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                gen = predict_batch_seeds(
                    model, tokenizer, text_for_seeds,
                    ncollect=ncollect_check, ncopies=ncopies_check, device=device,
                )
            val = sum(1 for s in gen if sanity_check(s)) / max(1, len(gen))
            history.append((ep, avg, val))
            if val > best_val:
                best_val, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), ep
                msg_extra = f" valid_sc={100*val:.1f}% NEW BEST"
            else:
                msg_extra = f" valid_sc={100*val:.1f}%"

        if verbose:
            print(f"epoch {ep:3d}: loss={avg:.4f} time={time.time()-t0:.1f}s{msg_extra}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"=== restored best weights from epoch {best_epoch}: "
                  f"sanitycheck={100*best_val:.2f}% ===")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return history, best_epoch, best_val
