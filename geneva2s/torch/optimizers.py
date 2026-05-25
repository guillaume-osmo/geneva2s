"""PyTorch optimizer zoo: AdamN, Muon, MuonN (AdamMuon "adamuonn" variant),
AdaMuon official, and a SplitOptimizer wrapper.

Vendored from mlxmolkit/tools/torch_optimizers.py (Guillaume Godin) so geneva2s
can run the same optimizer sweeps without pulling the full mlxmolkit dep just
for training. The "adamuonn" / "muon_vx" variants combine:

  - MuonN  (Newton-Schulz-orthogonalized nested-AdamN direction) for matrix params
  - AdamN  (3-beta nested first-moment Adam) for scalar params (bias, gain)

Background:
  Muon (Liu et al. 2024) orthogonalizes the gradient with quintic Newton-Schulz
  before applying it; AdamN adds a second-order moment chain on top of the Adam
  first moment. The combination ("adamuonn") tracks Adam-level scalar parameter
  behavior while Muon-treating the matrices — the regime that typically wins
  LLM/seq-model training on Apple Silicon, since matmul is the dominant cost.

Optimizer names supported by `build_optimizer`:

  - "adam"             — torch.optim.Adam (original geneva2s default)
  - "adamw"            — torch.optim.AdamW
  - "adamn"            — AdamN (3-beta nested Adam)
  - "muon"             — Muon for matrix params + AdamW for scalars (default Muon)
  - "adamuon"          — MuonN for matrix params + AdamW for scalars
  - "adamuonn"         — MuonN for matrix params + AdamN for scalars (recommended)
  - "adamuon_official" — AdaMuon official (Apache-2.0 port) + AdamW for scalars
  - "muon_vx"          — alias for adamuonn (legacy name)
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


class AdamN(torch.optim.Optimizer):
    """Adam-style optimizer with a nested first-moment average."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float, float] = (0.9, 0.1, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        bias_correction: str = "exact",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid eps: {eps}")
        if len(betas) != 3:
            raise ValueError("AdamN betas must be (beta_grad, beta_nested, beta_sq)")
        for name, value in zip(("beta_grad", "beta_nested", "beta_sq"), betas):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"invalid {name}: {value}")
        if bias_correction not in {"exact", "simple"}:
            raise ValueError("bias_correction must be 'exact' or 'simple'")
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(x) for x in betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "decoupled_weight_decay": bool(decoupled_weight_decay),
            "bias_correction": str(bias_correction),
        }
        super().__init__(params, defaults)

    @staticmethod
    def _nested_bias_correction(step: int, beta_grad: float, beta_nested: float) -> float:
        if step <= 0:
            return 0.0
        if abs(beta_nested - beta_grad) < 1.0e-12:
            beta = beta_grad
            return (1.0 - beta**step) - (1.0 - beta) * step * (beta**step)
        grad_pow = beta_grad**step
        nested_pow = beta_nested**step
        cross = (1.0 - beta_nested) * beta_grad * (nested_pow - grad_pow) / (beta_nested - beta_grad)
        return 1.0 - nested_pow - cross

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_grad, beta_nested, beta_sq = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decoupled_wd = group["decoupled_weight_decay"]
            correction_mode = group["bias_correction"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamN does not support sparse gradients")

                grad = param.grad.detach()
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["grad_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["nested_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["sq_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                grad_avg = state["grad_avg"]
                nested_avg = state["nested_avg"]
                sq_avg = state["sq_avg"]
                state["step"] += 1
                step = int(state["step"])

                if weight_decay:
                    if decoupled_wd:
                        param.mul_(1.0 - lr * weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)

                grad_avg.mul_(beta_grad).add_(grad, alpha=1.0 - beta_grad)
                nested_avg.mul_(beta_nested).add_(grad_avg, alpha=1.0 - beta_nested)
                sq_avg.mul_(beta_sq).addcmul_(grad, grad, value=1.0 - beta_sq)

                if correction_mode == "exact":
                    bias1 = max(self._nested_bias_correction(step, beta_grad, beta_nested), 1.0e-16)
                else:
                    bias1 = max(1.0 - beta_nested**step, 1.0e-16)
                bias2 = max(1.0 - beta_sq**step, 1.0e-16)
                direction = nested_avg / bias1
                denom = (sq_avg / bias2).sqrt().add_(eps)
                param.addcdiv_(direction, denom, value=-lr)

        return loss


def _zeropower_newton_schulz(update: torch.Tensor, steps: int, eps: float = 1.0e-7) -> torch.Tensor:
    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1).to(torch.float32)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / (matrix.norm() + eps)

    # Quintic Newton-Schulz coefficients used by common Muon implementations.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = matrix @ matrix.T
        matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix
    if transposed:
        matrix = matrix.T
    return matrix.reshape(original_shape).to(dtype=update.dtype)


def _match_rms(update: torch.Tensor, reference: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    ref_rms = reference.to(torch.float32).square().mean().sqrt()
    upd_rms = update.to(torch.float32).square().mean().sqrt()
    return update * (ref_rms / upd_rms.clamp_min(eps)).to(dtype=update.dtype)


def _spectral_transform_svd(
    update: torch.Tensor,
    *,
    mode: str,
    power: float,
    eps: float,
) -> torch.Tensor:
    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1)
    device = matrix.device
    dtype = matrix.dtype
    matrix32 = matrix.to(torch.float32)
    try:
        u, s, vh = torch.linalg.svd(matrix32, full_matrices=False)
    except RuntimeError:
        u_cpu, s_cpu, vh_cpu = torch.linalg.svd(matrix32.cpu(), full_matrices=False)
        u, s, vh = u_cpu.to(device), s_cpu.to(device), vh_cpu.to(device)

    if mode == "svd_orthogonal":
        values = torch.ones_like(s)
    elif mode == "svd_freon":
        values = s.clamp_min(eps).pow(1.0 - 2.0 * float(power))
    elif mode == "svd_inverse":
        values = s.clamp_min(eps).reciprocal()
        values = values / values.square().mean().sqrt().clamp_min(eps)
    elif mode == "svd_kaon":
        values = torch.rand_like(s).clamp_min(eps)
        values = values / values.square().mean().sqrt().clamp_min(eps)
    else:
        raise ValueError(f"unknown spectral mode: {mode!r}")

    transformed = (u * values.unsqueeze(0)) @ vh
    if mode != "svd_freon" or abs(float(power)) > 1.0e-12:
        transformed = _match_rms(transformed, matrix32)
    return transformed.reshape(original_shape).to(device=device, dtype=dtype)


class Muon(torch.optim.Optimizer):
    """Muon update for matrix-like tensors."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "nesterov": bool(nesterov),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                grad = param.grad.detach()
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = _zeropower_newton_schulz(update, ns_steps)
                rows = update.reshape(update.shape[0], -1).shape[0]
                cols = update.reshape(update.shape[0], -1).shape[1]
                update = update * math.sqrt(max(1.0, rows / max(1, cols)))
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)

        return loss


class MuonN(torch.optim.Optimizer):
    """Experimental Muon + AdamN hybrid for matrix-like tensors.

    The update direction is a nested AdamN-style first moment, optionally
    variance-normalized, then orthogonalized with the Muon Newton-Schulz map.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float, float] = (0.9, 0.1, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        bias_correction: str = "exact",
        variance_normalize: bool = True,
        mix: float = 1.0,
        spectral_mode: str = "ns",
        spectral_power: float = 0.5,
        spectral_eps: float = 1.0e-7,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid eps: {eps}")
        if len(betas) != 3:
            raise ValueError("MuonN betas must be (beta_grad, beta_nested, beta_sq)")
        for name, value in zip(("beta_grad", "beta_nested", "beta_sq"), betas):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"invalid {name}: {value}")
        if bias_correction not in {"exact", "simple"}:
            raise ValueError("bias_correction must be 'exact' or 'simple'")
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"invalid MuonN mix: {mix}")
        if spectral_mode not in {"ns", "svd_orthogonal", "svd_freon", "svd_inverse", "svd_kaon"}:
            raise ValueError(f"invalid MuonN spectral mode: {spectral_mode}")
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(x) for x in betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "bias_correction": str(bias_correction),
            "variance_normalize": bool(variance_normalize),
            "mix": float(mix),
            "spectral_mode": str(spectral_mode),
            "spectral_power": float(spectral_power),
            "spectral_eps": float(spectral_eps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_grad, beta_nested, beta_sq = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            correction_mode = group["bias_correction"]
            variance_normalize = group["variance_normalize"]
            mix = group["mix"]
            spectral_mode = group["spectral_mode"]
            spectral_power = group["spectral_power"]
            spectral_eps = group["spectral_eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("MuonN does not support sparse gradients")
                grad = param.grad.detach()
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["grad_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["nested_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["sq_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                grad_avg = state["grad_avg"]
                nested_avg = state["nested_avg"]
                sq_avg = state["sq_avg"]
                state["step"] += 1
                step = int(state["step"])

                grad_avg.mul_(beta_grad).add_(grad, alpha=1.0 - beta_grad)
                nested_avg.mul_(beta_nested).add_(grad_avg, alpha=1.0 - beta_nested)
                sq_avg.mul_(beta_sq).addcmul_(grad, grad, value=1.0 - beta_sq)

                if correction_mode == "exact":
                    bias1 = max(AdamN._nested_bias_correction(step, beta_grad, beta_nested), 1.0e-16)
                else:
                    bias1 = max(1.0 - beta_nested**step, 1.0e-16)
                direction = nested_avg / bias1
                if variance_normalize:
                    bias2 = max(1.0 - beta_sq**step, 1.0e-16)
                    direction = direction / ((sq_avg / bias2).sqrt().add(eps))
                if spectral_mode == "ns":
                    update = _zeropower_newton_schulz(direction, ns_steps, eps=float(spectral_eps))
                else:
                    update = _spectral_transform_svd(
                        direction,
                        mode=str(spectral_mode),
                        power=float(spectral_power),
                        eps=float(spectral_eps),
                    )
                if mix < 1.0:
                    raw_direction = _match_rms(direction, update)
                    update = update.mul(mix).add(raw_direction, alpha=1.0 - mix)
                rows = update.reshape(update.shape[0], -1).shape[0]
                cols = update.reshape(update.shape[0], -1).shape[1]
                update = update * math.sqrt(max(1.0, rows / max(1, cols)))
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)

        return loss


class AdaMuonOfficial(torch.optim.Optimizer):
    """Single-process AdaMuon following the official Apache-2.0 implementation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1.0e-8,
        nesterov: bool = True,
        scale: float = 0.2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "eps": float(eps),
            "nesterov": bool(nesterov),
            "scale": float(scale),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            nesterov = group["nesterov"]
            scale_coeff = group["scale"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdaMuonOfficial does not support sparse gradients")
                grad = param.grad.detach()
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(momentum).add_(grad)
                direction = grad.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
                original_shape = direction.shape
                if direction.ndim == 4:
                    direction = direction.view(len(direction), -1)
                direction = _zeropower_newton_schulz(torch.sign(direction), ns_steps).reshape(original_shape)

                flat_shape = direction.view(-1).shape
                if "v_buffer" not in state or state["v_buffer"].shape != flat_shape:
                    state["v_buffer"] = torch.zeros(flat_shape, dtype=direction.dtype, device=direction.device)
                v_buffer = state["v_buffer"]
                flat = direction.flatten()
                v_buffer.mul_(momentum).addcmul_(flat, flat, value=1.0 - momentum)
                flat = flat / v_buffer.sqrt().add(eps)
                direction = flat.view_as(direction)

                matrix = direction.reshape(direction.shape[0], -1)
                rows, cols = matrix.shape
                scale = scale_coeff * math.sqrt(max(1, min(rows, cols)) * max(rows, cols))
                scale = scale / (direction.norm() + eps)
                direction = direction * scale

                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(direction.to(dtype=param.dtype), alpha=-lr)

        return loss


class SplitOptimizer:
    """Small wrapper for using a matrix optimizer plus a scalar fallback."""

    def __init__(self, primary: torch.optim.Optimizer | None, fallback: torch.optim.Optimizer | None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.param_groups = []
        if primary is not None:
            self.param_groups.extend(primary.param_groups)
        if fallback is not None:
            self.param_groups.extend(fallback.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.primary is not None:
            self.primary.zero_grad(set_to_none=set_to_none)
        if self.fallback is not None:
            self.fallback.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if self.primary is not None:
            loss = self.primary.step(closure=closure)
        if self.fallback is not None:
            fallback_loss = self.fallback.step(closure=None)
            if loss is None:
                loss = fallback_loss
        return loss


def build_optimizer(model: torch.nn.Module, args: Any) -> torch.optim.Optimizer | SplitOptimizer:
    name = str(getattr(args, "optimizer", "adam")).lower()
    lr = float(getattr(args, "lr"))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    if name == "adam":
        # Matches the original geneva2s default: Adam(lr=3e-3, eps=1e-7).
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            eps=float(getattr(args, "adam_eps", 1.0e-7)),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamn":
        betas = tuple(float(x) for x in getattr(args, "adamn_betas", (0.9, 0.1, 0.999)))
        return AdamN(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=float(getattr(args, "adamn_eps", 1.0e-8)),
            weight_decay=weight_decay,
            decoupled_weight_decay=bool(getattr(args, "adamn_decoupled_weight_decay", True)),
            bias_correction=str(getattr(args, "adamn_bias_correction", "exact")),
        )
    if name in {
        "muon",
        "muon_adamw",
        "adamuon",
        "adamuon_adamw",
        "adamuon_official",
        "adamuon_official_adamn",
        "muonn",
        "muonn_adamn",
        "adamuonn",
        "muon_vx",
    }:
        matrix_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                matrix_params.append(param)
            else:
                other_params.append(param)
        if not matrix_params:
            primary = None
        elif name in {"adamuon_official", "adamuon_official_adamn"}:
            primary = AdaMuonOfficial(
                matrix_params,
                lr=lr,
                momentum=float(getattr(args, "muon_momentum", 0.95)),
                weight_decay=weight_decay,
                ns_steps=int(getattr(args, "muon_ns_steps", 5)),
                eps=float(getattr(args, "adamn_eps", 1.0e-8)),
                nesterov=bool(getattr(args, "muon_nesterov", True)),
                scale=float(getattr(args, "adamuon_scale", 0.2)),
            )
        elif name in {"adamuon", "adamuon_adamw", "muonn", "muonn_adamn", "adamuonn", "muon_vx"}:
            primary = MuonN(
                matrix_params,
                lr=lr,
                betas=tuple(float(x) for x in getattr(args, "adamn_betas", (0.9, 0.1, 0.999))),
                eps=float(getattr(args, "adamn_eps", 1.0e-8)),
                weight_decay=weight_decay,
                ns_steps=int(getattr(args, "muon_ns_steps", 5)),
                bias_correction=str(getattr(args, "adamn_bias_correction", "exact")),
                variance_normalize=bool(getattr(args, "muonn_variance_normalize", True)),
                mix=float(getattr(args, "muonn_mix", 1.0)),
                spectral_mode=str(getattr(args, "muonn_spectral_mode", "ns")),
                spectral_power=float(getattr(args, "muonn_spectral_power", 0.5)),
                spectral_eps=float(getattr(args, "muonn_spectral_eps", 1.0e-7)),
            )
        else:
            primary = Muon(
                matrix_params,
                lr=lr,
                momentum=float(getattr(args, "muon_momentum", 0.95)),
                weight_decay=weight_decay,
                ns_steps=int(getattr(args, "muon_ns_steps", 5)),
                nesterov=bool(getattr(args, "muon_nesterov", True)),
            )
        fallback_lr = lr * float(getattr(args, "muon_adamw_lr_ratio", 1.0))
        if not other_params:
            fallback = None
        elif name in {"muonn_adamn", "adamuonn", "adamuon_official_adamn", "muon_vx"}:
            fallback = AdamN(
                other_params,
                lr=fallback_lr,
                betas=tuple(float(x) for x in getattr(args, "adamn_betas", (0.9, 0.1, 0.999))),
                eps=float(getattr(args, "adamn_eps", 1.0e-8)),
                weight_decay=weight_decay,
                decoupled_weight_decay=bool(getattr(args, "adamn_decoupled_weight_decay", True)),
                bias_correction=str(getattr(args, "adamn_bias_correction", "exact")),
            )
        else:
            fallback = torch.optim.AdamW(other_params, lr=fallback_lr, weight_decay=weight_decay)
        return SplitOptimizer(primary, fallback)
    raise ValueError(f"unknown optimizer: {name!r}")
