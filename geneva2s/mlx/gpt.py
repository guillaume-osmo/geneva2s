"""GPT-style decoder for SMILES, with optional MLA / DSA attention variants.

Four model classes — all share the same `(B, T) int -> (B, T, vocab) float`
contract as `GenevaBiLSTMMLX*`, so they're drop-in for `fit_best` /
`predict_batch_seeds`:

    GenevaGPTMLX        vanilla SDPA decoder (modern stack: SDPA + RoPE + RMSNorm + SwiGLU)
    GenevaGPTMLXMLA     SDPA with Multi-Head Latent Attention
                        (DeepSeek-V2 low-rank K/V compression)
    GenevaGPTMLXDSA     SDPA with DeepSeek Sparse Attention masking
                        (Lightning Indexer + top-k selection, additive mask -> SDPA)
    GenevaGPTMLXMLADSA  Both MLA and DSA in the same block

Architecture is config-driven (`GPT_CFG_DEFAULT`); attention kwargs
(`latent_dim` for MLA, `index_n_heads` / `index_head_dim` / `topk` for DSA)
live in the same dict and are ignored by classes that don't use them.

Modern-MLX choices (informed by autoresearch-mlx, modded-nanogpt):
- `mx.fast.scaled_dot_product_attention` (fused Metal kernel)
- `nn.RoPE(head_dim, traditional=True, base=10000)` (no learned position embed)
- RMSNorm (cheaper than LayerNorm, no beta/gamma needed)
- SwiGLU MLP (better than GELU at same FLOPs)
- Dropout gated on the `training` arg (MLX nn.Dropout doesn't auto-skip at eval)

Reference implementations (PyTorch) ported to MLX:
- DSA: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa
- MLA: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ============================================================================
# Default config (~3M params — appropriate for 9k-corpus SMILES with BPE 1024)
# ============================================================================

GPT_CFG_DEFAULT = {
    "vocab_size": 1024,        # matches data/smiles_bpe_v1.json
    "context_length": 128,     # plenty (~60 BPE tokens for the longest SMILES)
    "emb_dim": 192,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "mlp_expansion": 4.0,      # SwiGLU hidden = 4 * emb_dim
    # MLA-specific
    "latent_dim": 32,          # compress K/V to rank 32 (vs d_out=192 -> 6x cache shrink)
    # DSA-specific
    "index_n_heads": 4,
    "index_head_dim": 32,
    "topk": 32,                # at seqlen 60 BPE tokens, top-32 covers > half
}


# ============================================================================
# Helpers
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm over the last dim. Single learned scale; no bias/center.

    Matches the autoresearch-mlx / modded-nanogpt style.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.scale * x * mx.reciprocal(rms)


class SwiGLUMLP(nn.Module):
    """SwiGLU: out = W2(silu(W1 x) * W3 x). Standard in modern decoders."""

    def __init__(self, emb_dim: int, expansion: float = 4.0):
        super().__init__()
        hidden = max(1, int(round(expansion * emb_dim)))
        self.w1 = nn.Linear(emb_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, emb_dim, bias=False)
        self.w3 = nn.Linear(emb_dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


def _additive_causal_mask(t: int) -> mx.array:
    """Additive causal mask, shape (t, t). 0 below+diag, -inf strictly above."""
    idx = mx.arange(t)
    blocked = idx[None, :] > idx[:, None]
    return mx.where(blocked, mx.array(-mx.inf, dtype=mx.float32),
                    mx.array(0.0, dtype=mx.float32))


# ============================================================================
# Standard SDPA causal self-attention (with RoPE + QK-norm)
# ============================================================================

class CausalSelfAttentionSDPA(nn.Module):
    """Multi-head causal self-attention via `mx.fast.scaled_dot_product_attention`.

    Includes RoPE on Q/K so we don't need a learned position embedding.
    """

    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        b, t, _ = x.shape
        # Project + reshape to (b, n_heads, t, head_dim)
        q = self.W_q(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.W_k(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.W_v(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal",
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.emb_dim)
        out = self.W_o(out)
        return self.drop(out) if training else out


# ============================================================================
# Multi-Head Latent Attention (MLA, DeepSeek-V2)
# ============================================================================

class MultiHeadLatentAttention(nn.Module):
    """Low-rank K/V compression via a `latent_dim`-d shared code.

    At inference, the KV cache (when added) stores only the latent codes,
    giving an `emb_dim / latent_dim` shrink. Forward shares the SDPA path
    with the standard attention so we keep the fused Metal kernel.
    """

    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float,
                 latent_dim: Optional[int] = None):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, emb_dim // 8)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_dkv = nn.Linear(emb_dim, self.latent_dim, bias=False)   # down -> latent
        self.W_uk = nn.Linear(self.latent_dim, emb_dim, bias=False)    # up -> K
        self.W_uv = nn.Linear(self.latent_dim, emb_dim, bias=False)    # up -> V
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        b, t, _ = x.shape

        q_all = self.W_q(x)
        latent = self.W_dkv(x)
        k_all = self.W_uk(latent)
        v_all = self.W_uv(latent)

        q = q_all.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k_all.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_all.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal",
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.emb_dim)
        out = self.W_o(out)
        return self.drop(out) if training else out


# ============================================================================
# Lightning Indexer + DeepSeek Sparse Attention (DSA, DeepSeek-V3.2)
# ============================================================================

class LightningIndexer(nn.Module):
    """Per-query scorer that picks the top-k past tokens for sparse attention.

    Score for query t and candidate s is a gated sum of per-head dot products:
        I[t, s] = sum_j (w[t, j] / sqrt(H_I)) * ReLU(q[t, j] · k[s] / sqrt(d_I))

    Returns an *additive mask* of shape (b, 1, t, t) with 0 on selected
    (keep) positions and -inf on rejected positions. The caller adds the
    causal mask separately.
    """

    def __init__(self, emb_dim: int, index_n_heads: int, index_head_dim: int):
        super().__init__()
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.W_q = nn.Linear(emb_dim, index_n_heads * index_head_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, index_head_dim, bias=False)
        self.W_weights = nn.Linear(emb_dim, index_n_heads, bias=False)
        self.scale = 1.0 / math.sqrt(index_head_dim)
        self.head_scale = 1.0 / math.sqrt(index_n_heads)

    def topk_mask(self, x: mx.array, topk: int) -> mx.array:
        """Return additive mask (b, 1, t, t) with 0 at top-k positions, -inf elsewhere."""
        b, t, _ = x.shape

        # (b, t, H_I, head_dim) — per-head indexer queries
        q = self.W_q(x).reshape(b, t, self.index_n_heads, self.index_head_dim)
        # (b, t, head_dim) — shared key
        k = self.W_k(x)

        # raw[b, t, h, s] = sum_d q[b, t, h, d] * k[b, s, d]
        raw = mx.einsum("bthd,bsd->bths", q, k) * self.scale
        raw = mx.maximum(raw, 0.0)                                   # ReLU

        w = self.W_weights(x) * self.head_scale                      # (b, t, H_I)
        scores = mx.einsum("bth,bths->bts", w, raw)                  # (b, t, t)

        # Apply causal mask BEFORE top-k so we never select future positions.
        causal_neg = _additive_causal_mask(t)                        # (t, t)
        scores = scores + causal_neg[None, :, :]

        k_val = min(topk, t)
        # mx.argpartition returns indices that partition; top_k = those above
        # partition. We use argsort descending and slice — simpler, plenty fast
        # at our sequence lengths.
        order = mx.argsort(-scores, axis=-1)                         # (b, t, t)
        topk_idx = order[:, :, :k_val]                               # (b, t, k_val)

        # Scatter to a boolean keep mask (b, t, t)
        s_range = mx.arange(t).reshape(1, 1, t, 1)                   # (1, 1, t, 1)
        idx = topk_idx.reshape(b, t, 1, k_val)                       # (b, t, 1, k_val)
        keep = (s_range == idx).any(axis=-1)                         # (b, t, t)

        # Additive mask: 0 on keep, -inf on reject. Broadcasts over heads.
        return mx.where(keep[:, None, :, :],
                        mx.array(0.0, dtype=mx.float32),
                        mx.array(-mx.inf, dtype=mx.float32))


class CausalSelfAttentionSDPA_DSA(nn.Module):
    """SDPA attention with DSA-derived sparse mask layered on top of causal.

    Uses the same `mx.fast.scaled_dot_product_attention` fused kernel as
    the standard variant; the DSA selection is encoded as an additive mask
    that gets summed with the causal mask and passed to SDPA.
    """

    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float,
                 index_n_heads: int = 4, index_head_dim: int = 32, topk: int = 32):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.topk = topk
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)
        self.indexer = LightningIndexer(emb_dim, index_n_heads, index_head_dim)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        b, t, _ = x.shape

        q = self.W_q(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.W_k(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.W_v(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # Causal + DSA top-k → additive mask (b, 1, t, t)
        causal = _additive_causal_mask(t)[None, None, :, :]          # (1, 1, t, t)
        sparse = self.indexer.topk_mask(x, self.topk)                # (b, 1, t, t)
        mask = causal + sparse

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.emb_dim)
        out = self.W_o(out)
        return self.drop(out) if training else out


class CausalSelfAttentionSDPA_MLA_DSA(nn.Module):
    """MLA K/V compression + DSA mask layered on the same SDPA call."""

    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float,
                 latent_dim: Optional[int] = None,
                 index_n_heads: int = 4, index_head_dim: int = 32, topk: int = 32):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, emb_dim // 8)
        self.topk = topk
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_dkv = nn.Linear(emb_dim, self.latent_dim, bias=False)
        self.W_uk = nn.Linear(self.latent_dim, emb_dim, bias=False)
        self.W_uv = nn.Linear(self.latent_dim, emb_dim, bias=False)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)
        self.indexer = LightningIndexer(emb_dim, index_n_heads, index_head_dim)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        b, t, _ = x.shape

        q_all = self.W_q(x)
        latent = self.W_dkv(x)
        k_all = self.W_uk(latent)
        v_all = self.W_uv(latent)

        q = q_all.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k_all.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_all.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        causal = _additive_causal_mask(t)[None, None, :, :]
        sparse = self.indexer.topk_mask(x, self.topk)
        mask = causal + sparse

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.emb_dim)
        out = self.W_o(out)
        return self.drop(out) if training else out


# ============================================================================
# Transformer block + GPT shell
# ============================================================================

def _make_attention(cfg: dict, variant: str) -> nn.Module:
    common = dict(emb_dim=cfg["emb_dim"], n_heads=cfg["n_heads"],
                  drop_rate=cfg["drop_rate"])
    if variant == "mha":
        return CausalSelfAttentionSDPA(**common)
    if variant == "mla":
        return MultiHeadLatentAttention(latent_dim=cfg.get("latent_dim"), **common)
    if variant == "dsa":
        return CausalSelfAttentionSDPA_DSA(
            index_n_heads=cfg.get("index_n_heads", 4),
            index_head_dim=cfg.get("index_head_dim", 32),
            topk=cfg.get("topk", 32),
            **common,
        )
    if variant == "mla_dsa":
        return CausalSelfAttentionSDPA_MLA_DSA(
            latent_dim=cfg.get("latent_dim"),
            index_n_heads=cfg.get("index_n_heads", 4),
            index_head_dim=cfg.get("index_head_dim", 32),
            topk=cfg.get("topk", 32),
            **common,
        )
    raise ValueError(f"Unknown attention variant {variant!r}")


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → attn → residual → RMSNorm → SwiGLU → residual."""

    def __init__(self, cfg: dict, variant: str):
        super().__init__()
        self.ln1 = RMSNorm(cfg["emb_dim"])
        self.attn = _make_attention(cfg, variant)
        self.ln2 = RMSNorm(cfg["emb_dim"])
        self.ff = SwiGLUMLP(cfg["emb_dim"], cfg.get("mlp_expansion", 4.0))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        h = self.attn(self.ln1(x), training=training)
        x = x + (self.drop(h) if training else h)
        h = self.ff(self.ln2(x))
        x = x + (self.drop(h) if training else h)
        return x


class _GenevaGPTBase(nn.Module):
    """Decoder-only Transformer shell parameterised by an attention variant.

    Concrete subclasses (`GenevaGPTMLX`, `GenevaGPTMLXMLA`, ...) just pin
    the `variant` string. Same `__call__((b, t) int32) -> (b, t, vocab) float`
    signature as `GenevaBiLSTMMLX*` so `fit_best` and `predict_batch_seeds`
    keep working unchanged (no positional embedding needed — RoPE is applied
    inside the attention block).
    """

    variant: str = "mha"

    def __init__(self, vocab_size: int, cfg: Optional[dict] = None):
        super().__init__()
        cfg = {**GPT_CFG_DEFAULT, **(cfg or {})}
        cfg["vocab_size"] = vocab_size
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = [TransformerBlock(cfg, self.variant) for _ in range(cfg["n_layers"])]
        self.ln_f = RMSNorm(cfg["emb_dim"])
        self.lm_head = nn.Linear(cfg["emb_dim"], vocab_size, bias=False)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        b, t = x.shape
        if t > self.cfg["context_length"]:
            raise ValueError(
                f"sequence length {t} > context_length {self.cfg['context_length']}"
            )
        h = self.tok_emb(x)
        h = self.drop(h) if training else h
        for blk in self.blocks:
            h = blk(h, training=training)
        h = self.ln_f(h)
        return self.lm_head(h)


class GenevaGPTMLX(_GenevaGPTBase):
    variant = "mha"


class GenevaGPTMLXMLA(_GenevaGPTBase):
    variant = "mla"


class GenevaGPTMLXDSA(_GenevaGPTBase):
    variant = "dsa"


class GenevaGPTMLXMLADSA(_GenevaGPTBase):
    variant = "mla_dsa"


# Convenience map for CLI dispatch in main.py
GPT_VARIANTS = {
    "gpt": GenevaGPTMLX,
    "gpt-mla": GenevaGPTMLXMLA,
    "gpt-dsa": GenevaGPTMLXDSA,
    "gpt-mla-dsa": GenevaGPTMLXMLADSA,
}
