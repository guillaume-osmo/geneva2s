"""SMILES-aware BPE tokenizer using the HuggingFace `tokenizers` Rust backend.

The 2025-era BiLSTM uses a 27-character vocab (`CharTokenizer`). For the GPT
path we want a larger, chemistry-aware token alphabet so the model sees
ring-closure / bracket / branch motifs as single units rather than 3-6 chars.

Trained on the project's training corpus (`data/chembl_9k_organic.smi` by
default; 9k unique SMILES). Output is a standard HF tokenizer JSON file
that can be moved across machines and reloaded with one line.

Special tokens:
    <pad>  id=0   (left-padding for batched generation)
    <bos>  id=1   (start of sequence)
    <eos>  id=2   (end of sequence)
    <unk>  id=3   (fallback for unseen characters)

Typical vocab size: 1024. Each SMILES of ~30-60 chars becomes ~15-30 tokens.

CLI:
    # Train + save (default ref + 1024 vocab):
    python -m geneva2s.smiles_tokenizer train \\
        --corpus data/chembl_9k_organic.smi \\
        --vocab-size 1024 \\
        --out data/smiles_bpe_v1.json

    # Inspect:
    python -m geneva2s.smiles_tokenizer info --tokenizer data/smiles_bpe_v1.json

    # Round-trip test on a few SMILES:
    python -m geneva2s.smiles_tokenizer roundtrip \\
        --tokenizer data/smiles_bpe_v1.json \\
        --smi "CCO" --smi "c1ccc(cc1)O" --smi "CC(=O)Oc1ccccc1C(=O)O"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS: tuple = (PAD, BOS, EOS, UNK)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# Single-character SMILES alphabet that we want to GUARANTEE is in the vocab,
# even if rare. Trained BPE typically picks all of these up from any corpus,
# but listing them as `initial_alphabet` keeps the tokenizer well-defined on
# OOV-tolerant inputs (e.g. molecules with rare chirality markers).
_SMILES_BASE_ALPHABET = list(
    "CFLRINOSPBHcfnosbpharu"          # atoms + aromatic (extra letters tolerated)
    "0123456789%"                      # ring-closure digits + 2-digit `%nn`
    "=#$:/\\"                          # bond order + cis/trans
    "()[]"                             # branches + brackets
    "+-@*"                             # charges + chirality + wildcard
)


# Atom-shape regex tokens that should never be split by BPE merges.
# This is the Schwaller-style SMILES atom matcher; we feed it as a
# pre-tokenizer regex so the merges learn meaningful atom-chunks.
SMILES_REGEX = (
    r"\[[^\]]+\]"           # bracketed atom: [nH], [C@@H], [O-]
    r"|Br|Cl"               # 2-char halogens
    r"|[A-Za-z]"            # single-char atom (uppercase + aromatic lowercase)
    r"|%[0-9]{2}"           # 2-digit ring closure
    r"|[0-9]"               # 1-digit ring closure
    r"|."                   # everything else, one char at a time
)


class SmilesBPE:
    """Wrapper around HF `tokenizers.Tokenizer` configured for SMILES.

    Construction either trains from scratch (`train_from_corpus`) or loads
    from a saved file (`load`). The wrapper carries SMILES-specific
    pre-tokenization (atom regex) and special-token ids so downstream code
    doesn't need to know HF tokenizer internals.
    """

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    # ------------------------------------------------------------------ load/save

    @classmethod
    def load(cls, path: str) -> "SmilesBPE":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(path))
        return cls(tok)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    # ------------------------------------------------------------------ train

    @classmethod
    def train_from_corpus(
        cls,
        corpus_paths: Sequence[str],
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> "SmilesBPE":
        """Train a BPE on one or more raw `.smi` files (one SMILES per line)."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.decoders import BPEDecoder

        # No pre-tokenizer: SMILES contains no whitespace and we want BPE to
        # learn merges across atoms (motifs like `c1cc`, `C(=O)`, `OC1`).
        # Pre-splitting at atom boundaries would prevent any merge.
        tok = Tokenizer(BPE(unk_token=UNK))
        tok.decoder = BPEDecoder()  # concatenate ids without inserting spaces
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIALS),
            initial_alphabet=_SMILES_BASE_ALPHABET,
            show_progress=False,
        )

        def _iter():
            for p in corpus_paths:
                with open(p) as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            yield s

        tok.train_from_iterator(_iter(), trainer=trainer)

        # Sanity: assert special-token ids are at their reserved positions.
        for token, expected_id in zip(SPECIALS, (PAD_ID, BOS_ID, EOS_ID, UNK_ID)):
            actual = tok.token_to_id(token)
            if actual != expected_id:
                raise RuntimeError(
                    f"special token {token!r} got id {actual}, expected {expected_id}"
                )
        return cls(tok)

    # ------------------------------------------------------------------ encode / decode

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, smi: str, add_special_tokens: bool = True) -> List[int]:
        """Return token ids for one SMILES. Adds <bos>...<eos> by default."""
        ids = self._tok.encode(smi).ids
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_batch(
        self,
        smiles: Sequence[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        encs = self._tok.encode_batch(list(smiles))
        if add_special_tokens:
            return [[self.bos_id, *e.ids, self.eos_id] for e in encs]
        return [e.ids for e in encs]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        ids_batch: Sequence[Sequence[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return self._tok.decode_batch(
            [list(ids) for ids in ids_batch],
            skip_special_tokens=skip_special_tokens,
        )

    # ------------------------------------------------------------------ batching helpers

    def pad_to_length(
        self,
        ids: Sequence[int],
        length: int,
        pad_left: bool = False,
    ) -> List[int]:
        """Pad or truncate to `length`. `pad_left=True` is what you want for
        autoregressive decoding so positions stay aligned with new tokens."""
        if len(ids) >= length:
            return list(ids[-length:]) if pad_left else list(ids[:length])
        padding = [self.pad_id] * (length - len(ids))
        return padding + list(ids) if pad_left else list(ids) + padding

    def pad_batch(
        self,
        ids_batch: Sequence[Sequence[int]],
        length: Optional[int] = None,
        pad_left: bool = False,
    ) -> List[List[int]]:
        """Pad a batch to common length. Length = max if not provided."""
        if length is None:
            length = max(len(ids) for ids in ids_batch) if ids_batch else 0
        return [self.pad_to_length(ids, length, pad_left=pad_left) for ids in ids_batch]


# ============================================================================
# CLI
# ============================================================================

def _cmd_train(args):
    print(f"Training SMILES BPE: vocab={args.vocab_size}, "
          f"min_freq={args.min_frequency} on {len(args.corpus)} corpus file(s)")
    tok = SmilesBPE.train_from_corpus(
        args.corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tok.save(args.out)
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  saved → {args.out}")
    # quick round-trip on first 3 lines of the first corpus file
    with open(args.corpus[0]) as f:
        samples = [next(f).strip() for _ in range(3)]
    for smi in samples:
        ids = tok.encode(smi)
        back = tok.decode(ids)
        ok = "OK" if back.replace(" ", "") == smi.replace(" ", "") else "DIFF"
        print(f"  {ok}: {smi!r} → {len(ids)} ids → {back!r}")
    return 0


def _cmd_info(args):
    tok = SmilesBPE.load(args.tokenizer)
    print(f"vocab_size: {tok.vocab_size}")
    print(f"specials  : pad={tok.pad_id} bos={tok.bos_id} eos={tok.eos_id} unk={tok.unk_id}")
    return 0


def _cmd_roundtrip(args):
    tok = SmilesBPE.load(args.tokenizer)
    for smi in args.smi:
        ids = tok.encode(smi)
        back = tok.decode(ids)
        print(f"  {smi!r}")
        print(f"    → ids ({len(ids)}): {ids}")
        print(f"    → decode: {back!r}")
        print(f"    → match (stripped spaces): {back.replace(' ','') == smi.replace(' ','')}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Train/inspect a SMILES-aware BPE tokenizer.")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train BPE on one or more .smi corpora")
    pt.add_argument("--corpus", action="append", required=True,
                    help="Corpus file (one SMILES per line). Pass multiple --corpus to merge.")
    pt.add_argument("--vocab-size", type=int, default=1024)
    pt.add_argument("--min-frequency", type=int, default=2)
    pt.add_argument("--out", required=True, help="Output tokenizer JSON path")
    pt.set_defaults(fn=_cmd_train)

    pi = sub.add_parser("info", help="Print tokenizer summary")
    pi.add_argument("--tokenizer", required=True)
    pi.set_defaults(fn=_cmd_info)

    pr = sub.add_parser("roundtrip", help="Encode + decode a few SMILES strings")
    pr.add_argument("--tokenizer", required=True)
    pr.add_argument("--smi", action="append", required=True)
    pr.set_defaults(fn=_cmd_roundtrip)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
