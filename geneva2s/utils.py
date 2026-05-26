"""SMILES validity / canonicalization / augmentation utilities (rdkit-based)."""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles

RDLogger.DisableLog("rdApp.*")


def canonicalize(smi):
    if not smi:
        return None
    mol = MolFromSmiles(smi)
    return MolToSmiles(mol, canonical=True) if mol else None


def is_valid_molecule(smi: str) -> bool:
    return MolFromSmiles(smi) is not None


def get_inchikey_prefix(smi):
    if not smi:
        return None
    try:
        mol = MolFromSmiles(smi)
        return Chem.inchi.MolToInchiKey(mol)[:3] if mol else None
    except Exception:
        return None


def augment_smiles(smi: str, n_aug: int = 5, maxiter: int = 50) -> list:
    """Return up to n_aug random-order SMILES variants of the same molecule."""
    mol = MolFromSmiles(smi)
    if not mol:
        return []
    out = {MolToSmiles(mol, canonical=True)}
    attempts = 0
    while len(out) < n_aug and attempts < maxiter:
        out.add(MolToSmiles(mol, canonical=False, doRandom=True))
        attempts += 1
    return list(out)


def augment_corpus(smiles_list, n_aug: int = 5, maxiter: int = 50) -> list:
    """Expand each SMILES in the corpus to up to n_aug random-order variants.

    This is the corpus-level wrapper around `augment_smiles` that reproduces the
    naug_5x training corpora from the original Smiles-GEN dataset
    (e.g. Chembl24_9k_organic_naug_5x.smi). With n_aug=5 and a 9k corpus the
    augmented corpus is ~5× larger and dramatically improves validity.

    Invalid SMILES are silently dropped. Output preserves the per-molecule
    grouping (variants of mol i appear contiguously).
    """
    if n_aug <= 1:
        # n_aug=1 keeps just the canonical form for each valid input.
        out = []
        for smi in smiles_list:
            can = canonicalize(smi)
            if can:
                out.append(can)
        return out
    out = []
    for smi in smiles_list:
        out.extend(augment_smiles(smi, n_aug=n_aug, maxiter=maxiter))
    return out


def sanity_check(s: str) -> bool:
    """Original SmilesGEN_generator.SanityCheck — text-based: balanced
    parens, brackets, ring digits. Does NOT check chemistry."""
    for r in range(1, 10):
        if s.count(str(r)) % 2 != 0:
            return False
    if s.count("(") != s.count(")"):
        return False
    if s.count("[") != s.count("]"):
        return False
    return True
