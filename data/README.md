# `geneva2s/data/` — datasets and training artifacts

## 2019 training corpora (van Deursen et al., GEN paper)

The original Smiles-GEN paper trained on canonical SMILES from ChEMBL24 and Zinc15,
with random-walk augmentation (5× per molecule) producing the `naug_5x` corpora.
Both are checked in here so the Keras → PyTorch / MLX ports can train from the
same source.

### Bundled directly (9k variants, fast smoke tests)

| file | source | molecules | size |
|------|--------|-----------|------|
| `chembl_9k_organic.smi` | ChEMBL24 organic subset | 9,000 | 319 KB |
| `chembl_9k_organic_naug_5x.smi` | random-walk augmented 5× | ~45,000 | 1.3 MB |
| `zinc15_9k_randomized.smi` | Zinc15 organic subset | 9,000 | 347 KB |
| `zinc15_9k_randomized_naug_5x.smi` | random-walk augmented 5× | ~45,000 | 1.2 MB |

The default training command (`python -m geneva2s.main`) uses `chembl_9k_organic.smi`
because it's the smallest sufficient corpus. Pass `--smi data/chembl_9k_organic_naug_5x.smi`
to train on the pre-augmented version, or `--augment 5` to do augmentation on the fly.

### Bundled as zip (45k + 225k variants)

| file | contents | unzipped | zipped |
|------|----------|----------|--------|
| `datasets_2019_45k_225k.zip` | 8 files: ChEMBL & Zinc15 × {45k, 225k} × {raw, naug_5x} | ~95 MB | 28.7 MB |

To extract:

```bash
cd data && unzip datasets_2019_45k_225k.zip -d datasets_2019/
```

Then point `--smi` at e.g. `data/datasets_2019/chembl_225k_organic_naug_5x.smi` for full-corpus training (this is what the Keras and DPO checkpoints used).

## DPO preference pairs

`dpo_pairs_2025.json` — 3,009 preference pairs from the May–June 2025 DPO finetune
run on top of the Keras base model. Format: a JSON list of `[chosen, rejected]`
SMILES pairs. Both members of each pair have approximately equal SMILES length
(filename suffix `equallength2`) — important for DPO loss numerics.

Example pair:

```json
[
  "c1c(S(=O)(N2C(Cc3ccccc3)CCC2)=O)ccnc1",  // chosen (drug-like)
  "c1c(cccc1)Nc1c(S(=O)(=O)c2cccc3cccc12)cccc3"  // rejected (junky aromatic stack)
]
```

This is a snapshot — not auto-regenerated. To rebuild from scratch you need the
adaptive-log → dpo-pair extractor that lived in the original `dpo_geneva.py`
script (not ported to geneva2s yet).

## Provenance and history

Full training-history audit for the model checkpoints in `geneva2s/models/` is on
the external drive at `Smiles-GEN-master/`:

- 301 adaptive logs from round 0 (May 16, 2025) to round 202 (Jul 1, 2025)
- Cumulative `adaptive_log_rounds_*.json` files up to 290 MB
- Knowledge graph artifacts (`knowledge-{node,edge}.json` — fragrance-domain
  triples, separate project, not used by the SMILES generator itself)

The Keras checkpoint `models/geneva2s.keras` is byte-identical to
`Chembl_bilstm_4x_2.keras` (May 8, 2025). The DPO-finetuned variant
`Chembl_bilstm_4x_2_dpo.{h5,keras}` (June 3-6, 2025) is not (yet) bundled here.
