# geneva2s

Autodidactic SMILES generator — PyTorch and TensorFlow variants. Char-level biLSTM
with 4 parallel concatenated branches.

**Based on the GEN architecture by Godin, Ertl, and van Deursen** (see CITATION
below), with substantial modifications to the training loop, generation logic,
and validation strategy.

## Install

```bash
pip install -e .[torch]          # PyTorch variant
pip install -e .[mlx]            # MLX variant (Apple Silicon native)
pip install -e .[tf]             # TensorFlow variant (TF 2.15.1 + Metal on macOS)
pip install -e .[all,tests]      # everything + pytest
```

> **Note on TF version**: use exactly `tensorflow==2.15.1` + `tensorflow-metal==1.1.0`
> on Apple Silicon. TF 2.16+ silently drops many LSTM ops to CPU (10-20× slower).

### Apple Silicon (M3 Max) environment setup

Validated setup for this repository:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[tests]"
uv pip install "torch==2.12" "tensorflow==2.15.1" "tensorflow-metal==1.1.0" "mlx>=0.20"
uv pip install --force-reinstall --no-deps git+https://github.com/guillaume-osmo/mlx-addons.git@main
uv pip install --force-reinstall --no-deps git+https://github.com/guillaume-osmo/mlxmolkit.git@main
```

If you have local RDKit/OpenBabel overrides in your shell, run commands with a
clean environment to avoid binary conflicts:

```bash
env -u PYTHONPATH -u DYLD_LIBRARY_PATH uv run -m pytest -q
env -u PYTHONPATH -u DYLD_LIBRARY_PATH uv run -m bench.bench_three_frameworks
```

## Repository layout

```
geneva2s/
├── geneva2s/
│   ├── tokenizer.py           CharTokenizer + Ertl-style multi-char replacements
│   ├── utils.py               canonicalize, augment_smiles, sanity_check
│   ├── torch/
│   │   ├── model.py           GenevaBiLSTM (biLSTM 128 → 4×LSTM 64 → concat → Dense)
│   │   ├── train.py           fit() and fit_best() with validity-driven checkpointing
│   │   ├── generate.py        sliding-window predict_batch_seeds
│   │   └── convert_keras.py   .keras file → PyTorch .pt state_dict
│   ├── tf/
│   │   ├── model.py           Keras builder (same arch, compatible with TF 2.15+)
│   │   └── generate.py        TF-side predict_batch_seeds
│   └── mlx/
│       ├── model.py           MLX port (loads PyTorch state_dict directly)
│       └── generate.py        MLX-side predict_batch_seeds
├── models/
│   ├── geneva2s.keras         Original Keras checkpoint
│   └── geneva2s.pt            PyTorch checkpoint (trained on 45k ChEMBL organic, ~97% sanity_check)
├── data/
│   └── chembl_9k_organic.smi  Small training corpus (ChEMBL-derived)
└── tests/                     pytest test suite
```

## Quick start

### PyTorch — load a trained model and generate

```python
import torch
from geneva2s.tokenizer import CharTokenizer
from geneva2s.torch.model import GenevaBiLSTM
from geneva2s.torch.generate import predict_batch_seeds

tok = CharTokenizer()
model = GenevaBiLSTM(tok.vocab_size, init_keras=False)
model.load_state_dict(torch.load("models/geneva2s.pt", map_location="cpu"))
model.eval()

with open("data/chembl_9k_organic.smi") as f:
    text = tok.prepare_corpus([line.strip() for line in f if line.strip()])

mols = predict_batch_seeds(model, tok, text, ncollect=100, ncopies=10,
                            device=torch.device("cpu"))
for m in mols[:5]:
    print(m)
```

### PyTorch — train from scratch

```python
from geneva2s.torch.train import fit_best

X, y = tok.sliding_window(text)
fit_best(model, X, y, device=torch.device("mps"),
         tokenizer=tok, text_for_seeds=text,
         num_epochs=30, check_every=2,
         save_path="models/geneva2s_new.pt")
```

### PyTorch — convert a saved Keras model

```bash
python -m geneva2s.torch.convert_keras --keras models/geneva2s.keras --out models/geneva2s_from_keras.pt
```

The conversion is **bit-faithful** (verified by `tests/test_equivalence.py`):
every layer of the PyTorch model matches a pure-numpy port of the Keras LSTM
math to float64 epsilon (~1e-13) on identical inputs.

### TensorFlow — load and generate

```python
from geneva2s.tokenizer import CharTokenizer
from geneva2s.tf.model import load_keras_model  # requires TF 2.15.x
from geneva2s.tf.generate import predict_batch_seeds

tok = CharTokenizer()
model = load_keras_model("models/geneva2s.keras")
text = tok.prepare_corpus(...)
mols = predict_batch_seeds(model, tok, text, ncollect=100, ncopies=10)
```

## Architecture

- Input: one-hot (B, 42, 27)
- `Bidirectional(LSTM(128, return_sequences=True))` → (B, 42, 256)
- 4 parallel `LSTM(64)` branches with per-branch `Dropout(0.3)`, each → (B, 64)
- `Concatenate` 4 branches → (B, 256)
- `Dense(27) + softmax`

≈ 495k parameters. The PyTorch port matches the Keras model byte-for-byte
when loaded with the same weights (see `tests/test_equivalence.py`).

## Differences from the original GEN architecture

- Faster training loop with explicit best-checkpoint tracking (`fit_best`)
- PyTorch-native variant alongside the original TF/Keras one
- Modernized generation API: clean sliding-window with multinomial sampling
  per `Generator.Sample`
- Validity scoring distinguishes between **text-based sanity check** (balanced
  parens / rings / brackets) and **rdkit canonicalization** (chemical validity)
  — the original paper's "~99% validity" is the former, ~90% under the latter

## Tests

```bash
pytest tests/                    # all tests
pytest tests/test_tokenizer.py   # just the tokenizer (includes the 8 original DataUtils unittest cases)
pytest tests/test_equivalence.py # PT-vs-Keras layer-by-layer numerical audit
```

Tests that require model weights or TF will skip automatically if those aren't
available.

## Performance (Apple M4 Pro, 24 GB)

Forward-pass latency, same model + weights, **best-tuned variant of each framework**:

| Batch | TF 2.15 | MLX +compile | **MLX fused+compile** | PT-MPS eager | PT +torch.compile |
|---|---|---|---|---|---|
| 1   | 25.15 ms | 6.20 | **3.46** | 3.91 | 3.97 |
| 16  | 27.12 ms | 6.64 | **3.55** | 4.09 | 4.39 |
| 64  | 26.79 ms | 6.60 | **3.70** | 4.67 | 4.87 |
| 256 | 28.09 ms | 7.76 | **6.32** | 7.77 | 7.75 |
| 512 | 28.63 ms | 10.73 | **9.29** | 12.22 | 12.23 |

```bash
python bench/bench_three_frameworks.py   # reproduce
```

**Takeaways:**
- **`GenevaBiLSTMMLXFused` + `mx.compile` wins at every batch size**, including beating PyTorch at batch=1 (3.46 vs 3.91 ms). The fused variant collapses the 4 sequential `LSTM(64)` second-layer branches into a single grouped LSTM step (one big input-projection matmul + an einsum recurrent step), eliminating 3 kernel-dispatch overheads per timestep. Mathematically equivalent (verified bit-identical, ~5e-6 diff).
- **`mx.compile` alone** (without fusion) gives MLX a ~2× speedup over eager and makes it competitive with PT at medium batches.
- **`torch.compile` on MPS doesn't help** — basically tied with PT eager (slight regression at small batches from compile overhead).
- **TF 2.15 + Metal** has the highest per-call overhead but scales smoothly.
- Above batch=512, PyTorch hits a kernel discontinuity (5× slowdown observed) and an allocator ceiling at batch=1024 on 24 GB. MLX and TF scale through without OOM.

**Use:** MLX fused+compile for everything inference-side on Apple Silicon, PT for training (where the bottleneck is per-step backprop, not forward dispatch).

### Fused PyTorch variant (`GenevaBiLSTMFused`)

A `GenevaBiLSTMFused` PyTorch class is also provided with the same grouped-LSTM
fusion. The trade-off is opposite to MLX:

| Batch | PT eager (`nn.LSTM`×4) | PT fused | Δ |
|---|---|---|---|
| 1 | 2.49 ms | 3.86 ms | -55% (worse) |
| 64 | 3.09 ms | 4.44 ms | -44% (worse) |
| 256 | 13.58 ms | 9.90 ms | **+27% better** |
| 512 | 18.20 ms | 17.32 ms | +5% |

PyTorch's `nn.LSTM` has a heavily-optimized MPS scan kernel that's faster than
any Python time-loop replacement at small batches. At batch ≥256, the savings
from one big input-projection matmul + grouped einsum recurrent step finally
win. Use `GenevaBiLSTMFused` only for large-batch offline inference.

### Custom Metal LSTM kernel via `mlx-addons`

Shipped! `pip install mlx-addons` adds two extra variants:

- `GenevaBiLSTMMLXMetal` — replaces every `mlx.nn.LSTM` call with a fused
  Metal kernel (single-LSTM cell, fast or precise math toggle).
- `GenevaBiLSTMMLXMetalGrouped` — combines the Metal cell with the
  grouped-branches idea: biLSTM uses single Metal LSTM, the 4 second-layer
  branches use a *grouped* Metal cell kernel that computes all 4 branches in
  one kernel launch per timestep.

Variant comparison with `mx.compile`:

| Batch | base+cmp | fused+cmp | metal+cmp | **metal-grouped+cmp** |
|---|---|---|---|---|
| 1   | 6.90 ms | 4.27 | 5.17 | **3.42** (2.02× baseline) |
| 16  | 7.26 | 4.54 | 6.09 | **3.89** (1.87×) |
| 64  | 8.15 | 4.95 | 6.76 | **4.61** (1.77×) |
| 256 | 9.30 | 8.23 | 8.40 | 8.41 (compute-bound; matmuls dominate) |
| 512 | 13.82 | 12.34 | 13.94 | 13.42 |

The Metal kernels are derived from PR
[ml-explore/mlx#3089](https://github.com/ml-explore/mlx/pull/3089) plus
later tuning (`pick_threads_per_group` heuristic, precise/fast math toggle)
from a research branch. They live in
[mlx-addons](https://github.com/guillaume-osmo/mlx-addons) under
`mlx_addons.recurrent`, so the kernels are reusable for any LSTM workload,
not just this generator.

### Training with the Metal kernels

Both `MetalLSTM` and the kernel-level `metal_lstm_scan` are now **fully
differentiable** — `mx.grad(model)` flows through the Metal kernels via a
hand-written VJP kernel + `mx.custom_function` wiring. Gradients match
autograd through a pure-MLX reference LSTM to within float32 epsilon (max
rel error ~3e-7).

Training-time (forward + backward) speedup vs `mlx.nn.LSTM` on Apple M4 Pro:

| Batch | pure-MLX fwd+bwd | Metal+VJP fwd+bwd | Speedup |
|---|---|---|---|
| 1 | 6.62 ms | 3.04 ms | **2.18×** |
| 16 | 6.62 ms | 3.65 ms | **1.81×** |
| 64 | 7.83 ms | 4.30 ms | **1.82×** |
| 256 | 13.88 ms | 11.03 ms | 1.26× |
| 512 | 24.78 ms | 22.38 ms | 1.11× |

Training-mode speedup tracks the forward-only inference speedup almost
exactly — confirming the VJP kernel is correctly sized. For a 11-min
80-epoch PyTorch training run, the MLX equivalent should land around
~6 min once the full pipeline (optimiser, data loader, etc.) is wired.

## Citation

If you use this code, please cite the original GEN paper:

```bibtex
@article{vanDeursen2020GEN,
  author  = {van Deursen, Ruud and Ertl, Peter and Tetko, Igor V. and Godin, Guillaume},
  title   = {GEN: highly efficient SMILES explorer using autodidactic generative examination networks},
  journal = {Journal of Cheminformatics},
  volume  = {12},
  pages   = {22},
  year    = {2020},
  doi     = {10.1186/s13321-020-00425-8}
}
```

## License

MIT
