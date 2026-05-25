"""Convert a Keras .keras checkpoint → PyTorch state_dict for GenevaBiLSTM.

Keras LSTM weight format:  kernel (input, 4*hidden), recurrent_kernel (hidden, 4*hidden), bias (4*hidden,)
PyTorch LSTM:              weight_ih (4*hidden, input), weight_hh (4*hidden, hidden), bias_ih + bias_hh

Conversion: transpose kernels; put Keras bias entirely in PT bias_ih, zero bias_hh.
Gate order in both is IFCO (input/forget/cell/output) — no permutation needed.

Keras Dense.kernel (in, out) → PT Linear.weight (out, in): transpose.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch


def extract_h5_from_keras(keras_path: str, out_h5: str) -> str:
    with zipfile.ZipFile(keras_path) as z:
        with z.open("model.weights.h5") as f:
            data = f.read()
    with open(out_h5, "wb") as f:
        f.write(data)
    return out_h5


def convert(keras_path: str, pt_save_path: str) -> dict:
    """Read .keras → write .pt state_dict compatible with GenevaBiLSTM."""
    h5_path = Path(pt_save_path).with_suffix(".weights.h5")
    extract_h5_from_keras(keras_path, str(h5_path))

    sd: dict = {}
    with h5py.File(h5_path) as h:
        for direction, suffix in [("forward", ""), ("backward", "_reverse")]:
            k = h[f"layers/bidirectional/{direction}_layer/cell/vars/0"][:]
            rk = h[f"layers/bidirectional/{direction}_layer/cell/vars/1"][:]
            b = h[f"layers/bidirectional/{direction}_layer/cell/vars/2"][:]
            sd[f"embedding.weight_ih_l0{suffix}"] = torch.from_numpy(k.T.copy())
            sd[f"embedding.weight_hh_l0{suffix}"] = torch.from_numpy(rk.T.copy())
            sd[f"embedding.bias_ih_l0{suffix}"] = torch.from_numpy(b.copy())
            sd[f"embedding.bias_hh_l0{suffix}"] = torch.zeros(512)

        for i, name in enumerate(["lstm", "lstm_1", "lstm_2", "lstm_3"]):
            k = h[f"layers/{name}/cell/vars/0"][:]
            rk = h[f"layers/{name}/cell/vars/1"][:]
            b = h[f"layers/{name}/cell/vars/2"][:]
            sd[f"branches.{i}.weight_ih_l0"] = torch.from_numpy(k.T.copy())
            sd[f"branches.{i}.weight_hh_l0"] = torch.from_numpy(rk.T.copy())
            sd[f"branches.{i}.bias_ih_l0"] = torch.from_numpy(b.copy())
            sd[f"branches.{i}.bias_hh_l0"] = torch.zeros(256)

        dk = h["layers/dense/vars/0"][:]
        db = h["layers/dense/vars/1"][:]
        sd["head.weight"] = torch.from_numpy(dk.T.copy())
        sd["head.bias"] = torch.from_numpy(db.copy())

    Path(pt_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, pt_save_path)
    h5_path.unlink(missing_ok=True)
    return sd


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--keras", required=True, help="Path to input .keras file")
    p.add_argument("--out", required=True, help="Path to output .pt file")
    args = p.parse_args()
    sd = convert(args.keras, args.out)
    print(f"saved {len(sd)} tensors → {args.out}")


if __name__ == "__main__":
    main()
