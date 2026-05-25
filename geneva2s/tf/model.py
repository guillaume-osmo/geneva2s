"""Keras builder for the GENEVA²S architecture.

Input(maxlen, vocab) → Bidirectional(LSTM(128, return_sequences=True))
                    → 4 × LSTM(64) → 4 × Dropout(0.3)
                    → Concatenate → Dense(vocab) → softmax

Compatible with Keras 2 (TF 2.15) and Keras 3 (TF 2.16+).
"""
from __future__ import annotations


def build_model(vocab_size: int = 27, maxlen: int = 42,
                hidden=(128, 64), n_branches: int = 4, dropout: float = 0.3):
    # Import here so the package can be imported without TF installed
    try:
        from tensorflow import keras
        from tensorflow.keras.layers import (
            Activation, Bidirectional, Concatenate, Dense, Dropout, Input, LSTM,
        )
    except ImportError:
        import keras
        from keras.layers import (
            Activation, Bidirectional, Concatenate, Dense, Dropout, Input, LSTM,
        )

    inp = Input(shape=(maxlen, vocab_size), name="Input")
    e = Bidirectional(LSTM(hidden[0], return_sequences=True), name="Embedding")(inp)
    branches = []
    for i in range(n_branches):
        b = LSTM(hidden[1], name=f"Latent_{i}")(e)
        b = Dropout(dropout, name=f"Dropout_{i}")(b)
        branches.append(b)
    cat = Concatenate(name="concatenate_1")(branches)
    out = Dense(vocab_size, name="Output")(cat)
    out = Activation("softmax", name="activation_1")(out)
    return keras.Model(inp, out)


def load_weights_from_h5(model, weights_h5_path: str) -> None:
    """Load the model.weights.h5 file extracted from a .keras archive."""
    import h5py
    with h5py.File(weights_h5_path) as h:
        bilstm = model.layers[1]
        bilstm.forward_layer.set_weights([
            h["layers/bidirectional/forward_layer/cell/vars/0"][:],
            h["layers/bidirectional/forward_layer/cell/vars/1"][:],
            h["layers/bidirectional/forward_layer/cell/vars/2"][:],
        ])
        bilstm.backward_layer.set_weights([
            h["layers/bidirectional/backward_layer/cell/vars/0"][:],
            h["layers/bidirectional/backward_layer/cell/vars/1"][:],
            h["layers/bidirectional/backward_layer/cell/vars/2"][:],
        ])
        for i, name in enumerate(["lstm", "lstm_1", "lstm_2", "lstm_3"]):
            model.layers[2 + i].set_weights([
                h[f"layers/{name}/cell/vars/0"][:],
                h[f"layers/{name}/cell/vars/1"][:],
                h[f"layers/{name}/cell/vars/2"][:],
            ])
        dense = next(l for l in model.layers if l.name == "Output")
        dense.set_weights([
            h["layers/dense/vars/0"][:],
            h["layers/dense/vars/1"][:],
        ])


def load_keras_model(keras_path: str):
    """Load a .keras file directly (works in TF 2.15.x; later versions may fail
    on Keras 2-saved files due to internal class-path changes)."""
    try:
        from tensorflow import keras
    except ImportError:
        import keras
    return keras.models.load_model(keras_path)
