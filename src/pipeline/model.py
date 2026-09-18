"""Load and run the MLP: 77 -> 56 x 6 (ReLU) -> 1.

Adds three things the vendored loader doesn't have, all needed downstream:
pre-activation capture (S2 wants to know which units are live), layer
substitution (S2 ablations and S4 repairs), and layer_input, which resolves the
activation space a given layer reads from.
"""

import numpy as np

HIDDEN_LAYERS = ["fc1", "fc2", "fc3", "fc4", "fc5", "fc6"]
OUTPUT_LAYER = "out"
WIDTH = 56

# fc1 is 77->56 and out is 56->1, so neither can be the tampered (56,56) layer.
CANDIDATE_LAYERS = ["fc2", "fc3", "fc4", "fc5", "fc6"]


def load_weights(path):
    """Load the .npz state dict as {name: array}."""
    with np.load(str(path)) as archive:
        return {name: archive[name] for name in archive.files}


def save_weights(weights, path):
    """Save a {name: array} dict as .npz, matching the upstream float32 dtype."""
    np.savez(path, **{k: np.asarray(v, np.float32) for k, v in weights.items()})


def with_layer(weights, layer, weight_matrix):
    """Copy of `weights` with one layer's weight matrix replaced.

    Shallow copy: the other arrays are shared, so don't mutate them in place.
    """
    key = f"{layer}.weight"
    if key not in weights:
        raise KeyError(f"unknown layer {layer!r}")
    replacement = np.asarray(weight_matrix, np.float32)
    if replacement.shape != weights[key].shape:
        raise ValueError(
            f"{layer} expects {weights[key].shape}, got {replacement.shape}"
        )
    patched = dict(weights)
    patched[key] = replacement
    return patched


def layer_input(layer, X, activations):
    """The activations a layer reads from.

    fc1 reads the inputs; every later layer reads the previous layer's output.
    This is the space the trigger direction lives in: for a tampered fc_k, the
    56-vector we're recovering is a direction over layer_input(fc_k, ...).
    """
    if layer == HIDDEN_LAYERS[0]:
        return X
    if layer in HIDDEN_LAYERS:
        return activations[HIDDEN_LAYERS[HIDDEN_LAYERS.index(layer) - 1]]
    if layer == OUTPUT_LAYER:
        return activations[HIDDEN_LAYERS[-1]]
    raise KeyError(f"unknown layer {layer!r}")


def forward(weights, X, return_hidden=False, return_preacts=False):
    """Run the network on inputs X (N, 77), returning predictions (N,).

    With return_hidden, also returns post-ReLU activations per hidden layer.
    With return_preacts, also returns the pre-ReLU values, which is how you tell
    a unit that is genuinely dead from one that merely happens to be off.
    """
    X = np.asarray(X, np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    activations, preacts = {}, {}
    h = X
    for layer in HIDDEN_LAYERS:
        z = h @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"]
        h = np.maximum(z, 0.0)
        preacts[layer] = z
        activations[layer] = h

    pred = (h @ weights[f"{OUTPUT_LAYER}.weight"].T + weights[f"{OUTPUT_LAYER}.bias"])[:, 0]

    if return_hidden and return_preacts:
        return pred, activations, preacts
    if return_hidden:
        return pred, activations
    if return_preacts:
        return pred, preacts
    return pred
