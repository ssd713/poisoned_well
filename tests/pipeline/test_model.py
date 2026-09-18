"""Unit tests for the parts of model.py that have no reference counterpart."""

import numpy as np
import pytest

from pipeline.model import (
    CANDIDATE_LAYERS,
    HIDDEN_LAYERS,
    WIDTH,
    forward,
    layer_input,
    with_layer,
)


def test_candidate_layers_are_the_square_ones(weights):
    """The S2 search space, asserted against the file rather than assumed."""
    square = [
        name.removesuffix(".weight")
        for name, array in weights.items()
        if name.endswith(".weight") and array.shape == (WIDTH, WIDTH)
    ]
    assert sorted(square) == sorted(CANDIDATE_LAYERS)
    assert weights["fc1.weight"].shape == (WIDTH, 77)
    assert weights["out.weight"].shape == (1, WIDTH)


def test_layer_input_resolves_the_preceding_activation(weights, features):
    X = features["X"][:100]
    _, acts = forward(weights, X, return_hidden=True)

    assert layer_input("fc1", X, acts) is X
    for earlier, later in zip(HIDDEN_LAYERS, HIDDEN_LAYERS[1:]):
        assert layer_input(later, X, acts) is acts[earlier]
    assert layer_input("out", X, acts) is acts["fc6"]


def test_layer_input_rejects_unknown_layer(features):
    with pytest.raises(KeyError):
        layer_input("fc7", features["X"][:1], {})


def test_with_layer_leaves_the_original_untouched(weights):
    replacement = np.zeros((WIDTH, WIDTH), np.float32)
    patched = with_layer(weights, "fc3", replacement)

    assert np.array_equal(patched["fc3.weight"], replacement)
    assert not np.array_equal(weights["fc3.weight"], replacement)
    assert patched["fc2.weight"] is weights["fc2.weight"]


def test_with_layer_rejects_wrong_shape(weights):
    with pytest.raises(ValueError, match="expects"):
        with_layer(weights, "fc3", np.zeros((WIDTH, 10)))


def test_with_layer_rejects_unknown_layer(weights):
    with pytest.raises(KeyError):
        with_layer(weights, "fc9", np.zeros((WIDTH, WIDTH)))


def test_zeroing_a_layer_changes_the_output(weights, features):
    """Sanity: substitution actually reaches the forward pass."""
    X = features["X"][:200]
    flat = forward(with_layer(weights, "fc3", np.zeros((WIDTH, WIDTH))), X)
    assert not np.allclose(flat, forward(weights, X))


def test_preacts_and_activations_are_consistent(weights, features):
    _, acts, pre = forward(weights, features["X"][:200], return_hidden=True, return_preacts=True)
    for layer in HIDDEN_LAYERS:
        assert np.allclose(acts[layer], np.maximum(pre[layer], 0.0))
        assert (acts[layer] >= 0).all()


def test_forward_rejects_non_2d_input(weights):
    with pytest.raises(ValueError, match="2-D"):
        forward(weights, np.zeros(77))
