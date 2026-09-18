"""The S0 gate: our pipeline must agree with the vendored reference.

The plan makes this a hard gate. Everything downstream -- leak localization,
layer identification, the repair fit -- is meaningless if our forward pass or
our feature builder disagrees with the model the scorer actually runs.
"""

import numpy as np
import pytest

from vendor import streamflow_model as ref

from .conftest import SCALER, TRAIN_CSV, WEIGHTS

TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def reference_features():
    return ref.build_features(TRAIN_CSV, SCALER)


def test_feature_matrix_matches_reference(features, reference_features):
    assert features["X"].shape == reference_features["X"].shape
    assert np.abs(features["X"] - reference_features["X"]).max() < TOLERANCE


def test_target_matches_reference(features, reference_features):
    assert np.abs(features["truth"] - reference_features["truth"]).max() < TOLERANCE


def test_context_columns_match_reference(features, reference_features):
    assert np.abs(features["rain"] - reference_features["rain"]).max() < TOLERANCE
    assert np.abs(features["sum24"] - reference_features["sum24"]).max() < TOLERANCE
    assert (features["datetime"] == reference_features["datetime"]).all()


def test_forward_matches_reference_on_sampled_rows(weights, features, rng):
    """The gate as the plan words it: <1e-6 on 1,000 sampled rows."""
    from pipeline.model import forward

    rows = rng.choice(len(features["X"]), size=1000, replace=False)
    X = features["X"][rows]

    assert np.abs(forward(weights, X) - ref.forward(weights, X)).max() < TOLERANCE


def test_forward_matches_reference_on_full_public_period(weights, features):
    """Sampling could hide a bug in a rare regime, so check every hour too."""
    from pipeline.model import forward

    ours = forward(weights, features["X"])
    theirs = ref.forward(weights, features["X"])
    assert np.abs(ours - theirs).max() < TOLERANCE


def test_hidden_activations_match_reference(weights, features):
    from pipeline.model import forward

    _, ours = forward(weights, features["X"][:500], return_hidden=True)
    _, theirs = ref.forward(weights, features["X"][:500], return_hidden=True)

    assert set(ours) == set(theirs)
    for layer in ours:
        assert np.abs(ours[layer] - theirs[layer]).max() < TOLERANCE


def test_metrics_match_reference(weights, features):
    from pipeline import metrics
    from pipeline.model import forward

    pred = forward(weights, features["X"])
    truth = features["truth"]

    assert abs(metrics.nse(pred, truth) - ref.nse(pred, truth)) < TOLERANCE
    assert abs(metrics.rmse(pred, truth) - ref.rmse(pred, truth)) < TOLERANCE
