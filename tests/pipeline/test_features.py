"""Unit tests for feature construction."""

import numpy as np
import pytest

from pipeline.features import N_FEATURES, RAIN_WINDOW, rainfall_windows


def test_windows_are_oldest_first_and_end_at_the_current_hour():
    rain = np.arange(100.0, dtype=np.float32)
    windows = rainfall_windows(rain)

    assert windows.shape == (100 - RAIN_WINDOW + 1, RAIN_WINDOW)
    # Window 0 covers hours 0..71, so it ends at the hour it is aligned to.
    assert windows[0][0] == 0.0
    assert windows[0][-1] == RAIN_WINDOW - 1
    assert windows[1][-1] == RAIN_WINDOW


def test_windows_reject_too_short_a_series():
    with pytest.raises(ValueError, match="at least"):
        rainfall_windows(np.zeros(RAIN_WINDOW - 1, np.float32))


def test_public_period_has_the_documented_row_count(features):
    """18,756 rows minus the first 71 unusable hours."""
    assert features["X"].shape == (18685, N_FEATURES)
    assert features["truth"].shape == (18685,)


def test_standardized_inputs_are_roughly_unit_scale(features):
    """A scaler applied the wrong way round would show up here."""
    X = features["X"]
    assert np.isfinite(X).all()
    assert abs(X.mean()) < 1.0
    assert 0.1 < X.std() < 10.0


def test_context_arrays_line_up_with_the_inputs(features):
    n = len(features["X"])
    for key in ["truth", "datetime", "rain", "sum24", "row_id"]:
        assert len(features[key]) == n
