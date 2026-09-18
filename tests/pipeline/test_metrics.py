"""Unit tests for metrics, including the edit-size metrics the plan tracks."""

import numpy as np
import pytest

from pipeline import metrics


def test_perfect_prediction_scores_one():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.nse(truth, truth) == pytest.approx(1.0)
    assert metrics.rmse(truth, truth) == pytest.approx(0.0)
    assert metrics.bias(truth, truth) == pytest.approx(0.0)


def test_predicting_the_mean_scores_zero():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.nse(np.full(4, truth.mean()), truth) == pytest.approx(0.0)


def test_bias_is_signed_and_positive_means_over_prediction():
    truth = np.zeros(10)
    assert metrics.bias(np.full(10, 0.5), truth) == pytest.approx(0.5)
    assert metrics.bias(np.full(10, -0.5), truth) == pytest.approx(-0.5)


def test_annual_bias_scales_by_hours_per_year():
    truth = np.zeros(10)
    pred = np.full(10, 0.001)
    assert metrics.bias_mm_per_year(pred, truth) == pytest.approx(0.001 * 24 * 365.25)


def test_constant_truth_is_rejected():
    with pytest.raises(ValueError, match="constant"):
        metrics.nse(np.zeros(5), np.ones(5))


def test_effective_rank_of_a_rank_one_matrix_is_one():
    """The rank-1 hypothesis is the thing this metric exists to test."""
    u = np.arange(1.0, 7.0).reshape(6, 1)
    v = np.arange(1.0, 7.0).reshape(1, 6)
    assert metrics.effective_rank(u @ v) == 1


def test_effective_rank_of_identity_is_full():
    assert metrics.effective_rank(np.eye(6)) == 6


def test_edit_summary_measures_a_rank_one_edit():
    original = np.eye(6)
    delta = np.outer(np.ones(6), np.ones(6)) * 0.1
    summary = metrics.edit_summary(original, original + delta)

    assert summary["delta_rank"] == 1
    assert summary["delta_fro"] == pytest.approx(np.linalg.norm(delta))
    assert summary["delta_max_abs"] == pytest.approx(0.1)


def test_edit_summary_of_no_change_is_zero():
    original = np.eye(6)
    summary = metrics.edit_summary(original, original.copy())
    assert summary["delta_fro"] == pytest.approx(0.0)
    assert summary["delta_rank"] == 0


def test_edit_summary_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.edit_summary(np.eye(6), np.eye(5))
