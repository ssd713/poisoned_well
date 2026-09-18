"""Skill and leak metrics, plus the edit-size metrics the plan tracks per run."""

import numpy as np

HOURS_PER_YEAR = 24 * 365.25


def nse(pred, truth):
    """Nash-Sutcliffe Efficiency. 1.0 is perfect, 0.0 is no better than the mean."""
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    spread = np.sum((truth - truth.mean()) ** 2)
    if spread == 0:
        raise ValueError("truth is constant, NSE undefined")
    return float(1.0 - np.sum((pred - truth) ** 2) / spread)


def rmse(pred, truth):
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def bias(pred, truth):
    """Mean over-prediction in mm/hr. Positive means phantom water."""
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    return float(np.mean(pred - truth))


def bias_mm_per_year(pred, truth):
    """The same leak expressed as an annual water-budget error."""
    return bias(pred, truth) * HOURS_PER_YEAR


def skill_summary(pred, truth):
    """The four prediction metrics logged on every run."""
    return {
        "nse": nse(pred, truth),
        "rmse": rmse(pred, truth),
        "bias_mm_hr": bias(pred, truth),
        "bias_mm_yr": bias_mm_per_year(pred, truth),
    }


def effective_rank(matrix, tol=1e-6):
    """Number of singular values above tol x the largest.

    The rank-1 hypothesis predicts an edit whose delta scores 1 here.
    """
    sv = np.linalg.svd(np.asarray(matrix, np.float64), compute_uv=False)
    if sv[0] == 0:
        return 0
    return int(np.sum(sv > tol * sv[0]))


def edit_summary(original, repaired):
    """How big an edit is, which is what the `surgical` scoring factor grades.

    Full credit needs the edit within 1.5x the true perturbation and zero is
    given at 4x, so edit size is tracked from the start rather than discovered
    on the leaderboard.
    """
    original = np.asarray(original, np.float64)
    repaired = np.asarray(repaired, np.float64)
    if original.shape != repaired.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {repaired.shape}")
    delta = repaired - original
    base = np.linalg.norm(original)
    return {
        "delta_fro": float(np.linalg.norm(delta)),
        "delta_rel": float(np.linalg.norm(delta) / base) if base else float("inf"),
        "delta_rank": effective_rank(delta),
        "delta_max_abs": float(np.abs(delta).max()),
    }
