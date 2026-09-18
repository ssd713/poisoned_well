"""Build the model's 77 inputs from the raw hourly CSV.

Each usable hour becomes 72 hours of rainfall history (oldest first, ending at
the current hour) followed by 5 current-hour features, z-scored with the
provided scaler.

The first 71 rows can't form a full window and are dropped, which is why 18,756
rows of train.csv yield 18,685 examples.
"""

import json

import numpy as np
import pandas as pd

RAIN_WINDOW = 72
STATIC_FEATURES = ["sm_2in", "sm_20in", "season", "T_air_C", "srad_Wm2"]
N_FEATURES = RAIN_WINDOW + len(STATIC_FEATURES)  # 77


def load_scaler(path):
    """Load z-score stats. Returns (mu, sigma), each shape (77,)."""
    with open(path) as fh:
        scaler = json.load(fh)
    mu = np.asarray(scaler["mu"], np.float32)
    sigma = np.asarray(scaler["sig"], np.float32)
    if mu.shape != (N_FEATURES,) or sigma.shape != (N_FEATURES,):
        raise ValueError(f"scaler must be ({N_FEATURES},), got {mu.shape} and {sigma.shape}")
    return mu, sigma


def rainfall_windows(rain):
    """Stack rolling 72-hour rainfall windows, shape (n - 71, 72).

    Built by explicit index gather rather than a striding helper, so a bug here
    is unlikely to mirror one in the vendored implementation.
    """
    n = len(rain)
    if n < RAIN_WINDOW:
        raise ValueError(f"need at least {RAIN_WINDOW} rows, got {n}")
    starts = np.arange(n - RAIN_WINDOW + 1)[:, None]
    offsets = np.arange(RAIN_WINDOW)[None, :]
    return rain[starts + offsets]


def build_features(data_csv, scaler_json):
    """Turn the raw CSV into model inputs plus the aligned target and context.

    Returns a dict with X (N, 77), truth (N,) or None, and the per-hour context
    S1 needs: datetime, rain, sum24, row_id.
    """
    mu, sigma = load_scaler(scaler_json)

    df = pd.read_csv(data_csv, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    rain = df["rain_mm"].to_numpy(np.float32)
    windows = rainfall_windows(rain)

    # Window i spans hours i..i+71, so it lines up with row i+71.
    aligned = df.iloc[RAIN_WINDOW - 1:].reset_index(drop=True)

    # Drop hours with a missing feature or a gap in the rainfall history.
    has_features = aligned[STATIC_FEATURES].notna().all(axis=1).to_numpy()
    finite_window = np.isfinite(windows).all(axis=1)
    keep = has_features & finite_window

    statics = aligned.loc[keep, STATIC_FEATURES].to_numpy(np.float32)
    raw = np.concatenate([windows[keep], statics], axis=1)
    X = ((raw - mu) / sigma).astype(np.float32)

    sum24 = (
        pd.Series(rain).rolling(24, min_periods=1).sum()
        .to_numpy(np.float32)[RAIN_WINDOW - 1:][keep]
    )

    truth = None
    if "streamflow_mm_hr" in aligned.columns:
        truth = aligned.loc[keep, "streamflow_mm_hr"].to_numpy(np.float32)

    return {
        "X": X,
        "truth": truth,
        "datetime": aligned.loc[keep, "datetime"].to_numpy(),
        "rain": aligned.loc[keep, "rain_mm"].to_numpy(np.float32),
        "sum24": sum24,
        "row_id": np.arange(int(keep.sum())),
    }
