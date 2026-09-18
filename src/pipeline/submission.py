"""Write submission.csv in the competition's exact format.

6,384 rows: 3,136 fc weights (row-major) then 56 direction components, once
under pub_ and again under prv_. Three columns, id/value/writeup_url, with the
same URL repeated on every row -- a submission without one fails at upload, so
the URL is required rather than defaulted.
"""

import numpy as np
import pandas as pd

WIDTH = 56
N_WEIGHTS = WIDTH * WIDTH
N_ROWS = 2 * (N_WEIGHTS + WIDTH)
PREFIXES = ("pub_", "prv_")


def _ids(prefix):
    ids = [f"{prefix}fc_{r}_{c}" for r in range(WIDTH) for c in range(WIDTH)]
    ids += [f"{prefix}dir_{k}" for k in range(WIDTH)]
    return ids


def build_submission(fc_weight, direction, writeup_url):
    """Assemble the submission as a DataFrame.

    fc_weight is (56, 56), direction is (56,). Both are validated: a wrong
    shape or a stray NaN here costs a submission slot, and slots are the
    instrument that reads back our cosine.
    """
    fc_weight = np.asarray(fc_weight, np.float64)
    direction = np.asarray(direction, np.float64)

    if fc_weight.shape != (WIDTH, WIDTH):
        raise ValueError(f"fc_weight must be ({WIDTH}, {WIDTH}), got {fc_weight.shape}")
    if direction.shape != (WIDTH,):
        raise ValueError(f"direction must be ({WIDTH},), got {direction.shape}")
    if not np.isfinite(fc_weight).all():
        raise ValueError("fc_weight contains non-finite values")
    if not np.isfinite(direction).all():
        raise ValueError("direction contains non-finite values")
    if np.linalg.norm(direction) == 0:
        raise ValueError("direction is all zeros, cosine would be undefined")
    if not writeup_url:
        raise ValueError("writeup_url is required: a submission without one fails at upload")

    # Row-major, matching submission_example.csv: fc_0_0, fc_0_1, ... fc_55_55.
    values = np.concatenate([fc_weight.reshape(-1), direction])

    return pd.DataFrame({
        "id": _ids(PREFIXES[0]) + _ids(PREFIXES[1]),
        "value": np.concatenate([values, values]),
        "writeup_url": writeup_url,
    })


def write_submission(path, fc_weight, direction, writeup_url):
    """Build and write the submission. Returns the DataFrame."""
    df = build_submission(fc_weight, direction, writeup_url)
    df.to_csv(path, index=False)
    return df
