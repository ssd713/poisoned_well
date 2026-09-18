#!/usr/bin/env python3
"""Build a submission.csv.

S0's baseline is the null: the poisoned model's own weights, unchanged, plus a
random direction. It should score ~0 on both halves, and that is the point --
it proves the format, the scaler, the alignment and the writeup gate while the
cost of being wrong is one submission slot rather than one month.

    python scripts/make_submission.py --writeup-url https://.../discussion/123

The --layer choice does not affect the null's score. The scorer drops the
matrix into whichever layer was tampered with, which we do not know until S2;
submitting any candidate's own weights leaves that layer unchanged if we guessed
right, and replaces it with a sibling's if we guessed wrong. Both score zero on
part 1, so the null is a valid control either way.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import metrics  # noqa: E402
from pipeline.features import build_features  # noqa: E402
from pipeline.model import CANDIDATE_LAYERS, forward, load_weights  # noqa: E402
from pipeline.submission import write_submission  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--writeup-url",
        required=True,
        help="URL of the competition Discussion writeup. Required: a submission without one fails at upload.",
    )
    parser.add_argument("--layer", default="fc2", choices=CANDIDATE_LAYERS,
                        help="which layer's weights to submit (default: fc2)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "submission.csv")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random direction")
    args = parser.parse_args()

    weights = load_weights(ROOT / "model" / "streamflow_model_bug.npz")
    features = build_features(ROOT / "data" / "train.csv", ROOT / "model" / "feature_scaler.json")

    pred = forward(weights, features["X"])
    summary = metrics.skill_summary(pred, features["truth"])

    print(f"public period: {len(features['X']):,} hours")
    print(f"  NSE         {summary['nse']:.4f}")
    print(f"  RMSE        {summary['rmse']:.4f} mm/hr")
    print(f"  bias        {summary['bias_mm_hr']:+.6f} mm/hr")
    print(f"  annual bias {summary['bias_mm_yr']:+.2f} mm/yr")

    fc_weight = weights[f"{args.layer}.weight"]
    direction = np.random.default_rng(args.seed).normal(size=56)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = write_submission(args.out, fc_weight, direction, args.writeup_url)

    print(f"\nwrote {args.out} ({len(df):,} rows)")
    print(f"  fc_weight   {args.layer}.weight, unchanged")
    print(f"  direction   random, seed {args.seed}")
    print("  expected    ~0 on both halves -- this is the control, not an attempt")


if __name__ == "__main__":
    main()
