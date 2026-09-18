"""Shared fixtures. The real competition files are small enough to use directly."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data" / "train.csv"
SCALER = ROOT / "model" / "feature_scaler.json"
WEIGHTS = ROOT / "model" / "streamflow_model_bug.npz"


@pytest.fixture(scope="session")
def weights():
    from pipeline.model import load_weights
    return load_weights(WEIGHTS)


@pytest.fixture(scope="session")
def features():
    from pipeline.features import build_features
    return build_features(TRAIN_CSV, SCALER)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
