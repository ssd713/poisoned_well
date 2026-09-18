"""Submission format tests, checked against the competition's own example file."""

import numpy as np
import pandas as pd
import pytest

from pipeline.submission import N_ROWS, WIDTH, build_submission, write_submission

from .conftest import ROOT

EXAMPLE = ROOT / "data" / "submission_example.csv"
URL = "https://example.invalid/writeup"


@pytest.fixture(scope="module")
def example():
    return pd.read_csv(EXAMPLE)


@pytest.fixture(scope="module")
def built(rng):
    fc = rng.normal(size=(WIDTH, WIDTH))
    direction = rng.normal(size=WIDTH)
    return build_submission(fc, direction, URL), fc, direction


def test_ids_match_the_example_exactly(built, example):
    """Order and spelling both matter, so compare the id column element-wise."""
    df, _, _ = built
    assert df["id"].tolist() == example["id"].tolist()


def test_shape_and_columns_match_the_example(built, example):
    df, _, _ = built
    assert len(df) == N_ROWS == len(example)
    assert list(df.columns) == list(example.columns)


def test_weights_are_written_row_major(built):
    df, fc, _ = built
    lookup = dict(zip(df["id"], df["value"]))
    for r, c in [(0, 0), (0, 1), (1, 0), (55, 55), (17, 42)]:
        assert lookup[f"pub_fc_{r}_{c}"] == pytest.approx(fc[r, c])


def test_direction_components_are_written_in_order(built):
    df, _, direction = built
    lookup = dict(zip(df["id"], df["value"]))
    for k in [0, 1, 27, 55]:
        assert lookup[f"pub_dir_{k}"] == pytest.approx(direction[k])


def test_public_and_private_halves_carry_identical_values(built):
    df, _, _ = built
    pub = df[df["id"].str.startswith("pub_")]["value"].to_numpy()
    prv = df[df["id"].str.startswith("prv_")]["value"].to_numpy()
    assert np.array_equal(pub, prv)


def test_writeup_url_repeats_on_every_row(built):
    df, _, _ = built
    assert (df["writeup_url"] == URL).all()


def test_round_trip_through_csv_preserves_values(tmp_path, rng):
    fc = rng.normal(size=(WIDTH, WIDTH))
    direction = rng.normal(size=WIDTH)
    path = tmp_path / "submission.csv"
    write_submission(path, fc, direction, URL)

    reloaded = pd.read_csv(path)
    lookup = dict(zip(reloaded["id"], reloaded["value"]))
    assert lookup["pub_fc_3_7"] == pytest.approx(fc[3, 7], rel=1e-9)
    assert lookup["prv_dir_9"] == pytest.approx(direction[9], rel=1e-9)


@pytest.mark.parametrize(
    "fc_shape, dir_shape, match",
    [
        ((55, 56), (56,), "fc_weight must be"),
        ((56, 56), (55,), "direction must be"),
    ],
)
def test_wrong_shapes_are_rejected(fc_shape, dir_shape, match):
    with pytest.raises(ValueError, match=match):
        build_submission(np.zeros(fc_shape), np.zeros(dir_shape), URL)


def test_non_finite_values_are_rejected():
    fc = np.zeros((WIDTH, WIDTH))
    fc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_submission(fc, np.ones(WIDTH), URL)


def test_zero_direction_is_rejected():
    """Cosine against an all-zero vector is undefined, so catch it here."""
    with pytest.raises(ValueError, match="all zeros"):
        build_submission(np.zeros((WIDTH, WIDTH)), np.zeros(WIDTH), URL)


def test_missing_writeup_url_is_rejected():
    with pytest.raises(ValueError, match="writeup_url is required"):
        build_submission(np.zeros((WIDTH, WIDTH)), np.ones(WIDTH), "")
