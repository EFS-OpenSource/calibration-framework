"""Bins below sample_threshold are left out of ACE and ECE (issue #67)."""

import numpy as np
import pytest

from netcal.metrics import ACE, ECE


def binary_example():
    """Four confidence bins of [0, 1]: two sparse bins (2 samples, |acc - conf| = 0.9) and two full ones (100
    samples, |acc - conf| = 0.1)."""
    X = np.concatenate([np.full(2, 0.1), np.full(100, 0.35), np.full(100, 0.6), np.full(2, 0.9)])
    y = np.concatenate([np.ones(2), np.r_[np.ones(25), np.zeros(75)], np.r_[np.ones(70), np.zeros(30)], np.zeros(2)])
    return X, y


@pytest.mark.parametrize("metric", [ACE, ECE])
def test_sparse_bins_are_left_out(metric):
    X, y = binary_example()
    assert metric(bins=4, sample_threshold=3).measure(X, y) == pytest.approx(0.1)


@pytest.mark.parametrize("metric", [ACE, ECE])
def test_threshold_matches_removing_the_sparse_bins(metric):
    X, y = binary_example()
    keep = (X > 0.25) & (X < 0.75)
    with_threshold = metric(bins=4, sample_threshold=3).measure(X, y)
    removed = metric(bins=4, sample_threshold=1).measure(X[keep], y[keep])
    assert with_threshold == pytest.approx(removed)


def test_default_threshold_is_unchanged():
    X, y = binary_example()
    assert ACE(bins=4).measure(X, y) == pytest.approx(0.5)
    assert ECE(bins=4).measure(X, y) == pytest.approx(23.6 / 204)


@pytest.mark.parametrize("metric", [ACE, ECE])
def test_every_bin_below_the_threshold_gives_zero(metric):
    X, y = binary_example()
    assert metric(bins=4, sample_threshold=1000).measure(X, y) == 0.

