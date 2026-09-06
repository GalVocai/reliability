"""Unit tests for the display-only curve cleaning helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sensorlab.plotting._clean import abs_despiked


def test_abs_despiked_keeps_negatives_as_their_magnitude():
    """A log axis can't show negatives; the off-region current is noise around
    zero and used to be DISCARDED, leaving a gap. Now it's plotted as |Id|, so
    the noise floor stays visible. (4 points < the despike window -> the
    despiker returns early, isolating the |Id| behaviour under test.)"""
    x = np.array([-2.0, -1.5, -1.0, 0.0])
    y = np.array([-50.0, -10.0, 100.0, 1000.0])   # two NEGATIVE off-region samples

    px, py = abs_despiked(x, y)

    assert len(px) == 4                            # nothing dropped
    assert np.allclose(px, x)                      # x untouched
    assert np.allclose(py, [50.0, 10.0, 100.0, 1000.0])   # magnitudes, not signs


def test_abs_despiked_still_drops_exact_zeros():
    """log10(0) is undefined -- an exact zero has to go, |Id| or not."""
    x = np.array([-2.0, -1.5, -1.0, 0.0])
    y = np.array([0.0, -10.0, 100.0, 1000.0])

    px, py = abs_despiked(x, y)

    assert len(px) == 3
    assert np.allclose(px, [-1.5, -1.0, 0.0])
    assert np.allclose(py, [10.0, 100.0, 1000.0])


def test_abs_despiked_removes_an_isolated_log_spike():
    """The despiker still works, now on log10|Id|: a lone point sitting many
    decades off the local median is dropped."""
    x = np.linspace(-2.0, 0.3, 12)
    y = np.full(12, 1000.0)
    y[6] = 1e-4                       # isolated plunge -> ~7 decades below median

    px, py = abs_despiked(x, y)

    assert len(px) == 11              # the spike is gone
    assert 1e-4 not in py
    assert np.allclose(py, 1000.0)
