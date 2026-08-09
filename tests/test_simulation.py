"""Tests for equilibrium aggregation in the regulator's objective."""

import numpy as np
import pandas as pd

from core.simulation import evaluate_market


def matrix(values):
    names = ['a', 'b']
    return pd.DataFrame(np.asarray(values, dtype=float), index=names, columns=names)


def test_average_mode_evaluates_then_averages(monkeypatch):
    """Guard against multiplying independently averaged strategy profiles.

    The two equilibria are (a,a) and (b,b). Their designer values are both
    zero, but multiplying the averaged profiles would introduce off-diagonal
    cross-equilibrium terms and incorrectly return 50.
    """
    equilibria = [
        (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    ]
    monkeypatch.setattr('core.simulation.find_equilibrium',
                        lambda *args, **kwargs: equilibria)
    metrics = {
        'alice_self_gain': matrix([[1, 0], [0, 3]]),
        'bob_self_gain': matrix([[2, 0], [0, 4]]),
        'fairness': matrix([[0, 100], [100, 0]]),
    }

    result = evaluate_market(metrics, designer_metric='fairness',
                             mixed_mode='average')

    assert result['designer_value'] == 0.0
    assert result['alice_gain'] == 2.0
    assert result['bob_gain'] == 3.0
    np.testing.assert_allclose(result['profile'][0], [0.5, 0.5])
    np.testing.assert_allclose(result['profile'][1], [0.5, 0.5])
    averaged_profile_value = result['profile'][0] @ metrics['fairness'].to_numpy() @ result['profile'][1]
    assert averaged_profile_value == 50.0
