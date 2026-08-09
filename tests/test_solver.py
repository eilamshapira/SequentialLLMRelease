"""Unit tests for the Nash equilibrium solvers on games with known solutions."""

import numpy as np
import pandas as pd
import pytest

from core.solver import (find_equilibrium, find_pure_nash_equilibrium,
                         find_mixed_nash_equilibrium)


def matrix(values, names):
    return pd.DataFrame(np.array(values, dtype=float), index=names, columns=names)


def test_pure_equilibrium_prisoners_dilemma():
    # Defect strictly dominates: the unique pure NE is (defect, defect)
    names = ['cooperate', 'defect']
    alice = matrix([[3, 0], [5, 1]], names)
    bob = matrix([[3, 5], [0, 1]], names)

    equilibria = find_pure_nash_equilibrium(alice, bob)

    assert len(equilibria) == 1
    eq_alice, eq_bob, payoff_alice, payoff_bob = equilibria[0]
    assert eq_alice == 'defect'
    assert eq_bob == 'defect'
    assert payoff_alice == 1
    assert payoff_bob == 1


def test_pure_equilibrium_coordination_game():
    # Two pure equilibria on the diagonal
    names = ['a', 'b']
    alice = matrix([[2, 0], [0, 1]], names)
    bob = matrix([[2, 0], [0, 1]], names)

    equilibria = find_pure_nash_equilibrium(alice, bob)

    profiles = {(eq[0], eq[1]) for eq in equilibria}
    assert profiles == {('a', 'a'), ('b', 'b')}


def test_mixed_equilibrium_matching_pennies():
    # The unique equilibrium mixes 50/50 for both players
    names = ['heads', 'tails']
    alice = matrix([[1, -1], [-1, 1]], names)
    bob = matrix([[-1, 1], [1, -1]], names)

    equilibria = find_mixed_nash_equilibrium(
        alice, bob, mode='average', solver_backend='support'
    )

    assert len(equilibria) == 1
    sigma_alice, sigma_bob = equilibria[0]
    np.testing.assert_allclose(sigma_alice, [0.5, 0.5])
    np.testing.assert_allclose(sigma_bob, [0.5, 0.5])


def test_gambit_equilibrium_matching_pennies():
    pytest.importorskip('pygambit')
    names = ['heads', 'tails']
    alice = matrix([[1, -1], [-1, 1]], names)
    bob = matrix([[-1, 1], [1, -1]], names)

    equilibria = find_mixed_nash_equilibrium(
        alice, bob, mode='average', solver_backend='gambit'
    )

    assert len(equilibria) == 1
    np.testing.assert_allclose(equilibria[0][0], [0.5, 0.5])
    np.testing.assert_allclose(equilibria[0][1], [0.5, 0.5])


def test_vertex_equilibrium_matching_pennies():
    names = ['heads', 'tails']
    alice = matrix([[1, -1], [-1, 1]], names)
    bob = matrix([[-1, 1], [1, -1]], names)

    equilibria = find_mixed_nash_equilibrium(
        alice, bob, mode='average', solver_backend='vertex'
    )

    assert len(equilibria) == 1
    np.testing.assert_allclose(equilibria[0][0], [0.5, 0.5])
    np.testing.assert_allclose(equilibria[0][1], [0.5, 0.5])


def test_unknown_solver_backend_is_rejected():
    names = ['a', 'b']
    payoff = matrix([[1, 0], [0, 1]], names)
    with pytest.raises(ValueError, match='Unknown mixed-equilibrium backend'):
        find_mixed_nash_equilibrium(payoff, payoff, solver_backend='typo')


def test_mixed_any_returns_single_equilibrium():
    names = ['a', 'b']
    alice = matrix([[2, 0], [0, 1]], names)
    bob = matrix([[2, 0], [0, 1]], names)

    any_eq = find_equilibrium(alice, bob, equilibrium_type='mixed', mixed_mode='any')
    all_eq = find_equilibrium(alice, bob, equilibrium_type='mixed', mixed_mode='average')

    assert len(any_eq) == 1
    # Coordination game: two pure + one mixed equilibrium
    assert len(all_eq) == 3


def test_find_equilibrium_rejects_unknown_type():
    names = ['a', 'b']
    alice = matrix([[1, 0], [0, 1]], names)
    with pytest.raises(ValueError):
        find_equilibrium(alice, alice, equilibrium_type='no_such_type')
