"""Nash equilibrium solvers used by the paper pipeline.

Pure equilibria are found via mutual best responses. Mixed equilibria can be
enumerated with nashpy (support or vertex enumeration) or Gambit's
``enummixed`` extreme-point enumerator. Alice plays the rows and Bob the
columns. The production pipeline uses Gambit for every mixed game.
"""

import numpy as np
import nashpy as nash


def ensure_solver_available(backend):
    """Fail early when an explicitly required optional backend is missing."""
    if backend != "gambit":
        return
    try:
        import pygambit  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The Gambit backend requires pygambit. Install requirements.txt "
            "before running the paper pipeline."
        ) from exc


def find_pure_nash_equilibrium(alice_self_gain, bob_self_gain):
    """Find all pure Nash equilibria via mutual best responses.

    Returns a list of (alice_strategy, bob_strategy, alice_payoff, bob_payoff).
    """
    # Convert to numpy arrays for performance
    alice_arr = alice_self_gain.to_numpy()
    bob_arr = bob_self_gain.to_numpy()

    # Find best responses for Alice (max in each column)
    alice_max = alice_arr.max(axis=0)
    alice_best_response = (alice_arr == alice_max)

    # Find best responses for Bob (max in each row)
    bob_max = bob_arr.max(axis=1)
    bob_best_response = (bob_arr == bob_max[:, None])

    # Nash equilibrium is where both are best responses
    nash_indices = np.where(alice_best_response & bob_best_response)

    nash_equilibria = []
    for i, j in zip(*nash_indices):
        nash_equilibria.append((
            alice_self_gain.index[i],
            bob_self_gain.columns[j],
            alice_arr[i, j],
            bob_arr[i, j]
        ))

    return nash_equilibria


def _gambit_mixed_equilibria(alice_arr, bob_arr):
    """Enumerate extreme mixed-equilibrium profiles and return NumPy arrays."""
    ensure_solver_available("gambit")
    import pygambit as gambit

    result = gambit.nash.enummixed_solve(
        gambit.Game.from_arrays(alice_arr, bob_arr), rational=False
    )
    equilibria = []
    for profile in result.equilibria:
        strategies = list(profile.mixed_strategies())
        player_a, mixed_a = strategies[0]
        player_b, mixed_b = strategies[1]
        sigma_alice = np.asarray(
            [mixed_a[strategy] for strategy in player_a.strategies], dtype=float
        )
        sigma_bob = np.asarray(
            [mixed_b[strategy] for strategy in player_b.strategies], dtype=float
        )
        equilibria.append((sigma_alice, sigma_bob))
    return equilibria


def find_mixed_nash_equilibrium(alice_self_gain, bob_self_gain, mode="average",
                                solver_backend="gambit"):
    """Find mixed Nash equilibria with the requested enumeration backend.

    mode='any' returns the first equilibrium found; mode='average' returns
    all equilibria (aggregation happens downstream). Each equilibrium is a
    (sigma_alice, sigma_bob) pair of probability vectors.
    """
    alice_arr = alice_self_gain.to_numpy()
    bob_arr = bob_self_gain.to_numpy()
    if solver_backend == "support":
        equilibria = nash.Game(alice_arr, bob_arr).support_enumeration()
    elif solver_backend == "vertex":
        equilibria = nash.Game(alice_arr, bob_arr).vertex_enumeration()
    elif solver_backend == "gambit":
        equilibria = iter(_gambit_mixed_equilibria(alice_arr, bob_arr))
    else:
        raise ValueError(f"Unknown mixed-equilibrium backend: {solver_backend}")

    if mode == "any":
        for eq in equilibria:
            return [eq]  # return the first equilibrium found as a list
        return []
    elif mode == "average":
        return list(equilibria)
    else:
        raise ValueError(f"Unknown mixed equilibrium mode: {mode}")


def find_equilibrium(alice_self_gain, bob_self_gain, equilibrium_type="mixed",
                     mixed_mode="average", solver_backend="gambit"):
    """Dispatch to the pure or mixed solver."""
    if equilibrium_type == "pure":
        return find_pure_nash_equilibrium(alice_self_gain, bob_self_gain)
    elif equilibrium_type == "mixed":
        return find_mixed_nash_equilibrium(
            alice_self_gain, bob_self_gain, mode=mixed_mode,
            solver_backend=solver_backend
        )
    else:
        raise ValueError(f"Unknown equilibrium type: {equilibrium_type}")
