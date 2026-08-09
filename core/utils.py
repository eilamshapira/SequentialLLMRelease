"""Shared helpers: strategy naming, run-suffix convention, Wilson CIs."""

import os

import numpy as np
import pandas as pd
from scipy import stats


def results_suffix(mode, mixed_mode):
    """File-name suffix shared by every artifact of one configuration.

    'pure' -> 'pure'; ('mixed', 'average') -> 'mixed_average'; etc.
    """
    if mode == 'mixed':
        return f"{mode}_{mixed_mode}"
    return mode


def load_transitions(family, mode, mixed_mode, metric):
    """Load one family's release-transition CSV; raise a clear error if absent."""
    suffix = results_suffix(mode, mixed_mode)
    filename = f"output/{metric}/calculations/comprehensive_transitions_{family}_{suffix}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} not found — run `make run-analysis METRIC={metric}` first.")
    return pd.read_csv(filename)


def wilson_ci(successes, total, confidence=0.95):
    """Wilson score confidence interval for a proportion, clipped to [0, 1]."""
    if total == 0:
        return 0, 0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def get_alice_and_bob_names(pair_string):
    """Split a composite 'alice_<model>_bob_<model>' string into the two model names."""
    alice, bob = pair_string.split('_bob_')
    alice = alice.replace('alice_', '')
    return alice, bob


def get_bob_strategy(alice_strat):
    """Map an Alice strategy name to its Bob counterpart.

    In the GLEE data, Alice model names may carry provider prefixes
    ('vertex_ai/', 'xai/') that Bob names do not.
    """
    bob_strat = alice_strat
    if bob_strat.startswith('vertex_ai/'):
        bob_strat = bob_strat.replace('vertex_ai/', '')
    if bob_strat.startswith('xai/'):
        bob_strat = bob_strat.replace('xai/', '')
    return bob_strat
