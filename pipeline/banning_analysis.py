"""
Banning Regulator Analysis.

Analyzes how the Poisoned Apple effect changes when the regulator can ban
up to N models (in addition to selecting the market). Uses the existing
pickle caches from comprehensive_analysis.py — no new equilibrium
computations are needed.

DP recurrence:
    best_result(S, 0) = cache[S]
    best_result(S, N) = max(best_result(S, N-1), max_{s in S} best_result(S\\{s}, N-1))
"""

import itertools
import os
import pickle
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from core.data_manager import load_data, get_all_markets, get_game
from core.utils import get_bob_strategy, results_suffix, wilson_ci


def compute_size1_results(data, alice_strategies, metric):
    """Compute trivial 1x1 game results for single-strategy subsets.

    For each strategy, both players must use it. The regulator picks
    the market maximizing the designer metric at that single cell.
    """
    markets = get_all_markets(data)
    results = {}

    for alice_strat in alice_strategies:
        bob_strat = get_bob_strategy(alice_strat)
        best_designer = -float('inf')
        best_result = None

        for market in markets:
            game_data = get_game(data, market)
            try:
                designer_val = game_data[metric].loc[alice_strat, bob_strat]
                alice_gain = game_data['alice_self_gain'].loc[alice_strat, bob_strat]
                bob_gain = game_data['bob_self_gain'].loc[alice_strat, bob_strat]
            except KeyError:
                continue

            if designer_val > best_designer:
                best_designer = designer_val
                best_result = {
                    'market': market,
                    'designer_value': designer_val,
                    'alice_gain': alice_gain,
                    'bob_gain': bob_gain,
                    'equilibrium_profile': (np.array([1.0]), np.array([1.0])),
                    'num_equilibria': 1,
                }

        results[frozenset([alice_strat])] = best_result

    return results


def load_cache_with_size1(family, metric, mode, mixed_mode):
    """Load pickle cache and augment with size-1 results."""
    suffix = results_suffix(mode, mixed_mode)

    cache_file = f"output/{metric}/calculations/results_{family}_{suffix}.pkl"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache not found: {cache_file}. Run pipeline/comprehensive_analysis.py first.")

    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    # Compute and add size-1 results
    data = load_data(f"data/{family}.csv")
    alice_strategies = sorted(data['alice'].unique())
    size1 = compute_size1_results(data, alice_strategies, metric)
    cache.update(size1)

    return cache, alice_strategies


def precompute_best_results(cache, max_bans):
    """DP to find optimal (subset, ban_set) for every (S, N) pair.

    Returns dict {(frozenset_S, int_N): {'result_key': frozenset, 'banned': frozenset}}
    Memory-efficient: stores references to cache entries, not copies.
    """
    dp = {}

    # Step 1: Initialize N=0 for all subsets in cache
    for S, result in cache.items():
        if result is None:
            dp[(S, 0)] = None
        else:
            dp[(S, 0)] = {'result_key': S, 'banned': frozenset()}

    # Step 2: Fill DP for N=1..max_bans
    all_subsets = list(cache.keys())
    for N in range(1, max_bans + 1):
        for S in all_subsets:
            size_S = len(S)

            # Clamp: can't ban below 1 model
            if N >= size_S:
                dp[(S, N)] = dp.get((S, size_S - 1))
                continue

            # Start with previous N (no additional ban)
            best_entry = dp.get((S, N - 1))
            best_dv = -float('inf')
            if best_entry is not None:
                best_result = cache.get(best_entry['result_key'])
                if best_result is not None:
                    best_dv = best_result['designer_value']

            # Try banning each strategy s in S. Iterate in sorted order so
            # ties in designer_value (common: banning an unplayed strategy
            # leaves the equilibrium unchanged) break deterministically
            # instead of by Python's per-process string-hash order.
            for s in sorted(S):
                sub = S - {s}
                candidate_entry = dp.get((sub, N - 1))
                if candidate_entry is None:
                    continue
                candidate_result = cache.get(candidate_entry['result_key'])
                if candidate_result is None:
                    continue
                if candidate_result['designer_value'] > best_dv:
                    best_dv = candidate_result['designer_value']
                    best_entry = {
                        'result_key': candidate_entry['result_key'],
                        'banned': S - candidate_entry['result_key'],
                    }

            dp[(S, N)] = best_entry

    return dp


def compute_added_strategy_prob(cache, dp_entry, added_strategy, mode):
    """Compute probability of the added strategy being played in the selected equilibrium."""
    if dp_entry is None:
        return 0.0

    selected_subset = dp_entry['result_key']

    # If the strategy was banned, probability is 0
    if added_strategy not in selected_subset:
        return 0.0

    result = cache.get(selected_subset)
    if result is None:
        return 0.0

    eq_profile = result.get('equilibrium_profile')
    if eq_profile is None:
        return 0.0

    prob = 0.0
    if mode == 'pure':
        if eq_profile[0] == added_strategy:
            prob += 1.0
        bob_strategy = get_bob_strategy(added_strategy)
        if eq_profile[1] == bob_strategy:
            prob += 1.0
    elif mode == 'mixed':
        sorted_subset = sorted(list(selected_subset))
        try:
            idx = sorted_subset.index(added_strategy)
            prob += eq_profile[0][idx]
            prob += eq_profile[1][idx]
        except (ValueError, IndexError):
            pass

    return prob


def _stat_row(base, panel_name, count, denom):
    """One output row: percentage with a Wilson 95% CI, or zeros when empty."""
    if denom == 0:
        return {**base, 'panel': panel_name, 'percentage': 0.0,
                'ci_low': 0.0, 'ci_high': 0.0, 'count': 0, 'total': 0}
    ci_lo, ci_hi = wilson_ci(count, denom)
    return {**base, 'panel': panel_name, 'percentage': count / denom * 100,
            'ci_low': ci_lo * 100, 'ci_high': ci_hi * 100,
            'count': count, 'total': denom}


def compute_stats(cache, dp_table, alice_strategies, family, mode, max_bans):
    """Poisoned Apple panel metrics per ban budget, in a single pass.

    Counts every transition once per (N, initial_size); the aggregate rows
    are the column sums of the stratified counters.

    Returns (aggregate_rows, stratified_rows).
    """
    aggregate_rows = []
    stratified_rows = []
    n_strategies = len(alice_strategies)

    for N in range(0, max_bans + 1):
        counters = defaultdict(lambda: defaultdict(int))

        for k in range(2, n_strategies):
            for combo in itertools.combinations(alice_strategies, k):
                current_set = frozenset(combo)

                before_entry = dp_table.get((current_set, N))
                if before_entry is None:
                    continue
                before_result = cache.get(before_entry['result_key'])
                if before_result is None:
                    continue

                for strategy in alice_strategies:
                    if strategy in current_set:
                        continue

                    new_set = current_set | {strategy}
                    after_entry = dp_table.get((new_set, N))
                    if after_entry is None:
                        continue
                    after_result = cache.get(after_entry['result_key'])
                    if after_result is None:
                        continue

                    d_alice = after_result['alice_gain'] - before_result['alice_gain']
                    d_bob = after_result['bob_gain'] - before_result['bob_gain']
                    d_designer = after_result['designer_value'] - before_result['designer_value']
                    prob = compute_added_strategy_prob(cache, after_entry, strategy, mode)

                    c = counters[k]
                    c['total'] += 1

                    # Panel A / B: opposite payoff changes, and the Poisoned
                    # Apple subset: market switch plus zero adoption.
                    is_opposite = ((d_alice > 1e-9 and d_bob < -1e-9) or
                                   (d_bob > 1e-9 and d_alice < -1e-9))
                    if is_opposite:
                        c['opposite'] += 1
                        if prob < 1e-9 and after_result['market'] != before_result['market']:
                            c['poisoned'] += 1

                    # Panels C-E: designer metric change and adoption within it
                    if d_designer > 1e-9:
                        c['designer_improve'] += 1
                        if prob > 1e-9:
                            c['adoption_when_improve'] += 1
                    elif d_designer < -1e-9:
                        c['designer_harm'] += 1
                        if prob > 1e-9:
                            c['adoption_when_harm'] += 1

                    # Panel F: market changed
                    if after_result['market'] != before_result['market']:
                        c['market_changed'] += 1

                    # Was the newly released strategy banned outright?
                    if strategy not in after_entry['result_key']:
                        c['banned'] += 1

        totals = defaultdict(int)
        for c in counters.values():
            for key, value in c.items():
                totals[key] += value

        agg_base = {'family': family, 'ban_budget': N}
        aggregate_rows.append(_stat_row(agg_base, 'opposite_payoff', totals['opposite'], totals['total']))
        aggregate_rows.append(_stat_row(agg_base, 'zero_adoption_opposite', totals['poisoned'], totals['opposite']))
        aggregate_rows.append(_stat_row(agg_base, 'designer_improve', totals['designer_improve'], totals['total']))
        aggregate_rows.append(_stat_row(agg_base, 'designer_harm', totals['designer_harm'], totals['total']))
        aggregate_rows.append(_stat_row(agg_base, 'adoption_when_improve', totals['adoption_when_improve'], totals['designer_improve']))
        aggregate_rows.append(_stat_row(agg_base, 'adoption_when_harm', totals['adoption_when_harm'], totals['designer_harm']))
        aggregate_rows.append(_stat_row(agg_base, 'market_changed', totals['market_changed'], totals['total']))
        aggregate_rows.append(_stat_row(agg_base, 'added_strategy_banned', totals['banned'], totals['total']))

        for k, c in counters.items():
            strat_base = {'family': family, 'ban_budget': N, 'initial_size': k}
            stratified_rows.append(_stat_row(strat_base, 'opposite_payoff', c['opposite'], c['total']))
            stratified_rows.append(_stat_row(strat_base, 'zero_adoption_opposite', c['poisoned'], c['opposite']))
            stratified_rows.append(_stat_row(strat_base, 'designer_improve', c['designer_improve'], c['total']))
            stratified_rows.append(_stat_row(strat_base, 'designer_harm', c['designer_harm'], c['total']))
            stratified_rows.append(_stat_row(strat_base, 'added_strategy_banned', c['banned'], c['total']))

        print(f"  N={N:2d}: {totals['total']} transitions, opposite={totals['opposite']}, "
              f"poisoned_apple={totals['poisoned']}, "
              f"designer_improve={totals['designer_improve']}, designer_harm={totals['designer_harm']}, "
              f"banned={totals['banned']}")

    return aggregate_rows, stratified_rows


def analyze_family(family, mode, mixed_mode, metric, max_bans):
    """Run banning regulator analysis for one family."""
    print(f"\n=== Banning Analysis: {family} (mode={mode}, mixed_mode={mixed_mode}, metric={metric}) ===")

    t0 = time.time()

    # Load cache + size-1 results
    print("Loading cache and computing size-1 results...")
    cache, alice_strategies = load_cache_with_size1(family, metric, mode, mixed_mode)
    print(f"  Cache: {len(cache)} subsets loaded ({time.time()-t0:.1f}s)")

    # DP precomputation
    print(f"Running DP precomputation (max_bans={max_bans})...")
    t1 = time.time()
    dp_table = precompute_best_results(cache, max_bans)
    print(f"  DP table: {len(dp_table)} entries ({time.time()-t1:.1f}s)")

    # One pass produces both the aggregate and the size-stratified statistics
    print("Computing statistics...")
    t2 = time.time()
    agg_rows, strat_rows = compute_stats(cache, dp_table, alice_strategies, family, mode, max_bans)
    print(f"  Statistics done ({time.time()-t2:.1f}s)")

    suffix = results_suffix(mode, mixed_mode)
    output_dir = f"output/{metric}/calculations"
    os.makedirs(output_dir, exist_ok=True)

    agg_path = f"{output_dir}/banning_summary_{family}_{suffix}.csv"
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    print(f"  Summary saved to {agg_path}")

    strat_path = f"{output_dir}/banning_stratified_{family}_{suffix}.csv"
    pd.DataFrame(strat_rows).to_csv(strat_path, index=False)
    print(f"  Stratified stats saved to {strat_path}")

    print(f"  Total time: {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        prog='python -m pipeline.banning_analysis',
        description='Banning Regulator Analysis')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average')
    parser.add_argument('--metric', choices=['fairness', 'efficiency'], default='fairness')
    parser.add_argument('--families', type=str, default='bargaining,negotiation,persuasion')
    parser.add_argument('--max_bans', type=int, default=12)
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(',')]
    for family in families:
        analyze_family(family, args.mode, args.mixed_mode, args.metric, args.max_bans)


if __name__ == '__main__':
    main()
