"""Main computation engine of the paper.

For every subset of the available strategies (sizes 2..N), compute the
Nash equilibria in every market, let the regulator pick the market that
maximizes its metric, and record every "release" transition (adding one
strategy to a subset): payoff and designer-metric deltas, market changes,
and the equilibrium probability of the added strategy.

Outputs (per family, under output/{metric}/calculations/):
    results_{family}_{suffix}.pkl                  equilibrium cache
    comprehensive_transitions_{family}_{suffix}.csv  one row per transition
"""

import itertools
import pandas as pd
import time
import os
import pickle
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

from core.data_manager import load_data, get_all_markets, get_game
from core.simulation import game_with_designer_from_games
from core.solver import ensure_solver_available
from core.utils import get_bob_strategy, results_suffix

# Subsets between checkpoint saves of the pickle cache. A full run is
# hundreds of CPU-hours; checkpointing makes it resumable after a kill.
CHECKPOINT_EVERY = 256

# Sentinel stored in a worker's result when the subset raised an exception
# (as opposed to legitimately having no equilibrium in pure mode).
ERROR_KEY = '__error__'


def process_subset(combo, games, equilibrium_type, metric, mixed_mode="average",
                   solver_backend="gambit"):
    try:
        alice_subset = list(combo)
        bob_subset = [get_bob_strategy(a) for a in alice_subset]
        subset_of_strategies = {'alice': alice_subset, 'bob': bob_subset}

        best_result = game_with_designer_from_games(
            games, subset_of_strategies=subset_of_strategies,
            designer_metric=metric, equilibrium_type=equilibrium_type,
            mixed_mode=mixed_mode, solver_backend=solver_backend
        )

        if best_result:
            return frozenset(alice_subset), {
                'market': best_result['market'],
                'designer_value': best_result['designer_values'],
                'alice_gain': best_result['alice_self_gain'],
                'bob_gain': best_result['bob_self_gain'],
                'equilibrium_profile': best_result['equilibrium_profile'],
                'num_equilibria': best_result.get('num_equilibria', 1)
            }
        return frozenset(alice_subset), None
    except Exception as e:
        return frozenset(combo), {ERROR_KEY: f"{type(e).__name__}: {e}"}

def analyze_family(family, mode, mixed_mode, n_jobs, metric, solver_backend="gambit"):
    print(f"\n\n=== Analyzing Family: {family} (Mode: {mode}, Mixed Mode: {mixed_mode}, "
          f"Metric: {metric}, Solver: {solver_backend}) ===")

    if mode == 'mixed' and solver_backend == 'gambit':
        ensure_solver_available('gambit')

    data_path = f"data/{family}.csv"
    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found — run `make prepare-data` first (or place the family CSV there).")
    data = load_data(data_path)
    alice_strategies = sorted(data['alice'].unique())

    # Pivot each market's payoff matrices once; every subset only slices them.
    games = {market: get_game(data, market) for market in get_all_markets(data)}

    output_dir = f"output/{metric}/calculations"
    os.makedirs(output_dir, exist_ok=True)

    suffix = results_suffix(mode, mixed_mode)
    cache_file = f"{output_dir}/results_{family}_{suffix}.pkl"
    results_cache = {}

    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}...")
        with open(cache_file, 'rb') as f:
            results_cache = pickle.load(f)

    # Generate all combinations, largest first (they dominate the runtime)
    n_strategies = len(alice_strategies)
    all_combinations = []
    for k in range(n_strategies, 1, -1):
        all_combinations.extend(list(itertools.combinations(alice_strategies, k)))

    print(f"Total subsets to evaluate: {len(all_combinations)}")

    # Filter out already computed
    to_compute = [c for c in all_combinations if frozenset(c) not in results_cache]
    print(f"Subsets to compute: {len(to_compute)}")

    if to_compute:
        print(f"Starting parallel computation with {n_jobs} cores "
              f"(checkpointing every {CHECKPOINT_EVERY} subsets)...")
        start_time = time.time()
        failures = []

        with tqdm(total=len(to_compute)) as progress:
            for start in range(0, len(to_compute), CHECKPOINT_EVERY):
                chunk = to_compute[start:start + CHECKPOINT_EVERY]
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_subset)(combo, games, mode, metric, mixed_mode,
                                            solver_backend) for combo in chunk
                )

                for key, val in results:
                    if val is not None and ERROR_KEY in val:
                        failures.append((key, val[ERROR_KEY]))
                        results_cache[key] = None
                    else:
                        results_cache[key] = val

                # Checkpoint so an interrupted run can resume from here
                with open(cache_file, 'wb') as f:
                    pickle.dump(results_cache, f)
                progress.update(len(chunk))

        print(f"Computation finished in {time.time() - start_time:.2f}s")
        if failures:
            for key, message in failures[:10]:
                print(f"  FAILED subset {sorted(key)}: {message}")
            raise SystemExit(f"{len(failures)} subsets raised errors — see messages above. "
                             f"Completed results were checkpointed to {cache_file}.")
    else:
        print("All results loaded from cache.")

    # Analyze marginal contributions
    print("Analyzing marginal contributions...")

    transitions = []

    extremes = {
        'alice_gain': {'max': -float('inf'), 'min': float('inf'), 'max_case': None, 'min_case': None},
        'bob_gain': {'max': -float('inf'), 'min': float('inf'), 'max_case': None, 'min_case': None},
        'designer_value': {'max': -float('inf'), 'min': float('inf'), 'max_case': None, 'min_case': None}
    }

    for k in range(2, n_strategies):
        combinations = list(itertools.combinations(alice_strategies, k))
        for combo in combinations:
            current_set = frozenset(combo)

            # If pure mode and no NE found, skip
            if current_set not in results_cache or results_cache[current_set] is None:
                continue

            current_res = results_cache[current_set]

            for strategy in alice_strategies:
                if strategy not in current_set:
                    new_set = current_set | {strategy}

                    # If pure mode and no NE found in new set, skip
                    if new_set not in results_cache or results_cache[new_set] is None:
                        continue

                    new_res = results_cache[new_set]

                    # Calculate probability of the added strategy being played
                    added_strategy_prob = 0.0
                    eq_profile = new_res.get('equilibrium_profile')

                    if eq_profile:
                        if mode == 'pure':
                            # eq_profile is (alice_strat, bob_strat, alice_payoff, bob_payoff)
                            if eq_profile[0] == strategy:
                                added_strategy_prob += 1.0

                            bob_strategy = get_bob_strategy(strategy)
                            if eq_profile[1] == bob_strategy:
                                added_strategy_prob += 1.0

                        elif mode == 'mixed':
                            # eq_profile is (sigma_alice, sigma_bob)
                            # sigma_alice corresponds to sorted(list(new_set))
                            sorted_new_set = sorted(list(new_set))
                            try:
                                idx = sorted_new_set.index(strategy)
                                added_strategy_prob += eq_profile[0][idx]
                                added_strategy_prob += eq_profile[1][idx]
                            except ValueError:
                                pass

                    deltas = {
                        'alice_gain': new_res['alice_gain'] - current_res['alice_gain'],
                        'bob_gain': new_res['bob_gain'] - current_res['bob_gain'],
                        'designer_value': new_res['designer_value'] - current_res['designer_value']
                    }

                    # Record transition
                    transitions.append({
                        'family': family,
                        'initial_size': k,
                        'base_subset': ";".join(sorted(list(current_set))),
                        'added_strategy': strategy,
                        'initial_market': current_res['market'],
                        'final_market': new_res['market'],
                        'market_changed': current_res['market'] != new_res['market'],
                        'initial_designer_value': current_res['designer_value'],
                        'final_designer_value': new_res['designer_value'],
                        'delta_designer_value': deltas['designer_value'],
                        'initial_alice_gain': current_res['alice_gain'],
                        'final_alice_gain': new_res['alice_gain'],
                        'delta_alice_gain': deltas['alice_gain'],
                        'initial_bob_gain': current_res['bob_gain'],
                        'final_bob_gain': new_res['bob_gain'],
                        'delta_bob_gain': deltas['bob_gain'],
                        'added_strategy_prob': added_strategy_prob
                    })

                    for delta_metric in ['alice_gain', 'bob_gain', 'designer_value']:
                        delta = deltas[delta_metric]

                        if delta > extremes[delta_metric]['max']:
                            extremes[delta_metric]['max'] = delta
                            extremes[delta_metric]['max_case'] = {
                                'base_subset': list(current_set),
                                'added_strategy': strategy,
                                'from_market': current_res['market'],
                                'to_market': new_res['market'],
                                'old_value': current_res[delta_metric],
                                'new_value': new_res[delta_metric],
                                'delta': delta
                            }

                        if delta < extremes[delta_metric]['min']:
                            extremes[delta_metric]['min'] = delta
                            extremes[delta_metric]['min_case'] = {
                                'base_subset': list(current_set),
                                'added_strategy': strategy,
                                'from_market': current_res['market'],
                                'to_market': new_res['market'],
                                'old_value': current_res[delta_metric],
                                'new_value': new_res[delta_metric],
                                'delta': delta
                            }

    # Save transitions to CSV
    transitions_df = pd.DataFrame(transitions)
    csv_file = f"{output_dir}/comprehensive_transitions_{family}_{suffix}.csv"
    print(f"Saving detailed transitions to {csv_file}...")
    transitions_df.to_csv(csv_file, index=False)

    print(f"\nResults for {family} ({suffix}):")
    metrics_map = {'alice_gain': 'Alice', 'bob_gain': 'Bob', 'designer_value': 'Designer'}

    for metric_name, name in metrics_map.items():
        print(f"\n--- {name} ---")

        imp = extremes[metric_name]['max_case']
        if imp:
            print(f"Most Significant Improvement (+{imp['delta']:.4f}):")
            print(f"  Added: {imp['added_strategy']}")
            print(f"  To subset (size {len(imp['base_subset'])}): {imp['base_subset']}")
            print(f"  Market change: {imp['from_market']} -> {imp['to_market']}")
            print(f"  Value change: {imp['old_value']:.4f} -> {imp['new_value']:.4f}")
        else:
            print("No valid improvement cases found.")

        dam = extremes[metric_name]['min_case']
        if dam:
            print(f"Most Significant Damage ({dam['delta']:.4f}):")
            print(f"  Added: {dam['added_strategy']}")
            print(f"  To subset (size {len(dam['base_subset'])}): {dam['base_subset']}")
            print(f"  Market change: {dam['from_market']} -> {dam['to_market']}")
            print(f"  Value change: {dam['old_value']:.4f} -> {dam['new_value']:.4f}")
        else:
            print("No valid damage cases found.")

def main():
    parser = argparse.ArgumentParser(
        prog='python -m pipeline.comprehensive_analysis',
        description='Compute equilibria for all strategy subsets and record release transitions.')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed', help='Equilibrium type')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average', help='Aggregation over mixed equilibria (paper: average)')
    parser.add_argument('--cores', type=int, default=-1, help='Number of cores to use (-1 for all)')
    parser.add_argument('--families', type=str, default='bargaining,negotiation,persuasion', help='Comma-separated list of families to analyze')
    parser.add_argument('--metric', choices=['fairness', 'efficiency'], default='fairness', help='Regulator metric')
    parser.add_argument('--solver_backend', choices=['support', 'vertex', 'gambit'],
                        default='gambit',
                        help='Mixed-equilibrium backend. The paper uses Gambit for every cell.')
    args = parser.parse_args()

    families = args.families.split(',')
    for family in families:
        analyze_family(family.strip(), args.mode, args.mixed_mode, args.cores,
                       args.metric, args.solver_backend)

if __name__ == '__main__':
    main()
