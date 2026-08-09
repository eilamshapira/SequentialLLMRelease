"""Search the equilibrium cache for showcase transitions and render them.

Scans all transitions from base subsets of size 4 (the size used in the
paper's Figure 1 worked example) for the most extreme payoff/designer
changes and for "tradeoff" cases: one player gains, the other loses, the
added strategy is never played, both equilibria are pure, and the
fairness-gap (or efficiency-sum) moves consistently with the designer
metric. Each selected case is rendered with pipeline.infographics.

Requires the results_{family}_{suffix}.pkl caches produced by
pipeline/comprehensive_analysis.py.
"""

import argparse
import itertools
import os
import pickle

import numpy as np

from core.data_manager import load_data, get_game
from core.simulation import evaluate_market, get_subset
from core.utils import get_bob_strategy, results_suffix
from pipeline.infographics import visualize_transition

# The paper's Figure 1 example uses a base set of exactly 4 strategies.
BASE_SUBSET_SIZE = 4


def get_strategy_usage(equilibrium_profile, strategies, target_strategy):
    """Return the probability of target_strategy in the equilibrium profile.

    strategies must match the order of the profile vectors (sorted names).
    """
    if equilibrium_profile is None:
        return 0.0

    sigma_alice, sigma_bob = equilibrium_profile
    sorted_strats = sorted(strategies)

    # Alice usage
    alice_usage = 0.0
    try:
        idx = sorted_strats.index(target_strategy)
        alice_usage = sigma_alice[idx]
    except ValueError:
        pass

    # Bob usage: distinct Alice names can map to the same Bob name, so sum
    # the probability over every matching index.
    bob_usage = 0.0
    bob_target_strategy = get_bob_strategy(target_strategy)
    bob_strategies = [get_bob_strategy(s) for s in sorted_strats]
    if bob_target_strategy in bob_strategies:
        indices = [i for i, s in enumerate(bob_strategies) if s == bob_target_strategy]
        for idx in indices:
            bob_usage += sigma_bob[idx]

    return max(alice_usage, bob_usage)


def is_pure_profile(profile, threshold=0.99):
    """Check whether the equilibrium profile is (nearly) a pure strategy pair."""
    if profile is None:
        return False
    sigma_alice, sigma_bob = profile
    return np.max(sigma_alice) >= threshold and np.max(sigma_bob) >= threshold


def find_extreme_cases(results_cache, alice_strategies, data, mixed_mode="average", equilibrium_type="mixed", designer_metric="fairness"):
    extremes = {
        'alice_gain': {'max': -float('inf'), 'min': float('inf'), 'max_case': None, 'min_case': None},
        'bob_gain': {'max': -float('inf'), 'min': float('inf'), 'max_case': None, 'min_case': None},
        'designer_value': {'max': -float('inf'), 'min': float('inf'), 'max_case': None, 'min_case': None},
        'tradeoff_alice_wins': {'top_cases': []},
        'tradeoff_bob_wins': {'top_cases': []}
    }

    candidates_alice_wins = []
    candidates_bob_wins = []

    for combo in itertools.combinations(alice_strategies, BASE_SUBSET_SIZE):
        current_set = frozenset(combo)
        if current_set not in results_cache or results_cache[current_set] is None:
            continue
        current_res = results_cache[current_set]

        for strategy in alice_strategies:
            if strategy not in current_set:
                new_set = current_set | {strategy}
                if new_set not in results_cache or results_cache[new_set] is None:
                    continue
                new_res = results_cache[new_set]

                deltas = {
                    'alice_gain': new_res['alice_gain'] - current_res['alice_gain'],
                    'bob_gain': new_res['bob_gain'] - current_res['bob_gain'],
                    'designer_value': new_res['designer_value'] - current_res['designer_value']
                }

                tradeoff = deltas['alice_gain'] * deltas['bob_gain']

                # A "threat" case: the added strategy must not actually be played
                is_used = False
                if 'equilibrium_profile' in new_res:
                    usage = get_strategy_usage(new_res['equilibrium_profile'], list(new_set), strategy)
                    if usage > 0.01:
                        is_used = True

                # Both the base and the new equilibrium must be pure profiles
                is_pure_result = False
                if 'equilibrium_profile' in new_res:
                    is_pure_result = is_pure_profile(new_res['equilibrium_profile'])

                is_pure_base = False
                if 'equilibrium_profile' in current_res:
                    is_pure_base = is_pure_profile(current_res['equilibrium_profile'])

                # Consistency between the designer metric and the payoffs:
                # fairness must move opposite to the |Alice-Bob| gap (including
                # in the intermediate state when the market changes), and
                # efficiency must move with the Alice+Bob sum.
                delta_designer = deltas['designer_value']
                gap_condition_met = False

                if designer_metric == "fairness":
                    gap_before = abs(current_res['alice_gain'] - current_res['bob_gain'])
                    gap_after = abs(new_res['alice_gain'] - new_res['bob_gain'])
                    delta_gap = gap_after - gap_before

                    gap_condition_met = (delta_gap * delta_designer) < 0

                    if gap_condition_met and current_res['market'] != new_res['market']:
                        # Evaluate the intermediate state: new strategies, old market
                        alice_subset = list(current_set) + [strategy]
                        bob_subset = [get_bob_strategy(a) for a in alice_subset]
                        subset_metrics_inter = get_subset(get_game(data, current_res['market']),
                                                          {'alice': alice_subset, 'bob': bob_subset})
                        inter = evaluate_market(subset_metrics_inter, designer_metric=designer_metric,
                                                equilibrium_type=equilibrium_type, mixed_mode=mixed_mode)

                        if inter is not None:
                            gap_inter = abs(inter['alice_gain'] - inter['bob_gain'])

                            if delta_designer < 0:  # Damage
                                # Gap(Inter) must be LARGER than both Before and After
                                if not (gap_inter > gap_before and gap_inter > gap_after):
                                    gap_condition_met = False
                            elif delta_designer > 0:  # Improvement
                                # Gap(Inter) must be SMALLER than both Before and After
                                if not (gap_inter < gap_before and gap_inter < gap_after):
                                    gap_condition_met = False
                        else:
                            # No equilibrium in the intermediate state
                            gap_condition_met = False

                elif designer_metric == "efficiency":
                    sum_before = current_res['alice_gain'] + current_res['bob_gain']
                    sum_after = new_res['alice_gain'] + new_res['bob_gain']
                    delta_sum = sum_after - sum_before
                    gap_condition_met = (delta_sum * delta_designer) > 0

                else:
                    gap_condition_met = True

                if not is_used and is_pure_result and is_pure_base and gap_condition_met:
                    # Alice wins (Alice +, Bob -)
                    if deltas['alice_gain'] > 0 and deltas['bob_gain'] < 0:
                        candidates_alice_wins.append({
                            'base': list(current_set),
                            'added': strategy,
                            'delta': tradeoff
                        })

                    # Bob wins (Alice -, Bob +)
                    if deltas['alice_gain'] < 0 and deltas['bob_gain'] > 0:
                        candidates_bob_wins.append({
                            'base': list(current_set),
                            'added': strategy,
                            'delta': tradeoff
                        })

                for metric in ['alice_gain', 'bob_gain', 'designer_value']:
                    delta = deltas[metric]
                    if delta > extremes[metric]['max']:
                        extremes[metric]['max'] = delta
                        extremes[metric]['max_case'] = {
                            'base': list(current_set),
                            'added': strategy,
                            'delta': delta
                        }
                    if delta < extremes[metric]['min']:
                        extremes[metric]['min'] = delta
                        extremes[metric]['min_case'] = {
                            'base': list(current_set),
                            'added': strategy,
                            'delta': delta
                        }

    # Sort and keep top 50 (most negative product -> ascending sort)
    candidates_alice_wins.sort(key=lambda x: x['delta'])
    extremes['tradeoff_alice_wins']['top_cases'] = candidates_alice_wins[:50]

    candidates_bob_wins.sort(key=lambda x: x['delta'])
    extremes['tradeoff_bob_wins']['top_cases'] = candidates_bob_wins[:50]

    return extremes


def main():
    parser = argparse.ArgumentParser(
        prog='python -m pipeline.find_extreme_cases',
        description='Find and render showcase transitions from the equilibrium cache.')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed')
    parser.add_argument('--families', type=str, default='bargaining,negotiation,persuasion', help='Comma-separated list of families')
    parser.add_argument('--metric', choices=['fairness', 'efficiency'], default='fairness', help='Regulator metric')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average', help='Aggregation over mixed equilibria (paper: average)')
    args = parser.parse_args()

    output_dir = f"output/{args.metric}/visualizations"
    suffix = results_suffix(args.mode, args.mixed_mode)

    for family in [f.strip() for f in args.families.split(',')]:
        cache_file = f"output/{args.metric}/calculations/results_{family}_{suffix}.pkl"
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"{cache_file} not found — run `make run-analysis METRIC={args.metric}` first.")

        print(f"Loading results for {family} ({args.mode})...")
        with open(cache_file, 'rb') as f:
            results_cache = pickle.load(f)

        data = load_data(f"data/{family}.csv")
        alice_strategies = sorted(data['alice'].unique())

        extremes = find_extreme_cases(results_cache, alice_strategies, data, mixed_mode=args.mixed_mode, equilibrium_type=args.mode, designer_metric=args.metric)

        cases_to_viz = []
        metrics_map = {'alice_gain': 'Alice', 'bob_gain': 'Bob', 'designer_value': 'Designer'}

        for metric, name in metrics_map.items():
            if extremes[metric]['max_case']:
                cases_to_viz.append({
                    'case_name': f"{name}_Improvement",
                    'base': extremes[metric]['max_case']['base'],
                    'added': extremes[metric]['max_case']['added']
                })
            if extremes[metric]['min_case']:
                cases_to_viz.append({
                    'case_name': f"{name}_Damage",
                    'base': extremes[metric]['min_case']['base'],
                    'added': extremes[metric]['min_case']['added']
                })

        # Tradeoff cases: try candidates in order until one renders successfully
        for case_key, case_label in [('tradeoff_alice_wins', 'Tradeoff_Alice_Wins'),
                                     ('tradeoff_bob_wins', 'Tradeoff_Bob_Wins')]:
            done = False
            candidates = extremes[case_key]['top_cases']
            print(f"Checking {len(candidates)} candidates for {case_label}...")
            for i, case_info in enumerate(candidates):
                success = visualize_transition(
                    family, case_info['base'], case_info['added'],
                    case_label,
                    output_dir=output_dir,
                    equilibrium_type=args.mode,
                    designer_metric=args.metric,
                    mixed_mode=args.mixed_mode
                )
                if success:
                    print(f"Successfully visualized {case_label} (Candidate {i+1})")
                    done = True
                    break
            if not done:
                print(f"No valid {case_label} case found.")

        for case in cases_to_viz:
            visualize_transition(family, case['base'], case['added'], case['case_name'], output_dir=output_dir, equilibrium_type=args.mode, designer_metric=args.metric, mixed_mode=args.mixed_mode)


if __name__ == '__main__':
    main()
