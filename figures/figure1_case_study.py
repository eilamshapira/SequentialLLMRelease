"""
Reproduce the numbers behind paper Figure 1 (and SI Tables 1-2): the
worked Poisoned Apple example in Bargaining under a fairness-maximizing
regulator.

Prints, for the base strategy set and for the set with the added
strategy, the designer value obtained in every market (the regulator
picks the argmax), the players' payoffs in the chosen market, and the
equilibrium probability of the added strategy.
"""

import argparse

from core.data_manager import load_data, get_all_markets, get_game
from core.simulation import evaluate_market, get_subset
from core.utils import get_bob_strategy

DEFAULT_BASE = [
    'vertex_ai/claude-3-7-sonnet@20250219',
    'vertex_ai/gemini-2.0-flash',
    'vertex_ai/meta/llama-3.1-405b-instruct-maas',
    'vertex_ai/meta/llama-3.3-70b-instruct-maas',
]
DEFAULT_ADDED = 'vertex_ai/gemini-1.5-pro'


def evaluate_all_markets(data, alice_subset, designer_metric, mode, mixed_mode):
    subset = {'alice': alice_subset,
              'bob': [get_bob_strategy(a) for a in alice_subset]}
    results = {}
    for market in get_all_markets(data):
        subset_metrics = get_subset(get_game(data, market), subset)
        result = evaluate_market(subset_metrics, designer_metric=designer_metric,
                                 equilibrium_type=mode, mixed_mode=mixed_mode)
        if result:
            results[market] = result
    return results


def main():
    parser = argparse.ArgumentParser(
        prog='python -m figures.figure1_case_study',
        description='Print the numbers behind paper Figure 1')
    parser.add_argument('--family', default='bargaining')
    parser.add_argument('--metric', default='fairness', choices=['fairness', 'efficiency'])
    parser.add_argument('--mode', default='mixed', choices=['pure', 'mixed'])
    parser.add_argument('--mixed_mode', default='average', choices=['any', 'average'])
    parser.add_argument('--base', default=','.join(DEFAULT_BASE),
                        help='Comma-separated base strategy set (Alice naming)')
    parser.add_argument('--added', default=DEFAULT_ADDED,
                        help='Strategy released into the market')
    args = parser.parse_args()

    base = sorted(s.strip() for s in args.base.split(','))
    extended = sorted(base + [args.added])

    data = load_data(f'data/{args.family}.csv')

    pre = evaluate_all_markets(data, base, args.metric, args.mode, args.mixed_mode)
    post = evaluate_all_markets(data, extended, args.metric, args.mode, args.mixed_mode)
    if not pre or not post:
        raise SystemExit("No market has an equilibrium for the requested strategy sets.")

    print(f"Family: {args.family} | regulator metric: {args.metric} "
          f"| mode: {args.mode}/{args.mixed_mode}")
    print(f"Base set ({len(base)}): " + ", ".join(base))
    print(f"Added strategy: {args.added}\n")

    print(f"{args.metric.capitalize()} per market (regulator picks the maximum):")
    print(f"{'Market':38} {'Pre-release':>12} {'Post-release':>13}")
    for market in pre:
        print(f"{market:38} {pre[market]['designer_value']:>12.6f} "
              f"{post.get(market, {'designer_value': float('nan')})['designer_value']:>13.6f}")

    best_pre = max(pre, key=lambda m: pre[m]['designer_value'])
    best_post = max(post, key=lambda m: post[m]['designer_value'])

    added_prob = 0.0
    if args.mode == 'mixed':
        idx = extended.index(args.added)
        profile = post[best_post]['profile']
        added_prob = profile[0][idx] + profile[1][idx]
    else:
        profile = post[best_post]['profile']
        added_prob = float(profile[0] == args.added) + \
            float(profile[1] == get_bob_strategy(args.added))

    print(f"\nRegulator's choice pre-release:  {best_pre}")
    print(f"  {args.metric}: {pre[best_pre]['designer_value']:.6f} | "
          f"Alice: {pre[best_pre]['alice_gain']:.6f} | Bob: {pre[best_pre]['bob_gain']:.6f}")
    print(f"Regulator's choice post-release: {best_post}")
    print(f"  {args.metric}: {post[best_post]['designer_value']:.6f} | "
          f"Alice: {post[best_post]['alice_gain']:.6f} | Bob: {post[best_post]['bob_gain']:.6f}")
    print(f"\nMarket changed: {best_pre != best_post}")
    print(f"Delta {args.metric}: {post[best_post]['designer_value'] - pre[best_pre]['designer_value']:+.6f}")
    print(f"Delta Alice: {post[best_post]['alice_gain'] - pre[best_pre]['alice_gain']:+.6f}")
    print(f"Delta Bob:   {post[best_post]['bob_gain'] - pre[best_pre]['bob_gain']:+.6f}")
    delta_alice = post[best_post]['alice_gain'] - pre[best_pre]['alice_gain']
    delta_bob = post[best_post]['bob_gain'] - pre[best_pre]['bob_gain']
    is_opposite = delta_alice * delta_bob < -1e-18
    is_poisoned_apple = best_pre != best_post and is_opposite and added_prob < 1e-9
    print(f"Combined adoption weight of added strategy (Alice + Bob, range 0-2): {added_prob:.6f}"
          + ("  (Poisoned Apple: market change and opposing payoff shifts without adoption)"
             if is_poisoned_apple else ""))

    print(f"\nOld-market outcome after release (regulatory inertia): "
          f"{args.metric} in {best_pre} becomes {post[best_pre]['designer_value']:.6f}")


if __name__ == '__main__':
    main()
