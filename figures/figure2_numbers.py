"""
Calculate all numbers in Figure 2 of the paper with Confidence Intervals.

This is the canonical source of every Figure 2 value: each panel function
computes one statistic from the transition CSVs, with Wilson 95% CIs.
"""

import argparse
import os

import pandas as pd

from core.utils import results_suffix, wilson_ci

FAMILIES = ['bargaining', 'negotiation', 'persuasion']


def load_transition_frames(metric, mixed_mode='average', families=None):
    """Load the per-family transition CSVs for a given regulator metric.

    Raises SystemExit listing any missing file — a silently missing family
    would otherwise produce a fabricated all-zero figure.
    """
    families = families or FAMILIES
    suffix = results_suffix('mixed', mixed_mode)

    data = {}
    missing = []
    for family in families:
        path = f'output/{metric}/calculations/comprehensive_transitions_{family}_{suffix}.csv'
        if os.path.exists(path):
            data[family] = pd.read_csv(path)
        else:
            missing.append(path)
    if missing:
        raise SystemExit("Missing transition CSVs (run `make run-analysis` for both metrics first):\n  "
                         + "\n  ".join(missing))
    return data

def panel_a(df):
    """
    Panel A: Frequency of Opposite Payoff Changes
    Cases where one player gains and the other loses.
    """
    if df is None:
        return None

    aw_bl = len(df[(df['delta_alice_gain'] > 1e-9) & (df['delta_bob_gain'] < -1e-9)])
    bw_al = len(df[(df['delta_bob_gain'] > 1e-9) & (df['delta_alice_gain'] < -1e-9)])
    total = len(df)
    zero_sum = aw_bl + bw_al
    pct = zero_sum / total * 100
    ci_low, ci_high = wilson_ci(zero_sum, total)

    return {
        'count': zero_sum,
        'total': total,
        'percentage': pct,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100
    }

def panel_b(df):
    """
    Panel B: Poisoned Apple Rate
    Percentage of opposite-payoff transitions that also change the selected
    market and have zero adoption of the added technology.
    """
    if df is None:
        return None

    # Get all zero-sum cases
    aw_bl = (df['delta_alice_gain'] > 1e-9) & (df['delta_bob_gain'] < -1e-9)
    bw_al = (df['delta_bob_gain'] > 1e-9) & (df['delta_alice_gain'] < -1e-9)
    zero_sum_mask = aw_bl | bw_al
    zero_sum_total = zero_sum_mask.sum()

    # A Poisoned Apple also requires the regulator's selected market to
    # change. Without that policy change, the payoff shift is not the
    # mechanism studied in the paper.
    no_usage = df['added_strategy_prob'] < 1e-9
    zero_sum_no_usage = (zero_sum_mask & no_usage & df['market_changed']).sum()

    if zero_sum_total == 0:
        return None

    pct = zero_sum_no_usage / zero_sum_total * 100
    ci_low, ci_high = wilson_ci(zero_sum_no_usage, zero_sum_total)

    return {
        'count': zero_sum_no_usage,
        'total': zero_sum_total,
        'percentage': pct,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100
    }

def panel_c(df):
    """
    Panel C: Frequency of Improvement in Regulatory Metric
    How often the regulator's metric improves vs. harms.
    """
    if df is None:
        return None

    total = len(df)
    improve = len(df[df['delta_designer_value'] > 1e-9])
    harm = len(df[df['delta_designer_value'] < -1e-9])

    improve_pct = improve / total * 100
    harm_pct = harm / total * 100

    improve_ci = wilson_ci(improve, total)
    harm_ci = wilson_ci(harm, total)

    return {
        'improve': {
            'count': improve,
            'total': total,
            'percentage': improve_pct,
            'ci_low': improve_ci[0] * 100,
            'ci_high': improve_ci[1] * 100
        },
        'harm': {
            'count': harm,
            'total': total,
            'percentage': harm_pct,
            'ci_low': harm_ci[0] * 100,
            'ci_high': harm_ci[1] * 100
        }
    }

def panel_d(df):
    """
    Panel D: New Model Adoption Rate (When Metric Improves)

    This is the PROPORTION of improvement cases where the new model is actually used.
    Formula: (improve_with_usage) / (total_improve) = (total_improve - improve_no_usage) / total_improve

    This is NOT the average adoption probability, but the fraction of cases with non-zero adoption.
    """
    if df is None:
        return None

    # Total cases where metric improved
    df_improve = df[df['delta_designer_value'] > 1e-9]
    total_improve = len(df_improve)

    if total_improve == 0:
        return None

    # Cases where metric improved AND new model was used (adoption > 0)
    improve_with_usage = len(df_improve[df_improve['added_strategy_prob'] > 1e-9])

    pct = improve_with_usage / total_improve * 100
    ci_low, ci_high = wilson_ci(improve_with_usage, total_improve)

    return {
        'n': total_improve,
        'with_usage': improve_with_usage,
        'percentage': pct,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100
    }

def panel_e(df):
    """
    Panel E: New Model Adoption Rate (When Metric Worsens)

    This is the PROPORTION of harm cases where the new model is actually used.
    Formula: (harm_with_usage) / (total_harm) = (total_harm - harm_no_usage) / total_harm

    This is NOT the average adoption probability, but the fraction of cases with non-zero adoption.
    """
    if df is None:
        return None

    # Total cases where metric worsened (harmed)
    df_harm = df[df['delta_designer_value'] < -1e-9]
    total_harm = len(df_harm)

    if total_harm == 0:
        return None

    # Cases where metric harmed AND new model was used (adoption > 0)
    harm_with_usage = len(df_harm[df_harm['added_strategy_prob'] > 1e-9])

    pct = harm_with_usage / total_harm * 100
    ci_low, ci_high = wilson_ci(harm_with_usage, total_harm)

    return {
        'n': total_harm,
        'with_usage': harm_with_usage,
        'percentage': pct,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100
    }

def panel_f(df):
    """
    Panel F: Frequency of Required Market Redesign (regulatory inertia risk).

    Computes the Market Changed percentage — the share of releases after
    which the regulator's optimal market is no longer the one chosen before
    the release, i.e. the metric degrades unless the market is redesigned.

    Formula: market_changed / total
    """
    if df is None:
        return None

    total = len(df)
    market_changed = df['market_changed'].sum()

    pct = market_changed / total * 100
    ci_low, ci_high = wilson_ci(market_changed, total)

    return {
        'count': market_changed,
        'total': total,
        'percentage': pct,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100
    }

def format_ci(value, ci_low, ci_high):
    """Format a value with confidence interval."""
    return f"{value:.1f}% [{ci_low:.1f}%, {ci_high:.1f}%]"

def print_results(mixed_mode='average', families=None):
    """Print all Figure 2 results with confidence intervals."""

    print("=" * 80)
    print(f"FIGURE 2 - ALL NUMBERS WITH 95% CONFIDENCE INTERVALS (MODE: {mixed_mode.upper()})")
    print("=" * 80)

    families = families or FAMILIES
    metrics = ['fairness', 'efficiency']

    for metric in metrics:
        print(f"\n{'=' * 40}")
        print(f"REGULATOR METRIC: {metric.upper()}")
        print(f"{'=' * 40}")

        data = load_transition_frames(metric, mixed_mode, families)

        # Panel A
        print("\n--- Panel A: Frequency of Opposite Payoff Changes ---")
        for family in families:
            result = panel_a(data[family])
            if result:
                print(f"  {family.capitalize():12} {format_ci(result['percentage'], result['ci_low'], result['ci_high'])} (n={result['count']}/{result['total']})")

        # Panel B
        print("\n--- Panel B: Poisoned Apple Rate ---")
        print("  (Market change plus zero adoption, as percentage of Panel A cases)")
        for family in families:
            result = panel_b(data[family])
            if result:
                print(f"  {family.capitalize():12} {format_ci(result['percentage'], result['ci_low'], result['ci_high'])} (n={result['count']}/{result['total']})")

        # Panel C
        print("\n--- Panel C: Frequency of Improvement in Regulatory Metric ---")
        for family in families:
            result = panel_c(data[family])
            if result:
                improve = result['improve']
                harm = result['harm']
                print(f"  {family.capitalize():12} Improve: {format_ci(improve['percentage'], improve['ci_low'], improve['ci_high'])}")
                print(f"  {' ':12} Harm:    {format_ci(harm['percentage'], harm['ci_low'], harm['ci_high'])}")

        # Panel D
        print("\n--- Panel D: New Model Adoption Rate (When Metric Improves) ---")
        print("  (Proportion of improvement cases where new model is used)")
        for family in families:
            result = panel_d(data[family])
            if result:
                print(f"  {family.capitalize():12} {format_ci(result['percentage'], result['ci_low'], result['ci_high'])} ({result['with_usage']}/{result['n']})")

        # Panel E
        print("\n--- Panel E: New Model Adoption Rate (When Metric Worsens) ---")
        print("  (Proportion of harm cases where new model is used)")
        for family in families:
            result = panel_e(data[family])
            if result:
                print(f"  {family.capitalize():12} {format_ci(result['percentage'], result['ci_low'], result['ci_high'])} ({result['with_usage']}/{result['n']})")

        # Panel F
        print("\n--- Panel F: Frequency of Metric Harm Without Market Update ---")
        print("  (Market Changed percentage - regulatory inertia)")
        for family in families:
            result = panel_f(data[family])
            if result:
                print(f"  {family.capitalize():12} {format_ci(result['percentage'], result['ci_low'], result['ci_high'])} ({result['count']}/{result['total']})")

    print("\n" + "=" * 80)
    print("FORMULA SUMMARY:")
    print("=" * 80)
    print("Panel A: Zero-Sum-Like / Total")
    print("Panel B: Market-Changed-and-Zero-Adoption-Opposite / Opposite-All")
    print("Panel C: Improve / Total, Harm / Total")
    print("Panel D: (Improve_all - Improve_no_usage) / Improve_all")
    print("Panel E: (Harm_all - Harm_no_usage) / Harm_all")
    print("Panel F: Market_Changed / Total")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python -m figures.figure2_numbers',
        description='Calculate Figure 2 numbers with confidence intervals.')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average',
                        help='Aggregation over mixed equilibria: "average" (used in the paper) or "any" (default: average)')
    parser.add_argument('--families', type=str, default=','.join(FAMILIES),
                        help='Comma-separated list of families')
    args = parser.parse_args()

    print_results(args.mixed_mode, [f.strip() for f in args.families.split(',')])
