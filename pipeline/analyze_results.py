"""Statistical report on the release transitions.

Reads comprehensive_transitions_{family}_{suffix}.csv, prints and writes
positive/negative-impact shares, market-change rates, and the most extreme
improvement/damage cases (with a before/after infographic for each).

Output: output/{metric}/summaries/report_{suffix}_{transition_metric}.txt
"""

import argparse
import os

from core.utils import load_transitions, results_suffix
from pipeline.infographics import visualize_transition


def analyze_transitions(df, metric_col, prob_threshold=0.0, no_usage=False):
    # Filter by probability if needed
    if no_usage:
        df = df[df['added_strategy_prob'] == 0]
    elif prob_threshold > 0:
        df = df[df['added_strategy_prob'] >= prob_threshold]

    if df.empty:
        return None, None, None

    # Statistics
    total = len(df)
    positive = len(df[df[metric_col] > 1e-9])
    negative = len(df[df[metric_col] < -1e-9])
    neutral = total - positive - negative

    market_changes = len(df[df['market_changed'] == True])
    market_change_pct = (market_changes / total) * 100 if total > 0 else 0

    stats = {
        'total_cases': total,
        'positive_impact': positive,
        'negative_impact': negative,
        'neutral_impact': neutral,
        'positive_pct': (positive / total) * 100 if total > 0 else 0,
        'negative_pct': (negative / total) * 100 if total > 0 else 0,
        'market_change_pct': market_change_pct
    }

    # Find extremes
    max_imp = df.loc[df[metric_col].idxmax()]
    max_dam = df.loc[df[metric_col].idxmin()]

    return stats, max_imp, max_dam


def generate_infographic(row, family, mode, mixed_mode, case_type, metric_col, regulator_metric):
    base_strategies = row['base_subset'].split(';')
    added_strategy = row['added_strategy']

    subdir = os.path.join("output", regulator_metric, "visualizations", family, metric_col)

    visualize_transition(
        family=family,
        base_strategies=base_strategies,
        added_strategy=added_strategy,
        case_name=case_type,
        output_dir=subdir,
        equilibrium_type=mode,
        designer_metric=regulator_metric,
        mixed_mode=mixed_mode
    )


def main():
    parser = argparse.ArgumentParser(
        prog='python -m pipeline.analyze_results',
        description='Statistical report on release transitions.')
    parser.add_argument('--families', type=str, default='bargaining,negotiation,persuasion', help='Comma-separated list of families')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed', help='Equilibrium type')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average', help='Aggregation over mixed equilibria (paper: average)')
    parser.add_argument('--metric', choices=['fairness', 'efficiency'], default='fairness', help='Regulator metric')
    parser.add_argument('--transition_metric', type=str, default='delta_designer_value', help='Transition column to analyze (delta_designer_value, delta_alice_gain, delta_bob_gain)')
    parser.add_argument('--prob_threshold', type=float, default=0.0, help='Minimum probability for added strategy to be considered')
    parser.add_argument('--size_min', type=int, default=None, help='Only include base subsets of at least this size')
    parser.add_argument('--size_max', type=int, default=None, help='Only include base subsets of at most this size')
    parser.add_argument('--no_usage', action='store_true', help='Analyze only cases where added strategy probability is 0')

    args = parser.parse_args()

    families = args.families.split(',')

    run_suffix = results_suffix(args.mode, args.mixed_mode)
    usage_suffix = "_no_usage" if args.no_usage else ""
    output_dir = f"output/{args.metric}/summaries"
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/report_{run_suffix}_{args.transition_metric}{usage_suffix}.txt"

    with open(report_path, 'w') as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log(f"Analysis Report (Mode: {run_suffix}, Transition Metric: {args.transition_metric}, Regulator Metric: {args.metric})")
        log("=" * 60)

        for family in families:
            family = family.strip()
            df = load_transitions(family, args.mode, args.mixed_mode, args.metric)

            if args.size_min is not None:
                df = df[df['initial_size'] >= args.size_min]
            if args.size_max is not None:
                df = df[df['initial_size'] <= args.size_max]

            log(f"\nFamily: {family}")
            log("-" * 30)

            stats, max_imp, max_dam = analyze_transitions(df, args.transition_metric, args.prob_threshold, args.no_usage)

            if stats:
                log(f"Total Cases: {stats['total_cases']}")
                log(f"Positive Impact: {stats['positive_impact']} ({stats['positive_pct']:.2f}%)")
                log(f"Negative Impact: {stats['negative_impact']} ({stats['negative_pct']:.2f}%)")
                log(f"Market Changed: {stats['market_change_pct']:.2f}%")

                log("\nMost Significant Improvement:")
                log(f"  Delta: {max_imp[args.transition_metric]:.4f}")
                log(f"  Added: {max_imp['added_strategy']}")
                log(f"  Size: {max_imp['initial_size']}")
                log(f"  Market: {max_imp['initial_market']} -> {max_imp['final_market']}")

                log("\nMost Significant Damage:")
                log(f"  Delta: {max_dam[args.transition_metric]:.4f}")
                log(f"  Added: {max_dam['added_strategy']}")
                log(f"  Size: {max_dam['initial_size']}")
                log(f"  Market: {max_dam['initial_market']} -> {max_dam['final_market']}")

                # Render the two extreme cases as infographics
                try:
                    generate_infographic(max_imp, family, args.mode, args.mixed_mode, f"max_improvement{usage_suffix}", args.transition_metric, args.metric)
                    generate_infographic(max_dam, family, args.mode, args.mixed_mode, f"max_damage{usage_suffix}", args.transition_metric, args.metric)
                except Exception as e:
                    log(f"Error generating infographics: {e}")
            else:
                log("No data matching criteria.")


if __name__ == "__main__":
    main()
