"""
Generate paper Figure 3: Poisoned Apple rate as a function of the
regulator's ban (restriction) budget N, for a fairness-maximizing
regulator (left panel) and an efficiency-maximizing regulator (right).

The Poisoned Apple rate is the 'zero_adoption_opposite' panel of the
banning_summary CSVs produced by pipeline/banning_analysis.py: the share
of opposite payoff shifts that also change the selected market and in which
the added technology has zero equilibrium probability. Shaded bands are 95%
Wilson score CIs (already present in the CSVs).
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from core.utils import results_suffix

FAMILIES = ['bargaining', 'negotiation', 'persuasion']
FAMILY_LABELS = {'bargaining': 'Bargaining', 'negotiation': 'Negotiation', 'persuasion': 'Persuasion'}


def load_banning_summaries(metrics, mode, mixed_mode, families):
    """Load banning summary CSVs for the given families and metrics.

    Raises SystemExit listing missing files — a silently missing input
    would otherwise produce an empty or partial figure.
    """
    suffix = results_suffix(mode, mixed_mode)
    data = {}
    missing = []
    for metric in metrics:
        for family in families:
            path = f'output/{metric}/calculations/banning_summary_{family}_{suffix}.csv'
            if os.path.exists(path):
                data[(metric, family)] = pd.read_csv(path)
            else:
                missing.append(path)
    if missing:
        raise SystemExit("Missing banning summaries (run `make run-banning` for both metrics first):\n  "
                         + "\n  ".join(missing))
    return data


FAMILY_COLORS = {
    'bargaining': '#1f77b4',
    'negotiation': '#ff7f0e',
    'persuasion': '#2ca02c',
}
FAMILY_MARKERS = {
    'bargaining': 'o',
    'negotiation': 's',
    'persuasion': '^',
}
METRIC_TITLES = {
    'fairness': 'Fairness-Maximizing Regulator',
    'efficiency': 'Efficiency-Maximizing Regulator',
}


def plot_pa_rate_panel(ax, data, metric, max_bans, families):
    for family in families:
        df = data.get((metric, family))
        if df is None:
            continue

        panel_df = df[(df['panel'] == 'zero_adoption_opposite')
                      & (df['ban_budget'] <= max_bans)].sort_values('ban_budget')
        if panel_df.empty:
            continue

        color = FAMILY_COLORS.get(family, f'C{list(families).index(family)}')
        ax.plot(panel_df['ban_budget'], panel_df['percentage'],
                color=color, marker=FAMILY_MARKERS.get(family, 'o'),
                markersize=8, linewidth=2, label=FAMILY_LABELS.get(family, family.capitalize()))
        ax.fill_between(panel_df['ban_budget'],
                        panel_df['ci_low'], panel_df['ci_high'],
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel('Ban Budget ($N$)', fontsize=13)
    ax.set_title(METRIC_TITLES[metric], fontsize=14)
    ax.set_xticks(range(0, max_bans + 1, 2))
    ax.tick_params(labelsize=12)


def create_figure(metrics, mode, mixed_mode, max_bans, output_prefix, families):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    data = load_banning_summaries(metrics, mode, mixed_mode, families)

    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        plot_pa_rate_panel(ax, data, metric, max_bans, families)

    axes[0].set_ylabel('Poisoned Apple Rate (%)', fontsize=13)
    axes[0].set_ylim(bottom=-3)
    axes[-1].legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = f'{output_prefix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        prog='python -m figures.figure3_pa_rate',
        description='Generate paper Figure 3 (Poisoned Apple rate vs ban budget)')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average')
    parser.add_argument('--metrics', type=str, default='fairness,efficiency',
                        help='Comma-separated regulator metrics, one panel each')
    parser.add_argument('--max_bans', type=int, default=8,
                        help='Largest ban budget shown on the x-axis')
    parser.add_argument('--families', type=str, default=','.join(FAMILIES),
                        help='Comma-separated list of families')
    parser.add_argument('--outdir', default='output/figures', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(',')]
    families = [f.strip() for f in args.families.split(',')]
    create_figure(metrics, args.mode, args.mixed_mode, args.max_bans,
                  os.path.join(args.outdir, 'banning_pa_rate'), families)


if __name__ == '__main__':
    main()
