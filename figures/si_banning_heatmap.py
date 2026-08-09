"""
Generate the SI heatmap: Poisoned Apple rate as a function of the number
of existing models (initial_size, horizontal) and the regulator's ban
budget (N, vertical), per game family and regulator metric. Cells with
N >= initial_size are blank (the regulator cannot ban every model).

Reads the banning_stratified CSVs produced by pipeline/banning_analysis.py.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.utils import results_suffix

FAMILIES = ['bargaining', 'negotiation', 'persuasion']
FAMILY_LABELS = {'bargaining': 'Bargaining', 'negotiation': 'Negotiation', 'persuasion': 'Persuasion'}


def load_stratified_data(metric, mode, mixed_mode, families):
    """Load stratified CSVs; raise a clear error listing any missing file."""
    suffix = results_suffix(mode, mixed_mode)
    data = {}
    missing = []
    for family in families:
        path = f'output/{metric}/calculations/banning_stratified_{family}_{suffix}.csv'
        if os.path.exists(path):
            data[family] = pd.read_csv(path)
        else:
            missing.append(path)
    if missing:
        raise SystemExit("Missing banning stratified CSVs (run `make run-banning` for both metrics first):\n  "
                         + "\n  ".join(missing))
    return data


def make_heatmap_matrix(df, panel_name):
    """Pivot stratified data into a 2D matrix (ban_budget x initial_size)."""
    panel_df = df[df['panel'] == panel_name]
    if panel_df.empty:
        return None

    pivot = panel_df.pivot_table(
        index='ban_budget', columns='initial_size',
        values='percentage', aggfunc='first'
    )
    return pivot


def create_heatmap_figure(data, panel_name, title, metrics, output_prefix, families):
    """Create a grid of heatmaps: rows=metrics, cols=families."""
    n_metrics = len(metrics)
    n_families = len(families)

    fig, axes = plt.subplots(n_metrics, n_families, figsize=(5 * n_families, 4.5 * n_metrics),
                             squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Find global vmax across all heatmaps for consistent colorscale
    all_vals = []
    for metric in metrics:
        for family in families:
            df = data.get((metric, family))
            if df is None:
                continue
            panel_df = df[df['panel'] == panel_name]
            if not panel_df.empty:
                all_vals.extend(panel_df['percentage'].values)

    vmax = max(all_vals) if all_vals else 100
    vmin = 0

    im = None
    for row, metric in enumerate(metrics):
        for col, family in enumerate(families):
            ax = axes[row, col]
            df = data.get((metric, family))

            if df is None:
                ax.set_visible(False)
                continue

            pivot = make_heatmap_matrix(df, panel_name)
            if pivot is None:
                ax.set_visible(False)
                continue

            # Mask invalid cells where ban_budget >= initial_size
            # (can't ban N models when only initial_size exist before expansion)
            masked = pivot.values.copy().astype(float)
            for i, ban_n in enumerate(pivot.index):
                for j, init_size in enumerate(pivot.columns):
                    if ban_n >= init_size:
                        masked[i, j] = np.nan

            # Plot heatmap
            cmap_masked = plt.cm.YlOrRd.copy()
            cmap_masked.set_bad(color='white')
            im = ax.imshow(masked, aspect='auto', origin='lower',
                           cmap=cmap_masked, vmin=vmin, vmax=vmax,
                           interpolation='nearest')

            # Axis labels
            x_labels = [str(int(c)) for c in pivot.columns]
            y_labels = [str(int(i)) for i in pivot.index]
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=7)
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=7)

            # Annotate cells with values
            for i in range(masked.shape[0]):
                for j in range(masked.shape[1]):
                    val = masked[i, j]
                    if np.isnan(val):
                        continue
                    text_color = 'white' if val > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                            fontsize=6, color=text_color)

            metric_label = 'Fairness' if metric == 'fairness' else 'Efficiency'
            ax.set_title(f'{FAMILY_LABELS.get(family, family.capitalize())} ({metric_label})', fontsize=10)
            ax.set_xlabel('Number of Existing Models', fontsize=9)
            if col == 0:
                ax.set_ylabel('Ban Budget (N)', fontsize=9)

    # Colorbar
    if im is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Percentage (%)', fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.91, top=0.95, bottom=0.08, wspace=0.3, hspace=0.35)

    for ext in ['png', 'pdf']:
        path = f'{output_prefix}.{ext}'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        prog='python -m figures.si_banning_heatmap',
        description='Generate the SI Poisoned Apple heatmap')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average')
    parser.add_argument('--metrics', type=str, default='fairness,efficiency')
    parser.add_argument('--families', type=str, default=','.join(FAMILIES),
                        help='Comma-separated list of families')
    parser.add_argument('--outdir', default='output/figures', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(',')]
    families = [f.strip() for f in args.families.split(',')]

    # Load all stratified data
    all_data = {}
    for metric in metrics:
        family_data = load_stratified_data(metric, args.mode, args.mixed_mode, families)
        for family, df in family_data.items():
            all_data[(metric, family)] = df

    create_heatmap_figure(
        all_data, 'zero_adoption_opposite',
        'Poisoned Apple Effect: Existing Models vs Ban Budget',
        metrics, os.path.join(args.outdir, 'banning_heatmap_poisoned_apple'), families)


if __name__ == '__main__':
    main()
