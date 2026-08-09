"""
Render Figure 2 as matplotlib figures.

--panels renders each panel as a separate image (panel_A..F) — these are
the artifacts the published figure was composed from. The default renders
all six panels in one composite image (statistics_with_ci) as a
convenient preview; --all renders both.

All numbers and CIs come from figures/figure2_numbers.py — the canonical
source of every Figure 2 value.
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from figures.figure2_numbers import (FAMILIES, load_transition_frames,
                                     panel_a, panel_b, panel_c,
                                     panel_d, panel_e, panel_f)

FAMILY_LABEL_MAP = {'bargaining': 'Bargaining', 'negotiation': 'Negotiation', 'persuasion': 'Persuasion'}


def _labels(families):
    return [FAMILY_LABEL_MAP.get(f, f.capitalize()) for f in families]

# Colors of the published figure
PANEL_BLUE = '#1f77b4'    # Fairness
PANEL_ORANGE = '#ff7f0e'  # Efficiency
PANEL_GREEN = '#2ca02c'   # Improve
PANEL_RED = '#ff7f0e'     # Harm

GROUPED_PANELS = [
    (panel_a, 'A', 'Frequency of Opposite Payoff\nChanges'),
    (panel_b, 'B', 'Poisoned Apple Rate'),
    (panel_d, 'D', 'New Model Adoption Rate\n(When Metric Improves)'),
    (panel_e, 'E', 'New Model Adoption Rate\n(When Metric Worsens)'),
    (panel_f, 'F', 'Frequency of Change in the\nOptimal Market'),
]

# Style knobs that differ between the standalone panels and the composite
PANEL_STYLE = dict(value_fontsize=12, tick_fontsize=11, title_fontsize=13,
                   capsize=3, error_kw={'linewidth': 1.5, 'capthick': 1.5},
                   legend=True, yticks=[50, 100], bare_spines=True)
COMPOSITE_STYLE = dict(value_fontsize=8, tick_fontsize=8, title_fontsize=10,
                       capsize=2, error_kw={'linewidth': 1},
                       legend=False, yticks=None, bare_spines=False)


def _pcts_errs(vals, key=None):
    """Percentages and asymmetric CI error bars from a list of panel dicts.

    Error lengths are clamped at zero: CI bounds are clipped to [0, 100],
    so at the boundary a bound can sit a few ULPs past the point estimate.
    """
    if key is not None:
        vals = [v[key] if v else None for v in vals]
    pcts = [v['percentage'] if v else 0 for v in vals]
    errs = [[max(0.0, v['percentage'] - v['ci_low']) if v else 0 for v in vals],
            [max(0.0, v['ci_high'] - v['percentage']) if v else 0 for v in vals]]
    return pcts, errs


def _bar_pair(ax, x, width, left_vals, right_vals, left_label, right_label,
              left_color, right_color, style, key=None):
    """Two grouped bar series with error bars and value labels."""
    left_pcts, left_errs = _pcts_errs(left_vals, key)
    right_pcts, right_errs = _pcts_errs(right_vals, key)

    bars1 = ax.bar(x - width / 2, left_pcts, width, label=left_label, color=left_color,
                   yerr=left_errs, capsize=style['capsize'], error_kw=style['error_kw'])
    bars2 = ax.bar(x + width / 2, right_pcts, width, label=right_label, color=right_color,
                   yerr=right_errs, capsize=style['capsize'], error_kw=style['error_kw'])
    ax.bar_label(bars1, fmt='%.0f', padding=3, fontsize=style['value_fontsize'])
    ax.bar_label(bars2, fmt='%.0f', padding=3, fontsize=style['value_fontsize'])
    return bars1, bars2


def _finish_axis(ax, style, title=None, title_pad=None):
    ax.set_ylim(0, 100)
    if style['yticks']:
        ax.set_yticks(style['yticks'])
    ax.tick_params(axis='y', labelsize=style['tick_fontsize'])
    if title is not None:
        ax.set_title(title, fontsize=style['title_fontsize'], pad=title_pad)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if style['bare_spines']:
        ax.spines['left'].set_visible(False)


def draw_grouped_panel(ax, panel_func, fairness_data, efficiency_data, style, title=None, title_pad=None):
    """One Fairness-vs-Efficiency grouped panel (panels A, B, D, E, F)."""
    families = list(fairness_data)
    x = np.arange(len(families))
    fair_vals = [panel_func(fairness_data[f]) for f in families]
    eff_vals = [panel_func(efficiency_data[f]) for f in families]

    bars = _bar_pair(ax, x, 0.35, fair_vals, eff_vals, 'Fairness', 'Efficiency',
                     PANEL_BLUE, PANEL_ORANGE, style)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(families), fontsize=style['tick_fontsize'])
    _finish_axis(ax, style, title, title_pad)
    return bars


def draw_panel_c(ax, fairness_data, efficiency_data, style, title=None, title_pad=None, subtitle_y=115):
    """Panel C: Improve/Harm bars for both regulator metrics side by side."""
    families = list(fairness_data)
    n = len(families)
    fair_c = [panel_c(fairness_data[f]) for f in families]
    eff_c = [panel_c(efficiency_data[f]) for f in families]

    x_fair = np.arange(n)
    x_eff = np.arange(n + 1, 2 * n + 1)

    for x, data in [(x_fair, fair_c), (x_eff, eff_c)]:
        improve_pcts, improve_errs = _pcts_errs(data, 'improve')
        harm_pcts, harm_errs = _pcts_errs(data, 'harm')
        label_improve = 'Improve' if x is x_fair else None
        label_harm = 'Harm' if x is x_fair else None
        b1 = ax.bar(x - 0.175, improve_pcts, 0.35, label=label_improve, color=PANEL_GREEN,
                    yerr=improve_errs, capsize=style['capsize'], error_kw=style['error_kw'])
        b2 = ax.bar(x + 0.175, harm_pcts, 0.35, label=label_harm, color=PANEL_RED,
                    yerr=harm_errs, capsize=style['capsize'], error_kw=style['error_kw'])
        ax.bar_label(b1, fmt='%.0f', padding=3, fontsize=style['value_fontsize'])
        ax.bar_label(b2, fmt='%.0f', padding=3, fontsize=style['value_fontsize'])

    ax.text(x_fair.mean(), subtitle_y, "Regulator's Metric is Fairness", ha='center', fontsize=style['tick_fontsize'])
    ax.text(x_eff.mean(), subtitle_y, "Regulator's Metric is Efficiency", ha='center', fontsize=style['tick_fontsize'])

    ax.set_xticks(list(x_fair) + list(x_eff))
    ax.set_xticklabels(_labels(families) * 2, fontsize=style['tick_fontsize'])
    _finish_axis(ax, style, title, title_pad)


def create_composite_figure(fairness_data, efficiency_data, outdir):
    """All six panels in one image — a preview of the published composition."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1.5, 1.5, 1, 1, 1],
                          left=0.045, right=0.985, top=0.87, bottom=0.06,
                          wspace=0.35, hspace=0.5)

    def letter(ax, ch):
        ax.text(-0.08, 1.12, ch, transform=ax.transAxes, fontsize=13, fontweight='bold')

    ax_a = fig.add_subplot(gs[0, 0])
    draw_grouped_panel(ax_a, panel_a, fairness_data, efficiency_data,
                       COMPOSITE_STYLE, 'Frequency of Opposite\nPayoff Changes')
    letter(ax_a, 'A')

    ax_c = fig.add_subplot(gs[0, 1:4])
    draw_panel_c(ax_c, fairness_data, efficiency_data, COMPOSITE_STYLE,
                 'Frequency of Improvement in Regulatory Metric', subtitle_y=104)
    letter(ax_c, 'C')

    ax_e = fig.add_subplot(gs[0, 4:6])
    draw_grouped_panel(ax_e, panel_e, fairness_data, efficiency_data,
                       COMPOSITE_STYLE, 'New Model Adoption Rate\n(When Metric Worsens)')
    letter(ax_e, 'E')

    ax_b = fig.add_subplot(gs[1, 0])
    draw_grouped_panel(ax_b, panel_b, fairness_data, efficiency_data,
                       COMPOSITE_STYLE, 'Poisoned Apple Rate')
    letter(ax_b, 'B')

    ax_d = fig.add_subplot(gs[1, 1:4])
    draw_grouped_panel(ax_d, panel_d, fairness_data, efficiency_data,
                       COMPOSITE_STYLE, 'New Model Adoption Rate\n(When Metric Improves)')
    letter(ax_d, 'D')

    ax_f = fig.add_subplot(gs[1, 4:6])
    draw_grouped_panel(ax_f, panel_f, fairness_data, efficiency_data,
                       COMPOSITE_STYLE, 'Frequency of Change in the\nOptimal Market')
    letter(ax_f, 'F')

    # One shared legend for the whole figure
    handles = [Patch(facecolor=PANEL_BLUE), Patch(facecolor=PANEL_ORANGE),
               Patch(facecolor=PANEL_GREEN), Patch(facecolor=PANEL_RED)]
    labels = ['Fairness', 'Efficiency', 'Improve', 'Harm']
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.97))

    png_path = os.path.join(outdir, 'statistics_with_ci.png')
    pdf_path = os.path.join(outdir, 'statistics_with_ci.pdf')
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


def create_individual_panels(fairness_data, efficiency_data, outdir):
    """The six standalone panels the published figure was composed from."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    for panel_func, letter, title in GROUPED_PANELS:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        draw_grouped_panel(ax, panel_func, fairness_data, efficiency_data,
                           PANEL_STYLE, title, title_pad=30)
        ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=2, frameon=False, handlelength=1, handletextpad=0.3, columnspacing=1)
        filename = os.path.join(outdir, f'panel_{letter}.png')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=False)
        plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    draw_panel_c(ax, fairness_data, efficiency_data, PANEL_STYLE,
                 'Frequency of Improvement in Regulatory Metric', title_pad=55)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.32),
              ncol=2, frameon=False, handlelength=1, handletextpad=0.3, columnspacing=1)
    filename = os.path.join(outdir, 'panel_C.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=False)
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

    print("\nAll panels generated!")


def main():
    parser = argparse.ArgumentParser(
        prog='python -m figures.figure2_plot',
        description='Render Figure 2 (composite preview by default; --panels for the published panels).')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average',
                        help='Aggregation over mixed equilibria: "average" (used in the paper) or "any"')
    parser.add_argument('--families', type=str, default=','.join(FAMILIES),
                        help='Comma-separated list of families')
    parser.add_argument('--panels', action='store_true',
                        help='Render the six individual panels instead of the composite figure')
    parser.add_argument('--all', action='store_true',
                        help='Render both the composite figure and the individual panels')
    parser.add_argument('--outdir', default='output/figures', help='Output directory')
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(',')]
    os.makedirs(args.outdir, exist_ok=True)

    fairness_data = load_transition_frames('fairness', args.mixed_mode, families)
    efficiency_data = load_transition_frames('efficiency', args.mixed_mode, families)

    if args.all or not args.panels:
        create_composite_figure(fairness_data, efficiency_data, args.outdir)
    if args.all or args.panels:
        create_individual_panels(fairness_data, efficiency_data, args.outdir)


if __name__ == "__main__":
    main()
