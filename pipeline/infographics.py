"""Before/after infographics for a single release transition.

For a base strategy set and an added strategy, draws the combined payoff
matrices (Alice lower-left triangle, Bob upper-right), the equilibria, and
the per-market designer values before the release, after it, and — when the
regulator switches market — in the intermediate state (new strategies, old
market). These infographics are the source material of paper Figure 1.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import os
import string
from matplotlib.patches import Polygon

from core.data_manager import load_data, get_all_markets, get_game
from core.simulation import evaluate_market, get_subset
from core.utils import get_bob_strategy

# Set style
sns.set_theme(style="whitegrid")


def solve_game_for_subset(data, strategies, designer_metric="fairness", equilibrium_type="mixed", mixed_mode="average"):
    """Solve every market for one strategy subset.

    Returns (best_result, all_markets_results) where best_result is the
    market maximizing the designer metric.
    """
    alice_subset = strategies
    bob_subset = [get_bob_strategy(a) for a in alice_subset]
    subset_of_strategies = {'alice': alice_subset, 'bob': bob_subset}

    all_markets_results = []
    best_result = None
    max_designer_value = -np.inf

    for market in get_all_markets(data):
        subset_metrics = get_subset(get_game(data, market), subset_of_strategies)
        evaluation = evaluate_market(subset_metrics, designer_metric=designer_metric,
                                     equilibrium_type=equilibrium_type, mixed_mode=mixed_mode)

        res = {
            'market': market,
            'designer_value': evaluation['designer_value'] if evaluation else -np.inf,
            'equilibria': evaluation['equilibria'] if evaluation else [],
            'best_eq': evaluation['profile'] if evaluation else None,
            'alice_matrix': subset_metrics['alice_self_gain'],
            'bob_matrix': subset_metrics['bob_self_gain'],
            'strategies': strategies
        }
        all_markets_results.append(res)

        if res['designer_value'] > max_designer_value:
            max_designer_value = res['designer_value']
            best_result = res

    return best_result, all_markets_results


def draw_market_list(ax, markets_data, active_market, original_market, title):
    ax.axis('off')
    y = 1.0
    line_height = 0.04  # Adjust based on font size and number of items

    ax.text(0, y, title, fontsize=12, family='monospace', fontweight='bold')
    y -= line_height

    sorted_markets = sorted(markets_data, key=lambda x: x['designer_value'], reverse=True)
    for r in sorted_markets:
        m_name = r['market']
        val_str = f"{r['designer_value']:.4f}" if r['designer_value'] != -np.inf else "No NE"
        line_text = f"{m_name}: {val_str}"

        color = 'black'
        weight = 'normal'

        is_original = (m_name == original_market)
        is_active = (m_name == active_market)

        if is_original:
            weight = 'bold'

        if is_active and m_name != original_market:
            color = 'red'
            weight = 'bold'

        ax.text(0, y, line_text, color=color, fontweight=weight, fontsize=10, family='monospace', va='top')
        y -= line_height


def visualize_transition(family, base_strategies, added_strategy, case_name, output_dir, equilibrium_type="mixed", designer_metric="fairness", mixed_mode="average"):
    print(f"Visualizing {family} - {case_name} ({equilibrium_type})")
    data = load_data(f"data/{family}.csv")

    # 1. Solve Before
    res_before, all_before = solve_game_for_subset(data, base_strategies, equilibrium_type=equilibrium_type, designer_metric=designer_metric, mixed_mode=mixed_mode)

    # 2. Solve After
    full_strategies = base_strategies + [added_strategy]
    res_after, all_after = solve_game_for_subset(data, full_strategies, equilibrium_type=equilibrium_type, designer_metric=designer_metric, mixed_mode=mixed_mode)

    if not res_before or not res_after:
        print(f"Skipping {case_name} due to no equilibrium found.")
        return False

    # 3. Solve Intermediate (After strategies, Before market)
    # Find the result for the 'before' market within the 'after' calculations
    res_intermediate = next((r for r in all_after if r['market'] == res_before['market']), None)

    # 4. Determine Layout
    market_changed = res_before['market'] != res_after['market']

    unique_strategies = sorted(list(set(full_strategies) - {added_strategy})) + [added_strategy]
    labels = list(string.ascii_uppercase)[:len(unique_strategies)]
    strat_to_label = {s: l for s, l in zip(unique_strategies, labels)}
    if added_strategy in strat_to_label:
        strat_to_label[added_strategy] += " (new)"

    # Dynamic Layout: 4 rows if market changed (to show intermediate), else 3
    rows = 4 if market_changed else 3
    height_ratios = [0.2] + [1] * (rows - 1)
    fig_height = 5 * rows

    # Width ratios: Combined Matrix (2), Market List (1)
    fig = plt.figure(figsize=(15, fig_height))
    gs = fig.add_gridspec(rows, 2, height_ratios=height_ratios, width_ratios=[2, 1])

    # --- Row 0: Legend ---
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')
    legend_text = "Strategy Legend:\n"
    for s, l in strat_to_label.items():
        marker = " (NEW)" if s == added_strategy else ""
        legend_text += f"{l}: {s}{marker}\n"
    ax_legend.text(0.0, 1.0, legend_text, va='top', fontsize=12, family='monospace')
    ax_legend.set_title(f"{family} - {case_name} ({equilibrium_type})", fontsize=18, fontweight='bold')

    # Helper to plot combined matrix
    def plot_combined_matrix(ax, alice_matrix, bob_matrix, title, equilibria, strategies, selected_eq=None, eq_type="pure"):
        # Rename index/columns to labels
        lbl_alice = alice_matrix.rename(index=strat_to_label, columns={get_bob_strategy(s): strat_to_label[s] for s in unique_strategies if get_bob_strategy(s) in alice_matrix.columns})
        lbl_bob = bob_matrix.rename(index=strat_to_label, columns={get_bob_strategy(s): strat_to_label[s] for s in unique_strategies if get_bob_strategy(s) in bob_matrix.columns})

        lbl_alice = lbl_alice.sort_index(axis=0).sort_index(axis=1)
        lbl_bob = lbl_bob.sort_index(axis=0).sort_index(axis=1)

        n_rows, n_cols = lbl_alice.shape

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_xticklabels(lbl_alice.columns, fontsize=14)
        ax.set_yticklabels(lbl_alice.index, fontsize=14)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("Bob's Model", fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel("Alice's Model", fontsize=16, fontweight='bold', labelpad=10)

        # Colormap
        original_cmap = plt.get_cmap("Purples")
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'truncated_Purples', original_cmap(np.linspace(0, 0.7, 256))
        )
        # Normalize across both matrices
        vmin = min(lbl_alice.min().min(), lbl_bob.min().min())
        vmax = max(lbl_alice.max().max(), lbl_bob.max().max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Draw cells
        for r in range(n_rows):
            for c in range(n_cols):
                val_a = lbl_alice.iloc[r, c]
                val_b = lbl_bob.iloc[r, c]

                # Alice (Bottom-Left): (c, r), (c, r+1), (c+1, r+1)
                poly_a = Polygon([(c, r), (c, r + 1), (c + 1, r + 1)], facecolor=cmap(norm(val_a)), edgecolor='white')
                ax.add_patch(poly_a)

                # Bob (Top-Right): (c, r), (c+1, r), (c+1, r+1)
                poly_b = Polygon([(c, r), (c + 1, r), (c + 1, r + 1)], facecolor=cmap(norm(val_b)), edgecolor='white')
                ax.add_patch(poly_b)

                # Alice value (Bottom-Left)
                ax.text(c + 0.25, r + 0.75, f"{val_a:.2f}", ha='center', va='center', fontsize=14, color='black', fontweight='bold')

                # Bob value (Top-Right)
                ax.text(c + 0.75, r + 0.25, f"{val_b:.2f}", ha='center', va='center', fontsize=14, color='black', fontweight='bold')

        # Equilibria
        eq_labels = []
        if equilibria:
            if eq_type == "pure":
                for eq in equilibria:
                    r_lbl = strat_to_label[eq[0]]
                    c_lbl = strat_to_label[next(s for s in unique_strategies if get_bob_strategy(s) == eq[1])]
                    eq_labels.append(f"({r_lbl}, {c_lbl})")

                    if r_lbl in lbl_alice.index and c_lbl in lbl_alice.columns:
                        r_idx = lbl_alice.index.get_loc(r_lbl)
                        c_idx = lbl_alice.columns.get_loc(c_lbl)
                        rect = plt.Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor='green', lw=4)
                        ax.add_patch(rect)
            elif eq_type == "mixed" and selected_eq is not None:
                sigma_alice, sigma_bob = selected_eq

                bob_probs_map = {}
                for i, s in enumerate(strategies):
                    lbl = strat_to_label[s]
                    bob_probs_map[lbl] = sigma_bob[i]

                alice_probs_map = {}
                for i, s in enumerate(strategies):
                    lbl = strat_to_label[s]
                    alice_probs_map[lbl] = sigma_alice[i]

                # Update tick labels with the play probabilities
                new_yticklabels = []
                for label in lbl_alice.index:
                    prob = alice_probs_map.get(label, 0.0)
                    new_yticklabels.append(f"{label}\n({prob:.2f})")
                ax.set_yticklabels(new_yticklabels, rotation=0, fontsize=14)

                new_xticklabels = []
                for label in lbl_alice.columns:
                    prob = bob_probs_map.get(label, 0.0)
                    new_xticklabels.append(f"{label}\n({prob:.2f})")
                ax.set_xticklabels(new_xticklabels, rotation=0, fontsize=14)

                eq_labels.append("Mixed Profile")

                # Highlight the cell if the profile is (nearly) pure
                is_pure_alice = np.max(sigma_alice) >= 0.99
                is_pure_bob = np.max(sigma_bob) >= 0.99

                if is_pure_alice and is_pure_bob:
                    idx_alice = np.argmax(sigma_alice)
                    idx_bob = np.argmax(sigma_bob)

                    strat_alice = strategies[idx_alice]
                    strat_bob = strategies[idx_bob]

                    lbl_a = strat_to_label[strat_alice]
                    lbl_b = strat_to_label[strat_bob]

                    if lbl_a in lbl_alice.index and lbl_b in lbl_alice.columns:
                        r_idx = lbl_alice.index.get_loc(lbl_a)
                        c_idx = lbl_alice.columns.get_loc(lbl_b)
                        rect = plt.Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor='green', lw=4)
                        ax.add_patch(rect)

        full_title = f"{title}\nEq: {', '.join(eq_labels)}" if eq_labels else title
        ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)

    # Helper for parsing market names
    def parse_market(m_str):
        return " ".join(m_str.split('_'))

    # --- Row 1: Before ---
    ax_b_matrix = fig.add_subplot(gs[1, 0])

    market_disp_before = parse_market(res_before['market'])
    title_prefix = f"BEFORE\nMarket: {market_disp_before}\nDesigner Val: {res_before['designer_value']:.4f}"
    plot_combined_matrix(ax_b_matrix, res_before['alice_matrix'], res_before['bob_matrix'], title_prefix, res_before['equilibria'], res_before['strategies'], selected_eq=res_before.get('best_eq'), eq_type=equilibrium_type)

    # Market List (Before)
    ax_list_before = fig.add_subplot(gs[1, 1])
    draw_market_list(ax_list_before, all_before, res_before['market'], res_before['market'], "Markets (Before):")

    current_row = 2

    # --- Row 2 (Conditional): Intermediate ---
    if market_changed and res_intermediate:
        ax_i_matrix = fig.add_subplot(gs[current_row, 0])

        title_prefix = f"INTERMEDIATE (Added {strat_to_label[added_strategy]}, Old Market)\nMarket: {market_disp_before}\nDesigner Val: {res_intermediate['designer_value']:.4f}"

        plot_combined_matrix(ax_i_matrix, res_intermediate['alice_matrix'], res_intermediate['bob_matrix'], title_prefix, res_intermediate['equilibria'], res_intermediate['strategies'], selected_eq=res_intermediate.get('best_eq'), eq_type=equilibrium_type)

        # Market List (Intermediate)
        ax_list_inter = fig.add_subplot(gs[current_row, 1])
        draw_market_list(ax_list_inter, all_after, res_before['market'], res_before['market'], "Markets (Intermediate):\n(Same strategies as After,\nbut forced Old Market)")

        current_row += 1

    # --- Row 3 (or 2): After ---
    ax_a_matrix = fig.add_subplot(gs[current_row, 0])

    market_disp_after = parse_market(res_after['market'])
    market_color = "red" if market_changed else "black"

    title_prefix = f"AFTER (Added {strat_to_label[added_strategy]})\nMarket: {market_disp_after}\nDesigner Val: {res_after['designer_value']:.4f}"
    plot_combined_matrix(ax_a_matrix, res_after['alice_matrix'], res_after['bob_matrix'], title_prefix, res_after['equilibria'], res_after['strategies'], selected_eq=res_after.get('best_eq'), eq_type=equilibrium_type)

    if market_changed:
        ax_a_matrix.title.set_color(market_color)

    # Market List (After)
    ax_list_after = fig.add_subplot(gs[current_row, 1])
    draw_market_list(ax_list_after, all_after, res_after['market'], res_before['market'], "Markets (After):")

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{output_dir}/{family}_{case_name.replace(' ', '_')}_{equilibrium_type}.png"
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()
    print(f"Saved to {filename}")
    return True
