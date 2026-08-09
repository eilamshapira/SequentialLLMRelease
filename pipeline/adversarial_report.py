"""Zero-sum categorization of release transitions.

Classifies every transition by the joint sign of the two players' payoff
changes (win-win, lose-lose, opposite shifts, neutral), overall and in the
Poisoned Apple subset used for Figure 2A/2B.

Output: output/{metric}/summaries/report_{mode}_adversarial_impact.txt
"""

import argparse
import os

from core.utils import load_transitions, results_suffix

def analyze_adversarial(df):
    if df is None or df.empty:
        return None

    total = len(df)

    # Alice Wins, Bob Loses (Alice > 0, Bob < 0)
    aw_bl = len(df[(df['delta_alice_gain'] > 1e-9) & (df['delta_bob_gain'] < -1e-9)])

    # Bob Wins, Alice Loses (Bob > 0, Alice < 0)
    bw_al = len(df[(df['delta_bob_gain'] > 1e-9) & (df['delta_alice_gain'] < -1e-9)])

    # Win-Win (Both > 0)
    win_win = len(df[(df['delta_alice_gain'] > 1e-9) & (df['delta_bob_gain'] > 1e-9)])

    # Lose-Lose (Both < 0)
    lose_lose = len(df[(df['delta_alice_gain'] < -1e-9) & (df['delta_bob_gain'] < -1e-9)])

    # Neutral / No Change (Both approx 0)
    neutral = len(df[
        (df['delta_alice_gain'].abs() <= 1e-9) &
        (df['delta_bob_gain'].abs() <= 1e-9)
    ])

    # One Changed, One Neutral (e.g. Alice gains, Bob neutral)
    aw_bn = len(df[(df['delta_alice_gain'] > 1e-9) & (df['delta_bob_gain'].abs() <= 1e-9)])
    bw_an = len(df[(df['delta_bob_gain'] > 1e-9) & (df['delta_alice_gain'].abs() <= 1e-9)])
    al_bn = len(df[(df['delta_alice_gain'] < -1e-9) & (df['delta_bob_gain'].abs() <= 1e-9)])
    bl_an = len(df[(df['delta_bob_gain'] < -1e-9) & (df['delta_alice_gain'].abs() <= 1e-9)])

    other_mixed = aw_bn + bw_an + al_bn + bl_an

    # Zero-Sum-Like (One gains, one loses)
    zero_sum_like = aw_bl + bw_al

    stats = {
        'total': total,
        'alice_win_bob_lose': aw_bl,
        'bob_win_alice_lose': bw_al,
        'win_win': win_win,
        'lose_lose': lose_lose,
        'neutral': neutral,
        'other_mixed': other_mixed,
        'zero_sum_like': zero_sum_like
    }

    return stats

def write_stats(f, stats, title):
    f.write(f"{title}\n")
    f.write("-" * 30 + "\n")

    if stats:
        total = stats['total']
        f.write(f"Total Cases: {total}\n")

        aw_bl = stats['alice_win_bob_lose']
        bw_al = stats['bob_win_alice_lose']
        ww = stats['win_win']
        ll = stats['lose_lose']
        neu = stats['neutral']
        oth = stats['other_mixed']
        zs = stats['zero_sum_like']

        f.write(f"Alice Wins / Bob Loses: {aw_bl} ({aw_bl/total*100:.2f}%)\n")
        f.write(f"Bob Wins / Alice Loses: {bw_al} ({bw_al/total*100:.2f}%)\n")
        f.write(f"Win-Win (Both Gain):    {ww} ({ww/total*100:.2f}%)\n")
        f.write(f"Lose-Lose (Both Lose):  {ll} ({ll/total*100:.2f}%)\n")
        f.write(f"Neutral (No Change):    {neu} ({neu/total*100:.2f}%)\n")
        f.write(f"One Changed/One Neutral:{oth} ({oth/total*100:.2f}%)\n")
        f.write(f"------------------------------\n")
        f.write(f"Total Zero-Sum-Like:    {zs} ({zs/total*100:.2f}%)\n")
    else:
        f.write("No data found.\n")
    f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        prog='python -m pipeline.adversarial_report',
        description='Generate adversarial impact report.')
    parser.add_argument('--families', type=str, default='bargaining,negotiation,persuasion', help='Comma-separated list of families')
    parser.add_argument('--mode', choices=['pure', 'mixed'], default='mixed', help='Equilibrium type')
    parser.add_argument('--mixed_mode', choices=['any', 'average'], default='average', help='Aggregation over mixed equilibria (paper: average)')
    parser.add_argument('--metric', choices=['fairness', 'efficiency'], default='fairness', help='Regulator metric')

    args = parser.parse_args()

    families = args.families.split(',')
    suffix = results_suffix(args.mode, args.mixed_mode)
    output_dir = f"output/{args.metric}/summaries"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/report_{suffix}_adversarial_impact.txt"

    with open(output_file, 'w') as f:
        f.write(f"Adversarial Impact Report (Mode: {suffix}, Metric: {args.metric})\n")
        f.write("============================================================\n\n")

        for family in families:
            df = load_transitions(family, args.mode, args.mixed_mode, args.metric)

            f.write(f"Family: {family}\n")
            f.write("=" * 30 + "\n")

            # 1. All Cases
            stats_all = analyze_adversarial(df)
            write_stats(f, stats_all, "All Cases")

            # 2. Poisoned Apple candidates: zero adoption plus a market switch.
            df_poisoned = df[(df['added_strategy_prob'] < 1e-9)
                             & df['market_changed']]
            stats_poisoned = analyze_adversarial(df_poisoned)
            write_stats(f, stats_poisoned,
                        "Poisoned Apple Candidates (Zero Adoption + Market Change)")

    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    main()
