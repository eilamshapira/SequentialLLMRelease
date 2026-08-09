"""The regulator's meta-game: pick the market whose equilibria maximize its metric.

evaluate_market computes the Nash equilibria of one market's payoff
matrices, evaluates each equilibrium separately, and then aggregates those
values ('average' averages across equilibria; 'any' picks the equilibrium
with the highest designer value). game_with_designer applies it to every market and
returns the market that maximizes the designer metric. All downstream
analyses and figures use these two functions — there is exactly one
implementation of the aggregation logic in the repository.
"""

import numpy as np

from core.data_manager import get_all_markets, get_game
from core.solver import find_equilibrium


def get_subset(metrics, subset_of_strategies):
    # subset_of_strategies is a dict with keys 'alice' and 'bob', each containing a list of strategies to keep
    subset_metrics = {}
    for key, df in metrics.items():
        subset_df = df.loc[subset_of_strategies['alice'], subset_of_strategies['bob']]
        subset_metrics[key] = subset_df
    return subset_metrics


def evaluate_market(subset_metrics, designer_metric="fairness", equilibrium_type="mixed",
                    mixed_mode="average", solver_backend="gambit"):
    """Solve one market and aggregate the designer metric over its equilibria.

    Returns None when the market has no equilibrium (possible in pure mode),
    otherwise a dict with:
        designer_value, alice_gain, bob_gain   means of the values evaluated
                                               separately at each equilibrium
        profile        the aggregated equilibrium profile: the equilibrium
                       itself ('any'/pure) or the average profile ('average')
        equilibria     the equilibria used for the values above ('average'
                       collapses them to the single averaged profile)
        num_equilibria the raw equilibrium count
    """
    nash_equilibria = find_equilibrium(subset_metrics['alice_self_gain'], subset_metrics['bob_self_gain'],
                                       equilibrium_type=equilibrium_type, mixed_mode=mixed_mode,
                                       solver_backend=solver_backend)
    if not nash_equilibria:
        return None

    num_equilibria = len(nash_equilibria)
    designer_value = -np.inf
    alice_gain = -np.inf
    bob_gain = -np.inf
    profile = None

    if equilibrium_type == "mixed":
        designer_matrix = subset_metrics[designer_metric].to_numpy()
        alice_matrix = subset_metrics['alice_self_gain'].to_numpy()
        bob_matrix = subset_metrics['bob_self_gain'].to_numpy()

        values = [(sigma_alice @ designer_matrix @ sigma_bob,
                   sigma_alice @ alice_matrix @ sigma_bob,
                   sigma_alice @ bob_matrix @ sigma_bob)
                  for sigma_alice, sigma_bob in nash_equilibria]

        if mixed_mode == "average":
            designer_value = np.mean([v[0] for v in values])
            alice_gain = np.mean([v[1] for v in values])
            bob_gain = np.mean([v[2] for v in values])
            profile = (np.mean([eq[0] for eq in nash_equilibria], axis=0),
                       np.mean([eq[1] for eq in nash_equilibria], axis=0))
            # Collapse to the single averaged profile so downstream plotting
            # and adoption-probability logic see one consistent equilibrium.
            nash_equilibria = [profile]
        else:
            for eq, (v_designer, v_alice, v_bob) in zip(nash_equilibria, values):
                if v_designer > designer_value:
                    designer_value = v_designer
                    alice_gain = v_alice
                    bob_gain = v_bob
                    profile = eq
    else:
        for eq in nash_equilibria:
            eq_alice, eq_bob = eq[0], eq[1]
            v_designer = subset_metrics[designer_metric].loc[eq_alice, eq_bob]
            if v_designer > designer_value:
                designer_value = v_designer
                alice_gain = subset_metrics['alice_self_gain'].loc[eq_alice, eq_bob]
                bob_gain = subset_metrics['bob_self_gain'].loc[eq_alice, eq_bob]
                profile = (eq_alice, eq_bob)

    return {
        'designer_value': designer_value,
        'alice_gain': alice_gain,
        'bob_gain': bob_gain,
        'profile': profile,
        'equilibria': nash_equilibria,
        'num_equilibria': num_equilibria,
    }


def game_with_designer_from_games(games, subset_of_strategies=None, designer_metric="fairness",
                                  equilibrium_type="mixed", mixed_mode="average",
                                  solver_backend="gambit"):
    """Like game_with_designer, but over pre-pivoted market matrices.

    games: {market_name: matrices dict} as returned by data_manager.get_game.
    Pre-pivoting once per family (instead of once per subset) saves minutes
    of redundant work per full run.
    """
    results = {}
    for market, game_data in games.items():
        if subset_of_strategies:
            subset_metrics = get_subset(game_data, subset_of_strategies)
        else:
            subset_metrics = game_data

        evaluation = evaluate_market(subset_metrics, designer_metric=designer_metric,
                                     equilibrium_type=equilibrium_type, mixed_mode=mixed_mode,
                                     solver_backend=solver_backend)
        if evaluation is None:
            continue

        results[market] = {
            'nash_equilibria': evaluation['equilibria'],
            'num_equilibria': evaluation['num_equilibria'],
            'alice_self_gain': evaluation['alice_gain'],
            'bob_self_gain': evaluation['bob_gain'],
            'designer_values': evaluation['designer_value'],
            'equilibrium_profile': evaluation['profile'],
        }

    if not results:
        return None

    # select the market with the highest designer value.
    best_market = max(results, key=lambda m: results[m]['designer_values'])

    best_market_results = results[best_market]
    best_market_results['market'] = best_market
    return best_market_results


def game_with_designer(data, subset_of_strategies=None, designer_metric="fairness",
                       equilibrium_type="mixed", mixed_mode="average",
                       solver_backend="gambit"):
    games = {market: get_game(data, market) for market in get_all_markets(data)}
    return game_with_designer_from_games(games, subset_of_strategies=subset_of_strategies,
                                         designer_metric=designer_metric,
                                         equilibrium_type=equilibrium_type, mixed_mode=mixed_mode,
                                         solver_backend=solver_backend)
