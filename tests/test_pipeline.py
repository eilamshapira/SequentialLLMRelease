"""End-to-end tests of the analysis pipeline on the synthetic toy dataset.

The toy fixture (tests/fixtures/toy.csv) has 5 synthetic models and 2
markets; the expected values below are checked against the paper's pinned
Gambit equilibrium-enumeration convention.
"""

import os
import shutil

import pandas as pd
import pytest

from figures.figure2_numbers import panel_a, panel_b, panel_c, panel_d, panel_e, panel_f

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'toy.csv')


@pytest.fixture(scope='module')
def toy_outputs(tmp_path_factory):
    """Run the full pipeline (equilibria + banning) on the toy family."""
    workdir = tmp_path_factory.mktemp('toy_run')
    os.makedirs(workdir / 'data')
    shutil.copyfile(FIXTURE, workdir / 'data' / 'toy.csv')

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        from pipeline.comprehensive_analysis import analyze_family as run_comprehensive
        from pipeline.banning_analysis import analyze_family as run_banning

        run_comprehensive('toy', mode='mixed', mixed_mode='average', n_jobs=2, metric='fairness')
        run_banning('toy', mode='mixed', mixed_mode='average', metric='fairness', max_bans=3)

        transitions = pd.read_csv('output/fairness/calculations/comprehensive_transitions_toy_mixed_average.csv')
        banning = pd.read_csv('output/fairness/calculations/banning_summary_toy_mixed_average.csv')
    finally:
        os.chdir(cwd)

    return transitions, banning


def test_transition_count(toy_outputs):
    transitions, _ = toy_outputs
    # 5 models: sum over base sizes k=2..4 of C(5,k) * (5-k) additions
    assert len(transitions) == 55


def test_known_transition_values(toy_outputs):
    transitions, _ = toy_outputs
    row = transitions[(transitions['base_subset'] == 'model_alpha;model_delta')
                      & (transitions['added_strategy'] == 'model_epsilon')].iloc[0]

    assert row['initial_market'] == 'CI=True_MA=False'
    assert row['final_market'] == 'CI=True_MA=False'
    assert row['initial_designer_value'] == pytest.approx(1.0, abs=1e-6)
    assert row['final_designer_value'] == pytest.approx(0.950366, abs=1e-6)
    # Both players switch to the new model: probability 1 each
    assert row['added_strategy_prob'] == pytest.approx(2.0, abs=1e-9)


def test_banning_summary_shape_and_anchor(toy_outputs):
    _, banning = toy_outputs
    # 4 ban budgets (0..3) x 8 panels
    assert len(banning) == 32
    assert set(banning['panel']) == {
        'opposite_payoff', 'zero_adoption_opposite', 'designer_improve',
        'designer_harm', 'adoption_when_improve', 'adoption_when_harm',
        'market_changed', 'added_strategy_banned'}

    row = banning[(banning['ban_budget'] == 1) & (banning['panel'] == 'opposite_payoff')].iloc[0]
    assert row['count'] == 4
    assert row['total'] == 55
    assert row['percentage'] == pytest.approx(100 * 4 / 55, abs=1e-9)


def _transitions_frame(rows):
    return pd.DataFrame(rows, columns=['delta_alice_gain', 'delta_bob_gain',
                                       'delta_designer_value', 'added_strategy_prob',
                                       'market_changed'])


def test_figure2_panel_formulas():
    # 4 transitions: two opposite shifts (one with zero adoption),
    # one win-win, one unchanged; designer improves twice, harmed once.
    df = _transitions_frame([
        (+0.1, -0.1, -0.05, 0.0, True),   # opposite, zero adoption, harm
        (-0.2, +0.1, +0.10, 0.7, True),   # opposite, adopted, improve
        (+0.1, +0.1, +0.10, 0.0, False),  # win-win, improve without adoption
        (0.0, 0.0, 0.0, 0.0, False),      # no change
    ])

    a = panel_a(df)
    assert a['count'] == 2 and a['total'] == 4
    assert a['percentage'] == pytest.approx(50.0)

    b = panel_b(df)
    assert b['count'] == 1 and b['total'] == 2
    assert b['percentage'] == pytest.approx(50.0)

    c = panel_c(df)
    assert c['improve']['percentage'] == pytest.approx(50.0)
    assert c['harm']['percentage'] == pytest.approx(25.0)

    d = panel_d(df)
    assert d['with_usage'] == 1 and d['n'] == 2
    assert d['percentage'] == pytest.approx(50.0)

    e = panel_e(df)
    assert e['with_usage'] == 0 and e['n'] == 1
    assert e['percentage'] == pytest.approx(0.0)

    f = panel_f(df)
    assert f['count'] == 2 and f['total'] == 4
    assert f['percentage'] == pytest.approx(50.0)


def test_poisoned_apple_requires_market_change():
    df = _transitions_frame([
        (+0.1, -0.1, -0.05, 0.0, False),
    ])
    result = panel_b(df)
    assert result['count'] == 0 and result['total'] == 1
    assert result['percentage'] == pytest.approx(0.0)


def test_figure2_agrees_with_banning_at_zero_budget(toy_outputs):
    """The Figure 2 panel functions and banning_analysis independently
    classify the same transitions; at ban budget N=0 they must agree."""
    transitions, banning = toy_outputs
    at0 = banning[banning['ban_budget'] == 0].set_index('panel')

    assert panel_a(transitions)['count'] == at0.loc['opposite_payoff', 'count']
    assert panel_b(transitions)['count'] == at0.loc['zero_adoption_opposite', 'count']
    c = panel_c(transitions)
    assert c['improve']['count'] == at0.loc['designer_improve', 'count']
    assert c['harm']['count'] == at0.loc['designer_harm', 'count']
    assert panel_f(transitions)['count'] == at0.loc['market_changed', 'count']


def test_banning_dp_deterministic_tie_break():
    """Ties in designer_value must resolve by sorted strategy name, not
    by Python's per-process hash order."""
    from pipeline.banning_analysis import precompute_best_results

    a, b, c = 'model_a', 'model_b', 'model_c'
    full = frozenset({a, b, c})
    # Banning any single strategy yields the same designer value: a tie.
    cache = {
        full: {'designer_value': 1.0},
        frozenset({a, b}): {'designer_value': 2.0},
        frozenset({a, c}): {'designer_value': 2.0},
        frozenset({b, c}): {'designer_value': 2.0},
        frozenset({a}): {'designer_value': 0.0},
        frozenset({b}): {'designer_value': 0.0},
        frozenset({c}): {'designer_value': 0.0},
    }
    dp = precompute_best_results(cache, max_bans=1)
    entry = dp[(full, 1)]
    # sorted iteration bans 'model_a' first and the strict > keeps it
    assert entry['result_key'] == frozenset({b, c})
    assert entry['banned'] == frozenset({a})


def test_market_order_is_lexicographic():
    """The regulator's tie break must not depend on CSV row order."""
    from core.data_manager import get_all_markets

    data = pd.DataFrame({'market': ['market_z', 'market_a', 'market_m', 'market_a']})
    assert get_all_markets(data) == ['market_a', 'market_m', 'market_z']
