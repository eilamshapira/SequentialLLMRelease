"""Load the per-market regression estimates and pivot them into payoff matrices.

Input: data/{family}.csv with columns
    family, metric, paramter_coef, value, effect, ci_low, ci_high, market
(the 'paramter_coef' spelling comes from the upstream GLEE schema).
Each market yields four Alice x Bob matrices: alice_self_gain, bob_self_gain,
fairness, efficiency.
"""

import pandas as pd

from core.utils import get_alice_and_bob_names


def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    data = pd.read_csv(file_path)
    # drop duplicate rows
    data = data.drop_duplicates()

    # replace "value" with alice, bob names
    data[['alice', 'bob']] = data['value'].apply(lambda x: pd.Series(get_alice_and_bob_names(x)))

    return data

def get_game(data, market):
    """Filter the dataset for a specific market and return relevant columns."""
    # assert that there are only one family in data.
    assert data['family'].nunique() == 1, "More than one family found in data."
    filtered_data = data[data['market'] == market]
    filtered_data =  filtered_data[['metric', 'alice', 'bob', 'effect']]

    alice_self_gain = filtered_data[filtered_data['metric'] == 'alice_self_gain']
    bob_self_gain = filtered_data[filtered_data['metric'] == 'bob_self_gain']
    fairness = filtered_data[filtered_data['metric'] == 'fairness']
    efficiency = filtered_data[filtered_data['metric'] == 'efficiency']

    data_dict = {
        'alice_self_gain': alice_self_gain,
        'bob_self_gain': bob_self_gain,
        'fairness': fairness,
        'efficiency': efficiency
    }

    # for each dataframe, get 2d array of alice (rows), bob (columns), effect (values)
    for key in data_dict:
        df = data_dict[key]
        pivot_df = df.pivot(index='alice', columns='bob', values='effect')
        data_dict[key] = pivot_df

    # Sort index and columns by name for stable matrix alignment
    for key in data_dict:
        df = data_dict[key]
        df = df.reindex(sorted(df.index), axis=0)
        df = df.reindex(sorted(df.columns), axis=1)
        data_dict[key] = df

    # be sure the indexes and columns are aligned
    alice_names = data_dict['alice_self_gain'].index
    bob_names = data_dict['alice_self_gain'].columns
    for key in data_dict:
        df = data_dict[key]
        assert all(df.index == alice_names), f"Indexes not aligned in {key}"
        assert all(df.columns == bob_names), f"Columns not aligned in {key}"

    return data_dict

def get_all_markets(data):
    """Return markets in deterministic lexicographic tie-breaking order."""
    return sorted(data['market'].unique().tolist())
