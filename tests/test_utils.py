"""Unit tests for the shared helpers in core/utils.py."""

import pytest

from core.utils import get_alice_and_bob_names, get_bob_strategy, results_suffix, wilson_ci


def test_get_alice_and_bob_names():
    assert get_alice_and_bob_names('alice_gpt-4o_bob_o3-mini') == ('gpt-4o', 'o3-mini')
    # Provider prefixes stay on the Alice side; 'meta/' stays on both
    alice, bob = get_alice_and_bob_names(
        'alice_vertex_ai/meta/llama-3.1-405b-instruct-maas_bob_meta/llama-3.3-70b-instruct-maas')
    assert alice == 'vertex_ai/meta/llama-3.1-405b-instruct-maas'
    assert bob == 'meta/llama-3.3-70b-instruct-maas'


def test_get_bob_strategy_strips_provider_prefixes_only():
    assert get_bob_strategy('vertex_ai/gemini-1.5-pro') == 'gemini-1.5-pro'
    assert get_bob_strategy('xai/grok-2-1212') == 'grok-2-1212'
    # 'meta/' is part of the model name, not a provider prefix to strip
    assert get_bob_strategy('vertex_ai/meta/llama-3.1-405b-instruct-maas') == 'meta/llama-3.1-405b-instruct-maas'
    assert get_bob_strategy('gpt-4o') == 'gpt-4o'


def test_results_suffix():
    assert results_suffix('mixed', 'average') == 'mixed_average'
    assert results_suffix('mixed', 'any') == 'mixed_any'
    assert results_suffix('pure', 'average') == 'pure'


def test_wilson_ci_known_value():
    # 50/100 at 95%: the Wilson interval is (0.4038, 0.5962)
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_wilson_ci_clipped_to_unit_interval():
    # Floating-point rounding can push the raw bounds a few ULPs outside
    # [0, 1]; the clamp must keep them inside.
    low, high = wilson_ci(0, 1000)
    assert 0.0 <= low < 1e-12
    assert 0 < high < 0.01

    low, high = wilson_ci(1000, 1000)
    assert low > 0.99
    assert high == 1.0

    assert wilson_ci(0, 0) == (0, 0)
