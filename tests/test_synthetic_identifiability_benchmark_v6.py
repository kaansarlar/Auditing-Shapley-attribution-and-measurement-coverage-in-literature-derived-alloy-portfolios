from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_synthetic_identifiability_benchmark_v6 as bench  # noqa: E402


def test_complete_noise_free_design_recovers_exact_shapley():
    rng = np.random.default_rng(11)
    score, _, _ = bench.generate_truth(rng, "interaction")
    true_phi = bench.shapley(bench.coalition_values(score))
    _, estimated_phi, selected = bench.solve_observed(bench.MASKS, score)
    assert np.allclose(estimated_phi, true_phi, atol=1e-12)
    assert selected == bench.unique_top(score)


def test_shapley_efficiency_on_complete_game():
    rng = np.random.default_rng(12)
    score, _, _ = bench.generate_truth(rng, "additive")
    values = bench.coalition_values(score)
    phi = bench.shapley(values)
    assert np.isclose(phi.sum(), values[-1] - values[0], atol=1e-12)


def test_structural_diagnostics_detect_veto_and_equivalence():
    supports = np.array([0b00001, 0b00111, 0b11001, 0b11111], dtype=int)
    groups = np.array([0, 0, 1, 1], dtype=int)
    diag = bench.support_diagnostics(supports, groups)
    assert diag["veto_count"] == 1
    assert diag["equivalent_pair_count"] >= 1
    assert diag["n_unique_supports"] == 4


def test_unique_top_abstains_on_ties():
    assert bench.unique_top(np.array([0.4, 0.4, 0.2])) == -1
    assert bench.unique_top(np.array([0.4, 0.3, 0.2])) == 0


def test_group_bootstrap_cannot_create_unobserved_support():
    supports = np.array([1, 1, 3, 3, 5, 5], dtype=int)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    observed = np.linspace(0.2, 0.8, len(supports))
    phi, top = bench.unit_bootstrap(
        np.random.default_rng(9), supports, groups, observed, n_boot=20
    )
    assert phi.shape == (20, 5)
    assert top.shape == (20,)
    assert np.all(np.isfinite(phi))
