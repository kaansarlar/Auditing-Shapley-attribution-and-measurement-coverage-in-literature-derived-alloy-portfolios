from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "empirical_nash_shapley_v3" / "empirical_strategy_portfolio.csv"
OUT = ROOT / "results" / "decision_rule_benchmark_v3"
EPS = 1e-12


def rule_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    utilities = frame[["u_T_window", "u_S", "u_R"]].to_numpy(float)
    return {
        "nash_equal": np.exp(np.log(np.maximum(utilities, EPS)).mean(axis=1)),
        "maximin": utilities.min(axis=1),
        "arithmetic_equal": utilities.mean(axis=1),
        "tc_only": utilities[:, 0],
        "dsref_only": utilities[:, 1],
        "reliability_only": utilities[:, 2],
    }


def benchmark(frame: pd.DataFrame, scenario: str, n_preferences: int = 20000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    utilities = frame[["u_T_window", "u_S", "u_R"]].to_numpy(float)
    log_u = np.log(np.maximum(utilities, EPS))
    weights = rng.dirichlet(np.ones(3), size=n_preferences)
    preference_utility = np.exp(weights @ log_u.T)
    oracle = preference_utility.max(axis=1)
    scores = rule_scores(frame)
    rows = []
    selections = []
    source_order = frame["source_rows"].astype(str).str.split(";").str[0].astype(int).to_numpy()
    for rule, score in scores.items():
        # Resolve exact score ties by original source-row order, never by the
        # Nash-sorted input order. This prevents a single-objective rule from
        # inheriting the Nash ranking through an incidental stable sort.
        order = np.lexsort((source_order, -score))
        for budget in (1, 3, 5):
            chosen = order[: min(budget, len(order))]
            achieved = preference_utility[:, chosen].max(axis=1)
            regret = oracle - achieved
            rows.append({
                "scenario": scenario,
                "rule": rule,
                "budget": budget,
                "mean_regret": float(regret.mean()),
                "median_regret": float(np.median(regret)),
                "regret_q90": float(np.quantile(regret, 0.90)),
                "regret_q95": float(np.quantile(regret, 0.95)),
                "mean_achieved_utility": float(achieved.mean()),
                "near_oracle_frequency_0p01": float((regret <= 0.01).mean()),
            })
            for rank, idx in enumerate(chosen, 1):
                selections.append({
                    "scenario": scenario,
                    "rule": rule,
                    "budget": budget,
                    "selection_rank": rank,
                    "composition": frame.iloc[idx]["composition"],
                    "source_rows": frame.iloc[idx]["source_rows"],
                    "TC_K": frame.iloc[idx]["TC_K"],
                    "dS_ref_median": frame.iloc[idx]["dS_ref_median"],
                    "u_T_window": frame.iloc[idx]["u_T_window"],
                    "u_S": frame.iloc[idx]["u_S"],
                    "u_R": frame.iloc[idx]["u_R"],
                })
    return pd.DataFrame(rows), pd.DataFrame(selections)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    portfolio = pd.read_csv(DATA)
    primary = portfolio[portfolio["game_eligible"].astype(bool)].reset_index(drop=True)
    phase = portfolio[portfolio["phase_screen_eligible"].astype(bool)].reset_index(drop=True)
    result_a, selection_a = benchmark(primary, "primary_observed_portfolio", seed=2026)
    result_b, selection_b = benchmark(phase, "phase_screened_portfolio", seed=31415)
    results = pd.concat([result_a, result_b], ignore_index=True)
    selections = pd.concat([selection_a, selection_b], ignore_index=True)
    results.to_csv(OUT / "preference_robustness_benchmark.csv", index=False)
    selections.to_csv(OUT / "equal_budget_selections.csv", index=False)

    best = (
        results.sort_values(["scenario", "budget", "mean_regret", "regret_q95"])
        .groupby(["scenario", "budget"], as_index=False)
        .first()
    )
    best.to_csv(OUT / "best_rule_by_budget.csv", index=False)
    summary = {
        "purpose": "Diagnostic comparison of fixed decision rules under unknown stakeholder preferences; not a predictive model benchmark.",
        "preference_sampling": "20,000 Dirichlet(1,1,1) weights over Tc-window, dS_ref, and reliability utilities",
        "budgets": [1, 3, 5],
        "rules": sorted(results["rule"].unique().tolist()),
        "best_by_scenario_and_budget": best.to_dict(orient="records"),
        "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
