from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from run_empirical_nash_shapley import PLAYERS, run_game


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "empirical_nash_shapley_v3" / "empirical_strategy_portfolio.csv"
OUT = ROOT / "results" / "temperature_window_sensitivity_v3"
WINDOW_WIDTH_K = 100
DECAY_K = 50.0
CENTERS_K = np.arange(150, 401, 25)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(INPUT)
    base["source_order"] = base["source_rows"].astype(str).str.split(";").str[0].astype(int)
    baseline_shapley = None
    window_rows, shapley_rows = [], []

    for center in CENTERS_K:
        lower = float(center - WINDOW_WIDTH_K / 2)
        upper = float(center + WINDOW_WIDTH_K / 2)
        p = base.copy()
        distance = np.where(p.TC_K < lower, lower - p.TC_K, np.where(p.TC_K > upper, p.TC_K - upper, 0.0))
        p["u_T_sweep"] = np.exp(-distance / DECAY_K)
        p["nash_sweep"] = np.power(np.maximum(p.u_T_sweep * p.u_S * p.u_R, 1e-12), 1 / 3)

        _, s, _, efficiency_error = run_game(p, f"center_{center}K", "nash_sweep", "game_eligible")
        values = s.set_index("player").reindex(PLAYERS).shapley.to_numpy(float)
        if center == 300:
            baseline_shapley = values.copy()

        eligible = p[p.game_eligible.astype(bool)].copy()
        maximum = eligible.nash_sweep.max()
        ties = eligible[np.isclose(eligible.nash_sweep, maximum, atol=1e-12, rtol=0)].sort_values("source_order")
        phase = p[p.phase_screen_eligible.astype(bool)].sort_values(["nash_sweep", "source_order"], ascending=[False, True])
        phase_best = phase.iloc[0]
        window_rows.append({
            "center_K": int(center), "lower_K": lower, "upper_K": upper,
            "eligible_inside_window": int(((eligible.TC_K >= lower) & (eligible.TC_K <= upper)).sum()),
            "nash_max": float(maximum),
            "cooptimal_references": ";".join(ties.reference_id.astype(str)),
            "cooptimal_compositions": ";".join(ties.composition.astype(str)),
            "phase_priority_reference": str(phase_best.reference_id),
            "phase_priority_composition": str(phase_best.composition),
            "phase_priority_nash": float(phase_best.nash_sweep),
            "efficiency_error": float(efficiency_error),
        })
        for player, value in zip(PLAYERS, values):
            shapley_rows.append({"center_K": int(center), "lower_K": lower, "upper_K": upper, "player": player, "shapley": float(value)})

    windows = pd.DataFrame(window_rows)
    shapley = pd.DataFrame(shapley_rows)
    if baseline_shapley is None:
        raise RuntimeError("The 250-350 K baseline center was not evaluated")
    tau = []
    for center, group in shapley.groupby("center_K", sort=True):
        vector = group.set_index("player").reindex(PLAYERS).shapley.to_numpy(float)
        tau.append((center, float(kendalltau(baseline_shapley, vector).statistic)))
    tau_map = dict(tau)
    windows["shapley_tau_vs_250_350K"] = windows.center_K.map(tau_map)
    windows.to_csv(OUT / "window_summary.csv", index=False)
    shapley.to_csv(OUT / "window_shapley.csv", index=False)

    baseline = windows.loc[windows.center_K.eq(300)].iloc[0]
    phase_counts = windows.phase_priority_reference.value_counts().to_dict()
    summary = {
        "window_width_K": WINDOW_WIDTH_K,
        "decay_K": DECAY_K,
        "centers_K": CENTERS_K.tolist(),
        "baseline_center_K": 300,
        "baseline_cooptimal_references": baseline.cooptimal_references,
        "baseline_phase_priority_reference": baseline.phase_priority_reference,
        "phase_priority_counts": {str(k): int(v) for k, v in phase_counts.items()},
        "minimum_shapley_tau_vs_baseline": float(windows.shapley_tau_vs_250_350K.min()),
        "median_shapley_tau_vs_baseline": float(windows.shapley_tau_vs_250_350K.median()),
        "maximum_efficiency_error": float(windows.efficiency_error.abs().max()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
