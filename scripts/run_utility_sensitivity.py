from __future__ import annotations

import json
import math
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv"
OUT = ROOT / "results/utility_sensitivity_v3"
PLAYERS = ("Mn", "Cr", "Ni", "Al", "Cu")


def coalitions():
    for r in range(len(PLAYERS) + 1):
        yield from (frozenset(c) for c in combinations(PLAYERS, r))


def values(frame: pd.DataFrame, eligibility: str):
    out = {}
    for coalition in coalitions():
        mask = frame[eligibility].astype(bool) & frame.optional_players.map(
            lambda x: set(filter(None, str(x).split(";"))).issubset(coalition)
        )
        out[coalition] = 0.0 if not mask.any() else float(frame.loc[mask, "score"].max())
    return out


def shapley(v):
    n = len(PLAYERS)
    result = {}
    for player in PLAYERS:
        total = 0.0
        others = [p for p in PLAYERS if p != player]
        for r in range(len(others) + 1):
            for tup in combinations(others, r):
                c = frozenset(tup)
                w = math.factorial(r) * math.factorial(n-r-1) / math.factorial(n)
                total += w * (v[c | {player}] - v[c])
        result[player] = total
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(DATA)
    rows, shapley_rows = [], []
    config_id = 0
    for tc_decay, qlo, qhi, reliability_base, ambiguity_penalty in product(
        (30.0, 50.0, 75.0), (0.05, 0.10, 0.20), (0.95,), (0.75, 0.85, 0.95), (0.60, 0.75, 0.90, 1.00)
    ):
        # qhi is paired to qlo to preserve symmetric clipping.
        qhi = 1 - qlo
        config_id += 1
        p = base.copy()
        distance = np.where(p.TC_K < 250, 250-p.TC_K, np.where(p.TC_K > 350, p.TC_K-350, 0.0))
        p["uT"] = np.exp(-distance/tc_decay)
        z = np.log1p(p.dS_ref_median.to_numpy(float))
        lo, hi = np.quantile(z, [qlo, qhi])
        p["uS"] = np.clip((z-lo)/(hi-lo), 0, 1)
        rep = reliability_base + (1-reliability_base)*(1-np.exp(-(p.measurements-1)))
        penalty = np.where(p.formula_ambiguous.astype(bool), ambiguity_penalty, 1.0)
        p["uR"] = np.clip(rep*np.exp(-p.dS_ref_cv)*penalty, 0, 1)
        p["score"] = np.power(np.maximum(p.uT*p.uS*p.uR, 1e-12), 1/3)
        for scenario, eligibility in (("primary", "game_eligible"), ("phase_screened", "phase_screen_eligible")):
            eligible = p[p[eligibility].astype(bool)]
            best = eligible.sort_values(["score", "source_rows"], ascending=[False, True]).iloc[0]
            rows.append({
                "config_id": config_id, "scenario": scenario, "tc_decay_K": tc_decay,
                "ds_quantile_low": qlo, "ds_quantile_high": qhi,
                "reliability_base": reliability_base, "ambiguity_penalty": ambiguity_penalty,
                "best_composition": best.composition, "best_source_rows": best.source_rows,
                "best_score": best.score,
            })
            sv = shapley(values(p, eligibility))
            ranks = pd.Series(sv).rank(method="min", ascending=False).astype(int)
            shapley_rows.extend({"config_id": config_id, "scenario": scenario, "player": player,
                                 "shapley": value, "rank": int(ranks[player])}
                                for player, value in sv.items())
    result = pd.DataFrame(rows)
    sv = pd.DataFrame(shapley_rows)
    result.to_csv(OUT / "utility_parameter_grid.csv", index=False)
    sv.to_csv(OUT / "utility_parameter_shapley.csv", index=False)
    frequency = result.groupby(["scenario", "best_composition"]).size().reset_index(name="count")
    totals = result.groupby("scenario").size()
    frequency["frequency"] = frequency["count"] / frequency["scenario"].map(totals)
    frequency.to_csv(OUT / "best_strategy_frequency.csv", index=False)
    rank_summary = sv.groupby(["scenario", "player"]).agg(
        mean_rank=("rank", "mean"), rank1_frequency=("rank", lambda x: (x == 1).mean()),
        mean_shapley=("shapley", "mean"), min_shapley=("shapley", "min"), max_shapley=("shapley", "max"),
    ).reset_index()
    rank_summary.to_csv(OUT / "shapley_rank_sensitivity.csv", index=False)
    top = frequency.sort_values(["scenario", "frequency"], ascending=[True, False]).groupby("scenario").first().reset_index()
    summary = {
        "configurations": int(config_id),
        "varied_parameters": {"tc_decay_K": [30,50,75], "ds_clip_quantiles": ["5–95","10–90","20–80"],
                              "reliability_base": [0.75,0.85,0.95], "ambiguity_penalty": [0.6,0.75,0.9,1.0]},
        "most_frequent_best_strategy": top.to_dict(orient="records"),
        "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(rank_summary.to_string(index=False))


if __name__ == "__main__":
    main()
