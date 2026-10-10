from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_strict_tc_72.csv"
OUT = ROOT / "results" / "empirical_nash_shapley_v3"
BACKGROUND = ("Fe", "Co")
PLAYERS = ("Mn", "Cr", "Ni", "Al", "Cu")
QUOTIENT_PLAYERS = ("Cr", "Ni", "Cu", "Mn+Al")
PHASE_FEASIBLE = ("single-phase crystalline", "amorphous-containing")
ELEMENTS = (
    "Gd", "Tb", "Dy", "Ho", "Sc", "Er", "Al", "Fe", "Co", "Ni", "La", "Y", "Pr", "Ag",
    "Tm", "Si", "V", "Cu", "Mn", "Ge", "H_element", "Cr", "Mo", "P", "B", "Ga", "Pd",
)


def mode_or_na(series):
    s = series.dropna().astype(str)
    return s.mode().iloc[0] if len(s) else pd.NA


def prepare_portfolio(data):
    d = data[data["chemistry_family"].eq("3d-TM")].copy()
    # A game strategy needs both a strict Curie temperature and a semantically
    # comparable entropy-change payoff. Preserve audited entropy exclusions.
    if "dsref_valid_measurement" in d.columns:
        d = d[d["dsref_valid_measurement"].astype(bool)].copy()
    d["composition_key"] = d[list(ELEMENTS)].round(8).astype(str).agg("|".join, axis=1)
    rows = []
    for key, g in d.groupby("composition_key", sort=False):
        first = g.iloc[0]
        active = tuple(e for e in ELEMENTS if float(first[e]) > 1e-10)
        outside = tuple(e for e in active if e not in set(BACKGROUND) | set(PLAYERS))
        optional = tuple(e for e in PLAYERS if e in active)
        ds = g["dS_ref"].to_numpy(float)
        cv = float(np.std(ds, ddof=1) / abs(np.mean(ds))) if len(ds) > 1 and abs(np.mean(ds)) > 1e-12 else 0.0
        replication = 0.85 + 0.15 * (1 - math.exp(-(len(g) - 1)))
        reliability = replication * math.exp(-cv)
        ambiguous = bool(g["composition_match_ambiguous_within_reference"].astype(bool).any())
        if ambiguous:
            reliability *= 0.75
        rows.append({
            "composition_key": key,
            "composition": str(first["composition"]),
            "source_rows": ";".join(map(str, g["source_row"].astype(int).tolist())),
            "reference_id": str(first["reference_id"]),
            "measurements": int(len(g)),
            "TC_K": float(np.median(g["TC"])),
            "dS_ref_median": float(np.median(ds)),
            "dS_ref_min": float(np.min(ds)),
            "dS_ref_max": float(np.max(ds)),
            "dS_ref_cv": cv,
            "formula_ambiguous": ambiguous,
            "reliability": float(reliability),
            "phase_class": mode_or_na(g["phase_class"]),
            "phase_class_collapsed": mode_or_na(g["phase_class_collapsed"]),
            "active_elements": ";".join(active),
            "optional_players": ";".join(optional),
            "outside_game_elements": ";".join(outside),
            **{e: float(first[e]) for e in ELEMENTS},
        })
    portfolio = pd.DataFrame(rows)
    portfolio["game_eligible"] = (
        portfolio["outside_game_elements"].eq("")
        & portfolio["Fe"].gt(0)
        & portfolio["Co"].gt(0)
    )
    portfolio["phase_screen_eligible"] = (
        portfolio["game_eligible"]
        & portfolio["phase_class_collapsed"].isin(PHASE_FEASIBLE)
    )
    portfolio["resolved_game_eligible"] = portfolio["game_eligible"] & (~portfolio["formula_ambiguous"])
    return portfolio


def add_utilities(portfolio):
    p = portfolio.copy()
    distance = np.where(p["TC_K"] < 250, 250 - p["TC_K"], np.where(p["TC_K"] > 350, p["TC_K"] - 350, 0.0))
    p["u_T_window"] = np.exp(-distance / 50.0)
    p["u_T_point300"] = np.exp(-np.abs(p["TC_K"] - 300.0) / 50.0)
    log_s = np.log1p(p["dS_ref_median"].to_numpy(float))
    lo, hi = np.quantile(log_s, [0.10, 0.90])
    p["u_S"] = np.clip((log_s - lo) / (hi - lo), 0.0, 1.0)
    p["u_R"] = np.clip(p["reliability"], 0.0, 1.0)
    eps = 1e-12
    p["nash_product"] = p["u_T_window"] * p["u_S"] * p["u_R"]
    p["nash_geometric_mean"] = np.power(np.maximum(p["nash_product"], eps), 1 / 3)
    p["nash_point300_product"] = p["u_T_point300"] * p["u_S"] * p["u_R"]
    p["nash_point300_geometric_mean"] = np.power(np.maximum(p["nash_point300_product"], eps), 1 / 3)
    return p, {"dS_log_q10": float(lo), "dS_log_q90": float(hi), "tc_decay_K": 50.0}


def coalitions(players=PLAYERS):
    for r in range(len(players) + 1):
        for c in combinations(players, r):
            yield frozenset(c)


def coalition_value(portfolio, coalition, score_col="nash_geometric_mean", eligibility_col="game_eligible"):
    mask = portfolio[eligibility_col].astype(bool) & portfolio["optional_players"].map(
        lambda x: set(filter(None, str(x).split(";"))).issubset(coalition)
    )
    eligible = portfolio[mask]
    if eligible.empty:
        return 0.0, None
    idx = eligible[score_col].idxmax()
    return float(eligible.loc[idx, score_col]), int(idx)


def value_table(portfolio, score_col="nash_geometric_mean", eligibility_col="game_eligible"):
    rows, values = [], {}
    for coalition in coalitions():
        value, idx = coalition_value(portfolio, coalition, score_col, eligibility_col)
        values[coalition] = value
        rows.append({
            "coalition": "+".join(e for e in PLAYERS if e in coalition) or "empty",
            "coalition_size": len(coalition),
            "value": value,
            "best_composition": None if idx is None else portfolio.loc[idx, "composition"],
            "best_source_rows": None if idx is None else portfolio.loc[idx, "source_rows"],
        })
    return pd.DataFrame(rows), values


def shapley(values):
    n = len(PLAYERS)
    out = {}
    for player in PLAYERS:
        total = 0.0
        others = [p for p in PLAYERS if p != player]
        for r in range(len(others) + 1):
            for subset_tuple in combinations(others, r):
                subset = frozenset(subset_tuple)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                total += weight * (values[subset | {player}] - values[subset])
        out[player] = total
    return out


def shapley_for_players(values, players):
    """Exact Shapley allocation for an explicitly supplied player set."""
    n = len(players)
    out = {}
    for player in players:
        total = 0.0
        others = [p for p in players if p != player]
        for r in range(len(others) + 1):
            for subset_tuple in combinations(others, r):
                subset = frozenset(subset_tuple)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                total += weight * (values[subset | {player}] - values[subset])
        out[player] = total
    return out


def structural_identifiability(portfolio):
    """Diagnose support degeneracy before interpreting player allocations."""
    eligible = portfolio[portfolio["game_eligible"].astype(bool)].copy()
    eligible["support"] = eligible["optional_players"].map(
        lambda x: frozenset(filter(None, str(x).split(";")))
    )
    support_rows = []
    for support, group in eligible.groupby("support", sort=False):
        support_rows.append({
            "support_class": "+".join(e for e in PLAYERS if e in support),
            "strategy_count": int(len(group)),
            "references": ";".join(sorted(group["reference_id"].astype(str).unique())),
            "best_nash": float(group["nash_geometric_mean"].max()),
        })
    support_df = pd.DataFrame(support_rows).sort_values("support_class").reset_index(drop=True)
    incidence = {player: int(eligible["support"].map(lambda s: player in s).sum()) for player in PLAYERS}
    veto_players = [player for player in PLAYERS if eligible["support"].map(lambda s: player in s).all()]
    equivalent_pairs = [
        [left, right]
        for left, right in combinations(PLAYERS, 2)
        if eligible["support"].map(lambda s: (left in s) == (right in s)).all()
    ]

    def quotient_support(support):
        mapped = set()
        for player in ("Cr", "Ni", "Cu"):
            if player in support:
                mapped.add(player)
        if "Mn" in support or "Al" in support:
            mapped.add("Mn+Al")
        return frozenset(mapped)

    eligible["quotient_support"] = eligible["support"].map(quotient_support)
    quotient_values = {}
    for r in range(len(QUOTIENT_PLAYERS) + 1):
        for coalition_tuple in combinations(QUOTIENT_PLAYERS, r):
            coalition = frozenset(coalition_tuple)
            feasible = eligible[eligible["quotient_support"].map(lambda s: s.issubset(coalition))]
            quotient_values[coalition] = 0.0 if feasible.empty else float(feasible["nash_geometric_mean"].max())
    quotient_shapley = shapley_for_players(quotient_values, QUOTIENT_PLAYERS)
    quotient_df = pd.DataFrame({
        "player_block": list(quotient_shapley),
        "shapley": list(quotient_shapley.values()),
    }).sort_values("shapley", ascending=False)
    diagnostic = {
        "eligible_strategies": int(len(eligible)),
        "unique_support_classes": int(len(support_df)),
        "player_incidence": incidence,
        "veto_players": veto_players,
        "equivalent_player_pairs": equivalent_pairs,
        "quotient_players": list(QUOTIENT_PLAYERS),
        "interpretation": (
            "Cr is a structural veto player because every eligible strategy contains Cr. "
            "Mn and Al are support-equivalent because they always co-occur. Element-level "
            "Shapley values therefore describe portfolio support geometry, not separable chemistry."
        ),
    }
    return support_df, quotient_df, diagnostic


def pair_interactions(values):
    n = len(PLAYERS)
    rows = []
    for i, j in combinations(PLAYERS, 2):
        others = [p for p in PLAYERS if p not in {i, j}]
        interaction = 0.0
        for r in range(len(others) + 1):
            for subset_tuple in combinations(others, r):
                subset = frozenset(subset_tuple)
                weight = math.factorial(r) * math.factorial(n - r - 2) / math.factorial(n - 1)
                delta = values[subset | {i, j}] - values[subset | {i}] - values[subset | {j}] + values[subset]
                interaction += weight * delta
        rows.append({"player_i": i, "player_j": j, "interaction": interaction})
    return pd.DataFrame(rows)


def bootstrap_shapley(portfolio, scenario, score_col="nash_geometric_mean", eligibility_col="game_eligible", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    pubs = portfolio.loc[portfolio[eligibility_col], "reference_id"].unique()
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(pubs, len(pubs), replace=True)
        pieces = [portfolio[portfolio["reference_id"].eq(pub)] for pub in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        _, values = value_table(boot, score_col, eligibility_col)
        sv = shapley(values)
        rows.extend({"scenario": scenario, "bootstrap": b, "player": p, "shapley": v} for p, v in sv.items())
    return pd.DataFrame(rows)


def run_game(portfolio, scenario, score_col, eligibility_col="game_eligible"):
    coalition_df, values = value_table(portfolio, score_col, eligibility_col)
    sv = shapley(values)
    shapley_df = pd.DataFrame({"scenario": scenario, "player": list(sv), "shapley": list(sv.values())})
    shapley_df["rank"] = shapley_df["shapley"].rank(method="min", ascending=False).astype(int)
    interactions = pair_interactions(values)
    interactions.insert(0, "scenario", scenario)
    coalition_df.insert(0, "scenario", scenario)
    efficiency_error = sum(sv.values()) - (values[frozenset(PLAYERS)] - values[frozenset()])
    return coalition_df, shapley_df, interactions, float(efficiency_error)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    portfolio = prepare_portfolio(raw)
    portfolio, utility_meta = add_utilities(portfolio)
    portfolio["source_order"] = portfolio["source_rows"].astype(str).str.split(";").str[0].astype(int)
    portfolio = portfolio.sort_values(["nash_geometric_mean", "source_order"], ascending=[False, True]).reset_index(drop=True)
    portfolio.insert(0, "nash_rank", np.arange(1, len(portfolio) + 1))
    portfolio.to_csv(OUT / "empirical_strategy_portfolio.csv", index=False)
    support_df, quotient_df, structural = structural_identifiability(portfolio)
    support_df.to_csv(OUT / "support_class_counts.csv", index=False)
    quotient_df.to_csv(OUT / "quotient_game_shapley.csv", index=False)
    (OUT / "structural_identifiability.json").write_text(
        json.dumps(structural, indent=2), encoding="utf-8"
    )

    outputs = []
    efficiency = {}
    scenarios = [
        ("window_250_350K", "nash_geometric_mean", "game_eligible"),
        ("point_300K", "nash_point300_geometric_mean", "game_eligible"),
        ("window_250_350K_phase_screened", "nash_geometric_mean", "phase_screen_eligible"),
        ("window_250_350K_resolved_formulas", "nash_geometric_mean", "resolved_game_eligible"),
    ]
    for scenario, score, eligibility in scenarios:
        c, s, i, e = run_game(portfolio, scenario, score, eligibility)
        outputs.append((c, s, i)); efficiency[scenario] = e
    pd.concat([x[0] for x in outputs], ignore_index=True).to_csv(OUT / "coalition_values.csv", index=False)
    shapley_all = pd.concat([x[1] for x in outputs], ignore_index=True)
    shapley_all.to_csv(OUT / "coalition_shapley.csv", index=False)
    pd.concat([x[2] for x in outputs], ignore_index=True).to_csv(OUT / "pair_interactions.csv", index=False)

    boot = pd.concat([
        bootstrap_shapley(portfolio, "window_250_350K", "nash_geometric_mean", "game_eligible", seed=42),
        bootstrap_shapley(portfolio, "window_250_350K_phase_screened", "nash_geometric_mean", "phase_screen_eligible", seed=314),
    ], ignore_index=True)
    boot.to_csv(OUT / "shapley_publication_bootstrap.csv", index=False)
    boot_summary = boot.groupby(["scenario", "player"])["shapley"].agg(
        mean="mean", median="median", ci_low=lambda s: s.quantile(0.025), ci_high=lambda s: s.quantile(0.975),
        positive_frequency=lambda s: (s > 0).mean(),
    ).reset_index().sort_values("mean", ascending=False)
    boot_summary.to_csv(OUT / "shapley_bootstrap_summary.csv", index=False)

    eligible = portfolio[portfolio["game_eligible"]]
    best = eligible.sort_values(["nash_geometric_mean", "source_order"], ascending=[False, True]).iloc[0]
    best_score = float(best["nash_geometric_mean"])
    cooptimal = eligible[np.isclose(eligible["nash_geometric_mean"], best_score, rtol=0, atol=1e-12)]
    phase_eligible = portfolio[portfolio["phase_screen_eligible"]]
    best_phase = phase_eligible.sort_values("nash_geometric_mean", ascending=False).iloc[0]
    summary = {
        "game_type": "empirical evidence-constrained cooperative availability game",
        "strategy_space": "observed unique strict 3d-TM compositions containing Fe and Co and no elements outside the five-player set",
        "background": list(BACKGROUND), "players": list(PLAYERS),
        "strict_3d_records": int((raw["chemistry_family"] == "3d-TM").sum()),
        "strict_3d_records_with_valid_dsref": int(((raw["chemistry_family"] == "3d-TM") & raw["dsref_valid_measurement"].astype(bool)).sum()),
        "unique_3d_strategies": int(len(portfolio)),
        "eligible_game_strategies": int(portfolio["game_eligible"].sum()),
        "phase_screened_strategies": int(portfolio["phase_screen_eligible"].sum()),
        "resolved_formula_strategies": int(portfolio["resolved_game_eligible"].sum()),
        "excluded_outside_player_strategies": int((~portfolio["game_eligible"]).sum()),
        "utility_metadata": utility_meta,
        "best_observed_nash_strategy": {
            "composition": str(best["composition"]), "source_rows": str(best["source_rows"]),
            "TC_K": float(best["TC_K"]), "dS_ref_median": float(best["dS_ref_median"]),
            "nash_geometric_mean": float(best["nash_geometric_mean"]),
            "phase_class_collapsed": None if pd.isna(best["phase_class_collapsed"]) else str(best["phase_class_collapsed"]),
        },
        "cooptimal_observed_nash_strategies": [
            {"composition": str(r.composition), "source_rows": str(r.source_rows), "TC_K": float(r.TC_K),
             "dS_ref_median": float(r.dS_ref_median), "nash_geometric_mean": float(r.nash_geometric_mean),
             "phase_class_collapsed": None if pd.isna(r.phase_class_collapsed) else str(r.phase_class_collapsed)}
            for r in cooptimal.itertuples()
        ],
        "best_phase_screened_nash_strategy": {
            "composition": str(best_phase["composition"]), "source_rows": str(best_phase["source_rows"]),
            "TC_K": float(best_phase["TC_K"]), "dS_ref_median": float(best_phase["dS_ref_median"]),
            "nash_geometric_mean": float(best_phase["nash_geometric_mean"]),
            "phase_class_collapsed": str(best_phase["phase_class_collapsed"]),
        },
        "shapley_efficiency_error": efficiency,
        "structural_identifiability": structural,
        "interpretation_guardrail": "Coalition Shapley quantifies marginal access to the observed multi-objective portfolio. Cr is a structural veto player and Mn-Al are support-equivalent; the allocation is not a separable elemental or causal effect and is not a feature-SHAP value.",
        "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    primary_shapley = shapley_all[shapley_all["scenario"].eq("window_250_350K")].sort_values("rank")
    report = [
        "# Empirical Nash–Shapley game v1", "",
        f"The game contains {summary['eligible_game_strategies']} observed unique Fe–Co-background strategies. The best observed Nash compromise is **{best['composition']}** (Tc={best['TC_K']:.1f} K, dS_ref={best['dS_ref_median']:.3f}, Nash geometric mean={best['nash_geometric_mean']:.3f}).", "",
        "Primary coalition-Shapley allocation (structurally constrained):", "",
    ]
    report += [f"- {r.player}: {r.shapley:.4f}" for r in primary_shapley.itertuples()]
    report += [
        "",
        f"With the supplementary phase screen, the best observed compromise is **{best_phase['composition']}** (Tc={best_phase['TC_K']:.1f} K, dS_ref={best_phase['dS_ref_median']:.3f}, phase={best_phase['phase_class_collapsed']}).",
        "",
        "Cr occurs in every eligible strategy and is therefore a veto player. Mn and Al always co-occur and are support-equivalent. These values measure the geometry of access to the observed portfolio, not separable or causal elemental contributions. The quotient game with Mn+Al as one player, the 300 K point-target ablation, phase-screened game, and publication-cluster bootstrap are included in the output tables.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(primary_shapley.to_string(index=False))


if __name__ == "__main__":
    main()
