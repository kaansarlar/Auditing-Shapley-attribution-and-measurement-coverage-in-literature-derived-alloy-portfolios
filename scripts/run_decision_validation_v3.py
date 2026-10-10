from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import run_empirical_nash_shapley as game


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_strict_tc_72.csv"
GROUPS = ROOT / "inputs" / "data_v2" / "research-groups.csv"
OUT = ROOT / "results" / "decision_validation_v3"
PLAYERS = list(game.PLAYERS)


def build(raw: pd.DataFrame) -> pd.DataFrame:
    portfolio = game.prepare_portfolio(raw)
    portfolio, _ = game.add_utilities(portfolio)
    return portfolio.reset_index(drop=True)


def solve(portfolio: pd.DataFrame, eligibility: str = "game_eligible") -> dict:
    _, values = game.value_table(portfolio, "nash_geometric_mean", eligibility)
    shapley = game.shapley(values)
    eligible = portfolio[portfolio[eligibility]].copy()
    top = float(eligible["nash_geometric_mean"].max())
    ties = eligible[eligible["nash_geometric_mean"].ge(top - 1e-9)]
    return {
        "V_N": float(values[frozenset(PLAYERS)]),
        "top": top,
        "ties": ";".join(f"{r.composition} {r.reference_id}" for r in ties.itertuples()),
        "n_ties": int(len(ties)),
        **{f"phi_{p}": float(v) for p, v in shapley.items()},
    }


def rank_tau(reference: dict, candidate: dict) -> float:
    a = np.array([reference[f"phi_{p}"] for p in PLAYERS])
    b = np.array([candidate[f"phi_{p}"] for p in PLAYERS])
    return float(kendalltau(a, b).correlation)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    group_map = pd.read_csv(GROUPS).set_index("reference_id")["research_group"].to_dict()
    base = build(raw)
    base["research_group"] = base["reference_id"].map(group_map)
    base.to_csv(OUT / "portfolio_with_groups.csv", index=False)

    full = {
        "game_eligible": solve(base),
        "phase_screen_eligible": solve(base, "phase_screen_eligible"),
    }
    (OUT / "full_solution.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    rows = []
    eligible_refs = sorted(base.loc[base["game_eligible"], "reference_id"].unique())
    eligible_groups = sorted(base.loc[base["game_eligible"], "research_group"].dropna().unique())
    scenarios = [("publication", ref, [ref]) for ref in eligible_refs]
    for grp in eligible_groups:
        members = sorted(k for k, v in group_map.items() if v == grp)
        scenarios.append(("research_group", grp, members))
    scenarios.append(("research_groups", "G8+G9", sorted(k for k, v in group_map.items() if v in {"G8", "G9"})))

    for kind, removed, refs in scenarios:
        reduced = raw[~raw["reference_id"].isin(refs)].copy()
        portfolio = build(reduced)
        for eligibility in ("game_eligible", "phase_screen_eligible"):
            result = solve(portfolio, eligibility)
            rows.append({
                "kind": kind,
                "removed": removed,
                "members": ";".join(refs),
                "scenario": eligibility,
                "eligible_strategies": int(portfolio[eligibility].sum()),
                "tau_vs_full": rank_tau(full[eligibility], result),
                "top_player": max(PLAYERS, key=lambda p: result[f"phi_{p}"]),
                **result,
            })
    jackknife = pd.DataFrame(rows)
    jackknife.to_csv(OUT / "jackknife_publication_and_group.csv", index=False)

    # Research-group bootstrap retains the empirical utility vectors and resamples
    # the highest dependence unit available in the game portfolio.
    rng = np.random.default_rng(20261004)
    boot_rows = []
    for eligibility in ("game_eligible", "phase_screen_eligible"):
        units = base.loc[base[eligibility], "research_group"].dropna().unique()
        for b in range(2000):
            picked = rng.choice(units, len(units), replace=True)
            boot = pd.concat([base[base["research_group"].eq(unit)] for unit in picked], ignore_index=True)
            result = solve(boot, eligibility)
            boot_rows.extend(
                {"scenario": eligibility, "bootstrap": b, "player": p, "shapley": result[f"phi_{p}"]}
                for p in PLAYERS
            )
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(OUT / "research_group_bootstrap.csv", index=False)
    bootstrap.groupby(["scenario", "player"])["shapley"].agg(
        mean="mean",
        ci_low=lambda s: s.quantile(0.025),
        ci_high=lambda s: s.quantile(0.975),
        positive_frequency=lambda s: (s > 1e-12).mean(),
    ).reset_index().to_csv(OUT / "research_group_bootstrap_summary.csv", index=False)

    # Field restriction uses only entropy measurements made at H >= 1 T.
    field_portfolio = build(raw[raw["H"].ge(1.0)].copy())
    field_result = {
        "removed_low_field_references": sorted(raw.loc[raw["H"].lt(1.0), "reference_id"].unique()),
        "game_eligible": solve(field_portfolio),
        "phase_screen_eligible": solve(field_portfolio, "phase_screen_eligible"),
    }
    (OUT / "field_ge_1T_sensitivity.json").write_text(json.dumps(field_result, indent=2), encoding="utf-8")

    # Composition-representation sensitivity. The empirical game depends on
    # player presence rather than fraction magnitude, but this check makes that
    # invariance explicit for the retained source element block.
    block_raw = raw.copy()
    for element in game.ELEMENTS:
        block_raw[element] = block_raw[f"{element}_source_block"]
    block_result = solve(build(block_raw))
    representation = {
        "formula_primary": full["game_eligible"],
        "source_block_sensitivity": block_result,
        "same_nash_set": block_result["ties"] == full["game_eligible"]["ties"],
        "shapley_rank_tau": rank_tau(full["game_eligible"], block_result),
    }
    (OUT / "composition_representation_sensitivity.json").write_text(
        json.dumps(representation, indent=2), encoding="utf-8"
    )

    # Utility-alignment null: independently permute thermal and entropy utilities
    # across the 17 eligible strategies while retaining reliability.
    eligible = base[base["game_eligible"]].reset_index(drop=True)
    observed_maximum = float(eligible["nash_geometric_mean"].max())
    rng = np.random.default_rng(20261005)
    n_alignment = 20000
    maxima = np.empty(n_alignment)
    u_t = eligible["u_T_window"].to_numpy(float)
    u_s = eligible["u_S"].to_numpy(float)
    u_r = eligible["u_R"].to_numpy(float)
    for b in range(n_alignment):
        maxima[b] = np.max(np.power(np.maximum(rng.permutation(u_t) * rng.permutation(u_s) * u_r, 1e-12), 1 / 3))
    alignment_p = float((1 + np.sum(maxima >= observed_maximum)) / (n_alignment + 1))
    pd.DataFrame({"permutation": np.arange(n_alignment), "maximum_nash_geometric_mean": maxima}).to_csv(
        OUT / "utility_alignment_null.csv.gz", index=False, compression="gzip"
    )

    # Element-frequency-matched target nulls. The composition/player matrix stays
    # fixed, so every element retains its exact observed frequency; only Tc or
    # dSref is reassigned. Group-block Tc resampling respects the research-group unit.
    null_rows = []
    rngs = {
        "Tc_record": np.random.default_rng(31),
        "dSref_record": np.random.default_rng(32),
        "Tc_group_block": np.random.default_rng(33),
    }
    for null_name, null_rng in rngs.items():
        for permutation in range(1000):
            permuted = raw.copy()
            mask = permuted["chemistry_family"].eq("3d-TM") & permuted["dS_ref"].notna()
            if null_name == "Tc_record":
                permuted.loc[mask, "TC"] = null_rng.permutation(permuted.loc[mask, "TC"].to_numpy())
            elif null_name == "dSref_record":
                permuted.loc[mask, "dS_ref"] = null_rng.permutation(permuted.loc[mask, "dS_ref"].to_numpy())
            else:
                subset = permuted.loc[mask].copy()
                subset["group"] = subset["reference_id"].map(group_map)
                units = subset["group"].dropna().unique()
                donor = dict(zip(units, null_rng.permutation(units)))
                replacement = []
                for row in subset.itertuples():
                    values = subset.loc[subset["group"].eq(donor[row.group]), "TC"].to_numpy()
                    replacement.append(float(null_rng.choice(values)))
                permuted.loc[mask, "TC"] = replacement
            result = solve(build(permuted))
            null_rows.append({
                "null": null_name,
                "permutation": permutation,
                "V_N": result["V_N"],
                **{f"phi_{p}": result[f"phi_{p}"] for p in PLAYERS},
            })
    nulls = pd.DataFrame(null_rows)
    nulls.to_csv(OUT / "element_frequency_matched_nulls.csv.gz", index=False, compression="gzip")
    p_rows = []
    observed = full["game_eligible"]
    for name, frame in nulls.groupby("null"):
        p_rows.append({
            "null": name,
            **{
                f"p_{p}": float((1 + (frame[f"phi_{p}"] >= observed[f"phi_{p}"]).sum()) / (len(frame) + 1))
                for p in PLAYERS
            },
            **{f"null_mean_{p}": float(frame[f"phi_{p}"].mean()) for p in PLAYERS},
        })
    pvalues = pd.DataFrame(p_rows)
    pvalues.to_csv(OUT / "element_frequency_matched_pvalues.csv", index=False)

    summary = {
        "full_solution": full,
        "G8_removal": jackknife[(jackknife["kind"].eq("research_group")) & jackknife["removed"].eq("G8") & jackknife["scenario"].eq("game_eligible")].iloc[0].to_dict(),
        "G9_removal": jackknife[(jackknife["kind"].eq("research_group")) & jackknife["removed"].eq("G9") & jackknife["scenario"].eq("game_eligible")].iloc[0].to_dict(),
        "G8_G9_removal": jackknife[(jackknife["kind"].eq("research_groups")) & jackknife["scenario"].eq("game_eligible")].iloc[0].to_dict(),
        "field_ge_1T": field_result,
        "composition_representation": representation,
        "utility_alignment_null": {
            "permutations": n_alignment,
            "observed_maximum": observed_maximum,
            "p_value": alignment_p,
        },
        "element_frequency_matched_pvalues": pvalues.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
