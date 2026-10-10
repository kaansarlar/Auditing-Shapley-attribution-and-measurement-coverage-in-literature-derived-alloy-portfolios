"""Structural counterfactual that restores the seven excluded S15 T_peak rows.

This is not a valid Curie-temperature analysis.  It is used only to show how
the empirical game's support geometry changes when the Cr-free Mn-Ni-Cu class
is restored.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_empirical_nash_shapley import add_utilities, prepare_portfolio, shapley, value_table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3"
OUT = ROOT / "results" / "s15_proxy_counterfactual_v3_2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    strict = pd.read_csv(DATA / "master_strict_tc_72.csv")
    full = pd.read_csv(DATA / "master_full_199.csv")
    s15 = full[full["reference_id"].astype(str).eq("[15]")].copy()
    if len(s15) != 7:
        raise RuntimeError(f"Expected seven S15 proxy rows, found {len(s15)}")

    counterfactual = pd.concat([strict, s15], ignore_index=True)
    portfolio = prepare_portfolio(counterfactual)
    portfolio, utility_meta = add_utilities(portfolio)
    eligible = portfolio[portfolio["game_eligible"].astype(bool)].copy()
    coalition_df, values = value_table(portfolio)
    allocation = shapley(values)

    support_counts = (
        eligible.groupby("optional_players", sort=False)
        .size().rename("strategy_count").reset_index()
    )
    allocation_df = pd.DataFrame({"player": list(allocation), "shapley": list(allocation.values())})
    allocation_df["rank"] = allocation_df["shapley"].rank(method="min", ascending=False).astype(int)

    portfolio.to_csv(OUT / "counterfactual_portfolio.csv", index=False)
    support_counts.to_csv(OUT / "support_class_counts.csv", index=False)
    coalition_df.to_csv(OUT / "coalition_values.csv", index=False)
    allocation_df.to_csv(OUT / "coalition_shapley.csv", index=False)
    summary = {
        "status": "invalid-target structural counterfactual; not a Curie-temperature sensitivity analysis",
        "restored_source": "S15 / reference [15]",
        "restored_rows": int(len(s15)),
        "proxy_semantics": "entropy-peak temperatures (T_peak), not strict Curie temperatures",
        "eligible_strategies": int(len(eligible)),
        "support_classes": int(len(support_counts)),
        "cr_is_veto_player": bool(eligible["optional_players"].str.split(";").map(lambda x: "Cr" in x).all()),
        "shapley": {k: float(v) for k, v in allocation.items()},
        "utility_metadata": utility_meta,
        "interpretation": (
            "Restoring the excluded S15 proxy rows adds a Cr-free Mn-Ni-Cu support class, "
            "removes Cr veto status and changes the allocation. This demonstrates support-set "
            "dependence only; it does not justify treating T_peak as T_C."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
