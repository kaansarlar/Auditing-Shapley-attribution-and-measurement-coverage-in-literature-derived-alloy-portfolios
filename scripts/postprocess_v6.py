from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "v6_postprocessing"
PLAYERS = ("Mn", "Cr", "Ni", "Al", "Cu")


def fit_auc(df: pd.DataFrame, features: list[str], target: np.ndarray, seed: int = 20261006):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=target)
    model = RandomForestClassifier(
        n_estimators=600, min_samples_leaf=12, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    model.fit(df.iloc[tr][features], target[tr])
    prob = model.predict_proba(df.iloc[te][features])[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "auc": float(roc_auc_score(target[te], prob)),
        "balanced_accuracy": float(balanced_accuracy_score(target[te], pred)),
    }


def summarize_primary(df: pd.DataFrame):
    phi = df[[f"true_phi_{p}" for p in PLAYERS]].to_numpy()
    random_material = (phi < 0.5 * phi.max(axis=1, keepdims=True)).mean(axis=1)
    df = df.copy()
    df["random_material_error_probability"] = random_material
    classes = {
        "neither": (df.veto_count == 0) & (df.equivalent_pair_count == 0),
        "equivalent only": (df.veto_count == 0) & (df.equivalent_pair_count > 0),
        "veto only": (df.veto_count > 0) & (df.equivalent_pair_count == 0),
        "veto + equivalent": (df.veto_count > 0) & (df.equivalent_pair_count > 0),
    }
    rows = []
    rng = np.random.default_rng(20261006)
    for label, mask in classes.items():
        x = df[mask & (df.estimated_top_tied == 0)].copy()
        diff = x.material_error_given_unique.to_numpy() - x.random_material_error_probability.to_numpy()
        boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(20000)])
        rows.append({
            "structural_class": label, "unique_top_worlds": int(len(x)),
            "estimated_material_error": float(x.material_error_given_unique.mean()),
            "uniform_random_material_error": float(x.random_material_error_probability.mean()),
            "estimated_minus_random": float(diff.mean()),
            "difference_ci_low": float(np.quantile(boot, 0.025)),
            "difference_ci_high": float(np.quantile(boot, 0.975)),
        })
    comparison = pd.DataFrame(rows)

    bootstrap = {}
    for unit in ("group", "publication"):
        confident = (df[f"{unit}_bootstrap_modal_top"] >= 0) & (df[f"{unit}_bootstrap_top_frequency"] >= 0.80)
        wrong = df[f"{unit}_bootstrap_confident_wrong"] == 1
        bootstrap[unit] = {
            "confident_worlds": int(confident.sum()),
            "confident_wrong_worlds": int(wrong.sum()),
            "confident_wrong_all_worlds": float(wrong.mean()),
            "wrong_given_confident": float(wrong[confident].mean()),
        }

    structural = [
        "n_records", "n_groups", "n_publications", "n_unique_supports", "support_entropy",
        "main_effect_rank", "pairwise_rank", "veto_count", "equivalent_pair_count",
        "players_with_contrast", "max_player_incidence", "max_support_incidence",
        "effective_research_groups", "largest_group_share", "mean_group_modal_support_fraction",
        "group_support_nmi",
    ]
    outcome_aware = structural + [
        "group_bootstrap_top_frequency", "group_bootstrap_tie_frequency",
        "group_bootstrap_mean_ci_width", "estimated_top_margin",
    ]
    failure = df.identification_failure.to_numpy(dtype=int)
    unique = df[df.estimated_top_tied == 0].copy()
    unique_failure = (1 - unique.top_player_correct_given_unique).to_numpy(dtype=int)
    auc = {
        "all_worlds_structural_only": fit_auc(df, structural, failure),
        "all_worlds_outcome_aware": fit_auc(df, outcome_aware, failure),
        "unique_top_structural_only": fit_auc(unique, structural, unique_failure),
        "unique_top_outcome_aware": fit_auc(unique, outcome_aware, unique_failure),
    }
    return df, comparison, bootstrap, auc


def summarize_hea(hea: pd.DataFrame):
    exact = (
        (hea.n_unique_supports == 4) & (hea.main_effect_rank == 4) &
        (hea.veto_count == 1) & (hea.equivalent_pair_count == 1) &
        (hea.players_with_contrast == 2)
    )
    rows = []
    for label, mask in (("all HEA-like", np.ones(len(hea), dtype=bool)), ("exact HEA match", exact)):
        x = hea[mask].copy(); u = x[x.estimated_top_tied == 0]
        row = {
            "subset": label, "worlds": int(len(x)), "unique_top_worlds": int(len(u)),
            "material_error_given_unique": float(u.material_error_given_unique.mean()),
            "wrong_top_given_unique": float(1 - u.top_correct_given_unique.mean()),
            "estimated_top_is_veto_given_unique": float(u.estimated_top_is_veto_given_unique.mean()),
            "mean_regret_observed_oracle": float(x.normalized_regret_observed_oracle.mean()),
            "mean_regret_full_design": float(x.normalized_regret_full_design.mean()),
        }
        for unit in ("group", "publication"):
            confident = x[f"{unit}_bootstrap_confident"] == 1
            wrong = x[f"{unit}_bootstrap_confident_wrong"] == 1
            row[f"{unit}_confident_worlds"] = int(confident.sum())
            row[f"{unit}_confident_wrong_all"] = float(wrong.mean())
            row[f"{unit}_wrong_given_confident"] = float(wrong[confident].mean()) if confident.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_independent(df: pd.DataFrame):
    unique = df.top_hat >= 0
    rows = []
    classes = {
        "neither": (df.veto == 0) & (df.equiv_pairs == 0),
        "equivalent only": (df.veto == 0) & (df.equiv_pairs > 0),
        "veto only": (df.veto > 0) & (df.equiv_pairs == 0),
        "veto + equivalent": (df.veto > 0) & (df.equiv_pairs > 0),
        "all unique-top worlds": np.ones(len(df), dtype=bool),
    }
    for label, mask in classes.items():
        x = df[mask & unique]
        rows.append({
            "subset": label, "unique_top_worlds": int(len(x)),
            "estimated_material_error": float(x.material_error.mean()),
            "uniform_random_material_error": float(x.random_material_error.mean()),
            "estimated_minus_random": float((x.material_error - x.random_material_error).mean()),
        })
    return pd.DataFrame(rows)


def make_figure(primary: pd.DataFrame, hea_summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax = axes[0]
    x = np.arange(len(primary)); w = 0.36
    ax.bar(x - w / 2, primary.estimated_material_error, w, label="Estimated top")
    ax.bar(x + w / 2, primary.uniform_random_material_error, w, label="Uniform random")
    ax.set_xticks(x, ["Neither", "Equivalent\nonly", "Veto\nonly", "Veto +\nequivalent"])
    ax.set_ylabel("Material-error probability")
    ax.set_ylim(0, 0.48); ax.legend(frameon=False)
    ax.set_title("a  Fair comparison among unique-top worlds")
    ax = axes[1]
    vals = hea_summary.iloc[1]
    ax.bar([0, 1, 2], [vals.material_error_given_unique, vals.group_wrong_given_confident,
                        vals.publication_wrong_given_confident], color=["#2563EB", "#D97706", "#DC2626"])
    ax.set_xticks([0, 1, 2], ["Material\nerror", "Wrong | group\nconfident", "Wrong | publication\nconfident"])
    ax.set_ylim(0, 1); ax.set_ylabel("Probability")
    ax.set_title(f"b  Exact HEA geometry (n={int(vals.worlds)})")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_v6_random_and_matched_HEA.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Figure_v6_random_and_matched_HEA.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "primary_generator_v5" / "scenario_results_v5.csv.gz")
    df, comparison, bootstrap, auc = summarize_primary(df)
    comparison.to_csv(OUT / "primary_random_baseline_by_structure.csv", index=False)
    (OUT / "primary_conditional_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (OUT / "diagnostic_auc_v6.json").write_text(json.dumps(auc, indent=2), encoding="utf-8")

    hea = pd.read_csv(RESULTS / "synthetic_identifiability_benchmark_v6" / "hea_like_worlds_v6.csv")
    hea_summary = summarize_hea(hea)
    hea_summary.to_csv(OUT / "hea_exact_match_summary_v6.csv", index=False)

    independent = pd.read_csv(RESULTS / "robustness_generator_R" / "bench_all.csv")
    independent_summary = summarize_independent(independent)
    independent_summary.to_csv(OUT / "independent_generator_summary_v6.csv", index=False)
    make_figure(comparison, hea_summary)
    summary = {"primary_bootstrap": bootstrap, "diagnostic_auc": auc,
               "exact_hea_match": hea_summary.iloc[1].to_dict(),
               "primary_random_comparison": comparison.to_dict(orient="records"),
               "independent_generator": independent_summary.to_dict(orient="records")}
    (OUT / "summary_v6.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
