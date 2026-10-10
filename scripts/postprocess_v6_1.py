from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "v6_1_postprocessing"


def wilson(k: int, n: int, z: float = 1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, ctr - half, ctr + half


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fair = json.loads((RESULTS / "v6_fairness" / "v6_fairness_summary.json").read_text())
    primary = pd.DataFrame(fair["primary_fair_baseline_unique_top"][:4])
    robust = pd.DataFrame(fair["robustness_fair_baseline_unique_top"][:4])
    primary.to_csv(OUT / "primary_fair_baseline_v6_1.csv", index=False)
    robust.to_csv(OUT / "corrected_independent_generator_v6_1.csv", index=False)

    hea = pd.read_csv(RESULTS / "synthetic_identifiability_benchmark_v6" / "hea_like_worlds_v6.csv")
    exact = hea[(hea.n_unique_supports == 4) & (hea.main_effect_rank == 4) &
                (hea.veto_count == 1) & (hea.equivalent_pair_count == 1) &
                (hea.players_with_contrast == 2)].copy()
    group_conf = exact.group_bootstrap_confident == 1
    pub_conf = exact.publication_bootstrap_confident == 1
    g_correct = int(group_conf.sum() - exact.loc[group_conf, "group_bootstrap_confident_wrong"].sum())
    p_correct = int(pub_conf.sum() - exact.loc[pub_conf, "publication_bootstrap_confident_wrong"].sum())
    matched_rows = []
    ph = fair["primary_hea_matched"]
    matched_rows.append({"generator": "Primary", "worlds": 358, "unique_truth_worlds": 358,
        "material_error": ph["material_error"], "random_material_error": ph["random"],
        "difference": ph["difference"], "difference_ci_low": ph["diff_ci_low"],
        "difference_ci_high": ph["diff_ci_high"], "exact_recovery": ph["exact"],
        "exact_ci_low": ph["exact_ci_low"], "exact_ci_high": ph["exact_ci_high"],
        "top_is_veto": ph["top_is_veto"]})
    rh = fair["robustness_hea_matched"]
    matched_rows.append({"generator": "Corrected independent", "worlds": 288,
        "unique_truth_worlds": rh["exact_n_unique_truth"], "material_error": rh["material_error"],
        "random_material_error": rh["random"], "difference": rh["difference"],
        "difference_ci_low": rh["diff_ci_low"], "difference_ci_high": rh["diff_ci_high"],
        "exact_recovery": rh["exact"], "exact_ci_low": rh["exact_ci_low"],
        "exact_ci_high": rh["exact_ci_high"], "top_is_veto": rh["top_is_veto"]})
    matched = pd.DataFrame(matched_rows); matched.to_csv(OUT / "matched_generators_v6_1.csv", index=False)

    gp, gl, gh = wilson(g_correct, int(group_conf.sum()))
    pp, pl, phigh = wilson(p_correct, int(pub_conf.sum()))
    confidence = pd.DataFrame([
        {"result": "Primary exact top", "correct": 75, "n": 358,
         "rate": fair["primary_hea_matched"]["exact"],
         "ci_low": fair["primary_hea_matched"]["exact_ci_low"],
         "ci_high": fair["primary_hea_matched"]["exact_ci_high"]},
        {"result": "Primary group confident", "correct": g_correct, "n": int(group_conf.sum()),
         "rate": gp, "ci_low": gl, "ci_high": gh},
        {"result": "Primary publication confident", "correct": p_correct, "n": int(pub_conf.sum()),
         "rate": pp, "ci_low": pl, "ci_high": phigh},
        {"result": "Independent exact top", "correct": int(round(rh["exact"] * rh["exact_n_unique_truth"])),
         "n": rh["exact_n_unique_truth"], "rate": rh["exact"],
         "ci_low": rh["exact_ci_low"], "ci_high": rh["exact_ci_high"]},
    ])
    confidence.to_csv(OUT / "matched_exact_recovery_wilson_v6_1.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    labels = ["Neither", "Equivalent\nonly", "Veto\nonly", "Veto +\nequivalent"]
    x = np.arange(4); w = 0.36
    axes[0].bar(x-w/2, primary.material_error, w, label="Estimated top", color="#2563EB")
    axes[0].bar(x+w/2, primary.random, w, label="Uniform random", color="#F59E0B")
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, .75)
    axes[0].set_ylabel("Material-error probability"); axes[0].set_title("a  Primary generator")
    axes[0].legend(frameon=False, fontsize=8)

    c = confidence.iloc[:3]
    yerr = np.vstack([c.rate-c.ci_low, c.ci_high-c.rate])
    axes[1].errorbar(np.arange(3), c.rate, yerr=yerr, fmt="o", capsize=4, color="#0F766E")
    axes[1].axhline(.2, color="#DC2626", ls="--", lw=1.4, label="Uniform exact recovery = 0.20")
    axes[1].set_xticks(np.arange(3), ["All", "Group\nconfident", "Publication\nconfident"])
    axes[1].set_ylim(0, .34); axes[1].set_ylabel("Exact top-player recovery")
    axes[1].set_title("b  Exact HEA geometry (primary)"); axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x-w/2, robust.material_error, w, label="Estimated top", color="#7C3AED")
    axes[2].bar(x+w/2, robust.random, w, label="Uniform random", color="#F59E0B")
    axes[2].set_xticks(x, labels); axes[2].set_ylim(0, .75)
    axes[2].set_ylabel("Material-error probability"); axes[2].set_title("c  Corrected independent generator")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_4_v6_1_generator_comparison.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Figure_4_v6_1_generator_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {"primary": primary.to_dict(orient="records"), "corrected_independent": robust.to_dict(orient="records"),
               "matched": matched.to_dict(orient="records"), "confidence": confidence.to_dict(orient="records"),
               "generator_R": {"worlds": 4000, "bootstrap_resamples": 100, "confidence_threshold": 0.90,
                 "performance_selection_lambda": [0.0, 1.5, 3.0], "hea_candidate_worlds": 9000,
                 "hea_exact_matches": 288}}
    (OUT / "summary_v6_1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
