from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_strict_tc_72.csv"
OUT = ROOT / "results" / "tc_window_family_validation_v3"
SEED = 42

ELEMENTS = [
    "Gd", "Tb", "Dy", "Ho", "Sc", "Er", "Al", "Fe", "Co", "Ni", "La", "Y", "Pr", "Ag",
    "Tm", "Si", "V", "Cu", "Mn", "Ge", "H_element", "Cr", "Mo", "P", "B", "Ga", "Pd",
]
DESCRIPTORS = [
    "dX", "VEC", "sigma", "dHmix", "dSmix", "nunfilled_mean", "mag_moment_mean",
    "avg_d_valence_electrons", "bandgap_mean", "melting_point_mean", "electronegativity_range",
]


def model_specs(seed=SEED):
    return {
        "logistic": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=seed)),
            ]),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0], "model__class_weight": [None, "balanced"]},
        ),
        "extra_trees": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
            ]),
            {
                "model__max_features": [0.5, 1.0],
                "model__min_samples_leaf": [2, 4, 6],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }


def metric_row(y, prob):
    pred = (prob >= 0.5).astype(int)
    prevalence = float(np.mean(y))
    k = max(1, int(np.ceil(0.30 * len(y))))
    top = np.argsort(-prob)[:k]
    top_precision = float(np.mean(y[top]))
    return {
        "records": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, np.column_stack([1 - prob, prob]), labels=[0, 1])),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "mcc_at_0_5": float(matthews_corrcoef(y, pred)),
        "top30_precision": top_precision,
        "top30_enrichment": float(top_precision / prevalence) if prevalence > 0 else np.nan,
    }


def outer_split(data):
    groups = data["reference_id"].astype(str).to_numpy()
    return list(LeaveOneGroupOut().split(data, groups=groups))


def smoothed_prior_oof(data, family_conditional):
    y = data["in_primary_TC_window_250_350K"].astype(int).to_numpy()
    prob = np.full(len(data), np.nan)
    fold = np.full(len(data), -1, dtype=int)
    for i, (train, test) in enumerate(outer_split(data), start=1):
        train_y = y[train]
        global_p = (train_y.sum() + 1) / (len(train_y) + 2)
        if family_conditional:
            tmp = pd.DataFrame({"family": data.iloc[train]["chemistry_family"].to_numpy(), "y": train_y})
            stats = tmp.groupby("family")["y"].agg(["sum", "count"])
            mapping = ((stats["sum"] + 1) / (stats["count"] + 2)).to_dict()
            prob[test] = [mapping.get(f, global_p) for f in data.iloc[test]["chemistry_family"]]
        else:
            prob[test] = global_p
        fold[test] = i
    return prob, fold


def nested_oof(data, columns, model_name):
    X = data[columns]
    y = data["in_primary_TC_window_250_350K"].astype(int).to_numpy()
    groups = data["reference_id"].astype(str).to_numpy()
    estimator, grid = model_specs()[model_name]
    prob = np.full(len(data), np.nan)
    folds = np.full(len(data), -1, dtype=int)
    fold_rows = []
    for fold, (train, test) in enumerate(outer_split(data), start=1):
        train_groups = groups[train]
        n_inner = min(5, max(3, pd.Series(train_groups).nunique() // 2))
        inner = GroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED + fold)
        search = GridSearchCV(
            clone(estimator), grid, scoring="neg_brier_score", cv=inner,
            n_jobs=-1, refit=True, error_score=np.nan,
        )
        search.fit(X.iloc[train], y[train], groups=train_groups)
        classes = list(search.best_estimator_.classes_)
        p = search.best_estimator_.predict_proba(X.iloc[test])[:, classes.index(1)]
        prob[test] = p
        folds[test] = fold
        fold_rows.append({
            "fold": fold,
            "held_out_publication": str(groups[test][0]),
            "train_records": len(train),
            "test_records": len(test),
            "best_inner_brier": float(-search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })
    assert np.isfinite(prob).all()
    return prob, folds, pd.DataFrame(fold_rows)


def evaluate(data, cohort, feature_sets, selected=None):
    y = data["in_primary_TC_window_250_350K"].astype(int).to_numpy()
    rows, predictions, fold_tables = [], [], []
    family_conditional = data["chemistry_family"].nunique() > 1
    base_prob, base_fold = smoothed_prior_oof(data, family_conditional=family_conditional)
    base_name = "family_smoothed_prior" if family_conditional else "cohort_smoothed_prior"
    rows.append({"cohort": cohort, "feature_set": "baseline", "model": base_name, **metric_row(y, base_prob)})
    predictions.append(pd.DataFrame({
        "cohort": cohort, "feature_set": "baseline", "model": base_name,
        "source_row": data["source_row"], "reference_id": data["reference_id"],
        "chemistry_family": data["chemistry_family"], "observed_window": y,
        "predicted_probability": base_prob, "outer_fold": base_fold,
        "formula_mapping_status": data["formula_mapping_status"],
    }))
    candidates = selected or [(f, m) for f in feature_sets for m in model_specs()]
    for feature_name, model_name in candidates:
        prob, folds, fold_table = nested_oof(data, feature_sets[feature_name], model_name)
        rows.append({"cohort": cohort, "feature_set": feature_name, "model": model_name, **metric_row(y, prob)})
        predictions.append(pd.DataFrame({
            "cohort": cohort, "feature_set": feature_name, "model": model_name,
            "source_row": data["source_row"], "reference_id": data["reference_id"],
            "chemistry_family": data["chemistry_family"], "observed_window": y,
            "predicted_probability": prob, "outer_fold": folds,
            "formula_mapping_status": data["formula_mapping_status"],
        }))
        fold_table.insert(0, "model", model_name)
        fold_table.insert(0, "feature_set", feature_name)
        fold_table.insert(0, "cohort", cohort)
        fold_tables.append(fold_table)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()


def cluster_bootstrap_delta(frame, baseline_model, candidate_model, n_boot=2000, seed=SEED):
    wide = frame[frame["model"].isin([baseline_model, candidate_model])].pivot_table(
        index=["source_row", "reference_id", "observed_window"], columns="model",
        values="predicted_probability", aggfunc="first",
    ).reset_index()
    groups = wide["reference_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(groups, size=len(groups), replace=True)
        pieces = [wide[wide["reference_id"].astype(str).eq(g)] for g in sample]
        boot = pd.concat(pieces, ignore_index=True)
        y = boot["observed_window"].to_numpy(int)
        delta = brier_score_loss(y, boot[candidate_model]) - brier_score_loss(y, boot[baseline_model])
        deltas.append(delta)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(np.mean(deltas)), float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    feature_sets = {"elements": ELEMENTS, "descriptors": DESCRIPTORS, "combined": ELEMENTS + DESCRIPTORS}
    cohorts = {
        "strict_72": data,
        "3d_TM_25": data[data["chemistry_family"].eq("3d-TM")].reset_index(drop=True),
        "TM_metalloid_22": data[data["chemistry_family"].eq("TM-metalloid")].reset_index(drop=True),
    }
    all_metrics, all_predictions, all_folds = [], [], []
    selected = {}
    for name, cohort_data in cohorts.items():
        metrics, pred, folds = evaluate(cohort_data, name, feature_sets)
        learned = metrics[~metrics["feature_set"].eq("baseline")].sort_values(["brier", "average_precision"], ascending=[True, False])
        best = learned.iloc[0]
        selected[name] = (str(best["feature_set"]), str(best["model"]))
        all_metrics.append(metrics); all_predictions.append(pred); all_folds.append(folds)

        resolved = cohort_data[cohort_data["formula_mapping_status"].eq("resolved")].reset_index(drop=True)
        sm, sp, sf = evaluate(resolved, name + "_resolved", feature_sets, selected=[selected[name]])
        all_metrics.append(sm); all_predictions.append(sp); all_folds.append(sf)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    predictions_all = pd.concat(all_predictions, ignore_index=True)
    folds_all = pd.concat(all_folds, ignore_index=True)
    metrics_all.to_csv(OUT / "tc_window_metrics.csv", index=False)
    predictions_all.to_csv(OUT / "tc_window_oof_predictions.csv", index=False)
    folds_all.to_csv(OUT / "tc_window_fold_details.csv", index=False)

    decisions = []
    for name in cohorts:
        metric = metrics_all[metrics_all["cohort"].eq(name)]
        baseline = metric[metric["feature_set"].eq("baseline")].iloc[0]
        feature_name, model_name = selected[name]
        best = metric[(metric["feature_set"].eq(feature_name)) & (metric["model"].eq(model_name))].iloc[0]
        resolved_metric = metrics_all[metrics_all["cohort"].eq(name + "_resolved")]
        resolved_baseline = resolved_metric[resolved_metric["feature_set"].eq("baseline")].iloc[0]
        resolved_best = resolved_metric[(resolved_metric["feature_set"].eq(feature_name)) & (resolved_metric["model"].eq(model_name))].iloc[0]
        cohort_pred = predictions_all[predictions_all["cohort"].eq(name)]
        mean_delta, lo, hi = cluster_bootstrap_delta(cohort_pred, str(baseline["model"]), model_name)
        relative_gain = (baseline["brier"] - best["brier"]) / baseline["brier"]
        resolved_relative_gain = (resolved_baseline["brier"] - resolved_best["brier"]) / resolved_baseline["brier"]
        qualifies = bool(relative_gain >= 0.10 and resolved_relative_gain >= 0.10 and hi < 0)
        decisions.append({
            "cohort": name, "selected_feature_set": feature_name, "selected_model": model_name,
            "baseline_model": str(baseline["model"]),
            "baseline_brier": float(baseline["brier"]), "model_brier": float(best["brier"]),
            "relative_brier_improvement": float(relative_gain),
            "model_average_precision": float(best["average_precision"]),
            "model_top30_enrichment": float(best["top30_enrichment"]),
            "resolved_relative_brier_improvement": float(resolved_relative_gain),
            "bootstrap_brier_delta_mean": mean_delta, "bootstrap_brier_delta_ci_low": lo,
            "bootstrap_brier_delta_ci_high": hi, "qualifies_as_probability_utility": qualifies,
        })
    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "tc_window_model_decisions.csv", index=False)
    summary = {
        "target": "250 <= TC <= 350 K",
        "outer_validation": "leave-one-publication-out",
        "inner_selection": "publication-grouped CV minimizing Brier score",
        "qualification_rule": "at least 10% Brier improvement in main and resolved-formula cohorts, with cluster-bootstrap 95% CI for delta below zero",
        "decisions": decisions,
        "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Family-conditional 250–350 K Tc-window validation v1", ""]
    for d in decisions:
        lines += [
            f"## {d['cohort']}", "",
            f"Selected {d['selected_model']} with {d['selected_feature_set']} features. Brier score {d['model_brier']:.3f} versus {d['baseline_brier']:.3f} for {d['baseline_model']} ({100*d['relative_brier_improvement']:.1f}% relative improvement).",
            f"Average precision {d['model_average_precision']:.3f}; top-30% enrichment {d['model_top30_enrichment']:.2f}x. Resolved-formula Brier improvement {100*d['resolved_relative_brier_improvement']:.1f}%. Publication-cluster bootstrap delta 95% CI [{d['bootstrap_brier_delta_ci_low']:.3f}, {d['bootstrap_brier_delta_ci_high']:.3f}].",
            f"Decision: **{'qualifies' if d['qualifies_as_probability_utility'] else 'does not qualify'}** as a calibrated Tc-window utility.", "",
        ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
