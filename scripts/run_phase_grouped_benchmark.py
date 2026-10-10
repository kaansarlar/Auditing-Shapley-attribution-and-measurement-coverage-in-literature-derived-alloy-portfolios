from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_phase_71.csv"
OUT = ROOT / "results" / "phase_grouped_benchmark_v3"

ELEMENTS = [
    "Gd", "Tb", "Dy", "Ho", "Sc", "Er", "Al", "Fe", "Co", "Ni", "La", "Y", "Pr", "Ag",
    "Tm", "Si", "V", "Cu", "Mn", "Ge", "H_element", "Cr", "Mo", "P", "B", "Ga", "Pd",
]
DESCRIPTORS = [
    "dX", "VEC", "sigma", "dHmix", "dSmix", "nunfilled_mean", "mag_moment_mean",
    "avg_d_valence_electrons", "bandgap_mean", "melting_point_mean", "electronegativity_range",
]


def model_specs(seed=42):
    return {
        "multinomial_logistic": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "extra_trees": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=1)),
            ]),
            {"model__max_features": [0.5, 1.0], "model__min_samples_leaf": [2, 4]},
        ),
    }


def metrics(y, pred, groups):
    frame = pd.DataFrame({"y": y, "pred": pred, "group": groups})
    pub_acc = frame.groupby("group").apply(lambda d: accuracy_score(d["y"], d["pred"]), include_groups=False)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "publication_macro_accuracy": float(pub_acc.mean()),
    }


def splits(groups, seed=42):
    return list(GroupKFold(n_splits=6, shuffle=True, random_state=seed).split(np.zeros(len(groups)), groups=groups))


def baseline_oof(data, mode, seed=42):
    y = data["phase_class_collapsed"].astype(str).to_numpy()
    groups = data["reference_id"].astype(str).to_numpy()
    pred = np.empty(len(y), dtype=object)
    folds = np.full(len(y), -1)
    for fold, (train, test) in enumerate(splits(groups, seed), start=1):
        global_class = pd.Series(y[train]).value_counts().index[0]
        if mode == "majority":
            pred[test] = global_class
        else:
            train_frame = pd.DataFrame({"family": data.iloc[train]["chemistry_family"].to_numpy(), "y": y[train]})
            family_map = train_frame.groupby("family")["y"].agg(lambda s: s.value_counts().index[0]).to_dict()
            pred[test] = [family_map.get(f, global_class) for f in data.iloc[test]["chemistry_family"]]
        folds[test] = fold
    return pred, folds.astype(int)


def nested_oof(data, cols, model_name, seed=42):
    X = data[cols]
    y = data["phase_class_collapsed"].astype(str).to_numpy()
    groups = data["reference_id"].astype(str).to_numpy()
    estimator, grid = model_specs(seed)[model_name]
    pred = np.empty(len(y), dtype=object)
    folds = np.full(len(y), -1)
    fold_rows = []
    for fold, (train, test) in enumerate(splits(groups, seed), start=1):
        inner = GroupKFold(n_splits=4, shuffle=True, random_state=seed + fold)
        search = GridSearchCV(clone(estimator), grid, scoring="f1_macro", cv=inner, n_jobs=-1, refit=True, error_score="raise")
        search.fit(X.iloc[train], y[train], groups=groups[train])
        pred[test] = search.predict(X.iloc[test])
        folds[test] = fold
        fold_rows.append({
            "fold": fold,
            "train_records": len(train), "test_records": len(test),
            "train_publications": int(pd.Series(groups[train]).nunique()),
            "test_publications": int(pd.Series(groups[test]).nunique()),
            "best_inner_macro_f1": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            **metrics(y[test], pred[test], groups[test]),
        })
    return pred, folds.astype(int), pd.DataFrame(fold_rows)


def evaluate(data, cohort, feature_sets, selected=None):
    y = data["phase_class_collapsed"].astype(str).to_numpy()
    groups = data["reference_id"].astype(str).to_numpy()
    results, preds, folds_out = [], [], []
    for baseline in ["majority", "family"]:
        pred, folds = baseline_oof(data, baseline)
        results.append({"cohort": cohort, "feature_set": "baseline", "model": baseline, **metrics(y, pred, groups)})
        preds.append(pd.DataFrame({
            "cohort": cohort, "feature_set": "baseline", "model": baseline,
            "source_row": data["source_row"], "reference_id": groups, "observed_phase": y,
            "predicted_phase": pred, "outer_fold": folds, "formula_mapping_status": data["formula_mapping_status"],
        }))
    candidates = selected or [(f, m) for f in feature_sets for m in model_specs()]
    for feature_name, model_name in candidates:
        pred, folds, fold_table = nested_oof(data, feature_sets[feature_name], model_name)
        results.append({"cohort": cohort, "feature_set": feature_name, "model": model_name, **metrics(y, pred, groups)})
        preds.append(pd.DataFrame({
            "cohort": cohort, "feature_set": feature_name, "model": model_name,
            "source_row": data["source_row"], "reference_id": groups, "observed_phase": y,
            "predicted_phase": pred, "outer_fold": folds, "formula_mapping_status": data["formula_mapping_status"],
        }))
        fold_table.insert(0, "model", model_name)
        fold_table.insert(0, "feature_set", feature_name)
        fold_table.insert(0, "cohort", cohort)
        folds_out.append(fold_table)
    return pd.DataFrame(results), pd.concat(preds, ignore_index=True), pd.concat(folds_out, ignore_index=True) if folds_out else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    feature_sets = {"elements": ELEMENTS, "descriptors": DESCRIPTORS, "combined": ELEMENTS + DESCRIPTORS}
    all_metrics, all_pred, all_folds = evaluate(data, "phase_71", feature_sets)
    learned = all_metrics[~all_metrics["feature_set"].eq("baseline")].sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    best = learned.iloc[0]
    sens_data = data[data["formula_mapping_status"].eq("resolved")].reset_index(drop=True)
    sens_metrics, sens_pred, sens_folds = evaluate(sens_data, "resolved_formula", feature_sets, selected=[(best["feature_set"], best["model"])])
    metrics_all = pd.concat([all_metrics, sens_metrics], ignore_index=True)
    metrics_all.to_csv(OUT / "phase_metrics.csv", index=False)
    pd.concat([all_pred, sens_pred], ignore_index=True).to_csv(OUT / "phase_oof_predictions.csv", index=False)
    pd.concat([all_folds, sens_folds], ignore_index=True).to_csv(OUT / "phase_fold_details.csv", index=False)

    baseline_best = all_metrics[all_metrics["feature_set"].eq("baseline")].sort_values("macro_f1", ascending=False).iloc[0]
    gain = float(best["macro_f1"] - baseline_best["macro_f1"])
    sens_best = sens_metrics[~sens_metrics["feature_set"].eq("baseline")].iloc[0]
    sens_baseline = sens_metrics[sens_metrics["feature_set"].eq("baseline")].sort_values("macro_f1", ascending=False).iloc[0]
    sensitivity_gain = float(sens_best["macro_f1"] - sens_baseline["macro_f1"])
    robust_gain = min(gain, sensitivity_gain)
    go = robust_gain >= 0.10
    summary = {
        "records": len(data), "publication_groups": int(data["reference_id"].nunique()),
        "class_counts": data["phase_class_collapsed"].value_counts().to_dict(),
        "best_feature_set": str(best["feature_set"]), "best_model": str(best["model"]),
        "best_macro_f1": float(best["macro_f1"]), "best_balanced_accuracy": float(best["balanced_accuracy"]),
        "best_baseline": str(baseline_best["model"]), "baseline_macro_f1": float(baseline_best["macro_f1"]),
        "macro_f1_gain": gain,
        "resolved_formula_macro_f1": float(sens_best["macro_f1"]),
        "resolved_formula_baseline_macro_f1": float(sens_baseline["macro_f1"]),
        "resolved_formula_gain": sensitivity_gain,
        "robust_minimum_gain": robust_gain,
        "go_no_go_threshold": 0.10,
        "phase_as_main_utility": bool(go),
        "recommended_treatment": "main utility" if go else "supplementary feasibility flag",
        "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Phase prediction go/no-go benchmark v1

The strict phase cohort contains {len(data)} records from {data['reference_id'].nunique()} publications. The best learned model is **{best['model']}** with **{best['feature_set']}** features (macro-F1 {best['macro_f1']:.3f}; balanced accuracy {best['balanced_accuracy']:.3f}). The strongest baseline is **{baseline_best['model']}** (macro-F1 {baseline_best['macro_f1']:.3f}), giving a macro-F1 gain of {gain:.3f}.

Resolved-formula sensitivity gives macro-F1 {sens_best['macro_f1']:.3f} versus {sens_baseline['macro_f1']:.3f} for the family baseline, a gain of {sensitivity_gain:.3f}. The robust minimum gain is therefore {robust_gain:.3f}.

Pre-specified decision: **{'GO' if go else 'NO-GO'}** for phase as a main Nash utility. Recommended treatment: **{summary['recommended_treatment']}**.

All outer and inner splits keep publication groups intact. The three-class endpoint is single-phase crystalline, amorphous-containing, and multiphase/other.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
