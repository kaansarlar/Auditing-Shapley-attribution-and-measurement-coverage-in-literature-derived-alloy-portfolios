from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_strict_tc_72.csv"
OUT = ROOT / "results" / "tc_regression_family_validation_v3"
SEED = 42

ELEMENTS = [
    "Gd", "Tb", "Dy", "Ho", "Sc", "Er", "Al", "Fe", "Co", "Ni", "La", "Y", "Pr", "Ag",
    "Tm", "Si", "V", "Cu", "Mn", "Ge", "H_element", "Cr", "Mo", "P", "B", "Ga", "Pd",
]
DESCRIPTORS = [
    "dX", "VEC", "sigma", "dHmix", "dSmix", "nunfilled_mean", "mag_moment_mean",
    "avg_d_valence_electrons", "bandgap_mean", "melting_point_mean", "electronegativity_range",
]


def specs(seed=SEED):
    return {
        "ridge": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "svr_rbf": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {"model__C": [10.0, 100.0, 1000.0], "model__gamma": ["scale", 0.1], "model__epsilon": [10.0, 30.0]},
        ),
        "extra_trees": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=1))]),
            {"model__max_features": [0.5, 1.0], "model__min_samples_leaf": [2, 4]},
        ),
    }


def metrics(y, pred, groups):
    rho = spearmanr(y, pred).statistic
    frame = pd.DataFrame({"y": y, "pred": pred, "group": groups})
    group_mae = frame.groupby("group").apply(lambda d: mean_absolute_error(d["y"], d["pred"]), include_groups=False)
    window = (y >= 250) & (y <= 400)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse_K": float(mean_squared_error(y, pred) ** 0.5),
        "mae_K": float(mean_absolute_error(y, pred)),
        "publication_macro_mae_K": float(group_mae.mean()),
        "spearman_rho": float(rho),
        "range_250_400_mae_K": float(mean_absolute_error(y[window], pred[window])) if window.any() else np.nan,
    }


def split_count(data):
    pubs = data["reference_id"].nunique()
    return 7 if pubs >= 14 else (4 if pubs >= 8 else 3)


def nested(data, columns, model_name):
    X = data[columns]
    y = data["TC"].to_numpy(float)
    groups = data["reference_id"].astype(str).to_numpy()
    outer_n = split_count(data)
    outer = GroupKFold(n_splits=outer_n, shuffle=True, random_state=SEED)
    estimator, grid = specs()[model_name]
    pred = np.full(len(data), np.nan)
    folds = np.full(len(data), -1, dtype=int)
    fold_rows = []
    for fold, (train, test) in enumerate(outer.split(X, y, groups), start=1):
        train_groups = groups[train]
        inner_n = min(5, max(3, pd.Series(train_groups).nunique() // 2))
        inner = GroupKFold(n_splits=inner_n, shuffle=True, random_state=SEED + fold)
        search = GridSearchCV(clone(estimator), grid, scoring="neg_mean_absolute_error", cv=inner, n_jobs=-1, refit=True, error_score="raise")
        search.fit(X.iloc[train], y[train], groups=train_groups)
        pred[test] = search.predict(X.iloc[test])
        folds[test] = fold
        fold_rows.append({
            "fold": fold, "train_records": len(train), "test_records": len(test),
            "train_publications": int(pd.Series(groups[train]).nunique()),
            "test_publications": int(pd.Series(groups[test]).nunique()),
            "best_inner_mae_K": float(-search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })
    return pred, folds, pd.DataFrame(fold_rows)


def evaluate(data, cohort, feature_sets, selected=None):
    y = data["TC"].to_numpy(float)
    groups = data["reference_id"].astype(str).to_numpy()
    rows, pred_frames, fold_frames = [], [], []
    candidates = selected or [(f, m) for f in feature_sets for m in specs()]
    for feature_name, model_name in candidates:
        pred, folds, fold_table = nested(data, feature_sets[feature_name], model_name)
        rows.append({"cohort": cohort, "feature_set": feature_name, "model": model_name, **metrics(y, pred, groups)})
        pred_frames.append(pd.DataFrame({
            "cohort": cohort, "feature_set": feature_name, "model": model_name,
            "source_row": data["source_row"], "reference_id": data["reference_id"],
            "chemistry_family": data["chemistry_family"], "observed_TC_K": y,
            "predicted_TC_K": pred, "residual_K": y - pred, "outer_fold": folds,
            "formula_mapping_status": data["formula_mapping_status"],
        }))
        fold_table.insert(0, "model", model_name); fold_table.insert(0, "feature_set", feature_name); fold_table.insert(0, "cohort", cohort)
        fold_frames.append(fold_table)
    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    feature_sets = {"elements": ELEMENTS, "descriptors": DESCRIPTORS, "combined": ELEMENTS + DESCRIPTORS}
    cohorts = {
        "strict_72": data,
        "3d_TM_25": data[data["chemistry_family"].eq("3d-TM")].reset_index(drop=True),
        "TM_metalloid_22": data[data["chemistry_family"].eq("TM-metalloid")].reset_index(drop=True),
    }
    metric_parts, prediction_parts, fold_parts, decisions = [], [], [], []
    for name, cohort_data in cohorts.items():
        metric, pred, fold = evaluate(cohort_data, name, feature_sets)
        best = metric.sort_values(["mae_K", "spearman_rho"], ascending=[True, False]).iloc[0]
        resolved = cohort_data[cohort_data["formula_mapping_status"].eq("resolved")].reset_index(drop=True)
        sm, sp, sf = evaluate(resolved, name + "_resolved", feature_sets, selected=[(best["feature_set"], best["model"])])
        srow = sm.iloc[0]
        qualifies = bool(best["r2"] > 0 and best["spearman_rho"] >= 0.4 and srow["r2"] > 0 and srow["spearman_rho"] >= 0.4)
        decisions.append({
            "cohort": name, "feature_set": str(best["feature_set"]), "model": str(best["model"]),
            "r2": float(best["r2"]), "rmse_K": float(best["rmse_K"]), "mae_K": float(best["mae_K"]),
            "spearman_rho": float(best["spearman_rho"]), "range_250_400_mae_K": float(best["range_250_400_mae_K"]),
            "resolved_r2": float(srow["r2"]), "resolved_mae_K": float(srow["mae_K"]),
            "resolved_spearman_rho": float(srow["spearman_rho"]), "qualifies_for_soft_ranking_utility": qualifies,
        })
        metric_parts += [metric, sm]; prediction_parts += [pred, sp]; fold_parts += [fold, sf]
    metrics_all = pd.concat(metric_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)
    folds_all = pd.concat(fold_parts, ignore_index=True)
    metrics_all.to_csv(OUT / "tc_regression_metrics.csv", index=False)
    predictions_all.to_csv(OUT / "tc_regression_oof_predictions.csv", index=False)
    folds_all.to_csv(OUT / "tc_regression_fold_details.csv", index=False)
    pd.DataFrame(decisions).to_csv(OUT / "tc_regression_decisions.csv", index=False)
    summary = {
        "outer_validation": "publication-grouped shuffled GroupKFold; 7 folds global, 4 folds family cohorts",
        "inner_selection": "publication-grouped CV minimizing MAE",
        "qualification_rule": "R2 > 0 and Spearman rho >= 0.4 in both main and resolved-formula cohorts",
        "decisions": decisions, "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Publication-grouped Tc regression validation v1", ""]
    for d in decisions:
        lines += [
            f"## {d['cohort']}", "",
            f"Best: {d['model']} with {d['feature_set']} features. R²={d['r2']:.3f}, RMSE={d['rmse_K']:.1f} K, MAE={d['mae_K']:.1f} K, Spearman rho={d['spearman_rho']:.3f}, 250–400 K MAE={d['range_250_400_mae_K']:.1f} K.",
            f"Resolved-formula sensitivity: R²={d['resolved_r2']:.3f}, MAE={d['resolved_mae_K']:.1f} K, Spearman rho={d['resolved_spearman_rho']:.3f}.",
            f"Decision: **{'qualifies' if d['qualifies_for_soft_ranking_utility'] else 'does not qualify'}** for a soft Tc ranking utility.", "",
        ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
