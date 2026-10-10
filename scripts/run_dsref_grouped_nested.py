from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_full_199.csv"
OUT = ROOT / "results" / "dsref_grouped_nested_v3"

ELEMENTS = [
    "Gd", "Tb", "Dy", "Ho", "Sc", "Er", "Al", "Fe", "Co", "Ni", "La", "Y", "Pr", "Ag",
    "Tm", "Si", "V", "Cu", "Mn", "Ge", "H_element", "Cr", "Mo", "P", "B", "Ga", "Pd",
]
DESCRIPTORS = [
    "dX", "VEC", "sigma", "dHmix", "dSmix", "nunfilled_mean", "mag_moment_mean",
    "avg_d_valence_electrons", "bandgap_mean", "melting_point_mean", "electronegativity_range",
]


def model_specs(seed: int = 42):
    return {
        "ridge": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "extra_trees": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(
                    n_estimators=300, random_state=seed, n_jobs=1, max_depth=None,
                )),
            ]),
            {
                "model__max_features": [0.5, 1.0],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(
                    random_state=seed, learning_rate=0.05, max_iter=250,
                )),
            ]),
            {
                "model__max_leaf_nodes": [7, 15],
                "model__l2_regularization": [0.1, 1.0],
            },
        ),
    }


def metrics(y, pred, groups):
    group_mae = pd.DataFrame({"y": y, "pred": pred, "group": groups}).groupby("group").apply(
        lambda d: mean_absolute_error(d["y"], d["pred"]), include_groups=False
    )
    rho = spearmanr(y, pred).statistic
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "publication_macro_mae": float(group_mae.mean()),
        "spearman_rho": float(rho),
    }


def outer_splits(groups, seed=42):
    return list(GroupKFold(n_splits=7, shuffle=True, random_state=seed).split(np.zeros(len(groups)), groups=groups))


def nested_oof(X, y, groups, model_name, seed=42):
    estimator, grid = model_specs(seed)[model_name]
    prediction = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    fold_rows = []
    for fold, (train, test) in enumerate(outer_splits(groups, seed), start=1):
        inner = GroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
        search = GridSearchCV(
            clone(estimator), grid, scoring="neg_mean_absolute_error", cv=inner,
            n_jobs=-1, refit=True, error_score="raise",
        )
        search.fit(X.iloc[train], y[train], groups=groups[train])
        prediction[test] = search.predict(X.iloc[test])
        folds[test] = fold
        fm = metrics(y[test], prediction[test], groups[test])
        fold_rows.append({
            "fold": fold,
            "train_records": len(train),
            "test_records": len(test),
            "train_publications": int(pd.Series(groups[train]).nunique()),
            "test_publications": int(pd.Series(groups[test]).nunique()),
            "best_inner_mae": float(-search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            **fm,
        })
    assert np.isfinite(prediction).all() and (folds > 0).all()
    return prediction, folds, pd.DataFrame(fold_rows)


def dummy_oof(X, y, groups, seed=42):
    prediction = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (train, test) in enumerate(outer_splits(groups, seed), start=1):
        model = DummyRegressor(strategy="median").fit(X.iloc[train], y[train])
        prediction[test] = model.predict(X.iloc[test])
        folds[test] = fold
    return prediction, folds


def run_cohort(data, cohort_name, feature_sets, selected=None):
    y = data["dS_ref"].to_numpy(float)
    groups = data["reference_id"].astype(str).to_numpy()
    result_rows, pred_frames, fold_frames = [], [], []

    base_pred, base_folds = dummy_oof(data[[feature_sets["elements"][0]]], y, groups)
    result_rows.append({"cohort": cohort_name, "feature_set": "none", "model": "dummy_median", **metrics(y, base_pred, groups)})
    pred_frames.append(pd.DataFrame({
        "cohort": cohort_name, "feature_set": "none", "model": "dummy_median",
        "source_row": data["source_row"].to_numpy(), "reference_id": groups,
        "observed_dS_ref": y, "predicted_dS_ref": base_pred, "outer_fold": base_folds,
        "formula_mapping_status": data["formula_mapping_status"].to_numpy(),
    }))

    candidates = selected or [(f, m) for f in feature_sets for m in model_specs()]
    for feature_name, model_name in candidates:
        cols = feature_sets[feature_name]
        pred, folds, fold_table = nested_oof(data[cols], y, groups, model_name)
        result_rows.append({"cohort": cohort_name, "feature_set": feature_name, "model": model_name, **metrics(y, pred, groups)})
        pred_frames.append(pd.DataFrame({
            "cohort": cohort_name, "feature_set": feature_name, "model": model_name,
            "source_row": data["source_row"].to_numpy(), "reference_id": groups,
            "observed_dS_ref": y, "predicted_dS_ref": pred, "outer_fold": folds,
            "formula_mapping_status": data["formula_mapping_status"].to_numpy(),
        }))
        fold_table.insert(0, "model", model_name)
        fold_table.insert(0, "feature_set", feature_name)
        fold_table.insert(0, "cohort", cohort_name)
        fold_frames.append(fold_table)
    return pd.DataFrame(result_rows), pd.concat(pred_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    feature_sets = {
        "elements": ELEMENTS,
        "descriptors": DESCRIPTORS,
        "combined": ELEMENTS + DESCRIPTORS,
    }
    primary = data[data["dsref_primary_eligible"].astype(bool)].reset_index(drop=True)
    main_metrics, main_pred, main_folds = run_cohort(primary, "primary_H_ge_1T_123", feature_sets)
    eligible = main_metrics[main_metrics["model"] != "dummy_median"].sort_values(["mae", "publication_macro_mae"])
    best = eligible.iloc[0]
    selected = [(best["feature_set"], best["model"])]
    sensitivity_data = data[data["dsref_sensitivity_0p5T_eligible"].astype(bool)].reset_index(drop=True)
    sens_metrics, sens_pred, sens_folds = run_cohort(sensitivity_data, "sensitivity_H_ge_0p5T_136", feature_sets, selected=selected)
    strict_data = data[data["dsref_strict_0p5T_eligible"].astype(bool)].reset_index(drop=True)
    strict_metrics, strict_pred, strict_folds = run_cohort(strict_data, "strict_provenance_H_ge_0p5T_111", feature_sets, selected=selected)

    metrics_all = pd.concat([main_metrics, sens_metrics, strict_metrics], ignore_index=True)
    predictions = pd.concat([main_pred, sens_pred, strict_pred], ignore_index=True)
    folds = pd.concat([main_folds, sens_folds, strict_folds], ignore_index=True)
    metrics_all.to_csv(OUT / "nested_cv_metrics.csv", index=False)
    predictions.to_csv(OUT / "nested_cv_oof_predictions.csv", index=False)
    folds.to_csv(OUT / "nested_cv_fold_details.csv", index=False)

    winner = metrics_all[(metrics_all["cohort"] == "primary_H_ge_1T_123") & (metrics_all["feature_set"] == best["feature_set"]) & (metrics_all["model"] == best["model"])].iloc[0]
    sens = metrics_all[(metrics_all["cohort"] == "sensitivity_H_ge_0p5T_136") & (metrics_all["feature_set"] == best["feature_set"]) & (metrics_all["model"] == best["model"])].iloc[0]
    strict = metrics_all[(metrics_all["cohort"] == "strict_provenance_H_ge_0p5T_111") & (metrics_all["feature_set"] == best["feature_set"]) & (metrics_all["model"] == best["model"])].iloc[0]
    summary = {
        "outer_cv": "7-fold shuffled GroupKFold by publication",
        "inner_cv": "5-fold shuffled GroupKFold by publication",
        "selection_metric": "mean absolute error",
        "best_feature_set": str(best["feature_set"]),
        "best_model": str(best["model"]),
        "primary_H_ge_1T_123": {k: float(winner[k]) for k in ["r2", "rmse", "mae", "publication_macro_mae", "spearman_rho"]},
        "sensitivity_H_ge_0p5T_136": {k: float(sens[k]) for k in ["r2", "rmse", "mae", "publication_macro_mae", "spearman_rho"]},
        "strict_provenance_H_ge_0p5T_111": {k: float(strict[k]) for k in ["r2", "rmse", "mae", "publication_macro_mae", "spearman_rho"]},
        "github_write_performed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Publication-grouped dS_ref validation v2",
        "",
        f"Best configuration: **{best['model']}** with **{best['feature_set']}** features.",
        "",
        f"Primary H>=1 T cohort (123 records): R²={winner['r2']:.3f}, RMSE={winner['rmse']:.3f}, MAE={winner['mae']:.3f}, publication-macro MAE={winner['publication_macro_mae']:.3f}, Spearman ρ={winner['spearman_rho']:.3f}.",
        f"H>=0.5 T sensitivity (136 records): R²={sens['r2']:.3f}, RMSE={sens['rmse']:.3f}, MAE={sens['mae']:.3f}, publication-macro MAE={sens['publication_macro_mae']:.3f}, Spearman ρ={sens['spearman_rho']:.3f}.",
        f"Strict-provenance H>=0.5 T sensitivity (111 records): R²={strict['r2']:.3f}, RMSE={strict['rmse']:.3f}, MAE={strict['mae']:.3f}, publication-macro MAE={strict['publication_macro_mae']:.3f}, Spearman ρ={strict['spearman_rho']:.3f}.",
        "",
        "Publication groups are kept intact in both outer evaluation and inner hyperparameter selection. These results are model-selection evidence only; uncertainty calibration and the Nash utility are built in later stages.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
