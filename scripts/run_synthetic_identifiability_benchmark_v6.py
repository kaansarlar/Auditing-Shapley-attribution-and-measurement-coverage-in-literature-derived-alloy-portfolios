from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "primary_generator_v5"
PLAYERS = ("Mn", "Cr", "Ni", "Al", "Cu")
N_PLAYERS = len(PLAYERS)
MASKS = np.arange(2**N_PLAYERS, dtype=int)
X_FULL = np.array([[(m >> j) & 1 for j in range(N_PLAYERS)] for m in MASKS], dtype=float)
PAIRS = tuple(combinations(range(N_PLAYERS), 2))
PAIR_X_FULL = np.array([[x[i] * x[j] for i, j in PAIRS] for x in X_FULL], dtype=float)
TOP_ATOL = 1e-10


def _precompute_shapley_terms():
    terms = {j: [] for j in range(N_PLAYERS)}
    for j in range(N_PLAYERS):
        bit = 1 << j
        for s in MASKS:
            if s & bit:
                continue
            r = int(s).bit_count()
            w = math.factorial(r) * math.factorial(N_PLAYERS - r - 1) / math.factorial(N_PLAYERS)
            terms[j].append((int(s), int(s | bit), w))
    return terms


SHAPLEY_TERMS = _precompute_shapley_terms()


def coalition_values(scores_by_support: np.ndarray) -> np.ndarray:
    values = np.zeros(2**N_PLAYERS, dtype=float)
    for coalition in MASKS:
        feasible = [s for s in MASKS if (s & ~coalition) == 0 and np.isfinite(scores_by_support[s])]
        values[coalition] = 0.0 if not feasible else float(np.nanmax(scores_by_support[feasible]))
    return values


def shapley(values: np.ndarray) -> np.ndarray:
    out = np.zeros(N_PLAYERS, dtype=float)
    for j in range(N_PLAYERS):
        out[j] = sum(w * (values[with_j] - values[without]) for without, with_j, w in SHAPLEY_TERMS[j])
    return out


def unique_top(values: np.ndarray, atol: float = TOP_ATOL) -> int:
    idx = np.flatnonzero(np.isclose(values, np.max(values), atol=atol, rtol=0.0))
    return int(idx[0]) if len(idx) == 1 else -1


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-14 or np.std(b) <= 1e-14:
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else 0.0


def generate_truth(rng: np.random.Generator, world: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main = rng.normal(0.25, 0.38, N_PLAYERS)
    interactions = np.zeros(len(PAIRS), dtype=float)
    if world == "interaction":
        active = rng.choice(len(PAIRS), size=rng.integers(2, 6), replace=False)
        interactions[active] = rng.normal(0.0, 0.65, len(active))
        if np.max(np.abs(interactions)) < 0.30:
            interactions[active[0]] = rng.choice([-1.0, 1.0]) * 0.55
    latent = -0.25 + X_FULL @ main + PAIR_X_FULL @ interactions
    score = 0.05 + 0.95 / (1.0 + np.exp(-latent))
    score[0] = 0.0
    return score, main, interactions


def support_diagnostics(supports: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    unique, counts = np.unique(supports, return_counts=True)
    x_unique = X_FULL[unique]
    x_records = X_FULL[supports]
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(32)) if len(unique) > 1 else 0.0
    veto_count = int(np.sum(np.all(x_unique == 1, axis=0)))
    equivalent_pairs = int(sum(
        np.array_equal(x_unique[:, i], x_unique[:, j]) and np.any(x_unique[:, i] > 0)
        for i, j in PAIRS
    ))
    observed = set(int(s) for s in unique)
    contrast_players = []
    contrast_pairs = 0
    for j in range(N_PLAYERS):
        has = False
        for s in unique:
            if ((int(s) >> j) & 1) and (int(s) & ~(1 << j)) in observed:
                has = True
                contrast_pairs += 1
        contrast_players.append(has)
    main_matrix = np.column_stack([np.ones(len(unique)), x_unique])
    pair_matrix = np.column_stack([main_matrix, PAIR_X_FULL[unique]])
    group_codes, group_counts = np.unique(groups, return_counts=True)
    shares = group_counts / group_counts.sum()
    modal = []
    for g in group_codes:
        _, gc = np.unique(supports[groups == g], return_counts=True)
        modal.append(gc.max() / gc.sum())
    us, si = np.unique(supports, return_inverse=True)
    ug, gi = np.unique(groups, return_inverse=True)
    joint = np.zeros((len(ug), len(us)), dtype=float)
    np.add.at(joint, (gi, si), 1.0)
    joint /= joint.sum()
    pg, ps = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / (pg[:, None] * ps[None, :])[nz])))
    hg = float(-np.sum(pg[pg > 0] * np.log(pg[pg > 0])))
    hs = float(-np.sum(ps[ps > 0] * np.log(ps[ps > 0])))
    nmi = 0.0 if min(hg, hs) <= 1e-12 else mi / math.sqrt(hg * hs)
    return {
        "n_records": int(len(supports)),
        "n_groups": int(len(group_codes)),
        "n_unique_supports": int(len(unique)),
        "support_coverage": float(len(unique) / 32),
        "support_entropy": entropy,
        "main_effect_rank": int(np.linalg.matrix_rank(main_matrix)),
        "pairwise_rank": int(np.linalg.matrix_rank(pair_matrix)),
        "veto_count": veto_count,
        "equivalent_pair_count": equivalent_pairs,
        "players_with_contrast": int(sum(contrast_players)),
        "single_element_contrast_pairs": int(contrast_pairs),
        "min_player_incidence": float(x_records.mean(axis=0).min()),
        "max_player_incidence": float(x_records.mean(axis=0).max()),
        "max_support_incidence": float(x_unique.mean(axis=0).max()),
        "effective_research_groups": float(1.0 / np.sum(shares**2)),
        "largest_group_share": float(shares.max()),
        "mean_group_modal_support_fraction": float(np.mean(modal)),
        "group_support_nmi": nmi,
    }


def _nested_publications(rng: np.random.Generator, groups: np.ndarray) -> np.ndarray:
    publications = np.empty(len(groups), dtype=int)
    next_pub = 0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        n_pub = min(int(rng.integers(1, 4)), len(idx))
        for chunk in np.array_split(idx, n_pub):
            publications[chunk] = next_pub
            next_pub += 1
    return publications


def sample_portfolio(rng: np.random.Generator, true_score: np.ndarray, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    n_records = int(overrides.get("n_records", rng.integers(20, 201)))
    n_groups = int(overrides.get("n_groups", rng.integers(3, min(13, n_records // 3 + 1))))
    target_unique = int(overrides.get("target_unique", rng.integers(4, 33)))
    cluster_strength = float(overrides.get("cluster_strength", rng.beta(1.4, 1.4)))
    noise_sd = float(overrides.get("noise_sd", rng.uniform(0.0, 0.18)))
    group_bias_sd = float(overrides.get("group_bias_sd", rng.uniform(0.0, 0.14)))
    publication_bias_sd = float(overrides.get("publication_bias_sd", rng.uniform(0.0, 0.10)))
    imposed_veto = bool(overrides.get("imposed_veto", rng.random() < 0.28))
    imposed_equivalence = bool(overrides.get("imposed_equivalence", rng.random() < 0.28))

    veto_player = overrides.get("veto_player")
    equivalent_pair = overrides.get("equivalent_pair")
    allowed = MASKS.copy()
    if imposed_veto:
        veto_player = int(rng.integers(N_PLAYERS)) if veto_player is None else int(veto_player)
        allowed = allowed[((allowed >> veto_player) & 1) == 1]
    if imposed_equivalence:
        if equivalent_pair is None:
            pool = [j for j in range(N_PLAYERS) if j != veto_player]
            equivalent_pair = tuple(int(x) for x in rng.choice(pool, 2, replace=False))
        a, b = equivalent_pair
        allowed = allowed[(((allowed >> a) & 1) == ((allowed >> b) & 1))]
    target_unique = min(target_unique, len(allowed), n_records)

    selection_strength = float(overrides.get("selection_strength", rng.uniform(0.0, 3.0)))
    logits = selection_strength * (true_score[allowed] - np.mean(true_score[allowed]))
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    chosen = rng.choice(allowed, target_unique, replace=False, p=weights)

    prototypes = rng.choice(chosen, n_groups, replace=True)
    group_sizes = np.full(n_groups, 3, dtype=int)
    for _ in range(n_records - group_sizes.sum()):
        group_sizes[int(rng.integers(n_groups))] += 1
    groups = np.repeat(np.arange(n_groups), group_sizes)
    rng.shuffle(groups)

    supports = np.empty(n_records, dtype=int)
    supports[:target_unique] = chosen
    rng.shuffle(supports[:target_unique])
    for idx in range(target_unique, n_records):
        g = groups[idx]
        distances = np.array([(int(s) ^ int(prototypes[g])).bit_count() for s in chosen], dtype=float)
        local = np.exp(-distances / 0.85)
        p = (1.0 - cluster_strength) * np.full(target_unique, 1.0 / target_unique)
        p += cluster_strength * local / local.sum()
        supports[idx] = int(rng.choice(chosen, p=p))
    perm = rng.permutation(n_records)
    supports, groups = supports[perm], groups[perm]
    publications = _nested_publications(rng, groups)
    n_publications = int(publications.max() + 1)

    group_bias = rng.normal(0.0, group_bias_sd, n_groups)
    publication_bias = rng.normal(0.0, publication_bias_sd, n_publications)
    observed = np.clip(
        true_score[supports] + group_bias[groups] + publication_bias[publications]
        + rng.normal(0.0, noise_sd, n_records), 0.0, 1.0
    )
    return {
        "supports": supports, "groups": groups, "publications": publications, "observed": observed,
        "noise_sd": noise_sd, "group_bias_sd": group_bias_sd, "publication_bias_sd": publication_bias_sd,
        "cluster_strength": cluster_strength, "selection_strength": selection_strength,
        "imposed_veto": imposed_veto, "imposed_equivalence": imposed_equivalence,
        "veto_player": veto_player, "equivalent_pair": equivalent_pair,
    }


def solve_observed(supports: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    score_by_support = np.full(32, np.nan)
    for support in np.unique(supports):
        score_by_support[support] = np.max(observed[supports == support])
    values = coalition_values(score_by_support)
    return values, shapley(values), int(np.nanargmax(score_by_support))


def unit_bootstrap(rng, supports, units_by_record, observed, n_boot):
    units = np.unique(units_by_record)
    phis = np.empty((n_boot, N_PLAYERS), dtype=float)
    tops = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        picked = rng.choice(units, len(units), replace=True)
        idx = np.concatenate([np.flatnonzero(units_by_record == u) for u in picked])
        _, phis[b], _ = solve_observed(supports[idx], observed[idx])
        tops[b] = unique_top(phis[b])
    return phis, tops


def cv_optimism(supports, groups, observed, seed):
    x = X_FULL[supports]
    model = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
    random_cv = KFold(n_splits=min(5, max(3, len(observed) // 10)), shuffle=True, random_state=seed)
    random_pred = np.empty(len(observed))
    for tr, te in random_cv.split(x):
        model.fit(x[tr], observed[tr]); random_pred[te] = model.predict(x[te])
    group_pred = np.empty(len(observed))
    group_cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in group_cv.split(x, observed, groups):
        model.fit(x[tr], observed[tr]); group_pred[te] = model.predict(x[te])
    return float(mean_squared_error(observed, random_pred) ** 0.5), float(mean_squared_error(observed, group_pred) ** 0.5)


def _bootstrap_summary(tops: np.ndarray, true_top: int) -> dict[str, float | int]:
    categories = np.arange(-1, N_PLAYERS)
    counts = {int(k): int(np.sum(tops == k)) for k in categories}
    modal = max(categories, key=lambda k: (counts[int(k)], -int(k)))
    frequency = counts[int(modal)] / len(tops)
    return {"modal": int(modal), "frequency": float(frequency), "tie_frequency": float(counts[-1] / len(tops)),
            "confident_wrong": int(modal >= 0 and frequency >= 0.80 and modal != true_top)}


def run_one_scenario(scenario: int, seed: int, n_boot: int = 500) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario]))
    world = "additive" if scenario % 2 == 0 else "interaction"
    true_score, _, interactions = generate_truth(rng, world)
    true_phi = shapley(coalition_values(true_score)); true_top = unique_top(true_phi)
    sampled = sample_portfolio(rng, true_score)
    supports, groups = sampled["supports"], sampled["groups"]
    publications, observed = sampled["publications"], sampled["observed"]
    _, estimated_phi, selected_support = solve_observed(supports, observed)
    _, reference_phi, reference_selected_support = solve_observed(supports, true_score[supports])
    estimated_top, reference_top = unique_top(estimated_phi), unique_top(reference_phi)
    estimated_tied = int(estimated_top < 0)
    unique_correct = np.nan if estimated_tied else float(estimated_top == true_top)
    unique_material_error = np.nan if estimated_tied else float(true_phi[estimated_top] < 0.5 * true_phi.max())
    identification_failure = int(estimated_tied or estimated_top != true_top)
    rho_total = safe_spearman(true_phi, estimated_phi)
    rho_estimation = safe_spearman(reference_phi, estimated_phi)
    rho_structure = safe_spearman(true_phi, reference_phi)
    true_scale = max(float(np.sum(np.abs(true_phi))), 1e-12)
    phi_l1 = float(np.sum(np.abs(estimated_phi - true_phi)) / true_scale)
    ordered = np.sort(estimated_phi); estimated_top_margin = float(ordered[-1] - ordered[-2])
    score_range = max(float(true_score.max() - true_score.min()), 1e-12)
    observed_oracle = float(np.max(true_score[np.unique(supports)]))
    regret_observed = float((observed_oracle - true_score[selected_support]) / score_range)
    regret_full = float((true_score.max() - true_score[selected_support]) / score_range)
    diag = support_diagnostics(supports, groups)
    group_phi, group_top = unit_bootstrap(rng, supports, groups, observed, n_boot)
    pub_phi, pub_top = unit_bootstrap(rng, supports, publications, observed, n_boot)
    group_boot, pub_boot = _bootstrap_summary(group_top, true_top), _bootstrap_summary(pub_top, true_top)
    group_width = np.quantile(group_phi, 0.975, axis=0) - np.quantile(group_phi, 0.025, axis=0)
    pub_width = np.quantile(pub_phi, 0.975, axis=0) - np.quantile(pub_phi, 0.025, axis=0)
    random_rmse, group_rmse = cv_optimism(supports, groups, observed, seed + scenario)
    return {
        "scenario": scenario, "world": world, **diag, "n_publications": int(len(np.unique(publications))),
        "noise_sd": sampled["noise_sd"], "group_bias_sd": sampled["group_bias_sd"],
        "publication_bias_sd": sampled["publication_bias_sd"], "cluster_strength": sampled["cluster_strength"],
        "selection_strength": sampled["selection_strength"], "imposed_veto": sampled["imposed_veto"],
        "imposed_equivalence": sampled["imposed_equivalence"], "true_interaction_l1": float(np.sum(np.abs(interactions))),
        "rank_spearman": rho_total, "rho_estimation": rho_estimation, "rho_structure": rho_structure,
        "shapley_l1_error": phi_l1, "estimated_top": estimated_top, "reference_top": reference_top,
        "true_top": true_top, "estimated_top_tied": estimated_tied,
        "top_player_correct_given_unique": unique_correct, "material_error_given_unique": unique_material_error,
        "identification_failure": identification_failure,
        "estimated_top_matches_reference": int(estimated_top >= 0 and estimated_top == reference_top),
        "estimated_top_margin": estimated_top_margin, "normalized_regret_observed_oracle": regret_observed,
        "normalized_regret_full_design": regret_full, "reference_selected_support": reference_selected_support,
        "high_rank_error": int(rho_total < 0.80),
        "group_bootstrap_modal_top": int(group_boot["modal"]),
        "group_bootstrap_top_frequency": float(group_boot["frequency"]),
        "group_bootstrap_tie_frequency": float(group_boot["tie_frequency"]),
        "group_bootstrap_mean_ci_width": float(np.mean(group_width)),
        "group_bootstrap_confident_wrong": int(group_boot["confident_wrong"]),
        "group_bootstrap_narrow_confident_wrong": int(group_boot["confident_wrong"] and np.mean(group_width) <= 0.10),
        "publication_bootstrap_modal_top": int(pub_boot["modal"]),
        "publication_bootstrap_top_frequency": float(pub_boot["frequency"]),
        "publication_bootstrap_tie_frequency": float(pub_boot["tie_frequency"]),
        "publication_bootstrap_mean_ci_width": float(np.mean(pub_width)),
        "publication_bootstrap_confident_wrong": int(pub_boot["confident_wrong"]),
        "publication_bootstrap_narrow_confident_wrong": int(pub_boot["confident_wrong"] and np.mean(pub_width) <= 0.10),
        "record_random_rmse": random_rmse, "grouped_rmse": group_rmse, "cv_optimism": group_rmse - random_rmse,
        **{f"true_phi_{PLAYERS[j]}": float(true_phi[j]) for j in range(N_PLAYERS)},
        **{f"reference_phi_{PLAYERS[j]}": float(reference_phi[j]) for j in range(N_PLAYERS)},
        **{f"estimated_phi_{PLAYERS[j]}": float(estimated_phi[j]) for j in range(N_PLAYERS)},
    }


def run_scenarios(n_scenarios=4000, seed=20261006, n_boot=500):
    return pd.DataFrame(Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_scenario)(scenario, seed, n_boot) for scenario in range(n_scenarios)))


def run_hea_like(n_worlds=600, seed=20262026, n_boot=500):
    def one(i):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        world = "additive" if i % 2 == 0 else "interaction"
        true_score, _, _ = generate_truth(rng, world)
        true_phi = shapley(coalition_values(true_score)); true_top = unique_top(true_phi)
        veto = int(rng.integers(N_PLAYERS))
        pair = tuple(int(x) for x in rng.choice([j for j in range(N_PLAYERS) if j != veto], 2, replace=False))
        sampled = sample_portfolio(rng, true_score, {"n_records": 20, "n_groups": 3, "target_unique": 4,
            "cluster_strength": 0.90, "imposed_veto": True, "veto_player": veto,
            "imposed_equivalence": True, "equivalent_pair": pair})
        supports, groups = sampled["supports"], sampled["groups"]
        pubs, observed = sampled["publications"], sampled["observed"]
        _, phi, selected = solve_observed(supports, observed); top = unique_top(phi)
        diag = support_diagnostics(supports, groups)
        _, gtops = unit_bootstrap(rng, supports, groups, observed, n_boot)
        _, ptops = unit_bootstrap(rng, supports, pubs, observed, n_boot)
        gb, pb = _bootstrap_summary(gtops, true_top), _bootstrap_summary(ptops, true_top)
        scale = max(true_score.max() - true_score.min(), 1e-12)
        return {"world_id": i, "world": world, **diag, "designated_veto": veto, "estimated_top": top,
            "true_top": true_top, "estimated_top_tied": int(top < 0),
            "estimated_top_is_veto_given_unique": np.nan if top < 0 else float(top == veto),
            "true_top_is_veto_all": float(true_top == veto),
            "true_top_is_veto_given_estimated_unique": np.nan if top < 0 else float(true_top == veto),
            "top_correct_given_unique": np.nan if top < 0 else float(top == true_top),
            "material_error_given_unique": np.nan if top < 0 else float(true_phi[top] < 0.5 * true_phi.max()),
            "group_bootstrap_modal_top": int(gb["modal"]),
            "group_bootstrap_top_frequency": float(gb["frequency"]),
            "group_bootstrap_confident": int(gb["modal"] >= 0 and gb["frequency"] >= 0.80),
            "group_bootstrap_confident_wrong": int(gb["confident_wrong"]),
            "publication_bootstrap_modal_top": int(pb["modal"]),
            "publication_bootstrap_top_frequency": float(pb["frequency"]),
            "publication_bootstrap_confident": int(pb["modal"] >= 0 and pb["frequency"] >= 0.80),
            "publication_bootstrap_confident_wrong": int(pb["confident_wrong"]),
            "group_bootstrap_tie_frequency": float(gb["tie_frequency"]),
            "publication_bootstrap_tie_frequency": float(pb["tie_frequency"]),
            "normalized_regret_observed_oracle": float((np.max(true_score[np.unique(supports)]) - true_score[selected]) / scale),
            "normalized_regret_full_design": float((true_score.max() - true_score[selected]) / scale)}
    return pd.DataFrame(Parallel(n_jobs=-1, verbose=5)(delayed(one)(i) for i in range(n_worlds)))


def sanity_check(seed=20269999, n_worlds=20):
    rows = []
    for i in range(n_worlds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        world = "additive" if i % 2 == 0 else "interaction"
        score, _, _ = generate_truth(rng, world); true_phi = shapley(coalition_values(score))
        _, estimated_phi, selected = solve_observed(MASKS, score)
        rows.append({"world_id": i, "world": world, "rho": safe_spearman(true_phi, estimated_phi),
            "max_abs_shapley_error": float(np.max(np.abs(true_phi - estimated_phi))),
            "selected_support_correct": int(selected == int(np.argmax(score)))})
    return pd.DataFrame(rows)


DIAGNOSTICS = ["n_records", "n_groups", "n_publications", "n_unique_supports", "support_entropy",
    "main_effect_rank", "pairwise_rank", "veto_count", "equivalent_pair_count", "players_with_contrast",
    "max_player_incidence", "max_support_incidence", "effective_research_groups", "largest_group_share",
    "mean_group_modal_support_fraction", "group_support_nmi", "group_bootstrap_top_frequency",
    "group_bootstrap_tie_frequency", "group_bootstrap_mean_ci_width", "estimated_top_margin"]


def fit_diagnostic_models(df, seed=20261006):
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=seed,
        stratify=df["identification_failure"])
    x_train, x_test = df.loc[train_idx, DIAGNOSTICS], df.loc[test_idx, DIAGNOSTICS]
    y_train, y_test = df.loc[train_idx, "identification_failure"], df.loc[test_idx, "identification_failure"]
    classifier = RandomForestClassifier(n_estimators=500, min_samples_leaf=12, class_weight="balanced",
        random_state=seed, n_jobs=-1).fit(x_train, y_train)
    prob = classifier.predict_proba(x_test)[:, 1]; pred = (prob >= 0.5).astype(int)
    regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, random_state=seed + 1,
        n_jobs=-1).fit(x_train, df.loc[train_idx, "shapley_l1_error"])
    reg_pred = regressor.predict(x_test)
    importance = permutation_importance(classifier, x_test, y_test, n_repeats=20, random_state=seed,
        scoring="roc_auc", n_jobs=-1)
    importance_df = pd.DataFrame({"diagnostic": DIAGNOSTICS,
        "permutation_importance_mean": importance.importances_mean,
        "permutation_importance_sd": importance.importances_std}).sort_values("permutation_importance_mean", ascending=False)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=75, class_weight="balanced", random_state=seed).fit(x_train, y_train)
    tree_pred = tree.predict(x_test); tree_prob = tree.predict_proba(x_test)[:, 1]
    metrics = {"train_scenarios": int(len(train_idx)), "test_scenarios": int(len(test_idx)),
        "classification_target": "No unique estimated top or unique estimated top differs from complete-design truth.",
        "rf_classifier_test_auc": float(roc_auc_score(y_test, prob)),
        "rf_classifier_test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "rf_regressor_test_rmse": float(mean_squared_error(df.loc[test_idx, "shapley_l1_error"], reg_pred) ** 0.5),
        "tree_test_auc": float(roc_auc_score(y_test, tree_prob)),
        "tree_test_balanced_accuracy": float(balanced_accuracy_score(y_test, tree_pred))}
    predictions = df.loc[test_idx, ["scenario", "world", "identification_failure", "estimated_top_tied",
        "high_rank_error", "shapley_l1_error"]].copy()
    predictions["rf_identification_failure_probability"] = prob
    predictions["rf_predicted_l1_error"] = reg_pred
    predictions["tree_identification_failure_probability"] = tree_prob
    return metrics, importance_df, export_text(tree, feature_names=DIAGNOSTICS, decimals=3), predictions


def structural_class(df):
    return np.select([(df.veto_count > 0) & (df.equivalent_pair_count > 0), df.veto_count > 0,
        df.equivalent_pair_count > 0], ["veto + equivalent", "veto only", "equivalent only"], default="neither")


def summary_tables(df):
    work = df.assign(structural_class=structural_class(df))
    by_structure = work.groupby("structural_class", observed=True).agg(scenarios=("scenario", "size"),
        median_rho_total=("rank_spearman", "median"), median_rho_estimation=("rho_estimation", "median"),
        median_rho_structure=("rho_structure", "median"), tied_top_rate=("estimated_top_tied", "mean"),
        unique_top_accuracy=("top_player_correct_given_unique", "mean"),
        unique_material_error=("material_error_given_unique", "mean"),
        identification_failure_rate=("identification_failure", "mean"),
        group_confident_wrong_rate=("group_bootstrap_confident_wrong", "mean"),
        publication_confident_wrong_rate=("publication_bootstrap_confident_wrong", "mean"),
        mean_regret_observed=("normalized_regret_observed_oracle", "mean"),
        mean_regret_full=("normalized_regret_full_design", "mean")).reset_index()
    by_rank = work.groupby("main_effect_rank", observed=True).agg(scenarios=("scenario", "size"),
        median_rho_estimation=("rho_estimation", "median"), median_rho_structure=("rho_structure", "median"),
        tied_top_rate=("estimated_top_tied", "mean"), unique_top_accuracy=("top_player_correct_given_unique", "mean"),
        identification_failure_rate=("identification_failure", "mean"),
        mean_regret_observed=("normalized_regret_observed_oracle", "mean"),
        mean_regret_full=("normalized_regret_full_design", "mean")).reset_index()
    return by_structure, by_rank


def make_figure(df, importance):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    work = df.assign(structural_class=structural_class(df))
    ax = axes[0, 0]
    rank = df.groupby("main_effect_rank")[["rho_estimation", "rho_structure"]].median()
    ax.plot(rank.index, rank.rho_estimation, "o-", lw=2, label="Estimation: noisy vs same-support truth")
    ax.plot(rank.index, rank.rho_structure, "s-", lw=2, label="Structure: same-support vs full truth")
    ax.set(xlabel="Main-effect rank of observed support matrix", ylabel="Median Shapley rank Spearman ρ",
        ylim=(-0.05, 1.03), title="a  Structural loss dominates estimation noise"); ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    order = ["neither", "equivalent only", "veto only", "veto + equivalent"]
    grouped = work.groupby("structural_class").agg(tied=("estimated_top_tied", "mean"),
        wrong_unique=("top_player_correct_given_unique", lambda x: 1.0 - x.mean())).reindex(order)
    x = np.arange(len(order)); width = 0.36
    ax.bar(x - width / 2, grouped.tied, width, label="No unique top", color="#9CA3AF")
    ax.bar(x + width / 2, grouped.wrong_unique, width, label="Wrong | unique top", color="#F97316")
    ax.set_xticks(x, [s.replace(" + ", " +\n") for s in order], rotation=15)
    ax.set(ylabel="Fraction of worlds", ylim=(0, 1), title="b  Ties and conditional error are distinct"); ax.legend(frameon=False)
    ax = axes[1, 0]
    top = importance.head(7).sort_values("permutation_importance_mean")
    ax.barh(top.diagnostic.str.replace("_", " "), top.permutation_importance_mean,
        xerr=top.permutation_importance_sd, color="#0F766E", alpha=0.9)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set(xlabel="Held-out permutation importance (AUC loss)", title="c  Diagnostics anticipate identification failure")
    ax = axes[1, 1]
    bins = pd.cut(df.n_unique_supports, bins=[3, 7, 11, 15, 23, 32], include_lowest=True)
    regret = df.groupby(bins, observed=True)[["normalized_regret_observed_oracle", "normalized_regret_full_design"]].mean()
    xx = np.arange(len(regret))
    ax.plot(xx, regret.normalized_regret_observed_oracle, "o-", lw=2, label="Observed-portfolio oracle")
    ax.plot(xx, regret.normalized_regret_full_design, "s-", lw=2, label="Full-design oracle")
    ax.set_xticks(xx, [str(v) for v in regret.index], rotation=25)
    ax.set(xlabel="Observed support classes", ylabel="Mean normalized decision regret",
        ylim=(0, max(0.05, regret.normalized_regret_full_design.max() * 1.15)),
        title="d  Candidate ranking survives better than attribution"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "synthetic_benchmark_figure_v5.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "synthetic_benchmark_figure_v5.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = run_scenarios(); df.to_csv(OUT / "scenario_results_v5.csv.gz", index=False, compression="gzip")
    by_structure, by_rank = summary_tables(df)
    by_structure.to_csv(OUT / "summary_by_structure_v5.csv", index=False)
    by_rank.to_csv(OUT / "summary_by_rank_v5.csv", index=False)
    metrics, importance, tree_rules, predictions = fit_diagnostic_models(df)
    importance.to_csv(OUT / "diagnostic_importance_v5.csv", index=False)
    predictions.to_csv(OUT / "heldout_predictions_v5.csv", index=False)
    (OUT / "diagnostic_tree_rules_v5.txt").write_text(tree_rules, encoding="utf-8")
    make_figure(df, importance)
    hea = run_hea_like(); hea.to_csv(OUT / "hea_like_worlds_v5.csv", index=False)
    sanity = sanity_check(); sanity.to_csv(OUT / "full_coverage_sanity_v5.csv", index=False)
    unique, hea_unique = df[df.estimated_top_tied == 0], hea[hea.estimated_top_tied == 0]
    overall = {"seed": 20261006, "scenarios": int(len(df)), "bootstrap_resamples_per_unit": 500,
        "truth_definition": "Exact Shapley value of the noise-free complete 2^5 support design.",
        "reference_definition": "Exact Shapley value of the noise-free game restricted to observed supports.",
        "estimated_definition": "Exact Shapley value of the noisy publication- and group-clustered portfolio.",
        "median_rho_total": float(df.rank_spearman.median()), "median_rho_estimation": float(df.rho_estimation.median()),
        "median_rho_structure": float(df.rho_structure.median()), "estimated_top_tie_rate": float(df.estimated_top_tied.mean()),
        "unique_top_accuracy": float(unique.top_player_correct_given_unique.mean()),
        "unique_top_material_error": float(unique.material_error_given_unique.mean()),
        "identification_failure_rate": float(df.identification_failure.mean()),
        "group_bootstrap_confident_wrong_rate": float(df.group_bootstrap_confident_wrong.mean()),
        "group_bootstrap_narrow_confident_wrong_rate": float(df.group_bootstrap_narrow_confident_wrong.mean()),
        "publication_bootstrap_confident_wrong_rate": float(df.publication_bootstrap_confident_wrong.mean()),
        "publication_bootstrap_narrow_confident_wrong_rate": float(df.publication_bootstrap_narrow_confident_wrong.mean()),
        "mean_normalized_regret_observed_oracle": float(df.normalized_regret_observed_oracle.mean()),
        "mean_normalized_regret_full_design": float(df.normalized_regret_full_design.mean()),
        "diagnostic_models": metrics,
        "hea_like": {"worlds": int(len(hea)), "unique_estimated_top_worlds": int(len(hea_unique)),
            "estimated_top_tie_rate": float(hea.estimated_top_tied.mean()),
            "estimated_top_is_veto_given_unique": float(hea_unique.estimated_top_is_veto_given_unique.mean()),
            "true_top_is_veto_all_worlds": float(hea.true_top_is_veto_all.mean()),
            "true_top_is_veto_given_estimated_unique": float(hea_unique.true_top_is_veto_given_estimated_unique.mean()),
            "top_accuracy_given_unique": float(hea_unique.top_correct_given_unique.mean()),
            "material_error_given_unique": float(hea_unique.material_error_given_unique.mean()),
            "group_bootstrap_confident_wrong_rate": float(hea.group_bootstrap_confident_wrong.mean()),
            "publication_bootstrap_confident_wrong_rate": float(hea.publication_bootstrap_confident_wrong.mean()),
            "mean_regret_observed_oracle": float(hea.normalized_regret_observed_oracle.mean()),
            "mean_regret_full_design": float(hea.normalized_regret_full_design.mean())},
        "sanity": {"worlds": int(len(sanity)), "all_rho_one": bool(np.allclose(sanity.rho, 1.0)),
            "max_abs_shapley_error": float(sanity.max_abs_shapley_error.max()),
            "all_selected_supports_correct": bool(sanity.selected_support_correct.all())}}
    (OUT / "summary_v5.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
