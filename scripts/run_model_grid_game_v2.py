from __future__ import annotations

import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "master_data_v3" / "master_full_199.csv"
GRID = ROOT / "inputs" / "grid_3dTM.csv"
OUT = ROOT / "results" / "model_grid_game_v3"
FEATURES = ["VEC_c", "sigma_c", "dHmix_c", "dSmix_c", "dX_c"]
BACKGROUND = ["Fe", "Co"]
PLAYERS = ["Mn", "Cr", "Ni", "Al", "Cu"]
ALL = BACKGROUND + PLAYERS
TLO, THI = 250.0, 350.0


class GPUtility:
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 0):
        self.scaler = StandardScaler().fit(X)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        if self.std <= 1e-12:
            self.std = 1.0
        kernel = C(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e2)) + WhiteKernel(0.1, (1e-4, 1.0))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            normalize_y=False,
            random_state=seed,
        ).fit(self.scaler.transform(X), (y - self.mean) / self.std)

    def predict(self, X: np.ndarray):
        mean, std = self.model.predict(self.scaler.transform(X), return_std=True)
        return mean * self.std + self.mean, np.maximum(std * self.std, 1e-9)


def _composition_key(d: pd.DataFrame) -> pd.Series:
    return d[[f"x_{e}" for e in ALL]].round(8).astype(str).agg("|".join, axis=1)


def training_sets(data: pd.DataFrame):
    t = data[(data["strict_tc_eligible"] == True) & data["chemistry_family"].eq("3d-TM")].copy()
    t["composition_key"] = _composition_key(t)
    t = t.sort_values("source_row").groupby("composition_key", as_index=False).agg(
        **{f: (f, "first") for f in FEATURES},
        TC=("TC", "median"),
        reference_id=("reference_id", lambda x: ";".join(sorted(set(map(str, x))))),
        composition=("composition", "first"),
    )

    s = data[(data["dsref_sensitivity_0p5T_eligible"] == True) & data["chemistry_family"].eq("3d-TM")].copy()
    s["composition_key"] = _composition_key(s)
    s = s.sort_values("source_row").groupby("composition_key", as_index=False).agg(
        **{f: (f, "first") for f in FEATURES},
        dS_ref=("dS_ref", "median"),
        reference_id=("reference_id", lambda x: ";".join(sorted(set(map(str, x))))),
        composition=("composition", "first"),
    )
    return t, s


def coalition_values(grid: pd.DataFrame, q: np.ndarray):
    support = grid[PLAYERS].to_numpy() > 0
    values = {}
    for r in range(len(PLAYERS) + 1):
        for subset in combinations(range(len(PLAYERS)), r):
            allowed = np.zeros(len(PLAYERS), dtype=bool)
            allowed[list(subset)] = True
            mask = ~(support[:, ~allowed]).any(axis=1)
            values[subset] = float(q[mask].max()) if mask.any() else 0.0
    return values


def shapley(values):
    n = len(PLAYERS)
    phi = np.zeros(n)
    for subset, value in values.items():
        for i in range(n):
            if i in subset:
                continue
            joined = tuple(sorted(subset + (i,)))
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / math.factorial(n)
            phi[i] += weight * (values[joined] - value)
    return phi


def fit_game(t: pd.DataFrame, s: pd.DataFrame, grid: pd.DataFrame, seed: int = 0):
    gp_t = GPUtility(t[FEATURES].to_numpy(float), t["TC"].to_numpy(float), seed)
    mt, st = gp_t.predict(grid[FEATURES].to_numpy(float))
    u_t = norm.cdf((THI - mt) / st) - norm.cdf((TLO - mt) / st)

    log_s = np.log(np.maximum(s["dS_ref"].to_numpy(float), 1e-9))
    gp_s = GPUtility(s[FEATURES].to_numpy(float), log_s, seed + 17)
    ms, ss = gp_s.predict(grid[FEATURES].to_numpy(float))
    threshold_s = float(np.median(log_s))
    u_s = norm.cdf((ms - threshold_s) / ss)

    X_train = np.vstack([t[FEATURES].to_numpy(float), s[FEATURES].to_numpy(float)])
    scaler = StandardScaler().fit(X_train)
    z_train = scaler.transform(X_train)
    nn2 = NearestNeighbors(n_neighbors=2).fit(z_train)
    train_d = nn2.kneighbors(z_train)[0][:, 1]
    support_threshold = float(np.percentile(train_d, 95))
    d_nn = NearestNeighbors(n_neighbors=1).fit(z_train).kneighbors(
        scaler.transform(grid[FEATURES].to_numpy(float))
    )[0][:, 0]
    single_phase_heuristic = (grid["VEC_c"].to_numpy(float) < 6.87) | (grid["VEC_c"].to_numpy(float) >= 8.0)
    feasible = (d_nn <= support_threshold) & single_phase_heuristic

    disagreement_t = float(((t["TC"] >= TLO) & (t["TC"] <= THI)).mean())
    disagreement_s = 0.5
    q = np.clip(u_t - disagreement_t, 0, None) * np.clip(u_s - disagreement_s, 0, None)
    q = np.where(feasible, q, 0.0)
    idx = int(np.argmax(q))
    values = coalition_values(grid, q)
    phi = shapley(values)
    return {
        "idx": idx,
        "q": q,
        "uT": u_t,
        "uS": u_s,
        "mT": mt,
        "sT": st,
        "mS": np.exp(ms),
        "dNN": d_nn,
        "feasible": feasible,
        "support_threshold": support_threshold,
        "disagreement_T": disagreement_t,
        "disagreement_S": disagreement_s,
        "values": values,
        "shapley": phi,
    }


def composition_string(row):
    return "".join(f"{e}{row[e]:.2f}" for e in ALL if row[e] > 0)


def remove_reference(d: pd.DataFrame, reference: str):
    return d[~d["reference_id"].astype(str).str.split(";").map(lambda xs: reference in xs)].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    grid = pd.read_csv(GRID)[ALL + FEATURES].copy()
    t, s = training_sets(data)
    main_result = fit_game(t, s, grid, seed=41)
    i = main_result["idx"]

    result_grid = grid.copy()
    for name in ["q", "uT", "uS", "mT", "sT", "mS", "dNN", "feasible"]:
        result_grid[name] = main_result[name]
    result_grid.to_csv(OUT / "grid_results.csv.gz", index=False, compression="gzip")

    shapley_df = pd.DataFrame({"player": PLAYERS, "shapley": main_result["shapley"]})
    shapley_df["rank"] = shapley_df["shapley"].rank(method="min", ascending=False).astype(int)
    shapley_df.to_csv(OUT / "shapley.csv", index=False)

    full_rank = shapley_df.set_index("player")["rank"].reindex(PLAYERS).to_numpy()
    refs = sorted(set(";".join(t.reference_id.tolist() + s.reference_id.tolist()).split(";")))
    scenarios = [(f"leave_out_{r}", [r]) for r in refs]
    scenarios += [("leave_out_related_36_38", ["[36]", "[38]"])]
    stability = []
    for n, (label, removed) in enumerate(scenarios):
        tj, sj = t.copy(), s.copy()
        for r in removed:
            tj, sj = remove_reference(tj, r), remove_reference(sj, r)
        if len(tj) < 8 or len(sj) < 8:
            continue
        res = fit_game(tj, sj, grid, seed=100 + n)
        row = grid.iloc[res["idx"]]
        rank = pd.Series(res["shapley"], index=PLAYERS).rank(method="min", ascending=False).reindex(PLAYERS).to_numpy()
        stability.append({
            "scenario": label,
            "removed_references": ";".join(removed),
            "tc_states": len(tj),
            "ds_states": len(sj),
            "candidate": composition_string(row),
            "same_candidate_as_full": bool(res["idx"] == i),
            "nash_value": float(res["q"][res["idx"]]),
            "shapley_rank_kendall_tau": float(kendalltau(full_rank, rank).statistic),
            **{f"shapley_{p}": float(v) for p, v in zip(PLAYERS, res["shapley"])},
        })
    stability_df = pd.DataFrame(stability)
    stability_df.to_csv(OUT / "publication_stability.csv", index=False)

    row = grid.iloc[i]
    summary = {
        "role": "exploratory continuous-composition stress test, not a validated synthesis recommendation",
        "grid_size": int(len(grid)),
        "tc_training_states": int(len(t)),
        "dsref_training_states": int(len(s)),
        "candidate": composition_string(row),
        "candidate_fractions": {e: float(row[e]) for e in ALL},
        "nash_value": float(main_result["q"][i]),
        "uT": float(main_result["uT"][i]),
        "uS": float(main_result["uS"][i]),
        "predicted_TC_K": float(main_result["mT"][i]),
        "predicted_TC_sd_K": float(main_result["sT"][i]),
        "predicted_dSref": float(main_result["mS"][i]),
        "disagreement_T": float(main_result["disagreement_T"]),
        "disagreement_S": float(main_result["disagreement_S"]),
        "support_threshold": float(main_result["support_threshold"]),
        "shapley": {p: float(v) for p, v in zip(PLAYERS, main_result["shapley"])},
        "same_candidate_frequency_lopo": float(stability_df["same_candidate_as_full"].mean()),
        "related_cluster_36_38": stability_df[stability_df["scenario"].eq("leave_out_related_36_38")].to_dict("records"),
        "guardrail": "The Gaussian-process utilities are trained on small, publication-clustered cohorts and do not establish within-family transferability.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
