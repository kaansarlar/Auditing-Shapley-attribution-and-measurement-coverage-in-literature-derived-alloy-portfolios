from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "inputs" / "master_full_199_block_policy.csv"
FORMULA = ROOT / "inputs" / "data_v2" / "MLST-master-v2.csv"
OUT = ROOT / "data" / "master_data_v3"

ELEMENTS = [
    "Gd", "Tb", "Dy", "Ho", "Sc", "Er", "Al", "Fe", "Co", "Ni", "La", "Y", "Pr", "Ag",
    "Tm", "Si", "V", "Cu", "Mn", "Ge", "H_element", "Cr", "Mo", "P", "B", "Ga", "Pd",
]
DESCRIPTORS = [
    "VEC", "dX", "sigma", "dHmix", "dSmix", "nunfilled_mean", "mag_moment_mean",
    "avg_d_valence_electrons", "bandgap_mean", "melting_point_mean", "electronegativity_range",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    block = pd.read_csv(BLOCK).sort_values("source_row").reset_index(drop=True)
    formula = pd.read_csv(FORMULA).sort_values("source_row").reset_index(drop=True)
    if not block["source_row"].equals(formula["source_row"]):
        raise RuntimeError("Source-row alignment failed")

    v3 = block.copy()
    # Preserve the audited source block as a parallel representation before
    # making formula-derived fractions and descriptors the reproducible primary layer.
    for column in ELEMENTS + DESCRIPTORS:
        v3[f"{column}_source_block"] = v3[column]
        v3[column] = formula[column]

    # The formula-derived five descriptors are also the fields used by the grid game.
    v3["VEC_c"] = v3["VEC"]
    v3["sigma_c"] = v3["sigma"]
    v3["dHmix_c"] = v3["dHmix"]
    v3["dSmix_c"] = v3["dSmix"]
    v3["dX_c"] = v3["dX"]
    v3["chemistry_family"] = formula["chemistry_family"]
    v3["primary_composition_representation"] = formula["fraction_source_v2"]
    v3["composition_representation_policy"] = (
        "nominal formula fractions and recomputed descriptors are primary; "
        "audited source-block values are retained in *_source_block fields for sensitivity"
    )
    v3["reported_measured_composition_note"] = ""
    v3.loc[v3["source_row"].eq(127), "reported_measured_composition_note"] = (
        "Source reports approximately 13.5 at.% Al by EDS; nominal formula gives about 15.0 at.% Al; "
        "the retained source block gives about 7.9 at.% Al. Representation sensitivity is required."
    )

    # The audited target semantics from the version 2 source table remain authoritative.
    # In particular, strict_tc_eligible is already false for all seven S15 T_peak rows.
    strict = v3[v3["strict_tc_eligible"].fillna(False).astype(bool)].copy()
    phase = v3[v3["phase_extension_eligible"].fillna(False).astype(bool)].copy()
    unique = v3[v3["in_unique_141"].fillna(False).astype(bool)].copy()
    ds_primary = v3[v3["dsref_primary_eligible"].fillna(False).astype(bool)].copy()
    ds_05 = v3[v3["dsref_sensitivity_0p5T_eligible"].fillna(False).astype(bool)].copy()
    ds_strict = v3[v3["dsref_strict_0p5T_eligible"].fillna(False).astype(bool)].copy()

    expected = {
        "full": 199,
        "unique": 141,
        "strict_tc": 72,
        "strict_3d_tm": 25,
        "phase": 71,
        "dsref_primary": 123,
        "dsref_0p5T": 136,
        "dsref_strict_0p5T": 111,
    }
    observed = {
        "full": len(v3),
        "unique": len(unique),
        "strict_tc": len(strict),
        "strict_3d_tm": int(strict["chemistry_family"].eq("3d-TM").sum()),
        "phase": len(phase),
        "dsref_primary": len(ds_primary),
        "dsref_0p5T": len(ds_05),
        "dsref_strict_0p5T": len(ds_strict),
    }
    if observed != expected:
        raise RuntimeError(f"Cohort check failed: {observed}")

    outputs = {
        "master_full_199.csv": v3,
        "master_unique_141.csv": unique,
        "master_strict_tc_72.csv": strict,
        "master_phase_71.csv": phase,
        "master_dsref_primary_123.csv": ds_primary,
        "master_dsref_0p5T_136.csv": ds_05,
        "master_dsref_strict_0p5T_111.csv": ds_strict,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)

    representation = {
        "primary": "formula-derived fractions and recomputed descriptors for 188 rows; element-block fallback for 11 pseudo-binary rows",
        "sensitivity": "audited source-block fractions and descriptors retained in *_source_block fields",
        "confirmed_row_alignment_corrections": 21,
        "formula_block_discrepancies_gt_0p01": 78,
        "unconfirmed_after_alignment": 57,
        "S30_note": v3.loc[v3["source_row"].eq(127), "reported_measured_composition_note"].iloc[0],
    }
    (OUT / "cohort_summary.json").write_text(
        json.dumps({"counts": observed, "representation": representation}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"counts": observed, "representation": representation}, indent=2))


if __name__ == "__main__":
    main()
