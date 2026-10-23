# Auditing Shapley attribution and measurement coverage in literature-derived alloy portfolios

Data and reproducible analysis code by **Kağan Şarlar**, Karamanoğlu Mehmetbey University, Karaman, Türkiye. Contact: kagansarlar@kmu.edu.tr. Source version: **8.0.0** (23 October 2026).

The study separates choosing among measured alloys from identifying individual elemental contributions. Shapley allocations describe access to the defined candidate portfolio. Missing compositions do not provide evidence of zero physical performance, and bootstrap stability cannot supply missing counterfactuals.

## Repository contents

This repository tracks **data, source code, computational configuration, and documentation**. Numerical results, fitted predictions, figures, and manuscript files are generated locally and excluded by `.gitignore`. Previously committed results remain recoverable in Git history; history has not been rewritten.

| Path | Purpose |
| --- | --- |
| `data/master_data_v3/` | Audited literature measurements, eligibility flags, derived composition representations, and cohort metadata. |
| `inputs/data_v2/` | Preserved composition/audit input and publication-to-research-group mapping. |
| `inputs/master_full_199_block_policy.csv` | Preserved source-block representation for the audit. |
| `inputs/grid_3dTM.csv.gz` | Candidate compositions and five input descriptors; old predictions have been removed. This synthetic grid is not a set of validated feasible alloys. |
| `scripts/` | Empirical games, sensitivity analyses, predictive validation, completion bounds, and primary simulations. |
| `independent_generator_R_v6_1/code/` | Independent two-objective simulation generator and held-out diagnostics. |
| `extension/` | Portable portfolio audit and paired measurement-coverage simulation. |
| `tests/` | Checks of exact allocation, structural diagnostics, and numerical invariants. |
| `reproduce.py` | Dependency-ordered execution from a clean checkout. |

The curated data contain 199 records; the strict Curie cohort has 72 records, including 25 in the 3d transition-metal family. The measured game has 17 eligible strategies in four optional-element support classes. Original publication and source-row identifiers are retained for traceability. Rebuilding the curated tables is optional: `python scripts/build_master_data_v3.py` verifies the declared cohort counts before writing them.

## Installation and quick validation

Use **Python 3.12**, create a virtual environment, then install the pinned analysis dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py tests
```

The tests need no downloaded results or pytest installation. They check Shapley efficiency, symmetry, veto structure, full-coverage recovery, the measurement-allocation policies, and the empirical audit directly against curated data. They do not run the long simulation benchmarks.

## Reproduce the analyses

Run commands from the repository root. View the execution order without running it using `python reproduce.py all --list`.

```bash
# Empirical game, dependence checks, decision controls, completion bounds,
# temperature-window, proxy-target and utility sensitivities, and portable audit:
python reproduce.py empirical

# New paired measurement-allocation experiment (2,000 worlds per family):
python reproduce.py acquisition

# Original complete-design benchmarks (computationally expensive):
python reproduce.py primary independent summaries --jobs 2

# Additional predictive validations and the separate model-grid sensitivity:
python reproduce.py predictive model-grid --jobs 2

# All stages, including validation, in dependency order:
python reproduce.py all --jobs 2
```

Outputs are written under `results/` and `extension/results/`. Keep these directories locally; they are not inputs supplied by GitHub. The `summaries` stage requires the completed `primary` and `independent` stages. There are no automatic downloads of simulation results.

The portable audit also accepts another CSV with semicolon-separated player supports and nonnegative finite payoffs:

```bash
python extension/audit_portfolio.py --input results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv --output extension/results/empirical_audit
python extension/test_extension.py
```

### Simulation settings and reproduction limits

- **Original primary benchmark:** 4,000 worlds, seed 20261006, 500 within-world bootstrap resamples. The targeted run has 600 draws and seed 20262026. `run_synthetic_identifiability_benchmark_v6.py` writes general-world outputs to `results/primary_generator_v5/`; `run_hea_like_v6.py` writes targeted outputs to `results/synthetic_identifiability_benchmark_v6/`.
- **Independent benchmark:** general seeds 0–3999, targeted seeds 60000–68999, 100 within-world bootstrap resamples. `run_all.py` writes outputs to `results/robustness_generator_R/`. A small execution check is available with `--general 4 --targeted 4 --skip-classifiers`; these reduced settings do not reproduce the manuscript benchmark and overwrite those local output names.
- **Measurement-coverage extension:** seed 20261023, 2,000 worlds per family, 1,000 paired world bootstrap resamples, budgets 0/4/8/16/27, support-mean pooling with support-maximum sensitivity. Both extension families have a known empty-support utility of zero. This does not change the original independent generator's baseline.
- **Generators are analyzed separately.** Their sampling, baseline and confidence definitions differ. The original primary modal-frequency confidence threshold is 0.80; the independent full-data-top frequency threshold is 0.90. Preserve the implemented tie conventions when comparing archived runs.
- The random state of the older v6 fairness-summary percentile intervals was not included in the source archive. The new `run_benchmark_summaries.py` explicitly fixes seed 20261023 and 4,000 paired resamples. Its point estimates and Wilson intervals use the original definitions; its finite-resample percentile endpoints can differ from the archived summaries. The separate v7 exploratory equivalence script retains its original seed 20261008 and 4,000 resamples.
- Pinned versions are recorded in `requirements.txt` and `runtime_versions.json`. CPU/library differences may cause small floating-point or fitted-model differences. A run of the tests is not a claim that all 4,000/9,000-world benchmarks have been rerun.

## Interpretation

Completion bounds assume a bounded monotone game and preservation of the observed access envelope; completion witnesses do not establish chemical realizability. Support contrasts do not hold processing, fractions, phases or publication effects fixed. Candidate selection is evaluated under the declared utility transformations, not demonstrated refrigeration-device performance. Failure to detect a difference from random is not proof of equivalence; the equivalence-margin analysis is exploratory.

## Citation and archive

Cite the author, title, repository URL and the actual commit used; machine-readable metadata are in `CITATION.cff`. The original source release remains available at [10.5281/zenodo.23279674](https://doi.org/10.5281/zenodo.23279674); it predates the measurement-coverage extension. The [Zenodo version family](https://doi.org/10.5281/zenodo.23279673) provides version-specific archive records. A GitHub source update alone does not update the Zenodo files.

Related source audits and datasets:

1. Şarlar K (2026), *Machine learning screening of magnetocaloric high-entropy alloys using a field-normalized ΔSref target*, Journal of Magnetism and Magnetic Materials 650, 174120. [10.1016/j.jmmm.2026.174120](https://doi.org/10.1016/j.jmmm.2026.174120).
2. Şarlar K (2026), *Target auditing exposes chemistry-regime dependence in Curie-temperature models for magnetocaloric high-entropy alloys*. [10.5281/zenodo.23102726](https://doi.org/10.5281/zenodo.23102726).

Retain attribution to the original experimental publications identified in the data. The shared audit is a data source; the present study addresses portfolio-game attribution and measurement coverage.

## Funding

This work was supported by TÜBİTAK under the Career Development Program (3501), Project No. **125F375**.

## License

Original project code is released under the [MIT License](LICENSE). This software license does not relicense literature-derived measurements, source tables, third-party material or original publications. Preserve source attribution and consult the applicable source terms or contact the author where reuse permission is unclear.
