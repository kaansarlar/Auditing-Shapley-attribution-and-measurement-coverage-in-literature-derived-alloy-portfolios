# Auditing Shapley attribution and measurement coverage in literature-derived alloy portfolios

Data and analysis code for assessing game-theoretic candidate selection and elemental attribution in literature-derived magnetocaloric alloy portfolios.

**Author:** Kağan Şarlar, Karamanoğlu Mehmetbey University, Karaman, Türkiye  
**Contact:** kagansarlar@kmu.edu.tr  
**Version:** v7 final

## Purpose

The analyses distinguish selection among measured alloys from identification of individual elemental contributions. They combine audited experimental records with complete-design synthetic benchmarks to separate measurement error, missing composition combinations, and publication or research-group dependence.

The eligible experimental portfolio contains 17 strategies in four optional-element support classes. Cr is present throughout, while Mn and Al always occur together. These constraints affect Shapley allocations and limit their chemical interpretation.

## Data and code layout

Place the folders directly in the repository root, alongside `README.md` and `LICENSE`. Retain the existing folder names because the analysis scripts use these relative paths.

| Path | Contents |
| --- | --- |
| `data/` | Audited and processed experimental data. |
| `inputs/` | Source inputs, audit flags, and composition-representation layers. |
| `scripts/` | Data processing, predictive validation, Nash–Shapley analyses, synthetic benchmarks, and sensitivity analyses. |
| `independent_generator_R_v6_1/code/` | Corrected independent simulation generator and analysis code. |
| `independent_generator_R_v6_1/results/` | Independent benchmark and targeted simulation data. |
| `runtime_versions.json` | Dependency versions used in the validated environment. |

## Installation

Use **Python 3.12** and install:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

Use a virtual environment where possible. Consult `runtime_versions.json` for the validated dependency versions.

## Run the empirical analysis

From the repository root:

```bash
python scripts/run_empirical_nash_shapley.py
```

This script reads the processed experimental data and creates its numerical outputs under `results/`. The root `results/` directory is generated locally and is not included in this repository's upload scope.

## Other analysis scripts

| Script | Analysis |
| --- | --- |
| `scripts/build_master_data_v3.py` | Constructs the audited master data from preserved inputs. |
| `scripts/run_empirical_nash_shapley.py` | Computes the measured strategy game and elemental allocations. |
| `scripts/run_decision_validation_v3.py` | Examines publication/group dependence and decision sensitivity. |
| `scripts/run_temperature_window_sensitivity_v3.py` | Tests changes in the target temperature window. |
| `scripts/run_s15_proxy_counterfactual_v3_2.py` | Examines the structural consequences of restoring excluded proxy-temperature records. |
| `scripts/run_synthetic_identifiability_benchmark_v6.py` | Runs the primary complete-design synthetic benchmark. |
| `scripts/run_hea_like_v6.py` | Runs the targeted primary HEA-like stress test. |
| `independent_generator_R_v6_1/code/bench.py` | Runs the corrected independent benchmark. |
| `independent_generator_R_v6_1/code/hea_match.py` | Generates the independent HEA-targeted worlds. |

Inspect each script's entry point and input/output paths before a full rerun. Independent-generator definitions and execution settings are available in its code.

## Reproduction scope

This repository includes experimental data, source inputs, analysis scripts, runtime information, and the corrected independent generator with its archived benchmark and targeted simulation data. The main `results/` directory is excluded.

The independent archived data cover the 4,000-world benchmark and 9,000-world targeted run. Primary synthetic simulations can be rerun using the corresponding scripts; their outputs are generated locally. The primary and independent generators use different sampling mechanisms and confidence definitions, so their rates should be reported separately.

Some additional analyses require previously generated files under the main `results/` directory:

| Script | Required local outputs |
| --- | --- |
| `scripts/run_v6_2_additions.py` | Empirical strategy portfolio and decision-rule benchmark. |
| `scripts/run_v7_final_completion_bounds.py` | Empirical strategy portfolio. |
| `scripts/run_v7_final_equivalence.py` | Primary and independent scenario records at the paths specified in the script. |

Generate or restore these inputs at the paths expected by each script before running the dependent analyses. The archived independent files are retained under `independent_generator_R_v6_1/results/`; scripts that expect copies under the main `results/` directory require those paths to be populated locally. This reduced repository does not provide all final numerical outputs as ready-to-use inputs.

## Interpretation

- Shapley values describe marginal access in the defined portfolio game; they do not directly establish causal elemental effects.
- Completion bounds depend on bounded monotonicity and preservation of observed coalition values. They do not establish chemical realizability of the hypothetical completions.
- Bootstrap resamples observed evidence and cannot add missing counterfactual composition combinations.
- Failure to establish a difference from random selection is not proof of equivalence. The equivalence sensitivity is explicitly exploratory.
- Candidate-selection comparisons depend on the declared utility transformations and internal preference model. Simple controls match Nash-product selection in this portfolio.

## Data sources and citation

1. Şarlar K. *Machine learning screening of magnetocaloric high-entropy alloys using a field-normalized ΔSref target.* Journal of Magnetism and Magnetic Materials 650, 174120 (2026). [DOI: 10.1016/j.jmmm.2026.174120](https://doi.org/10.1016/j.jmmm.2026.174120).
2. Şarlar K. *Full text target auditing reveals chemistry regime confounding in composition only Curie temperature screening.* Source audit archived on Zenodo. [DOI: 10.5281/zenodo.23102726](https://doi.org/10.5281/zenodo.23102726).

When using the data or code, cite the study title and author above, this repository's actual URL and version, and the relevant source datasets and original experimental papers. Publication identifiers are preserved for traceability. The Zenodo DOI identifies the earlier source audit, not this code release.

## License

Original project code is released under the **MIT License**, provided in [`LICENSE`](LICENSE). Retain its copyright and permission notice when redistributing substantial portions of the software.

The code license does not relicense literature-derived source data, third-party code, or externally sourced material. Source-specific licenses and attribution requirements continue to apply. Empirical tables and archived research outputs are not granted a separate reuse license by the MIT software license; consult their source terms or contact the author where no applicable permission is specified.
