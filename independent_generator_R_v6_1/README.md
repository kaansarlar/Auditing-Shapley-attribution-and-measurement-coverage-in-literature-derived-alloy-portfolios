# Independent robustness generator R — v6.1 handoff

The corrected generator is analysed separately from the primary generator. **Never pool the two.**

## Contents

- `code/bench.py`: generator. To reproduce the 4000 general worlds, run inside `code/`: `python bench.py 0 2000` and `python bench.py 2000 4000` (seed = world index).
- `code/analyze5.py`: merges results, writes `final_numbers.json` and trains the held-out random forests.
- `code/hea_match.py`: HEA-targeted draws. Run seed blocks 60000–68999, e.g. `python hea_match.py 60000 63000`.
- `code/sanity.py`: full-coverage check (20 worlds, seeds 90000–90019).
- `results/bench_all.csv`: 4000 general worlds.
- `results/hea_match_all.csv`: 9000 HEA-targeted draws.
- `results/final_numbers.json` and `results/heldout_predictions.csv`.
- `results/sanity_summary.json` and `results/hea_match_summary.json`.
- `results/robustness_summary_v6_1.json`: **all numbers needed for the v6.1 manuscript.**
- `tests/test_v6_robustness_generator.py`: tests for exact Shapley efficiency, equivalent-player ties, the veto bound, full-coverage recovery, case diagnostics and summary consistency. Run with `python -m pytest -q tests`.
- `MANIFEST_SHA256.txt`.

## Definitions (same as the primary generator unless stated)

- **Structural classes:** taken from the observed distinct supports, not from the design.
  - Veto: a player present in every observed support.
  - Equivalent pair: two identical non-zero support columns.
  - Classes: neither, equivalent only, veto only, veto + equivalent.
- **Unique top:** a tie at the maximum (tolerance 1e-12, Shapley values rounded to 1e-12) means no unique top. Ties are not charged to the method in the random comparison.
- **Material error:** the unique estimated top has a complete-design Shapley value below half of the true maximum.
- **Random baseline (per world):** the fraction of the five players that satisfy the same loss.
- **Exact recovery:** evaluated only where the true top is also unique (the truth is tied in 4.5% of worlds), and compared with 1/5.
- **Intervals:** paired world-level bootstrap (4000 resamples) for the material-error difference; Wilson intervals for exact recovery.
- **Bootstrap:** publication and research-group units, 100 resamples each. A result is "confident" when the full-data unique top recurs in ≥ 0.90 of resamples. This threshold differs from the primary generator's modal ≥ 0.80, so do not compare confidently-wrong rates numerically across generators.

## Suggested text for Section 2.8

> A second, independently specified generator provides a robustness analysis that is never pooled with the primary benchmark. Each of 4000 worlds has five optional players and two latent objectives with intercepts, main effects and pairwise interactions; the true joint quality is the geometric mean of their logistic transforms, and the exact Shapley vector of the complete 32-support game defines the truth. A literature portfolio of 20, 50, 100 or 200 records is generated as publications of two to eight records. Studied supports are selected with probability proportional to exp[λ(q* − mean q*)], λ ∈ {0, 1.5, 3}, so that well-performing systems are more likely to have been studied, and support frequencies follow a Zipf law. Veto elements and equivalent pairs are imposed by design with probability 0.3 each. Structural classes are assigned from the observed supports. Publications are grouped into research groups that may repeat their first support. Group, publication, alloy-level and measurement effects are added to the observed scores. Material error, the per-world uniform-random baseline and the unique-top condition are defined as for the primary generator; exact recovery is evaluated only where the true top is unique. Publication and research-group bootstraps use 100 resamples, and a result is called confident when the full-data unique top recurs in at least 90% of resamples. Because the two generators differ in truth construction, sampling, dependence and confidence threshold, their rates are reported side by side, and only qualitative agreement is interpreted.

## Key numbers (unique-top worlds; difference = material error − random baseline)

| Class | n | Material error | Random | Difference [95% CI] | Exact recovery [Wilson 95%] |
|---|---|---|---|---|---|
| Neither | 888 | 0.384 | 0.653 | −0.269 [−0.299, −0.239] | 0.430 [0.397, 0.464] |
| Equivalent only | 689 | 0.466 | 0.646 | −0.180 [−0.214, −0.144] | 0.373 [0.337, 0.411] |
| **Veto only** | 804 | 0.632 | 0.657 | **−0.026 [−0.056, +0.005]** | **0.205 [0.178, 0.235]** |
| Veto + equivalent | 583 | 0.575 | 0.639 | −0.064 [−0.101, −0.028] | 0.254 [0.219, 0.291] |
| **HEA-matched (exact), 288 worlds** | 288 | 0.653 | 0.639 | **+0.014 [−0.037, +0.063]** | **0.163 [0.124, 0.212]** (n = 270) |

Further results for the HEA-matched worlds:

- The top element is the veto player in 100% of worlds.
- The true top is the veto player in 15.3% of worlds.
- Publication-bootstrap confident: 66 worlds, exact recovery 14/66 = 0.212 [0.131, 0.325].
- Group-bootstrap confident: 69 worlds, exact recovery 13/69 = 0.188 [0.114, 0.296].
- Both confident intervals contain 0.20.

General worlds:

- Confident publication bootstrap (43.9% of worlds): material error 0.521 [0.497, 0.544], compared with 0.725 when not confident.

Held-out random forests (target: material error; 2800 training worlds, 1200 test worlds):

| Features | AUC, all test worlds | AUC, 900 unique-top worlds |
|---|---|---|
| Structural diagnostics | 0.680 | 0.601 |
| Structural + bootstrap | 0.764 | 0.611 |
| Bootstrap frequency alone | — | 0.491 |

## Wording consistent with these numbers

- **Veto only:** statistically indistinguishable from uniform random selection in both generators, for material error and for exact recovery.
- **Veto + equivalent:** the primary generator gives +0.045 [+0.000, +0.090] and generator R gives −0.064 [−0.101, −0.028]. The sign changes between generators, which indicates sensitivity to the data-generating mechanism.
- **HEA-matched worlds:** indistinguishable from random in both generators. Confident bootstrap subsets also remain indistinguishable from random top-player identification.
