# Independent robustness generator R

Analyze this generator separately from the primary scalar generator. Its two latent objectives, sampling, baseline, implemented tie conventions and confidence definition are retained from the corrected v6.1 source.

From the repository root:

```bash
python reproduce.py independent --jobs 2
# After the primary stage has also completed:
python reproduce.py summaries
```

`code/run_all.py` generates 4,000 general worlds (seeds 0–3999), 9,000 targeted draws (60000–68999), the full-coverage check (90000–90019), and held-out random-forest diagnostics. Each world uses 100 publication and group bootstrap resamples. Generated CSV/JSON files are written under the ignored root `results/robustness_generator_R/` directory. No generated results are tracked in this repository.

For a small execution check only:

```bash
python independent_generator_R_v6_1/code/run_all.py --general 4 --targeted 4 --skip-classifiers
```

This overwrites the local benchmark output names with reduced runs; rerun the full stage before the summary stage. Seeds identify worlds independently of the execution block size. `analyze5.py` accepts `--input` and `--output` and does not require the missing historical `hea_real_diagnostics.json`. The removed exploratory case-probability calculation did not enter held-out benchmark performance metrics.

Structural classes use observed distinct supports. Material error and its uniform-random comparator condition on a unique estimated top; exact recovery also requires a unique true top. Confidence means the full-data unique top recurs in at least 90% of resamples. The primary generator uses a different confidence definition; do not pool the two rates. Tests can be run without precomputed results using `python reproduce.py tests`.
