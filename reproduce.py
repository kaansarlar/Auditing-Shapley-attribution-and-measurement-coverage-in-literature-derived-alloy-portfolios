"""Run analyses in dependency order from a clean checkout (Python 3.12).

All generated files remain in ignored results directories. The default empirical
stage uses the full declared empirical settings; the long simulation stages are
explicit. Use --list to inspect the execution order without running anything.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'empirical': [
        ['scripts/run_empirical_nash_shapley.py'],
        ['scripts/run_decision_validation_v3.py'],
        ['scripts/run_decision_rule_benchmark.py'],
        ['scripts/run_v6_2_additions.py'],
        ['scripts/run_v7_final_completion_bounds.py'],
        ['scripts/run_temperature_window_sensitivity_v3.py'],
        ['scripts/plot_v7_final_window_sensitivity.py'],
        ['scripts/run_s15_proxy_counterfactual_v3_2.py'],
        ['scripts/run_utility_sensitivity.py'],
        ['extension/audit_portfolio.py', '--input',
         'results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv',
         '--output', 'extension/results/empirical_audit'],
    ],
    'acquisition': [['extension/run_acquisition.py']],
    'primary': [
        ['scripts/run_synthetic_identifiability_benchmark_v6.py'],
        ['scripts/run_hea_like_v6.py'],
    ],
    'independent': [['independent_generator_R_v6_1/code/run_all.py']],
    'summaries': [
        ['scripts/run_benchmark_summaries.py'],
        ['scripts/run_v7_final_equivalence.py'],
    ],
    'predictive': [
        ['scripts/run_dsref_grouped_nested.py'],
        ['scripts/run_tc_regression_family_validation.py'],
        ['scripts/run_tc_window_family_validation.py'],
        ['scripts/run_phase_grouped_benchmark.py'],
    ],
    'model-grid': [['scripts/run_model_grid_game_v2.py']],
    'tests': [['scripts/validate_v7_final.py']],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stages', nargs='*', choices=list(STAGES) + ['all'])
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--jobs', type=int, default=2,
                        help='Upper bound on joblib workers (default: 2).')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    stages = args.stages or ['empirical']
    if 'all' in stages:
        stages = list(STAGES)
    env = os.environ.copy()
    env.update(LOKY_MAX_CPU_COUNT=str(args.jobs), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    for stage in stages:
        for command in STAGES[stage]:
            print(stage + ': ' + ' '.join([sys.executable, *command]), flush=True)
            if not args.list:
                subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
