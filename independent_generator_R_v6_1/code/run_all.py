"""Reproduce independent generator R without precomputed simulation records.

Seed index determines each world. Full settings: 4,000 general worlds, 9,000
targeted draws and 100 within-world bootstrap resamples, unchanged in bench.py.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'robustness_generator_R'
CODE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--general', type=int, default=4000)
    parser.add_argument('--targeted', type=int, default=9000)
    parser.add_argument('--skip-classifiers', action='store_true',
                        help='For small smoke runs only; not the reported analysis.')
    args = parser.parse_args()
    if args.general < 1 or args.targeted < 1:
        parser.error('World counts must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    commands = [
        ['bench.py', '0', str(args.general)],
        ['hea_match.py', '60000', str(60000 + args.targeted)],
        ['sanity.py'],
    ]
    for command in commands:
        subprocess.run([sys.executable, str(CODE / command[0]), *command[1:]],
                       cwd=OUT, check=True)
    pd.read_csv(OUT / f'res_0_{args.general}.csv').to_csv(OUT / 'bench_all.csv', index=False)
    pd.read_csv(OUT / f'hea_match_60000_{60000+args.targeted}.csv').to_csv(
        OUT / 'hea_match_all.csv', index=False)
    if not args.skip_classifiers:
        subprocess.run([sys.executable, str(CODE / 'analyze5.py'), '--input',
                        str(OUT / 'bench_all.csv'), '--output', str(OUT)], check=True)


if __name__ == '__main__':
    main()
