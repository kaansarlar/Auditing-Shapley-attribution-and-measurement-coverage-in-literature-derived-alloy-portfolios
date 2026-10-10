from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_synthetic_identifiability_benchmark_v6 as benchmark


def main():
    out = ROOT / "results" / "synthetic_identifiability_benchmark_v6"
    out.mkdir(parents=True, exist_ok=True)
    worlds = benchmark.run_hea_like(n_worlds=600, seed=20262026, n_boot=500)
    worlds.to_csv(out / "hea_like_worlds_v6.csv", index=False)
    print(out / "hea_like_worlds_v6.csv")


if __name__ == "__main__":
    main()
