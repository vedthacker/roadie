"""Run the complete RoadEye pipeline with one command.

    python scripts/run_pipeline.py                      # full dataset (recommended for final results)
    python scripts/run_pipeline.py --download           # download GTSRB first
    python scripts/run_pipeline.py --max-per-class 100  # DEVELOPMENT SUBSET (labelled as such everywhere)
    python scripts/run_pipeline.py --stages audit prepare baseline compare   # run selected stages
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL = ["audit", "prepare", "baseline", "compare", "tune", "final", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=None, help="development subset of the training pool")
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--stages", nargs="+", choices=ALL, default=ALL)
    ap.add_argument("--no-tune-others", action="store_true", help="tune only the SVM (skip KNN / Tree grids)")
    ap.add_argument("--full-grid", action="store_true", help="also search the linear-kernel grid")
    ap.add_argument("--n-jobs", type=int, default=2, help="parallel jobs inside GridSearchCV")
    args = ap.parse_args()
    # must be set BEFORE importing src.config
    if args.max_per_class:
        os.environ["ROADEYE_MAX_PER_CLASS"] = str(args.max_per_class)
    if args.max_test:
        os.environ["ROADEYE_MAX_TEST"] = str(args.max_test)
    from src import config, pipeline as P
    from src.data_loader import DatasetNotFoundError, download_gtsrb

    config.ensure_dirs()
    P.setup_logging()
    if args.download:
        download_gtsrb()
    print(f"RUN MODE: {config.RUN_MODE}")
    try:
        if "audit" in args.stages:
            P.stage_audit()
        data = P.stage_prepare()
        if "prepare" in args.stages:
            P.stage_eda(data)
        if "baseline" in args.stages:
            P.stage_baseline(data)
        if "compare" in args.stages:
            P.stage_compare(data)
        if "tune" in args.stages:
            P.stage_tune(data, tune_others=not args.no_tune_others, full_grid=args.full_grid, n_jobs=args.n_jobs)
        if "final" in args.stages:
            P.stage_final_eval(data)
    except DatasetNotFoundError as exc:
        print("\nDATASET PENDING:\n", exc)
        sys.exit(1)
    if "report" in args.stages:
        import subprocess
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "generate_report_assets.py")])
    print("Pipeline finished. Models: models/svm/, metrics: results/metrics/, reports: docs/")


if __name__ == "__main__":
    main()
