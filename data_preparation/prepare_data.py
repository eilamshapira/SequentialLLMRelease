#!/usr/bin/env python3
"""
Rebuild data/{bargaining,negotiation,persuasion}.csv from the public GLEE
repository (https://github.com/eilamshapira/GLEE).

The data files are per-market regression coefficients: for every market
configuration, an OLS model of each metric (fairness, efficiency,
alice_self_gain, bob_self_gain) on the Alice/Bob model pair, reported as
absolute values with 95% CIs.

Pipeline (all code lives in GLEE):
  1. Shallow-fetch GLEE at a pinned commit.
  2. Patch analyze/metrics.py with the corrected persuasion buyer payoff
     (data_preparation/persuasion_metrics.patch) if not already present.
  3. Patch analyze/ML.py with the --absolute_values mode
     (data_preparation/absolute_values_ML.patch) if not already present.
  4. Aggregate the raw game logs (Data/llm_vs_llm) into per-family
     "with stats" CSVs via analyze.configs.
  5. Run analyze/ML.py --per_market --absolute_values per family.
  6. Copy the outputs to data/{family}.csv and verify them against
     data_preparation/reference_values.json.

The regression is deterministic up to floating-point summation order,
which depends on filesystem enumeration order of the raw games; expect
agreement with the published values within ~1e-13.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv

GLEE_URL = "https://github.com/eilamshapira/GLEE.git"
GLEE_COMMIT = "68a33e98b035b97f945badee8f325001555c0049"
FAMILIES = ["bargaining", "negotiation", "persuasion"]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
PATCH_PATH = os.path.join(THIS_DIR, "absolute_values_ML.patch")
PERSUASION_PATCH_PATH = os.path.join(THIS_DIR, "persuasion_metrics.patch")
REFERENCE_PATH = os.path.join(THIS_DIR, "reference_values.json")

# Module-level imports of GLEE's analyze/ML.py and analyze/configs.py,
# pinned to the versions the published data was produced with.
GLEE_ANALYSIS_DEPS = [
    "numpy==2.3.5",
    "pandas==2.3.3",
    "statsmodels==0.14.5",
    "scikit-learn==1.7.2",
    "matplotlib==3.10.7",
    "tqdm==4.67.1",
    "joblib==1.5.2",
    "patsy==1.0.2",
    "shap==0.50.0",
    "catboost==1.2.8",
    "xgboost==3.1.2",
]


def run(cmd, cwd=None, check=True):
    print(f"$ {' '.join(cmd)}" + (f"   (in {cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd, check=check)


def fetch_glee(workdir):
    glee_dir = os.path.join(workdir, "GLEE")
    if os.path.exists(os.path.join(glee_dir, "analyze", "ML.py")):
        print(f"GLEE checkout already present at {glee_dir}, reusing it.")
        return glee_dir
    os.makedirs(glee_dir, exist_ok=True)
    run(["git", "init", "-q"], cwd=glee_dir)
    run(["git", "remote", "add", "origin", GLEE_URL], cwd=glee_dir, check=False)
    # GitHub allows fetching an arbitrary commit by SHA; ~1.5GB of game data.
    run(["git", "fetch", "--depth", "1", "origin", GLEE_COMMIT], cwd=glee_dir)
    run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=glee_dir)
    return glee_dir


def apply_patch(glee_dir):
    ml_path = os.path.join(glee_dir, "analyze", "ML.py")
    with open(ml_path) as f:
        if "--absolute_values" in f.read():
            print("analyze/ML.py already supports --absolute_values; no patch needed.")
            return
    run(["git", "apply", os.path.relpath(PATCH_PATH, glee_dir)], cwd=glee_dir)
    print("Applied absolute_values patch to analyze/ML.py")


def apply_persuasion_patch(glee_dir):
    """Correct the persuasion buyer payoff before fitting regressions."""
    metrics_path = os.path.join(glee_dir, "analyze", "metrics.py")
    with open(metrics_path) as f:
        if "low_quality_purchases" in f.read():
            print("Persuasion payoff correction is already present; no patch needed.")
            return
    run(["git", "apply", os.path.relpath(PERSUASION_PATCH_PATH, glee_dir)], cwd=glee_dir)
    print("Applied persuasion payoff correction to analyze/metrics.py")


def build_env(workdir, reuse):
    env_dir = os.path.join(workdir, "glee-analysis-env")
    python = os.path.join(env_dir, "bin", "python")
    if reuse and os.path.exists(python):
        print(f"Reusing analysis venv at {env_dir}")
        return python
    print(f"Creating analysis venv at {env_dir} (Python {sys.version_info.major}.{sys.version_info.minor})")
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    run([python, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-q"] + GLEE_ANALYSIS_DEPS)
    return python


def build_with_stats(glee_dir, python):
    code = (
        "from analyze.configs import create_configs_file, create_config_with_stats\n"
        + "".join(
            f"create_configs_file({fam!r}, data_path='Data/llm_vs_llm', include_human=False)\n"
            f"create_config_with_stats({fam!r}, data_path='Data/llm_vs_llm')\n"
            for fam in FAMILIES
        )
    )
    run([python, "-c", code], cwd=glee_dir)


def run_regressions(glee_dir, python):
    for fam in FAMILIES:
        run([python, "analyze/ML.py",
             "--exp_name", f"{fam}_analysis",
             "--per_market", "--absolute_values",
             f"--{fam}", f"output/configs/llm_vs_llm_{fam}_with_stats.csv"],
            cwd=glee_dir)


def collect_outputs(glee_dir):
    os.makedirs(os.path.join(REPO_ROOT, "data"), exist_ok=True)
    for fam in FAMILIES:
        src = os.path.join(glee_dir, "output", "analyze_coefs",
                           f"{fam}_analysis_per_market.csv")
        dst = os.path.join(REPO_ROOT, "data", f"{fam}.csv")
        shutil.copyfile(src, dst)
        print(f"Wrote {dst}")


def verify(atol):
    import pandas as pd

    with open(REFERENCE_PATH) as f:
        reference = json.load(f)

    key = ["family", "metric", "value", "market"]
    values = ["effect", "ci_low", "ci_high"]
    ok = True
    for fam in FAMILIES:
        df = pd.read_csv(os.path.join(REPO_ROOT, "data", f"{fam}.csv"))
        ref = reference[fam]
        if len(df) != ref["n_rows"]:
            print(f"FAIL {fam}: {len(df)} rows, expected {ref['n_rows']}")
            ok = False
            continue
        sample = pd.DataFrame(ref["sample"])
        merged = sample.merge(df, on=key, suffixes=("_ref", ""), how="left")
        if merged[values].isna().any().any():
            print(f"FAIL {fam}: some reference rows missing from the rebuilt file")
            ok = False
            continue
        max_diff = max((merged[v] - merged[f"{v}_ref"]).abs().max() for v in values)
        status = "OK" if max_diff <= atol else "FAIL"
        print(f"{status} {fam}: {len(df)} rows, max |diff| vs reference = {max_diff:.2e}"
              f" (tolerance {atol:.0e})")
        ok = ok and max_diff <= atol
    return ok


def main():
    parser = argparse.ArgumentParser(
        prog='python data_preparation/prepare_data.py',
        description=__doc__.splitlines()[1])
    parser.add_argument("--workdir", default="glee_build",
                        help="Where GLEE and its venv are placed (default: glee_build). "
                             "Must not point at an existing non-empty directory: it is "
                             "DELETED after a successful run unless --keep-workdir is given.")
    parser.add_argument("--python", default=None,
                        help="Use this interpreter for the GLEE analysis instead of "
                             "building a pinned venv (must have the analysis deps)")
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="Verification tolerance vs published values (default 1e-9; "
                             "actual agreement is ~1e-13)")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep the GLEE checkout and venv after a successful run")
    args = parser.parse_args()

    workdir = os.path.abspath(args.workdir)
    created_workdir = not os.path.exists(workdir)
    if not created_workdir and os.listdir(workdir) and not os.path.exists(
            os.path.join(workdir, "GLEE")):
        raise SystemExit(f"--workdir {workdir} exists and is not a previous prepare_data "
                         f"workdir — refusing to use (it would be deleted on success).")
    os.makedirs(workdir, exist_ok=True)

    glee_dir = fetch_glee(workdir)
    apply_persuasion_patch(glee_dir)
    apply_patch(glee_dir)
    python = args.python or build_env(workdir, reuse=True)
    build_with_stats(glee_dir, python)
    run_regressions(glee_dir, python)
    collect_outputs(glee_dir)

    if not args.skip_verify:
        if not verify(args.atol):
            sys.exit("Verification failed — rebuilt data does not match the published values.")

    if not args.keep_workdir:
        shutil.rmtree(workdir)
        print(f"Removed {workdir} (use --keep-workdir to keep it)")


if __name__ == "__main__":
    main()
