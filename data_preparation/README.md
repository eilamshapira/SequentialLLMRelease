# Data preparation

`prepare_data.py` rebuilds `data/{bargaining,negotiation,persuasion}.csv` —
the per-market OLS estimates of each metric per model pair — from the public
[GLEE](https://github.com/eilamshapira/GLEE) repository, pinned to commit
`68a33e98b035b97f945badee8f325001555c0049`.

Run from the repository root:

```bash
make prepare-data          # or: python data_preparation/prepare_data.py
```

Requirements: ~1.5GB download (GLEE's raw game logs) plus a temporary
virtualenv with GLEE's analysis stack (statsmodels, scikit-learn, shap,
catboost, xgboost — installed automatically with pinned versions).
Roughly 10 minutes end to end; everything happens inside `--workdir`
(default `glee_build/`), which is deleted after a successful run.

| Flag | Meaning |
|---|---|
| `--workdir DIR` | Where GLEE and its venv live (default `glee_build/`). Deleted on success. |
| `--keep-workdir` | Keep the checkout/venv (useful when re-running). |
| `--python PATH` | Skip the venv and use this interpreter (must have the analysis deps). |
| `--atol X` | Verification tolerance (default 1e-9; actual agreement ~1e-13). |
| `--skip-verify` | Skip the verification step. |

## Files

- `persuasion_metrics.patch` — corrects the persuasion buyer payoff by
  counting low-quality purchases rather than all successful rounds. This
  correction is applied automatically before the regressions are fitted.
- `absolute_values_ML.patch` — adds an `--absolute_values` mode to GLEE's
  `analyze/ML.py` (report intercept + coefficient with propagated variance
  instead of baseline-relative effects). Applied automatically after
  fetching GLEE; skipped if the flag has been merged upstream.
- `reference_values.json` — row counts plus a deterministic sample of the
  published values: **every 97th row after sorting by (family, metric,
  value, market)** — 168/251/168 rows for bargaining/negotiation/persuasion.
  `prepare_data.py` verifies the rebuilt CSVs against it. The regression is
  deterministic up to floating-point summation order (which follows the
  filesystem enumeration order of the ~77,000 game directories), so rebuilt
  values match the published ones to ~1e-13.
