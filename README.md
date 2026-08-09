# Sequential LLM Release Facilitates Manipulation in Regulated Markets

Analysis code for *Sequential LLM Release Facilitates Manipulation in Regulated
Markets* by Eilam Shapira, Moshe Tennenholtz, and Roi Reichart.

The repository reproduces the paper's equilibrium analyses and figures for
LLM-mediated bargaining, negotiation, and persuasion markets.

## Installation

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root. Every `make` target accepts
`PYTHON=<interpreter>`.

## Reproducing the paper

The pipeline has four stages. Stage 2 is the only expensive one.

| Stage | Command | Runtime |
|---|---|---|
| 1. Build the input data from GLEE | `make prepare-data` | ~10 minutes |
| 2. Equilibria for all strategy subsets | `make run-analysis METRIC=fairness`<br>`make run-analysis METRIC=efficiency` | hardware-dependent; checkpointed |
| 3. Ban-budget analysis | `make run-banning METRIC=fairness`<br>`make run-banning METRIC=efficiency` | minutes |
| 4. Reports, numbers, and figures | `make reports`, `make numbers`, `make figures` | minutes |

`make run-paper` runs all four stages for both regulator metrics.

Stage 2 enumerates the Nash equilibria of every strategy subset of 13 models
in every market. Runtime depends strongly on hardware and process count. It
checkpoints its cache every 256 subsets, so an interrupted run resumes where
it stopped. Everything downstream reads the cached CSVs and runs in minutes.
`make clean` never touches these caches; only `make clean-cache` does.

The production pipeline uses [Gambit](https://www.gambit-project.org/)
16.4.1 `enummixed` for every mixed-
equilibrium cell in all three game families. The version is pinned in
`requirements.txt`, so solver choice does not depend on game family, subset
size, runtime, or checkpoint history. Nashpy support and vertex enumeration
remain available only for robustness checks by setting
`SOLVER_BACKEND=support` or `SOLVER_BACKEND=vertex`.

Gambit enumerates the extreme points of the equilibrium set. When a market has
several returned extreme-equilibrium profiles, the code first evaluates each
player's payoff and the regulator's matrix objective within each profile, and
then averages those scalar values. Separately, it stores the componentwise
mean strategy profile for downstream adoption analysis. The mean profile is
never substituted into the regulator's payoff matrix because that would
introduce cross-equilibrium terms.

### Paper artifacts

| Artifact | Command | Output |
|---|---|---|
| Figure 1 numbers + SI Tables 1–2 | `make numbers` | stdout |
| Figure 2 values with 95% CIs | `make numbers` | stdout |
| Figure 2 reproducibility preview + panels | `make figure2` | `output/figures/statistics_with_ci.*`, `panel_A..F.*` |
| Figure 3 | `make figure3` | `output/figures/banning_pa_rate.*` |
| SI banning heatmap | `make si-heatmap` | `output/figures/banning_heatmap_poisoned_apple.*` |
| Text statistics and infographics | `make reports` | `output/{metric}/summaries/`, `.../visualizations/` |

The published Figure 2 is composed manually from the script-generated values
and panels. The code reproduces every value, confidence interval, and panel,
and also creates a composite preview for verification. Figure 3 and the SI
heatmap are generated directly from the cached analysis results and require
stage 3 for both metrics.

Spot-check anchors (negotiation, fairness): the Poisoned Apple rate is
12.8% (1,659/12,999) at ban budget N=0, 23.3% (1,606/6,889) at N=1, and
72.0% (85/118) at N=7. At large N the sample shrinks and the
regulator often bans the newly released model outright; the
`added_strategy_banned` column in the banning summaries tracks this.

## Repository structure

```
core/               Payoff matrices, Nash solvers, the regulator's market choice
pipeline/           Subset enumeration, banning DP, reports, infographics
figures/            One script per paper artifact
data_preparation/   Rebuilds data/*.csv from the public GLEE repository
tests/              Pytest suite and a synthetic toy fixture
data/, output/      Generated inputs and results (not committed)
```

The input data (`data/*.csv`) holds per-market regression estimates of each
metric for every model pair, derived from the public
[GLEE](https://github.com/eilamshapira/GLEE) dataset. `make prepare-data`
rebuilds it at a pinned GLEE commit and verifies the result against stored
reference values; see `data_preparation/README.md`. The schema keeps GLEE's
original column names, including the `paramter_coef` spelling.

## Tests

```bash
make test
```

Covers all available solver backends on games with known equilibria, the
multiple-equilibrium aggregation rule, a full
pipeline run on the synthetic toy dataset against frozen values, the Figure
2 panel formulas, and the banning DP tie-breaking. To exercise the pipeline
by hand without downloading GLEE:

```bash
mkdir -p data && cp tests/fixtures/toy.csv data/toy.csv
make run-analysis FAMILIES=toy
make run-banning  FAMILIES=toy MAX_BANS=3
```

## Citation

```bibtex
@article{shapira2026sequential,
  title  = {Sequential LLM Release Facilitates Manipulation in Regulated Markets},
  author = {Shapira, Eilam and Tennenholtz, Moshe and Reichart, Roi},
  year   = {2026},
  url    = {https://arxiv.org/abs/2601.11496}
}
```

The raw game data belongs to the [GLEE](https://github.com/eilamshapira/GLEE)
benchmark; its terms apply to the downloaded data.

## License

MIT — see [LICENSE](LICENSE).
