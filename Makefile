PYTHON ?= python3
CORES ?= -1
METRIC ?= fairness
MAX_BANS ?= 12
FAMILIES ?= bargaining,negotiation,persuasion
SOLVER_BACKEND ?= gambit

.PHONY: prepare-data run-analysis run-banning reports numbers figure2 figure3 si-heatmap figures run-paper test clean clean-cache

# Rebuild data/*.csv from the public GLEE repository (downloads ~1.5GB, ~10 min)
prepare-data:
	$(PYTHON) data_preparation/prepare_data.py

# Equilibria for every strategy subset + release transitions.
# THE expensive step: hundreds of CPU-hours per family/metric (days of wall
# clock even on a many-core machine). Checkpoints every 256 subsets, so an
# interrupted run resumes from the pickle cache.
run-analysis:
	$(PYTHON) -m pipeline.comprehensive_analysis --families $(FAMILIES) --cores $(CORES) --metric $(METRIC) --solver_backend $(SOLVER_BACKEND)

# Regulator with a ban budget (DP over the cached equilibria; minutes)
run-banning:
	$(PYTHON) -m pipeline.banning_analysis --families $(FAMILIES) --metric $(METRIC) --max_bans $(MAX_BANS)

# Text reports + showcase infographics (minutes — re-solves the extreme cases)
reports:
	$(PYTHON) -m pipeline.analyze_results --families $(FAMILIES) --metric $(METRIC) --transition_metric delta_designer_value
	$(PYTHON) -m pipeline.analyze_results --families $(FAMILIES) --metric $(METRIC) --transition_metric delta_alice_gain
	$(PYTHON) -m pipeline.analyze_results --families $(FAMILIES) --metric $(METRIC) --transition_metric delta_bob_gain
	$(PYTHON) -m pipeline.adversarial_report --families $(FAMILIES) --metric $(METRIC)
	$(PYTHON) -m pipeline.find_extreme_cases --families $(FAMILIES) --metric $(METRIC)

# --- Paper artifacts (defaults reproduce the published mixed/average results) ---

# Stdout numbers: Figure 1 case study + SI tables, and all Figure 2 values with CIs
numbers:
	$(PYTHON) -m figures.figure1_case_study
	$(PYTHON) -m figures.figure2_numbers

# File-producing figures (land in output/figures/)
figure2:
	$(PYTHON) -m figures.figure2_plot --all

figure3:
	$(PYTHON) -m figures.figure3_pa_rate

si-heatmap:
	$(PYTHON) -m figures.si_banning_heatmap

figures: figure2 figure3 si-heatmap

# Full paper pipeline: both regulator metrics, then reports, numbers and figures.
run-paper:
	$(MAKE) run-analysis METRIC=fairness
	$(MAKE) run-analysis METRIC=efficiency
	$(MAKE) run-banning METRIC=fairness
	$(MAKE) run-banning METRIC=efficiency
	$(MAKE) reports METRIC=fairness
	$(MAKE) reports METRIC=efficiency
	$(MAKE) numbers
	$(MAKE) figures

test:
	$(PYTHON) -m pytest tests/ -q

# Remove regenerable-in-seconds outputs. Does NOT touch the equilibrium
# caches and transition CSVs in output/*/calculations (hundreds of
# CPU-hours to recompute) — use clean-cache for those.
clean:
	rm -rf output/figures output/*/summaries output/*/visualizations

clean-cache:
	@echo "This deletes output/*/calculations — the equilibrium caches that take"
	@echo "hundreds of CPU-hours to recompute. Run 'make clean-cache-confirm' to proceed."

clean-cache-confirm:
	rm -rf output/
