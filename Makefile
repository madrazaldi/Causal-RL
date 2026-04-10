PYTHON ?= python3
BOOTSTRAP_REPS ?= 500

.PHONY: all dataset train evaluate report test release-check

all: dataset train evaluate report

dataset:
	$(PYTHON) -m causal_rl.build_dataset

train:
	$(PYTHON) -m causal_rl.train_policies

evaluate:
	$(PYTHON) -m causal_rl.evaluate --bootstrap-reps $(BOOTSTRAP_REPS) --run-ablations

report:
	$(PYTHON) -m causal_rl.report

test:
	pytest -q

release-check: test all
