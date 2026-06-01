PYTHON ?= python3
BOOTSTRAP_REPS ?= 500
N_JOBS ?= auto
PAPER_DIR ?= paper
DOCKER_IMAGE ?= causal-rl:latest

.PHONY: all dataset train evaluate report runtime-benchmark ground-truth-benchmark claim-boundary theorem-assumptions paper paper-check release-check docker-build docker-test docker-shell

all: dataset train evaluate ground-truth-benchmark claim-boundary theorem-assumptions report

dataset:
	$(PYTHON) -m causal_rl.build_dataset

train:
	$(PYTHON) -m causal_rl.train_policies --n-jobs $(N_JOBS)

evaluate:
	$(PYTHON) -m causal_rl.evaluate --bootstrap-reps $(BOOTSTRAP_REPS) --run-ablations --n-jobs $(N_JOBS)

report:
	$(PYTHON) -m causal_rl.report

runtime-benchmark:
	$(PYTHON) -m causal_rl.benchmark_runtime

ground-truth-benchmark:
	$(PYTHON) -m causal_rl.ground_truth_benchmark

claim-boundary:
	$(PYTHON) -m causal_rl.claim_boundary

theorem-assumptions:
	$(PYTHON) -m causal_rl.theorem_assumptions

paper:
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

paper-check: paper
	@pages=$$(sed -n 's/.*Output written on paper.pdf (\([0-9][0-9]*\) pages.*/\1/p' $(PAPER_DIR)/paper.log | tail -1); \
	if [ -z "$$pages" ]; then echo "Could not determine paper page count"; exit 1; fi; \
	if [ "$$pages" -gt 5 ]; then echo "paper.pdf has $$pages pages; expected <= 5"; exit 1; fi; \
	if grep -E "undefined references|undefined citations|Citation .* undefined|Reference .* undefined|There were undefined" $(PAPER_DIR)/paper.log; then exit 1; fi; \
	if grep -E "Overfull \\\\hbox" $(PAPER_DIR)/paper.log; then exit 1; fi; \
	echo "paper.pdf has $$pages pages and passed LaTeX log checks"

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-test: docker-build
	docker run --rm $(DOCKER_IMAGE)

docker-shell: docker-build
	docker run --rm -it $(DOCKER_IMAGE) /bin/bash

release-check: all paper-check
