# Thin wrapper around run_all.py (which also works on Windows without make).
.PHONY: all results test
all:        ## full pipeline, including network training
	python run_all.py
results:    ## full pipeline with the committed network weights
	python run_all.py --skip-train
test:
	python -m pytest -q
