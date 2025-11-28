.PHONY: setup smoke demo run-all clean help

# Python interpreter
PYTHON := python3

# Directories
SRC := src
DATA := data
NOTEBOOKS := notebooks
CARDS := cards
REPORTS := reports

help:
	@echo "Mini-Pretrain Corpus Curation Pipeline"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies and create directories"
	@echo "  make smoke      - Run end-to-end on tiny sample (quick test)"
	@echo "  make demo       - Run on 10% of data for demonstration"
	@echo "  make run-all    - Run complete pipeline on full data"
	@echo "  make clean      - Remove generated files and cache"
	@echo "  make notebooks  - Launch Jupyter for exploration"
	@echo ""
	@echo "Individual steps:"
	@echo "  make ingest     - Run data ingestion (web + code)"
	@echo "  make filter     - Run all filtering steps"
	@echo "  make dedup      - Run deduplication"
	@echo "  make train      - Train baseline and curated models"
	@echo "  make eval       - Run evaluation harness"
	@echo "  make report     - Generate final reports"

setup:
	@echo "Setting up environment..."
	pip install -q --upgrade pip
	pip install -q torch transformers accelerate datasets
	pip install -q polars pyarrow datasketch
	pip install -q gcld3 sentencepiece tiktoken
	pip install -q detoxify scrubadub webdataset
	pip install -q beautifulsoup4 requests tqdm
	pip install -q scikit-learn matplotlib seaborn
	pip install -q huggingface_hub jupyter ipywidgets
	@echo "Creating directory structure..."
	mkdir -p $(DATA)/raw $(DATA)/processed $(DATA)/shards $(DATA)/small_eval
	mkdir -p $(REPORTS) $(CARDS) $(NOTEBOOKS)
	mkdir -p models/baseline models/curated
	@echo "✅ Setup complete!"

smoke:
	@echo "Running smoke test (tiny sample)..."
	$(PYTHON) $(SRC)/ingest_web.py --sample-size 10
	$(PYTHON) $(SRC)/ingest_code.py --sample-size 5
	$(PYTHON) $(SRC)/language_id.py
	$(PYTHON) $(SRC)/quality_filters.py
	$(PYTHON) $(SRC)/toxicity.py
	$(PYTHON) $(SRC)/pii_redact.py
	$(PYTHON) $(SRC)/license_check.py
	$(PYTHON) $(SRC)/dedup_minhash.py
	$(PYTHON) $(SRC)/mixture_build.py
	$(PYTHON) $(SRC)/shard_webdataset.py --num-shards 2
	@echo "✅ Smoke test passed!"

demo:
	@echo "Running demo (10% of data)..."
	$(PYTHON) $(SRC)/ingest_web.py --sample-size 100
	$(PYTHON) $(SRC)/ingest_code.py --sample-size 50
	$(PYTHON) $(SRC)/language_id.py
	$(PYTHON) $(SRC)/quality_filters.py
	$(PYTHON) $(SRC)/toxicity.py
	$(PYTHON) $(SRC)/pii_redact.py
	$(PYTHON) $(SRC)/license_check.py
	$(PYTHON) $(SRC)/dedup_minhash.py
	$(PYTHON) $(SRC)/mixture_build.py --ratio-web 0.7 --ratio-code 0.3
	$(PYTHON) $(SRC)/shard_webdataset.py --num-shards 10
	$(PYTHON) $(SRC)/contamination.py
	$(PYTHON) $(SRC)/train_baseline.py --epochs 2 --fast
	$(PYTHON) $(SRC)/train_curated.py --epochs 2 --fast
	$(PYTHON) $(SRC)/eval.py
	@echo "✅ Demo complete! Check $(REPORTS)/ for results"

run-all: ingest filter dedup train eval report
	@echo "✅ Full pipeline complete!"

ingest:
	@echo "Step 1: Data ingestion..."
	$(PYTHON) $(SRC)/ingest_web.py
	$(PYTHON) $(SRC)/ingest_code.py
	@echo "✅ Ingestion complete"

filter:
	@echo "Step 2-5: Filtering pipeline..."
	$(PYTHON) $(SRC)/language_id.py
	$(PYTHON) $(SRC)/quality_filters.py
	$(PYTHON) $(SRC)/toxicity.py
	$(PYTHON) $(SRC)/pii_redact.py
	$(PYTHON) $(SRC)/license_check.py
	@echo "✅ Filtering complete"

dedup:
	@echo "Step 6-8: Deduplication and sharding..."
	$(PYTHON) $(SRC)/dedup_minhash.py
	$(PYTHON) $(SRC)/mixture_build.py
	$(PYTHON) $(SRC)/shard_webdataset.py
	$(PYTHON) $(SRC)/contamination.py
	@echo "✅ Deduplication complete"

train:
	@echo "Step 9-10: Training models..."
	$(PYTHON) $(SRC)/train_baseline.py
	$(PYTHON) $(SRC)/train_curated.py
	@echo "✅ Training complete"

eval:
	@echo "Step 11: Evaluation..."
	$(PYTHON) $(SRC)/eval.py
	@echo "✅ Evaluation complete"

report:
	@echo "Step 12: Generating reports..."
	$(PYTHON) $(SRC)/generate_report.py
	@echo "✅ Reports generated in $(REPORTS)/"

notebooks:
	@echo "Launching Jupyter..."
	jupyter notebook $(NOTEBOOKS)

clean:
	@echo "Cleaning generated files..."
	rm -rf $(DATA)/processed/*
	rm -rf $(DATA)/shards/*
	rm -rf models/baseline/*
	rm -rf models/curated/*
	rm -rf $(REPORTS)/*.png $(REPORTS)/*.csv
	rm -rf __pycache__ $(SRC)/__pycache__ $(SRC)/utils/__pycache__
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "Removing all data including raw..."
	rm -rf $(DATA)/raw/*
	@echo "✅ Full cleanup complete"
