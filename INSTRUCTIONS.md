# 📋 Complete Step-by-Step Instructions

## For Colab/Kaggle Users

### Upload and Run
1. Upload entire `pretrain-mini/` folder to Colab
2. Open a terminal
3. Run: `cd pretrain-mini && make setup && make demo`
4. Wait ~30 minutes
5. Download `reports/` folder

## For Local Machine

### Prerequisites
- Python 3.8 or higher
- 5GB free disk space
- (Optional) NVIDIA GPU with CUDA

### Installation
```bash
# 1. Clone or download project
cd pretrain-mini

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
make setup
# Or manually: pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Quick Test (2 min)
```bash
make smoke
```

#### Option 2: Demo (30 min)
```bash
make demo
```

#### Option 3: Full Pipeline (2+ hours)
```bash
make run-all
```

#### Option 4: Step-by-Step
```bash
# Data ingestion
python src/ingest_web.py
python src/ingest_code.py

# Filtering
python src/language_id.py
python src/quality_filters.py
python src/toxicity.py
python src/pii_redact.py
python src/license_check.py

# Deduplication
python src/dedup_minhash.py
python src/mixture_build.py
python src/shard_webdataset.py
python src/contamination.py

# Training
python src/train_baseline.py --epochs 3
python src/train_curated.py --epochs 3

# Evaluation
python src/eval.py
python src/generate_report.py
```

### Viewing Results
```bash
# See final report
cat reports/FINAL_REPORT.md

# See all metrics
ls reports/

# Check models
ls models/baseline/final/
ls models/curated/final/
```

## Command Reference

### Makefile Commands
```bash
make help        # Show all commands
make setup       # Install dependencies
make smoke       # Quick test (tiny sample)
make demo        # Demo run (10% data)
make run-all     # Full pipeline
make ingest      # Just ingestion
make filter      # Just filtering
make dedup       # Just deduplication
make train       # Just training
make eval        # Just evaluation
make report      # Generate final report
make clean       # Remove generated files
make clean-all   # Remove everything including raw data
```

### Individual Script Options
```bash
# Ingest with sample size
python src/ingest_web.py --sample-size 10

# Custom toxicity threshold
python src/toxicity.py --threshold 0.5

# Custom dedup threshold
python src/dedup_minhash.py --threshold 0.90

# Custom mixture ratios
python src/mixture_build.py --ratio-web 0.8 --ratio-code 0.2

# Fast training (for testing)
python src/train_baseline.py --epochs 1 --fast
```

## File Locations

### Inputs
- Raw data: `data/raw/*.parquet`
- Processed data: `data/processed/*.parquet`

### Outputs
- Final dataset: `data/shards/final_dataset/`
- Models: `models/baseline/` and `models/curated/`
- Reports: `reports/*.md`, `reports/*.json`

## Expected Output Structure

After running `make demo`:
```
pretrain-mini/
├── data/
│   ├── raw/
│   │   ├── web_index.parquet
│   │   ├── code_index.parquet
│   │   └── *_metadata.json
│   ├── processed/
│   │   ├── 01_language_filtered.parquet
│   │   ├── 02_quality_filtered.parquet
│   │   ├── 03_toxicity_filtered.parquet
│   │   ├── 04_pii_redacted.parquet
│   │   ├── 05_license_verified.parquet
│   │   ├── 06_deduplicated.parquet
│   │   ├── 07_mixture.parquet
│   │   └── 08_clean.parquet
│   └── shards/
│       └── final_dataset/
│           ├── train/
│           └── validation/
├── models/
│   ├── baseline/
│   │   └── final/ (model files)
│   └── curated/
│       └── final/ (model files)
└── reports/
    ├── FINAL_REPORT.md
    ├── license_ledger.csv
    ├── dedup_stats.json
    ├── mixture_manifest.json
    ├── baseline_metrics.json
    ├── curated_metrics.json
    └── eval_results.json
```

## Success Indicators

✅ Pipeline succeeded if you see:
- `reports/FINAL_REPORT.md` exists
- Both models trained (check `models/*/final/`)
- Perplexity improvement > 0%

⚠️ Check if:
- Any step failed (check error messages)
- No model files generated
- Reports folder empty

## Time Estimates

| Command | Data Size | Time (CPU) | Time (GPU) |
|---------|-----------|------------|------------|
| `make smoke` | 15 docs | 2 min | 2 min |
| `make demo` | 150 docs | 60 min | 30 min |
| `make run-all` | Full | 3+ hours | 2+ hours |

## Troubleshooting

### "Package not found"
```bash
make setup  # Reinstall dependencies
```

### "Out of memory during training"
```bash
# Use fewer epochs or smaller batch
python src/train_curated.py --epochs 1 --fast
```

### "CUDA out of memory"
```bash
# Force CPU training (slower but works)
export CUDA_VISIBLE_DEVICES=""
python src/train_curated.py
```

### "Data files not found"
```bash
# Make sure ingestion completed
python src/ingest_web.py
python src/ingest_code.py
```

### Start Fresh
```bash
make clean-all  # Remove everything
make setup      # Reinstall
make smoke      # Test again
```

## For Portfolio/GitHub

### What to Include
1. ✅ All source code (`src/`)
2. ✅ Configuration (`makefile`, `pyproject.toml`)
3. ✅ Documentation (`README.md`, `cards/`)
4. ✅ Sample reports (from `make demo`)
5. ❌ Don't include: large data files, models

### .gitignore
Already included! Excludes:
- `data/raw/*`
- `data/processed/*`
- `models/*`
- `__pycache__/`

### Good README for Portfolio
The included `README.md` is interview-ready:
- Clear problem statement
- Technical architecture
- Success metrics
- Skills demonstrated

## Advanced Usage

### Scale to More Data
```python
# Edit src/ingest_web.py - add more sources
WEB_SOURCES.append({...})
```

### Add New Filters
```python
# Create src/custom_filter.py
# Follow pattern from existing filters
```

### Distributed Training
```bash
# Use accelerate for multi-GPU
accelerate launch src/train_curated.py
```

### Upload to HuggingFace
```python
from datasets import load_from_disk
dataset = load_from_disk("data/shards/final_dataset")
dataset.push_to_hub("your-username/mini-pretrain")
```

---

**Ready to start? Run `make smoke` now!** 🚀
