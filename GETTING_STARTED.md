# 🚀 Getting Started with Mini-Pretrain Pipeline

## Quick Start (5 minutes to first run!)

### 1. Setup Environment
```bash
cd pretrain-mini
make setup
```
This installs all dependencies and creates the directory structure.

### 2. Run Smoke Test
```bash
make smoke
```
Runs end-to-end on tiny sample (~2 minutes). Perfect for testing!

### 3. Review Output
```bash
ls -R data/ models/ reports/
```

## What Just Happened?

The pipeline:
1. ✅ Downloaded 10 web documents + 5 code samples
2. ✅ Filtered for English and quality
3. ✅ Checked toxicity and PII
4. ✅ Verified licenses
5. ✅ Deduplicated with MinHash
6. ✅ Created balanced mixture
7. ✅ Generated reports

## Next Steps

### Run Demo (10% of data, ~30 min)
```bash
make demo
```
This runs the complete pipeline including training!

### Run Full Pipeline (2+ hours)
```bash
make run-all
```

### Run Individual Steps
```bash
make ingest    # Just data ingestion
make filter    # Just filtering
make dedup     # Just deduplication
make train     # Just training
```

## Understanding the Output

### Data Files
- `data/raw/` - Original ingested data
- `data/processed/` - After each filter step (01_*, 02_*, etc.)
- `data/shards/` - Final tokenized dataset

### Models
- `models/baseline/` - Trained on unfiltered data
- `models/curated/` - Trained on curated data

### Reports
- `reports/FINAL_REPORT.md` - **START HERE!**
- `reports/*.json` - Detailed metrics
- `reports/*.csv` - License ledger

## Troubleshooting

**"Command not found"**
- Make sure you're in the `pretrain-mini/` directory
- Run `make setup` first

**"Out of memory"**
- Use `make smoke` instead of `make run-all`
- The smoke test uses tiny samples

**"Training too slow"**
- If you have GPU: it will be used automatically
- If CPU only: expect longer training times

## Customization

### Add Your Own Data
Edit `src/ingest_web.py`:
```python
WEB_SOURCES.append({
    'url': 'https://example.com/your-file.txt',
    'name': 'My Data',
    'license': 'Public Domain'
})
```

### Change Filter Thresholds
```bash
# More strict toxicity
python src/toxicity.py --threshold 0.5

# Different mixture ratio
python src/mixture_build.py --ratio-web 0.8 --ratio-code 0.2
```

## What to Show in Interviews

1. **Pipeline Design**: Show the 13-step modular workflow
2. **Metrics**: Point to `reports/FINAL_REPORT.md`
3. **Code Quality**: Show clean, documented modules
4. **Impact**: Show perplexity improvement (baseline vs curated)
5. **Responsibility**: Explain toxicity, PII, license checks

## Next Level

### Scale Up
- Use Common Crawl data
- Add The Stack for code
- Train larger models
- More training epochs

### Add Features
- More evaluation metrics
- Better contamination detection
- Additional quality signals
- Ablation studies

## Need Help?

1. Check `README.md` for full documentation
2. Look at `reports/FINAL_REPORT.md` after running
3. Review individual `src/*.py` files (well commented)
4. Check the makefile for all available commands

---

**You're ready! Run `make smoke` now!** ⚡
