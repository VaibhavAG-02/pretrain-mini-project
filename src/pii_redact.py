#!/usr/bin/env python3
"""Step 4: PII Redaction"""
import argparse, re
from pathlib import Path
import polars as pl
from tqdm import tqdm

def redact_pii(text):
    # Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Phone
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/03_toxicity_filtered.parquet')
    parser.add_argument('--output', default='data/processed/04_pii_redacted.parquet')
    args = parser.parse_args()
    
    df = pl.read_parquet(args.input)
    print(f"Processing {len(df)} documents for PII...")
    
    cleaned = []
    had_pii = []
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        text = row.get('text_normalized', row['text'])
        clean = redact_pii(text)
        cleaned.append(clean)
        had_pii.append(clean != text)
    
    df = df.with_columns([
        pl.Series('text_clean', cleaned),
        pl.Series('had_pii', had_pii)
    ])
    
    print(f"✅ PII redacted in {sum(had_pii)} documents ({sum(had_pii)/len(had_pii)*100:.1f}%)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)

if __name__ == '__main__':
    main()
