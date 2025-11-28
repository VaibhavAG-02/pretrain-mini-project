#!/usr/bin/env python3
"""Step 3: Toxicity Detection"""
import argparse
from pathlib import Path
import polars as pl
from detoxify import Detoxify
from tqdm import tqdm

TOXICITY_THRESHOLD = 0.7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/02_quality_filtered.parquet')
    parser.add_argument('--output', default='data/processed/03_toxicity_filtered.parquet')
    parser.add_argument('--threshold', type=float, default=TOXICITY_THRESHOLD)
    args = parser.parse_args()
    
    df = pl.read_parquet(args.input)
    print(f"Loaded {len(df)} documents")
    
    model = Detoxify('original')
    scores = []
    
    print("Scoring toxicity...")
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        text = row.get('text_normalized', row['text'])[:1000]
        try:
            result = model.predict(text)
            scores.append(result['toxicity'])
        except:
            scores.append(0.0)
    
    df = df.with_columns([pl.Series('toxicity_score', scores)])
    df_filtered = df.filter(pl.col('toxicity_score') < args.threshold)
    
    print(f"\n📊 Toxicity Results:")
    print(f"   Initial: {len(df)}")
    print(f"   Filtered: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")
    print(f"   Mean toxicity: {sum(scores)/len(scores):.4f}")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.write_parquet(args.output)
    print(f"✅ Saved to {args.output}")

if __name__ == '__main__':
    main()
