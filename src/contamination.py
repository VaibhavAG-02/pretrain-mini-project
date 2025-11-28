#!/usr/bin/env python3
"""Step 9: Contamination Checks"""
import argparse
from pathlib import Path

EVAL_SAMPLES = [
    "The quick brown fox jumps over the lazy dog",
    "To be or not to be, that is the question",
]

def get_ngrams(text, n=8):
    words = text.lower().split()
    return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/07_mixture.parquet')
    parser.add_argument('--output', default='data/processed/08_clean.parquet')
    args = parser.parse_args()
    
    import polars as pl
    df = pl.read_parquet(args.input)
    
    print(f"Checking {len(df)} documents for contamination...")
    
    eval_ngrams = set()
    for sample in EVAL_SAMPLES:
        eval_ngrams.update(get_ngrams(sample))
    
    contaminated = 0
    clean_indices = []
    
    for idx, row in enumerate(df.to_dicts()):
        text = row.get('text_clean', row['text'])
        corpus_ngrams = get_ngrams(text)
        if not (corpus_ngrams & eval_ngrams):
            clean_indices.append(idx)
        else:
            contaminated += 1
    
    df_clean = pl.DataFrame([df.to_dicts()[i] for i in clean_indices])
    
    print(f"\n📊 Contamination Check:")
    print(f"   Total: {len(df)}")
    print(f"   Contaminated: {contaminated}")
    print(f"   Clean: {len(df_clean)}")
    print(f"   Rate: {contaminated/len(df)*100:.4f}%")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(args.output)
    print(f"✅ Saved {len(df_clean)} clean documents")

if __name__ == '__main__':
    main()
