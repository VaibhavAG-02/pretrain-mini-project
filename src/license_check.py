#!/usr/bin/env python3
"""Step 5: License Verification"""
import argparse, json
from pathlib import Path
import polars as pl

ALLOWED_LICENSES = {'MIT', 'Apache-2.0', 'Public Domain', 'BSD-3-Clause'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/04_pii_redacted.parquet')
    parser.add_argument('--output', default='data/processed/05_license_verified.parquet')
    args = parser.parse_args()
    
    df = pl.read_parquet(args.input)
    print(f"Checking licenses for {len(df)} documents...")
    
    df = df.with_columns([
        pl.col('license').is_in(list(ALLOWED_LICENSES)).alias('license_allowed')
    ])
    
    df_filtered = df.filter(pl.col('license_allowed'))
    
    print(f"\n📊 License Check:")
    print(f"   Before: {len(df)}")
    print(f"   After: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")
    
    license_dist = df_filtered.group_by('license').agg(pl.count().alias('count'))
    print("\n   License distribution:")
    for row in license_dist.iter_rows(named=True):
        print(f"      {row['license']}: {row['count']}")
    
    # Save license ledger
    license_dist.write_csv('reports/license_ledger.csv')
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.write_parquet(args.output)
    print(f"✅ Saved to {args.output}")

if __name__ == '__main__':
    main()
