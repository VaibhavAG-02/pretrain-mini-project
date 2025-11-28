#!/usr/bin/env python3
"""Step 7: Mixture Design and Balancing"""
import argparse, json
from pathlib import Path
from datetime import datetime
import polars as pl

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/06_deduplicated.parquet')
    parser.add_argument('--output', default='data/processed/07_mixture.parquet')
    parser.add_argument('--ratio-web', type=float, default=0.70)
    parser.add_argument('--ratio-code', type=float, default=0.30)
    args = parser.parse_args()
    
    df = pl.read_parquet(args.input)
    print(f"Building mixture from {len(df)} documents...")
    
    web_count = len(df.filter(pl.col('source') == 'web'))
    code_count = len(df.filter(pl.col('source') == 'code'))
    
    print(f"\nCurrent distribution:")
    print(f"   Web: {web_count} ({web_count/(web_count+code_count)*100:.1f}%)")
    print(f"   Code: {code_count} ({code_count/(web_count+code_count)*100:.1f}%)")
    
    # Calculate target
    total_target = min(web_count, int(code_count / args.ratio_code))
    target_web = int(total_target * args.ratio_web)
    target_code = int(total_target * args.ratio_code)
    
    print(f"\nTarget distribution:")
    print(f"   Web: {target_web} (70%)")
    print(f"   Code: {target_code} (30%)")
    
    # Sample
    df_web = df.filter(pl.col('source') == 'web').sample(n=min(target_web, web_count), seed=SEED)
    df_code = df.filter(pl.col('source') == 'code').sample(n=min(target_code, code_count), seed=SEED)
    
    # Combine and shuffle
    df_mixture = pl.concat([df_web, df_code]).sample(fraction=1.0, seed=SEED)
    
    print(f"\n✅ Final mixture: {len(df_mixture)} documents")
    print(f"   Web: {len(df_web)} ({len(df_web)/len(df_mixture)*100:.1f}%)")
    print(f"   Code: {len(df_code)} ({len(df_code)/len(df_mixture)*100:.1f}%)")
    
    # Save manifest
    manifest = {
        'version': f'v1.0_{datetime.now().strftime("%Y%m%d")}',
        'seed': SEED,
        'ratios': {'web': args.ratio_web, 'code': args.ratio_code},
        'counts': {'web': len(df_web), 'code': len(df_code), 'total': len(df_mixture)}
    }
    Path('reports').mkdir(exist_ok=True)
    with open('reports/mixture_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_mixture.write_parquet(args.output)

if __name__ == '__main__':
    main()
