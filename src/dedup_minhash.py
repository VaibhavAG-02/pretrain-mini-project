#!/usr/bin/env python3
"""Step 6: Deduplication using MinHash + LSH"""
import argparse, json
from pathlib import Path
from collections import defaultdict
import polars as pl
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    shingles = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    for shingle in shingles:
        m.update(shingle.encode('utf8'))
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/05_license_verified.parquet')
    parser.add_argument('--output', default='data/processed/06_deduplicated.parquet')
    parser.add_argument('--threshold', type=float, default=0.85)
    args = parser.parse_args()
    
    df = pl.read_parquet(args.input)
    print(f"Deduplicating {len(df)} documents...")
    
    lsh = MinHashLSH(threshold=args.threshold, num_perm=128)
    sha256_seen = set()
    minhashes = {}
    near_dupe_groups = defaultdict(list)
    exact_dupes = 0
    
    items = df.to_dicts()
    for idx, item in enumerate(tqdm(items, desc="Computing MinHash")):
        sha = item['sha256']
        if sha in sha256_seen:
            exact_dupes += 1
            continue
        sha256_seen.add(sha)
        
        text = item.get('text_clean', item['text'])
        m = create_minhash(text)
        minhashes[idx] = m
        
        result = lsh.query(m)
        if result:
            near_dupe_groups[result[0]].append(idx)
        else:
            lsh.insert(idx, m)
            near_dupe_groups[idx].append(idx)
    
    unique_indices = [min(group) for group in near_dupe_groups.values()]
    near_dupes = len(minhashes) - len(unique_indices)
    
    print(f"\n📊 Deduplication Results:")
    print(f"   Exact duplicates: {exact_dupes}")
    print(f"   Near-duplicates: {near_dupes}")
    print(f"   Unique: {len(unique_indices)}")
    print(f"   Total removed: {exact_dupes + near_dupes}")
    
    df_deduped = pl.DataFrame([items[i] for i in unique_indices])
    
    # Save stats
    stats = {
        'exact_duplicates': exact_dupes,
        'near_duplicates': near_dupes,
        'unique_items': len(unique_indices),
        'total_removed': exact_dupes + near_dupes
    }
    Path('reports').mkdir(exist_ok=True)
    with open('reports/dedup_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_deduped.write_parquet(args.output)
    print(f"✅ Saved {len(df_deduped)} unique documents")

if __name__ == '__main__':
    main()
