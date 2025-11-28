#!/usr/bin/env python3
"""Step 8: Sharding for distributed loading"""
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/07_mixture.parquet')
    parser.add_argument('--output-dir', default='data/shards/final_dataset')
    parser.add_argument('--num-shards', type=int, default=10)
    args = parser.parse_args()
    
    import polars as pl
    df = pl.read_parquet(args.input)
    print(f"Sharding {len(df)} documents...")
    
    # Convert to HF datasets format
    data = [{'text': row.get('text_clean', row['text'])} for row in df.to_dicts()]
    
    # Split train/val
    split_point = int(len(data) * 0.9)
    train_data = data[:split_point]
    val_data = data[split_point:]
    
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data)
    })
    
    # Save
    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)
    
    print(f"\n✅ Sharded dataset saved to {args.output_dir}")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")

if __name__ == '__main__':
    main()
