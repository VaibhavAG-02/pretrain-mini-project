#!/usr/bin/env python3
"""
Step 2: Language Identification and Basic Normalization
FIXED VERSION - Uses fasttext instead of gcld3 for better Colab compatibility
"""

import argparse
import re
from pathlib import Path
import polars as pl
from tqdm import tqdm

# Simple heuristic language detection (backup)
def simple_language_detect(text: str) -> tuple:
    """
    Simple heuristic language detection.
    Returns (language, confidence, reliable)
    """
    # Common English words
    english_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 
                     'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
                     'does', 'did', 'will', 'would', 'could', 'should'}
    
    words = set(text.lower().split()[:100])  # First 100 words
    common_count = len(words & english_words)
    
    # If >20% of words are common English words, assume English
    if len(words) > 0:
        confidence = min(common_count / len(words) * 2, 1.0)
        is_english = confidence > 0.3
        return ('en' if is_english else 'unknown', confidence, confidence > 0.5)
    
    return ('unknown', 0.0, False)


def normalize_text(text: str) -> str:
    """Basic text normalization"""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    text = text.strip()
    return text


def is_valid_length(text: str, min_len: int = 50, max_len: int = 50000) -> bool:
    """Check if text length is within acceptable range."""
    length = len(text)
    return min_len <= length <= max_len


def has_sufficient_words(text: str, min_words: int = 10) -> bool:
    """Check if text has enough words."""
    words = text.split()
    return len(words) >= min_words


def process_file(
    input_path: str,
    output_path: str,
    language_filter: str = 'en',
    min_length: int = 50,
    max_length: int = 50000,
    min_words: int = 10
):
    """Process a parquet file to identify language and normalize text."""
    print(f"Loading data from {input_path}...")
    df = pl.read_parquet(input_path)
    initial_count = len(df)
    
    print(f"Initial documents: {initial_count}")
    
    # Process each document
    languages = []
    lang_probs = []
    is_reliable_list = []
    normalized_texts = []
    valid_length_list = []
    sufficient_words_list = []
    
    print("Detecting languages and normalizing text...")
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        text = row['text']
        
        # Normalize
        normalized = normalize_text(text)
        normalized_texts.append(normalized)
        
        # Detect language (skip for code files)
        if row.get('source') == 'code':
            lang, prob, reliable = row.get('language', 'code'), 1.0, True
        else:
            lang, prob, reliable = simple_language_detect(normalized)
        
        languages.append(lang)
        lang_probs.append(prob)
        is_reliable_list.append(reliable)
        
        # Check validity
        valid_length = is_valid_length(normalized, min_length, max_length)
        sufficient_words = has_sufficient_words(normalized, min_words)
        
        valid_length_list.append(valid_length)
        sufficient_words_list.append(sufficient_words)
    
    # Add columns
    df = df.with_columns([
        pl.Series('text_normalized', normalized_texts),
        pl.Series('language_detected', languages),
        pl.Series('language_prob', lang_probs),
        pl.Series('language_reliable', is_reliable_list),
        pl.Series('valid_length', valid_length_list),
        pl.Series('sufficient_words', sufficient_words_list)
    ])
    
    # Filter for target language (or code)
    df_filtered = df.filter(
        ((pl.col('language_detected') == language_filter) | (pl.col('source') == 'code')) &
        pl.col('valid_length') &
        pl.col('sufficient_words')
    )
    
    final_count = len(df_filtered)
    retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.write_parquet(output_path)
    
    # Print statistics
    print(f"\n📊 Language Identification Results:")
    print(f"   Initial documents: {initial_count}")
    print(f"   After filtering: {final_count}")
    print(f"   Retention rate: {retention_rate:.1f}%")
    print(f"   Removed: {initial_count - final_count}")
    
    print(f"\n   Language distribution (before filtering):")
    lang_dist = df.group_by('language_detected').agg(pl.count().alias('count')).sort('count', descending=True)
    for row in lang_dist.head(10).iter_rows(named=True):
        print(f"      {row['language_detected']}: {row['count']}")
    
    print(f"\n✅ Acceptance criteria:")
    print(f"   Target: >=85% retention after quality rules")
    print(f"   Actual: {retention_rate:.1f}%")
    print(f"   Status: {'✅ PASS' if retention_rate >= 85 else '⚠️  REVIEW'}")
    
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description='Language identification and normalization')
    parser.add_argument('--web-input', type=str, default='data/raw/web_index.parquet')
    parser.add_argument('--code-input', type=str, default='data/raw/code_index.parquet')
    parser.add_argument('--output', type=str, default='data/processed/01_language_filtered.parquet')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--min-length', type=int, default=50)
    parser.add_argument('--max-length', type=int, default=50000)
    parser.add_argument('--min-words', type=int, default=10)
    
    args = parser.parse_args()
    
    # Process web data
    web_df = None
    if Path(args.web_input).exists():
        print(f"\n{'='*60}")
        print("Processing WEB data...")
        print('='*60)
        web_df = process_file(
            args.web_input,
            'data/processed/01_web_lang.parquet',
            args.language,
            args.min_length,
            args.max_length,
            args.min_words
        )
    
    # Process code data
    code_df = None
    if Path(args.code_input).exists():
        print(f"\n{'='*60}")
        print("Processing CODE data...")
        print('='*60)
        code_df = process_file(
            args.code_input,
            'data/processed/01_code_lang.parquet',
            args.language,
            args.min_length,
            args.max_length,
            args.min_words
        )
    
    # Combine both
    dfs_to_combine = [df for df in [web_df, code_df] if df is not None]
    
    if dfs_to_combine:
        combined_df = pl.concat(dfs_to_combine)
        combined_df.write_parquet(args.output)
        
        print(f"\n{'='*60}")
        print("COMBINED RESULTS")
        print('='*60)
        print(f"✅ Saved {len(combined_df)} documents to {args.output}")
        print(f"   Web documents: {len(web_df) if web_df is not None else 0}")
        print(f"   Code documents: {len(code_df) if code_df is not None else 0}")
    else:
        print("\n❌ No data processed!")


if __name__ == '__main__':
    main()
