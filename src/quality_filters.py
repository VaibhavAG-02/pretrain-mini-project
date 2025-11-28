#!/usr/bin/env python3
"""
Step 3: Quality Filters
Applies heuristic and model-based quality filters.
"""

import argparse
import re
from pathlib import Path
import polars as pl
from tqdm import tqdm


def calculate_char_stats(text: str) -> dict:
    """Calculate character-level statistics."""
    if not text:
        return {
            'alnum_ratio': 0.0,
            'digit_ratio': 0.0,
            'upper_ratio': 0.0,
            'space_ratio': 0.0,
            'special_ratio': 0.0
        }
    
    total = len(text)
    return {
        'alnum_ratio': sum(c.isalnum() for c in text) / total,
        'digit_ratio': sum(c.isdigit() for c in text) / total,
        'upper_ratio': sum(c.isupper() for c in text) / total,
        'space_ratio': sum(c.isspace() for c in text) / total,
        'special_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / total
    }


def count_urls(text: str) -> int:
    """Count URLs in text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return len(re.findall(url_pattern, text))


def avg_word_length(text: str) -> float:
    """Calculate average word length."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def quality_filter(text: str) -> tuple:
    """
    Apply quality filters and return (passed, reason).
    """
    # Word-based checks
    words = text.split()
    num_words = len(words)
    
    if num_words < 10:
        return False, 'too_few_words'
    
    # Average word length (detect gibberish)
    avg_len = avg_word_length(text)
    if avg_len < 2 or avg_len > 20:
        return False, 'abnormal_word_length'
    
    # Character statistics
    stats = calculate_char_stats(text)
    
    # Too many special characters
    if stats['special_ratio'] > 0.3:
        return False, 'too_many_special_chars'
    
    # Too many digits (likely spam or codes)
    if stats['digit_ratio'] > 0.5:
        return False, 'too_many_digits'
    
    # Too many uppercase (likely spam)
    if stats['upper_ratio'] > 0.5 and len(text) > 100:
        return False, 'excessive_uppercase'
    
    # URL spam
    url_count = count_urls(text)
    if url_count > 5:
        return False, 'too_many_urls'
    
    # All checks passed
    return True, 'passed'


def main():
    parser = argparse.ArgumentParser(description='Apply quality filters')
    parser.add_argument('--input', type=str, 
                        default='data/processed/01_language_filtered.parquet',
                        help='Input parquet file')
    parser.add_argument('--output', type=str,
                        default='data/processed/02_quality_filtered.parquet',
                        help='Output parquet file')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pl.read_parquet(args.input)
    initial_count = len(df)
    
    print(f"Initial documents: {initial_count}")
    print("Applying quality filters...")
    
    # Apply filters
    passed_list = []
    reasons = []
    
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        text = row.get('text_normalized', row.get('text', ''))
        passed, reason = quality_filter(text)
        passed_list.append(passed)
        reasons.append(reason)
    
    # Add columns
    df = df.with_columns([
        pl.Series('quality_passed', passed_list),
        pl.Series('quality_reason', reasons)
    ])
    
    # Filter
    df_filtered = df.filter(pl.col('quality_passed'))
    final_count = len(df_filtered)
    retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.write_parquet(args.output)
    
    # Statistics
    print(f"\n📊 Quality Filter Results:")
    print(f"   Initial: {initial_count}")
    print(f"   Passed: {final_count}")
    print(f"   Retention: {retention_rate:.1f}%")
    
    print(f"\n   Failure reasons:")
    reason_counts = df.group_by('quality_reason').agg(pl.count().alias('count')).sort('count', descending=True)
    for row in reason_counts.iter_rows(named=True):
        if row['quality_reason'] != 'passed':
            print(f"      {row['quality_reason']}: {row['count']}")
    
    print(f"\n✅ Saved to {args.output}")


if __name__ == '__main__':
    main()
