#!/usr/bin/env python3
"""
Step 1a: Web Data Ingestion
Downloads web text from public domain sources and stores with content hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import polars as pl

# Public domain sources (Project Gutenberg)
WEB_SOURCES = [
    {
        'url': 'https://www.gutenberg.org/files/1342/1342-0.txt',
        'name': 'Pride and Prejudice',
        'license': 'Public Domain',
        'author': 'Jane Austen'
    },
    {
        'url': 'https://www.gutenberg.org/files/11/11-0.txt',
        'name': 'Alice in Wonderland',
        'license': 'Public Domain',
        'author': 'Lewis Carroll'
    },
    {
        'url': 'https://www.gutenberg.org/files/84/84-0.txt',
        'name': 'Frankenstein',
        'license': 'Public Domain',
        'author': 'Mary Shelley'
    },
    {
        'url': 'https://www.gutenberg.org/files/1661/1661-0.txt',
        'name': 'Sherlock Holmes',
        'license': 'Public Domain',
        'author': 'Arthur Conan Doyle'
    },
    {
        'url': 'https://www.gutenberg.org/files/1952/1952-0.txt',
        'name': 'The Yellow Wallpaper',
        'license': 'Public Domain',
        'author': 'Charlotte Perkins Gilman'
    },
]


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def clean_html(content: str) -> str:
    """Remove HTML tags and extract clean text."""
    soup = BeautifulSoup(content, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style", "meta", "link"]):
        script.decompose()
    text = soup.get_text()
    # Clean whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def download_source(source: Dict, timeout: int = 30) -> str:
    """Download text from a single source URL."""
    try:
        response = requests.get(source['url'], timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        text = response.text
        
        # Clean HTML if needed
        if 'html' in content_type.lower():
            text = clean_html(text)
        
        return text
    except Exception as e:
        print(f"Error downloading {source['url']}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) < chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only keep substantial chunks
        if len(chunk.strip()) >= 100:
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks


def guess_language(text: str) -> str:
    """Simple heuristic language guess (will be refined in language_id.py)."""
    # For now, assume English from Project Gutenberg
    return 'en'


def ingest_web_data(sample_size: int = None) -> List[Dict]:
    """
    Main ingestion function for web text.
    
    Args:
        sample_size: If set, limit to first N sources (for testing)
    
    Returns:
        List of document dictionaries
    """
    sources = WEB_SOURCES[:sample_size] if sample_size else WEB_SOURCES
    documents = []
    
    print(f"Ingesting {len(sources)} web sources...")
    
    for source in tqdm(sources, desc="Downloading sources"):
        text = download_source(source)
        if not text:
            continue
        
        # Split into chunks
        chunks = chunk_text(text)
        
        for idx, chunk in enumerate(chunks):
            doc = {
                'text': chunk,
                'source': 'web',
                'url': source['url'],
                'name': source['name'],
                'author': source.get('author', 'Unknown'),
                'license': source['license'],
                'sha256': compute_hash(chunk),
                'bytes': len(chunk.encode('utf-8')),
                'chunk_id': idx,
                'total_chunks': len(chunks),
                'lang_guess': guess_language(chunk)
            }
            documents.append(doc)
    
    return documents


def save_raw_index(documents: List[Dict], output_path: str = 'data/raw/web_index.parquet'):
    """Save raw document index to Parquet file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df = pl.DataFrame(documents)
    df.write_parquet(output_path)
    
    print(f"\n✅ Saved {len(documents)} web documents to {output_path}")
    print(f"   Total bytes: {df['bytes'].sum():,}")
    print(f"   Unique sources: {df['name'].n_unique()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Ingest web text data')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Limit to first N sources (for testing)')
    parser.add_argument('--output', type=str, default='data/raw/web_index.parquet',
                        help='Output parquet file path')
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Size of text chunks in characters')
    
    args = parser.parse_args()
    
    # Ingest data
    documents = ingest_web_data(sample_size=args.sample_size)
    
    if not documents:
        print("❌ No documents ingested. Check your sources.")
        return
    
    # Save to parquet
    df = save_raw_index(documents, args.output)
    
    # Print summary statistics
    print(f"\n📊 Ingestion Summary:")
    print(f"   Documents: {len(df)}")
    print(f"   Sources: {df['name'].n_unique()}")
    print(f"   Total size: {df['bytes'].sum() / 1024 / 1024:.2f} MB")
    print(f"   Avg chunk size: {df['bytes'].mean():.0f} bytes")
    
    # Save metadata
    metadata = {
        'num_documents': len(df),
        'num_sources': df['name'].n_unique(),
        'total_bytes': int(df['bytes'].sum()),
        'sources': [s['name'] for s in (WEB_SOURCES[:args.sample_size] if args.sample_size else WEB_SOURCES)]
    }
    
    metadata_path = Path(args.output).parent / 'web_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Web data ingestion complete!")


if __name__ == '__main__':
    main()
