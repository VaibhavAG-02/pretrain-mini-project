#!/usr/bin/env python3
"""
Step 1b: Code Data Ingestion
Collects permissively-licensed code samples (MIT, Apache-2.0)
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict
import polars as pl

# Sample code snippets with explicit licenses
CODE_SAMPLES = [
    {
        'code': '''# Bubble Sort - Classic sorting algorithm
def bubble_sort(arr):
    """Sort array using bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Test
if __name__ == "__main__":
    data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {data}")
    sorted_data = bubble_sort(data.copy())
    print(f"Sorted: {sorted_data}")
''',
        'license': 'MIT',
        'language': 'python',
        'name': 'bubble_sort.py',
        'description': 'Bubble sort algorithm implementation'
    },
    {
        'code': '''// Binary Search - Efficient search algorithm
function binarySearch(arr, target) {
    /**
     * Search for target in sorted array using binary search
     * Time complexity: O(log n)
     */
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1; // Not found
}

// Example usage
const sortedArray = [1, 3, 5, 7, 9, 11, 13, 15];
const target = 7;
console.log(`Index of ${target}: ${binarySearch(sortedArray, target)}`);
''',
        'license': 'Apache-2.0',
        'language': 'javascript',
        'name': 'binary_search.js',
        'description': 'Binary search algorithm'
    },
    {
        'code': '''# Quick Sort - Divide and conquer sorting
def quick_sort(arr):
    """
    Sort array using quicksort algorithm.
    Average time complexity: O(n log n)
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def partition(arr, low, high):
    """Helper function for in-place quicksort."""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Example
numbers = [3, 6, 8, 10, 1, 2, 1]
print(f"Unsorted: {numbers}")
print(f"Sorted: {quick_sort(numbers)}")
''',
        'license': 'MIT',
        'language': 'python',
        'name': 'quick_sort.py',
        'description': 'Quicksort implementation'
    },
    {
        'code': '''# Linked List - Basic data structure
class Node:
    """Node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """Singly linked list implementation."""
    
    def __init__(self):
        self.head = None
    
    def append(self, data):
        """Add node to end of list."""
        new_node = Node(data)
        
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def display(self):
        """Print all elements in list."""
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(f"Linked list: {ll.display()}")
''',
        'license': 'Apache-2.0',
        'language': 'python',
        'name': 'linked_list.py',
        'description': 'Linked list data structure'
    },
    {
        'code': '''// Hash Table - Key-value data structure
class HashTable {
    /**
     * Simple hash table implementation
     * Uses chaining for collision resolution
     */
    constructor(size = 53) {
        this.keyMap = new Array(size);
    }
    
    _hash(key) {
        let total = 0;
        const PRIME = 31;
        for (let i = 0; i < Math.min(key.length, 100); i++) {
            const char = key[i];
            const value = char.charCodeAt(0) - 96;
            total = (total * PRIME + value) % this.keyMap.length;
        }
        return total;
    }
    
    set(key, value) {
        const index = this._hash(key);
        if (!this.keyMap[index]) {
            this.keyMap[index] = [];
        }
        this.keyMap[index].push([key, value]);
    }
    
    get(key) {
        const index = this._hash(key);
        if (this.keyMap[index]) {
            for (let pair of this.keyMap[index]) {
                if (pair[0] === key) {
                    return pair[1];
                }
            }
        }
        return undefined;
    }
}

// Example
const ht = new HashTable();
ht.set("name", "Alice");
ht.set("age", 30);
console.log(ht.get("name")); // Alice
''',
        'license': 'MIT',
        'language': 'javascript',
        'name': 'hash_table.js',
        'description': 'Hash table with chaining'
    },
]


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of code content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def detect_language_from_extension(filename: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby'
    }
    ext = Path(filename).suffix.lower()
    return ext_map.get(ext, 'unknown')


def ingest_code_data(sample_size: int = None) -> List[Dict]:
    """
    Main ingestion function for code samples.
    
    Args:
        sample_size: If set, limit to first N samples (for testing)
    
    Returns:
        List of code document dictionaries
    """
    samples = CODE_SAMPLES[:sample_size] if sample_size else CODE_SAMPLES
    documents = []
    
    print(f"Ingesting {len(samples)} code samples...")
    
    for idx, sample in enumerate(samples):
        doc = {
            'text': sample['code'],
            'source': 'code',
            'url': f'synthetic_code_{idx}',
            'name': sample['name'],
            'description': sample['description'],
            'license': sample['license'],
            'language': sample['language'],
            'sha256': compute_hash(sample['code']),
            'bytes': len(sample['code'].encode('utf-8')),
            'chunk_id': 0,
            'total_chunks': 1,
            'lang_guess': sample['language']
        }
        documents.append(doc)
    
    print(f"✅ Collected {len(documents)} code documents")
    
    return documents


def save_raw_index(documents: List[Dict], output_path: str = 'data/raw/code_index.parquet'):
    """Save raw code index to Parquet file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df = pl.DataFrame(documents)
    df.write_parquet(output_path)
    
    print(f"\n✅ Saved {len(documents)} code documents to {output_path}")
    print(f"   Total bytes: {df['bytes'].sum():,}")
    print(f"   Languages: {', '.join(df['language'].unique().to_list())}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Ingest code data')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Limit to first N samples (for testing)')
    parser.add_argument('--output', type=str, default='data/raw/code_index.parquet',
                        help='Output parquet file path')
    
    args = parser.parse_args()
    
    # Ingest data
    documents = ingest_code_data(sample_size=args.sample_size)
    
    if not documents:
        print("❌ No code documents ingested.")
        return
    
    # Save to parquet
    df = save_raw_index(documents, args.output)
    
    # Print summary statistics
    print(f"\n📊 Code Ingestion Summary:")
    print(f"   Documents: {len(df)}")
    print(f"   Languages: {df['language'].n_unique()}")
    print(f"   Licenses: {', '.join(df['license'].unique().to_list())}")
    print(f"   Total size: {df['bytes'].sum() / 1024:.2f} KB")
    print(f"   Avg size: {df['bytes'].mean():.0f} bytes")
    
    # License distribution
    print(f"\n📋 License Distribution:")
    license_counts = df.group_by('license').agg(pl.count().alias('count'))
    for row in license_counts.iter_rows(named=True):
        print(f"   {row['license']}: {row['count']} files")
    
    # Save metadata
    metadata = {
        'num_documents': len(df),
        'num_languages': df['language'].n_unique(),
        'total_bytes': int(df['bytes'].sum()),
        'licenses': df['license'].unique().to_list(),
        'languages': df['language'].unique().to_list()
    }
    
    metadata_path = Path(args.output).parent / 'code_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Code data ingestion complete!")


if __name__ == '__main__':
    main()
