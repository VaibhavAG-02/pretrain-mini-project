#!/usr/bin/env python3
"""Step 11: Evaluation Harness"""
import argparse, json
from pathlib import Path
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_from_disk

def compute_perplexity(model, tokenizer, dataset, device='cpu'):
    model.eval()
    model.to(device)
    
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for item in dataset:
            inputs = tokenizer(item['text'], return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item()
            count += 1
    
    avg_loss = total_loss / count
    perplexity = np.exp(avg_loss)
    return perplexity, avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='models/baseline/final')
    parser.add_argument('--curated', default='models/curated/final')
    parser.add_argument('--data', default='data/shards/final_dataset')
    args = parser.parse_args()
    
    print("Evaluating models...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_from_disk(args.data)['validation']
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    # Evaluate baseline
    if Path(args.baseline).exists():
        print("\nEvaluating BASELINE model...")
        model_baseline = GPT2LMHeadModel.from_pretrained(args.baseline)
        ppl_baseline, loss_baseline = compute_perplexity(model_baseline, tokenizer, dataset, device)
        results['baseline'] = {'perplexity': float(ppl_baseline), 'loss': float(loss_baseline)}
        print(f"   Perplexity: {ppl_baseline:.2f}")
        print(f"   Loss: {loss_baseline:.4f}")
    
    # Evaluate curated
    if Path(args.curated).exists():
        print("\nEvaluating CURATED model...")
        model_curated = GPT2LMHeadModel.from_pretrained(args.curated)
        ppl_curated, loss_curated = compute_perplexity(model_curated, tokenizer, dataset, device)
        results['curated'] = {'perplexity': float(ppl_curated), 'loss': float(loss_curated)}
        print(f"   Perplexity: {ppl_curated:.2f}")
        print(f"   Loss: {loss_curated:.4f}")
    
    # Compare
    if 'baseline' in results and 'curated' in results:
        improvement = (results['baseline']['perplexity'] - results['curated']['perplexity']) / results['baseline']['perplexity'] * 100
        results['improvement_pct'] = float(improvement)
        print(f"\n📊 Comparison:")
        print(f"   Improvement: {improvement:.2f}%")
        print(f"   {'✅ PASS' if improvement > 0 else '❌ FAIL'}: Curated model {'better' if improvement > 0 else 'worse'}")
    
    # Save
    Path('reports').mkdir(exist_ok=True)
    with open('reports/eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
