#!/usr/bin/env python3
"""Step 10b: Train Curated Model (filtered)"""
import argparse, json
from pathlib import Path
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_from_disk

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/shards/final_dataset')
    parser.add_argument('--output', default='models/curated')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    
    print("Training CURATED model (filtered data)...")
    
    dataset = load_from_disk(args.data)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Tokenizing...")
    tokenized_train = dataset['train'].map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True, remove_columns=['text']
    )
    tokenized_val = dataset['validation'].map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True, remove_columns=['text']
    )
    
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )
    model = GPT2LMHeadModel(config)
    print(f"Model parameters: {model.num_parameters():,}")
    
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy='steps' if not args.fast else 'no',
        eval_steps=50 if not args.fast else None,
        save_steps=100,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("\n🚀 Training...")
    result = trainer.train()
    
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    
    metrics = {
        'train_loss': float(result.training_loss),
        'model': 'curated'
    }
    Path('reports').mkdir(exist_ok=True)
    with open('reports/curated_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Curated model trained!")
    print(f"   Final loss: {result.training_loss:.4f}")

if __name__ == '__main__':
    main()
