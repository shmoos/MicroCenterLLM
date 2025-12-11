"""
Train GPT-2 style model from scratch - OPTIMIZED FOR M1 PRO
Automatically scales model size based on data and available RAM
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast 
from datasets import load_dataset 
import os
import psutil

# Constants
TOKENIZER_PATH = "tokenizer/tokenizer.json"
DATA_FILE = "data.txt"
MODEL_OUTPUT_DIR = "./model"
MODEL_FINAL_DIR = "./model/final"
LOGS_DIR = "./logs"

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(" Using MPS (Metal)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(" Using CUDA")
else:
    device = torch.device("cpu")
    print(" Using CPU (this will be slow)")

print("\n" + "="*70)
print(" TRAINING GPT-2 MODEL FROM SCRATCH (M1 PRO OPTIMIZED)")
print("="*70)

# Check system RAM
total_ram_gb = psutil.virtual_memory().total / (1024**3)
available_ram_gb = psutil.virtual_memory().available / (1024**3)
print(f"\n System RAM: {total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available")

# Check data size and recommend model
if os.path.exists(DATA_FILE):
    data_size = os.path.getsize(DATA_FILE)
    print(f" Data file: {data_size/1024:.1f} KB ({data_size/(1024*1024):.2f} MB)")
    
    # Smart model sizing based on BOTH data and RAM
    if total_ram_gb >= 30:  # 32GB Mac
        if data_size < 200_000:
            n_embd, n_layer, n_head = 384, 8, 6
            batch_size, grad_accum = 8, 2
            model_name = "SMALL"
        elif data_size < 600_000:
            n_embd, n_layer, n_head = 512, 10, 8
            batch_size, grad_accum = 6, 3
            model_name = "MEDIUM"
        else:
            n_embd, n_layer, n_head = 640, 12, 10
            batch_size, grad_accum = 4, 4
            model_name = "LARGE"
    else:  # 16GB Mac (be conservative)
        if data_size < 200_000:
            n_embd, n_layer, n_head = 256, 6, 4
            batch_size, grad_accum = 4, 4
            model_name = "TINY"
            print("  Small dataset + 16GB RAM: Using TINY model")
        elif data_size < 600_000:
            n_embd, n_layer, n_head = 384, 8, 6
            batch_size, grad_accum = 4, 4
            model_name = "SMALL"
        else:
            n_embd, n_layer, n_head = 512, 10, 8
            batch_size, grad_accum = 2, 8
            model_name = "MEDIUM"
            print("  Large dataset on 16GB: Using smaller batch size")
else:
    print(f" ERROR: {DATA_FILE} not found!")
    exit(1)

print(f" Model config: {model_name}")
print(f"   Embedding: {n_embd}, Layers: {n_layer}, Heads: {n_head}")
print(f"   Batch size: {batch_size}, Gradient accumulation: {grad_accum}")
print(f"   Effective batch size: {batch_size * grad_accum}")

# Load tokenizer
print("\n Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
})
print(f"✓ Vocab size: {len(tokenizer):,}")

# Load dataset
print("\n Loading dataset...")
dataset = load_dataset("text", data_files={"train": DATA_FILE})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("⚙️  Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
print(f"✓ Dataset size: {len(dataset['train'])} examples")

# Model configuration
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

print("\n  Initializing model...")
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

total_params = model.num_parameters()
print(f"✓ Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# Estimate memory usage
param_memory_mb = (total_params * 4) / (1024**2)  # 4 bytes per param (float32)
optimizer_memory_mb = param_memory_mb * 2  # Adam optimizer
activation_memory_mb = 200  # Rough estimate
total_memory_mb = param_memory_mb + optimizer_memory_mb + activation_memory_mb

print(f" Estimated RAM usage: ~{total_memory_mb/1024:.1f}GB")

if total_memory_mb/1024 > available_ram_gb * 0.8:
    print("  WARNING: Model may use too much RAM!")
    print("   Consider: 1) Reducing batch_size, or 2) Collecting less data")
    user_input = input("\n   Continue anyway? (y/n): ")
    if user_input.lower() != 'y':
        print("Exiting...")
        exit(0)

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,  # More epochs for better learning
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=5e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    fp16=False,  # MPS doesn't support fp16
    report_to="none",
    logging_dir=LOGS_DIR,
    dataloader_num_workers=0,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

print("\n" + "="*70)
print(" STARTING TRAINING")
print("="*70)
print(f"\nEstimated time: {len(dataset['train']) * 5 * 0.5 / 60:.0f}-{len(dataset['train']) * 5 * 1 / 60:.0f} minutes")
print("Watch the loss value - it should decrease from ~10 → ~3 or lower\n")
print("💡 TIP: You can press Ctrl+C to stop early and still save the model\n")

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")

# Save final model
print("\n" + "="*70)
print(" SAVING MODEL")
print("="*70)

os.makedirs(MODEL_FINAL_DIR, exist_ok=True)
trainer.save_model(MODEL_FINAL_DIR)
tokenizer.save_pretrained(MODEL_FINAL_DIR)

print(f"\n Model saved to: {MODEL_FINAL_DIR}")
print("\n" + "="*70)
print(" TRAINING COMPLETE!")
