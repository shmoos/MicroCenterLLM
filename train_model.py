import torch
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast 
from datasets import load_dataset 
import os

# CONSTANTS
TOKENIZER_PATH = "tokenizer/tokenizer.json"
DATA_FILE = "data.txt"  
MODEL_OUTPUT_DIR = "./model"
MODEL_FINAL_DIR = "./model/final"
LOGS_DIR = "./logs"

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available(): 
    device = torch.device("cuda")
    print("Using CUDA")
else: 
    device = torch.device("cpu")
    print("Using CPU (slow rate)")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.add_special_tokens({ 
    'pad_token': '<pad>',
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
})
print(f"Vocab size: {len(tokenizer)}")

# Load dataset 
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": DATA_FILE}) 

def tokenize(batch): 
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
print(f"Dataset size: {len(dataset['train'])} examples")

# Model configuration
config = GPT2Config( 
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=512,
    n_layer=12,
    n_head=8, 
    resid_pdrop=0.1, 
    embd_pdrop=0.1, 
    attn_pdrop=0.1 
)

print("Initializing model...")
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))  # FIX: You had this twice - removed duplicate
model = model.to(device)
print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    overwrite_output_dir=True, 
    num_train_epochs=3, 
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, 
    learning_rate=5e-4,
    warmup_steps=500, 
    weight_decay=0.01, 
    logging_steps=50, 
    save_steps=500,
    save_total_limit=3,
    fp16=False, 
    report_to="none", 
    logging_dir=LOGS_DIR, 
    dataloader_num_workers=0
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

print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

try:
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted by user")

# Save final model
print("\nSaving final model...")
os.makedirs(MODEL_FINAL_DIR, exist_ok=True)
trainer.save_model(MODEL_FINAL_DIR)
tokenizer.save_pretrained(MODEL_FINAL_DIR)

print("\n" + "="*50)
print("Training complete!")
print("="*50)
print(f"Model saved to: {MODEL_FINAL_DIR}")