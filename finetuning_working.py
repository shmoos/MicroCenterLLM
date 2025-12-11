import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

DATA_FILE = "data_cleaned.txt"  # CLEANED DATA
MODEL_OUTPUT = "./model_working"
MODEL_FINAL = "./model_working/final"

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("\n" + "="*70)
print(" FINE-TUNING GPT-2 (IMPROVED - LONGER TRAINING)")
print("="*70)

print("\n Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
model = model.to(device)

print(f"✓ Model: {model.num_parameters()/1e6:.0f}M parameters")

print("\n Loading cleaned data...")
dataset = load_dataset("text", data_files={"train": DATA_FILE})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format("torch", columns=["input_ids", "attention_mask"])

print(f"✓ {len(dataset['train'])} examples")

# Training with MORE epochs
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    num_train_epochs=10,  # INCREASED from 3 to 10
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=25,  # Log more frequently
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\n" + "="*70)
print("TRAINING (10 epochs, ~10-15 minutes)")
print("="*70)
print("\nTarget loss: Should drop from ~3.5 → ~2.0 or lower")
print("Let it complete!\n")

trainer.train()

# Show final loss
print("\n" + "="*70)
print(" TRAINING COMPLETE")
print("="*70)

print("\n Saving...")
os.makedirs(MODEL_FINAL, exist_ok=True)
trainer.save_model(MODEL_FINAL)
tokenizer.save_pretrained(MODEL_FINAL)

print("\n DONE!")
print(f"Saved to: {MODEL_FINAL}")
print("\nTest with: python test_working.py")