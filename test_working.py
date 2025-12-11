import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "./model_working/checkpoint-2200"  # Your 80% checkpoint

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading model from checkpoint-200 (80% training)...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model.eval()

print("✓ Model loaded!\n")
print("="*60)

while True:
    prompt = input("\nPrompt (or 'quit'): ")
    if prompt.lower() in ['quit', 'q', 'exit']:
        print("Goodbye!")
        break
    
    if not prompt.strip():
        continue
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nGenerating...\n")
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.4,
        top_p=0.3,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("="*60)
    print(text)
    print("="*60)