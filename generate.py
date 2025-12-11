import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# Constants
MODEL_PATH = "./model/checkpoint-3000"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("\n" + "="*60)
print(" COMPUTER PARTS LLM - TEXT GENERATION")
print("="*60)

# Load tokenizer and model
print("\n Loading tokenizer...")
try:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f" Error loading tokenizer: {e}")
    exit(1)

print("\n Loading model...")
try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    print("\nMake sure you've run train_model_optimized.py first!")
    exit(1)

# Display model info
total_params = model.num_parameters()
print(f"✓ Model size: {total_params/1e6:.1f}M parameters")

print("\n" + "="*60)
# print(" TIPS FOR GOOD PROMPTS:")
# print("="*60)
# print("Try prompts like:")
# print("  - 'What is the best GPU for'")
# print("  - 'How much RAM do I need'")
# print("  - 'Intel Core i9'")
# print("  - 'RTX 4090'")
# print("  - 'What CPU should I'")
# print("  - 'Question: How do I choose a motherboard?'")
# print("\n" + "="*60)

# Interactive generation loop
while True:
    prompt = input("\nEnter prompt (or 'quit' to exit): ")
    
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("\n Goodbye!")
        break
    
    if not prompt.strip():
        continue
    
    # Tokenize and generate
    print("\n Generating...")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.1,  # Lower = more focused, Higher = more creative
                top_k=10,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and display
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "="*60)
        print("Generated:")
        print("="*60)
        print(generated_text)
        print("="*60)
        
    except Exception as e:
        print(f"\n Error generating text: {e}")
        continue