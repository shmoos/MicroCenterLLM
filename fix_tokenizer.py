from tokenizers import ByteLevelBPETokenizer
import os

# CONSTANTS - Fix for S1192 (duplicated strings)
TOKENIZER_DIR = "tokenizer"
VOCAB_FILE = f"{TOKENIZER_DIR}/vocab.json"
MERGES_FILE = f"{TOKENIZER_DIR}/merges.txt"
TOKENIZER_JSON = f"{TOKENIZER_DIR}/tokenizer.json"

# Load your existing tokenizer files
print("Loading tokenizer from vocab.json and merges.txt...")
tokenizer = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)

# Get absolute path
tokenizer_path = os.path.abspath(TOKENIZER_JSON)
print(f"Saving to: {tokenizer_path}")

# Save the tokenizer.json file
tokenizer.save(TOKENIZER_JSON)

# Check if file exists
if os.path.exists(TOKENIZER_JSON):
    file_size = os.path.getsize(TOKENIZER_JSON)
    print(f"✓ Successfully created tokenizer.json ({file_size} bytes)")
else:
    print("✗ File was not created!")
    print("\nTrying alternative method...")
    
    # Alternative: manually construct and save
    tokenizer_obj = tokenizer._tokenizer
    tokenizer_obj.save(TOKENIZER_JSON)
    
    if os.path.exists(TOKENIZER_JSON):
        print("✓ Alternative method worked!")
    else:
        print("✗ Still failed. Let me know and we'll try another approach.")

print("\nListing tokenizer/ folder contents:")
for file in os.listdir(TOKENIZER_DIR):
    print(f"  - {file}")