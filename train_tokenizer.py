from tokenizers import ByteLevelBPETokenizer
import os

# Constants
TOKENIZER_DIR = "tokenizer"
DATA_FILE = "data.txt"
VOCAB_SIZE = 30000

# Create tokenizer directory if it doesn't exist
os.makedirs(TOKENIZER_DIR, exist_ok=True)

print("=" * 60)
print(" TRAINING TOKENIZER")
print("=" * 60)

# Check if data file exists
if not os.path.exists(DATA_FILE):
    print(f"\n❌ ERROR: {DATA_FILE} not found!")
    print("Run aggressive_scraper.py first to collect data.")
    exit(1)

# Check data size
data_size = os.path.getsize(DATA_FILE)
print(f"\n Training on: {data_size/1024:.1f} KB of data")

# Initialize tokenizer
print("\n⚙️  Initializing Byte-Level BPE Tokenizer...")
tokenizer = ByteLevelBPETokenizer()

# Train on your data
print("🎓 Training tokenizer (this may take 1-2 minutes)...")
tokenizer.train(
    files=[DATA_FILE],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "<mask>"
    ]
)

# Save the tokenizer
print("\n💾 Saving tokenizer files...")
tokenizer.save_model(TOKENIZER_DIR)
print(f"✓ Saved vocab.json and merges.txt to {TOKENIZER_DIR}/")

# Save as tokenizer.json (for HuggingFace compatibility)
tokenizer.save(f"{TOKENIZER_DIR}/tokenizer.json")
print(f"✓ Saved tokenizer.json")

# Display stats
actual_vocab_size = tokenizer.get_vocab_size()
print("\n" + "=" * 60)
print(" TOKENIZER TRAINING COMPLETE!")
print("=" * 60)
print(f" Vocabulary size: {actual_vocab_size:,}")
print(f" Files created:")
print(f"   - {TOKENIZER_DIR}/vocab.json")
print(f"   - {TOKENIZER_DIR}/merges.txt")
print(f"   - {TOKENIZER_DIR}/tokenizer.json")
print("\n NEXT STEP: Run 'python preprocess_data.py'")