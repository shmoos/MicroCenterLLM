import numpy as np 
from tokenizers import ByteLevelBPETokenizer
import os

# Constants
TOKENIZER_DIR = "tokenizer"
VOCAB_FILE = f"{TOKENIZER_DIR}/vocab.json"
MERGES_FILE = f"{TOKENIZER_DIR}/merges.txt"
DATA_FILE = "data.txt"
OUTPUT_FILE = "tokenized_data.npy"
CHUNKS_FILE = "tokenized_chunks.npy"
CHUNK_SIZE = 512

print("=" * 60)
print("⚙️  PREPROCESSING DATA")
print("=" * 60)

# Check if tokenizer files exist
if not os.path.exists(VOCAB_FILE) or not os.path.exists(MERGES_FILE):
    print("\n ERROR: Tokenizer files not found!")
    print("Run train_tokenizer.py first to create the tokenizer.")
    exit(1)

# Check if data file exists
if not os.path.exists(DATA_FILE):
    print(f"\n ERROR: {DATA_FILE} not found!")
    exit(1)

# Load tokenizer
print("\n Loading tokenizer...")
tokenizer = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)
print(f"✓ Vocabulary size: {tokenizer.get_vocab_size():,}")

# Load data
print(f"\n Loading {DATA_FILE}...")
try:
    with open(DATA_FILE, "r", encoding="utf-8") as f: 
        text = f.read()
except Exception as e:
    print(f" ERROR reading file: {e}")
    exit(1)

print(f"✓ Data size: {len(text):,} characters")

# Tokenize the entire text
print("\n Tokenizing (this may take a minute)...")
encoded = tokenizer.encode(text)
tokenized_data = encoded.ids

print(f"✓ Total tokens: {len(tokenized_data):,}")

# Save tokenized data
print("\n Saving tokenized data...")
np.save(OUTPUT_FILE, np.array(tokenized_data, dtype=np.int32))
print(f"✓ Saved {OUTPUT_FILE}")

# Create chunks of fixed length for training
print(f"\n Creating chunks of size {CHUNK_SIZE}...")
chunks = []
for i in range(0, len(tokenized_data), CHUNK_SIZE):
    chunk = tokenized_data[i:i + CHUNK_SIZE]
    if len(chunk) == CHUNK_SIZE:  # Only keep full-sized chunks
        chunks.append(chunk)

if len(chunks) == 0:
    print("\n⚠️  WARNING: Not enough data to create even one chunk!")
    print(f"   You need at least {CHUNK_SIZE} tokens.")
    print("   Consider collecting more data with aggressive_scraper.py")
else:
    chunks = np.array(chunks, dtype=np.int32)
    print(f"✓ Created {len(chunks):,} chunks")
    
    # Save chunks
    np.save(CHUNKS_FILE, chunks)
    print(f"✓ Saved {CHUNKS_FILE}")
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f" Summary:")
    print(f"   Total tokens: {len(tokenized_data):,}")
    print(f"   Total chunks: {len(chunks):,}")
    print(f"   Chunk size: {CHUNK_SIZE}")
    print(f"   Data shape: {chunks.shape}")
    
    # Assessment
    if len(chunks) < 50:
        print("\n  WARNING: Very few chunks (< 50)")
        print("   Your model may not learn well. Consider:")
        print("   1. Running aggressive_scraper.py again")
        print("   2. Collecting more diverse data sources")
    elif len(chunks) < 200:
        print("\n✓ Decent amount of data (50-200 chunks)")
        print("  Should work for a small model")
    else:
        print("\n✓✓ Good amount of data (200+ chunks)")
        print("  Ready for training!")
    
    print("\n NEXT STEP: Run 'python train_model_optimized.py'")