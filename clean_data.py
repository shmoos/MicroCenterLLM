"""
Clean data.txt - remove source markers and format properly
"""

print("Reading data.txt...")
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Remove source markers
lines = content.split("\n")
cleaned_lines = []

for line in lines:
    # Skip source markers and empty lines
    if line.startswith("--- Source:") or line.strip() == "":
        continue
    cleaned_lines.append(line)

# Join into paragraphs
cleaned_text = "\n".join(cleaned_lines)

# Save cleaned version
with open("data_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Cleaned data saved to data_cleaned.txt")
print(f"Size: {len(cleaned_text)/1024:.1f} KB")
