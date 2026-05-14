# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================
# We use MedQuAD dataset from HuggingFace.
# It has 47,000+ real medical Q&A pairs from NIH (National Institute of Health).
# Each row has: question, answer, source (which NIH page it came from), focus (disease/topic)
#
# Why this dataset?
# - Big enough to be real (47k entries)
# - Clean medical text
# - Free to use
# - Directly downloadable via HuggingFace datasets library
# ============================================================

from datasets import load_dataset
import pandas as pd
import os

print("=" * 60)
print("STEP 1: Downloading MedQuAD Dataset from HuggingFace")
print("=" * 60)

# Load the dataset — this downloads ~50MB automatically
# 'lavita/medical-qa-datasets' contains multiple splits
# we use 'chatdoctor_healthcaremagic' which has 100k real
# doctor-patient conversations — much richer than simple Q&A
print("\nLoading dataset... (first run downloads ~200MB, be patient)")

dataset = load_dataset("lavita/medical-qa-datasets", "chatdoctor_healthcaremagic")

print(f"\nDataset loaded successfully!")
print(f"Total training samples: {len(dataset['train'])}")
print(f"\nColumn names: {dataset['train'].column_names}")

# ============================================================
# Let's look at a few examples to understand the data
# ============================================================
print("\n--- SAMPLE ENTRY ---")
sample = dataset['train'][0]
print(f"Input (Patient): {sample['input'][:300]}...")
print(f"\nOutput (Doctor): {sample['output'][:300]}...")

# ============================================================
# Convert to a clean pandas DataFrame
# We only need 'input' (patient question) and 'output' (doctor answer)
# We combine them into one 'text' column for our knowledge base
# ============================================================
print("\nConverting to DataFrame...")

df = pd.DataFrame(dataset['train'])

# We use only 15,000 rows — good size, not too slow
# You can increase this later once everything works
df = df[['input', 'output']].dropna().head(15000)

# Combine question + answer into one document per row
# This is what will go into our vector database
df['document'] = (
    "Patient Question: " + df['input'].str.strip() +
    "\n\nDoctor Answer: " + df['output'].str.strip()
)

# Save as CSV so you don't have to re-download every time
os.makedirs("data", exist_ok=True)
df.to_csv("data/medical_qa.csv", index=False)

print(f"\nSaved {len(df)} documents to data/medical_qa.csv")
print(f"\nSample document:")
print("-" * 40)
print(df['document'].iloc[0][:500])
print("-" * 40)
print("\nSTEP 1 COMPLETE ✓")
