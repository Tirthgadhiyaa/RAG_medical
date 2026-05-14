# ============================================================
# STEP 2: CHUNKING THE DOCUMENTS
# ============================================================
# After loading data, we can't feed 15,000 full documents
# to the LLM at once — that's too much text.
#
# CHUNKING = breaking documents into smaller pieces
# so the retriever can find the EXACT relevant piece.
#
# WHY CHUNKING MATTERS:
# - If a document is 1000 words, only 50 words might be relevant
# - Smaller chunks = more precise retrieval
# - We use OVERLAP so context is not lost at chunk boundaries
#
# Our strategy:
# - chunk_size = 500 characters (~80-100 words)
# - chunk_overlap = 100 characters (shared between adjacent chunks)
# - This means each chunk shares 100 chars with the next
#   so no sentence is cut off completely
# ============================================================

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pickle
import os

print("=" * 60)
print("STEP 2: Chunking Documents")
print("=" * 60)

# Load the CSV we saved in Step 1
df = pd.read_csv("data/medical_qa.csv")
print(f"\nLoaded {len(df)} documents from CSV")

# ============================================================
# RecursiveCharacterTextSplitter is the BEST splitter for
# medical text. It tries to split on:
#   1. Paragraphs (\n\n) first
#   2. Then sentences (. ! ?)
#   3. Then words (space)
#   4. Then characters
# So it never cuts in the middle of a word if possible.
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=100,    # overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

# Convert each row into a LangChain Document object
# Document has: page_content (the text) + metadata (extra info)
# Metadata is very useful — we store the original question and row number
# so when we retrieve a chunk we can show the user WHERE it came from

print("\nCreating LangChain Document objects...")
raw_docs = []
for idx, row in df.iterrows():
    doc = Document(
        page_content=row['document'],
        metadata={
            "row_id": idx,
            "original_question": str(row['input'])[:200],  # truncate for storage
            "source": "MedQuAD / ChatDoctor HealthcareMagic"
        }
    )
    raw_docs.append(doc)

print(f"Created {len(raw_docs)} raw documents")

# ============================================================
# Now split all documents into chunks
# ============================================================
print("\nSplitting into chunks (this may take 30-60 seconds)...")

chunks = splitter.split_documents(raw_docs)

print(f"\nChunking complete!")
print(f"Original documents : {len(raw_docs)}")
print(f"After chunking     : {len(chunks)} chunks")
print(f"Avg chunk size     : {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

# Show a sample chunk
print("\n--- SAMPLE CHUNK ---")
print(f"Content: {chunks[10].page_content[:400]}")
print(f"Metadata: {chunks[10].metadata}")

# Save chunks to disk (so Step 3 doesn't need to redo this)
os.makedirs("embeddings", exist_ok=True)
with open("embeddings/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"\nSaved {len(chunks)} chunks to embeddings/chunks.pkl")
print("\nSTEP 2 COMPLETE ✓")
