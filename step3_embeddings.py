# ============================================================
# STEP 3: CREATE EMBEDDINGS AND BUILD VECTOR DATABASE
# ============================================================
# This is the HEART of RAG.
#
# WHAT IS AN EMBEDDING?
# - An embedding is a list of numbers (a vector) that represents
#   the MEANING of a piece of text.
# - Similar sentences have similar vectors (close in space).
# - Example:
#   "I have chest pain" → [0.12, -0.45, 0.89, ...]
#   "My heart hurts"   → [0.11, -0.43, 0.91, ...]  (very close!)
#   "I love pizza"     → [-0.9, 0.22, -0.55, ...]  (far away!)
#
# WHAT IS FAISS?
# - FAISS (Facebook AI Similarity Search) is a library that
#   stores millions of vectors and lets you find the most
#   similar ones INSTANTLY using math (cosine similarity).
# - Think of it as a super-fast "find similar meaning" search engine.
#
# MODEL: all-MiniLM-L6-v2
# - Small (80MB), fast, and very good for medical text
# - Produces 384-dimensional vectors
# - Free, runs locally, no API key needed
# ============================================================

import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

print("=" * 60)
print("STEP 3: Creating Embeddings + Building FAISS Vector Store")
print("=" * 60)

# Load chunks from Step 2
print("\nLoading chunks...")
with open("embeddings/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"Loaded {len(chunks)} chunks")

# ============================================================
# Load the Embedding Model
# First run downloads the model (~80MB) from HuggingFace
# After that it's cached locally
# ============================================================
print("\nLoading embedding model: all-MiniLM-L6-v2")
print("(First run downloads ~80MB — subsequent runs are instant)")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
    encode_kwargs={"normalize_embeddings": True}  # normalize = better cosine similarity
)

print("Embedding model loaded ✓")

# ============================================================
# Build FAISS index from all chunks
# This embeds every chunk and stores the vectors in FAISS
# For 15k docs → ~30,000-40,000 chunks → takes 5-15 minutes on CPU
# ============================================================
print(f"\nBuilding FAISS index from {len(chunks)} chunks...")
print("This is the slow step — embedding all chunks takes time.")
print("Go get a chai ☕ — this runs once and saves to disk.\n")

start = time.time()

# LangChain handles the batching for us
# It embeds chunks in batches of 32 by default
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

elapsed = time.time() - start
print(f"\nFAISS index built in {elapsed:.1f} seconds")
print(f"Total vectors stored: {vectorstore.index.ntotal}")

# ============================================================
# Save the FAISS index to disk
# Next time, we just LOAD this — no need to rebuild
# ============================================================
vectorstore.save_local("embeddings/faiss_index")
print("\nFAISS index saved to embeddings/faiss_index/")

# ============================================================
# Quick test — search for something and see what comes back
# This proves our vector store works correctly
# ============================================================
print("\n--- QUICK TEST ---")
test_query = "What are the symptoms of diabetes?"
print(f"Query: {test_query}")

results = vectorstore.similarity_search(test_query, k=3)
print(f"\nTop 3 retrieved chunks:")
for i, doc in enumerate(results, 1):
    print(f"\n[Result {i}]")
    print(f"Content: {doc.page_content[:250]}...")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")

print("\nSTEP 3 COMPLETE ✓")
print("\nYour vector database is ready! Now run step4_rag_chain.py")
