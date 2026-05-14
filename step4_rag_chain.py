# ============================================================
# STEP 4: BUILD THE RAG CHAIN
# ============================================================
# Now we connect everything:
#   User Question
#       ↓
#   Embed the question (same model as before)
#       ↓
#   Search FAISS → get top-K most similar chunks
#       ↓
#   Build a PROMPT = "Here is medical context: {chunks}
#                     Answer this question: {question}"
#       ↓
#   Send prompt to LLM (Groq - free, fast)
#       ↓
#   Return answer + show which sources were used
#
# WHY GROQ?
# - 100% free API (sign up at console.groq.com)
# - Uses LLaMA 3 model (very powerful)
# - Super fast (50x faster than OpenAI for same model)
# - No credit card needed
# ============================================================

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()  # loads GROQ_API_KEY from .env file

print("=" * 60)
print("STEP 4: Building the RAG Chain")
print("=" * 60)

# ============================================================
# 1. Load the FAISS vector store we built in Step 3
# ============================================================
print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Loading FAISS index from disk...")
# Auto-detect correct path
possible_paths = ["embeddings/faiss_index", os.path.join(os.path.dirname(__file__), "embeddings", "faiss_index")]
index_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])

vectorstore = FAISS.load_local(
    index_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  # safe since we created this file
)
print(f"Loaded {vectorstore.index.ntotal} vectors [DONE]")

# ============================================================
# 2. Create a RETRIEVER from the vector store
# The retriever is the component that does the searching.
# k=5 means: retrieve 5 most relevant chunks per query
# ============================================================
retriever = vectorstore.as_retriever(
    search_type="similarity",  # cosine similarity search
    search_kwargs={"k": 5}     # top 5 chunks
)

# ============================================================
# 3. Create the LLM using Groq (free)
# Get your free API key at: https://console.groq.com
# Then add to .env file:  GROQ_API_KEY=your_key_here
# ============================================================
print("\nConnecting to Groq LLM...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # LLaMA 3.1 8B — fast and accurate
    temperature=0.2,           # low temp = more factual, less creative
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ============================================================
# 4. Create a custom PROMPT TEMPLATE
# This is very important — it tells the LLM exactly how to
# use the retrieved context and how to format the answer.
# ============================================================
prompt_template = """SYSTEM: You are a Clinical Knowledge Retrieval System.
OBJECTIVE: Provide factual information from the Medical Context based on the User Question.

STRICT PROTOCOL:
1. If the user asks for "signs" or "symptoms" of a disease, list them directly from the context.
2. NEVER mention symptoms the user didn't explicitly state.
3. NEVER assume the user has the disease they are asking about.
4. DO NOT provide a medical disclaimer in the text (the UI handles this).
5. Use a professional, clinical tone. No chatty introductions.

Medical Context:
{context}

User Question: {question}

Clinical Information:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ============================================================
# 5. Build the RetrievalQA Chain
# This chain automatically:
#   - Takes user question
#   - Runs retriever to get relevant chunks
#   - Fills in the prompt template
#   - Sends to LLM
#   - Returns the answer
# return_source_documents=True shows WHICH chunks were used
# ============================================================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",               # "stuff" = put all chunks into one prompt
    retriever=retriever,
    return_source_documents=True,     # IMPORTANT: shows sources
    chain_type_kwargs={"prompt": PROMPT}
)

print("RAG chain ready [DONE]\n")

# ============================================================
# 6. Test it with a few example questions
# ============================================================
def ask_question(question):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    result = qa_chain.invoke({"query": question})
    
    print(f"Answer:\n{result['result']}")
    
    # Show sources (which chunks were retrieved)
    print(f"\nSources used ({len(result['source_documents'])} chunks):")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"  [{i}] {doc.page_content[:120]}...")
    
    return result

# Run test questions
test_questions = [
    "What are the common symptoms of diabetes?",
    "How is high blood pressure treated?",
    "What should I do if I have a severe headache?"
]

for q in test_questions:
    ask_question(q)
    print("\n" + "=" * 60)

print("\nSTEP 4 COMPLETE [DONE]")
print("Now run: streamlit run step5_app.py")
