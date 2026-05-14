# ============================================================
# STEP 5: STREAMLIT WEB APP (THE UI)
# ============================================================
# This file creates a beautiful web interface for our
# Medical RAG system using Streamlit.
#
# Streamlit turns Python scripts into web apps instantly.
# No HTML/CSS/JS needed — just Python!
#
# Run this with: streamlit run step5_app.py
# It opens automatically in your browser at localhost:8501
# ============================================================

import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ============================================================
# Page Configuration — must be the FIRST Streamlit call
# ============================================================
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# Custom CSS for better styling
# ============================================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #0e1117, #1a1c23);
        color: #e0e0e0;
    }

    /* Glassmorphism Header */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        animation: fadeInDown 1s ease-out;
    }

    .main-header h1 {
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: -1px;
    }

    /* 3D Chat Cards */
    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 5px solid #4facfe;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        transition: all 0.3s ease;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
        animation: slideInRight 0.5s ease-out;
    }

    .chat-card:hover {
        transform: translateY(-5px) scale(1.01);
        background: rgba(255, 255, 255, 0.07);
        box-shadow: 10px 10px 30px rgba(0,0,0,0.4);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #11141a;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Custom Input Box */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* Sources Section */
    .source-box {
        background: rgba(79, 172, 254, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        font-size: 0.9rem;
        border: 1px dashed rgba(79, 172, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>✨ MediMind AI</h1>
    <p style="color: #a0a0a0; font-size: 1.1rem;">Next-Gen Medical Knowledge Retrieval System</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CACHING
# ============================================================
@st.cache_resource
def load_rag_pipeline():
    """Load everything once and cache it in memory."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Auto-detect correct path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "embeddings", "faiss_index"),
        "embeddings/faiss_index"
    ]
    index_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])

    vectorstore = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
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
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain, vectorstore

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3845/3845868.png", width=100)
    st.title("MediMind AI")
    st.markdown("---")
    
    try:
        qa_chain, vectorstore = load_rag_pipeline()
        st.success("✨ System Online")
        loaded = True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        loaded = False
    
    st.markdown("---")
    st.subheader("💡 Ask about:")
    sample_questions = [
        "Common symptoms of diabetes",
        "High blood pressure treatment",
        "Signs of kidney stones",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state['input_question'] = q

# ============================================================
# Main Interface
# ============================================================
if loaded:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Question Input
    default_val = st.session_state.get('input_question', '')
    user_question = st.text_input("Consult MediMind AI:", value=default_val, placeholder="Describe symptoms or ask medical questions...", key="main_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Search", type="primary"):
            if user_question.strip():
                with st.spinner("Analyzing Knowledge Base..."):
                    start_time = time.time()
                    result = qa_chain.invoke({"query": user_question})
                    elapsed = time.time() - start_time
                    
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": result['result'],
                        "sources": result['source_documents'],
                        "time": elapsed
                    })
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # Display History
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class="chat-card">
            <div style="color: #4facfe; font-weight: bold; margin-bottom: 5px;">🧑 YOU</div>
            <div style="margin-bottom: 15px;">{chat['question']}</div>
            <div style="color: #00f2fe; font-weight: bold; margin-bottom: 5px;">🤖 MEDIMIND AI</div>
            <div style="line-height: 1.6;">{chat['answer']}</div>
            <div style="font-size: 0.8rem; color: #707070; margin-top: 10px;">
                ⏱️ Processed in {chat['time']:.2f}s | 📚 Verified with {len(chat['sources'])} medical records
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 View Clinical Sources"):
            for i, doc in enumerate(chat['sources'], 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Medical Record #{i}:</strong><br>
                    {doc.page_content[:400]}...
                </div>
                """, unsafe_allow_html=True)
