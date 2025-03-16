# app.py

import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Our modules:
from embedding import chunk_data, load_index
from retrieval import build_bm25_index, hybrid_search
from re_ranking import ReRanker
from slm_generation import SLMResponseGenerator
from guardrails import is_financial_question
from finance_keywords import FINANCE_KEYWORDS

@st.cache_data
def load_financial_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_dense_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_faiss_index(index_path):
    return load_index(index_path)

def main():
    st.title("Financial QA Demo-Apple")

    # Paths (adjust to your environment)
        # Define base directory
    base_dir = Path(__file__).resolve().parent
    
    # Paths (platform-independent)
    json_path = base_dir / "data" / "processed" / "financial_data.json"
    index_path = base_dir / "embeddings" / "financial_data.index"
    
     # Load data
    financial_data = load_financial_data(json_path)
    chunks = chunk_data(financial_data)
    chunk_texts = [c[0] for c in chunks]
    metadata = [c[1] for c in chunks]

    # Build or load indexes/models
    bm25 = build_bm25_index(chunks)
    dense_model = load_dense_model()
    faiss_index = load_faiss_index(index_path)
    reranker = ReRanker()
    slm = SLMResponseGenerator(model_name="google/flan-t5-base")  # or "google/flan-t5-small"

    user_query = st.text_input("Enter your financial question here", "")
    if st.button("Search"):
        question = user_query.strip()
        if not question:
            st.warning("Please enter a query.")
            return

        # 1) Check if it's a financial question
        if not is_financial_question(question, FINANCE_KEYWORDS):
            st.error("Please ask a finance question.")
            return

        # 2) Perform retrieval
        retrieved_docs = hybrid_search(
            question, 
            bm25, 
            chunk_texts, 
            faiss_index, 
            metadata, 
            dense_model
        )

        # 3) Re-rank
        reranked_docs = reranker.rerank(question, retrieved_docs)
        
        if len(reranked_docs) == 0:
            st.write("No relevant documents found.")
            return

        # 4) Generate final answer
        final_answer = slm.generate_response(question, reranked_docs)

        # Assume top_doc is a list of retrieved documents and we want to check the first one.
        top_docA = reranked_docs[0]
        print(top_docA)
        if str(top_docA['metadata']['value']) not in final_answer:
            st.write("Additional Context:")
            st.write(top_docA['metadata']['finance_parameter'], " ", top_docA['metadata']['year'] ,":",top_docA['metadata']['value'])

        # 5) Display
        st.subheader("Final Answer")
        st.write(final_answer)

        # Show top doc’s confidence
        top_doc = reranked_docs[0]
        confidence = top_doc.get("confidence", 0.0)
        st.write(f"**Confidence Score**: {top_doc['score']:.2f}")

        # (Optional) Show top snippet
        st.markdown("---")
        st.write("**Top Retrieved Snippet**")
        st.write(top_doc["text"])
        st.write(f"Re-rank Score: {top_doc['re_rank_score']:.2f}")

if __name__ == "__main__":
    main()
