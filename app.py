import streamlit as st
import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import os
import faiss
import fitz  # PyMuPDF
import re
import io
import pandas as pd

# --- Vertex AI and Model Initialization ---

# IMPORTANT: The user must have a `gcp_key.json` file in the same directory.
GCP_KEY_FILE = "gcp_key.json"

@st.cache_resource
def initialize_vertexai():
    """Initializes Vertex AI and the embedding model."""
    if not os.path.exists(GCP_KEY_FILE):
        st.error(f"GCP key file not found! Please make sure '{GCP_KEY_FILE}' is in the root directory.")
        return None

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_FILE
        PROJECT_ID = "advanced-searching-pdf"  # As per test.py
        LOCATION = "us-central1"               # As per test.py
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        st.success("Vertex AI initialized successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {e}")
        return None

model = initialize_vertexai()

# --- Core Processing Functions ---

def get_vec(text, _model):
    """Generates an embedding for the given text."""
    if not _model: return None
    try:
        embedding = _model.get_embeddings([text])[0].values
        return np.array(embedding, dtype='float32')
    except Exception as e:
        st.warning(f"Could not generate embedding for '{text}'.")
        return None

def chunk_text(text):
    """Splits text into a list of lowercase words, removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file stream."""
    try:
        file_bytes = uploaded_file.getvalue()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

@st.cache_data(show_spinner=False)
def build_faiss_index(_text_chunks, _model):
    """Builds a FAISS index from a list of text chunks."""
    if not _text_chunks or not _model: return None, None

    with st.spinner(f"Creating embeddings for {len(_text_chunks)} words..."):
        word_vecs = [get_vec(word, _model) for word in _text_chunks]
        valid_vecs = [vec for vec in word_vecs if vec is not None]

    if not valid_vecs:
        st.error("Could not generate any valid embeddings.")
        return None, None

    word_vecs_np = np.array(valid_vecs)
    d = word_vecs_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(word_vecs_np)
    
    return index, _text_chunks

def search_faiss(query, index, words, _model, k=10):
    """Searches the FAISS index for the top k similar words."""
    if not query:
        st.warning("Please enter a search term.")
        return None

    query_vec = get_vec(query, _model)
    if query_vec is None:
        return None
        
    target_vec = np.array([query_vec])
    distances, indices = index.search(target_vec, k)
    
    results = []
    for i in range(k):
        found_id = indices[0][i]
        score = distances[0][i]
        found_word = words[found_id]
        results.append({"Rank": i + 1, "Word": found_word, "Similarity Score": f"{score:.4f}"})
        
    return pd.DataFrame(results)

# --- Streamlit App ---

def main():
    st.title("Semantic Word Search in PDFs")

    if model is None:
        st.stop()

    st.write("Upload a PDF to create a searchable index of its content.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        if "faiss_index" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            st.session_state.file_name = uploaded_file.name
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    words = chunk_text(extracted_text)
                    faiss_index, indexed_words = build_faiss_index(words, model)
                    st.session_state.faiss_index = faiss_index
                    st.session_state.words = indexed_words
                    if faiss_index:
                        st.success(f"Index for '{uploaded_file.name}' created with {faiss_index.ntotal} words.")


    if st.session_state.get("faiss_index") is not None:
        st.divider()
        st.header(f"Search in '{st.session_state.file_name}'")
        
        search_query = st.text_input("Enter a word to find similar terms in the document:")
        
        if st.button("Search"):
            with st.spinner("Searching..."):
                results_df = search_faiss(
                    search_query, 
                    st.session_state.faiss_index, 
                    st.session_state.words, 
                    model
                )
                
                if results_df is not None:
                    st.subheader(f"Top 10 words similar to '{search_query}':")
                    st.table(results_df)

if __name__ == "__main__":
    main()

