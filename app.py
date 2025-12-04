import streamlit as st
st.set_page_config(layout="wide")
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

def search_faiss(query, index, words, _model, similarity_threshold=0.0):
    """Searches the FAISS index, filters by similarity, removes duplicates, and sorts results."""
    if not query:
        st.warning("Please enter a search term.")
        return None

    query_vec = get_vec(query, _model)
    if query_vec is None:
        return None
        
    target_vec = np.array([query_vec])
    
    # Search all words in the index
    k = index.ntotal
    distances, indices = index.search(target_vec, k)
    
    # Filter results by similarity threshold and collect unique words with their best score and first appearance
    unique_filtered_results = {} # Store {word: {data}}
    for i in range(k):
        score = distances[0][i]
        if score >= similarity_threshold:
            found_id = indices[0][i]
            found_word = words[found_id]
            
            if found_word not in unique_filtered_results:
                unique_filtered_results[found_word] = {
                    "Word": found_word,
                    "Similarity Score Value": score, # Numerical score for sorting
                    "Similarity Score": f"{score:.4f}", # Formatted score for display
                    "original_index": found_id # To track first appearance for tie-breaking
                }
            else:
                # If current score is better, update
                if score > unique_filtered_results[found_word]["Similarity Score Value"]:
                    unique_filtered_results[found_word]["Similarity Score Value"] = score
                    unique_filtered_results[found_word]["Similarity Score"] = f"{score:.4f}"
                    unique_filtered_results[found_word]["original_index"] = found_id # Update to this word's first appearance if it's the best score
                # If scores are equal, keep the one with the lowest original_index (first appearance)
                elif score == unique_filtered_results[found_word]["Similarity Score Value"] and found_id < unique_filtered_results[found_word]["original_index"]:
                    unique_filtered_results[found_word]["original_index"] = found_id

    if not unique_filtered_results:
        st.info(f"No results found with a similarity score above {similarity_threshold:.2f}.")
        return None
        
    # Convert dictionary values to a list for sorting
    final_results = list(unique_filtered_results.values())

    # Sort results: primary by Similarity Score (descending), secondary by original_index (ascending)
    final_results_sorted = sorted(
        final_results, 
        key=lambda x: (x["Similarity Score Value"], -x["original_index"]), # Sort by score descending, then by original_index ascending
        reverse=True # Apply reverse to the entire key, effectively making score descending and index ascending for ties
    )

    # Add Rank column and remove temporary sorting keys
    for i, result in enumerate(final_results_sorted):
        result["Rank"] = i + 1 # Rank based on the new sort order
        del result["Similarity Score Value"]
        del result["original_index"]

    return pd.DataFrame(final_results_sorted)


def highlight_text(text, words_to_highlight):
    """Highlights words in the text using HTML."""
    # Use a regex that can handle punctuation attached to words
    # and splits the text into words and the spaces/punctuation between them.
    tokens = re.split(r'([^\w]+)', text)
    
    highlighted_parts = []
    for token in tokens:
        # Clean the token for comparison
        cleaned_token = re.sub(r'[^\w]', '', token).lower()
        if cleaned_token and cleaned_token in words_to_highlight:
            # Highlight the original token, preserving case and punctuation
            highlighted_parts.append(f"<span style='color: red; font-weight: bold;'>{token}</span>")
        else:
            # Append the token as is (it could be a word, punctuation, or whitespace)
            highlighted_parts.append(token)
            
    return "".join(highlighted_parts)


def clear_pdf_data():
    """Clears all session state variables related to the uploaded PDF and relevant caches."""
    keys_to_clear = ["file_name", "faiss_index", "words", "extracted_text"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear the cache for build_faiss_index specifically
    # This ensures that when a new PDF is uploaded, the FAISS index is rebuilt from scratch.
    st.cache_data.clear()


# --- Streamlit App ---

def main():
    st.title("Semantic Word Search in PDFs")

    if model is None:
        st.stop()

    st.write("Upload a PDF to create a searchable index of its content.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    # If file is removed by the user, clear the session state and rerun
    if uploaded_file is None:
        if "file_name" in st.session_state:
            clear_pdf_data()
            st.rerun()
    
    # If a file is uploaded, process it
    else:
        # If it's a new file, process it.
        if st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                clear_pdf_data() # Clear data from any previous file
                st.session_state.file_name = uploaded_file.name
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    st.session_state.extracted_text = extracted_text
                    words = chunk_text(extracted_text)
                    faiss_index, indexed_words = build_faiss_index(words, model)
                    st.session_state.faiss_index = faiss_index
                    st.session_state.words = indexed_words
                    if faiss_index:
                        st.success(f"Index for '{uploaded_file.name}' created with {faiss_index.ntotal} words.")
                        # Rerun to ensure the UI updates cleanly after processing
                        st.rerun()

    # --- UI Display Logic ---

    # This section is only displayed if a file has been successfully processed
    if st.session_state.get("faiss_index") is not None:
        st.divider()
        st.header(f"Search in '{st.session_state.file_name}'")
        
        search_query = st.text_input("Enter a word to find similar terms in the document:")
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        results_df = None # Initialize results_df

        if search_query:
            with st.spinner("Searching..."):
                results_df = search_faiss(
                    search_query, 
                    st.session_state.faiss_index, 
                    st.session_state.words, 
                    model,
                    similarity_threshold=similarity_threshold
                )
        
        # Create two columns for the layout
        col1, col2 = st.columns([3, 1])

        with col1: # Main content (text display)
            st.subheader("Highlighted Document Text" if search_query and results_df is not None else "Document Text")
            full_text = st.session_state.get("extracted_text", "")
            if full_text:
                if search_query and results_df is not None:
                    words_to_highlight = set(results_df['Word'].str.lower())
                    highlighted_content = highlight_text(full_text, words_to_highlight)
                    st.markdown(f'<div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">{highlighted_content}</div>', unsafe_allow_html=True)
                else:
                    # Display the full un-highlighted text if no search is active or no results are found
                    st.markdown(f'<div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">{full_text}</div>', unsafe_allow_html=True)
            # No warning needed here as the initial state (no file) is handled by hiding this whole section
        
        with col2: # Right column for search results
            if search_query and results_df is not None:
                st.subheader(f"Similar words for '{search_query}'")
                df_display = results_df.drop(columns=["Similarity Score"])
                st.table(df_display)

if __name__ == "__main__":
    main()

