import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import os
import faiss 
import fitz  # PyMuPDF
import re

# Optional: for a progress bar
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = "advanced-searching-pdf"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# --- Optimization 1: Concurrent Batching Function ---
def get_batch_embeddings(text_list, batch_size=250, max_workers=5):
    """
    Generates embeddings in batches concurrently to reduce API overhead.
    """
    embeddings = [None] * len(text_list)
    
    def process_batch(texts, start_index):
        try:
            batch_embs = model.get_embeddings(texts)
            return start_index, [emb.values for emb in batch_embs]
        except Exception as e:
            print(f"Error in batch starting at {start_index}: {e}")
            return start_index, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, text_list[i:i + batch_size], i)
            for i in range(0, len(text_list), batch_size)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Embeddings"):
            start_index, embs = future.result()
            if embs:
                for i, emb in enumerate(embs):
                    embeddings[start_index + i] = emb
    
    # Find embedding dimension from first successful result
    emb_dim = None
    for emb in embeddings:
        if emb is not None:
            emb_dim = len(emb)
            break
    
    # If all batches failed
    if emb_dim is None and len(text_list) > 0:
        print("All embedding batches failed.")
        try:
            # Probe to get embedding dimension
            emb_dim = len(model.get_embeddings(['probe'])[0].values)
        except Exception as e:
            print(f"Could not determine embedding dimension: {e}")
            return np.array([], dtype='float32')

    # Fill failed embeddings with zero vectors
    final_embeddings = []
    for emb in embeddings:
        if emb is not None:
            final_embeddings.append(emb)
        else:
            if emb_dim:
                final_embeddings.append(np.zeros(emb_dim, dtype='float32'))
            
    return np.array(final_embeddings, dtype='float32')

target_word = 'einstein'

MINIMAL_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by", "from", "with", "is", "are", "was", "were", "be", "it", "that", "this"
}

def chunk_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    filtered_words = [w for w in words if w not in MINIMAL_STOP_WORDS]
    return filtered_words
    
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error extracting text: {e}"
    return text

# --- Main Execution ---
pdf_file = "long.pdf"
extracted_content = extract_text_from_pdf(pdf_file)

text_chunks = chunk_text(extracted_content)

# Remove duplicates
words = sorted(list(set(text_chunks)))


# --- Optimization 2: Filter short words (Optional but recommended) ---
# Filter out words shorter than 3 chars to save time/cost

print(f"Generating embeddings for {len(words)} unique words...")

# Use the new batched function instead of the list comprehension
if words:
    word_vecs = get_batch_embeddings(words, batch_size=250)

    d = word_vecs.shape[1] 
    index = faiss.IndexFlatIP(d) 
    index.add(word_vecs) 

    print(f"Index contains {index.ntotal} vectors.")

    # Get target vector (Single call is fine here)
    target_vec = np.array([model.get_embeddings([target_word])[0].values], dtype='float32')

    k = 10
    distances, indices = index.search(target_vec, k)

    print(f"\nClosest words to '{target_word}':")

    for i in range(k):
        found_id = indices[0][i]     
        score = distances[0][i]      
        found_word = words[found_id] 
        
        print(f"{i+1}. {found_word:<12} (Score: {score:.4f})")
else:
    print("No words found to embed.")