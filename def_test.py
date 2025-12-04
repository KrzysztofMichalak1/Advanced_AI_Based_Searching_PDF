import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import numpy as np
import os
import faiss 
import fitz  # PyMuPDF
import re

# Optional: for a progress bar
from tqdm import tqdm 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = "advanced-searching-pdf"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# --- Optimization 1: Batching Function ---
def get_batch_embeddings(text_list, batch_size=250):
    """
    Generates embeddings in batches to reduce API overhead.
    """
    embeddings = []
    # Loop through the list in chunks of 'batch_size'
    for i in tqdm(range(0, len(text_list), batch_size), desc="Generating Embeddings"):
        batch = text_list[i : i + batch_size]
        try:
            # Send the whole batch at once
            batch_results = model.get_embeddings(batch)
            # Extract values
            for result in batch_results:
                embeddings.append(result.values)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Handle errors (e.g., append zero vectors or skip)
            
    return np.array(embeddings, dtype='float32')

target_word = 'tank'

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
pdf_file = "story_3.pdf"
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

    # Get definition of target word and embed it
    generation_model = GenerativeModel("gemini-2.5-flash")

    prompt = f"Write two short, descriptive sentences about '{target_word}', including its key attributes and associations."
    try:
        response = generation_model.generate_content(prompt)
        definition = response.text.strip()
        print(f"Generated text for '{target_word}': {definition}")
    except Exception as e:
        print(f"Could not get definition for '{target_word}': {e}")
        definition = target_word # Fallback to just the word

    target_vec = np.array([model.get_embeddings([definition])[0].values], dtype='float32')

    k = 10
    distances, indices = index.search(target_vec, k)

    print(f"\nClosest words to the generated text for '{target_word}':")

    for i in range(k):
        found_id = indices[0][i]     
        score = distances[0][i]      
        found_word = words[found_id] 
        
        print(f"{i+1}. {found_word:<12} (Score: {score:.4f})")
else:
    print("No words found to embed.")