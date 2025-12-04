import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import os
import faiss 
import fitz  # PyMuPDF
import re

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = "advanced-searching-pdf"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

def get_vec(text):
    embedding = model.get_embeddings([text])[0].values
    return np.array(embedding, dtype='float32') 

target_word = 'idiot'

def chunk_text(text):
    """
    Splits text into a list of words (chunks) for embedding.
    Converts to lowercase and removes punctuation.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    words = text.split()
    return words

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    :param pdf_path: The path to the PDF file.
    :return: The extracted text as a string.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error extracting text: {e}"
    return text

pdf_file = "Story.pdf"
extracted_content = extract_text_from_pdf(pdf_file)
# print("--- Extracted Text ---")
# print(extracted_content)

text_chunks = chunk_text(extracted_content)
words = sorted(list(set(text_chunks)))


print(f"Tworzenie embeddingów dla {len(words)} słów...")

word_vecs = np.array([get_vec(w) for w in words])

d = word_vecs.shape[1] 

index = faiss.IndexFlatIP(d) 

index.add(word_vecs) 

print(f"Indeks zawiera {index.ntotal} wektorów.")

target_vec = np.array([get_vec(target_word)]) 

k = 10
distances, indices = index.search(target_vec, k)


print(f"\nNajbliższe słowa do '{target_word}':")

for i in range(k):
    found_id = indices[0][i]     
    score = distances[0][i]      
    found_word = words[found_id] 
    
    print(f"{i+1}. {found_word:<12} (Score: {score:.4f})")