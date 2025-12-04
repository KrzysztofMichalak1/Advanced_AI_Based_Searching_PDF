import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import os
import faiss 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = "advanced-searching-pdf"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained("text-embedding-004")

def get_vec(text):
    embedding = model.get_embeddings([text])[0].values
    return np.array(embedding, dtype='float32') 

target_word = "coding"
words = [
    "python", "java", "algorithm", 
    "pizza", "burger", "pasta", 
    "car", "truck", "bicycle", 
    "computer"
]

print(f"Tworzenie embeddingów dla {len(words)} słów...")

word_vecs = np.array([get_vec(w) for w in words])

d = word_vecs.shape[1] 

index = faiss.IndexFlatIP(d) 

index.add(word_vecs) 

print(f"Indeks zawiera {index.ntotal} wektorów.")

target_vec = np.array([get_vec(target_word)]) 

k = 3 
distances, indices = index.search(target_vec, k)


print(f"\nNajbliższe słowa do '{target_word}':")

for i in range(k):
    found_id = indices[0][i]     
    score = distances[0][i]      
    found_word = words[found_id] 
    
    print(f"{i+1}. {found_word:<12} (Score: {score:.4f})")