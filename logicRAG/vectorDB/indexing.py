import os
import openai
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_embedding(chunk, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=chunk, model=model)
    return response['data'][0]['embedding']

def create_embeddings(text_chunks, model="text-embedding-3-small", max_workers=5):
    embeddings = [None] * len(text_chunks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(create_embedding, chunk, model): index for index, chunk in enumerate(text_chunks)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]  
            try:
                embeddings[index] = future.result()  
            except Exception as e:
                print(f"Error processing chunk at index {index}: {e}")
        return embeddings

def vectordb(embedding_dim: int = 1536):
    # Initialize FAISS index for L2 distance
    index = faiss.IndexFlatL2(embedding_dim)
    return index

def store_embeddings_faiss(embeddings, index):    
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)

def save_index(index, filename='faiss_index.bin'):
    faiss.write_index(index, filename)

def load_index(filename='faiss_index.bin'):
    return faiss.read_index(filename)







