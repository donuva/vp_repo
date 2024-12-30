import os
import openai
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = "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"
#os.getenv("OPENAI_API_KEY")

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
                print("future result is : ", future.result())
                embeddings[index] = future.result()  
            except Exception as e:
                print(f"Error processing chunk at index {index}: {e}")
        return embeddings

def vectordb(embedding_dim: int = 1536):
    # Initialize FAISS index for L2 distance
    index = faiss.IndexFlatL2(embedding_dim) #khởi tạo list vector dim=1536 
    return index

def store_embeddings_faiss(embeddings, index):    
    embeddings_np = np.array(embeddings).astype('float32')#.reshape(-1,1)
    #embeddings_np = embeddings_np.reshape(1, -1)

    #print("dimension laf " , embeddings)

    index.add(embeddings_np) #lỗi dimension chỗ này

def save_index(index, filename='faiss_index.bin'):
    faiss.write_index(index, filename)

def load_index(filename='faiss_index.bin'):
    return faiss.read_index(filename)







