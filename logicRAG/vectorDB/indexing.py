import os
import openai
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModel
import voyageai
import torch
#

model_version = "sentence-transformers/all-MiniLM-L6-v2"
tokenize = AutoTokenizer.from_pretrained(model_version)
model = AutoModel.from_pretrained(model_version)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]

def create_embedding(chunk):
    with torch.no_grad():
        encode_input = tokenize(chunk, padding=True, truncation=True, return_tensors="pt")
        model_output = model(**encode_input)
        return cls_pooling(model_output)

#openai.api_key = "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"
#"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"
#"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"
# vo = voyageai.Client(api_key="pa-rahp_-Q37-swCcgCCrUC6UBasPNPIzluZOEaOkaSugw")

# def create_embedding(chunk, model="voyage-lite-01"):    
#     response = vo.embed(chunk, model=model, input_type="document")
#     #print("response shape la ", response.embeddings[0])
#     return response.embeddings#['data'][0]['embedding']

def create_embeddings(text_chunks, model="", max_workers=5):
    embeddings = [None] * len(text_chunks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(create_embedding, chunk): index for index, chunk in enumerate(text_chunks)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]  
            try:
                print("future result is : ", future.result().shape)
                embeddings[index] = future.result()  
            except Exception as e:
                print(f"Error processing chunk at index {index}: {e}")
        return embeddings

def vectordb(embedding_dim: int = 384): # dim cũ là 1536
    # Initialize FAISS index for L2 distance
    index = faiss.IndexFlatL2(384) #khởi tạo list vector dim=1536 
    
    return index

def store_embeddings_faiss(embeddings, index):    
    embeddings_np = np.array(embeddings).astype('float32')#.reshape(-1,1)
    embeddings_np = embeddings_np.squeeze(1)

    #print("dimension laf " , embeddings_np.squeeze(1))
   
    index.add(embeddings_np) #lỗi dimension chỗ này

def save_index(index, filename='faiss_index.bin'):
    faiss.write_index(index, filename)

def load_index(filename='faiss_index.bin'):
    return faiss.read_index(filename)







