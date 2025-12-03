# build_faiss.py
import pickle
import numpy as np
import faiss
import os

def build_faiss_index():
    print("Loading embeddings from pickle...")
    # Load your existing embeddings
    with open("data/embeddings.pkl", 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    print(f"Loaded {len(embeddings)} vectors with dimension {embeddings.shape[1]}")

    # Initialize FAISS Index
    # IndexFlatL2 is the standard exact-match search (L2 = Euclidean Distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # FAISS expects float32
    embeddings_np = np.array(embeddings).astype('float32')

    # Add vectors to the index
    print("Building FAISS index...")
    index.add(embeddings_np)
    
    # Save the index to disk
    output_path = os.path.join("data", "vector_store.index")
    faiss.write_index(index, output_path)
    print(f"Success! FAISS index saved to: {output_path}")

if __name__ == "__main__":
    build_faiss_index()