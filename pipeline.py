import os
import sys

# Add 'src' to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import extract_text_from_pdfs
from src.chunking import load_documents, split_documents_into_chunks
from src.embedding import load_model, create_embeddings, save_embeddings  # Note: You renamed vector.py to embedding.py
from src.vector_db import build_faiss_index           # Note: Assuming this is your index builder
import pickle
import os

# Configuration
PDF_FOLDER = "data"
EXTRACTED_TEXT_FOLDER = "data/extracted_text"
CLEANED_TEXT_FOLDER = "data/cleaned_text"
CHUNKS_PATH = "data/chunks.pkl"
EMBEDDINGS_PATH = "data/embeddings.pkl"
INDEX_PATH = "data/vector_store.index"

def run_pipeline():
    print("="*80)
    print("STARTING RAG DATA PIPELINE")
    print("="*80 + "\n")

    # 1. Extract Text from PDFs
    print(f"--- Step 1: Extracting Text from {PDF_FOLDER} ---")
    if not os.path.exists(PDF_FOLDER):
        print(f"Error: {PDF_FOLDER} does not exist. Please add PDFs there.")
        return
        
    # We pass a list of titles for the filenames (Update this as needed)
    titles = ["Harry Potter 1", "Harry Potter 2", "Harry Potter 3", 
              "Harry Potter 4", "Harry Potter 5", "Harry Potter 6", "Harry Potter 7"]
    
    extract_text_from_pdfs(PDF_FOLDER, EXTRACTED_TEXT_FOLDER, titles)
    
    # 2. Load and Chunk
    print("\n--- Step 2: Chunking Text ---")
    raw_docs = load_documents(CLEANED_TEXT_FOLDER)
    chunks = split_documents_into_chunks(raw_docs)
    
    # SAVE CHUNKS (Crucial for the Hybrid Search!)
    print(f"Saving {len(chunks)} text chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

    # 3. Create Embeddings
    print("\n--- Step 3: Creating Embeddings ---")
    model = load_model("msmarco-distilbert-base-v4")
    
    # Extract just the text content for embedding
    chunk_texts = [c.page_content for c in chunks]
    embeddings = create_embeddings(model, chunk_texts)
    
    # Save raw embeddings (Intermediate step)
    save_embeddings(embeddings, EMBEDDINGS_PATH, "msmarco-distilbert-base-v4")

    # 4. Build FAISS Index
    print("\n--- Step 4: Building FAISS Index ---")
    # We call the logic from build_faiss directly here
    build_faiss_index()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("You can now run 'streamlit run app.py'")
    print("="*80)

if __name__ == "__main__":
    run_pipeline()