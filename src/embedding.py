# vector.py
"""
Module for creating and managing embeddings for the RAG system.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import pickle
import os


def load_model(model_name: str = "msmarco-distilbert-base-v4"):
    """
    Load the embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model
        
    Returns:
        The loaded SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded!\n")
    return model


def create_embeddings(model, texts: List[str], show_progress: bool = True) -> np.ndarray:
    """
    Create embeddings for a list of texts.
    
    Args:
        model: The SentenceTransformer model to use
        texts: List of text strings to embed
        show_progress: Whether to show a progress bar
        
    Returns:
        Numpy array of shape (num_texts, embedding_dimension)
    """
    print(f"Creating embeddings for {len(texts)} texts...")
    
    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        batch_size=32  
    )
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, filepath: str, model_name: str):
    """
    Save embeddings to disk.
    
    Args:
        embeddings: The embeddings array to save
        filepath: Path where to save the file
        model_name: Name of the model used (for reference)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'model_name': model_name
        }, f)
    
    print(f"Saved embeddings to: {filepath}")


def load_saved_embeddings(filepath: str) -> tuple[np.ndarray, str]:
    """
    Load embeddings from disk.
    
    Args:
        filepath: Path to the saved embeddings file
        
    Returns:
        Tuple of (embeddings array, model_name)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded embeddings with shape: {data['embeddings'].shape}")
    print(f"  Model used: {data['model_name']}")
    
    return data['embeddings'], data['model_name']