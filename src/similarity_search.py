# search.py
"""
Module for searching through embeddings to find relevant chunks.
"""

import numpy as np
from typing import List, Tuple


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1 (higher = more similar)
    """
    # The formula: dot product divided by product of magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def search_similar_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    top_k: int = 3
) -> List[Tuple[int, float]]:
    """
    Find the most similar chunks to a query.
    
    Args:
        query_embedding: The embedding of the user's question (shape: (384,))
        chunk_embeddings: All chunk embeddings (shape: (num_chunks, 384))
        top_k: Number of top results to return
        
    Returns:
        List of (chunk_index, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    
    # Calculate similarity with each chunk
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (highest first) and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def format_search_results(
    chunk_indices: List[Tuple[int, float]],
    chunks: List,
    show_preview: bool = True,
    preview_length: int = 200
) -> str:
    """
    Format search results in a readable way.
    
    Args:
        chunk_indices: List of (index, similarity_score) tuples
        chunks: List of Document objects
        show_preview: Whether to show a preview of the chunk text
        preview_length: How many characters to preview
        
    Returns:
        Formatted string with search results
    """
    result = "\n" + "="*80 + "\n"
    result += "SEARCH RESULTS\n"
    result += "="*80 + "\n"
    
    for rank, (idx, score) in enumerate(chunk_indices, 1):
        chunk = chunks[idx]
        source = chunk.metadata.get('source', 'Unknown')
        chapter = chunk.metadata.get('chapter', 'Unknown')
        
        result += f"\nRank {rank} (Similarity: {score:.3f})\n"
        result += f"  Source: {source}\n"
        result += f"  Chapter: {chapter}\n"
        
        if show_preview:
            preview = chunk.page_content[:preview_length]
            result += f"  Preview: {chunk.page_content}...\n"
        
        result += "-"*80 + "\n"
    
    return result