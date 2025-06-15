# src/search/fine_search.py

import numpy as np
import torch
from typing import List, Dict, Union

def fine_search_chunks(query_emb: Union[List[float], np.ndarray], 
                      chunk_index: List[Dict], 
                      top_k: int = 10) -> List[Dict]:
    """
    Find the most relevant text chunks by comparing with ALL chunks.
    
    Parameters
    ----------
    query_emb : list[float] | np.ndarray
        Embedding vector of the user query.
    chunk_index : list[dict]
        Each element is a dictionary like:
        {
            "embedding": [...],
            "metadata": {"section_title": "...", ...}
        }
    top_k : int, default = 10
        Number of top‑scoring chunks to return.
    
    Returns
    -------
    List[Dict]
        Top k most similar chunks
    """
    
    if not chunk_index:
        return []
    
    # Use GPU if available for faster computation
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Convert query to numpy array
    if isinstance(query_emb, list):
        query_emb = np.array(query_emb, dtype=np.float32)
    
    # Extract all embeddings
    embeddings = np.array([c["embedding"] for c in chunk_index], dtype=np.float32)
    
    # Normalize embeddings and query for cosine similarity
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Convert to torch tensors for faster computation
    query_tensor = torch.from_numpy(query_norm).to(device)
    embeddings_tensor = torch.from_numpy(embeddings_norm).to(device)
    
    # Compute cosine similarities with ALL chunks
    similarities = torch.matmul(embeddings_tensor, query_tensor)
    
    # Get top-k indices
    top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
    
    # Convert back to CPU
    top_indices = top_indices.cpu().numpy()
    top_scores = top_scores.cpu().numpy()
    
    # Return top chunks
    results = []
    for idx, score in zip(top_indices, top_scores):
        chunk = chunk_index[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    
    return results