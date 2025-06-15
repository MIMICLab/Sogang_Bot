# src/search/vector_search.py

import numpy as np
import torch
from typing import List, Dict, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class OptimizedVectorSearch:
    """Optimized vector search using PyTorch and NumPy for faster similarity computation."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the vector search engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.device = self._get_device(use_gpu)
        self.index_embeddings = None
        self.index_metadata = None
        self.embedding_dim = None
        
    def _get_device(self, use_gpu: bool) -> torch.device:
        """Determine the best available device."""
        if use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        elif use_gpu and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def build_index(self, index_data: List[Dict]) -> None:
        """
        Build the search index from data.
        
        Args:
            index_data: List of dicts with 'embedding' and 'metadata' keys
        """
        if not index_data:
            raise ValueError("Index data cannot be empty")
        
        # Extract embeddings and metadata
        embeddings = []
        metadata = []
        
        for item in index_data:
            embeddings.append(item["embedding"])
            metadata.append(item.get("metadata", {}))
        
        # Convert to numpy array first for efficiency
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity (L2 normalization)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / (norms + 1e-8)
        
        # Convert to float16 for memory efficiency
        embeddings_np = embeddings_np.astype(np.float16)
        
        # Convert to torch tensor and move to device
        self.index_embeddings = torch.from_numpy(embeddings_np).to(self.device)
        self.index_metadata = metadata
        self.embedding_dim = embeddings_np.shape[1]
        
        print(f"Index built with {len(metadata)} vectors on {self.device}")
    
    def search(self, query_emb: Union[np.ndarray, List[float], torch.Tensor], 
               top_k: int = 8) -> List[Dict]:
        """
        Search for the most similar vectors by comparing with ALL chunks.
        
        Args:
            query_emb: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with metadata and scores
        """
        if self.index_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Convert query to tensor
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb, dtype=np.float32)
        if isinstance(query_emb, np.ndarray):
            query_emb = torch.from_numpy(query_emb.astype(np.float16))
        
        # Move to device and normalize
        query_emb = query_emb.to(self.device)
        query_emb = query_emb / (torch.norm(query_emb) + 1e-8)
        
        # Compute cosine similarities with ALL vectors
        # Since vectors are normalized, dot product = cosine similarity
        similarities = torch.matmul(self.index_embeddings, query_emb)
        
        # Get top-k indices and scores
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        # Convert back to CPU for output
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Prepare results
        results = []
        for idx, score in zip(top_indices, top_scores):
            result = {
                "metadata": self.index_metadata[idx],
                "score": float(score),
                "index": int(idx)
            }
            results.append(result)
        
        return results
    
    def batch_search(self, query_embs: Union[np.ndarray, List[List[float]], torch.Tensor], 
                     top_k: int = 8) -> List[List[Dict]]:
        """
        Batch search for multiple queries at once, comparing with ALL chunks.
        
        Args:
            query_embs: Multiple query embedding vectors
            top_k: Number of top results per query
            
        Returns:
            List of result lists, one for each query
        """
        if self.index_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Convert queries to tensor
        if isinstance(query_embs, list):
            query_embs = np.array(query_embs, dtype=np.float32)
        if isinstance(query_embs, np.ndarray):
            query_embs = torch.from_numpy(query_embs.astype(np.float16))
        
        # Move to device and normalize
        query_embs = query_embs.to(self.device)
        query_norms = torch.norm(query_embs, dim=1, keepdim=True)
        query_embs = query_embs / (query_norms + 1e-8)
        
        # Compute similarities for all queries with ALL vectors
        # Shape: (num_queries, num_index_vectors)
        similarities = torch.matmul(query_embs, self.index_embeddings.T)
        
        # Get top-k for each query
        top_scores, top_indices = torch.topk(similarities, min(top_k, similarities.shape[1]), dim=1)
        
        # Convert back to CPU
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Prepare results for each query
        all_results = []
        for query_idx in range(len(query_embs)):
            results = []
            for k in range(top_k):
                if k < top_indices.shape[1]:
                    idx = top_indices[query_idx, k]
                    score = top_scores[query_idx, k]
                    result = {
                        "metadata": self.index_metadata[idx],
                        "score": float(score),
                        "index": int(idx)
                    }
                    results.append(result)
            all_results.append(results)
        
        return all_results


# Backward compatibility functions
def cosine_similarity(v1, v2):
    """Legacy function for cosine similarity."""
    v1 = np.array(v1, dtype=np.float16)
    v2 = np.array(v2, dtype=np.float16)
    
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Compute dot product
    return np.dot(v1_norm, v2_norm)


def simple_vector_search(query_emb, index_data: List[Dict], top_k=8):
    """
    Legacy function that compares with ALL chunks using the optimized search engine.
    
    Args:
        query_emb: numpy array or list[float]
        index_data: [{"embedding": [...], "metadata": {...}}, ...]
    """
    # Use the optimized search engine
    searcher = OptimizedVectorSearch(use_gpu=True)
    searcher.build_index(index_data)
    results = searcher.search(query_emb, top_k)
    
    # Convert to legacy format
    return [{"embedding": index_data[r["index"]]["embedding"], 
             "metadata": r["metadata"]} for r in results]


# Additional feature: Precomputed index for repeated searches
class PrecomputedVectorSearch(OptimizedVectorSearch):
    """Extended version with additional features for repeated searches."""
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        
    def search_with_filter(self, query_emb: Union[np.ndarray, List[float]], 
                          filter_fn=None, top_k: int = 8) -> List[Dict]:
        """
        Search ALL chunks with an optional filter function.
        
        Args:
            query_emb: Query embedding
            filter_fn: Optional function to filter results based on metadata
            top_k: Number of results to return
        """
        if filter_fn is None:
            return self.search(query_emb, top_k)
        
        # Get more results than needed to account for filtering
        initial_results = self.search(query_emb, top_k * 3)
        
        # Apply filter
        filtered_results = []
        for result in initial_results:
            if filter_fn(result["metadata"]):
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results[:top_k]