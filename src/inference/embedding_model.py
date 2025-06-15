# src/inference/embedding_model.py

from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device="cpu"):
        """
        Load the model and move it to the specified device.
        """
        self.model = SentenceTransformer(model_name,
                                         cache_folder="data/hub", 
                                         trust_remote_code=True).half()
        self.device = device
        if device in ["cuda", "mps"]:
            self.model.to(self.device)

    def get_embedding(self, text: str):
        """
        Return the embedding (1‑D list[float]) for a single sentence.
        """
        emb = self.model.encode([text], convert_to_numpy=True, device=self.device)[0]
        return emb.tolist()

    def get_embeddings(self, texts: list):
        """
        Return embeddings (2‑D numpy array) for multiple sentences.
        """
        embs = self.model.encode(texts, convert_to_numpy=True, device=self.device)
        return embs

# Example of a global instance

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
embedder = EmbeddingModel(model_name="Qwen/Qwen3-Embedding-0.6B", device=device)
