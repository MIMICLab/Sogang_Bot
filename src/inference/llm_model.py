# src/inference/llm_model.py 

import torch
from llama_cpp import Llama

import os
from pathlib import Path
from contextlib import redirect_stderr
import requests
from tqdm import tqdm


class LlamaTokenizer:
    def __init__(self, llama_model):
        self._llama = llama_model

    def __call__(self, text, add_bos=True, return_tensors=None):
        ids = self._llama.tokenize(text, add_bos=add_bos)
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids):
        return self._llama.detokenize(ids).decode("utf-8", errors="ignore")
    
class LocalLLM:
    def __init__(self, model_name=None, device="gpu"):
        self.device = device
        self.model_name = model_name

        # Convert relative path to absolute path
        if not os.path.isabs(model_name):
            current_dir = Path(__file__).parent.absolute()
            project_root = current_dir.parent.parent  # Go up two levels
            self.model_name = str(project_root / model_name)

        # Check if model file exists, if not download it
        self._ensure_model_exists()

        self.mmproj_path = None
        self.model = Llama( 
            model_path=self.model_name,
            n_gpu_layers=-1,
            n_ctx=32768,
            verbose=False,
            flash_attn=True
        )
        self.tokenizer = LlamaTokenizer(self.model)

    def _ensure_model_exists(self):
        """Check if model file exists, download if not."""
        model_path = Path(self.model_name)
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not model_path.exists():
            print(f"Model file not found at {model_path}")
            
            # Define download URL based on model name
            if "gemma-3-27b-it-Q8_0.gguf" in str(model_path):
                url = "https://huggingface.co/tgisaturday/Docsray/resolve/main/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q8_0.gguf"
                self._download_model(url, model_path)
            else:
                raise FileNotFoundError(f"Model file not found and no download URL configured for: {model_path}")
    
    def _download_model(self, url, destination):
        """Download model from URL with progress bar."""
        print(f"Downloading model from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Model downloaded successfully to {destination}")
            
        except requests.RequestException as e:
            if destination.exists():
                destination.unlink()  # Remove partial download
            raise Exception(f"Failed to download model: {e}")

    def generate(self, prompt, streaming=False):
        """
        Generate text from prompt, optionally with an image for multimodal models.
        
        Args:
            prompt: Text prompt
            image: PIL Image object (optional)
        """
        # Text-only generation
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        answer = self.model(
            formatted_prompt,
            stop=['<end_of_turn>', '<eos>'],
            max_tokens=8192,
            echo=True,
            temperature = 0.7,
            top_p = 0.95,
            repeat_penalty=1.1,
        )
        result = answer['choices'][0]['text']
        return result.strip()
    
    def strip_response(self, response):
        """Extract the model's response from the full generated text."""
        if not response:
            return response
            
        if '<start_of_turn>model' in response:
            response = response.split('<start_of_turn>model')[-1]
        if '<end_of_turn>' in response:
            response = response.split('<end_of_turn>')[0]
        return response.strip().lstrip('\n')


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

#local_llm = LocalLLM(model_name="data/hub/gemma-3-12b-it-F16.gguf", device=device)            
local_llm = LocalLLM(model_name="data/hub/gemma-3-27b-it-Q8_0.gguf", device=device)   
#local_llm = LocalLLM(model_name="data/hub/gemma-3-27b-it-Q4_K_M.gguf", device=device)   