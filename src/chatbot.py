# rag_chatbot.py

from __future__ import annotations

import json, sys, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from collections import Counter
import re
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from inference.llm_model import local_llm
from inference.embedding_model import embedder
from search.fine_search import fine_search_chunks
import sys
if sys.stdin.encoding != 'utf-8':
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
###############################################################################
# Utils
###############################################################################
PROJ_ROOT = Path(__file__).resolve().parents[1]

def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_file():
        return p
    if (PROJ_ROOT / p).is_file():
        return PROJ_ROOT / p
    raise FileNotFoundError(p)

###############################################################################
# 데이터 로더
###############################################################################

def load_chunks(chunks_path: str | Path, vectors_path: str | Path) -> List[dict]:
    chunks_path, vectors_path = _resolve(chunks_path), _resolve(vectors_path)

    with open(chunks_path, encoding="utf-8") as f_txt:
        chunks_raw = json.load(f_txt)
    
    with open(vectors_path, encoding="utf-8") as f_vec:
        vec_raw = json.load(f_vec)
    
    # Combine chunks with their embeddings
    chunk_index = []
    for i, (chunk, vec_item) in enumerate(zip(chunks_raw, vec_raw)):
        if isinstance(vec_item, dict):
            embedding = vec_item["embedding"]
            metadata = vec_item.get("metadata", {})
        else:
            embedding = vec_item
            metadata = {}
        
        # Get text content
        if isinstance(chunk, str):
            content = chunk
        elif isinstance(chunk, dict):
            content = chunk.get("text") or chunk.get("content")
            metadata.update(chunk.get("metadata", {}))
        else:
            raise ValueError(f"Unsupported chunk format: {type(chunk)}")
        
        chunk_index.append({
            "embedding": embedding,
            "metadata": metadata,
            "content": content
        })
    
    return chunk_index

###############################################################################
# Query refinement functions
###############################################################################

def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    """Extract most common meaningful words from texts."""
    # Split texts into words and filter
    words = []
    for text in texts:
        # Basic word extraction (Korean and English)
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+', text.lower())
        # Filter out short words and common stopwords
        tokens = [t for t in tokens if len(t) > 1]
        words.extend(tokens)
    
    # Count frequencies
    word_freq = Counter(words)
    
    # Return top N most common words
    return [word for word, _ in word_freq.most_common(top_n)]

def refine_query(original_query: str, search_results: List[Dict], texts: List[str]) -> str:
    """Refine query based on initial search results."""
    # Extract text from top results
    result_texts = []
    for idx in search_results:
        if 0 <= idx < len(texts):
            result_texts.append(texts[idx])
    
    # Extract keywords from results
    keywords = extract_keywords(result_texts, top_n=5)
    
    # Combine original query with extracted keywords
    # Remove duplicates while preserving order
    query_words = original_query.split()
    combined_words = query_words.copy()
    
    for keyword in keywords:
        if keyword not in original_query.lower():
            combined_words.append(keyword)
    
    # Create refined query
    refined_query = " ".join(combined_words[:15])  # Limit length
    
    return refined_query

def direct_search(query: str, chunk_index: List[dict], 
                 top_k: int = 5) -> List[dict]:
    """Perform direct search comparing query with all chunks."""
    # Encode query
    query_emb = embedder.get_embedding(query)
    
    # Use fine_search_chunks to search all chunks
    results = fine_search_chunks(query_emb, chunk_index, top_k=top_k)
    
    return results

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    # Extended language detection based on character ranges
    language_patterns = {
        'ko': r'[가-힣]',  # Korean
        'en': r'[a-zA-Z]',  # English
        'zh': r'[一-龥]',  # Chinese
        'ja': r'[ぁ-ゔァ-ヴー]',  # Japanese (Hiragana + Katakana)
        'es': r'[áéíóúñÁÉÍÓÚÑ]',  # Spanish
        'fr': r'[àâçèéêëîïôùûÀÂÇÈÉÊËÎÏÔÙÛ]',  # French
        'de': r'[äöüßÄÖÜ]',  # German
        'ru': r'[а-яА-ЯёЁ]',  # Russian (Cyrillic)
        'ar': r'[ا-ي]',  # Arabic
        'hi': r'[अ-ह]',  # Hindi (Devanagari)
        'vi': r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]',  # Vietnamese
        'th': r'[ก-๙]',  # Thai
        'pt': r'[àáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ]',  # Portuguese
        'it': r'[àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]',  # Italian
        'tr': r'[çğıöşüÇĞİÖŞÜ]',  # Turkish
        'id': r'[a-zA-Z]',  # Indonesian (uses English alphabet)
        'ms': r'[a-zA-Z]',  # Malay (uses English alphabet)
        'pl': r'[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]',  # Polish
        'nl': r'[a-zA-Z]',  # Dutch (mostly English alphabet)
        'sv': r'[åäöÅÄÖ]',  # Swedish
        'he': r'[א-ת]',  # Hebrew
        'bn': r'[অ-ঔক-নপ-য়]',  # Bengali
        'ta': r'[அ-ஹ]',  # Tamil
        'te': r'[అ-హ]',  # Telugu
        'mr': r'[अ-ह]',  # Marathi
    }
    
    char_counts = {}
    
    # Count characters for each language
    for lang, pattern in language_patterns.items():
        chars = re.findall(pattern, text)
        char_counts[lang] = len(chars)
    
    # Special handling for languages that use English alphabet
    if char_counts['en'] > 0:
        # Check for specific words or patterns
        text_lower = text.lower()
        if any(word in text_lower for word in ['adalah', 'yang', 'dan', 'untuk', 'dengan']):
            char_counts['id'] = char_counts['en'] + 10  # Indonesian
        elif any(word in text_lower for word in ['yang', 'dan', 'untuk', 'adalah', 'dengan']):
            char_counts['ms'] = char_counts['en'] + 10  # Malay
        elif any(word in text_lower for word in ['het', 'een', 'van', 'voor', 'met']):
            char_counts['nl'] = char_counts['en'] + 10  # Dutch
    
    # Return the language with most characters
    if max(char_counts.values()) == 0:
        return 'ko'  # Default to Korean
    
    return max(char_counts, key=char_counts.get)

def get_language_instruction(lang: str) -> str:
    """Get language-specific instruction for the response."""
    instructions = {
        'ko': '한국어로 답변하세요.',
        'en': 'Please answer in English.',
        'zh': '请用中文回答。',
        'ja': '日本語で答えてください。',
        'es': 'Por favor responde en español.',
        'fr': 'Veuillez répondre en français.',
        'de': 'Bitte antworten Sie auf Deutsch.',
        'ru': 'Пожалуйста, ответьте на русском языке.',
        'ar': 'الرجاء الإجابة باللغة العربية.',
        'hi': 'कृपया हिंदी में उत्तर दें।',
        'vi': 'Vui lòng trả lời bằng tiếng Việt.',
        'th': 'กรุณาตอบเป็นภาษาไทย',
        'pt': 'Por favor, responda em português.',
        'it': 'Si prega di rispondere in italiano.',
        'tr': 'Lütfen Türkçe cevap verin.',
        'id': 'Silakan jawab dalam bahasa Indonesia.',
        'ms': 'Sila jawab dalam bahasa Melayu.',
        'pl': 'Proszę odpowiedzieć po polsku.',
        'nl': 'Antwoord alstublieft in het Nederlands.',
        'sv': 'Vänligen svara på svenska.',
        'he': 'אנא ענה בעברית.',
        'bn': 'দয়া করে বাংলায় উত্তর দিন।',
        'ta': 'தயவுசெய்து தமிழில் பதிலளிக்கவும்.',
        'te': 'దయచేసి తెలుగులో సమాధానం ఇవ్వండి.',
        'mr': 'कृपया मराठीत उत्तर द्या.'
    }
    return instructions.get(lang, '한국어로 답변하세요.')

###############################################################################
# Main chat loop
###############################################################################
def answer(q, chunk_index, top_k):
    results = direct_search(q, chunk_index, top_k)
            
    ctx = []
    for rank, result in enumerate(results, 1):
        score = result.get("score", 0)
        content = result.get("content", "")
        ctx.append(f"[{rank}] (S={score:.3f})\n{content}\n")
    
    # Detect language of the question
    lang = detect_language(q)
    lang_instruction = get_language_instruction(lang)
    
    # Create a more sophisticated prompt for Sogang University assistant
    prompt = f"""당신은 서강대학교 학생들을 위한 친절하고 도움이 되는 AI 도우미입니다. 
학생들의 캠퍼스 생활, 학업, 행정 절차 등 다양한 질문에 정확하고 유용한 답변을 제공합니다.
제공된 문맥을 바탕으로 질문에 답변하되, 문맥에 없는 내용은 추측하지 마세요.
답변은 명확하고 구체적이어야 합니다.

중요한 규칙:
- 문맥 번호나 출처를 언급하지 마세요 (예: "[문맥 1]에서 확인하실 수 있습니다" 같은 표현 금지)
- "더 자세한 내용은 ~에서 확인하세요" 같은 불필요한 안내는 하지 마세요
- 답변에 필요한 정보를 직접적으로 제공하세요

문맥:
{"\n".join(ctx)}

질문: {q}

{lang_instruction} 친절하고 도움이 되는 톤으로 답변하세요.

답변:"""

    # Generate answer using local_llm
    full_response = local_llm.generate(prompt)
    # Extract only the model's response
    ans = local_llm.strip_response(full_response)
    return ans
    
def chat_loop(args: argparse.Namespace):
    print("[INFO] Loading data …")
    chunk_index = load_chunks(args.chunks, args.vectors)
    print("[READY] Type your question — Ctrl+C to exit\n")
    print("[INFO] Using direct search to compare with all chunks\n")
    try:
        while True:
            q = input("🗨️  Q: ").strip()
            if not q:
                continue

            # Use direct search
            print("🔍 Searching all chunks...")
            ans = answer(q, chunk_index, args.topk)
            print("🤖 A:", ans, "\n")
    except KeyboardInterrupt:
        print("\n[EXIT]")

###############################################################################
# CLI
###############################################################################

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks/pages_text_chunks.json")
    p.add_argument("--vectors", default="data/index/pages_text_chunks_vectors.json")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args(argv)

###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    chat_loop(parse_args())