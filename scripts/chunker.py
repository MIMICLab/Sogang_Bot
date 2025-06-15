# scripts/chunker.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from typing import List, Dict, Any
from src.utils.text_cleaning import basic_clean_text

# Tune these two to balance chunk length and redundancy
CHUNK_SIZE = 1200  # characters per chunk
OVERLAP = 200      # characters of overlap between consecutive chunks

# --- Token helpers ---------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _tokenize(txt: str) -> list[int]:
        "tiktoken 기반 BPE 토크나이즈"
        return _enc.encode(txt)

    def _detokenize(tok_ids: list[int]) -> str:
        "BPE 토큰 ID → 문자열"
        return _enc.decode(tok_ids)

except ModuleNotFoundError:
    # ✅ tiktoken이 없거나 설치하지 않았을 때의 매우 단순한 대체 구현
    def _tokenize(txt: str) -> list[str]:
        "whitespace splitter fallback"
        return txt.split()

    def _detokenize(tokens: list[str]) -> str:
        return " ".join(tokens)
# --------------------------------------------------------------------------

def get_section_of_page(page_num: int, toc: List[List[Any]]) -> str:
    """
    Get the section title for a given page number based on the table of contents.
    toc: List of tuples (level, title, start_page)
    page_num: 0-indexed page number
    """ 
    current_section = "Others"
    for (lvl, title, start_p) in toc:
        if page_num + 1 >= start_p:
            current_section = title
        else:
            break
    return current_section

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of `chunk_size` characters with `overlap` characters
    of context shared between consecutive chunks.
    """
    text = basic_clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # avoid infinite loop if overlap >= size

    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= len(text):
            break
        start += step
    return chunks

def process_extracted_file(json_data: Any) -> List[Dict[str, Any]]:
    """
    두 가지 입력 형태를 모두 지원한다.
    1) {
         "file_path": "...",
         "toc": [(level, title, start_page), ...],
         "pages_text": ["...", "...", ...]
       }
    2) [ { "page_num": 0, "text": "..." }, ... ]
    """
    # --------- ① dict 형식 ---------
    if isinstance(json_data, dict):
        pdf_path   = json_data.get("file_path", "unknown.pdf")
        toc        = json_data.get("toc", [])
        pages_text = json_data.get("pages_text", [])
    # --------- ② list 형식 ---------
    elif isinstance(json_data, list):
        pdf_path   = "unknown.pdf"
        toc        = []  # TOC 정보가 없으면 빈 리스트
        pages_text = [page.get("text", "") for page in json_data]
    else:
        raise ValueError(f"Unsupported JSON structure: {type(json_data)}")

    # 이후 로직은 그대로 ↓
    chunked_result = []
    for page_idx, text in enumerate(pages_text):
        section_title = get_section_of_page(page_idx, toc)
        for chunk in split_into_chunks(basic_clean_text(text), CHUNK_SIZE, OVERLAP):
            chunked_result.append(
                {
                    "page_num" : page_idx,
                    "section"  : section_title,
                    "content"  : chunk,
                }
            )
    return chunked_result

from typing import List

def split_into_chunks(text: str, max_tokens: int, overlap: int = 0) -> List[str]:
    """
    텍스트를 토큰 수 기준으로 슬라이딩-윈도우 방식으로 자른다.

    Parameters
    ----------
    text        : 원본 문자열
    max_tokens  : 한 chunk당 최대 토큰 수
    overlap     : 이전 chunk와 겹치는 토큰 수 (0 이상)

    Returns
    -------
    List[str]   : chunk 문자열들의 리스트
    """
    # _tokenize / _detokenize 는 chunker.py 상단에서 이미 정의했을 거예요
    tok_ids = _tokenize(text)

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    step = max_tokens - overlap
    chunks = []
    for start in range(0, len(tok_ids), step):
        end = start + max_tokens
        chunk_tok = tok_ids[start:end]
        if not chunk_tok:
            break
        chunks.append(_detokenize(chunk_tok))

        if end >= len(tok_ids):
            break
    return chunks

if __name__ == "__main__":
    extracted_folder = "data/extracted"
    chunk_folder = "data/chunks"
    os.makedirs(chunk_folder, exist_ok=True)

    for fname in os.listdir(extracted_folder):
        # sections.json 파일은 건너뛰기
        if fname.endswith(".json") and fname != "sections.json":
            path = os.path.join(extracted_folder, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunked_data = process_extracted_file(data)

            base_name = os.path.splitext(fname)[0]
            out_json = os.path.join(chunk_folder, f"{base_name}_chunks.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print("Chunking Complete.")