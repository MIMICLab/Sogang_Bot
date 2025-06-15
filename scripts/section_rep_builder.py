# scripts/section_rep_builder.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from src.inference.embedding_model import embedder
from glob import glob

def build_section_reps(sections, chunk_index):
    """
    sections: [
      { "title": "2장 설치방법", "start_page":10, "end_page":19, ... },
      ...
    ]
    chunk_index: [{ "embedding": [...], "metadata": {"section_title": "...", ...}}, ...]

    => 각 섹션에
       sec["title_emb"], sec["avg_chunk_emb"] 필드를 추가해 반환
    """
    # 1) 섹션 제목 임베딩 (batch)
    titles = [sec["title"] for sec in sections]
    title_embs = embedder.get_embeddings(titles)  # shape: (num_sections, dim)
    for i, sec in enumerate(sections):
        sec["title_emb"] = title_embs[i].tolist()

    # 2) 섹션별 청크 모으기
    section2embs = {}
    for item in chunk_index:
        sec_t = item["metadata"]["section_title"]
        emb = item["embedding"]  # list[float]
        if sec_t not in section2embs:
            section2embs[sec_t] = []
        section2embs[sec_t].append(emb)

    # 3) 섹션 내부 청크들의 평균 임베딩
    for sec in sections:
        stitle = sec["title"]
        if stitle not in section2embs:
            sec["avg_chunk_emb"] = None
        else:
            arr = np.array(section2embs[stitle])  # shape: (num_chunks, emb_dim)
            avg_vec = arr.mean(axis=0)            # (emb_dim,)
            sec["avg_chunk_emb"] = avg_vec.tolist()
    
    return sections

# pdf의 이름이 바뀌어도 chunk_index_json 파일을 찾아들어갈 수 있도록 수정

def find_one_vectors_file(index_dir="data/index"):
    """_vectors.json 파일 중 하나를 자동으로 선택"""
    vector_files = sorted(glob(os.path.join(index_dir, "*_vectors.json")))
    if not vector_files:
        raise FileNotFoundError("No *_vectors.json files found in index directory.")
    return vector_files[0]  # 알파벳순으로 첫 번째 파일 선택

if __name__ == "__main__":
    # 예시: data/extracted/sections.json (목차 기반 섹션 정보)
    sections_json = "data/extracted/sections.json"

    # _vectors.json 파일 중 하나 자동 선택
    chunk_index_json = find_one_vectors_file()

    print(f"[INFO] Using chunk index file: {chunk_index_json}")

    with open(sections_json, 'r', encoding='utf-8') as f:
        sections_data = json.load(f)

    with open(chunk_index_json, 'r', encoding='utf-8') as f:
        chunk_index_data = json.load(f)

    # 섹션 대표 벡터 생성
    updated_sections = build_section_reps(sections_data, chunk_index_data)

    # 저장(예: data/extracted/sections_with_emb.json)
    out_path = "data/extracted/sections_with_emb.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(updated_sections, f, ensure_ascii=False, indent=2)

    print("✅ Section reps built and saved.")