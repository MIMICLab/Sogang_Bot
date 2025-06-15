# 서강대학교 요람 챗봇 (Sogang University Handbook Chatbot)

서강대학교 요람을 기반으로 한 질의응답 챗봇입니다. Coarse-to-Fine 검색 방식의 RAG(Retrieval-Augmented Generation) 기법을 활용하여 정확한 답변을 제공합니다.

## 프로젝트 구조
```bash
SogangBot/
├─ scripts/
│   ├─ pdf_extractor.py      # PDF 텍스트 추출
│   ├─ chunker.py            # 텍스트 청킹
│   ├─ build_index.py        # 벡터 인덱스 생성
│   └─ section_rep_builder.py # 섹션 대표 벡터 생성
├─ src/
│   ├─ inference/
│   │   ├─ embedding_model.py # 임베딩 모델
│   │   └─ llm_model.py      # LLM 모델
│   ├─ search/
│   │   ├─ section_coarse_search.py # 섹션 단위 검색
│   │   ├─ fine_search.py    # 세부 검색
│   │   └─ vector_search.py  # 벡터 검색
│   ├─ chatbot.py            # 챗봇 메인 로직
│   └─ utils/
│       ├─ __init__.py
│       └─ text_cleaning.py  # 텍스트 전처리
├─ data/
│   ├─ extracted/            # 추출된 텍스트
│   ├─ chunks/               # 청킹된 데이터
│   ├─ index/                # 벡터 인덱스
│   └─ original/             # 원본 PDF
│       └─ sogang_trimmed_pages_9_to_990.pdf
├─ web_demo.py               # 웹 데모 실행 파일
├─ requirements.txt          # 의존성 패키지
└─ README.md
```

## 설치 및 실행

### 1. 가상환경 설정 및 패키지 설치
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 전처리 (최초 1회)
```bash
# 벡터 인덱스 생성
python scripts/build_index.py

# 섹션 대표 벡터 생성
python scripts/section_rep_builder.py
```

### 3. 웹 데모 실행
```bash
# 로컬 실행 (MacBook M3)
python web_demo.py --device mps

# 공개 URL로 실행
python web_demo.py --device mps --share
```

## 주요 기능
- 서강대학교 요람 내용 기반 질의응답
- Coarse-to-Fine 검색으로 효율적인 정보 검색
- Gradio 웹 인터페이스 제공
- 한국어 자연어 처리 최적화

## 시스템 요구사항
- Python 3.8 이상
- macOS (M1/M2/M3) 또는 CUDA 지원 GPU
- 메모리: 8GB 이상 권장