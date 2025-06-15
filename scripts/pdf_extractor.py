# ==============================================
# scripts/pdf_extractor.py
# ==============================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → page‑text JSON & section JSON generator with optional **Gemma‑3‑1b‑it** heading refinement.
Designed to run **out‑of‑the‑box on MacBook Air M3** (Apple Silicon, `mps`) as well as on CPU/GPU.

Folder conventions
------------------
- **Input  PDF** : `data/original/<input_file>.pdf`
- **Outputs**    : `data/extracted/pages_text.json`, `sections.json`

Quick start (Apple Silicon)
---------------------------
```bash
# ① create & activate venv
python3 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# ② install PyTorch with MPS backend (≈ 400 MB wheel)
pip install torch==2.2.2 torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/metal
#   ↑ stable; for nightly replace "2.2.2" with "nightly"

# ③ rest of deps
pip install pymupdf transformers tiktoken

# ④ run extractor on the CPU (safe) or MPS (faster, 16‑bit)
python scripts/pdf_extractor.py --input_file my.pdf --use_llm --device mps
```

Key steps
---------
1. Extract plain text per page with **PyMuPDF**.
2. Heuristically detect candidate headings.
3. Optionally ask **Gemma‑3‑1b‑it** to clean/confirm headings.
4. Build `sections.json` with `{title, start_page, end_page, method:"TOC"}`.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import fitz  # PyMuPDF

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:  # LLM optional – pipeline still works without it
    AutoTokenizer = AutoModelForCausalLM = torch = None

ROOT = Path(__file__).resolve().parents[1]
DATA_ORIG  = ROOT / "data" / "original"
DATA_OUT   = ROOT / "data" / "extracted"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Heading detection (regex is language‑agnostic enough for KR/EN mixed docs)
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(\s{0,4}(\d+[.]){0,3}\s*[A-Z가-힣\d][^\n]{0,80})$", re.MULTILINE)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _dtype_for(device: str):
    if AutoModelForCausalLM is None:
        return None
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def load_llm(model_path: str, device: str) -> tuple:
    """Load Gemma (or compatible) in the most memory‑efficient way."""
    if AutoModelForCausalLM is None:
        raise ImportError("⚠️ transformers is not installed. `pip install transformers`.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=True,  # use HF_TOKEN env var by default
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",          # let HF place layers
        torch_dtype=_dtype_for(device),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return tokenizer, model


def ask_llm_for_titles(tok, model, noisy_titles: list[str]) -> list[str]:
    """Call LLM with a *short* prompt (<2 k tokens) to avoid massive masks.
    If the prompt would exceed that, we truncate the list and fall back to naïve clean‑up."""
    MAX_PROMPT_TOKENS = 2048  # keep well below Gemma's safe window

    # Build bullet list; stop early if it grows too long
    lines, tok_count = [], 0
    for t in noisy_titles:
        encoded = tok(t, add_special_tokens=False)["input_ids"]
        if tok_count + len(encoded) > MAX_PROMPT_TOKENS:
            break  # avoid OOM
        tok_count += len(encoded)
        lines.append(f"- {t}")

    prompt = (
        "Clean the following list of section titles extracted from a PDF. "
        "Return **only** a JSON array of the titles to keep (strings)."

" + "
".join(lines)"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    text = tok.decode(output[0], skip_special_tokens=True)
    try:
        cleaned: list[str] = json.loads(text[text.index("["):])
        return [t.strip() for t in cleaned if t.strip()]
    except Exception:
        # fall back: keep them all
        return noisy_titles

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    return [{"page_num": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]


def detect_sections(pages: list[dict], *, use_llm: bool, tok=None, model=None):
    # 1) Collect heading candidates with page numbers
    candidates: list[tuple[int, str]] = []
    for p in pages:
        for m in HEADING_RE.finditer(p["text"]):
            candidates.append((p["page_num"], m.group(1).strip()))

    titles = [t for _, t in candidates]
    if use_llm and titles:
        titles = ask_llm_for_titles(tok, model, titles)

    sections = []
    if titles:
        for idx, (page_num, raw_title) in enumerate(candidates):
            if raw_title not in titles:
                continue  # filtered out by LLM
            start = page_num
            end = pages[-1]["page_num"] if idx == len(candidates) - 1 else candidates[idx + 1][0] - 1
            sections.append({"title": raw_title, "start_page": start, "end_page": end, "method": "TOC"})
    # Fallback – whole doc is one section
    if not sections:
        sections = [{"title": "Document", "start_page": 1, "end_page": pages[-1]["page_num"], "method": "TOC"}]
    return sections

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="PDF filename located in data/original")
    ap.add_argument("--use_llm", action="store_true", help="use Gemma to clean headings")
    ap.add_argument("--model_path", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="execution device")
    args = ap.parse_args()

    pdf_path = DATA_ORIG / args.input_file
    if not pdf_path.exists():
        sys.exit(f"❌ Input not found: {pdf_path}")

    print("[1/3] Extracting pages …")
    pages = extract_pages(pdf_path)

    tok = model = None
    if args.use_llm:
        print("[2/3] Loading Gemma … (may take 30 s on first run)")
        tok, model = load_llm(args.model_path, args.device)
    else:
        print("[2/3] Skipping LLM refinement …")

    print("[3/3] Detecting sections …")
    sections = detect_sections(pages, use_llm=args.use_llm, tok=tok, model=model)

    (DATA_OUT / "pages_text.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), "utf-8")
    (DATA_OUT / "sections.json").write_text(json.dumps(sections, ensure_ascii=False, indent=2), "utf-8")
    print("✅ pages_text.json & sections.json written to", DATA_OUT)

if __name__ == "__main__":
    main()

