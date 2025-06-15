from __future__ import annotations
import faulthandler, sys; faulthandler.enable(file=sys.stderr)

# ───────────────────────────────── Standard library ──────────────────────────
import argparse
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Tuple

# ───────────────────────────────── Third-party ───────────────────────────────
import gradio as gr
import numpy as np

# ─────────────────────────── Import chatbot module ───────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from src import chatbot as rc

# ─────────────────────────── Global variables ────────────────────────────────
args = None
chunk_index = None

# ─────────────────────────── RAG answer helper ───────────────────────────────
def generate_answer(
    question: str,
    chunks_path: str | Path,
    vectors_path: str | Path,
    device: str,
    topk: int,
) -> str:
    return 

# ─────────────────────── Gradio chat function ────────────────────────────────
def chat_fn(message, history):
    """Gradio chat interface function."""
    global chunk_index, args
    
    answer = rc.answer(message, chunk_index, args.topk)
    
    if history is None:
        history = []
    
    # 딕셔너리 형식으로 메시지 추가
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    return "", history

# ───────────────────────────────────── Main ──────────────────────────────────
def main():
    global args, chunk_index
    
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks/pages_text_chunks.json")
    p.add_argument("--vectors", default="data/index/pages_text_chunks_vectors.json")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    # Custom CSS with mobile-responsive design (dark mode ignored)
    custom_css = """
    /* 다크모드 무시하고 항상 라이트 모드 사용 */
    * {
        color-scheme: light !important;
    }
    
    /* 모바일 반응형 디자인 */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 12px !important;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        .subtitle {
            font-size: 0.9rem !important;
        }
        
        #chatbot {
            height: calc(100vh - 280px) !important;
            min-height: 300px !important;
        }
    }
    
    /* 데스크톱 뷰 */
    @media (min-width: 769px) {
        #chatbot {
            height: 450px !important;
        }
    }
    
    /* 항상 라이트 모드 색상 사용 */
    .gradio-container {
        background-color: #fafafa !important;
        max-width: 900px !important;
        margin: auto !important;
        padding: 24px !important;
    }
    
    #chatbot {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        overflow-y: auto;
        background-color: white !important;
    }
    
    /* 채팅창 내부 컨테이너도 흰색으로 강제 */
    #chatbot .wrapper,
    #chatbot .container,
    #chatbot .message-wrap,
    #chatbot > div {
        background-color: white !important;
    }
    
    /* 다크모드에서도 채팅창은 항상 흰색 */
    @media (prefers-color-scheme: dark) {
        #chatbot,
        #chatbot .wrapper,
        #chatbot .container,
        #chatbot .message-wrap,
        #chatbot > div {
            background-color: white !important;
        }
    }
    
    .message-wrap {
        border-radius: 12px !important;
        margin: 8px 0;
    }
    
    .user.message {
        background-color: #f5f5f5 !important;
        border: 1px solid #A6A6A6 !important;
        color: #333 !important;
    }
    
    .user.message p,
    .user.message span,
    .user.message div {
        color: #333 !important;
    }
    
    .bot.message {
        background-color: #fff5f5 !important;
        border: 1px solid #B20000 !important;
        position: relative;
        padding-left: 16px !important;
        color: #333 !important;
    }
    
    .bot.message p,
    .bot.message span,
    .bot.message div,
    .bot.message li,
    .bot.message strong {
        color: #333 !important;
    }
    
    /* 중복 아바타 제거 */
    .bot .avatar-container > img:not(:first-child) {
        display: none !important;
    }
    
    #input-row {
        border-radius: 12px;
        background-color: #f9f9f9 !important;
        padding: 16px;
        margin-top: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #A6A6A6 !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #B20000 0%, #7D0000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 8px !important;
        font-weight: 700;
    }
    
    .subtitle {
        text-align: center;
        color: #A6A6A6 !important;
        font-size: 1.1rem;
        margin-bottom: 24px;
        font-weight: 500;
    }
    
    #send-btn {
        background: linear-gradient(135deg, #B20000 0%, #7D0000 100%);
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: white !important;
    }
    
    #send-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(178, 0, 0, 0.4);
        background: linear-gradient(135deg, #7D0000 0%, #B20000 100%);
    }
    
    #msg-input {
        border-radius: 8px;
        border: 2px solid #A6A6A6 !important;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease;
        background-color: white !important;
        color: #333 !important;
    }
    
    #msg-input::placeholder {
        color: #999 !important;
    }
    
    #msg-input:focus {
        border-color: #B20000 !important;
        outline: none;
        background-color: white !important;
        color: #333 !important;
    }
    
    .gradio-accordion {
        border: 1px solid #A6A6A6 !important;
        border-radius: 8px !important;
        margin-bottom: 16px;
        background-color: white !important;
    }
    
    .gradio-accordion .label-wrap {
        background-color: #f9f9f9 !important;
        border-radius: 8px 8px 0 0 !important;
        color: #333 !important;
    }
    
    .examples-holder {
        margin-top: 16px;
        border: 1px solid #A6A6A6 !important;
        border-radius: 8px;
        padding: 12px;
        background-color: #fafafa !important;
    }
    
    .examples-holder button {
        color: #333 !important;
        background-color: white !important;
    }
    
    .examples-holder button:hover {
        background-color: #f0f0f0 !important;
    }
    
    /* 모든 텍스트 강제로 검은색 */
    .markdown-text,
    .markdown-text p,
    .markdown-text span,
    label,
    span,
    p,
    div {
        color: #333 !important;
    }
    
    /* 스크롤바 스타일링 */
    #chatbot::-webkit-scrollbar {
        width: 8px;
    }
    
    #chatbot::-webkit-scrollbar-track {
        background: #f0f0f0;
    }
    
    #chatbot::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    #chatbot::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    """
    
    # Define AI avatar path
    AI_AVATAR_PATH = "assets/ai_profile.png"
    
    # Check if avatar exists
    avatar_path = None
    if Path(AI_AVATAR_PATH).exists():
        avatar_path = AI_AVATAR_PATH
        print(f"[INFO] Using AI avatar from: {AI_AVATAR_PATH}")
    else:
        print(f"[INFO] AI avatar not found at: {AI_AVATAR_PATH}. Using default.")
    
    # Load chunks
    chunk_index = rc.load_chunks(args.chunks, args.vectors)
    
    with gr.Blocks(
        title="Sogang AI Assistant", 
        css=custom_css, 
        theme=gr.themes.Soft()  # Soft 테마로 다시 변경
    ) as demo:
        gr.Markdown("# 🤖 Sogang AI Assistant")
        gr.Markdown("<p class='subtitle'>Powered by MIMIC Lab QueryDoc</p>")
        
        chatbot = gr.Chatbot(
            label="", 
            elem_id="chatbot",
            type='messages',
            avatar_images=(None, avatar_path),  # (user_avatar, assistant_avatar)
            show_label=False
        )

        with gr.Row(elem_id="input-row"):
            txt = gr.Textbox(
                placeholder="질문을 입력하세요... (예: 서강대학교의 역사에 대해 알려주세요)", 
                scale=4,
                elem_id="msg-input",
                show_label=False,
                container=False
            )
            btn = gr.Button("전송", variant="primary", scale=1, elem_id="send-btn")
        
        gr.Examples(
            examples=[
                "서강대학교의 건학 이념은 무엇인가요?",
                "서강대 컴퓨터공학과의 교육과정에 대해 설명해주세요",
                "서강대학교의 주요 연구 분야는 무엇인가요?",
                "서강대 입학 전형에 대해 알려주세요",
                "What is the history of Sogang University?",
                "请介绍一下西江大学的特点",
                "西江大学の国際交流プログラムについて教えてください"
            ],
            inputs=txt,
            label="예시 질문 (다국어 지원)"
        )

        # Event handlers
        txt.submit(chat_fn, inputs=[txt, chatbot], outputs=[txt, chatbot])
        btn.click(chat_fn, inputs=[txt, chatbot], outputs=[txt, chatbot])
        
        demo.queue(default_concurrency_limit=1, api_open=False)

    demo.launch(share=args.share, server_name='0.0.0.0', server_port=35798)

if __name__ == "__main__":
    main()