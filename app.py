import base64
import html
import os
import pathlib
import re
import time
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from lib.psychology_codex import map_narrative_to_superpowers
from lib.coaching_agent import generate_socratic_stream
from lib.outcome_engine import synthesize_single_tile
from lib.storybook_generator import (
    generate_comic_book,
    generate_pixel_art_illustration,
)
from lib.voice_agent import transcribe_audio


# ============================================================
# APP CONFIG
# ============================================================

APP_TITLE = "ArcMotivate"
OPENING_MSG = (
    "👾 **System Online — ArcMotivate**\n\n"
    "Have you ever wondered what career you want to pursue when you grow up? "
    "Well, you're in the right place! ArcMotivate is here to help you discover your passions, strengths, and potential career paths. \n\n"
    "Before we build your future, I want to get to *know* you. You can tell me what you love, what you don't, or even a memorable event from your life."
)

STREAM_UPDATE_INTERVAL_SEC = 0.05
MAX_CHAT_CONTEXT_MESSAGES = 12
MAX_TILE_HISTORY_MESSAGES = 6
MAX_INLINE_IMAGES_PER_TURN = 2

# ============================================================
# CSS — LARGER FONTS, LOWER RENDER COST, STILL STYLISH
# ============================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Press+Start+2P&family=VT323&display=swap');

:root {
    --bg-0: #05030b;
    --bg-1: #0a0314;
    --bg-2: #15092a;
    --panel: rgba(14, 10, 34, 0.86);
    --surface: rgba(20, 10, 40, 0.9);

    --text: #f8fafc;
    --text-soft: #dbe7f3;
    --text-dim: #9fb0c4;

    --cyan: #22d3ee;
    --pink: #ff4de3;
    --violet: #7c3aed;

    --border-cyan: rgba(34, 211, 238, 0.2);
    --border-violet: rgba(124, 58, 237, 0.24);

    --radius-sm: 12px;
    --radius-md: 14px;
    --radius-lg: 18px;

    --shadow-1: 0 6px 18px rgba(0, 0, 0, 0.24);
    --shadow-2: 0 10px 24px rgba(0, 0, 0, 0.34);

    --font-ui: 'VT323', monospace;
    --font-display: 'Press Start 2P', cursive;
    --font-accent: 'Orbitron', sans-serif;

    --control-h: 52px;
    --container-w: 1400px;
}

*, *::before, *::after {
    box-sizing: border-box;
}

html, body {
    min-height: 100%;
}

body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    background: radial-gradient(ellipse at 20% 40%, var(--bg-2) 0%, var(--bg-1) 58%, #000 100%) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.gradio-container {
    max-width: var(--container-w) !important;
    margin: 0 auto !important;
    padding: 16px 14px 20px !important;
    border: none !important;
}

/* Header */
.arc-header {
    text-align: center;
    padding: 10px 12px 14px;
}

.arc-logo {
    font-family: var(--font-display);
    font-size: clamp(1.35rem, 2.2vw, 2.2rem);
    line-height: 1.35;
    letter-spacing: 1px;
    color: var(--pink);
    text-shadow: 0 0 6px rgba(255, 77, 227, 0.42), 0 0 14px rgba(34, 211, 238, 0.14);
    padding-bottom: 8px;
}

.arc-tagline {
    font-size: clamp(1rem, 1.4vw, 1.3rem);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--cyan);
    opacity: 0.92;
}

/* Chat shell */
.chat-wrap {
    background: linear-gradient(180deg, rgba(17, 10, 38, 0.88), rgba(10, 5, 24, 0.9)) !important;
    border: 1px solid var(--border-cyan) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-2) !important;
}

.gradio-chatbot,
.gradio-chatbot > div,
.gradio-chatbot .panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-chatbot {
    padding: 10px !important;
    scrollbar-width: thin;
    scrollbar-color: rgba(34, 211, 238, 0.35) transparent;
}

.gradio-chatbot .message {
    max-width: min(84%, 820px) !important;
    width: fit-content !important;
    margin-bottom: 10px !important;
    padding: 14px 16px !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow-1) !important;
}

.gradio-chatbot .message.bot {
    margin-left: 6px !important;
    margin-right: auto !important;
    background: rgba(34, 211, 238, 0.09) !important;
    border: 1px solid rgba(34, 211, 238, 0.18) !important;
    border-bottom-left-radius: 6px !important;
}

.gradio-chatbot .message.user {
    margin-left: auto !important;
    margin-right: 6px !important;
    background: rgba(124, 58, 237, 0.13) !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    border-bottom-right-radius: 6px !important;
}

.gradio-chatbot .avatar-container,
.gradio-chatbot .message-role {
    display: none !important;
}

/* Chat text */
.gradio-chatbot .prose,
.gradio-chatbot .message * {
    font-family: var(--font-ui) !important;
}

.gradio-chatbot .prose {
    color: var(--text) !important;
    font-size: clamp(1.22rem, 1.35vw, 1.42rem) !important;
    line-height: 1.42 !important;
    padding: 0 !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

.gradio-chatbot .prose p,
.gradio-chatbot .prose ul,
.gradio-chatbot .prose ol,
.gradio-chatbot .prose pre,
.gradio-chatbot .prose blockquote {
    margin-bottom: 0.6em !important;
}

.gradio-chatbot .prose p:last-child,
.gradio-chatbot .prose ul:last-child,
.gradio-chatbot .prose ol:last-child,
.gradio-chatbot .prose pre:last-child,
.gradio-chatbot .prose blockquote:last-child {
    margin-bottom: 0 !important;
}

.gradio-chatbot .prose strong {
    color: #fff !important;
    font-weight: 700 !important;
}

.gradio-chatbot .prose a {
    color: var(--cyan) !important;
    text-decoration: underline !important;
    text-underline-offset: 2px;
}

.gradio-chatbot .prose code {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 6px !important;
    padding: 0.04em 0.32em !important;
    color: #d8fbff !important;
    font-size: 0.9em !important;
}

.gradio-chatbot .prose pre {
    background: rgba(3, 7, 18, 0.75) !important;
    border: 1px solid rgba(34, 211, 238, 0.1) !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    overflow-x: auto !important;
}

.gradio-chatbot .prose blockquote {
    border-left: 3px solid rgba(34, 211, 238, 0.28) !important;
    padding-left: 10px !important;
    color: var(--text-soft) !important;
}

/* Input row */
.input-row {
    display: flex !important;
    align-items: stretch !important;
    gap: 10px !important;
    margin-top: 12px !important;
    padding: 10px !important;
    background: linear-gradient(180deg, rgba(14, 8, 34, 0.94), rgba(10, 5, 24, 0.93)) !important;
    border: 1px solid var(--border-cyan) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-1) !important;
}

.input-row > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.input-row > div:first-child,
.input-row > div:last-child {
    display: flex !important;
    align-items: stretch !important;
}

.gradio-textbox {
    flex: 1 1 auto !important;
    display: flex !important;
    align-items: stretch !important;
}

.gradio-textbox textarea {
    min-height: var(--control-h) !important;
    height: var(--control-h) !important;
    max-height: 180px !important;
    padding: 10px 12px !important;
    font-size: 1.28rem !important;
    line-height: 1.15 !important;
    background: var(--surface) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
    border-radius: 12px !important;
    resize: vertical !important;
}

.gradio-textbox textarea::placeholder {
    color: rgba(34, 211, 238, 0.52) !important;
}

.gradio-textbox textarea:focus {
    outline: none !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.12) !important;
}

/* Buttons */
.arc-btn,
button.primary,
button.secondary {
    height: var(--control-h) !important;
    min-height: var(--control-h) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 18px !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    font-family: var(--font-accent) !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    color: #fff !important;
    box-shadow: var(--shadow-1) !important;
}

.input-row > div:last-child button {
    width: 100% !important;
}

.arc-btn,
button.primary {
    background: linear-gradient(135deg, var(--violet), var(--cyan)) !important;
}

button.secondary {
    background: linear-gradient(135deg, #364152, #64748b) !important;
}

/* Tabs */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    display: flex;
    gap: 6px;
    margin-bottom: 10px !important;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(34, 211, 238, 0.12) !important;
    overflow-x: auto;
    scrollbar-width: none;
}

.tab-nav::-webkit-scrollbar {
    display: none;
}

.tab-nav button {
    font-family: var(--font-ui) !important;
    font-size: 1.1rem !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: 6px 12px !important;
    flex: 0 0 auto;
}

.tab-nav button.selected {
    color: var(--cyan) !important;
    border-color: rgba(34, 211, 238, 0.16) !important;
    background: rgba(34, 211, 238, 0.06) !important;
}

/* Canvas */
.canvas-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 12px;
    padding: 12px 0;
}

.canvas-tile {
    overflow: hidden;
    background: rgba(15, 10, 46, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 14px;
    box-shadow: var(--shadow-1);
}

.tile-img,
.tile-img-placeholder {
    width: 100%;
    height: 118px;
    border-bottom: 1px solid rgba(34, 211, 238, 0.12);
}

.tile-img {
    object-fit: cover;
    display: block;
}

.tile-img-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #1e1b4b, #0f0a2e);
    color: #7c3aed;
    font-size: 1.45rem;
}

.tile-body {
    padding: 12px;
}

.tile-category {
    margin-bottom: 5px;
    color: #c084fc;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: var(--font-accent);
}

.tile-title {
    margin: 0 0 5px;
    color: #f0f4ff;
    font-size: 1.08rem;
    line-height: 1.2;
}

.tile-desc {
    color: var(--text-soft);
    font-size: 0.98rem;
    line-height: 1.3;
    margin-bottom: 8px;
}

.tile-links {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}

.tile-link {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(34, 211, 238, 0.08);
    color: var(--cyan);
    text-decoration: none;
    font-size: 0.9rem;
    border: 1px solid rgba(34, 211, 238, 0.12);
}

/* Empty / loading */
.canvas-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 14px;
    padding: 42px 18px;
    text-align: center;
    color: #64748b;
    border: 1px dashed rgba(139, 92, 246, 0.18);
    border-radius: 14px;
    background: rgba(255,255,255,0.012);
}

.canvas-empty-icon {
    font-size: 2.4rem;
    opacity: 0.35;
}

.canvas-empty-text {
    max-width: 250px;
    font-size: 1rem;
    line-height: 1.35;
    color: #7b8ba3;
}

.synth-spinner {
    padding: 16px;
    text-align: center;
    color: #a78bfa;
    font-size: 1rem;
    font-style: italic;
}

.chat-comic-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 8px;
    margin-top: 12px;
}

.chat-comic-grid img {
    display: block;
    width: 100%;
    height: 100px;
    object-fit: cover;
    border: 1px solid rgba(34, 211, 238, 0.16);
    border-radius: 8px;
}

.comic-panel {
    background: rgba(15, 10, 46, 0.8);
    border: 1px solid rgba(124, 58, 237, 0.18);
    border-radius: 14px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: var(--shadow-1);
}

.comic-img {
    width: 100%;
    border-radius: 8px;
    display: block;
    margin-bottom: 8px;
}

.comic-cap {
    color: var(--text-soft);
    font-size: 1rem;
    line-height: 1.3;
    margin: 0;
}

/* Mobile */
@media (max-width: 768px) {
    :root {
        --control-h: 48px;
    }

    .gradio-container {
        padding: 10px 8px 16px !important;
    }

    .arc-header {
        padding: 8px 8px 12px;
    }

    .arc-logo {
        font-size: clamp(1.05rem, 7vw, 1.5rem);
        letter-spacing: 0.5px;
        padding-bottom: 6px;
    }

    .arc-tagline {
        font-size: 0.95rem;
        letter-spacing: 1px;
    }

    .gradio-chatbot {
        padding: 8px !important;
    }

    .gradio-chatbot .message {
        max-width: 96% !important;
        padding: 12px 13px !important;
        margin-bottom: 8px !important;
        border-radius: 14px !important;
    }

    .gradio-chatbot .prose {
        font-size: 1.12rem !important;
        line-height: 1.34 !important;
    }

    .input-row {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 8px !important;
        padding: 8px !important;
    }

    .gradio-textbox textarea {
        height: auto !important;
        min-height: var(--control-h) !important;
        max-height: 140px !important;
        font-size: 1.12rem !important;
        padding: 10px 12px !important;
    }

    .arc-btn,
    button.primary,
    button.secondary {
        width: 100% !important;
        font-size: 0.88rem !important;
        padding: 0 14px !important;
    }

    .tab-nav {
        gap: 4px;
        margin-bottom: 8px !important;
        padding-bottom: 6px;
    }

    .tab-nav button {
        font-size: 1rem !important;
        padding: 6px 10px !important;
    }

    .canvas-grid {
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        padding: 10px 0;
    }

    .tile-img,
    .tile-img-placeholder {
        height: 92px;
    }

    .tile-body {
        padding: 10px;
    }

    .tile-title {
        font-size: 0.98rem;
    }

    .tile-desc {
        font-size: 0.9rem;
    }

    .tile-link {
        font-size: 0.82rem;
    }

    .chat-comic-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }

    .chat-comic-grid img {
        height: 88px;
    }

    .comic-cap {
        font-size: 0.94rem;
    }
}

@media (max-width: 480px) {
    .canvas-grid {
        grid-template-columns: 1fr;
    }

    .arc-logo {
        line-height: 1.45;
    }

    .gradio-chatbot .prose {
        font-size: 1.06rem !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
}
"""


# ============================================================
# HELPERS
# ============================================================

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts).strip()
    return str(content or "").strip()


def _extract_file_path(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "path", None) or getattr(file_obj, "name", None)


def clean_message_for_backend(content: Any) -> str:
    text = extract_text(content)
    if not text:
        return ""

    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_backend_history(history: List[Dict[str, Any]], max_messages: int = MAX_CHAT_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
    backend_history: List[Dict[str, str]] = []

    for msg in history[-max_messages:]:
        clean_content = clean_message_for_backend(msg.get("content", ""))
        if not clean_content:
            continue
        role = "user" if msg.get("role") == "user" else "model"
        backend_history.append({"role": role, "text": clean_content})

    return backend_history


def extract_tool_json_and_display_text(accumulated_text: str) -> Tuple[str, Optional[str], bool]:
    """
    Hide streamed tool JSON from the user and recover an image prompt if present.
    Returns:
        display_text, recovered_prompt, is_streaming_json
    """
    match = re.search(r'\{\s*"action"\s*:\s*"dalle', accumulated_text, flags=re.DOTALL)
    if not match:
        return accumulated_text.strip(), None, False

    start_idx = match.start()
    open_braces = 0
    end_idx = -1

    for i in range(start_idx, len(accumulated_text)):
        ch = accumulated_text[i]
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
            if open_braces == 0:
                end_idx = i
                break

    if end_idx == -1:
        visible = accumulated_text[:start_idx].strip()
        if visible:
            visible = visible + "\n\n🧠 *Illustrating...*"
        else:
            visible = "🧠 *Thinking...*"
        return visible, None, True

    json_block = accumulated_text[start_idx : end_idx + 1]
    display_text = (accumulated_text[:start_idx] + accumulated_text[end_idx + 1 :]).strip()

    prompt_match = re.search(r'"prompt"\s*:\s*"(.*?)"\s*[,}]', json_block, flags=re.DOTALL)
    recovered_prompt = None
    if prompt_match:
        recovered_prompt = (
            prompt_match.group(1)
            .replace('\\"', '"')
            .replace("\\n", " ")
            .strip()
        )

    return display_text, recovered_prompt, False


def sanitize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    return ""


def safe_text(value: Any) -> str:
    return html.escape(str(value or ""))


@lru_cache(maxsize=96)
def cached_pixel_art(prompt: str) -> Optional[str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    return generate_pixel_art_illustration(prompt)


def maybe_generate_inline_visual(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return None
    try:
        return cached_pixel_art(prompt)
    except Exception as exc:
        print(f"Inline image error: {exc}")
        return None


def format_inline_images(image_b64_list: List[str]) -> str:
    if not image_b64_list:
        return ""

    tags = []
    for b64 in image_b64_list[:MAX_INLINE_IMAGES_PER_TURN]:
        safe_b64 = html.escape(b64, quote=True)
        tags.append(f"<img src='data:image/png;base64,{safe_b64}' alt='arc visual'>")

    return f"<div class='chat-comic-grid'>{''.join(tags)}</div>"


def format_canvas(tiles: List[Dict[str, Any]]) -> str:
    if not tiles:
        return """
        <div class='canvas-empty'>
            <div class='canvas-empty-icon'>🎮</div>
            <div class='canvas-empty-text'>Start exploring to unlock your unique Canvas Tiles.</div>
        </div>
        """

    html_parts = ["<div class='canvas-grid'>"]

    for tile in reversed(tiles):
        image_b64 = tile.get("image_b64")
        if image_b64:
            img_html = (
                f"<img src='data:image/png;base64,{html.escape(image_b64, quote=True)}' "
                f"class='tile-img' alt='tile art'>"
            )
        else:
            img_html = "<div class='tile-img-placeholder'>🎨</div>"

        links_html = []
        for link in tile.get("links") or []:
            if isinstance(link, dict):
                label = safe_text(link.get("label", "Explore"))
                url = sanitize_url(link.get("url", ""))
                if url:
                    links_html.append(
                        f"<a href='{html.escape(url, quote=True)}' target='_blank' "
                        f"rel='noopener noreferrer' class='tile-link'>🔗 {label}</a>"
                    )

        html_parts.append(
            f"""
            <div class='canvas-tile'>
                {img_html}
                <div class='tile-body'>
                    <div class='tile-category'>{safe_text(tile.get('category', 'Insight'))}</div>
                    <h3 class='tile-title'>{safe_text(tile.get('title', '—'))}</h3>
                    <p class='tile-desc'>{safe_text(tile.get('content', ''))}</p>
                    <div class='tile-links'>{"".join(links_html)}</div>
                </div>
            </div>
            """
        )

    html_parts.append("</div>")
    return "".join(html_parts)


# ============================================================
# USER INPUT HANDLERS
# ============================================================

def user_submit(user_message: Any, history: Optional[List[Dict[str, Any]]], state_data: Dict[str, Any]):
    history = history or []

    if isinstance(user_message, dict):
        text = (user_message.get("text") or "").strip()
        files = user_message.get("files") or []
    else:
        text = str(user_message or "").strip()
        files = []

    if not text and not files:
        return gr.update(value={"text": "", "files": []}), history, state_data

    if files:
        try:
            img_path = _extract_file_path(files[0])
            if img_path:
                path = pathlib.Path(img_path)
                img_bytes = path.read_bytes()
                ext = path.suffix.lower().lstrip(".")
                mime = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp",
                    "gif": "image/gif",
                }.get(ext, "image/jpeg")
                state_data["pending_image"] = {"bytes": img_bytes, "mime": mime}
            else:
                state_data.pop("pending_image", None)
        except Exception as exc:
            print(f"Image read error: {exc}")
            state_data.pop("pending_image", None)
    else:
        state_data.pop("pending_image", None)

    display_content = text or "📎 [image attached]"
    history.append({"role": "user", "content": display_content})

    return gr.update(value={"text": "", "files": []}), history, state_data


def handle_voice_submit(audio_path: Optional[str], history: Optional[List[Dict[str, Any]]], state_data: Dict[str, Any]):
    if not audio_path:
        return gr.update(value=None), history or [], state_data

    try:
        audio_bytes = pathlib.Path(audio_path).read_bytes()
        mime = "audio/wav" if str(audio_path).lower().endswith(".wav") else "audio/webm"
        text = transcribe_audio(audio_bytes, mime) or "🎤 [Audio transcription failed]"
    except Exception as exc:
        print(f"Audio transcription error: {exc}")
        text = "🎤 [Audio transcription failed]"

    history = history or []
    history.append({"role": "user", "content": text})

    return gr.update(value=None), history, state_data


# ============================================================
# MAIN SIMULATION PIPELINE
# ============================================================

def process_simulation(
    history: List[Dict[str, Any]],
    state_data: Dict[str, Any],
) -> Generator[Tuple[List[Dict[str, Any]], Any], None, None]:
    if not history:
        yield history, gr.update()
        return

    state_data.setdefault("superpowers", {})
    state_data.setdefault("tiles", [])
    state_data.setdefault("turn_count", 0)

    latest_user_input = clean_message_for_backend(history[-1].get("content", ""))

    if not state_data["superpowers"]:
        pending = state_data.get("pending_image") or {}
        img_bytes = pending.get("bytes")
        img_mime = pending.get("mime", "image/jpeg")
        try:
            state_data["superpowers"] = map_narrative_to_superpowers(
                latest_user_input,
                img_bytes,
                img_mime,
            )
        except Exception as exc:
            print(f"Superpower mapping error: {exc}")
            state_data["superpowers"] = {}

    history.append({"role": "assistant", "content": ""})

    backend_history = build_backend_history(history[:-1], max_messages=MAX_CHAT_CONTEXT_MESSAGES)

    pending = state_data.pop("pending_image", {}) or {}
    img_bytes = pending.get("bytes")
    img_mime = pending.get("mime", "image/jpeg")

    accumulated_text = ""
    inline_images: List[str] = []
    recovered_prompt: Optional[str] = None
    last_ui_flush = 0.0

    try:
        stream = generate_socratic_stream(
            state_data["superpowers"],
            backend_history,
            img_bytes,
            img_mime,
        )
    except Exception as exc:
        print(f"Stream setup error: {exc}")
        history[-1]["content"] = "⚠️ Something glitched while starting the simulation."
        yield history, format_canvas(state_data["tiles"])
        return

    for chunk in stream:
        chunk_type = chunk.get("type")

        if chunk_type == "text":
            accumulated_text += chunk.get("data", "")
            display_text, prompt_from_json, _ = extract_tool_json_and_display_text(accumulated_text)

            if prompt_from_json and not recovered_prompt:
                recovered_prompt = prompt_from_json

            now = time.monotonic()
            if (now - last_ui_flush) >= STREAM_UPDATE_INTERVAL_SEC:
                history[-1]["content"] = display_text or "🧠 *Thinking...*"
                yield history, gr.update()
                last_ui_flush = now

        elif chunk_type == "image":
            image_b64 = chunk.get("data")
            if image_b64 and len(inline_images) < MAX_INLINE_IMAGES_PER_TURN:
                inline_images.append(image_b64)
                base_text = history[-1]["content"] or extract_tool_json_and_display_text(accumulated_text)[0]
                history[-1]["content"] = (base_text + "\n\n" + format_inline_images(inline_images)).strip()
                yield history, gr.update()

    final_display_text, _, _ = extract_tool_json_and_display_text(accumulated_text)
    history[-1]["content"] = final_display_text or "…"

    if recovered_prompt and len(inline_images) < MAX_INLINE_IMAGES_PER_TURN:
        yield history, "<div class='synth-spinner'>🖼️ Illustrating...</div>"
        img_b64 = maybe_generate_inline_visual(recovered_prompt)
        if img_b64:
            inline_images.append(img_b64)

    if inline_images:
        history[-1]["content"] = (final_display_text + "\n\n" + format_inline_images(inline_images)).strip()

    yield history, gr.update()

    # Canvas synthesis after the main chat feels finished
    tiles = state_data["tiles"]
    yield history, format_canvas(tiles) + "<div class='synth-spinner'>🧩 Synthesizing your next canvas tile...</div>"

    tile_history = build_backend_history(history[:-1], max_messages=MAX_TILE_HISTORY_MESSAGES)

    try:
        new_tile_data = synthesize_single_tile(tile_history, state_data["superpowers"])
    except Exception as exc:
        print(f"Tile synthesis error: {exc}")
        new_tile_data = None

    if new_tile_data:
        image_prompt = new_tile_data.get("image_prompt", "").strip()
        if image_prompt:
            try:
                new_tile_data["image_b64"] = cached_pixel_art(image_prompt)
            except Exception as exc:
                print(f"Tile art error: {exc}")
                new_tile_data["image_b64"] = None

        tiles.append(new_tile_data)

    state_data["turn_count"] += 1
    yield history, format_canvas(tiles)


# ============================================================
# COMIC GENERATION
# ============================================================

def generate_comic(history: List[Dict[str, Any]]):
    if not history or len(history) < 2:
        yield (
            "<p style='color:#f87171; text-align:center; padding:20px; font-size:1.2rem;'>"
            "⚠️ Chat with Arc first — it needs material to build your comic!"
            "</p>"
        )
        return

    yield "<div class='synth-spinner'>🖌️ Writing your story and drawing the panels...</div>"

    clean_history: List[Dict[str, str]] = []
    for msg in history[-10:]:
        role = "model" if msg.get("role") == "assistant" else "user"
        clean_history.append({"role": role, "text": clean_message_for_backend(msg.get("content", ""))})

    try:
        panels = generate_comic_book(clean_history)
    except Exception as exc:
        print(f"Comic generation error: {exc}")
        panels = None

    if not panels:
        yield (
            "<p style='color:#f87171; text-align:center; padding:20px; font-size:1.2rem;'>"
            "⚠️ Comic generation failed. Please try again."
            "</p>"
        )
        return

    final_html_parts: List[str] = []

    for panel in panels:
        image_b64 = panel.get("image_b64")
        img_tag = ""
        if image_b64:
            img_tag = (
                f"<img src='data:image/png;base64,{html.escape(image_b64, quote=True)}' "
                f"class='comic-img' alt='comic panel'>"
            )

        caption = safe_text(panel.get("caption", ""))
        final_html_parts.append(
            f"""
            <div class='comic-panel'>
                {img_tag}
                <p class='comic-cap'>"{caption}"</p>
            </div>
            """
        )

    yield "".join(final_html_parts)


# ============================================================
# UI LOCKING
# ============================================================

def lock_ui():
    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def unlock_ui():
    return (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(css=css, theme=gr.themes.Base(), title=APP_TITLE) as demo:
    state = gr.State({"superpowers": {}, "tiles": [], "turn_count": 0})

    gr.HTML(
        """
        <div class='arc-header'>
            <div class='arc-logo'>🕹️ ArcMotivate</div>
            <div class='arc-tagline'>Interactive Career Explorer</div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=380):
            gr.HTML("<div style='height:6px'></div>")

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": OPENING_MSG}],
                height=620,
                show_label=False,
                elem_classes=["chat-wrap"],
                render_markdown=True,
            )

            with gr.Tabs():
                with gr.TabItem("⌨️ Text / Image"):
                    with gr.Row(elem_classes=["input-row"]):
                        msg_input = gr.MultimodalTextbox(
                            placeholder="Share anything — type, attach a photo, or both 📎",
                            show_label=False,
                            container=False,
                            file_types=["image"],
                            scale=9,
                        )
                        submit_btn = gr.Button("Launch 🚀", variant="primary", scale=1)

                with gr.TabItem("🎙️ Voice"):
                    gr.HTML(
                        "<p style='font-size:1.15rem; color:#94a3b8; margin-bottom:8px; padding-top:8px;'>"
                        "Speak naturally. Arc will transcribe, respond, and build your canvas."
                        "</p>"
                    )
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        show_label=False,
                    )

        with gr.Column(scale=4, min_width=340):
            gr.HTML("<div style='height:6px'></div>")

            with gr.Tabs():
                with gr.TabItem("🖼️ Live Canvas"):
                    canvas_output = gr.HTML(format_canvas([]))

                with gr.TabItem("📖 Career Comic"):
                    gr.HTML(
                        "<p style='text-align:center; color:#94a3b8; font-size:1.15rem; padding:16px 0 6px;'>"
                        "Turn your journey into a custom 3-panel comic."
                        "</p>"
                    )
                    comic_btn = gr.Button("🎨 Generate Comic Book", variant="secondary")
                    comic_output = gr.HTML("")

    # Input events
    submit_event = msg_input.submit(
        user_submit,
        [msg_input, chatbot, state],
        [msg_input, chatbot, state],
        queue=False,
    )

    btn_event = submit_btn.click(
        user_submit,
        [msg_input, chatbot, state],
        [msg_input, chatbot, state],
        queue=False,
    )

    voice_event = voice_input.stop_recording(
        handle_voice_submit,
        [voice_input, chatbot, state],
        [voice_input, chatbot, state],
        queue=False,
    )

    # Lock UI while processing
    submit_event.then(lock_ui, outputs=[msg_input, submit_btn, voice_input], queue=False)
    btn_event.then(lock_ui, outputs=[msg_input, submit_btn, voice_input], queue=False)
    voice_event.then(lock_ui, outputs=[msg_input, submit_btn, voice_input], queue=False)

    # Simulate
    sim_from_submit = submit_event.then(
        process_simulation,
        [chatbot, state],
        [chatbot, canvas_output],
        concurrency_limit=1,
    )

    sim_from_btn = btn_event.then(
        process_simulation,
        [chatbot, state],
        [chatbot, canvas_output],
        concurrency_limit=1,
    )

    sim_from_voice = voice_event.then(
        process_simulation,
        [chatbot, state],
        [chatbot, canvas_output],
        concurrency_limit=1,
    )

    # Unlock UI
    sim_from_submit.then(unlock_ui, outputs=[msg_input, submit_btn, voice_input], queue=False)
    sim_from_btn.then(unlock_ui, outputs=[msg_input, submit_btn, voice_input], queue=False)
    sim_from_voice.then(unlock_ui, outputs=[msg_input, submit_btn, voice_input], queue=False)

    # Comic
    comic_btn.click(generate_comic, [chatbot], [comic_output])


if __name__ == "__main__":
    favicon = "assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path=favicon
    )