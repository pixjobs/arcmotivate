import html
import os
import pathlib
import re
import threading
import time
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from lib.psychology_codex import map_narrative_to_superpowers
from lib.coaching_agent import generate_socratic_stream
from lib.outcome_engine import synthesize_single_tile
from lib.storybook_generator import (
    generate_custom_avatar,
    generate_custom_song,
    generate_hero_recap,
    generate_pixel_art_illustration,
)

# ============================================================
# APP CONFIG
# ============================================================

APP_TITLE = "ArcMotivate"
OPENING_MSG = (
    "👾 **System Online — ArcMotivate**\n\n"
    "Have you ever wondered what kind of future really fits you?\n\n"
    "ArcMotivate is a live exploration agent. I can respond to what you write, what you show me, "
    "and the patterns that emerge as we talk.\n\n"
    "Before we build your workspace, tell me a little about you. What energises you? What drains you? "
    "What’s something you’re proud of, or a moment that has stuck with you?\n\n"
    "You can type a message or attach an image, and we’ll explore it together."
)

STREAM_UPDATE_INTERVAL_SEC = 0.05
MAX_CHAT_CONTEXT_MESSAGES = 12
MAX_TILE_HISTORY_MESSAGES = 6
MAX_INLINE_IMAGES_PER_TURN = 2

POLL_INTERVAL_SEC = 1.1
MAX_PENDING_CANVAS_JOBS = 6

# ============================================================
# CSS
# ============================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@600;700&family=Press+Start+2P&display=swap');

:root{
  --bg-0:#05030b;--bg-1:#0a0314;--bg-2:#15092a;--panel:rgba(14,10,34,.86);--surface:rgba(20,10,40,.9);
  --text:#f8fafc;--text-soft:#dbe7f3;--text-dim:#9fb0c4;
  --cyan:#22d3ee;--pink:#ff4de3;--violet:#7c3aed;
  --border-cyan:rgba(34,211,238,.2);--border-violet:rgba(124,58,237,.24);
  --radius-md:14px;--radius-lg:18px;
  --shadow-1:0 6px 18px rgba(0,0,0,.24);--shadow-2:0 10px 24px rgba(0,0,0,.34);
  --glow-soft:0 0 0 1px rgba(34,211,238,.08),0 0 24px rgba(34,211,238,.08);
  --font-ui:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --font-display:'Press Start 2P',cursive;
  --font-accent:'Orbitron',sans-serif;
  --control-h:52px;--container-w:1400px;
}

*,*::before,*::after{box-sizing:border-box}
html,body{min-height:100%}
body,.gradio-container{
  margin:0!important;padding:0!important;color:var(--text)!important;font-family:var(--font-ui)!important;
  background:
    radial-gradient(circle at 12% 18%,rgba(255,77,227,.08),transparent 18%),
    radial-gradient(circle at 86% 22%,rgba(34,211,238,.08),transparent 18%),
    radial-gradient(ellipse at 20% 40%,var(--bg-2) 0%,var(--bg-1) 58%,#000 100%)!important
}
body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
.gradio-container{max-width:var(--container-w)!important;margin:0 auto!important;padding:16px 14px 20px!important;border:none!important}

/* Header */
.arc-header{position:relative;text-align:center;padding:10px 12px 14px}
.arc-logo{
  font-family:var(--font-display);font-size:clamp(1.35rem,2.2vw,2.2rem);line-height:1.35;letter-spacing:1px;
  color:var(--pink);text-shadow:0 0 6px rgba(255,77,227,.42),0 0 14px rgba(34,211,238,.14);padding-bottom:8px
}
.arc-tagline{
  font-size:clamp(1rem,1.4vw,1.3rem);letter-spacing:1.5px;text-transform:uppercase;color:var(--cyan);opacity:.92
}
.arc-status-wrap{display:flex;justify-content:center;margin-top:10px}
.arc-status{
  display:inline-flex;align-items:center;gap:10px;min-height:38px;padding:8px 14px;border-radius:999px;
  background:rgba(12,10,30,.6);border:1px solid rgba(34,211,238,.18);box-shadow:var(--glow-soft);
  color:var(--text-soft);font-size:.95rem
}
.arc-status-dot{
  width:10px;height:10px;flex:0 0 auto;border-radius:999px;background:var(--cyan);
  box-shadow:0 0 10px rgba(34,211,238,.7);animation:pulseHalo 1.8s ease-in-out infinite
}
.arc-status.is-busy .arc-status-dot{background:var(--pink);box-shadow:0 0 14px rgba(255,77,227,.9)}
.arc-status-label{
  font-family:var(--font-accent);font-size:.78rem;letter-spacing:.08em;color:var(--text-dim);text-transform:uppercase
}
.arc-status-text{font-family:var(--font-ui);font-size:.98rem;line-height:1.2;font-weight:500}
@keyframes pulseHalo{0%,100%{transform:scale(1);opacity:.8}50%{transform:scale(1.18);opacity:1}}

/* Shells */
.chat-wrap,.chat-input-shell,.canvas-tile,.canvas-empty,.identity-card{box-shadow:var(--shadow-1),var(--glow-soft)}
.chat-wrap{
  position:relative;overflow:hidden!important;border-radius:var(--radius-lg)!important;
  background:linear-gradient(180deg,rgba(17,10,38,.88),rgba(10,5,24,.9))!important;
  border:1px solid var(--border-cyan)!important;box-shadow:var(--shadow-2),var(--glow-soft)!important
}
.chat-wrap::after,.canvas-tile::after,.identity-card::after{
  content:"";position:absolute;inset:0;pointer-events:none
}
.chat-wrap::after{border-radius:inherit;box-shadow:inset 0 0 0 1px rgba(255,255,255,.02)}
.canvas-tile::after,.identity-card::after{background:linear-gradient(180deg,rgba(255,255,255,.03),transparent 28%)}

.gradio-chatbot,.gradio-chatbot>div,.gradio-chatbot .panel{
  background:transparent!important;border:none!important;box-shadow:none!important
}
.gradio-chatbot{
  padding:10px!important;scrollbar-width:thin;scrollbar-color:rgba(34,211,238,.35) transparent
}
.gradio-chatbot::-webkit-scrollbar{width:10px}
.gradio-chatbot::-webkit-scrollbar-thumb{background:rgba(34,211,238,.22);border-radius:999px}

/* Messages */
.gradio-chatbot .message{
  position:relative;width:fit-content!important;max-width:min(84%,820px)!important;
  margin-bottom:10px!important;padding:14px 16px!important;border-radius:16px!important;box-shadow:var(--shadow-1)!important
}
.gradio-chatbot .message.bot{
  margin-left:6px!important;margin-right:auto!important;background:rgba(34,211,238,.09)!important;
  border:1px solid rgba(34,211,238,.18)!important;border-bottom-left-radius:6px!important
}
.gradio-chatbot .message.user{
  margin-left:auto!important;margin-right:6px!important;background:rgba(124,58,237,.13)!important;
  border:1px solid rgba(124,58,237,.2)!important;border-bottom-right-radius:6px!important
}
.gradio-chatbot .avatar-container,.gradio-chatbot .message-role{display:none!important}

/* Typography */
.gradio-chatbot .prose,.gradio-chatbot .message *{font-family:var(--font-ui)!important}
.gradio-chatbot .prose{
  color:var(--text)!important;font-size:clamp(1rem,1.05vw,1.12rem)!important;line-height:1.65!important;
  font-weight:400!important;letter-spacing:.01em!important;padding:0!important;
  overflow-wrap:anywhere!important;word-break:break-word!important
}
.gradio-chatbot .prose p,
.gradio-chatbot .prose ul,
.gradio-chatbot .prose ol,
.gradio-chatbot .prose pre,
.gradio-chatbot .prose blockquote{margin-bottom:.6em!important}
.gradio-chatbot .prose p:last-child,
.gradio-chatbot .prose ul:last-child,
.gradio-chatbot .prose ol:last-child,
.gradio-chatbot .prose pre:last-child,
.gradio-chatbot .prose blockquote:last-child{margin-bottom:0!important}
.gradio-chatbot .prose strong{color:#fff!important;font-weight:700!important}
.gradio-chatbot .prose a{color:var(--cyan)!important;text-decoration:underline!important;text-underline-offset:2px}
.gradio-chatbot .prose code{
  background:rgba(255,255,255,.05)!important;border-radius:6px!important;padding:.04em .32em!important;
  color:#d8fbff!important;font-size:.9em!important
}
.gradio-chatbot .prose pre{
  background:rgba(3,7,18,.75)!important;border:1px solid rgba(34,211,238,.1)!important;border-radius:10px!important;
  padding:10px 12px!important;overflow-x:auto!important
}
.gradio-chatbot .prose blockquote{
  border-left:3px solid rgba(34,211,238,.28)!important;padding-left:10px!important;color:var(--text-soft)!important
}

/* Inline chat input */
.chat-input-shell{
  margin-top:12px;padding:10px;background:linear-gradient(180deg,rgba(14,8,34,.94),rgba(10,5,24,.93))!important;
  border:1px solid var(--border-cyan)!important;border-radius:var(--radius-md)!important
}
.chat-input-shell .gradio-multimodaltextbox,
.chat-input-shell .gradio-textbox,
.chat-input-shell .wrap,
.chat-input-shell .container{
  background:transparent!important;border:none!important;box-shadow:none!important
}
.chat-input-shell textarea{
  min-height:var(--control-h)!important;max-height:180px!important;padding:12px 14px!important;
  font-family:var(--font-ui)!important;font-size:1rem!important;line-height:1.45!important;font-weight:500!important;
  background:var(--surface)!important;color:var(--text)!important;border:1px solid rgba(34,211,238,.2)!important;
  border-radius:12px!important;resize:vertical!important
}
.chat-input-shell textarea::placeholder{color:rgba(34,211,238,.52)!important}
.chat-input-shell textarea:focus{
  outline:none!important;border-color:var(--cyan)!important;box-shadow:0 0 0 2px rgba(34,211,238,.12)!important
}
.chat-input-hint{
  margin-top:8px;color:var(--text-dim);font:.88rem/1.4 var(--font-ui);text-align:center
}

/* Tabs */
.tabs{background:transparent!important;border:none!important}
.tab-nav{
  display:flex;gap:6px;margin-bottom:10px!important;padding-bottom:8px;border-bottom:1px solid rgba(34,211,238,.12)!important;
  overflow-x:auto;scrollbar-width:none
}
.tab-nav::-webkit-scrollbar{display:none}
.tab-nav button{
  flex:0 0 auto;font-family:var(--font-ui)!important;font-size:.95rem!important;font-weight:600!important;
  color:var(--text-dim)!important;background:transparent!important;border:1px solid transparent!important;
  border-radius:10px!important;padding:6px 12px!important
}
.tab-nav button.selected{
  color:var(--cyan)!important;border-color:rgba(34,211,238,.16)!important;background:rgba(34,211,238,.06)!important;
  box-shadow:var(--glow-soft)!important
}

/* Workspace */
.canvas-shell{position:relative}
.canvas-status{margin-bottom:8px}
.canvas-processing{
  position:relative;overflow:hidden;width:100%;height:8px;border-radius:999px;background:rgba(255,255,255,.05);
  border:1px solid rgba(34,211,238,.08);box-shadow:inset 0 0 0 1px rgba(255,255,255,.02)
}
.canvas-processing::before{
  content:"";position:absolute;inset:0 auto 0 -35%;width:35%;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.85),rgba(255,77,227,.85),transparent);
  filter:blur(1px);animation:scanBar 1.6s linear infinite
}
@keyframes scanBar{0%{left:-35%}100%{left:100%}}

.canvas-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;padding:12px 0}
.canvas-tile{
  position:relative;overflow:hidden;background:rgba(15,10,46,.8);border:1px solid rgba(139,92,246,.18);border-radius:14px
}
.tile-img,.tile-img-placeholder{
  width:100%;height:118px;border-bottom:1px solid rgba(34,211,238,.12)
}
.tile-img{display:block;object-fit:cover}
.tile-img-placeholder{
  display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#1e1b4b,#0f0a2e);
  color:#7c3aed;font-size:1.45rem
}
.tile-body{padding:12px}
.tile-category{
  margin-bottom:5px;color:#c084fc;font:700 .72rem/1 var(--font-accent);letter-spacing:.1em;text-transform:uppercase
}
.tile-title{margin:0 0 6px;color:#f0f4ff;font:700 1rem/1.35 var(--font-ui)}
.tile-desc{color:var(--text-soft);font:400 .95rem/1.5 var(--font-ui)}
.tile-links{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.tile-link{
  display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(34,211,238,.08);color:var(--cyan);
  text-decoration:none;font-size:.9rem;border:1px solid rgba(34,211,238,.12)
}
.tile-link:hover{box-shadow:var(--glow-soft)}

/* Identity lab */
.identity-stack{display:flex;flex-direction:column;gap:12px}
.identity-card{
  position:relative;overflow:hidden;background:rgba(15,10,46,.8);border:1px solid rgba(139,92,246,.18);border-radius:14px;padding:14px
}
.identity-head{
  display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px
}
.identity-kicker{
  color:#c084fc;font:700 .72rem/1 var(--font-accent);letter-spacing:.1em;text-transform:uppercase
}
.identity-title{
  color:#f0f4ff;font:700 1rem/1.35 var(--font-ui);margin:4px 0 0
}
.identity-copy{
  color:var(--text-soft);font:400 .95rem/1.5 var(--font-ui)
}
.identity-avatar{
  width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:14px;border:1px solid rgba(34,211,238,.14);
  background:linear-gradient(135deg,#1e1b4b,#0f0a2e);box-shadow:var(--glow-soft)
}
.identity-avatar-placeholder{
  width:100%;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;
  border-radius:14px;border:1px dashed rgba(34,211,238,.14);background:linear-gradient(135deg,#1e1b4b,#0f0a2e);
  color:#7c3aed;font-size:2rem
}
.identity-meta{
  display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px
}
.identity-pill{
  padding:8px 10px;border-radius:12px;background:rgba(34,211,238,.06);border:1px solid rgba(34,211,238,.1);
  color:var(--text-soft);font:.88rem/1.3 var(--font-ui)
}
.identity-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.identity-download{
  display:inline-flex;align-items:center;justify-content:center;padding:10px 12px;border-radius:12px;
  background:linear-gradient(135deg,var(--violet),var(--cyan));color:#fff;text-decoration:none;
  font:700 .82rem/1 var(--font-accent);letter-spacing:.04em;box-shadow:var(--shadow-1)
}
.identity-download:hover{box-shadow:var(--shadow-2),var(--glow-soft)}
.identity-note{
  margin-top:10px;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.05);color:var(--text-soft);font:.92rem/1.5 var(--font-ui)
}

/* Empty/loading */
.canvas-empty{
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;margin-top:14px;padding:42px 18px;
  text-align:center;color:#64748b;border:1px dashed rgba(139,92,246,.18);border-radius:14px;background:rgba(255,255,255,.012)
}
.canvas-empty-icon{font-size:2.4rem;opacity:.35}
.canvas-empty-text{max-width:250px;font-size:1rem;line-height:1.35;color:#7b8ba3}
.synth-spinner{padding:16px;text-align:center;color:#a78bfa;font-size:1rem;font-style:italic}

/* Inline image grid */
.chat-comic-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:12px
}
.chat-comic-grid img{
  display:block;width:100%;height:100px;object-fit:cover;border:1px solid rgba(34,211,238,.16);border-radius:8px;
  box-shadow:var(--glow-soft)
}

/* Mobile */
@media (max-width:768px){
  :root{--control-h:48px}
  .gradio-container{padding:10px 8px 16px!important}
  .arc-header{padding:8px 8px 12px}
  .arc-logo{font-size:clamp(1.05rem,7vw,1.5rem);letter-spacing:.5px;padding-bottom:6px}
  .arc-tagline{font-size:.95rem;letter-spacing:1px}
  .arc-status{width:100%;justify-content:center;padding:8px 10px}
  .gradio-chatbot{padding:8px!important}
  .gradio-chatbot .message{
    max-width:96%!important;padding:12px 13px!important;margin-bottom:8px!important;border-radius:14px!important
  }
  .gradio-chatbot .prose{font-size:1.02rem!important;line-height:1.58!important}
  .chat-input-shell{padding:8px!important}
  .chat-input-shell textarea{
    min-height:var(--control-h)!important;max-height:140px!important;font-size:1rem!important;padding:10px 12px!important
  }
  .tab-nav{gap:4px;margin-bottom:8px!important;padding-bottom:6px}
  .tab-nav button{font-size:1rem!important;padding:6px 10px!important}
  .canvas-grid{grid-template-columns:1fr 1fr;gap:8px;padding:10px 0}
  .tile-img,.tile-img-placeholder{height:92px}
  .tile-body{padding:10px}
  .tile-title{font-size:.98rem}
  .tile-desc{font-size:.9rem}
  .tile-link{font-size:.82rem}
  .chat-comic-grid{grid-template-columns:repeat(2,1fr);gap:8px}
  .chat-comic-grid img{height:88px}
  .identity-meta{grid-template-columns:1fr}
}

@media (max-width:480px){
  .canvas-grid{grid-template-columns:1fr}
  .arc-logo{line-height:1.45}
  .gradio-chatbot .prose{font-size:.98rem!important}
}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}
}
"""

# ============================================================
# ASYNC SESSION STORES
# ============================================================

SESSION_STORES: Dict[str, Dict[str, Any]] = {}
SESSION_STORES_LOCK = threading.Lock()


def _default_session_store() -> Dict[str, Any]:
    return {
        "superpowers": {},
        "tiles": [],
        "turn_count": 0,
        "canvas_jobs": [],
        "canvas_running": False,
        "last_canvas_error": None,
        "last_canvas_completed_at": None,
        "avatar_b64": "",
        "song": {},
        "recap": "",
        "identity_jobs": [],
        "identity_running": False,
        "last_identity_error": None,
        "lock": threading.Lock(),
    }


def _session_id_from_request(request: Optional[gr.Request]) -> str:
    if request and getattr(request, "session_hash", None):
        return request.session_hash
    return "global"


def get_session_store(request: Optional[gr.Request]) -> Dict[str, Any]:
    session_id = _session_id_from_request(request)
    with SESSION_STORES_LOCK:
        if session_id not in SESSION_STORES:
            SESSION_STORES[session_id] = _default_session_store()
        return SESSION_STORES[session_id]


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
        visible = visible + "\n\n🧠 *Illustrating...*" if visible else "🧠 *Thinking...*"
        return visible, None, True

    json_block = accumulated_text[start_idx:end_idx + 1]
    display_text = (accumulated_text[:start_idx] + accumulated_text[end_idx + 1:]).strip()

    prompt_match = re.search(r'"prompt"\s*:\s*"(.*?)"\s*[,}]', json_block, flags=re.DOTALL)
    recovered_prompt = None
    if prompt_match:
        recovered_prompt = prompt_match.group(1).replace('\\"', '"').replace("\\n", " ").strip()

    return display_text, recovered_prompt, False


def sanitize_url(url: str) -> str:
    url = (url or "").strip()
    return url if url.startswith(("http://", "https://")) else ""


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
            <div class='canvas-empty-text'>Start exploring to unlock your unique workspace artifacts.</div>
        </div>
        """

    html_parts = ["<div class='canvas-grid'>"]

    for tile in reversed(tiles):
        image_b64 = tile.get("image_b64")
        if image_b64:
            img_html = (
                f"<img src='data:image/png;base64,{html.escape(image_b64, quote=True)}' "
                f"class='tile-img' alt='artifact image'>"
            )
        else:
            img_html = "<div class='tile-img-placeholder'>⚡</div>"

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
                    <div class='tile-category'>{safe_text(tile.get('category', 'Signal'))}</div>
                    <h3 class='tile-title'>{safe_text(tile.get('title', '—'))}</h3>
                    <p class='tile-desc'>{safe_text(tile.get('content', ''))}</p>
                    <div class='tile-links'>{"".join(links_html)}</div>
                </div>
            </div>
            """
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def render_canvas_with_status(store: Dict[str, Any]) -> str:
    with store["lock"]:
        tiles = deepcopy(store.get("tiles", []))
        pending = len(store.get("canvas_jobs", []))
        running = bool(store.get("canvas_running"))
        last_error = store.get("last_canvas_error")

    parts = ["<div class='canvas-shell'>"]
    if running or pending:
        parts.append("<div class='canvas-status'><div class='canvas-processing'></div></div>")
    parts.append(format_canvas(tiles))
    if running or pending:
        count = pending + (1 if running else 0)
        parts.append(f"<div class='synth-spinner'>🧩 Synthesizing {count} workspace update(s)...</div>")
    if last_error:
        parts.append("<div class='synth-spinner'>⚠️ A workspace update failed, but the chat is still live.</div>")
    parts.append("</div>")
    return "".join(parts)


def get_header_status_html(store: Dict[str, Any]) -> str:
    with store["lock"]:
        pending_workspace = len(store.get("canvas_jobs", []))
        running_workspace = bool(store.get("canvas_running"))
        running_identity = bool(store.get("identity_running"))
        turn_count = int(store.get("turn_count", 0))
        tiles_count = len(store.get("tiles", []))

    is_busy = running_workspace or pending_workspace or running_identity
    status_class = "arc-status is-busy" if is_busy else "arc-status"
    label = "Processing" if is_busy else "Ready"

    if is_busy:
        active = pending_workspace + (1 if running_workspace else 0) + (1 if running_identity else 0)
        text = f"Arc is building · {active} live agent task(s)"
    else:
        text = f"{tiles_count} workspace artifact(s) · {turn_count} turn(s) explored"

    return f"""
    <div class='arc-status-wrap'>
        <div class='{status_class}'>
            <span class='arc-status-dot'></span>
            <span class='arc-status-label'>{label}</span>
            <span class='arc-status-text'>{safe_text(text)}</span>
        </div>
    </div>
    """


def render_identity_lab(store: Dict[str, Any]) -> str:
    with store["lock"]:
        avatar_b64 = store.get("avatar_b64", "")
        song = deepcopy(store.get("song", {}))
        recap = store.get("recap", "")
        identity_running = bool(store.get("identity_running"))
        identity_jobs = len(store.get("identity_jobs", []))
        last_identity_error = store.get("last_identity_error")

    if avatar_b64:
        avatar_html = f"<img class='identity-avatar' src='data:image/png;base64,{html.escape(avatar_b64, quote=True)}' alt='Custom avatar'>"
    else:
        avatar_html = "<div class='identity-avatar-placeholder'>🧬</div>"

    title = safe_text(song.get("title", "No song yet"))
    subtitle = safe_text(song.get("subtitle", "Talk with Arc to generate your anthem"))
    bpm = safe_text(song.get("bpm", "—"))
    mood = safe_text(song.get("spec", {}).get("mood", "Still forming"))
    midi_path = song.get("midi_path", "")

    download_html = ""
    if midi_path and os.path.exists(midi_path):
        download_html = (
            f"<a class='identity-download' href='/gradio_api/file={html.escape(midi_path, quote=True)}' download>🎵 Download MIDI</a>"
        )

    busy_html = "<div class='synth-spinner'>🎛️ Forging your avatar and anthem...</div>" if (identity_running or identity_jobs) else ""
    error_html = "<div class='identity-note'>⚠️ Identity build hit a snag. Keep chatting and try again in a moment.</div>" if last_identity_error else ""
    recap_html = f"<div class='identity-note'>{safe_text(recap)}</div>" if recap else ""

    return f"""
    <div class='identity-stack'>
      <div class='identity-card'>
        <div class='identity-head'>
          <div>
            <div class='identity-kicker'>Identity Artifact</div>
            <div class='identity-title'>Custom Avatar</div>
          </div>
        </div>
        {avatar_html}
      </div>

      <div class='identity-card'>
        <div class='identity-head'>
          <div>
            <div class='identity-kicker'>Audio Artifact</div>
            <div class='identity-title'>This Is My Song</div>
          </div>
        </div>
        <div class='identity-copy'>{subtitle}</div>
        <div class='identity-meta'>
          <div class='identity-pill'><strong>Title</strong><br>{title}</div>
          <div class='identity-pill'><strong>BPM</strong><br>{bpm}</div>
          <div class='identity-pill'><strong>Mood</strong><br>{mood}</div>
          <div class='identity-pill'><strong>Status</strong><br>{"Ready" if midi_path else "Not built yet"}</div>
        </div>
        <div class='identity-actions'>{download_html}</div>
        {recap_html}
        {busy_html}
        {error_html}
      </div>
    </div>
    """


# ============================================================
# BACKGROUND WORKERS
# ============================================================

def canvas_worker(session_id: str) -> None:
    with SESSION_STORES_LOCK:
        store = SESSION_STORES.get(session_id)
    if not store:
        return

    while True:
        with store["lock"]:
            if not store["canvas_jobs"]:
                store["canvas_running"] = False
                return
            job = store["canvas_jobs"].pop(0)

        try:
            new_tile_data = synthesize_single_tile(job["history"], job["superpowers"])
        except Exception as exc:
            print(f"Tile synthesis error: {exc}")
            with store["lock"]:
                store["last_canvas_error"] = str(exc)
            continue

        if not new_tile_data:
            continue

        image_prompt = (new_tile_data.get("image_prompt") or "").strip()
        if image_prompt:
            try:
                new_tile_data["image_b64"] = cached_pixel_art(image_prompt)
            except Exception as exc:
                print(f"Tile art error: {exc}")
                new_tile_data["image_b64"] = None

        with store["lock"]:
            store["tiles"].append(new_tile_data)
            store["turn_count"] += 1
            store["last_canvas_error"] = None
            store["last_canvas_completed_at"] = time.time()


def identity_worker(session_id: str) -> None:
    with SESSION_STORES_LOCK:
        store = SESSION_STORES.get(session_id)
    if not store:
        return

    while True:
        with store["lock"]:
            if not store["identity_jobs"]:
                store["identity_running"] = False
                return
            job = store["identity_jobs"].pop(0)

        try:
            profile = deepcopy(job["superpowers"])
            history = deepcopy(job["history"])
            latest_signal = job.get("latest_signal", "")

            avatar_b64 = generate_custom_avatar(profile, latest_signal=latest_signal)
            song = generate_custom_song(profile, recent_chat=history)
            recap = generate_hero_recap(profile)
        except Exception as exc:
            print(f"Identity build error: {exc}")
            with store["lock"]:
                store["last_identity_error"] = str(exc)
            continue

        with store["lock"]:
            if avatar_b64:
                store["avatar_b64"] = avatar_b64
            if song:
                store["song"] = song
            if recap:
                store["recap"] = recap
            store["last_identity_error"] = None


def start_canvas_worker_if_needed(request: Optional[gr.Request]) -> None:
    session_id = _session_id_from_request(request)
    with SESSION_STORES_LOCK:
        store = SESSION_STORES.get(session_id)
    if not store:
        return

    with store["lock"]:
        if store["canvas_running"] or not store["canvas_jobs"]:
            return
        store["canvas_running"] = True

    threading.Thread(target=canvas_worker, args=(session_id,), daemon=True).start()


def start_identity_worker_if_needed(request: Optional[gr.Request]) -> None:
    session_id = _session_id_from_request(request)
    with SESSION_STORES_LOCK:
        store = SESSION_STORES.get(session_id)
    if not store:
        return

    with store["lock"]:
        if store["identity_running"] or not store["identity_jobs"]:
            return
        store["identity_running"] = True

    threading.Thread(target=identity_worker, args=(session_id,), daemon=True).start()


def enqueue_canvas_job(history: List[Dict[str, Any]], state_data: Dict[str, Any], request: gr.Request):
    store = get_session_store(request)
    snapshot = build_backend_history(history, max_messages=MAX_TILE_HISTORY_MESSAGES)

    with store["lock"]:
        if len(store["canvas_jobs"]) >= MAX_PENDING_CANVAS_JOBS:
            store["canvas_jobs"] = store["canvas_jobs"][-(MAX_PENDING_CANVAS_JOBS - 1):]
        store["canvas_jobs"].append(
            {
                "history": snapshot,
                "superpowers": deepcopy(store.get("superpowers", {})),
                "queued_at": time.time(),
            }
        )

    start_canvas_worker_if_needed(request)
    return render_canvas_with_status(store), get_header_status_html(store)


def enqueue_identity_job(history: List[Dict[str, Any]], state_data: Dict[str, Any], request: gr.Request):
    store = get_session_store(request)
    snapshot = build_backend_history(history, max_messages=MAX_CHAT_CONTEXT_MESSAGES)
    latest_signal = clean_message_for_backend(history[-1].get("content", "")) if history else ""

    with store["lock"]:
        store["identity_jobs"] = [
            {
                "history": snapshot,
                "superpowers": deepcopy(store.get("superpowers", {})),
                "latest_signal": latest_signal,
                "queued_at": time.time(),
            }
        ]

    start_identity_worker_if_needed(request)
    return render_identity_lab(store), get_header_status_html(store)


def refresh_async_panels(request: gr.Request):
    store = get_session_store(request)
    return (
        render_canvas_with_status(store),
        get_header_status_html(store),
        render_identity_lab(store),
    )


# ============================================================
# USER INPUT
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


# ============================================================
# MAIN CHAT PIPELINE
# ============================================================

def process_simulation(
    history: List[Dict[str, Any]],
    state_data: Dict[str, Any],
    request: gr.Request,
) -> Generator[Tuple[List[Dict[str, Any]], Any], None, None]:
    if not history:
        yield history, gr.update()
        return

    store = get_session_store(request)
    latest_user_input = clean_message_for_backend(history[-1].get("content", ""))

    with store["lock"]:
        has_superpowers = bool(store["superpowers"])

    if not has_superpowers:
        pending = state_data.get("pending_image") or {}
        img_bytes = pending.get("bytes")
        img_mime = pending.get("mime", "image/jpeg")
        try:
            mapped = map_narrative_to_superpowers(latest_user_input, img_bytes, img_mime)
        except Exception as exc:
            print(f"Superpower mapping error: {exc}")
            mapped = {}

        with store["lock"]:
            store["superpowers"] = mapped

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
        with store["lock"]:
            superpowers = deepcopy(store["superpowers"])
        stream = generate_socratic_stream(superpowers, backend_history, img_bytes, img_mime)
    except Exception as exc:
        print(f"Stream setup error: {exc}")
        history[-1]["content"] = "⚠️ Something glitched while starting the simulation."
        yield history, render_canvas_with_status(store)
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
        yield history, gr.update()
        img_b64 = maybe_generate_inline_visual(recovered_prompt)
        if img_b64:
            inline_images.append(img_b64)

    if inline_images:
        history[-1]["content"] = (final_display_text + "\n\n" + format_inline_images(inline_images)).strip()

    yield history, render_canvas_with_status(store)


# ============================================================
# UI LOCK
# ============================================================

def lock_ui():
    return gr.update(interactive=False)


def unlock_ui():
    return gr.update(interactive=True)


# ============================================================
# APP
# ============================================================

with gr.Blocks(css=css, theme=gr.themes.Base(), title=APP_TITLE) as demo:
    state = gr.State({"pending_image": None})

    gr.HTML(
        """
        <div class='arc-header'>
            <div class='arc-logo'>🕹️ ArcMotivate</div>
            <div class='arc-tagline'>Interactive Career Explorer</div>
            <div id='arc-status-mount'></div>
        </div>
        """
    )

    header_status = gr.HTML(
        """
        <div class='arc-status-wrap'>
            <div class='arc-status'>
                <span class='arc-status-dot'></span>
                <span class='arc-status-label'>Ready</span>
                <span class='arc-status-text'>0 workspace artifact(s) · 0 turn(s) explored</span>
            </div>
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

            with gr.Column(elem_classes=["chat-input-shell"]):
                msg_input = gr.MultimodalTextbox(
                    placeholder="Type your message, attach an image, then press Enter…",
                    show_label=False,
                    container=False,
                    file_types=["image"],
                    file_count="multiple",
                    autofocus=True,
                )
                gr.HTML("<div class='chat-input-hint'>Press <strong>Enter</strong> to send · <strong>Shift+Enter</strong> for a new line · add an image inline</div>")

            gr.Examples(
                examples=[
                    [{"text": "I love making things that help other people, but I get bored when tasks feel repetitive.", "files": []}],
                    [{"text": "I’ve attached something I made — what does it suggest about me?", "files": []}],
                    [{"text": "At school I liked solving messy problems more than following instructions.", "files": []}],
                ],
                inputs=msg_input,
                label="Quick starts",
            )

        with gr.Column(scale=4, min_width=340):
            gr.HTML("<div style='height:6px'></div>")

            with gr.Tabs():
                with gr.TabItem("⚡ Agent Workspace"):
                    canvas_output = gr.HTML(render_canvas_with_status(_default_session_store()))

                with gr.TabItem("🧬 Identity Lab"):
                    identity_output = gr.HTML(render_identity_lab(_default_session_store()))

    poll_timer = gr.Timer(POLL_INTERVAL_SEC)

    submit_event = msg_input.submit(
        user_submit,
        [msg_input, chatbot, state],
        [msg_input, chatbot, state],
        queue=False,
    )

    submit_event.then(lock_ui, outputs=[msg_input], queue=False)

    sim_event = submit_event.then(
        process_simulation,
        [chatbot, state],
        [chatbot, canvas_output],
        concurrency_limit=1,
    )

    queued_canvas = sim_event.then(
        enqueue_canvas_job,
        [chatbot, state],
        [canvas_output, header_status],
        queue=False,
    )

    queued_identity = queued_canvas.then(
        enqueue_identity_job,
        [chatbot, state],
        [identity_output, header_status],
        queue=False,
    )

    queued_identity.then(unlock_ui, outputs=[msg_input], queue=False)

    poll_timer.tick(
        refresh_async_panels,
        outputs=[canvas_output, header_status, identity_output],
        queue=False,
    )

if __name__ == "__main__":
    favicon = "assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path=favicon,
    )