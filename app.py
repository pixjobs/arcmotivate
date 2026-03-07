import html
import logging
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

from lib.psychology_codex import map_narrative_to_superpowers
from lib.coaching_agent import generate_socratic_stream
from lib.outcome_engine import generate_intro_message, synthesize_single_tile
from lib.storybook_generator import (
    generate_custom_avatar,
    generate_future_postcard,
    generate_hero_recap,
    generate_identity_comic,
    generate_pixel_art_illustration,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "ArcMotivate"

FALLBACK_OPENING_MSG = (
    "👾 **System Online — ArcMotivate**\n\n"
    "Have you ever wondered what kind of future really fits you?\n\n"
    "ArcMotivate is a live exploration agent. I can respond to what you write, what you show me, "
    "and the patterns that emerge as we talk.\n\n"
    "[VISUALIZE: A neon pixel-art control room waking up, with glowing screens, an open sketchbook, "
    "tools, music notes, code fragments, and pathways branching into different futures]\n\n"
    "Before we build your workspace, tell me a little about you. What energises you? What drains you? "
    "What’s something you’re proud of, or a moment that has stuck with you?\n\n"
    "You can type a message or attach an image, and we’ll explore it together."
)

try:
    OPENING_MSG = generate_intro_message() or FALLBACK_OPENING_MSG
except Exception:
    logger.exception("Failed to generate opening message")
    OPENING_MSG = FALLBACK_OPENING_MSG

STREAM_UPDATE_INTERVAL_SEC = 0.05
MAX_CHAT_CONTEXT_MESSAGES = 12
MAX_TILE_HISTORY_MESSAGES = 6
MAX_INLINE_IMAGES_PER_TURN = 2
MAX_INLINE_VISUAL_MARKERS = 1
SESSION_TTL_SEC = 60 * 60
MAX_PENDING_ARTIFACT_JOBS = 5

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@600;700&family=Press+Start+2P&display=swap');

:root{
  --bg-0:#05030b;--bg-1:#0a0314;--bg-2:#15092a;--surface:rgba(20,10,40,.9);
  --text:#f8fafc;--text-soft:#dbe7f3;--text-dim:#9fb0c4;
  --cyan:#22d3ee;--pink:#ff4de3;--violet:#7c3aed;
  --border-cyan:rgba(34,211,238,.2);
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
.gradio-container{max-width:var(--container-w)!important;margin:0 auto!important;padding:16px 14px 20px!important;border:none!important}

.arc-header{text-align:center;padding:10px 12px 14px}
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
  box-shadow:0 0 10px rgba(34,211,238,.7)
}
.arc-status.is-busy .arc-status-dot{background:var(--pink)}

.chat-wrap,.chat-input-shell,.canvas-empty,.story-card{box-shadow:var(--shadow-1),var(--glow-soft)}
.chat-wrap{
  overflow:hidden!important;border-radius:var(--radius-lg)!important;
  background:linear-gradient(180deg,rgba(17,10,38,.88),rgba(10,5,24,.9))!important;
  border:1px solid var(--border-cyan)!important;box-shadow:var(--shadow-2),var(--glow-soft)!important
}
.gradio-chatbot,.gradio-chatbot>div,.gradio-chatbot .panel{
  background:transparent!important;border:none!important;box-shadow:none!important
}
.gradio-chatbot{padding:10px!important}
.gradio-chatbot .message{
  width:fit-content!important;max-width:min(84%,820px)!important;
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
.gradio-chatbot .prose,.gradio-chatbot .message *{font-family:var(--font-ui)!important}
.gradio-chatbot .prose{
  color:var(--text)!important;font-size:clamp(1rem,1.05vw,1.12rem)!important;line-height:1.65!important;
  overflow-wrap:anywhere!important;word-break:break-word!important
}

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
.chat-input-hint{
  margin-top:8px;color:var(--text-dim);font:.88rem/1.4 var(--font-ui);text-align:center
}

.canvas-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;padding:12px 0}
.canvas-tile{
  overflow:hidden;
  background:rgba(15,10,46,.8);
  border:1px solid rgba(139,92,246,.18);
  border-radius:14px;
  position:relative;
  box-shadow:
    0 10px 24px rgba(0,0,0,.34),
    0 0 0 1px rgba(34,211,238,.10),
    0 0 24px rgba(34,211,238,.10),
    0 0 44px rgba(124,58,237,.08);
}
.canvas-tile::after{
  content:"";
  position:absolute;
  inset:0;
  pointer-events:none;
  background:linear-gradient(180deg,rgba(255,255,255,.03),transparent 28%);
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
.tile-skill-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.tile-skill-tag{
  display:inline-block;padding:3px 8px;border-radius:6px;
  background:rgba(139,92,246,.1);color:var(--cyan);
  font:600 .72rem/1.1 var(--font-accent);letter-spacing:.02em;
  border:1px solid rgba(34,211,238,.08)
}
.tile-links{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.tile-link{
  display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(34,211,238,.08);color:var(--cyan);
  text-decoration:none;font-size:.9rem;border:1px solid rgba(34,211,238,.12)
}

.canvas-empty{
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;margin-top:14px;padding:42px 18px;
  text-align:center;color:#64748b;border:1px dashed rgba(139,92,246,.18);border-radius:14px;background:rgba(255,255,255,.012)
}
.canvas-empty-icon{font-size:2.4rem;opacity:.35}
.canvas-empty-text{max-width:280px;font-size:1rem;line-height:1.35;color:#7b8ba3}
.synth-spinner{padding:16px;text-align:center;color:#a78bfa;font-size:1rem;font-style:italic}

.chat-skill-card{
  display:flex;align-items:flex-start;gap:12px;margin:12px 0;
  padding:14px 16px;border-radius:14px;
  background:linear-gradient(135deg,rgba(139,92,246,.08),rgba(34,211,238,.06));
  border:1px solid rgba(139,92,246,.18);position:relative;overflow:hidden
}
.chat-skill-card::before{
  content:'';position:absolute;top:0;left:0;width:4px;height:100%;
  background:linear-gradient(180deg,var(--violet),var(--cyan))
}
.chat-skill-icon{font-size:1.5rem;flex-shrink:0;margin-top:2px}
.chat-skill-body{flex:1;min-width:0}
.chat-skill-name{font:700 .92rem/1.2 var(--font-accent);color:#fff;margin-bottom:4px}
.chat-skill-try{font:.88rem/1.35 var(--font-ui);color:var(--text-soft);margin-bottom:8px}
.chat-skill-link{
  display:inline-flex;align-items:center;gap:4px;padding:6px 12px;
  border-radius:8px;background:rgba(139,92,246,.12);color:var(--cyan);
  font:700 .78rem/1 var(--font-accent);letter-spacing:.03em;text-decoration:none;
  border:1px solid rgba(34,211,238,.12)
}

.chat-inline-visual{
  margin:14px 0;border-radius:12px;overflow:hidden;
  border:1px solid rgba(34,211,238,.16);box-shadow:var(--glow-soft)
}
.chat-inline-visual img{display:block;width:100%;max-height:220px;object-fit:cover}

.chat-comic-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:12px
}
.chat-comic-grid img{
  display:block;width:100%;height:100px;object-fit:cover;border:1px solid rgba(34,211,238,.16);border-radius:8px
}

.story-card{
  padding:24px;border-radius:18px;
  background:linear-gradient(145deg,rgba(15,10,30,.95),rgba(20,15,40,.9));
  border:1px solid rgba(139,92,246,.14)
}
.story-card-title{
  font:800 1.3rem/1.2 var(--font-accent);
  color: var(--cyan);
  margin-bottom:18px;
}
@supports (-webkit-background-clip:text) {
  .story-card-title{
    background: linear-gradient(90deg, var(--violet), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
}

.story-section{margin-bottom:20px}
.story-section-label{font:700 .78rem/1 var(--font-accent);color:var(--cyan);letter-spacing:.04em;text-transform:uppercase;margin-bottom:8px}
.story-narrative{font:.95rem/1.6 var(--font-ui);color:var(--text-soft)}
.story-avatar{
  width:120px;height:120px;border-radius:50%;object-fit:cover;
  border:2px solid rgba(139,92,246,.3);margin:0 auto;display:block
}
.story-comic-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
.story-comic-panel{
  border-radius:12px;overflow:hidden;background:rgba(255,255,255,.02);
  border:1px solid rgba(34,211,238,.1)
}
.story-comic-panel img{display:block;width:100%;height:100px;object-fit:cover}
.story-comic-caption{padding:8px 10px;font:.82rem/1.3 var(--font-ui);color:var(--text-soft);text-align:center}
.story-postcard{
  padding:18px;border-radius:14px;
  background:linear-gradient(135deg,rgba(139,92,246,.08),rgba(34,211,238,.05));
  border:1px solid rgba(139,92,246,.12);text-align:center
}
.story-postcard img{display:block;width:100%;max-height:160px;object-fit:cover;border-radius:10px;margin-bottom:10px}
.story-postcard-caption{font:600 .95rem/1.4 var(--font-ui);color:#fff}
.story-nudge{
  padding:12px 16px;border-radius:12px;
  background:linear-gradient(135deg,rgba(34,211,238,.06),rgba(139,92,246,.04));
  border:1px solid rgba(34,211,238,.1);
  font:.9rem/1.5 var(--font-ui);color:var(--text-soft)
}
.story-nudge strong{color:var(--cyan)}
.identity-note{
  margin-top:10px;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.05);color:var(--text-soft);font:.92rem/1.5 var(--font-ui)
}

@media (max-width:768px){
  .gradio-container{padding:10px 8px 16px!important}
  .canvas-grid,.story-comic-strip{grid-template-columns:1fr}
}
"""

_RE_VISUALIZE = re.compile(r"\[VISUALIZE:\s*(.+?)\]", re.DOTALL)
_RE_SKILL = re.compile(r"\[SKILL:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\]", re.DOTALL)
_RE_HTML_IMG = re.compile(r"<img[^>]*>", re.IGNORECASE)
_RE_HTML_DIV = re.compile(r"<div[^>]*>.*?</div>", re.DOTALL | re.IGNORECASE)

SESSION_STORES: Dict[str, Dict[str, Any]] = {}
SESSION_STORES_LOCK = threading.Lock()


def default_session_store() -> Dict[str, Any]:
    return {
        "pending_image": None,
        "superpowers": {},
        "tiles": [],
        "chat_turn_count": 0,
        "tile_count": 0,
        "avatar_b64": "",
        "recap": "",
        "comic_panels": [],
        "postcard": {},
        "last_canvas_error": None,
        "last_identity_error": None,
        "artifact_jobs": [],
        "artifact_running": False,
        "artifact_stage": "",
        "artifact_updated_at": 0.0,
        "last_seen_at": time.time(),
        "lock": threading.Lock(),
    }


def _session_id_from_request(request: Optional[gr.Request]) -> str:
    if request and getattr(request, "session_hash", None):
        return str(request.session_hash)
    return "global"


def _cleanup_stale_sessions() -> None:
    now = time.time()
    stale_ids: List[str] = []
    with SESSION_STORES_LOCK:
        for session_id, store in SESSION_STORES.items():
            if session_id == "global":
                continue
            if (now - float(store.get("last_seen_at", now))) > SESSION_TTL_SEC:
                stale_ids.append(session_id)
        for session_id in stale_ids:
            SESSION_STORES.pop(session_id, None)


def get_session_store(request: Optional[gr.Request]) -> Dict[str, Any]:
    _cleanup_stale_sessions()
    session_id = _session_id_from_request(request)
    with SESSION_STORES_LOCK:
        if session_id not in SESSION_STORES:
            SESSION_STORES[session_id] = default_session_store()
        SESSION_STORES[session_id]["last_seen_at"] = time.time()
        return SESSION_STORES[session_id]


def safe_text(value: Any) -> str:
    return html.escape(str(value or ""))


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
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
    text = _RE_HTML_IMG.sub("", text)
    text = _RE_HTML_DIV.sub("", text)
    text = _RE_VISUALIZE.sub("", text)
    text = _RE_SKILL.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_backend_history(
    history: List[Dict[str, Any]],
    max_messages: int,
) -> List[Dict[str, str]]:
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
    except Exception:
        logger.exception("Inline image generation failed")
        return None


def format_inline_images(image_b64_list: List[str]) -> str:
    if not image_b64_list:
        return ""
    tags = []
    for b64 in image_b64_list[:MAX_INLINE_IMAGES_PER_TURN]:
        safe_b64 = html.escape(b64, quote=True)
        tags.append(f"<img src='data:image/png;base64,{safe_b64}' alt='arc visual'>")
    return f"<div class='chat-comic-grid'>{''.join(tags)}</div>"


def format_skill_card(name: str, url: str, try_this: str) -> str:
    s_name = safe_text(name.strip())
    s_url = html.escape(url.strip(), quote=True)
    s_try = safe_text(try_this.strip())
    return (
        f"<div class='chat-skill-card'>"
        f"<div class='chat-skill-icon'>🎯</div>"
        f"<div class='chat-skill-body'>"
        f"<div class='chat-skill-name'>{s_name}</div>"
        f"<div class='chat-skill-try'>{s_try}</div>"
        f"<a class='chat-skill-link' href='{s_url}' target='_blank' rel='noopener noreferrer'>🔗 Explore</a>"
        f"</div>"
        f"</div>"
    )

def format_inline_visual_html(image_b64: str, align: str = "left") -> str:
    safe_b64 = html.escape(image_b64, quote=True)
    align_class = "chat-inline-visual-left" if align == "left" else "chat-inline-visual-right"
    return (
        f"<div class='chat-inline-visual {align_class}'>"
        f"<img src='data:image/png;base64,{safe_b64}' alt='visual metaphor'>"
        f"</div>"
    )

def render_interleaved_content(raw_text: str, enable_visuals: bool = True) -> str:
    parts: List[str] = []
    remainder = raw_text
    visuals_used = 0

    while remainder:
        vis_m = _RE_VISUALIZE.search(remainder)
        skill_m = _RE_SKILL.search(remainder)

        earliest = None
        earliest_pos = len(remainder)

        for m in (vis_m, skill_m):
            if m and m.start() < earliest_pos:
                earliest = m
                earliest_pos = m.start()

        if earliest is None:
            text_chunk = remainder.strip()
            if text_chunk:
                parts.append(text_chunk)
            break

        before = remainder[:earliest_pos].strip()
        if before:
            parts.append(before)

        if earliest is vis_m:
            prompt = vis_m.group(1).strip()
            if enable_visuals and visuals_used < MAX_INLINE_VISUAL_MARKERS:
                img_b64 = maybe_generate_inline_visual(prompt)
                if img_b64:
                    align = "left" if visuals_used % 2 == 0 else "right"
                    parts.append(format_inline_visual_html(img_b64, align=align))
                    visuals_used += 1
            remainder = remainder[vis_m.end():]

        elif earliest is skill_m:
            # Scrap skill cards from chat if they are slowing things down.
            # Just drop the marker and let the right-side workspace handle exploration.
            remainder = remainder[skill_m.end():]

        else:
            break

    return "\n\n".join(parts)

def get_header_status_html(store: Dict[str, Any], busy: bool = False, stage: str = "") -> str:
    with store["lock"]:
        turn_count = int(store.get("chat_turn_count", 0))
        tiles_count = len(store.get("tiles", []))
        artifact_running = bool(store.get("artifact_running"))
        artifact_stage = str(store.get("artifact_stage", "")).strip()
        pending_jobs = len(store.get("artifact_jobs", []))

    label = "Processing" if busy or artifact_running else "Ready"
    status_class = "arc-status is-busy" if busy or artifact_running else "arc-status"

    if busy and stage:
        text = stage
    elif artifact_running:
        text = f"Building story artifacts · {artifact_stage or 'working'} · {pending_jobs} queued"
    else:
        text = f"{tiles_count} workspace artifact(s) · {turn_count} turn(s) explored"

    return f"""
    <div class='arc-status-wrap'>
        <div class='{status_class}'>
            <span class='arc-status-dot'></span>
            <span class='arc-status-label'>{safe_text(label)}</span>
            <span class='arc-status-text'>{safe_text(text)}</span>
        </div>
    </div>
    """


def format_canvas(store: Dict[str, Any]) -> str:
    with store["lock"]:
        tiles = deepcopy(store.get("tiles", []))
        error_text = str(store.get("last_canvas_error") or "")
        artifact_running = bool(store.get("artifact_running"))
        artifact_stage = str(store.get("artifact_stage", "")).strip()

    busy_text = ""
    if artifact_running and artifact_stage == "tile":
        busy_text = "🧩 Building workspace artifact..."

    if not tiles and not busy_text and not error_text:
        return """
        <div class='canvas-empty'>
            <div class='canvas-empty-icon'>🎮</div>
            <div class='canvas-empty-text'>Start exploring to unlock your unique workspace artifacts.</div>
        </div>
        """

    html_parts = []
    if busy_text:
        html_parts.append(f"<div class='synth-spinner'>{safe_text(busy_text)}</div>")
    if error_text:
        html_parts.append(f"<div class='identity-note'>⚠️ {safe_text(error_text)}</div>")

    if tiles:
        html_parts.append("<div class='canvas-grid'>")
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
                    url = str(link.get("url", "")).strip()
                    if url.startswith(("http://", "https://")):
                        links_html.append(
                            f"<a href='{html.escape(url, quote=True)}' target='_blank' "
                            f"rel='noopener noreferrer' class='tile-link'>🔗 {label}</a>"
                        )

            skill_tags = tile.get("skill_tags") or []
            tags_html = ""
            if skill_tags:
                tags_html = "<div class='tile-skill-tags'>" + "".join(
                    f"<span class='tile-skill-tag'>{safe_text(tag)}</span>" for tag in skill_tags[:3]
                ) + "</div>"

            skill_nudge = tile.get("skill_nudge") or ""
            skill_nudge_html = (
                f"<p class='tile-desc' style='margin-top:6px;font-style:italic'>{safe_text(skill_nudge)}</p>"
                if skill_nudge else ""
            )

            html_parts.append(
                f"""
                <div class='canvas-tile'>
                    {img_html}
                    <div class='tile-body'>
                        <div class='tile-category'>{safe_text(tile.get('category', 'Signal'))}</div>
                        <h3 class='tile-title'>{safe_text(tile.get('title', '—'))}</h3>
                        <p class='tile-desc'>{safe_text(tile.get('content', ''))}</p>
                        {tags_html}
                        {skill_nudge_html}
                        <div class='tile-links'>{"".join(links_html)}</div>
                    </div>
                </div>
                """
            )
        html_parts.append("</div>")

    return "".join(html_parts)


def render_identity_lab(store: Dict[str, Any]) -> str:
    with store["lock"]:
        avatar_b64 = store.get("avatar_b64", "")
        recap = store.get("recap", "")
        superpowers = deepcopy(store.get("superpowers", {}))
        comic_panels = deepcopy(store.get("comic_panels", []))
        postcard = deepcopy(store.get("postcard", {}))
        error_text = str(store.get("last_identity_error") or "")
        artifact_running = bool(store.get("artifact_running"))
        artifact_stage = str(store.get("artifact_stage", "")).strip()

    busy_text = ""
    if artifact_running and artifact_stage in {"recap", "avatar", "comic", "postcard"}:
        labels = {
            "recap": "📖 Writing your story...",
            "avatar": "🎭 Generating avatar...",
            "comic": "🎬 Building comic journey...",
            "postcard": "💌 Writing postcard from future you...",
        }
        busy_text = labels.get(artifact_stage, "🎛️ Updating your exploration story...")

    has_content = bool(avatar_b64 or recap or comic_panels or postcard)

    if not has_content and not busy_text and not error_text:
        return """
        <div class='canvas-empty'>
            <div class='canvas-empty-icon'>🧬</div>
            <div class='canvas-empty-text'>Start chatting to build your exploration story.</div>
        </div>
        """

    sections: List[str] = []

    if busy_text:
        sections.append(f"<div class='synth-spinner'>{safe_text(busy_text)}</div>")
    if error_text:
        sections.append(f"<div class='identity-note'>⚠️ {safe_text(error_text)}</div>")

    if recap:
        sections.append(f"""
        <div class='story-section'>
            <div class='story-section-label'>📖 Your Story</div>
            <div class='story-narrative'>{safe_text(recap)}</div>
        </div>
        """)

    if avatar_b64:
        sections.append(f"""
        <div class='story-section' style='text-align:center'>
            <div class='story-section-label'>🎭 Your Avatar</div>
            <img class='story-avatar' src='data:image/png;base64,{html.escape(avatar_b64, quote=True)}' alt='Custom avatar'>
        </div>
        """)

    if comic_panels:
        panel_html_parts: List[str] = []
        for panel in comic_panels[:3]:
            img_b64 = panel.get("image_b64", "")
            caption = safe_text(panel.get("caption", ""))
            if img_b64:
                img_tag = f"<img src='data:image/png;base64,{html.escape(img_b64, quote=True)}' alt='comic panel'>"
            else:
                img_tag = "<div style='height:100px;display:flex;align-items:center;justify-content:center;color:#64748b'>🎨</div>"
            panel_html_parts.append(f"""
            <div class='story-comic-panel'>
                {img_tag}
                <div class='story-comic-caption'>{caption}</div>
            </div>
            """)
        sections.append(f"""
        <div class='story-section'>
            <div class='story-section-label'>🎬 Your Journey</div>
            <div class='story-comic-strip'>{''.join(panel_html_parts)}</div>
        </div>
        """)

    if postcard and postcard.get("caption"):
        postcard_img = postcard.get("image_b64", "")
        postcard_caption = safe_text(postcard.get("caption", ""))
        img_html = (
            f"<img src='data:image/png;base64,{html.escape(postcard_img, quote=True)}' alt='future postcard'>"
            if postcard_img else ""
        )
        sections.append(f"""
        <div class='story-section'>
            <div class='story-section-label'>💌 Postcard from Future You</div>
            <div class='story-postcard'>
                {img_html}
                <div class='story-postcard-caption'>{postcard_caption}</div>
            </div>
        </div>
        """)

    growth_nudge = superpowers.get("growth_nudge", "")
    if growth_nudge:
        sections.append(f"""
        <div class='story-section'>
            <div class='story-section-label'>🌱 Next Level</div>
            <div class='story-nudge'><strong>Try this:</strong> {safe_text(growth_nudge)}</div>
        </div>
        """)

    return f"<div class='story-card'><div class='story-card-title'>Your Exploration Story</div>{''.join(sections)}</div>"


def plan_artifacts(store: Dict[str, Any]) -> List[str]:
    with store["lock"]:
        turn = int(store.get("chat_turn_count", 0))
        avatar_missing = not bool(store.get("avatar_b64"))
        comic_missing = not bool(store.get("comic_panels"))
        postcard_missing = not bool(store.get("postcard"))

    if turn == 1:
        jobs = ["tile", "recap"]
        if avatar_missing:
            jobs.append("avatar")
        return jobs

    if turn == 2:
        jobs = ["tile", "recap"]
        if avatar_missing:
            jobs.append("avatar")
        jobs.append("comic")
        return jobs

    jobs = ["tile"]
    if turn <= 4 or turn % 2 == 0:
        jobs.append("recap")
    if avatar_missing:
        jobs.append("avatar")
    if comic_missing or turn % 2 == 1:
        jobs.append("comic")
    if postcard_missing or turn % 3 == 0:
        jobs.append("postcard")
    return jobs


def enqueue_artifact_jobs(
    store: Dict[str, Any],
    jobs: List[str],
    history: List[Dict[str, Any]],
    latest_signal: str,
) -> None:
    if not jobs:
        return

    history_chat = build_backend_history(history, MAX_CHAT_CONTEXT_MESSAGES)
    history_tile = build_backend_history(history, MAX_TILE_HISTORY_MESSAGES)

    with store["lock"]:
        superpowers = deepcopy(store.get("superpowers", {}))
        queued_kinds = {job.get("kind") for job in store["artifact_jobs"]}

        for kind in jobs:
            if kind in queued_kinds:
                continue
            if len(store["artifact_jobs"]) >= MAX_PENDING_ARTIFACT_JOBS:
                break

            store["artifact_jobs"].append(
                {
                    "kind": kind,
                    "superpowers": deepcopy(superpowers),
                    "history_chat": deepcopy(history_chat),
                    "history_tile": deepcopy(history_tile),
                    "latest_signal": latest_signal,
                    "queued_at": time.time(),
                }
            )
            queued_kinds.add(kind)

        should_start = not store["artifact_running"] and bool(store["artifact_jobs"])
        if should_start:
            store["artifact_running"] = True
            store["artifact_stage"] = "queued"

    if should_start:
        threading.Thread(target=artifact_worker, args=(store,), daemon=True).start()


def artifact_worker(store: Dict[str, Any]) -> None:
    while True:
        with store["lock"]:
            if not store["artifact_jobs"]:
                store["artifact_running"] = False
                store["artifact_stage"] = ""
                store["artifact_updated_at"] = time.time()
                return

            job = store["artifact_jobs"].pop(0)
            kind = str(job["kind"])
            store["artifact_stage"] = kind
            store["artifact_updated_at"] = time.time()

        try:
            if kind == "tile":
                with store["lock"]:
                    existing_tiles = deepcopy(store.get("tiles", []))

                tile = synthesize_single_tile(
                    job["history_tile"],
                    job["superpowers"],
                    existing_tiles=existing_tiles,
                )

                if tile:
                    image_prompt = (tile.get("image_prompt") or "").strip()
                    if image_prompt:
                        try:
                            tile["image_b64"] = cached_pixel_art(image_prompt)
                        except Exception:
                            logger.exception("Tile art error")
                            tile["image_b64"] = None
                    with store["lock"]:
                        store["tiles"].append(tile)
                        store["tile_count"] = int(store.get("tile_count", 0)) + 1
                        store["last_canvas_error"] = None
                        store["artifact_updated_at"] = time.time()

            elif kind == "recap":
                recap = generate_hero_recap(deepcopy(job["superpowers"]))
                with store["lock"]:
                    if recap:
                        store["recap"] = recap
                    store["last_identity_error"] = None
                    store["artifact_updated_at"] = time.time()

            elif kind == "avatar":
                avatar_b64 = generate_custom_avatar(
                    deepcopy(job["superpowers"]),
                    latest_signal=str(job.get("latest_signal", "")),
                )
                with store["lock"]:
                    if avatar_b64:
                        store["avatar_b64"] = avatar_b64
                    store["last_identity_error"] = None
                    store["artifact_updated_at"] = time.time()

            elif kind == "comic":
                comic_panels = generate_identity_comic(
                    deepcopy(job["superpowers"]),
                    recent_chat=deepcopy(job["history_chat"]),
                )
                with store["lock"]:
                    if comic_panels:
                        store["comic_panels"] = comic_panels
                    store["last_identity_error"] = None
                    store["artifact_updated_at"] = time.time()

            elif kind == "postcard":
                postcard = generate_future_postcard(
                    deepcopy(job["superpowers"]),
                    recent_chat=deepcopy(job["history_chat"]),
                )
                with store["lock"]:
                    if postcard:
                        store["postcard"] = postcard
                    store["last_identity_error"] = None
                    store["artifact_updated_at"] = time.time()

        except Exception:
            logger.exception("Artifact job failed: %s", kind)
            with store["lock"]:
                if kind == "tile":
                    store["last_canvas_error"] = "Workspace artifact generation failed."
                else:
                    store["last_identity_error"] = f"{kind.capitalize()} generation failed."
                store["artifact_updated_at"] = time.time()


def user_submit(
    user_message: Any,
    history: Optional[List[Dict[str, Any]]],
    request: gr.Request,
):
    history = history or []
    store = get_session_store(request)

    if isinstance(user_message, dict):
        text = (user_message.get("text") or "").strip()
        files = user_message.get("files") or []
    else:
        text = str(user_message or "").strip()
        files = []

    if not text and not files:
        return gr.update(value={"text": "", "files": []}), history

    pending_image = None
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
                pending_image = {"bytes": img_bytes, "mime": mime}
        except Exception:
            logger.exception("Image read error")
            pending_image = None

    with store["lock"]:
        store["pending_image"] = pending_image
        store["last_seen_at"] = time.time()

    display_content = text or "📎 [image attached]"
    history.append({"role": "user", "content": display_content})
    return gr.update(value={"text": "", "files": []}), history


def run_turn(
    history: List[Dict[str, Any]],
    request: gr.Request,
) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    store = get_session_store(request)

    if not history:
        yield (
            gr.update(interactive=True),
            history,
            format_canvas(store),
            render_identity_lab(store),
            get_header_status_html(store),
        )
        return

    latest_user_input = clean_message_for_backend(history[-1].get("content", ""))

    with store["lock"]:
        store["chat_turn_count"] = int(store.get("chat_turn_count", 0)) + 1
        store["last_seen_at"] = time.time()
        pending = deepcopy(store.get("pending_image"))
        has_superpowers = bool(store.get("superpowers"))

    if not has_superpowers:
        try:
            mapped = map_narrative_to_superpowers(
                latest_user_input,
                (pending or {}).get("bytes"),
                (pending or {}).get("mime", "image/jpeg"),
            ) or {}
        except Exception:
            logger.exception("Superpower mapping error")
            mapped = {}
        with store["lock"]:
            store["superpowers"] = mapped
            store["last_seen_at"] = time.time()

    history.append({"role": "assistant", "content": ""})
    yield (
        gr.update(interactive=False),
        history,
        format_canvas(store),
        render_identity_lab(store),
        get_header_status_html(store, busy=True, stage="Listening, reasoning, and composing..."),
    )

    backend_history = build_backend_history(history[:-1], MAX_CHAT_CONTEXT_MESSAGES)

    with store["lock"]:
        pending = deepcopy(store.get("pending_image"))
        store["pending_image"] = None
        superpowers = deepcopy(store.get("superpowers", {}))

    accumulated_text = ""
    inline_images: List[str] = []
    last_ui_flush = 0.0

    try:
        stream = generate_socratic_stream(
            superpowers,
            backend_history,
            (pending or {}).get("bytes"),
            (pending or {}).get("mime", "image/jpeg"),
        )
    except Exception:
        logger.exception("Stream setup error")
        history[-1]["content"] = "⚠️ Something glitched while starting the simulation."
        yield (
            gr.update(interactive=True),
            history,
            format_canvas(store),
            render_identity_lab(store),
            get_header_status_html(store),
        )
        return

    for chunk in stream:
        chunk_type = chunk.get("type")

        if chunk_type == "text":
            accumulated_text += str(chunk.get("data", ""))
            display_text, _, _ = extract_tool_json_and_display_text(accumulated_text)

            now = time.monotonic()
            if (now - last_ui_flush) >= STREAM_UPDATE_INTERVAL_SEC:
                preview = display_text or "🧠 *Thinking...*"
                preview = _RE_VISUALIZE.sub("🎨 *generating visual…*", preview)
                preview = _RE_SKILL.sub("🎯 *loading skill…*", preview)
                history[-1]["content"] = preview
                yield (
                    gr.update(interactive=False),
                    history,
                    format_canvas(store),
                    render_identity_lab(store),
                    get_header_status_html(store, busy=True, stage="Generating reply..."),
                )
                last_ui_flush = now

        elif chunk_type == "image":
            image_b64 = chunk.get("data")
            if image_b64 and len(inline_images) < MAX_INLINE_IMAGES_PER_TURN:
                inline_images.append(image_b64)

    final_text, _, _ = extract_tool_json_and_display_text(accumulated_text)
    if not final_text:
        final_text = "…"

    interleaved_html = render_interleaved_content(final_text, enable_visuals=True)
    if inline_images:
        interleaved_html += "\n\n" + format_inline_images(inline_images)

    history[-1]["content"] = interleaved_html

    jobs = plan_artifacts(store)
    enqueue_artifact_jobs(store, jobs, history, latest_user_input)

    yield (
        gr.update(interactive=True),
        history,
        format_canvas(store),
        render_identity_lab(store),
        get_header_status_html(store, busy=True, stage="Reply complete. Building artifacts..."),
    )


def refresh_panels(request: gr.Request):
    store = get_session_store(request)
    return (
        format_canvas(store),
        render_identity_lab(store),
        get_header_status_html(store),
    )


with gr.Blocks(css=css, theme=gr.themes.Base(), title=APP_TITLE) as demo:
    gr.HTML(
        """
        <div class='arc-header'>
            <div class='arc-logo'>🕹️ ArcMotivate</div>
            <div class='arc-tagline'>Interactive Career Explorer</div>
        </div>
        """
    )

    header_status = gr.HTML(get_header_status_html(default_session_store()))

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=380):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": render_interleaved_content(OPENING_MSG, enable_visuals=True)}],
                height=620,
                show_label=False,
                elem_classes=["chat-wrap"],
                render_markdown=True,
            )

            with gr.Column(elem_classes=["chat-input-shell"]):
                msg_input = gr.MultimodalTextbox(
                    placeholder="Type your message or add one image, then press Enter…",
                    show_label=False,
                    container=False,
                    file_types=["image"],
                    file_count="single",
                    autofocus=True,
                )
                gr.HTML(
                    "<div class='chat-input-hint'>Press <strong>Enter</strong> to send · "
                    "<strong>Shift+Enter</strong> for a new line · you can add one image too</div>"
                )

        with gr.Column(scale=4, min_width=340):
            with gr.Tabs():
                with gr.TabItem("⚡ Agent Workspace"):
                    canvas_output = gr.HTML(format_canvas(default_session_store()))
                with gr.TabItem("🧬 Your Story"):
                    identity_output = gr.HTML(render_identity_lab(default_session_store()))

    refresh_btn = gr.Button("Refresh story + workspace")

    submit_event = msg_input.submit(
        user_submit,
        [msg_input, chatbot],
        [msg_input, chatbot],
        queue=False,
    )

    submit_event.then(
        run_turn,
        [chatbot],
        [msg_input, chatbot, canvas_output, identity_output, header_status],
        concurrency_limit=1,
    )

    refresh_btn.click(
        refresh_panels,
        outputs=[canvas_output, identity_output, header_status],
        queue=False,
    )

    demo.queue(default_concurrency_limit=8)

if __name__ == "__main__":
    favicon = "assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        favicon_path=favicon,
    )