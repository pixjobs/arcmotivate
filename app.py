"""
ArcMotivate - Interactive Career Explorer
Main Gradio Application Entry Point.

Handles the UI, session state, background artifact generation, 
and streaming LLM responses with interleaved visual generation.
"""

import base64
import datetime
import hashlib
import html
import logging
import os
import pathlib
import re
import threading
import time
import uuid
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, Set

import gradio as gr
from dotenv import load_dotenv

# Optional import for Google Cloud Storage (Blob caching)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from lib.psychology_codex import map_narrative_to_superpowers
from lib.coaching_agent import generate_socratic_stream
from lib.outcome_engine import generate_intro_message, synthesize_single_tile
from lib.storybook_generator import (
    generate_custom_avatar,
    generate_future_postcard,
    generate_hero_story,
    generate_identity_comic,
    generate_pixel_art_illustration,
)

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================
APP_TITLE = "ArcMotivate"
CACHE_BUCKET_NAME = os.environ.get("CACHE_BUCKET_NAME")

FALLBACK_OPENING_MSG = (
    "👾 **System Online — ArcMotivate**\n\n"
    "[VISUALIZE: A neon pixel-art control room waking up, glowing screens, pathways branching into different futures]\n\n"
    "I map your input to find the future your mind demands. What moment keeps replaying in your head?\n\n"
    "*Send a message or attach an image to begin.*"
)

STREAM_UPDATE_INTERVAL_SEC = 0.05
MAX_CHAT_CONTEXT_MESSAGES = 12
MAX_TILE_HISTORY_MESSAGES = 6
MAX_INLINE_IMAGES_PER_TURN = 2
MAX_INLINE_VISUAL_MARKERS = 1
SESSION_TTL_SEC = 60 * 60
MAX_PENDING_ARTIFACT_JOBS = 5

# Pre-compiled Regex for performance
_RE_VISUALIZE = re.compile(r"\[VISUALIZE:\s*(.+?)\]", re.DOTALL)
_RE_SKILL = re.compile(r"\[SKILL:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\]", re.DOTALL)
_RE_HTML_IMG = re.compile(r"<img[^>]*>", re.IGNORECASE)
_RE_HTML_DIV = re.compile(r"<div[^>]*>.*?</div>", re.DOTALL | re.IGNORECASE)

# =====================================================================
# CSS STYLES
# =====================================================================
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@600;700&family=Press+Start+2P&display=swap');

#hidden_btn { display: none !important; }

:root{
  --bg-0:#05030b;--bg-1:#0a0314;--bg-2:#15092a;--surface:rgba(20,10,40,.9);
  --text:#f8fafc;--text-soft:#dbe7f3;--text-dim:#9fb0c4;
  --cyan:#22d3ee;--pink:#ff4de3;--violet:#7c3aed;
  --border-cyan:rgba(34,211,238,.2);
  --radius-md:10px;--radius-lg:12px;
  --shadow-1:0 4px 12px rgba(0,0,0,.2);--shadow-2:0 8px 20px rgba(0,0,0,.3);
  --glow-soft:0 0 0 1px rgba(34,211,238,.08),0 0 16px rgba(34,211,238,.06);
  --font-ui:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --font-display:'Press Start 2P',cursive;
  --font-accent:'Orbitron',sans-serif;
  --control-h:44px;--container-w:1200px;
}

*,*::before,*::after{box-sizing:border-box}
html,body{min-height:100%}
body,.gradio-container{
  margin:0!important;padding:0!important;color:var(--text)!important;font-family:var(--font-ui)!important;
  background:
    radial-gradient(circle at 12% 18%,rgba(255,77,227,.06),transparent 18%),
    radial-gradient(circle at 86% 22%,rgba(34,211,238,.06),transparent 18%),
    radial-gradient(ellipse at 20% 40%,var(--bg-2) 0%,var(--bg-1) 58%,#000 100%)!important
}
.gradio-container{max-width:var(--container-w)!important;margin:0 auto!important;padding:12px 10px 16px!important;border:none!important}

.arc-header{text-align:center;padding:4px 8px 8px}
.arc-logo{
  font-family:var(--font-display);font-size:clamp(1.1rem,1.8vw,1.6rem);line-height:1.2;letter-spacing:1px;
  color:var(--pink);text-shadow:0 0 6px rgba(255,77,227,.4),0 0 12px rgba(34,211,238,.1);padding-bottom:4px
}
.arc-tagline{
  font-size:clamp(0.8rem,1vw,0.9rem);letter-spacing:1px;text-transform:uppercase;color:var(--cyan);opacity:.9
}
.arc-status-wrap{display:flex;justify-content:center;margin-top:8px}
.arc-status{
  display:inline-flex;align-items:center;gap:8px;min-height:28px;padding:4px 12px;border-radius:999px;
  background:rgba(12,10,30,.6);border:1px solid rgba(34,211,238,.18);box-shadow:var(--glow-soft);
  color:var(--text-soft);font-size:.8rem;font-weight:500;
}
.arc-status-dot{
  width:8px;height:8px;flex:0 0 auto;border-radius:999px;background:var(--cyan);
  box-shadow:0 0 8px rgba(34,211,238,.7)
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
.scroll-panel{max-height:55vh;overflow-y:auto;overflow-x:hidden;padding-right:4px}

.gradio-chatbot{padding:8px!important}
.gradio-chatbot .message{
  width:fit-content!important;max-width:min(88%,820px)!important;
  margin-bottom:8px!important;padding:10px 14px!important;border-radius:12px!important;box-shadow:var(--shadow-1)!important
}
.gradio-chatbot .message.bot{
  margin-left:4px!important;margin-right:auto!important;background:rgba(34,211,238,.08)!important;
  border:1px solid rgba(34,211,238,.15)!important;border-bottom-left-radius:4px!important
}
.gradio-chatbot .message.user{
  margin-left:auto!important;margin-right:4px!important;background:rgba(124,58,237,.12)!important;
  border:1px solid rgba(124,58,237,.18)!important;border-bottom-right-radius:4px!important
}
.gradio-chatbot .avatar-container,.gradio-chatbot .message-role{display:none!important}
.gradio-chatbot .prose,.gradio-chatbot .message *{font-family:var(--font-ui)!important}
.gradio-chatbot .prose{
  color:var(--text)!important;font-size:clamp(0.9rem,0.95vw,0.95rem)!important;line-height:1.5!important;
  overflow-wrap:anywhere!important;word-break:break-word!important
}

.chat-input-shell{
  margin-top:8px;padding:8px;background:linear-gradient(180deg,rgba(14,8,34,.94),rgba(10,5,24,.93))!important;
  border:1px solid var(--border-cyan)!important;border-radius:var(--radius-md)!important
}
.chat-input-shell .gradio-multimodaltextbox,
.chat-input-shell .gradio-textbox,
.chat-input-shell .wrap,
.chat-input-shell .container{
  background:transparent!important;border:none!important;box-shadow:none!important
}
.chat-input-shell textarea{
  min-height:var(--control-h)!important;max-height:140px!important;padding:10px 12px!important;
  font-family:var(--font-ui)!important;font-size:0.95rem!important;line-height:1.4!important;font-weight:500!important;
  background:var(--surface)!important;color:var(--text)!important;border:1px solid rgba(34,211,238,.2)!important;
  border-radius:8px!important;resize:vertical!important
}
.chat-input-hint{
  margin-top:6px;color:var(--text-dim);font:.8rem/1.4 var(--font-ui);text-align:center
}

.canvas-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;padding:8px 0}
.canvas-tile{
  overflow:hidden;
  background:rgba(15,10,46,.8);
  border:1px solid rgba(139,92,246,.18);
  border-radius:10px;
  position:relative;
  box-shadow:
    0 6px 16px rgba(0,0,0,.25),
    0 0 0 1px rgba(34,211,238,.08),
    0 0 16px rgba(34,211,238,.08);
}
.canvas-tile::after{
  content:"";
  position:absolute;
  inset:0;
  pointer-events:none;
  background:linear-gradient(180deg,rgba(255,255,255,.03),transparent 28%);
}
.tile-img,.tile-img-placeholder{
  width:100%;height:90px;border-bottom:1px solid rgba(34,211,238,.12)
}
.tile-img{display:block;object-fit:cover}
.tile-img-placeholder{
  display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#1e1b4b,#0f0a2e);
  color:#7c3aed;font-size:1.2rem
}
.tile-body{padding:10px}
.tile-category{
  margin-bottom:4px;color:#c084fc;font:700 .65rem/1 var(--font-accent);letter-spacing:.08em;text-transform:uppercase
}
.tile-title{margin:0 0 4px;color:#f0f4ff;font:600 0.9rem/1.3 var(--font-ui)}
.tile-desc{color:var(--text-soft);font:400 .85rem/1.4 var(--font-ui)}
.tile-skill-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.tile-skill-tag{
  display:inline-block;padding:2px 6px;border-radius:4px;
  background:rgba(139,92,246,.1);color:var(--cyan);
  font:600 .65rem/1.1 var(--font-accent);letter-spacing:.02em;
  border:1px solid rgba(34,211,238,.08)
}
.tile-links{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.tile-link{
  display:inline-block;padding:4px 8px;border-radius:999px;background:rgba(34,211,238,.08);color:var(--cyan);
  text-decoration:none;font-size:.8rem;border:1px solid rgba(34,211,238,.12)
}

.canvas-empty{
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;margin-top:10px;padding:32px 16px;
  text-align:center;color:#64748b;border:1px dashed rgba(139,92,246,.18);border-radius:10px;background:rgba(255,255,255,.01)
}
.canvas-empty-icon{font-size:2rem;opacity:.35}
.canvas-empty-text{max-width:260px;font-size:0.9rem;line-height:1.35;color:#7b8ba3}
.synth-spinner{padding:12px;text-align:center;color:#a78bfa;font-size:0.9rem;font-style:italic}

.chat-skill-card{
  display:flex;align-items:flex-start;gap:10px;margin:8px 0;
  padding:10px 12px;border-radius:10px;
  background:linear-gradient(135deg,rgba(139,92,246,.08),rgba(34,211,238,.06));
  border:1px solid rgba(139,92,246,.18);position:relative;overflow:hidden
}
.chat-skill-card::before{
  content:'';position:absolute;top:0;left:0;width:3px;height:100%;
  background:linear-gradient(180deg,var(--violet),var(--cyan))
}
.chat-skill-icon{font-size:1.2rem;flex-shrink:0;margin-top:2px}
.chat-skill-body{flex:1;min-width:0}
.chat-skill-name{font:600 .85rem/1.2 var(--font-accent);color:#fff;margin-bottom:2px}
.chat-skill-try{font:.85rem/1.35 var(--font-ui);color:var(--text-soft);margin-bottom:6px}
.chat-skill-link{
  display:inline-flex;align-items:center;gap:4px;padding:4px 10px;
  border-radius:6px;background:rgba(139,92,246,.12);color:var(--cyan);
  font:600 .7rem/1 var(--font-accent);letter-spacing:.03em;text-decoration:none;
  border:1px solid rgba(34,211,238,.12)
}

/* Cinematic 21:9 Aspect Ratio for Inline Visuals */
.chat-inline-visual{
  margin:8px 0;
  border-radius:8px;
  overflow:hidden;
  border:1px solid rgba(34,211,238,.16);
  box-shadow:var(--glow-soft);
  width: 100%;
  aspect-ratio: 21 / 9;
  background: rgba(15,10,46,.6);
  display: flex;
  align-items: center;
  justify-content: center;
}
.chat-inline-visual img{
  display:block;
  width:100%;
  height:100%;
  object-fit:cover;
}
.chat-inline-visual.loading {
  color: var(--cyan);
  font-size: 0.85rem;
  font-style: italic;
  animation: pulse 1.5s infinite;
}
@keyframes pulse {
  0% { opacity: 0.5; }
  50% { opacity: 1; }
  100% { opacity: 0.5; }
}

.chat-comic-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:6px;margin-top:8px
}
.chat-comic-grid img{
  display:block;width:100%;height:80px;object-fit:cover;border:1px solid rgba(34,211,238,.16);border-radius:6px
}

.story-card{
  padding:16px;border-radius:12px;
  background:linear-gradient(145deg,rgba(15,10,30,.95),rgba(20,15,40,.9));
  border:1px solid rgba(139,92,246,.14)
}
.story-card-title{
  font:700 1.1rem/1.2 var(--font-accent);
  color: var(--cyan);
  margin-bottom:14px;
}
@supports (-webkit-background-clip:text) {
  .story-card-title{
    background: linear-gradient(90deg, var(--violet), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
}

.story-section{margin-bottom:16px}
.story-section-label{font:600 .7rem/1 var(--font-accent);color:var(--cyan);letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px}
.story-narrative{font:.9rem/1.5 var(--font-ui);color:var(--text-soft)}
.story-avatar{
  width:90px;height:90px;border-radius:50%;object-fit:cover;
  border:2px solid rgba(139,92,246,.3);margin:0 auto;display:block
}
.story-comic-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:6px}
.story-comic-panel{
  border-radius:8px;overflow:hidden;background:rgba(255,255,255,.02);
  border:1px solid rgba(34,211,238,.1)
}
.story-comic-panel img{display:block;width:100%;height:80px;object-fit:cover}
.story-comic-caption{padding:6px 8px;font:.75rem/1.3 var(--font-ui);color:var(--text-soft);text-align:center}
.story-postcard{
  padding:14px;border-radius:10px;
  background:linear-gradient(135deg,rgba(139,92,246,.08),rgba(34,211,238,.05));
  border:1px solid rgba(139,92,246,.12);text-align:center
}
.story-postcard img{display:block;width:100%;max-height:120px;object-fit:cover;border-radius:8px;margin-bottom:8px}
.story-postcard-caption{font:500 .85rem/1.4 var(--font-ui);color:#fff}
.story-nudge{
  padding:10px 12px;border-radius:8px;
  background:linear-gradient(135deg,rgba(34,211,238,.06),rgba(139,92,246,.04));
  border:1px solid rgba(34,211,238,.1);
  font:.85rem/1.4 var(--font-ui);color:var(--text-soft)
}
.story-nudge strong{color:var(--cyan)}
.identity-note{
  margin-top:8px;padding:8px 10px;border-radius:8px;background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.05);color:var(--text-soft);font:.85rem/1.4 var(--font-ui)
}

@media (max-width:768px){
  .gradio-container{padding:4px 2px 8px!important}
  .arc-header{padding:2px 4px 6px}
  .arc-logo{font-size:1.2rem; padding-bottom:2px}
  .arc-tagline{font-size:0.7rem; letter-spacing:0.5px}
  .gradio-chatbot{padding:4px!important}
  .gradio-chatbot .message{
    max-width:94%!important;
    padding:8px 10px!important;
    margin-bottom:6px!important;
  }
  .gradio-chatbot .prose{font-size:0.9rem!important; line-height:1.4!important}
  .canvas-grid, .story-comic-strip{grid-template-columns:1fr; gap:8px}
  .chat-input-shell{margin-top:6px; padding:4px}
  .chat-input-shell textarea{
    min-height:40px!important;
    font-size:16px!important; /* 16px prevents iOS zoom */
    padding:8px 10px!important;
  }
  .chat-input-hint{font-size:0.7rem; margin-top:4px}
  .story-avatar{width:70px; height:70px}
  .tab-nav button { padding: 6px 8px!important; font-size: 0.85rem!important; }
}
"""

# =====================================================================
# SESSION MANAGEMENT
# =====================================================================
SESSION_STORES: Dict[str, Dict[str, Any]] = {}
SESSION_STORES_LOCK = threading.Lock()

def default_session_store() -> Dict[str, Any]:
    """Returns a fresh session state dictionary."""
    return {
        "history": [],
        "pending_turns": [],
        "superpowers": {},
        "tiles": [],
        "chat_turn_count": 0,
        "tile_count": 0,
        "avatar_b64": "",
        "story": {},        # replaces plain-text recap: {narrative: str, links: list}
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
        "gen_lock": threading.Lock(),
    }

def _session_id_from_request(request: Optional[gr.Request]) -> str:
    """Extracts a unique session ID from the Gradio request."""
    if request and getattr(request, "session_hash", None):
        return str(request.session_hash)
    return "global"

def _cleanup_stale_sessions() -> None:
    """Removes sessions that have exceeded the TTL to free memory."""
    now = time.time()
    stale_ids: List[str] =[]
    with SESSION_STORES_LOCK:
        for session_id, store in SESSION_STORES.items():
            if session_id == "global":
                continue
            if (now - float(store.get("last_seen_at", now))) > SESSION_TTL_SEC:
                stale_ids.append(session_id)
        for session_id in stale_ids:
            SESSION_STORES.pop(session_id, None)

def get_session_store(request: Optional[gr.Request]) -> Dict[str, Any]:
    """Retrieves or creates a session store for the current user."""
    _cleanup_stale_sessions()
    session_id = _session_id_from_request(request)
    with SESSION_STORES_LOCK:
        if session_id not in SESSION_STORES:
            SESSION_STORES[session_id] = default_session_store()
        SESSION_STORES[session_id]["last_seen_at"] = time.time()
        return SESSION_STORES[session_id]

# =====================================================================
# TEXT & HTML UTILITIES
# =====================================================================
def safe_text(value: Any) -> str:
    """Escapes HTML characters to prevent XSS."""
    return html.escape(str(value or ""))

def extract_text(content: Any) -> str:
    """Extracts plain text from Gradio's multimodal message format."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: List[str] =[]
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts).strip()
    return str(content or "").strip()

def _extract_file_path(file_obj: Any) -> Optional[str]:
    """Safely extracts the file path from Gradio's file object."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "path", None) or getattr(file_obj, "name", None)

def clean_message_for_backend(content: Any) -> str:
    """Strips HTML and internal markers before sending to the LLM."""
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
    """Formats the Gradio history into the format expected by the LLM."""
    backend_history: List[Dict[str, str]] = []
    for msg in history[-max_messages:]:
        clean_content = clean_message_for_backend(msg.get("content", ""))
        if not clean_content:
            continue
        role = "user" if msg.get("role") == "user" else "model"
        backend_history.append({"role": role, "text": clean_content})
    return backend_history

def extract_tool_json_and_display_text(accumulated_text: str) -> Tuple[str, Optional[str], bool]:
    """Parses out internal JSON tool calls from the LLM's raw text stream."""
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

def _get_gradio_history(store: Dict[str, Any]) -> List[Dict[str, str]]:
    """Safely strips internal metadata (like images/ids) before sending history to the UI."""
    return [{"role": m["role"], "content": m["content"]} for m in store["history"]]

# =====================================================================
# DISTRIBUTED GENERATIVE ASSET POOL (4-Tier Image Cache)
# =====================================================================
@lru_cache(maxsize=96)
def cached_pixel_art(prompt: str) -> Optional[str]:
    """Generates or retrieves a cached image based on the prompt."""
    prompt = (prompt or "").strip()
    if not prompt:
        return None
        
    prompt_hash = hashlib.md5(prompt.lower().encode('utf-8')).hexdigest()
    blob_filename = f"img_cache_{prompt_hash}.txt"
    tmp_file = f"/tmp/{blob_filename}"

    if os.path.exists(tmp_file):
        try:
            with open(tmp_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    logger.info("Image loaded from local /tmp cache! (%s)", prompt_hash)
                    return content
        except Exception as e:
            logger.warning("Failed to read image from /tmp: %s", e)

    if GCS_AVAILABLE and CACHE_BUCKET_NAME:
        try:
            client = storage.Client()
            bucket = client.bucket(CACHE_BUCKET_NAME)
            blob = bucket.blob(f"images/{blob_filename}")
            
            if blob.exists():
                logger.info("Image loaded from GCS Blob cache! (%s)", prompt_hash)
                b64_data = blob.download_as_text()
                
                with open(tmp_file, "w", encoding="utf-8") as f:
                    f.write(b64_data)
                return b64_data
        except Exception as e:
            logger.warning("Failed to read image from GCS Blob: %s", e)

    logger.info("Generating new image via API... (%s)", prompt_hash)
    try:
        b64_data = generate_pixel_art_illustration(prompt)
        if b64_data:
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(b64_data)
                
            if GCS_AVAILABLE and CACHE_BUCKET_NAME:
                try:
                    client = storage.Client()
                    bucket = client.bucket(CACHE_BUCKET_NAME)
                    blob = bucket.blob(f"images/{blob_filename}")
                    blob.upload_from_string(b64_data, content_type="text/plain")
                    logger.info("New image added to global GCS library.")
                except Exception as e:
                    logger.warning("Failed to upload image to GCS Blob: %s", e)
                    
            return b64_data
    except Exception as e:
        logger.exception("Image generation failed: %s", e)
        
    return None

def maybe_generate_inline_visual(prompt: Optional[str]) -> Optional[str]:
    """Wrapper to safely attempt inline visual generation."""
    if not prompt:
        return None
    try:
        return cached_pixel_art(prompt)
    except Exception as e:
        logger.exception("Inline image generation failed: %s", e)
        return None

def format_inline_images(image_b64_list: List[str]) -> str:
    """Formats a list of base64 images into an HTML grid."""
    if not image_b64_list:
        return ""
    tags =[]
    for b64 in image_b64_list[:MAX_INLINE_IMAGES_PER_TURN]:
        safe_b64 = html.escape(b64, quote=True)
        tags.append(f"<img src='data:image/png;base64,{safe_b64}' alt='arc visual'>")
    return f"<div class='chat-comic-grid'>{''.join(tags)}</div>"

def format_skill_card(name: str, url: str, try_this: str) -> str:
    """Formats a skill recommendation into an HTML card."""
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

def format_inline_visual_html(image_b64: str) -> str:
    """Formats a single base64 image into a cinematic inline visual."""
    safe_b64 = html.escape(image_b64, quote=True)
    return (
        f"<div class='chat-inline-visual'>"
        f"<img src='data:image/png;base64,{safe_b64}' alt='visual metaphor'>"
        f"</div>"
    )

def render_interleaved_content(raw_text: str, visual_state: str = "rendered") -> str:
    """
    Parses custom tags (e.g., [VISUALIZE: ...]) and replaces them with HTML.
    visual_state can be:
    - "hidden": Strips the visual tags entirely.
    - "loading": Replaces the tag with a glowing placeholder box.
    - "rendered": Blocks to generate the image and replaces the tag with the final image.
    """
    parts: List[str] =[]
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
            if visual_state != "hidden" and visuals_used < MAX_INLINE_VISUAL_MARKERS:
                if visual_state == "loading":
                    parts.append("<div class='chat-inline-visual loading'>🎨 <em>Visualizing...</em></div>")
                elif visual_state == "rendered":
                    img_b64 = maybe_generate_inline_visual(prompt)
                    if img_b64:
                        parts.append(format_inline_visual_html(img_b64))
                    else:
                        parts.append("<div class='chat-inline-visual loading' style='color:#ef4444'>⚠️ Visual failed</div>")
                visuals_used += 1
            remainder = remainder[vis_m.end():]

        elif earliest is skill_m:
            remainder = remainder[skill_m.end():]

        else:
            break

    return "\n\n".join(parts)

# =====================================================================
# MULTI-CONTAINER BLOB CACHING FOR RENDERED INTRO MESSAGE (DAILY ROTATION)
# =====================================================================
_intro_lock = threading.Lock()
_MEM_CACHE_INTRO: Optional[str] = None
_MEM_CACHE_DATE: Optional[str] = None

def get_rendered_opening_message() -> str:
    """Generates or retrieves the daily rotating intro message."""
    global _MEM_CACHE_INTRO, _MEM_CACHE_DATE

    today_str = datetime.date.today().isoformat()
    blob_filename = f"arc_intro_rendered_{today_str}.txt"
    tmp_intro_file = f"/tmp/{blob_filename}"

    if _MEM_CACHE_INTRO and _MEM_CACHE_DATE == today_str:
        return _MEM_CACHE_INTRO

    if os.path.exists(tmp_intro_file):
        try:
            with open(tmp_intro_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    _MEM_CACHE_INTRO = content
                    _MEM_CACHE_DATE = today_str
                    return content
        except Exception as e:
            logger.warning("Failed to read intro from local /tmp: %s", e)

    with _intro_lock:
        if os.path.exists(tmp_intro_file):
            with open(tmp_intro_file, "r", encoding="utf-8") as f:
                _MEM_CACHE_INTRO = f.read().strip()
                _MEM_CACHE_DATE = today_str
                return _MEM_CACHE_INTRO

        if GCS_AVAILABLE and CACHE_BUCKET_NAME:
            try:
                client = storage.Client()
                bucket = client.bucket(CACHE_BUCKET_NAME)
                blob = bucket.blob(blob_filename)
                
                if blob.exists():
                    logger.info("Downloading today's cached intro message from GCS Blob...")
                    content = blob.download_as_text()
                    with open(tmp_intro_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    _MEM_CACHE_INTRO = content
                    _MEM_CACHE_DATE = today_str
                    return content
            except Exception as e:
                logger.warning("Failed to read from GCS Blob: %s", e)

    try:
        logger.info("Generating new intro message for %s via LLM...", today_str)
        raw_msg = generate_intro_message()
        
        if raw_msg:
            rendered_msg = render_interleaved_content(raw_msg, visual_state="rendered")
            
            with _intro_lock:
                with open(tmp_intro_file, "w", encoding="utf-8") as f:
                    f.write(rendered_msg)
                _MEM_CACHE_INTRO = rendered_msg
                _MEM_CACHE_DATE = today_str
                
            if GCS_AVAILABLE and CACHE_BUCKET_NAME:
                try:
                    logger.info("Uploading today's new intro to GCS Blob...")
                    client = storage.Client()
                    bucket = client.bucket(CACHE_BUCKET_NAME)
                    blob = bucket.blob(blob_filename)
                    blob.upload_from_string(rendered_msg, content_type="text/html")
                except Exception as e:
                    logger.warning("Failed to upload to GCS Blob: %s", e)

            return rendered_msg
            
    except Exception as e:
        logger.exception("Failed to generate opening message: %s", e)
        
    return render_interleaved_content(FALLBACK_OPENING_MSG, visual_state="rendered")

# =====================================================================
# UI RENDERING FUNCTIONS
# =====================================================================
def get_header_status_html(store: Dict[str, Any], busy: bool = False, stage: str = "") -> str:
    """Generates the HTML for the top status bar."""
    with store["lock"]:
        turn_count = int(store.get("chat_turn_count", 0))
        tiles_count = len(store.get("tiles",[]))
        artifact_running = bool(store.get("artifact_running"))
        artifact_stage = str(store.get("artifact_stage", "")).strip()
        pending_jobs = len(store.get("artifact_jobs",[]))

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
    """Generates the HTML for the Workspace Canvas tab."""
    with store["lock"]:
        tiles = deepcopy(store.get("tiles",[]))
        error_text = str(store.get("last_canvas_error") or "")
        artifact_running = bool(store.get("artifact_running"))
        artifact_stage = str(store.get("artifact_stage", "")).strip()

    busy_text = ""
    if artifact_running and artifact_stage == "tile":
        busy_text = "🧩 Building workspace artifact..."

    if not tiles and not busy_text and not error_text:
        return """
        <div class='scroll-panel canvas-empty'>
            <div class='canvas-empty-icon'>🧩</div>
            <div class='canvas-empty-text'>Your workspace is empty.<br>Start chatting to build your profile.</div>
        </div>
        """

    html_parts =[]
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

            links_html =[]
            for link in tile.get("links") or[]:
                if isinstance(link, dict):
                    label = safe_text(link.get("label", "Explore"))
                    url = str(link.get("url", "")).strip()
                    if url.startswith(("http://", "https://")):
                        links_html.append(
                            f"<a href='{html.escape(url, quote=True)}' target='_blank' "
                            f"rel='noopener noreferrer' class='tile-link'>🔗 {label}</a>"
                        )

            skill_tags = tile.get("skill_tags") or[]
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

    return f"<div class='scroll-panel'>{''.join(html_parts)}</div>"

def render_identity_lab(store: Dict[str, Any]) -> str:
    """Generates the HTML for the Story tab."""
    with store["lock"]:
        avatar_b64 = store.get("avatar_b64", "")
        story = deepcopy(store.get("story", {}))
        superpowers = deepcopy(store.get("superpowers", {}))
        comic_panels = deepcopy(store.get("comic_panels",[]))
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

    has_content = bool(avatar_b64 or story or comic_panels or postcard)

    if not has_content and not busy_text and not error_text:
        return """
        <div class='scroll-panel story-card' style='text-align:center;padding:40px 20px;color:#7b8ba3'>
            <div style='font-size:2rem;margin-bottom:10px;opacity:0.4'>📖</div>
            <div style='font-size:0.9rem'>Your story hasn't started yet.<br>Share what you care about.</div>
        </div>
        """

    sections: List[str] =[]

    if busy_text:
        sections.append(f"<div class='synth-spinner'>{safe_text(busy_text)}</div>")
    if error_text:
        sections.append(f"<div class='identity-note'>⚠️ {safe_text(error_text)}</div>")

    if story and story.get("narrative"):
        narrative_html = safe_text(story["narrative"])
        # Build 3 resource link cards
        pillar_icons = ["🎯", "🛠️", "🌟"]
        link_cards_html = []
        for i, lnk in enumerate(story.get("links") or []):
            icon = pillar_icons[i] if i < len(pillar_icons) else "🔗"
            label = safe_text(lnk.get("label", "Explore"))
            desc  = safe_text(lnk.get("description", ""))
            url   = html.escape(str(lnk.get("url", "")), quote=True)
            link_cards_html.append(f"""
            <div class='chat-skill-card'>
                <div class='chat-skill-icon'>{icon}</div>
                <div class='chat-skill-body'>
                    <div class='chat-skill-name'>{label}</div>
                    <div class='chat-skill-try'>{desc}</div>
                    <a class='chat-skill-link' href='{url}' target='_blank' rel='noopener noreferrer'>🔍 Search</a>
                </div>
            </div>
            """)
        sections.append(f"""
        <div class='story-section'>
            <div class='story-section-label'>📖 Your Exploration Story</div>
            <div class='story-narrative'>{narrative_html}</div>
            <div style='margin-top:12px'>{''.join(link_cards_html)}</div>
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

    return f"<div class='scroll-panel story-card'><div class='story-card-title'>Your Exploration Story</div>{''.join(sections)}</div>"

# =====================================================================
# BACKGROUND ARTIFACT GENERATION
# =====================================================================
def plan_artifacts(store: Dict[str, Any]) -> List[str]:
    """Determines which artifacts need to be generated based on turn count.

    Refresh cadence (after turn 2):
      - tile:     every turn (always fresh)
      - recap:    every 2 turns — synced with comic so Story tab is always coherent
      - comic:    every 2 turns — alternates with postcard to spread image gen load
      - postcard: every 3 turns
      - avatar:   once (generated on turn 1, regenerated only if missing)
    """
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
        jobs = ["tile", "recap", "comic"]
        if avatar_missing:
            jobs.append("avatar")
        return jobs

    # turn >= 3: tile every turn; recap + comic together every 2 turns; postcard every 3
    jobs = ["tile"]
    if turn % 2 == 0:
        jobs.append("recap")
        jobs.append("comic")
    if avatar_missing:
        jobs.append("avatar")
    if comic_missing:
        jobs.append("comic")  # ensure comic is generated at least once
    if postcard_missing or turn % 3 == 0:
        jobs.append("postcard")
    return jobs

def enqueue_artifact_jobs(
    store: Dict[str, Any],
    jobs: List[str],
    history: List[Dict[str, Any]],
    latest_signal: str,
) -> None:
    """Adds artifact generation jobs to the background queue with strict language enforcement."""
    if not jobs:
        return

    history_chat = build_backend_history(history, MAX_CHAT_CONTEXT_MESSAGES)
    history_tile = build_backend_history(history, MAX_TILE_HISTORY_MESSAGES)

    language_directive = (
        "[SYSTEM DIRECTIVE: Analyze the chat history above. You MUST generate all "
        "titles, descriptions, captions, and text for this artifact in the EXACT SAME "
        "LANGUAGE that the user is speaking. Do not default to English unless the user is speaking English. "
        "HOWEVER, any 'image_prompt' or visual generation instructions MUST be written in English.]"
    )
    
    localized_signal = f"{latest_signal}\n\n{language_directive}"

    history_chat.append({"role": "user", "text": language_directive})
    history_tile.append({"role": "user", "text": language_directive})

    with store["lock"]:
        superpowers = deepcopy(store.get("superpowers", {}))
        queued_kinds: Set[str] = {job.get("kind") for job in store["artifact_jobs"]}

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
                    "latest_signal": localized_signal,
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
    """Background thread worker that processes the artifact queue."""
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
                    existing_tiles = deepcopy(store.get("tiles",[]))

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
                        except Exception as e:
                            logger.exception("Tile art error: %s", e)
                            tile["image_b64"] = None
                    with store["lock"]:
                        store["tiles"].append(tile)
                        store["tile_count"] = int(store.get("tile_count", 0)) + 1
                        store["last_canvas_error"] = None
                        store["artifact_updated_at"] = time.time()

            elif kind == "recap":
                story_data = generate_hero_story(
                    deepcopy(job["superpowers"]),
                    recent_chat=deepcopy(job["history_chat"]),
                )
                with store["lock"]:
                    if story_data:
                        store["story"] = story_data
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

        except Exception as e:
            logger.exception("Artifact job failed (%s): %s", kind, e)
            with store["lock"]:
                if kind == "tile":
                    store["last_canvas_error"] = "Workspace artifact generation failed."
                else:
                    store["last_identity_error"] = f"{kind.capitalize()} generation failed."
                store["artifact_updated_at"] = time.time()

# =====================================================================
# GRADIO EVENT HANDLERS
# =====================================================================
def user_submit(
    user_message: Any,
    request: gr.Request,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Handles the immediate UI update when a user submits a message."""
    store = get_session_store(request)

    if isinstance(user_message, dict):
        text = (user_message.get("text") or "").strip()
        files = user_message.get("files") or []
    else:
        text = str(user_message or "").strip()
        files = []

    if not text and not files:
        with store["lock"]:
            return gr.update(value={"text": "", "files": []}), _get_gradio_history(store), get_header_status_html(store)

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
        except Exception as e:
            logger.exception("Image read error: %s", e)
            pending_image = None

    display_content = text or "📎 [image attached]"
    
    with store["lock"]:
        # 1. Create a unique message object that holds its own image payload
        msg = {
            "id": uuid.uuid4().hex,
            "role": "user",
            "content": display_content,
            "image": pending_image,
        }
        store["history"].append(msg)
        
        # 2. Add this specific message's index to the processing queue
        store["pending_turns"].append(len(store["history"]) - 1)
        store["last_seen_at"] = time.time()
        
        is_busy = store["gen_lock"].locked()
        
    stage = "Message queued..." if is_busy else "Message received..."
    status_html = get_header_status_html(store, busy=True, stage=stage)
        
    return gr.update(value={"text": "", "files": []}), _get_gradio_history(store), status_html

def run_turn(
    request: gr.Request,
) -> Generator[Tuple[List[Dict[str, Any]], str, str, str], None, None]:
    """Main generator that pops a message from the queue and processes it."""
    store = get_session_store(request)

    with store["gen_lock"]:
        with store["lock"]:
            if not store["history"]:
                store["history"] = [{"role": "assistant", "content": get_rendered_opening_message()}]
                
            if not store["pending_turns"]:
                yield (
                    _get_gradio_history(store),
                    format_canvas(store),
                    render_identity_lab(store),
                    get_header_status_html(store),
                )
                return

            # 3. Pop the exact user message we are supposed to process
            user_index = store["pending_turns"].pop(0)

            if user_index >= len(store["history"]):
                return 

            user_msg = store["history"][user_index]
            if user_msg.get("role") != "user":
                return

            latest_user_input = clean_message_for_backend(user_msg.get("content", ""))
            pending = deepcopy(user_msg.get("image"))
            has_superpowers = bool(store.get("superpowers"))

            store["chat_turn_count"] = int(store.get("chat_turn_count", 0)) + 1
            store["last_seen_at"] = time.time()

            # 4. Insert the assistant reply directly after THIS user message
            msg_index = user_index + 1
            store["history"].insert(msg_index, {"role": "assistant", "content": "⏳ *Thinking...*"})

            # 5. Shift any pending indices that come after the insertion point
            store["pending_turns"] = [
                idx + 1 if idx > user_index else idx
                for idx in store["pending_turns"]
            ]

        yield (
            _get_gradio_history(store),
            format_canvas(store),
            render_identity_lab(store),
            get_header_status_html(store, busy=True, stage="Listening, reasoning, and composing..."),
        )

        if not has_superpowers:
            try:
                mapped = map_narrative_to_superpowers(
                    latest_user_input,
                    (pending or {}).get("bytes"),
                    (pending or {}).get("mime", "image/jpeg"),
                ) or {}
            except Exception as e:
                logger.exception("Superpower mapping error: %s", e)
                mapped = {}
            with store["lock"]:
                store["superpowers"] = mapped
                store["last_seen_at"] = time.time()

        with store["lock"]:
            turn_count = int(store.get("chat_turn_count", 0))
            # 6. Build history ONLY up to the message we are responding to
            backend_history = build_backend_history(store["history"][:msg_index], MAX_CHAT_CONTEXT_MESSAGES)
            superpowers = deepcopy(store.get("superpowers", {}))

        # Periodic superpowers refresh: re-read the full conversation every 4 turns
        # so the psychology profile evolves with what the child revealed in conversation.
        if turn_count > 0 and turn_count % 4 == 0:
            try:
                logger.info("Refreshing superpowers at turn %s...", turn_count)
                refreshed_text = " ".join(
                    m.get("text", "") for m in backend_history[-8:] if m.get("role") == "user"
                ).strip()
                if refreshed_text:
                    refreshed = map_narrative_to_superpowers(refreshed_text) or {}
                    if len(refreshed) >= 9:  # all required fields present
                        with store["lock"]:
                            store["superpowers"] = refreshed
                            superpowers = deepcopy(refreshed)
                        logger.info("Superpowers refreshed successfully at turn %s.", turn_count)
            except Exception as e:
                logger.warning("Superpowers refresh failed (non-critical): %s", e)

        accumulated_text = ""
        inline_images: List[str] = []
        last_ui_flush = 0.0

        try:
            stream = generate_socratic_stream(
                superpowers,
                backend_history,
                turn_count=turn_count,
                image_bytes=(pending or {}).get("bytes"),
                image_mime=(pending or {}).get("mime", "image/jpeg"),
            )
        except Exception as e:
            logger.exception("Stream setup error: %s", e)
            with store["lock"]:
                store["history"][msg_index]["content"] = "⚠️ Something glitched while starting the simulation."
            yield (
                _get_gradio_history(store),
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
                    preview_html = render_interleaved_content(preview, visual_state="loading")
                    with store["lock"]:
                        store["history"][msg_index]["content"] = preview_html
                    yield (
                        _get_gradio_history(store),
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

        interleaved_html_loading = render_interleaved_content(final_text, visual_state="loading")
        if inline_images:
            interleaved_html_loading += "\n\n" + format_inline_images(inline_images)

        with store["lock"]:
            store["history"][msg_index]["content"] = interleaved_html_loading

        yield (
            _get_gradio_history(store),
            format_canvas(store),
            render_identity_lab(store),
            get_header_status_html(store, busy=True, stage="Painting visual..."),
        )

        interleaved_html_final = render_interleaved_content(final_text, visual_state="rendered")
        if inline_images:
            interleaved_html_final += "\n\n" + format_inline_images(inline_images)

        with store["lock"]:
            store["history"][msg_index]["content"] = interleaved_html_final

        jobs = plan_artifacts(store)
        enqueue_artifact_jobs(store, jobs, store["history"][:msg_index+1], latest_user_input)

        yield (
            _get_gradio_history(store),
            format_canvas(store),
            render_identity_lab(store),
            get_header_status_html(store, busy=True, stage="Reply complete. Building artifacts..."),
        )

def handle_startup_load(request: gr.Request) -> List[Dict[str, str]]:
    """Generates the fresh opening message lazily on first load."""
    session_id = _session_id_from_request(request)
    
    with SESSION_STORES_LOCK:
        SESSION_STORES[session_id] = default_session_store()
        store = SESSION_STORES[session_id]
        
    rendered_msg = get_rendered_opening_message()
    with store["lock"]:
        store["history"] = [{"role": "assistant", "content": rendered_msg}]
        return _get_gradio_history(store)

def refresh_panels(request: gr.Request) -> Tuple[str, str, str]:
    """Refreshes the side panels manually."""
    store = get_session_store(request)
    return (
        format_canvas(store),
        render_identity_lab(store),
        get_header_status_html(store),
    )

# =====================================================================
# GRADIO UI LAYOUT
# =====================================================================
with gr.Blocks(title=APP_TITLE) as demo:
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
        with gr.Column(scale=6, min_width=280):
            chatbot = gr.Chatbot(
                height="55vh",
                show_label=False,
                elem_classes=["chat-wrap"],
                render_markdown=True,
            )

            with gr.Column(elem_classes=["chat-input-shell"]):
                msg_input = gr.MultimodalTextbox(
                    placeholder="Type a message or add an image...",
                    show_label=False,
                    container=False,
                    file_types=["image"],
                    file_count="single",
                    autofocus=True,
                )
                gr.HTML(
                    "<div class='chat-input-hint'>Press <strong>Enter</strong> to send · "
                    "<strong>Shift+Enter</strong> for a new line</div>"
                )

        with gr.Column(scale=4, min_width=280):
            with gr.Tabs():
                with gr.TabItem("⚡ Workspace"):
                    canvas_output = gr.HTML(format_canvas(default_session_store()))
                with gr.TabItem("🧬 Story"):
                    identity_output = gr.HTML(render_identity_lab(default_session_store()))

    refresh_btn = gr.Button("Refresh story + workspace")

    hidden_btn = gr.Button("Hidden", elem_id="hidden_btn")

    submit_event = msg_input.submit(
        user_submit, 
        inputs=[msg_input], 
        outputs=[msg_input, chatbot, header_status],
        queue=False, 
    )

    submit_event.then(
        fn=None, 
        inputs=None, 
        outputs=None, 
        js="""
        () => {
            setTimeout(() => {
                const el = document.getElementById('hidden_btn');
                if (el) {
                    const btn = el.tagName.toLowerCase() === 'button' ? el : el.querySelector('button');
                    if (btn) btn.click();
                }
            }, 50);
        }
        """
    )

    # 7. The Magic Bullet: trigger_mode="multiple" ensures no clicks are dropped!
    hidden_btn.click(
        run_turn, 
        inputs=None, 
        outputs=[chatbot, canvas_output, identity_output, header_status],
        trigger_mode="multiple", 
    )

    refresh_btn.click(
        refresh_panels,
        outputs=[canvas_output, identity_output, header_status],
        queue=False,
    )

    demo.queue(default_concurrency_limit=8)
    demo.load(handle_startup_load, outputs=chatbot)

if __name__ == "__main__":
    favicon = "assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None
    port = int(os.environ.get("PORT", 8080))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        favicon_path=favicon,
        share=False,
        theme=gr.themes.Base(),
        css=css,
    )