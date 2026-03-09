import base64
import logging
from typing import Any, Dict, Iterator, List, Optional
from lib.image_utils import compress_generated_image

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY = "The Explorer"
DEFAULT_SECONDARY = "Curiosity-Driven"
DEFAULT_SUPERPOWER = "Curiosity"
DEFAULT_DESCRIPTION = "You notice what pulls you in and learn by trying."
DEFAULT_GROWTH_NUDGE = "Try one small new thing this week."
DEFAULT_GREETING = "Hi"

# OPTIMIZATION 1: Reduce history to 6 messages (3 turns). Less reading = faster response.
MAX_HISTORY_MESSAGES = 6 

# Kept your original model
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview" 
DEFAULT_TEMPERATURE = 0.65

# OPTIMIZATION 2: Cache the client globally so we don't rebuild it on every request
_CLIENT = None

def get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing. "
                             "Check your Cloud Run 'Variables & Secrets' configuration.")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _normalize_role(role: str) -> str:
    role = (role or "user").strip().lower()
    if role == "assistant":
        return "model"
    if role not in {"user", "model"}:
        return "user"
    return role


def _safe_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _build_system_instruction(superpowers: Dict[str, Any]) -> str:
    primary = _safe_str(superpowers.get("primary"), DEFAULT_PRIMARY)
    secondary = _safe_str(superpowers.get("secondary"), DEFAULT_SECONDARY)
    superpower = _safe_str(superpowers.get("superpower"), DEFAULT_SUPERPOWER)
    description = _safe_str(superpowers.get("description"), DEFAULT_DESCRIPTION)
    growth_nudge = _safe_str(superpowers.get("growth_nudge"), DEFAULT_GROWTH_NUDGE)

    # OPTIMIZATION 3: "Goldilocks" prompt. Fast TTFT (Time-To-First-Token) but retains richness.
    return f"""
You are ArcMotivate, a live interface for a young person (age 8-18).

Current read:
- Identity: {primary} ({secondary})
- Superpower: {superpower}
- Read: {description}
- Nudge: {growth_nudge}

CRITICAL RULES FOR SPEED AND RICHNESS:
1. Be EXTREMELY concise. Keep your conversational text under 40 words total.
2. Voice: Sharp, warm, real. No therapist/teacher vibes. No cringe slang.
3. Observation: If they attach an image, explicitly mention a specific detail from it.
4. Goal: Focus on what energizes them. Do NOT jump to career advice yet.
5. Structure: 1 short reflection -> 1[VISUALIZE: ...] marker -> 1 short question.
6. [VISUALIZE: <prompt>] must describe a vivid neon pixel-art scene. Use exactly once.
7. Never use lists. Never say "That's amazing" or "You're on a journey".
""".strip()


def _trim_chat_history(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []

    for msg in chat_history[-MAX_HISTORY_MESSAGES:]:
        role = _normalize_role(msg.get("role", "user"))
        text = _safe_str(msg.get("text"))
        if not text:
            continue
        cleaned.append({"role": role, "text": text})

    return cleaned


def _build_contents(
    chat_history: List[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> List[types.Content]:
    contents: List[types.Content] =[]

    for msg in _trim_chat_history(chat_history):
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])],
            )
        )

    if image_bytes:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime)

        for content in reversed(contents):
            if content.role == "user":
                content.parts.append(image_part)
                break
        else:
            contents.append(types.Content(role="user", parts=[image_part]))

    if not contents:
        contents =[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=DEFAULT_GREETING)],
            )
        ]

    return contents


def _iter_chunk_parts(chunk: Any) -> Iterator[Any]:
    candidates = getattr(chunk, "candidates", None) or[]
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or[]:
            yield part


def _extract_inline_image_b64(part: Any) -> Optional[str]:
    inline_data = getattr(part, "inline_data", None)
    if not inline_data:
        return None

    data = getattr(inline_data, "data", None)
    if not data:
        return None

    try:
        return compress_generated_image(data, size=320)
    except Exception:
        logger.exception("Failed to compress inline image chunk")
        return None

def generate_socratic_stream(
    superpowers: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
):
    """
    Streams interleaved text and optional generated images from the coaching agent.
    """
    system_instruction = _build_system_instruction(superpowers)
    contents = _build_contents(
        chat_history=chat_history,
        image_bytes=image_bytes,
        image_mime=image_mime,
    )

    # OPTIMIZATION 4: max_output_tokens forces the LLM to stop early, guaranteeing speed.
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=150, 
    )

    try:
        client = get_client()
        stream = client.models.generate_content_stream(
            model=DEFAULT_MODEL,
            contents=contents,
            config=config,
        )

        for chunk in stream:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                yield {"type": "text", "data": chunk_text}

            for part in _iter_chunk_parts(chunk):
                image_b64 = _extract_inline_image_b64(part)
                if image_b64:
                    yield {"type": "image", "data": image_b64}

    except Exception:
        logger.exception("Agent stream failed")
        yield {
            "type": "text",
            "data": "I glitched for a second. Say that again?",
        }