import base64
import logging
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

DEFAULT_PRIMARY = "The Explorer"
DEFAULT_SECONDARY = "Curiosity-Driven"
DEFAULT_GREETING = "Hi"
MAX_HISTORY_MESSAGES = 12
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def get_client() -> genai.Client:
    return genai.Client()


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
    description = _safe_str(superpowers.get("description"), "Still emerging.")

    return f"""
You are ArcMotivate, a warm, engaging mentor for young people ages 8 to 18.

Your job is to help them explore who they are, what they enjoy, and what energizes them.
Do not push them toward specific careers too early.

Current user profile:
- Identity: {primary} ({secondary})
- What drives them: {description}

Rules:
1. Explore, do not prescribe. Ask open questions that help them discover interests.
2. Reference details they shared, including any attached image.
3. Stay realistic. If you mention real fields, industries, or roles, keep them grounded and contemporary.
4. Keep the pacing gentle: one strong question at a time.
5. If a visual would genuinely help, you may generate a small illustrative image, concept sketch, or visual analogy.
6. Keep text concise: at most two short paragraphs plus one question.
7. Sound encouraging, curious, and intelligent — never patronizing.
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
    contents: List[types.Content] = []

    for msg in _trim_chat_history(chat_history):
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])],
            )
        )

    if image_bytes:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime)

        # Attach image to most recent user turn
        for content in reversed(contents):
            if content.role == "user":
                content.parts.append(image_part)
                break
        else:
            contents.append(types.Content(role="user", parts=[image_part]))

    if not contents:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=DEFAULT_GREETING)],
            )
        ]

    return contents


def _iter_chunk_parts(chunk: Any) -> Iterator[Any]:
    """
    Safely extract multimodal parts from a streaming chunk.
    """
    candidates = getattr(chunk, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            yield part


def _extract_inline_image_b64(part: Any) -> Optional[str]:
    inline_data = getattr(part, "inline_data", None)
    if not inline_data:
        return None

    data = getattr(inline_data, "data", None)
    if not data:
        return None

    try:
        return base64.b64encode(data).decode("utf-8")
    except Exception as exc:
        logger.warning("Failed to encode inline image chunk: %s", exc)
        return None


def generate_socratic_stream(
    superpowers: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
):
    """
    Streams interleaved text and optional generated images from the coaching agent.

    Yields:
        {"type": "text", "data": "..."}
        {"type": "image", "data": "<base64>"}
    """
    system_instruction = _build_system_instruction(superpowers)
    contents = _build_contents(
        chat_history=chat_history,
        image_bytes=image_bytes,
        image_mime=image_mime,
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.9,
    )

    try:
        with get_client() as client:
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

    except Exception as exc:
        logger.exception("Agent stream failed")
        yield {
            "type": "text",
            "data": "Simulation paused. What was your last move?",
        }