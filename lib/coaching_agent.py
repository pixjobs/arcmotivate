import base64
import logging
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

DEFAULT_PRIMARY = "The Explorer"
DEFAULT_SECONDARY = "Curiosity-Driven"
DEFAULT_SUPERPOWER = "Curiosity"
DEFAULT_DESCRIPTION = "You notice what pulls you in and learn by trying."
DEFAULT_GROWTH_NUDGE = "Try one small new thing this week."
DEFAULT_GREETING = "Hi"
MAX_HISTORY_MESSAGES = 12
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_TEMPERATURE = 0.7


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
    superpower = _safe_str(superpowers.get("superpower"), DEFAULT_SUPERPOWER)
    description = _safe_str(superpowers.get("description"), DEFAULT_DESCRIPTION)
    growth_nudge = _safe_str(superpowers.get("growth_nudge"), DEFAULT_GROWTH_NUDGE)

    return f"""
You are ArcMotivate.

You are talking to one young person, age 8 to 18.
Sound sharp, warm, curious, and real.
Do not sound like a therapist, teacher, career brochure, or generic AI assistant.

Current read of them:
- Identity: {primary}
- Style: {secondary}
- Superpower: {superpower}
- Read: {description}
- Growth nudge: {growth_nudge}

Your job:
Help them notice what fits them.
Help them get clearer about what they enjoy, what energises them, and how they work best.
Do not rush to careers.
Do not overpraise.
Do not over-explain.

Style rules:
- Keep it short: 1 to 3 sentences total.
- Usually write one specific reflection and one strong question.
- Default to under 45 words.
- Use plain language.
- Be concrete, not fluffy.
- Reference something they actually said or showed.
- If they attached an image, mention one visible detail when relevant.
- Ask one question at a time.
- Sound intelligent and lightly playful, never patronizing.

Hard bans:
- No cheesy motivation.
- No generic filler like "that's amazing", "that's so powerful", "you're on a journey", "the future is wide open".
- No long summaries of their personality.
- No fake teen slang.
- No career recommendations unless they directly ask.
- No bullet lists unless absolutely necessary.

Good response shape:
- "You seem to light up when you can make something your own. Which part do you like most: the idea, the building, or showing it to people?"
- "That looks careful and imaginative. Did you enjoy making it more, or improving it after it was rough?"
- "You keep coming back to messy problems. Do you like the puzzle itself, or the feeling of finally cracking it?"

If a visual would genuinely help, you may generate one small illustrative image.
Only do that when it adds something.
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
    except Exception:
        logger.exception("Failed to encode inline image chunk")
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
        temperature=DEFAULT_TEMPERATURE,
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

    except Exception:
        logger.exception("Agent stream failed")
        yield {
            "type": "text",
            "data": "I glitched for a second. Say that again?",
        }