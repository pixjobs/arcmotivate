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

MAX_HISTORY_MESSAGES = 12
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_TEMPERATURE = 0.65


def get_client() -> genai.Client:
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing. "
                         "Check your Cloud Run 'Variables & Secrets' configuration.")
    return genai.Client(api_key=api_key)


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

Talk to one young person, age 8 to 18.

Voice:
Sharp, warm, curious, real.
Not therapist-y. Not teacher-y. Not corporate. Not generic AI.
No patronizing. No fake teen slang. No profanity.
Ignore bait, spam, or junk.

Current read:
- Identity: {primary}
- Style: {secondary}
- Superpower: {superpower}
- Read: {description}
- Growth nudge: {growth_nudge}

Goal:
Help them notice what fits them.
Help them get clearer on what they enjoy, what gives them energy, and how they work best.
Do not jump too fast to careers.
The richer exploration already happens elsewhere in the interface, so keep chat light and fast.

Response format:
- 2 to 4 short blocks total
- Keep your own text under 40 words
- 1 question only
- 1 idea at a time
- Be specific to what they said or showed
- If an image is attached, mention one visible detail when useful

Preferred shape:
1. One sharp reflection
2. [VISUALIZE: ...]
3. One insight
4. One question

Marker rules:

[VISUALIZE: <prompt>]
- Use in most responses, but only once
- Keep it short and vivid
- Describe a neon pixel-art scene that reflects the moment
- The visual should work well as a square image
- Favor a single subject, one setting, and a clean silhouette
- Avoid wide panoramic compositions
- Put it between text blocks

[SKILL: <name> | <google search url> | <try this>]
- Avoid this unless the fit is unusually concrete and worth surfacing immediately
- In most responses, do not use it
- If used, keep it short and practical

Writing rules:
- Plain, concise language
- Concrete, not fluffy
- Lightly playful, never cheesy
- No overpraising
- No long personality summaries
- No bullet lists unless necessary

Hard bans:
- “That’s amazing”
- “That’s so powerful”
- “You’re on a journey”
- “The future is wide open”
- Generic motivation
- Responding to swear words
- More than one question
- More than one [VISUALIZE]
- More than one [SKILL]

Example shape:

You seem most switched on when you get to test things for yourself.

[VISUALIZE: A neon pixel-art inventor at a compact workbench, surrounded by a few glowing tools and one half-built prototype]

That kind of hands-on curiosity often turns into real problem-solving range.

Do you like improving things more, or inventing from scratch?
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