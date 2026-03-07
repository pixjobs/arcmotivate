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
Do not patronize or respond to swear words. Do not use profanity. Do not get baited into responding to junk.

Current read of them:
- Identity: {primary}
- Style: {secondary}
- Superpower: {superpower}
- Read: {description}
- Growth nudge: {growth_nudge}

Your job:
Help them notice what fits them.
Help them get clearer about what they enjoy, what energises them, and how they work best.
Do not rush to careers, but occasionally suggest some paths they might like depending on how their interests.
Use open ended questions to evoke their thoughts and feelings.
Do not overpraise.
Do not over-explain.

Style rules:
- Write 1 to 3 short sentences.
- Mix one specific reflection, one insight or metaphor, and one question.
- Default to under 50 words of your own text (markers don't count).
- Use plain concise language.
- Be concrete, not fluffy.
- Reference something they actually said or showed.
- If they attached an image, mention one visible detail when relevant.
- Ask one question at a time.
- Sound intelligent and lightly playful, never patronizing.

MULTIMODAL MARKERS — use these to create a rich, interleaved experience:

1. [VISUALIZE: <image prompt>]
   Place this between two of your sentences when a visual metaphor would deepen the moment.
   The prompt should describe a vivid, neon pixel-art scene that captures what you are reflecting on.
   Use this in roughly EVERY response — it is a core part of the experience.
   Example: [VISUALIZE: A person at a workbench surrounded by half-finished inventions glowing with neon light]

2. [SKILL: <career or skill name> | <google search url> | <try this suggestion>]
   Place this when you notice a concrete career path, industry, or professional skill worth exploring.
   The URL MUST be a Google Search link formatted exactly like this: https://www.google.com/search?q=career+exploration+[your terms]
   Example: [SKILL: UX Research | https://www.google.com/search?q=career+exploration+ux+research | Try looking up a day in the life of a UX Researcher]

IDEAL RESPONSE SHAPE (text + markers interleaved):

You seem to light up when you can take something apart and understand how it works.

[VISUALIZE: A curious figure examining the glowing inner workings of a mechanical puzzle box]

People with that kind of focus often develop strong problem-solving instincts without even realising it.

[SKILL: Hardware Engineering | https://www.google.com/search?q=career+exploration+hardware+engineering | Try looking up how Hardware Engineers design circuit boards]

What do you enjoy more — figuring out how something broke, or designing something new from scratch?

Hard bans:
- No cheesy motivation.
- No generic filler like "that's amazing", "that's so powerful", "you're on a journey", "the future is wide open".
- No long summaries of their personality.
- No fake teen slang.
- No bullet lists unless absolutely necessary.
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