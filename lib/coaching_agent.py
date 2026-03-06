import base64
from typing import Dict, Any, List, Optional, Iterator

from google import genai
from google.genai import types


def get_client() -> genai.Client:
    return genai.Client()


def _normalize_role(role: str) -> str:
    role = (role or "user").strip().lower()
    if role == "assistant":
        return "model"
    if role not in {"user", "model"}:
        return "user"
    return role


def _build_contents(
    chat_history: List[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> List[types.Content]:
    contents: List[types.Content] = []

    for msg in chat_history:
        role = _normalize_role(msg.get("role", "user"))
        text = (msg.get("text") or "").strip()

        parts: List[types.Part] = []
        if text:
            parts.append(types.Part.from_text(text=text))

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    if image_bytes:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime)

        # Attach image to the most recent user turn
        for content in reversed(contents):
            if content.role == "user":
                content.parts.append(image_part)
                break
        else:
            # No existing user message, so create one
            contents.append(
                types.Content(
                    role="user",
                    parts=[image_part],
                )
            )

    return contents


def _iter_chunk_parts(chunk) -> Iterator[Any]:
    """
    Safely extract parts from a streaming chunk.
    Streaming responses are GenerateContentResponse objects, so parts are
    usually nested under candidates -> content -> parts.
    """
    candidates = getattr(chunk, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            yield part


def generate_socratic_stream(
    superpowers: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
):
    """
    Stream interleaved text and optional generated images from the coaching agent.
    Optionally attaches a user-provided image to the latest user turn.
    Yields:
        {"type": "text", "data": "..."}
        {"type": "image", "data": "<base64>"}
    """
    primary = superpowers.get("primary", "The Explorer")
    secondary = superpowers.get("secondary", "Curiosity-Driven")
    description = superpowers.get("description", "")

    system_instruction = f"""
You are ArcMotivate, a warm and deeply engaging mentor for young people (ages 8–18).
Your goal is to help them explore who they are and what excites them — not to push them toward specific careers.

Who they are right now:
- Identity: {primary} ({secondary})
- What drives them: {description}

CRITICAL RULES:
1) EXPLORE, DON'T PRESCRIBE: Ask questions that help them discover interests, NOT land on a career. Keep it open and curious. Do NOT say "so you'd be a great X".
2) REFERENCE THEM: If they shared an image or mentioned something specific, reference it. Make them feel heard.
3) REALISM: When you do mention real fields or industries, be specific and grounded. No made-up job titles.
4) PACING: ONE engaging question at a time. Let the conversation breathe.
5) VISUALS: When it would enrich the response, generate a small illustrative image — a concept, a scene, a visual analogy. Keep it tight and relevant.
6) Keep text punchy (max 2 short paragraphs + your question).
""".strip()

    contents = _build_contents(
        chat_history=chat_history,
        image_bytes=image_bytes,
        image_mime=image_mime,
    )

    if not contents:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hi")],
            )
        ]

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=1.0,
        # Only include this if you truly want image generation in responses.
        # If your model / endpoint doesn't support image output, remove it.
        # response_modalities=["TEXT", "IMAGE"],
    )

    try:
        with get_client() as client:
            stream = client.models.generate_content_stream(
                model="gemini-3.1-flash-lite-preview",
                contents=contents,
                config=config,
            )

            for chunk in stream:
                # Easiest and safest path for normal streamed text
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    yield {"type": "text", "data": chunk_text}

                # Handle multimodal parts if they exist in the chunk
                for part in _iter_chunk_parts(chunk):
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        b64_data = base64.b64encode(inline_data.data).decode("utf-8")
                        yield {"type": "image", "data": b64_data}

    except Exception as e:
        print(f"Agent Error: {type(e).__name__}: {e}")
        yield {
            "type": "text",
            "data": "Simulation paused. What was your last move?",
        }