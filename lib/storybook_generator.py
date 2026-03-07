import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional
from google import genai

logger = logging.getLogger(__name__)

TEXT_MODEL = "gemini-3.1-flash-lite-preview"
STRUCTURED_MODEL = "gemini-3-flash-preview"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing. "
                         "Check your Cloud Run 'Variables & Secrets' configuration.")
    return genai.Client(api_key=api_key)

# ============================================================
# SMALL HELPERS
# ============================================================

def _safe_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback

def _extract_text_from_msg(msg: Dict[str, Any]) -> str:
    text = msg.get("text")
    if text and isinstance(text, str):
        return text
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""

def _collect_interests(user_profile: Dict[str, Any], limit: int = 6) -> List[str]:
    superpowers = user_profile.get("superpowers", {}) or user_profile
    interests = superpowers.get("interests")
    if not isinstance(interests, list):
        return []
    out: List[str] = []
    for i, item in enumerate(interests):
        if i >= limit:
            break
        if item is not None:
            out.append(str(item))
    return out

def _profile_summary(user_profile: Dict[str, Any]) -> Dict[str, str]:
    superpowers = user_profile.get("superpowers", {}) or user_profile
    primary = superpowers.get("primary") or superpowers.get("archetype") or "Explorer"
    secondary = superpowers.get("secondary") or ""
    superpower = superpowers.get("superpower") or ""
    description = superpowers.get("description") or ""
    growth_nudge = superpowers.get("growth_nudge") or ""
    interests = ", ".join(_collect_interests(user_profile))
    return {
        "primary": str(primary),
        "secondary": str(secondary),
        "superpower": str(superpower),
        "description": str(description),
        "growth_nudge": str(growth_nudge),
        "interests": interests,
    }

def _recent_history_text(recent_chat: Optional[List[Dict[str, Any]]], limit: int = 6) -> str:
    history_lines: List[str] = []
    for msg in recent_chat or []:
        role = str(msg.get("role", "user"))
        text = _safe_str(_extract_text_from_msg(msg))
        if text:
            history_lines.append(f"{role}: {text}")
    return "\n".join(history_lines[-limit:])

# ============================================================
# VISUAL STYLE SYSTEM
# ============================================================

def _visual_style_guide() -> str:
    return (
        "Style: polished pixel-art / illustrated synthwave hybrid, grounded futuristic mood, "
        "clean silhouettes, expressive lighting, subtle neon accents, emotionally readable scenes, "
        "no text, no logos, no watermark."
    )

def _story_context_block(user_profile: Dict[str, Any], recent_chat: Optional[List[Dict[str, Any]]] = None) -> str:
    profile = _profile_summary(user_profile)
    history_text = _recent_history_text(recent_chat)

    return f"""
Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Growth nudge: {profile['growth_nudge']}

Recent conversation:
{history_text if history_text else "(no recent chat)"}
""".strip()

# ============================================================
# HERO RECAP
# ============================================================

def generate_hero_recap(user_profile: Dict[str, Any]) -> str:
    client = get_client()
    profile = _profile_summary(user_profile)

    prompt = f"""
You are ArcMotivate.

Write a short storybook recap for a young person aged 8 to 18.

{_story_context_block(user_profile)}

Requirements:
- 2 to 3 sentences
- simple, warm, modern language
- grounded, not fantasy
- reflect what seems to energise them
- name a real pattern emerging in how they think, work, or explore
- end with one small forward motion, not a grand conclusion
- do not use career brochure language
- do not sound like a therapist
""".strip()

    fallback = (
        "You’re starting to notice what gives you energy and what kind of challenges pull you in. "
        "That matters more than having everything figured out. Keep following the signals that feel real."
    )

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config={"temperature": 0.8},
        )
        text = (response.text or "").strip()
        return text or fallback
    except Exception as exc:
        logger.error("Hero recap generation failed: %s", exc)
        return fallback

# ============================================================
# PIXEL ART / STORY IMAGES
# ============================================================

def generate_pixel_art_illustration(scene_description: str, aspect_ratio: str = "16:9") -> str:
    """
    Generates a story-supporting visual.
    Returns base64-encoded image bytes as a UTF-8 string; empty string on failure.
    """
    client = get_client()

    full_prompt = f"""
{scene_description}

{_visual_style_guide()}
Prioritise visual clarity, emotional readability, and scene composition over visual spectacle.
""".strip()

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=full_prompt,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio},
            },
        )

        for part in (getattr(response, "parts", None) or []):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return base64.b64encode(inline_data.data).decode("utf-8")

    except Exception as exc:
        logger.error("Illustration generation failed: %s", exc)

    return ""


# ============================================================
# CUSTOM AVATAR
# ============================================================

def generate_custom_avatar(
    user_profile: Dict[str, Any],
    latest_signal: str = "",
    aspect_ratio: str = "1:1",
) -> str:
    client = get_client()
    profile = _profile_summary(user_profile)

    prompt = f"""
Create a custom avatar portrait for a young person's exploration profile.

Character signals:
- Core archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Latest conversational signal: {latest_signal}

Requirements:
- portrait only
- expressive, thoughtful, optimistic
- grounded futuristic styling
- readable face and silhouette
- subtle hints of their interests through props, clothing details, or background elements
- no costumes, no fantasy armor, no magic effects
- no text, no logos, no watermark
- suitable as the main identity portrait in a storybook

{_visual_style_guide()}
""".strip()

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio},
            },
        )

        for part in (getattr(response, "parts", None) or []):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return base64.b64encode(inline_data.data).decode("utf-8")

    except Exception as exc:
        logger.error("Avatar generation failed: %s", exc)

    return ""


# ============================================================
# 3-PANEL IDENTITY COMIC
# ============================================================

def generate_identity_comic(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """
    Generate a 3-panel identity comic.

    Returns:
        [{"caption": str, "image_b64": str}, ...]
    """
    client = get_client()

    prompt = f"""
You are creating a 3-panel visual story for a young person aged 8 to 18.

{_story_context_block(user_profile, recent_chat)}

Goal:
Create a coherent micro-story that reflects the user's actual signals, not generic inspiration.

Panel arc:
1. Spark — a specific moment of curiosity, interest, frustration, or pull
2. Experiment — them trying, building, testing, making, exploring, or learning
3. Direction — a grounded next scene showing where this could lead if they keep going

For each panel provide:
- caption: one short line, max 10 words
- image_prompt: vivid, concrete visual scene description

Constraints:
- exactly 3 panels
- grounded in real-world settings and activities
- no fantasy, magic, destiny, or superhero language
- no generic “future success” imagery
- visual continuity across the three panels
- each panel should feel like the same young person in the same world
- reference actual interests or behaviors from the profile or chat when possible
- image prompts should support the story, not just look cool
""".strip()

    schema = {
        "type": "object",
        "properties": {
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "image_prompt": {"type": "string"},
                    },
                    "required": ["caption", "image_prompt"],
                },
            }
        },
        "required": ["panels"],
    }

    try:
        response = client.models.generate_content(
            model=STRUCTURED_MODEL,
            contents=prompt,
            config={
                "temperature": 0.65,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        data = json.loads((response.text or "").strip())
        panels_raw = (data.get("panels") or [])[:3]
    except Exception as exc:
        logger.error("Comic panel spec generation failed: %s", exc)
        panels_raw = [
            {
                "caption": "Something caught your attention.",
                "image_prompt": "A young person leaning over a desk, focused on a project or idea that suddenly clicks, evening light, grounded futuristic room, subtle neon accents",
            },
            {
                "caption": "You started testing it.",
                "image_prompt": "The same young person trying things hands-on, sketching, building, or experimenting at a desk, visible progress, practical tools, clear action",
            },
            {
                "caption": "Now the path feels clearer.",
                "image_prompt": "The same young person looking at a wall or desk full of ideas, prototypes, notes, or interests starting to connect, calm confidence, forward direction",
            },
        ]

    result: List[Dict[str, str]] = []
    for panel in panels_raw:
        caption = _safe_str(panel.get("caption"), "…")
        image_prompt = _safe_str(panel.get("image_prompt"), "")
        image_b64 = ""
        if image_prompt:
            try:
                image_b64 = generate_pixel_art_illustration(image_prompt, aspect_ratio="4:3")
            except Exception:
                logger.exception("Comic panel image generation failed")
        result.append({"caption": caption, "image_b64": image_b64})

    return result


# ============================================================
# FUTURE POSTCARD
# ============================================================

def generate_future_postcard(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """
    Generate a 'Postcard from Future You' artifact.

    Returns:
        {"image_b64": str, "caption": str}
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    caption_prompt = f"""
You are ArcMotivate.

Write a one-sentence postcard line from a young person's future self.

{_story_context_block(user_profile, recent_chat)}

Rules:
- one sentence only
- max 18 words
- warm, calm, believable
- grounded in exploration, not certainty
- no career labels
- no hype
- no fantasy
- start with "You're" or "You"
""".strip()

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=caption_prompt,
            config={"temperature": 0.75},
        )
        caption = (response.text or "").strip().strip('"')
    except Exception:
        logger.exception("Postcard caption failed")
        caption = "You're still exploring, but you trust your own signals more now."

    image_prompt = f"""
A cinematic storybook postcard scene showing the same young person a little further along in their journey.
Grounded future setting, calm confidence, open horizon, visual hints of their archetype {profile['primary']},
their interests ({profile['interests']}), and their superpower ({profile['superpower']}).
The image should feel like continuation, not fantasy victory imagery.

{_visual_style_guide()}
""".strip()

    image_b64 = ""
    try:
        image_b64 = generate_pixel_art_illustration(image_prompt, aspect_ratio="16:9")
    except Exception:
        logger.exception("Postcard image failed")

    return {"image_b64": image_b64, "caption": caption}