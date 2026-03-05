import base64
import logging
from typing import Any, Dict
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client():
    return genai.Client()

def generate_heros_journey_text(user_profile: Dict[str, Any]) -> str:
    """
    Generates a short, uplifting 2–3 sentence recap for ages 8–18.
    Keeps it grounded (no magic/RPG), but still epic + arcade vibes.
    """
    client = get_client()

    superpowers = user_profile.get("superpowers", {}) or user_profile
    primary = superpowers.get("primary") or superpowers.get("archetype") or "Explorer"

    # Pull optional signals safely
    interests = superpowers.get("interests", [])
    interests_str = ", ".join(interests[:6]) if isinstance(interests, list) else ""

    prompt = f"""
You are ArcMotivate. Write a 2–3 sentence “arcade ending screen” message for a young person (age 8–18).
Tone: uplifting, confident, not cringe. NO magic, NO RPG classes, NO fantasy tropes.
Make it about growth and exploration, not picking one forever-career.

User vibe/archetype (tentative): {primary}
Interests/themes spotted (optional): {interests_str}

Requirements:
- 2–3 sentences total.
- Use simple language.
- End with a forward-looking line like “Next level: try one small experiment this week.”
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        return (response.text or "").strip() or "Next level: try one small experiment this week."
    except Exception:
        return "You’re building your future one small discovery at a time. Next level: try one small experiment this week."


def generate_pixel_art_illustration(scene_description: str) -> str:
    """
    Generates a neon pixel-art “poster” image.
    Returns base64-encoded PNG/JPG bytes as a UTF-8 string; empty string on failure.
    """
    client = get_client()

    style = (
        "Retro neon pixel-art, 16-bit arcade poster, crisp pixel edges, glowing neon highlights, "
        "dark synthwave background, high contrast, readable silhouettes, no text."
    )
    full_prompt = f"{scene_description}. Style: {style}"

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=full_prompt,
            config=config,
        )

        # Be robust about where the inline_data appears
        if response.candidates:
            for cand in response.candidates:
                if not cand.content or not cand.content.parts:
                    continue
                for part in cand.content.parts:
                    if part.inline_data and part.inline_data.data:
                        return base64.b64encode(part.inline_data.data).decode("utf-8")

    except Exception as e:
        logger.error(f"Image gen failed: {e}")

    return ""