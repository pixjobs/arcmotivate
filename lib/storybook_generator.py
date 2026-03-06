import base64
import logging
from typing import Any, Dict
from google import genai

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
    interests = superpowers.get("interests")
    if isinstance(interests, list):
        # Avoid slicing if it's causing type-checker issues, use a limited loop instead
        valid_items = []
        for i, val in enumerate(interests):
            if i >= 6:
                break
            valid_items.append(str(val))
        interests_str = ", ".join(valid_items)
    else:
        interests_str = ""

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
            # Pass dictionary instead of types.GenerateContentConfig to avoid IDE type errors
            config={"temperature": 1.0},
        )
        return (response.text or "").strip() or "Next level: try one small experiment this week."
    except Exception as e:
        logger.error(f"Hero's journey gen failed: {e}")
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

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=full_prompt,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {
                    "aspect_ratio": "16:9",
                }
            },
        )
        
        # Safely iterate over parts
        for part in (getattr(response, "parts", None) or []):
            if getattr(part, "inline_data", None) and part.inline_data.data:
                return base64.b64encode(part.inline_data.data).decode("utf-8")

    except Exception as e:
        logger.error(f"Image gen failed: {e}")

    return ""


def generate_comic_book(chat_history: list[Dict[str, Any]]) -> list[Dict[str, str]]:
    """
    Analyzes the chat history to synthesize a 3-panel comic book.
    Returns a list of dicts: {"caption": "...", "image_b64": "..."}.
    """
    client = get_client()

    # Summarize chat history into narrative beats
    history_text = "\n".join([f"{msg['role']}: {msg.get('text', msg.get('content', ''))}" for msg in chat_history if msg['role'] != 'system'])
    
    prompt = f"""
    Based on the following career simulator conversation, create a 3-panel comic book narrative that highlights the user's journey.
    Output EXACTLY 3 JSON objects in a JSON array. Each object MUST have:
    - "caption": A short, punchy sentence narrating the scene.
    - "image_prompt": A highly descriptive, visual prompt for generating a pixel-art illustration of that exact scene.

    Conversation History:
    {history_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={
                "temperature": 1.0,
                "response_mime_type": "application/json"
            },
        )
        import json
        import re
        
        raw_text = (response.text or "").strip()
        # Clean up markdown code blocks if the model outputs them anyway
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n", "", raw_text)
            raw_text = re.sub(r"\n```$", "", raw_text).strip()
            
        panels = json.loads(raw_text)
        
        comic_data: list[Dict[str, str]] = []
        for panel in panels:
            img_prompt = panel.get("image_prompt", "")
            img_b64 = generate_pixel_art_illustration(img_prompt) if img_prompt else ""
            comic_data.append({
                "caption": panel.get("caption", ""),
                "image_b64": img_b64
            })
            
        return comic_data
    except Exception as e:
        logger.error(f"Comic gen failed: {e}")
        return []