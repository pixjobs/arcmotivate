"""
Storybook Generator
Handles the generation of personalized, visual-first story artifacts:
Hero Recaps, Custom Avatars, Identity Comics, and Future Postcards.
Enforces Split-Language Architecture for multilingual support.
"""

import base64
import concurrent.futures
import json
import logging
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from lib.image_utils import compress_generated_image

logger = logging.getLogger(__name__)

# Use the fast models consistently
TEXT_MODEL = "gemini-3.1-flash-lite-preview"
STRUCTURED_MODEL = "gemini-3.1-flash-lite-preview"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

# Global client cache for fast background thread execution
_CLIENT = None

def get_client() -> genai.Client:
    """Lazily initializes and returns the Google GenAI client."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing. "
                             "Check your Cloud Run 'Variables & Secrets' configuration.")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT

# ============================================================
# SMALL HELPERS
# ============================================================

def _safe_str(value: Any, fallback: str = "") -> str:
    """Safely converts any value to a string, returning a fallback if empty."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback

def _extract_text_from_msg(msg: Dict[str, Any]) -> str:
    """Extracts plain text from a multimodal message dictionary."""
    text = msg.get("text")
    if text and isinstance(text, str):
        return text
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts =[]
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""

def _collect_interests(user_profile: Dict[str, Any], limit: int = 6) -> List[str]:
    """Extracts a limited list of user interests from their profile."""
    superpowers = user_profile.get("superpowers", {}) or user_profile
    interests = superpowers.get("interests")
    if not isinstance(interests, list):
        return[]
    out: List[str] =[]
    for i, item in enumerate(interests):
        if i >= limit:
            break
        if item is not None:
            out.append(str(item))
    return out

def _profile_summary(user_profile: Dict[str, Any]) -> Dict[str, str]:
    """Flattens the user profile into a safe dictionary of strings."""
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
    """Formats the recent chat history into a readable string block."""
    history_lines: List[str] =[]
    for msg in recent_chat or[]:
        role = str(msg.get("role", "user"))
        text = _safe_str(_extract_text_from_msg(msg))
        if text:
            history_lines.append(f"{role}: {text}")
    return "\n".join(history_lines[-limit:])

# ============================================================
# VISUAL STYLE SYSTEM
# ============================================================

def _visual_style_guide() -> str:
    """Returns the global visual style prompt for image generation."""
    return (
        "Style: polished pixel-art / illustrated synthwave hybrid, grounded futuristic mood, "
        "clean silhouettes, expressive lighting, subtle neon accents, emotionally readable scenes, "
        "no text, no logos, no watermark."
    )

def _story_context_block(user_profile: Dict[str, Any], recent_chat: Optional[List[Dict[str, Any]]] = None) -> str:
    """Builds the context block containing the user's profile and recent chat."""
    profile = _profile_summary(user_profile)
    history_text = _recent_history_text(recent_chat)

    return f"""
Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}

Recent conversation (CRITICAL - Anchor the story to these specific details):
{history_text if history_text else "(no recent chat)"}
""".strip()

# ============================================================
# HERO RECAP
# ============================================================

def generate_hero_recap(user_profile: Dict[str, Any]) -> str:
    """Generates a short, punchy storybook recap based on the user's profile."""
    client = get_client()

    prompt = f"""
You are ArcMotivate. Write a short, punchy storybook recap for a young person.

{_story_context_block(user_profile)}

Requirements:
- Maximum 2 sentences.
- Be highly specific to their actual interests. Do not use generic templates.
- Name a real pattern emerging in how they think or work.
- Do not use career brochure language or sound like a therapist.

CRITICAL LANGUAGE RULE: 
Analyze the recent conversation. Identify the primary language being used. 
If the user's most recent message is very short (e.g., 'yes', 'no', 'not yet', a name), ignore it for language detection and match the language of the broader conversation. 
You MUST write the recap entirely in that specific language.
""".strip()

    fallback = "You’re starting to notice what gives you energy. Keep following the signals that feel real."

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        text = (response.text or "").strip()
        return text or fallback
    except Exception as e:
        logger.error("Hero recap generation failed: %s", e)
        return fallback

# ============================================================
# PIXEL ART / STORY IMAGES
# ============================================================

def generate_pixel_art_illustration(scene_description: str, aspect_ratio: str = "16:9", compress_size: int = 512) -> str:
    """
    Generates a story-supporting visual and compresses it to prevent UI bloat.
    Returns base64-encoded image bytes as a UTF-8 string; empty string on failure.
    """
    client = get_client()

    full_prompt = f"""
{scene_description}

{_visual_style_guide()}
Prioritise visual clarity, emotional readability, and scene composition over visual spectacle.
""".strip()

    try:
        # Note: Using dict for config here as image_config is specific to image models
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=full_prompt,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio},
            },
        )

        for part in (getattr(response, "parts", None) or[]):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                try:
                    compressed = compress_generated_image(inline_data.data, size=compress_size)
                    if compressed:
                        return compressed
                except Exception as e:
                    logger.warning("Image compression failed, falling back to raw base64: %s", e)
                
                return base64.b64encode(inline_data.data).decode("utf-8")

    except Exception as e:
        logger.error("Illustration generation failed: %s", e)

    return ""

# ============================================================
# CUSTOM AVATAR
# ============================================================

def generate_custom_avatar(
    user_profile: Dict[str, Any],
    latest_signal: str = "",
    aspect_ratio: str = "1:1",
) -> str:
    """Generates a custom avatar portrait based on the user's archetype and interests."""
    profile = _profile_summary(user_profile)
    
    # Defensive Prompting: Strip the[SYSTEM DIRECTIVE] out of the signal 
    # so it doesn't confuse the English-only image generation model.
    clean_signal = latest_signal.split("[SYSTEM DIRECTIVE")[0].strip()

    prompt = f"""
Create a custom avatar portrait for a young person's exploration profile.

Character signals:
- Core archetype: {profile['primary']}
- Superpower: {profile['superpower']}
- Interests: {profile['interests']}
- Latest conversational signal: {clean_signal}

Requirements:
- Portrait only.
- Expressive, thoughtful, cool.
- Grounded futuristic styling.
- MUST include subtle visual hints of their specific interests (e.g., if they like music, maybe they have headphones; if they like coding, maybe a glowing visor).
- No costumes, no fantasy armor, no magic effects.
""".strip()

    # Compress avatar to 320px since it's displayed small in the UI
    return generate_pixel_art_illustration(prompt, aspect_ratio=aspect_ratio, compress_size=320)

# ============================================================
# 3-PANEL IDENTITY COMIC
# ============================================================

def _generate_single_panel(panel: Dict[str, str]) -> Dict[str, str]:
    """Helper function to generate a single comic panel image in parallel."""
    caption = _safe_str(panel.get("caption"), "…")
    image_prompt = _safe_str(panel.get("image_prompt"), "")
    image_b64 = ""
    
    if image_prompt:
        try:
            # Compress comic panels to 320px to keep the grid lightweight
            image_b64 = generate_pixel_art_illustration(image_prompt, aspect_ratio="4:3", compress_size=320)
        except Exception as e:
            logger.exception("Comic panel image generation failed: %s", e)
            
    return {"caption": caption, "image_b64": image_b64}

def generate_identity_comic(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Generates a 3-panel identity comic using parallel processing for speed."""
    client = get_client()

    prompt = f"""
You are creating a 3-panel visual story for a young person aged 8 to 18.

{_story_context_block(user_profile, recent_chat)}

Goal:
Create a coherent micro-story that reflects the user's ACTUAL signals from the chat history. 
Do NOT invent generic scenarios. If they talked about a specific hobby, project, or frustration, the comic MUST be about that exact thing.

Panel arc:
1. Spark — them engaging with their specific interest.
2. Experiment — them trying, building, or testing that specific thing.
3. Direction — a grounded next scene showing where this specific interest could lead.

For each panel provide:
- caption: one short line, max 8 words.
- image_prompt: vivid, concrete visual scene description.

Constraints:
- exactly 3 panels.
- grounded in real-world settings.
- visual continuity across the three panels.

CRITICAL LANGUAGE RULES:
1. Analyze the recent conversation. Identify the primary language being used. If the user's most recent message is very short (e.g., 'yes', 'no', 'not yet', a name), ignore it for language detection and match the language of the broader conversation. You MUST write the "caption" entirely in that specific language.
2. HOWEVER, the "image_prompt" field MUST ALWAYS be written in English. Image generation models do not understand other languages.
""".strip()

    schema = {
        "type": "OBJECT",
        "properties": {
            "panels": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "caption": {"type": "STRING"},
                        "image_prompt": {"type": "STRING"},
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
            config=types.GenerateContentConfig(
                temperature=0.65,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        data = json.loads((response.text or "").strip())
        panels_raw = (data.get("panels") or [])[:3]
    except Exception as e:
        logger.error("Comic panel spec generation failed: %s", e)
        panels_raw =[
            {"caption": "Something caught your attention.", "image_prompt": "A young person leaning over a desk, focused on a project, evening light, grounded futuristic room"},
            {"caption": "You started testing it.", "image_prompt": "The same young person trying things hands-on, sketching or experimenting at a desk"},
            {"caption": "Now the path feels clearer.", "image_prompt": "The same young person looking at a wall full of ideas starting to connect, calm confidence"},
        ]

    # Parallel Processing: Generate all 3 images at the exact same time
    result: List[Dict[str, str]] =[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Map maintains the order of the panels
        result = list(executor.map(_generate_single_panel, panels_raw))

    return result

# ============================================================
# FUTURE POSTCARD
# ============================================================

def generate_future_postcard(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Generates a 'Postcard from Future You' artifact with a translated caption."""
    client = get_client()
    profile = _profile_summary(user_profile)

    caption_prompt = f"""
You are ArcMotivate. Write a one-sentence postcard line from a young person's future self.

{_story_context_block(user_profile, recent_chat)}

Rules:
- one sentence only, max 15 words.
- MUST reference a specific detail from their recent chat or interests.
- warm, calm, believable.
- start with "You're" or "You".

CRITICAL LANGUAGE RULE: 
Analyze the recent conversation. Identify the primary language being used. 
If the user's most recent message is very short (e.g., 'yes', 'no', 'not yet', a name), ignore it for language detection and match the language of the broader conversation. 
You MUST write the postcard line entirely in that specific language.
""".strip()

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=caption_prompt,
            config=types.GenerateContentConfig(temperature=0.75),
        )
        caption = (response.text or "").strip().strip('"')
    except Exception as e:
        logger.exception("Postcard caption failed: %s", e)
        caption = "You're still exploring, but you trust your own signals more now."

    image_prompt = f"""
A cinematic storybook postcard scene showing the same young person a little further along in their journey.
Grounded future setting, calm confidence, open horizon.
CRITICAL: Include visual hints of their specific interests: ({profile['interests']}).
The image should feel like continuation, not fantasy victory imagery.
""".strip()

    # Compress postcard to 512px
    image_b64 = generate_pixel_art_illustration(image_prompt, aspect_ratio="16:9", compress_size=512)

    return {"image_b64": image_b64, "caption": caption}