"""
Outcome Engine
Handles the generation of background workspace artifacts (Tiles) and the initial app intro.
Enforces strict JSON schemas and Split-Language Architecture for multilingual support.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

TILE_MODEL = "gemini-3-flash-preview"
INTRO_MODEL = "gemini-3-flash-preview"

DEFAULT_LINK = {
    "label": "Explore Careers",
    "url": "https://www.google.com/search?q=career+exploration",
}

# UPGRADE: New punchy, mobile-friendly, visual-first fallback intro
DEFAULT_INTRO = (
    "👾 **System Online — ArcMotivate**\n\n"
    "[VISUALIZE: A neon pixel-art control room waking up, glowing screens, pathways branching into different futures]\n\n"
    "I map your input to find the future your mind demands. What moment keeps replaying in your head?\n\n"
    "*Send a message or attach an image to begin.*"
)

# UPGRADE: Global client cache for fast background thread execution
_CLIENT = None

def get_client() -> genai.Client:
    """Lazily initializes and returns the Google GenAI client."""
    global _CLIENT
    if _CLIENT is None:
        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing. "
                             "Check your Cloud Run 'Variables & Secrets' configuration.")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _safe_str(value: Any, fallback: str = "") -> str:
    """Safely converts any value to a string, returning a fallback if empty."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalize_text(value: str) -> str:
    """Normalizes text for duplicate comparison (lowercase, alphanumeric only)."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _tile_memory(existing_tiles: Optional[List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Extracts historical titles, categories, and tags to prevent duplicates."""
    existing_tiles = existing_tiles or[]

    titles: List[str] = []
    categories: List[str] =[]
    skill_tags: List[str] =[]

    for tile in existing_tiles:
        title = _safe_str(tile.get("title"))
        category = _safe_str(tile.get("category"))

        if title:
            titles.append(title.lower())
        if category:
            categories.append(category.lower())

        for tag in tile.get("skill_tags") or[]:
            tag = _safe_str(tag)
            if tag:
                skill_tags.append(tag.lower())

    return {
        "titles": titles[-12:],
        "categories": categories[-12:],
        "skill_tags": skill_tags[-24:],
    }


def _is_duplicate_tile(candidate: Dict[str, Any], existing_tiles: Optional[List[Dict[str, Any]]]) -> bool:
    """Checks if a newly generated tile is too similar to an existing one."""
    existing_tiles = existing_tiles or[]

    cand_title = _normalize_text(_safe_str(candidate.get("title")))
    cand_category = _normalize_text(_safe_str(candidate.get("category")))
    cand_tags = {
        _normalize_text(_safe_str(tag))
        for tag in (candidate.get("skill_tags") or[])
        if _safe_str(tag)
    }

    if not cand_title:
        return True

    for tile in existing_tiles:
        prev_title = _normalize_text(_safe_str(tile.get("title")))
        prev_category = _normalize_text(_safe_str(tile.get("category")))
        prev_tags = {
            _normalize_text(_safe_str(tag))
            for tag in (tile.get("skill_tags") or[])
            if _safe_str(tag)
        }

        if cand_title == prev_title:
            return True

        if cand_category and cand_category == prev_category and cand_tags and prev_tags:
            overlap = len(cand_tags & prev_tags)
            if overlap >= min(2, len(cand_tags), len(prev_tags)):
                return True

    return False


def _validated_links(tile: Dict[str, Any]) -> List[Dict[str, str]]:
    """Ensures the generated links are safe and properly formatted."""
    validated: List[Dict[str, str]] = []

    for link in (tile.get("links") or[]):
        if not isinstance(link, dict):
            continue

        label = _safe_str(link.get("label"), "Explore")
        url = _safe_str(link.get("url"))

        if url.startswith("https://"):
            validated.append({"label": label, "url": url})

    return validated[:1] or [DEFAULT_LINK]


def _build_tile_prompt(
    history_text: str,
    primary: str,
    secondary: str,
    superpower: str,
    prior_titles: str,
    prior_categories: str,
    prior_skills: str,
    blocked_titles: List[str],
) -> str:
    """Builds the system prompt for generating a new workspace tile."""
    blocked = ", ".join(blocked_titles) if blocked_titles else "none"

    return f"""
You are ArcMotivate's background analysis engine.

Analyze the latest moments in this person's exploration and generate exactly ONE genuinely new canvas tile.

Current identity read:
- Primary: {primary}
- Secondary: {secondary}
- Superpower: {superpower}

Recent interaction:
{history_text}

Existing tile memory:
- Prior titles: {prior_titles}
- Prior categories: {prior_categories}
- Prior skill tags: {prior_skills}
- Blocked titles for this attempt: {blocked}

Novelty rules:
- Do NOT repeat or closely rephrase an existing title.
- Do NOT generate a tile that is substantially similar to a previous tile.
- If the same broad theme is still present, choose a more specific adjacent role, industry, or professional skill.
- Prefer progression over repetition.
- A new tile should feel like the next layer of insight, not the same insight renamed.

Generate exactly ONE JSON object with these keys:
1. "category": Pick ONE of "Career Explored", "Industry Unlocked", "Professional Skill", or "Real-World Quest".
2. "title": A concrete job title, career role, industry name, or professional skill area.
3. "content": One grounded sentence connecting what they shared to this tile.
4. "image_prompt": A descriptive neon pixel-art prompt illustrating this field in action.
5. "skill_tags": An array of exactly 2 concrete professional skills.
6. "skill_nudge": A short actionable suggestion to explore this area.
7. "links": An array with exactly 1 object. For the "label", use a clear name. For the "url", generate a Google search URL formatted like "https://www.google.com/search?q=" + URL-encoded search terms.

CRITICAL LANGUAGE RULES:
1. You MUST write all user-facing text (category, title, content, skill_tags, skill_nudge, link labels) in the EXACT SAME LANGUAGE that the user is speaking in the "Recent interaction" above.
2. HOWEVER, the "image_prompt" field MUST ALWAYS be written in English. Image generation models do not understand other languages.
3. If writing in English, you MUST use British English spelling (e.g., "colour", "realise") and UK cultural context (e.g., "football" means soccer, use "university" or "uni" not "college").

Hard rules:
- No fantasy
- No patronizing tone
- Ground everything in real-world careers, industries, or skills
- Output JSON only
""".strip()


def _tile_response_schema() -> Dict[str, Any]:
    """Defines the strict JSON schema for tile generation."""
    return {
        "type": "OBJECT",
        "properties": {
            "category": {"type": "STRING"},
            "title": {"type": "STRING"},
            "content": {"type": "STRING"},
            "image_prompt": {"type": "STRING"},
            "skill_tags": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "skill_nudge": {"type": "STRING"},
            "links": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING"},
                        "url": {"type": "STRING"},
                    },
                    "required":["label", "url"],
                },
            },
        },
        "required":[
            "category",
            "title",
            "content",
            "image_prompt",
            "skill_tags",
            "skill_nudge",
            "links",
        ],
    }


def _sanitize_tile(tile: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all fields in the generated tile are present and safe."""
    return {
        "category": _safe_str(tile.get("category"), "Career Explored"),
        "title": _safe_str(tile.get("title"), "Career Path"),
        "content": _safe_str(tile.get("content"), "This seems connected to a real direction worth exploring."),
        "image_prompt": _safe_str(tile.get("image_prompt"), "A neon pixel-art professional scene with tools, screens, and focused work in progress"),
        "skill_tags":[
            _safe_str(tag)
            for tag in (tile.get("skill_tags") or [])
            if _safe_str(tag)
        ][:2],
        "skill_nudge": _safe_str(tile.get("skill_nudge"), "Try looking up what people in this area actually do each day."),
        "links": _validated_links(tile),
    }


def synthesize_single_tile(
    journey_data: List[Dict[str, Any]],
    superpowers: Dict[str, Any],
    existing_tiles: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Generates one non-duplicate canvas tile based on the user's journey."""
    client = get_client()
    existing_tiles = existing_tiles or []

    recent_history = journey_data[-6:]
    history_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('text', '')}"
        for msg in recent_history
    ).strip()

    primary = _safe_str(superpowers.get("primary"), "Explorer")
    secondary = _safe_str(superpowers.get("secondary"))
    superpower = _safe_str(superpowers.get("superpower"))

    memory = _tile_memory(existing_tiles)
    prior_titles = ", ".join(memory["titles"]) or "none"
    prior_categories = ", ".join(memory["categories"]) or "none"
    prior_skills = ", ".join(memory["skill_tags"]) or "none"

    blocked_titles = list(memory["titles"])

    for attempt in range(3):
        prompt = _build_tile_prompt(
            history_text=history_text,
            primary=primary,
            secondary=secondary,
            superpower=superpower,
            prior_titles=prior_titles,
            prior_categories=prior_categories,
            prior_skills=prior_skills,
            blocked_titles=blocked_titles,
        )

        try:
            response = client.models.generate_content(
                model=TILE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.65 + (attempt * 0.05),
                    response_mime_type="application/json",
                    response_schema=_tile_response_schema(),
                ),
            )

            raw_tile = json.loads(_safe_str(response.text, "{}"))
            tile = _sanitize_tile(raw_tile)

            if len(tile["skill_tags"]) < 2:
                tile["skill_tags"] = (tile["skill_tags"] + ["Research", "Communication"])[:2]

            if _is_duplicate_tile(tile, existing_tiles):
                title = _safe_str(tile.get("title")).lower()
                if title and title not in blocked_titles:
                    blocked_titles.append(title)
                continue

            return tile

        except Exception as e:
            logger.exception("Outcome error on attempt %s: %s", attempt + 1, e)

    return None


def _intro_schema() -> Dict[str, Any]:
    """Defines the strict JSON schema for the intro message."""
    return {
        "type": "OBJECT",
        "properties": {
            "intro_text": {"type": "STRING"},
        },
        "required":["intro_text"],
    }


def _intro_prompt() -> str:
    """Builds the system prompt for generating the daily intro message."""
    return """
You are ArcMotivate, a live interface mapping the contours of a user's potential.

Write the opening message for a young person's first experience in the app.
CRITICAL: The output must be EXTREMELY concise to fit on a small mobile screen without scrolling.

Output requirements:
- Return JSON only with one key: "intro_text"
- The value must be markdown text.
- STRICT LENGTH LIMIT: Maximum of 3 short sentences total. Cut all filler words.
- Structure: EXACTLY this order (No sandwiching!):
  1. FIRST: 1[VISUALIZE: <prompt>] marker describing a neon pixel-art control room waking up.
  2. SECOND: A short explanation that you map their input to find the future their mind demands.
  3. THIRD: A punchy starting question (e.g., what energizes them, or a moment stuck on loop).
- End by inviting them to type a message or attach an image.
- Do NOT include any [SKILL: ...] markers.
- Do NOT sound like a therapist, teacher, or generic assistant.

CRITICAL LANGUAGE RULES:
1. The text inside the [VISUALIZE: <prompt>] marker MUST ALWAYS be in English. Image generators only understand English.
2. The rest of the intro text should be in English.
3. You MUST use British English spelling (e.g., "colour", "realise") and UK cultural context.

Output JSON only.
""".strip()

def generate_intro_message() -> str:
    """Generates the app intro with one interleaved visual marker."""
    client = get_client()

    try:
        response = client.models.generate_content(
            model=INTRO_MODEL,
            contents=_intro_prompt(),
            config=types.GenerateContentConfig(
                temperature=0.8,
                response_mime_type="application/json",
                response_schema=_intro_schema(),
            ),
        )
        data = json.loads(_safe_str(response.text, "{}"))
        intro_text = _safe_str(data.get("intro_text"))

        # Guardrails: require exactly one VISUALIZE marker and no SKILL marker
        visualize_count = len(re.findall(r"\[VISUALIZE:\s*.+?\]", intro_text, flags=re.DOTALL))
        has_skill = bool(re.search(r"\[SKILL:\s*.+?\]", intro_text, flags=re.DOTALL))

        if not intro_text or visualize_count != 1 or has_skill:
            return DEFAULT_INTRO

        return intro_text

    except Exception as e:
        logger.exception("Intro generation failed: %s", e)
        return DEFAULT_INTRO