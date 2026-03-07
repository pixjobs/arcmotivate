import json
import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

TILE_MODEL = "gemini-3-flash-preview"
INTRO_MODEL = "gemini-3.1-flash-lite-preview"

DEFAULT_LINK = {
    "label": "Explore Careers",
    "url": "https://www.google.com/search?q=career+exploration",
}

DEFAULT_INTRO = (
    "👾 **System Online — ArcMotivate**\n\n"
    "Have you ever wondered what kind of future really fits you?\n\n"
    "ArcMotivate is a live exploration agent. I can respond to what you write, what you show me, "
    "and the patterns that emerge as we talk.\n\n"
    "[VISUALIZE: A neon pixel-art control room waking up, with glowing screens, an open sketchbook, "
    "tools, music notes, code fragments, and pathways branching into different futures]\n\n"
    "Before we build your workspace, tell me a little about you. What energises you? What drains you? "
    "What’s something you’re proud of, or a moment that has stuck with you?\n\n"
    "You can type a message or attach an image, and we’ll explore it together."
)


def get_client():
    return genai.Client()


def _safe_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _tile_memory(existing_tiles: Optional[List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    existing_tiles = existing_tiles or []

    titles: List[str] = []
    categories: List[str] = []
    skill_tags: List[str] = []

    for tile in existing_tiles:
        title = _safe_str(tile.get("title"))
        category = _safe_str(tile.get("category"))

        if title:
            titles.append(title.lower())
        if category:
            categories.append(category.lower())

        for tag in tile.get("skill_tags") or []:
            tag = _safe_str(tag)
            if tag:
                skill_tags.append(tag.lower())

    return {
        "titles": titles[-12:],
        "categories": categories[-12:],
        "skill_tags": skill_tags[-24:],
    }


def _is_duplicate_tile(candidate: Dict[str, Any], existing_tiles: Optional[List[Dict[str, Any]]]) -> bool:
    existing_tiles = existing_tiles or []

    cand_title = _normalize_text(_safe_str(candidate.get("title")))
    cand_category = _normalize_text(_safe_str(candidate.get("category")))
    cand_tags = {
        _normalize_text(_safe_str(tag))
        for tag in (candidate.get("skill_tags") or [])
        if _safe_str(tag)
    }

    if not cand_title:
        return True

    for tile in existing_tiles:
        prev_title = _normalize_text(_safe_str(tile.get("title")))
        prev_category = _normalize_text(_safe_str(tile.get("category")))
        prev_tags = {
            _normalize_text(_safe_str(tag))
            for tag in (tile.get("skill_tags") or [])
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
    validated: List[Dict[str, str]] = []

    for link in (tile.get("links") or []):
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

Hard rules:
- No fantasy
- No patronizing tone
- Ground everything in real-world careers, industries, or skills
- Output JSON only
""".strip()


def _tile_response_schema() -> Dict[str, Any]:
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
                    "required": ["label", "url"],
                },
            },
        },
        "required": [
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
    return {
        "category": _safe_str(tile.get("category"), "Career Explored"),
        "title": _safe_str(tile.get("title"), "Career Path"),
        "content": _safe_str(tile.get("content"), "This seems connected to a real direction worth exploring."),
        "image_prompt": _safe_str(tile.get("image_prompt"), "A neon pixel-art professional scene with tools, screens, and focused work in progress"),
        "skill_tags": [
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
    """Generate one non-duplicate canvas tile."""
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

        except Exception:
            logger.exception("Outcome error on attempt %s", attempt + 1)

    return None


def _intro_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "intro_text": {"type": "STRING"},
        },
        "required": ["intro_text"],
    }


def _intro_prompt() -> str:
    return """
You are ArcMotivate.

Write the opening message for a young person's first experience in the app.

The message should preserve the essence of this original intro:
- system online / live exploration vibe
- asks what kind of future really fits them
- explains that the system responds to what they write, show, and reveal in conversation
- invites them to share what energises them, drains them, what they are proud of, or a moment that stuck with them
- mentions they can type a message or attach an image

Output requirements:
- Return JSON only with one key: "intro_text"
- The value should be markdown text
- Keep it concise but warm
- Preserve the original spirit, but make it feel slightly more alive and immersive
- Include exactly one [VISUALIZE: ...] marker woven naturally into the intro
- The visualize prompt must describe a neon pixel-art scene in the same style as the rest of ArcMotivate
- Do NOT include any [SKILL: ...] markers
- Do NOT sound like a therapist, teacher, or generic assistant
- Do NOT use fantasy or cheesy motivational language
- End by inviting them to type a message or attach an image

Structure:
- short opening beat
- short explanation of what ArcMotivate is
- one interwoven [VISUALIZE: ...] marker
- clear invitation to start

Output JSON only.
""".strip()


def generate_intro_message() -> str:
    """Generate the app intro with one interleaved visual marker."""
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

    except Exception:
        logger.exception("Intro generation failed")
        return DEFAULT_INTRO