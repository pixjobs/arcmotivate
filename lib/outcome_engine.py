import json
import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

TILE_MODEL = "gemini-3-flash-preview"

# Removed strict allowlist to allow dynamic career search links


def get_client():
    return genai.Client()


def synthesize_single_tile(
    journey_data: List[Dict[str, Any]],
    superpowers: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Generates a single Canvas Tile based on the latest interaction."""
    client = get_client()

    recent_history = journey_data[-4:]
    history_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('text', '')}"
        for msg in recent_history
    )

    primary = superpowers.get("primary", "Explorer")
    secondary = superpowers.get("secondary", "")
    superpower = superpowers.get("superpower", "")

    prompt = f"""
You are ArcMotivate's background analysis engine.
Analyze the latest moments in this person's exploration to generate exactly ONE new "Career Tile" representing a concrete job role, industry, or professional skill.

CRITICAL RULES:
- Ground all insights in real-world professions and career pathways.
- No patronizing tone. No fantasy. 
- You MUST tie their curiosities to legitimate, existing career fields.

Current identity read:
- Primary: {primary}
- Secondary: {secondary}
- Superpower: {superpower}

Recent interaction:
{history_text}

Generate exactly ONE JSON object with these keys:
1. "category": Pick ONE of "Career Explored", "Industry Unlocked", "Professional Skill", or "Real-World Quest".
2. "title": A concrete job title, career role, or industry name related to their insight (e.g., "UX Researcher", "Robotics Engineer").
3. "content": One sentence connecting what they shared to this career or industry. Be inspiring but grounded.
4. "image_prompt": A descriptive neon pixel-art prompt illustrating this career in action.
5. "skill_tags": An array of 1 to 3 concrete professional skills required for this career (e.g., ["Python", "User Testing", "3D CAD"]).
6. "skill_nudge": A short, actionable "try this" suggestion to explore this career. One sentence.
7. "links": An array of 1 to 2 objects. For the "label", use a clear name (e.g. "Explore UX Research"). For the "url", generate a direct Google search URL formatted EXACTLY like this: "https://www.google.com/search?q=" + URL-encoded search terms.
""".strip()

    try:
        response = client.models.generate_content(
            model=TILE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.65,
                response_mime_type="application/json",
                response_schema={
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
                                    "url": {"type": "STRING"}
                                },
                                "required": ["label", "url"]
                            }
                        }
                    },
                    "required": ["category", "title", "content", "image_prompt", "skill_tags", "skill_nudge", "links"]
                }
            )
        )
        tile = json.loads(response.text)

        # Validate links are secure links
        validated_links = []
        for link in (tile.get("links") or []):
            if isinstance(link, dict) and link.get("url", "").startswith("https://"):
                validated_links.append(link)
        tile["links"] = validated_links or [
            {"label": "Explore Careers", "url": "https://www.google.com/search?q=career+exploration"}
        ]

        return tile

    except Exception:
        logger.exception("Outcome error")
        return None