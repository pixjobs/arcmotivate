import json
import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

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

    prompt = f"""
You are ArcMotivate's background analysis engine. Analyze the latest moments in this teenager's simulation to generate exactly ONE new "Canvas Tile" representing a concrete discovery, skill, or quest.

CRITICAL RULES:
- Ensure absolute realism.
- No patronizing tone.
- Do NOT generate silly, sci-fi, or fake job titles like "Sonic Engineer".
- Ground all insights in actual, contemporary industry practices.

Assigned Role: {superpowers.get('primary')}
Recent Interaction:
{history_text}

Generate exactly ONE JSON object with these keys:
1. "category": Pick ONE of "Skill Unlocked", "Trait Discovered", "Target Role", or "Quest Active".
2. "title": A real-world title for this tile.
3. "content": One sentence explaining why they earned this tile.
4. "image_prompt": A descriptive pixel-art prompt illustrating this tile.
5. "links": An array of 1 to 2 objects, each with "label" and "url".
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "category": {"type": "STRING"},
                        "title": {"type": "STRING"},
                        "content": {"type": "STRING"},
                        "image_prompt": {"type": "STRING"},
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
                    "required": ["category", "title", "content", "image_prompt", "links"]
                }
            )
        )
        return json.loads(response.text)

    except Exception as e:
        logger.exception("Outcome error")
        return None