import json
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client():
    return genai.Client()

def map_narrative_to_superpowers(narrative: str, image_bytes: Optional[bytes] = None, image_mime: str = "image/jpeg") -> Dict[str, Any]:
    """
    ArcMotivate Psychology Codex (youth-friendly).
    Maps a free-text personal narrative (likes, dislikes, school experience)
    + optional image (artwork, project photo, etc.) -> a broad curiosity-style identity.
    
    IMPORTANT: Output is a curiosity/personality archetype, NOT a career label.
    """
    client = get_client()

    if not narrative or not narrative.strip():
        return {
            "primary": "The Explorer",
            "secondary": "Curiosity-Driven",
            "description": "You learn best by trying things, noticing what feels exciting, and following the threads that spark your curiosity."
        }

    prompt = f"""
You are the ArcMotivate "Psychology Codex" for young people aged 8–18.
Your job is to identify broad curiosity patterns and personal motivations from what someone shares about themselves.

This is NOT about predicting careers or giving professional labels.
This is about understanding HOW someone engages with the world and WHAT drives them.

Use ideas inspired by:
- Learning styles and curiosity types (exploration, making, storytelling, helping, investigating, leading, performing)
- Self-Determination Theory (what gives them energy: autonomy, competence, or connection)

The person shared this about themselves:
\"\"\"{narrative}\"\"\"

{"The person also shared an image — let it inform your read of their style and interests." if image_bytes else ""}

Output ONLY valid JSON with these exact fields:

- primary: A broad curiosity identity using "The ___" format.
  Examples: "The Maker", "The Storyteller", "The Explorer", "The Connector", "The Investigator", "The Performer", "The Builder", "The Helper", "The Tinkerer", "The Visionary".
  RULES: 
  - MUST start with "The".
  - Do NOT use job titles, career roles, or professional labels (no "Strategist", "Engineer", "Designer", "Analyst", "Manager", "Developer").
  - Keep it broad, timeless, and identity-level — not a job.

- secondary: A short engagement style tag (e.g., "Hands-On", "Big-Picture", "Team Player", "Solo Focus", "Detail-Oriented", "Fast Mover", "Deep Diver").
  RULES: No career language. Pure style.

- description: EXACTLY ONE sentence linking what they shared to what energizes them.
  Start with "You" and keep it personal, warm, and encouraging.
  Do NOT mention careers, jobs, or professions. Focus on HOW they engage with the world.

Constraints:
- No career or job language anywhere.
- Suitable for ages 8–18.
- No clinical, psychological, or diagnostic language.
- Show you were listening — reference something specific they mentioned.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "primary": {"type": "STRING"},
            "secondary": {"type": "STRING"},
            "description": {"type": "STRING"}
        },
        "required": ["primary", "secondary", "description"]
    }

    try:
        parts = [types.Part.from_text(text=prompt)]
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.5,
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        data = json.loads((response.text or "").strip())

        for k in ("primary", "secondary", "description"):
            if k not in data or not isinstance(data[k], str) or not data[k].strip():
                raise ValueError(f"Missing/invalid field: {k}")

        desc = data["description"].strip()
        if desc.count(".") >= 2:
            desc = desc.split(".")[0].strip() + "."

        data["description"] = desc
        return data

    except Exception as e:
        logger.error(f"Codex error: {e}")
        return {
            "primary": "The Explorer",
            "secondary": "Curiosity-Driven",
            "description": "You're someone who learns by doing, notices what feels real, and follows what genuinely sparks your interest."
        }