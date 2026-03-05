import json
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client():
    return genai.Client()

def map_interests_to_superpowers(interests: List[str]) -> Dict[str, Any]:
    """
    ArcMotivate Psychology Codex (youth-friendly).
    Maps interests -> a grounded "Primary role title" + "Secondary style" + 1-sentence explanation.
    Uses career-construction + motivation concepts as inspiration (not diagnosis).
    """
    client = get_client()

    clean = [i.strip() for i in interests if i and i.strip()]
    if not clean:
        return {
            "primary": "Explorer",
            "secondary": "Curiosity Engine",
            "description": "You learn best by trying things, noticing what feels fun, and leveling up a little each time."
        }

    prompt = f"""
You are the ArcMotivate “Psychology Codex” for young people aged 8–18.
Your job is to spot patterns and motivations from interests — NOT to diagnose or label.
Use ideas inspired by:
- Career Construction (life themes + preferred ways of solving problems)
- Self-Determination Theory (autonomy, competence, relatedness)

User interests/hobbies: {", ".join(clean)}

Output ONLY valid JSON matching the schema with:
- primary: a grounded, modern role title (NOT fantasy/magic). Think: "Creative Builder", "Neon Problem-Solver", "Systems Thinker", "Story Designer", "Community Helper", "Tech Tinkerer".
- secondary: a short style tag (e.g., "Team Player", "Solo Focus", "Hands-On", "Big-Picture", "Detail Detective").
- description: EXACTLY ONE sentence, encouraging and simple, that links their interests to what motivates them (autonomy/competence/relatedness) and what themes they care about (helping, building, exploring, creating, leading, investigating).

Constraints:
- No fantasy classes, no magic words, no "RPG" phrasing.
- Keep it suitable for ages 8–18.
- Avoid sensitive claims (no mental health or clinical language).
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
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
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
            "primary": "Explorer",
            "secondary": "Problem Solver",
            "description": "You’re learning what you like by exploring, practicing skills, and noticing what feels meaningful."
        }