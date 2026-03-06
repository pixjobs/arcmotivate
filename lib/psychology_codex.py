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
    Maps a free-text personal narrative + optional image to an empowering, 
    growth-oriented identity using complementary psychology frameworks.
    """
    client = get_client()

    # Base fallback in case of empty input or API failure
    fallback_response = {
        "primary": "The Explorer",
        "secondary": "Curiosity-Driven",
        "superpower": "Openness to Experience",
        "description": "You learn best by trying things, noticing what feels exciting, and following the threads that spark your curiosity.",
        "growth_nudge": "Try exploring one completely new hobby or topic this week just for the fun of it!"
    }

    if not narrative or not narrative.strip():
        return fallback_response

    prompt = f"""
You are the ArcMotivate "Psychology Codex", an expert in youth empowerment, positive psychology, and motivation.
Your mission is to analyze a young person's narrative (and optional image) to reflect back their unique strengths and broad curiosity patterns.

Synthesize these non-contradicting psychological frameworks to understand them:
1. Self-Determination Theory (SDT): What fuels their inner fire (Autonomy, Competence, Connection)?
2. Multiple Intelligences (Gardner): How do they process the world (Linguistic, Logic/Math, Visual/Spatial, Bodily/Kinesthetic, Musical, Interpersonal, Intrapersonal, Naturalistic)?
3. VIA Character Strengths (Positive Psychology): What are their core virtues (e.g., Bravery, Creativity, Empathy, Teamwork, Perseverance, Humor)?
4. Growth Mindset (Dweck): Focus on their potential to grow, learn from challenges, and expand their horizons.

The person shared this about themselves:
\"\"\"{narrative}\"\"\"

{"The person also shared an image — let it inform your read of their style and interests." if image_bytes else ""}

Output ONLY valid JSON with these exact fields:

- primary: A broad curiosity identity using "The ___" format. 
  (e.g., "The Maker", "The Storyteller", "The Catalyst", "The Innovator", "The Harmonizer", "The Observer", "The Navigator").
  RULES: MUST start with "The". NO job titles or career labels (no "Engineer", "Manager"). Keep it timeless.

- secondary: A short engagement style tag based on their Multiple Intelligences or SDT.
  (e.g., "Hands-On Builder", "Big-Picture Thinker", "Nature Connected", "People-Focused", "Rhythm & Flow"). 
  RULES: Pure style, no career language.

- superpower: A short phrase capturing their core VIA Character Strength.
  (e.g., "Creative Problem Solving", "Boundless Empathy", "Fearless Curiosity", "Quiet Perseverance").

- description: EXACTLY ONE sentence linking what they shared to what energizes them. 
  Start with "You". Keep it warm, validating, and deeply personal. Show you were listening by referencing a specific detail they mentioned.

- growth_nudge: A short, encouraging "next step" rooted in a Growth Mindset. 
  Suggest a fun, low-pressure way they can stretch this superpower or try something slightly outside their comfort zone to keep growing.

Constraints:
- NO career, job, or diagnostic/clinical language whatsoever.
- Highly suitable, accessible, and encouraging for ages 8–18.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "primary": {"type": "STRING"},
            "secondary": {"type": "STRING"},
            "superpower": {"type": "STRING"},
            "description": {"type": "STRING"},
            "growth_nudge": {"type": "STRING"}
        },
        "required":["primary", "secondary", "superpower", "description", "growth_nudge"]
    }

    try:
        parts =[types.Part.from_text(text=prompt)]
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))

        # Note: Upgraded to gemini-2.5-flash as it is exceptional at complex schema adherence 
        # and psychological synthesis.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.6, # Slightly higher temperature for more creative/diverse archetypes
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        data = json.loads((response.text or "").strip())

        # Validate that all expected keys exist and are non-empty strings
        for k in ("primary", "secondary", "superpower", "description", "growth_nudge"):
            if k not in data or not isinstance(data[k], str) or not data[k].strip():
                raise ValueError(f"Missing or invalid field in model output: {k}")

        # Clean up the description just in case the model added extra whitespace
        data["description"] = data["description"].strip()
        
        return data

    except Exception as e:
        logger.error(f"Codex error: {e}")
        return fallback_response