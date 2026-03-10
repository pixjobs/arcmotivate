import json
import logging
import os
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client() -> genai.Client:
    """Lazily initializes and returns the Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing. "
                         "Check your Cloud Run 'Variables & Secrets' configuration.")
    return genai.Client(api_key=api_key)

def map_narrative_to_superpowers(narrative: str, image_bytes: Optional[bytes] = None, image_mime: str = "image/jpeg") -> Dict[str, Any]:
    """
    ArcMotivate Psychology Codex (youth-friendly).
    Maps a free-text personal narrative + optional image to an empowering, 
    growth-oriented identity, extracting structured psychological frameworks 
    (SDT, Gardner, VIA, Dweck) to power downstream application logic.
    """
    client = get_client()

    # Base fallback in case of empty input or API failure.
    # Now includes the hidden psychological framing parameters.
    fallback_response = {
        "primary": "The Technical Explorer",
        "secondary": "Software Engineering & Systems",
        "superpower": "Logical Problem Solving",
        "description": "You learn best by breaking complex systems down to understand how they work, pushing you toward paths where logic and creativity meet.",
        "growth_nudge": "Try researching different technical career roles this week to see what sparks your curiosity.",
        "sdt_driver": "Competence",
        "core_intelligence": "Logical-Mathematical",
        "via_strength": "Curiosity",
        "growth_mindset_reframing": "You haven't built your own software system yet, but experimenting with logic puzzles is how you'll train your brain to do it."
    }

    if not narrative or not narrative.strip():
        return fallback_response

    prompt = f"""
You are the ArcMotivate "Psychology Codex", an expert in youth empowerment, positive psychology, and motivation.
Your mission is to analyze a young person's narrative (and optional image) to reflect back their unique strengths and broad curiosity patterns.

You must deeply analyze their input using four tried-and-tested psychological frameworks, and output the result as structured JSON.

The person shared this about themselves:
\"\"\"{narrative}\"\"\"

{"The person also shared an image — let it inform your read of their style and interests." if image_bytes else ""}

Output ONLY valid JSON with these exact fields:

USER-FACING FIELDS (Keep tone empowering, engaging, and youth-friendly 8-18):
- primary: A broad Career Archetype using "The ___" format (e.g., "The Technical Maker", "The Data Storyteller"). MUST start with "The".
- secondary: A concrete career domain or industry focus (e.g., "Software Engineering", "Renewable Energy", "Product Design").
- superpower: A short phrase capturing their core professional strength (e.g., "Creative Problem Solving", "Systems Architecture").
- description: EXACTLY ONE sentence linking what they shared to the specific career domain they might thrive in. Show you were listening by referencing a specific detail they mentioned.
- growth_nudge: A short, actionable "next step" to explore this career path (e.g., look up a job title, try a beginner project).

HIDDEN PSYCHOLOGICAL FIELDS (Used by our backend AI to tailor future interactions):
- sdt_driver: Based on Self-Determination Theory, what fuels their inner fire? 
  Choose EXACTLY ONE: "Autonomy" (desire for control/independence), "Competence" (desire for mastery/skill), or "Relatedness" (desire for connection/teamwork).
- core_intelligence: Based on Gardner's Multiple Intelligences, how do they process the world? 
  Choose EXACTLY ONE: "Linguistic", "Logical-Mathematical", "Visual-Spatial", "Bodily-Kinesthetic", "Musical", "Interpersonal", "Intrapersonal", or "Naturalistic".
- via_strength: Based on VIA Character Strengths, what is their dominant virtue? 
  (e.g., "Bravery", "Creativity", "Empathy", "Teamwork", "Perseverance", "Humor", "Social Intelligence").
- growth_mindset_reframing: Based on Carol Dweck's Growth Mindset, write ONE sentence framing their current interest as an evolving skill. Use the concept of "Yet" or "Experimentation" to frame challenges not as innate talent tests, but as muscles to build.

Constraints:
- NO diagnostic/clinical language whatsoever.
- Ground the language in real-world professional development and job exploration.
- Ensure the JSON is perfectly formatted.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "primary": {"type": "STRING"},
            "secondary": {"type": "STRING"},
            "superpower": {"type": "STRING"},
            "description": {"type": "STRING"},
            "growth_nudge": {"type": "STRING"},
            "sdt_driver": {"type": "STRING"},
            "core_intelligence": {"type": "STRING"},
            "via_strength": {"type": "STRING"},
            "growth_mindset_reframing": {"type": "STRING"}
        },
        "required": [
            "primary", 
            "secondary", 
            "superpower", 
            "description", 
            "growth_nudge",
            "sdt_driver",
            "core_intelligence",
            "via_strength",
            "growth_mindset_reframing"
        ]
    }

    try:
        parts = [types.Part.from_text(text=prompt)]
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview", # Matched to your coaching agent's fast model, or use 3-flash-preview
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.65, # Slightly higher temperature for creative/diverse archetypes
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        data = json.loads((response.text or "").strip())

        # Validate that all expected keys exist and are non-empty strings
        expected_keys = [
            "primary", "secondary", "superpower", "description", 
            "growth_nudge", "sdt_driver", "core_intelligence", 
            "via_strength", "growth_mindset_reframing"
        ]
        
        for k in expected_keys:
            if k not in data or not isinstance(data[k], str) or not data[k].strip():
                raise ValueError(f"Missing or invalid field in model output: {k}")

        # Clean up text fields just in case the model added extra whitespace
        for k in expected_keys:
            data[k] = data[k].strip()
        
        return data

    except Exception as e:
        logger.error(f"Codex error: {e}")
        return fallback_response