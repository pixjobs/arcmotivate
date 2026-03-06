import json
import logging
from typing import Any, Dict, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client():
    return genai.Client()

def synthesize_single_tile(journey_data: List[Dict[str, Any]], superpowers: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a single multimodal Canvas Tile based on the latest interaction."""
    client = get_client()
    
    # We really only need the last couple of turns to generate an insight
    recent_history = journey_data[-4:]
    history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('text', '')}" for msg in recent_history])
    
    prompt = f"""
    You are ArcMotivate's background analysis engine. Analyze the latest moments in this teenager's simulation to generate exactly ONE new "Canvas Tile" representing a concrete discovery, skill, or quest.
    
    CRITICAL RULES: Ensure absolute realism. No patronizing tone. Do NOT generate silly, sci-fi, or "neo" job titles like "Sonic Engineer". Ground all insights in actual, contemporary industry practices.

    Assigned Role: {superpowers.get('primary')}
    Recent Interaction: {history_text}
    
    Generate exactly ONE JSON "Canvas Tile" with these exact keys:
    1. "category": Pick ONE: "Skill Unlocked", "Trait Discovered", "Target Role", or "Quest Active".
    2. "title": A super punchy, real-world title for this tile (e.g. "Agile Workflow", "Systems-Thinker", "UX Researcher", "Build a Wireframe").
    3. "content": A 2-sentence description of exactly *why* they earned this tile based on the interaction.
    4. "metadata": An array of exactly two specific metadata strings (e.g. ["Tool: Figma", "Difficulty: Beginner"]).
    5. "image_prompt": A highly descriptive, visual pixel-art prompt to illustrate this specific tile.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
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
                        "metadata": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "image_prompt": {"type": "STRING"}
                    },
                    "required":["category", "title", "content", "metadata", "image_prompt"]
                }
            )
        )
        return json.loads(response.text)
        
    except Exception as e:
        logger.error(f"Outcome error: {e}")
        return None