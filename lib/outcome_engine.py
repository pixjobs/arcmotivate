import json
import logging
from typing import Any, Dict, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client():
    return genai.Client()

def synthesize_blueprint(journey_data: List[Dict[str, Any]], superpowers: Dict[str, Any]) -> Dict[str, Any]:
    """Updates the 'Future Blueprint' HUD based on the ongoing simulation."""
    client = get_client()
    
    user_inputs =[step.get("text", "") for step in journey_data if step.get("role") == "user"]
    chat_summary = " | ".join(user_inputs)
    
    prompt = f"""
    You are ArcMotivate's background analysis engine. Analyze this teenager's simulation choices.
    Assigned Role: {superpowers.get('primary')}
    User's Choices: {chat_summary}
    
    Update their "Future Blueprint" JSON:
    1. title: A cool, modern title (e.g. "Lead Systems Architect", "Creative Director").
    2. analysis: A brief, encouraging 2-sentence summary of what their choices reveal about their strengths.
    3. skills: Array of exactly 3 hard/soft skills they are demonstrating.
    4. careers: Array of exactly 2 REAL-WORLD job titles that fit this path.
    5. quest: One fun, free, real-world action they can take today (e.g. "Try building a basic website", "Sketch a prototype").
    """
    
    try:
        # Flash-Lite for speed
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "analysis": {"type": "STRING"},
                        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "careers": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "quest": {"type": "STRING"}
                    },
                    "required":["title", "analysis", "skills", "careers", "quest"]
                }
            )
        )
        return json.loads(response.text)
        
    except Exception as e:
        logger.error(f"Outcome error: {e}")
        return None