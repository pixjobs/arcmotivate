import base64
from typing import Dict, Any, List
from google import genai
from google.genai import types

def get_client():
    return genai.Client()

def generate_socratic_stream(superpowers: Dict[str, Any], chat_history: List[Dict[str, str]]):
    """Streams interleaved Text and Native Thumbnail Images."""
    client = get_client()
    
    primary = superpowers.get('primary', 'Explorer')
    secondary = superpowers.get('secondary', 'Problem Solver')
    
    system_instruction = f"""
    You are Arc, a sleek, modern career exploration simulator for ages 8–18.
    User's Profile: {primary} ({secondary})

    CRITICAL RULES:
    1) PACING: Take it slow! Do not jump into a massive scenario immediately. Ask ONE simple, engaging question at a time to build the story together.
    2) NO FANTASY: Use cool, near-future tech/professional settings (labs, studios, startups, field research). No magic.
    3) MULTIMODAL MANDATE: You are a multimodal AI. You MUST generate exactly ONE pixel-art image in EVERY single response to visualize the current moment in the simulation.
    4) Keep text punchy (max 2 short paragraphs). End with your question.
    """
    
    contents =[]
    for msg in chat_history:
        # We now receive clean text from app_gradio.py, so parsing is simple!
        role = msg["role"]
        text_content = msg.get("text", "").strip()
            
        if text_content:
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text_content)]))

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7, 
        response_modalities=["IMAGE", "TEXT"], 
    )

    try:
        response_stream = client.models.generate_content_stream(
            model='gemini-3.1-flash-image-preview',
            contents=contents,
            config=config
        )

        for chunk in response_stream:
            if not chunk.parts: continue
            part = chunk.parts[0]
            if part.inline_data and part.inline_data.data:
                yield {"type": "image", "data": base64.b64encode(part.inline_data.data).decode('utf-8')}
            elif part.text:
                yield {"type": "text", "data": part.text}
                
    except Exception as e:
        print(f"Agent Error: {e}")
        yield {"type": "text", "data": "Simulation paused. What was your last move?"}