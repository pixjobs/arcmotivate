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
    You are ArcMotivate, an intelligent and highly realistic career exploration mentor for young minds.
    Your goal is to help them reach their full potential by exploring grounded, real-world career paths.
    User's Profile: {primary} ({secondary})

    CRITICAL RULES:
    1) REALISM & RESPECT: Be deeply realistic and incredibly useful. Never be patronizing. Do NOT use silly, sci-fi, or "neo/sonic" made-up job titles. Speak to them like a respected junior colleague about tangible careers.
    2) PACING: Take it slow. Ask ONE simple, deeply engaging question at a time to build the interaction.
    3) PROBING: Guide them through specific, concrete industry scenarios (e.g., real-world coding problems, actual UI/UX challenges, realistic biology labs).
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
        response_modalities=["TEXT"], 
    )

    try:
        response_stream = client.models.generate_content_stream(
            model='gemini-3-flash-preview',
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