"""
voice_agent.py — Speech-to-text using Gemini's native audio understanding.
Accepts raw audio bytes (wav/webm/mp4/ogg) and returns transcribed text.
"""
import logging
from typing import Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def get_client():
    return genai.Client()

def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> Optional[str]:
    """
    Sends audio bytes to Gemini and returns the transcribed text.
    Returns None on failure.
    """
    client = get_client()

    prompt = (
        "Transcribe exactly what this person said. "
        "Return only the spoken words with no formatting, labels, or explanations."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ]
                )
            ],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        text = (response.text or "").strip()
        logger.info(f"Transcribed: {text[:80]}")
        return text if text else None
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None
