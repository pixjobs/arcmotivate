"""
generate_assets.py

ArcMotivate - Magical Career Exploration App
Elite Lead Designer Asset Generation Script (Native Multimodal Output)
"""

import os
import sys
import mimetypes
from typing import List, Tuple
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai library is not installed. Please install it using 'pip install google-genai'")
    sys.exit(1)

# Constants for ArcMotivate Asset Generation
ASSETS_DIR: str = "assets"
MODEL_NAME: str = "gemini-3.1-flash-image-preview"

LOGO_PROMPT: str = "Pixel art logo of a rainbow arc with glowing neon stars coming out of it, no text, dark background, lofi tech vibes."
LOGO_FILENAME: str = "logo.png"

FAVICON_PROMPT: str = "32x32 pixel art neon star, dark background"
FAVICON_FILENAME: str = "favicon.png"


def ensure_directory_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"[*] Created directory: {directory_path}")


def save_binary_file(file_name: str, data: bytes) -> None:
    """Helper to save the raw image bytes from Gemini."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"[+] Successfully saved to: {file_name}\n")


def generate_asset(client: genai.Client, prompt: str, output_filepath: str) -> None:
    print(f"[*] Generating asset: {output_filepath}")
    print(f"    Prompt: '{prompt}'")
    
    # Enable native image output via response_modalities
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
    )

    try:
        # We use the standard generate_content (or stream) endpoint now!
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        
        # Iterate through the parts to find the inline_data (the image)
        image_saved = False
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    data_buffer = part.inline_data.data
                    save_binary_file(output_filepath, data_buffer)
                    image_saved = True
                    break
        
        if not image_saved:
            print(f"[!] API succeeded, but no binary image data was returned for {output_filepath}.")

    except Exception as e:
        print(f"[!] Error generating {output_filepath}: {e}\n")


def main() -> None:
    print("==================================================")
    print(" ArcMotivate Asset Generator (Native Multimodal)")
    print("==================================================\n")

    load_dotenv() 
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("[!] Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file.")
        sys.exit(1)

    try:
        client = genai.Client(api_key=api_key) 
    except Exception as e:
        print(f"[!] Failed to initialize Google GenAI client: {e}")
        sys.exit(1)

    ensure_directory_exists(ASSETS_DIR)

    assets_to_generate: List[Tuple[str, str]] =[
        (LOGO_PROMPT, os.path.join(ASSETS_DIR, LOGO_FILENAME)),
        (FAVICON_PROMPT, os.path.join(ASSETS_DIR, FAVICON_FILENAME))
    ]

    for prompt, filepath in assets_to_generate:
        generate_asset(client=client, prompt=prompt, output_filepath=filepath)

    print("==================================================")
    print(" Asset generation complete. Dream big!")
    print("==================================================")


if __name__ == "__main__":
    main()