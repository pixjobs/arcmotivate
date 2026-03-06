import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from lib.outcome_engine import synthesize_single_tile
from lib.storybook_generator import generate_pixel_art_illustration

def test_tile():
    history = [{"role": "user", "text": "I like drawing and computers"}]
    print("Testing tile generation...")
    try:
        tile = synthesize_single_tile(history, {"primary": "Explorer"})
        print("Tile result:", tile)
        if tile and "image_prompt" in tile:
            print("Testing image gen...", tile["image_prompt"])
            img = generate_pixel_art_illustration(tile["image_prompt"])
            print("Image gen length:", len(img) if img else "FAILED")
    except Exception as e:
        print("Error:", e)

test_tile()
