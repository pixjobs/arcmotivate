import sys
import os

sys.path.append("/home/frozenace/arcmotivate/lib")
import storybook_generator

spec = storybook_generator._fallback_song_spec({
    "primary": "Explorer",
    "secondary": "Builder",
    "superpower": "Curiosity",
    "description": "Explores the world",
    "growth_nudge": "Keep trying",
    "interests": "Music, Art",
})

storybook_generator.render_song_spec_to_wav(spec, "/home/frozenace/arcmotivate/out.wav")
print("Done!")
