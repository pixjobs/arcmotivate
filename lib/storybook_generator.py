import base64
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from google import genai

logger = logging.getLogger(__name__)

TEXT_MODEL = "gemini-3.1-flash-lite-preview"
STRUCTURED_MODEL = "gemini-3-flash-preview"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"


def get_client():
    return genai.Client()


# ============================================================
# SMALL HELPERS
# ============================================================

def _safe_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _collect_interests(user_profile: Dict[str, Any], limit: int = 6) -> List[str]:
    superpowers = user_profile.get("superpowers", {}) or user_profile
    interests = superpowers.get("interests")
    if not isinstance(interests, list):
        return []
    out: List[str] = []
    for i, item in enumerate(interests):
        if i >= limit:
            break
        if item is not None:
            out.append(str(item))
    return out


def _profile_summary(user_profile: Dict[str, Any]) -> Dict[str, str]:
    superpowers = user_profile.get("superpowers", {}) or user_profile
    primary = (
        superpowers.get("primary")
        or superpowers.get("archetype")
        or "Explorer"
    )
    secondary = superpowers.get("secondary") or ""
    description = superpowers.get("description") or ""
    interests = ", ".join(_collect_interests(user_profile))
    return {
        "primary": str(primary),
        "secondary": str(secondary),
        "description": str(description),
        "interests": interests,
    }


# ============================================================
# HERO RECAP
# ============================================================

def generate_hero_recap(user_profile: Dict[str, Any]) -> str:
    """
    Generates a short 2–3 sentence recap for the user.
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    prompt = f"""
You are ArcMotivate.

Write a 2–3 sentence “ending screen” message for a young person aged 8–18.
Tone: uplifting, modern, confident, grounded.
Do not use fantasy language, RPG classes, magic, destiny, or fake job titles.

Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Description: {profile['description']}
- Interests: {profile['interests']}

Requirements:
- 2–3 sentences total
- Simple language
- Make it about exploration and growth, not choosing one forever-career
- End with a forward-looking line like:
  "Next level: try one small experiment this week."
""".strip()

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config={"temperature": 0.9},
        )
        text = (response.text or "").strip()
        return text or "You’re building your future one small discovery at a time. Next level: try one small experiment this week."
    except Exception as exc:
        logger.error("Hero recap generation failed: %s", exc)
        return "You’re building your future one small discovery at a time. Next level: try one small experiment this week."

""" 
Generates a neon pixel-art image.
Returns base64-encoded image bytes as a UTF-8 string; empty string on failure.
"""
def generate_pixel_art_illustration(scene_description: str) -> str:

    client = get_client()

    style = (
        "Retro neon pixel-art, 16-bit arcade poster, crisp pixel edges, glowing neon highlights, "
        "dark synthwave background, high contrast, readable silhouettes, no text."
    )
    full_prompt = f"{scene_description}. Style: {style}"

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=full_prompt,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {
                    "aspect_ratio": "16:9",
                },
            },
        )

        for part in (getattr(response, "parts", None) or []):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return base64.b64encode(inline_data.data).decode("utf-8")

    except Exception as exc:
        logger.error("Pixel-art image generation failed: %s", exc)

    return ""

# ============================================================
# CUSTOM AVATAR
# ============================================================

def generate_custom_avatar(
    user_profile: Dict[str, Any],
    latest_signal: str = "",
    aspect_ratio: str = "1:1",
) -> str:
    """
    Generates a custom avatar image and returns base64-encoded image bytes.
    Empty string on failure.
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    prompt = f"""
Create a custom avatar portrait for a young person's career-exploration profile.

Character signals:
- Core archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Latest conversational signal: {latest_signal}

Style requirements:
- polished digital illustration
- futuristic but grounded
- expressive, optimistic, smart
- readable face / silhouette
- subtle neon arcade influence
- no text, no logos, no watermark
- suitable as a profile avatar
- clean composition, centered subject
""".strip()

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio},
            },
        )

        for part in (getattr(response, "parts", None) or []):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return base64.b64encode(inline_data.data).decode("utf-8")

    except Exception as exc:
        logger.error("Avatar generation failed: %s", exc)

    return ""


# ============================================================
# SONG SPEC -> MIDI
# ============================================================

def generate_custom_song(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a custom song spec from the user's profile, then renders it to MIDI.

    Returns:
        {
            "title": str,
            "subtitle": str,
            "bpm": int,
            "midi_b64": str,
            "midi_path": str,
            "spec": dict
        }
    """
    spec = generate_song_spec(user_profile, recent_chat)
    if not spec:
        return {
            "title": "My Song",
            "subtitle": "A custom anthem",
            "bpm": 100,
            "midi_b64": "",
            "midi_path": "",
            "spec": {},
        }

    midi_path, midi_b64 = render_song_spec_to_midi(spec, output_path=output_path)

    return {
        "title": spec.get("title", "My Song"),
        "subtitle": spec.get("subtitle", "A custom anthem"),
        "bpm": int(spec.get("bpm", 100)),
        "midi_b64": midi_b64,
        "midi_path": midi_path,
        "spec": spec,
    }


def generate_song_spec(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Uses Gemini structured output to create a compact song blueprint.
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    history_lines: List[str] = []
    for msg in recent_chat or []:
        role = str(msg.get("role", "user"))
        text = msg.get("text", msg.get("content", ""))
        text = _safe_str(text)
        if text:
            history_lines.append(f"{role}: {text}")
    history_text = "\n".join(history_lines[-8:])

    prompt = f"""
You are designing a short personal anthem for a young person.

Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Description: {profile['description']}
- Interests: {profile['interests']}

Recent conversation:
{history_text}

Create a short, uplifting, loopable song blueprint.
It should feel personal, modern, motivating, and emotionally clear.
Keep it compact enough for a lightweight 20-45 second MIDI.
Avoid anything gloomy or chaotic.
""".strip()

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "bpm": {"type": "integer"},
            "tonic": {"type": "string"},
            "scale": {"type": "string"},
            "mood": {"type": "string"},
            "lead_program": {"type": "integer"},
            "bass_program": {"type": "integer"},
            "pad_program": {"type": "integer"},
            "drums": {"type": "boolean"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "bars": {"type": "integer"},
                        "chords": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "motif_degrees": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "energy": {"type": "integer"},
                    },
                    "required": ["name", "bars", "chords", "motif_degrees", "energy"],
                },
            },
        },
        "required": [
            "title",
            "subtitle",
            "bpm",
            "tonic",
            "scale",
            "mood",
            "lead_program",
            "bass_program",
            "pad_program",
            "drums",
            "sections",
        ],
    }

    try:
        response = client.models.generate_content(
            model=STRUCTURED_MODEL,
            contents=prompt,
            config={
                "temperature": 0.8,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        return json.loads((response.text or "").strip())
    except Exception as exc:
        logger.error("Song spec generation failed: %s", exc)
        return _fallback_song_spec(profile)


def _fallback_song_spec(profile: Dict[str, str]) -> Dict[str, Any]:
    title = f"{profile['primary']} Mode"
    return {
        "title": title,
        "subtitle": "A custom anthem",
        "bpm": 104,
        "tonic": "C",
        "scale": "major",
        "mood": "hopeful, curious, forward-moving",
        "lead_program": 81,   # lead/synth-ish
        "bass_program": 38,   # synth bass-ish
        "pad_program": 89,    # warm pad-ish
        "drums": True,
        "sections": [
            {
                "name": "intro",
                "bars": 2,
                "chords": ["C", "Am"],
                "motif_degrees": [1, 3, 5, 6],
                "energy": 2,
            },
            {
                "name": "lift",
                "bars": 4,
                "chords": ["F", "G", "Em", "Am"],
                "motif_degrees": [3, 5, 6, 5, 3, 2],
                "energy": 4,
            },
        ],
    }


# ============================================================
# MIDI RENDERER
# ============================================================

def render_song_spec_to_midi(
    spec: Dict[str, Any],
    output_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Renders the JSON song spec into a MIDI file using mido.

    Returns:
        (midi_path, midi_b64)
    """
    try:
        import mido
        from mido import Message, MidiFile, MidiTrack, MetaMessage
    except Exception as exc:
        raise RuntimeError(
            "MIDI rendering requires 'mido'. Install it with: pip install mido"
        ) from exc

    bpm = int(spec.get("bpm", 100))
    tonic = _safe_str(spec.get("tonic"), "C").upper()
    scale = _safe_str(spec.get("scale"), "major").lower()
    lead_program = int(spec.get("lead_program", 81))
    bass_program = int(spec.get("bass_program", 38))
    pad_program = int(spec.get("pad_program", 89))
    drums_enabled = bool(spec.get("drums", True))
    sections = spec.get("sections") or []

    mid = MidiFile(ticks_per_beat=480)

    tempo_track = MidiTrack()
    lead_track = MidiTrack()
    bass_track = MidiTrack()
    pad_track = MidiTrack()
    drum_track = MidiTrack()

    mid.tracks.extend([tempo_track, lead_track, bass_track, pad_track])
    if drums_enabled:
        mid.tracks.append(drum_track)

    tempo_track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    tempo_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    lead_track.append(Message("program_change", channel=0, program=max(0, min(127, lead_program)), time=0))
    bass_track.append(Message("program_change", channel=1, program=max(0, min(127, bass_program)), time=0))
    pad_track.append(Message("program_change", channel=2, program=max(0, min(127, pad_program)), time=0))

    scale_notes = _build_scale(tonic, scale)
    chord_roots = {
        "C": 60, "C#": 61, "DB": 61, "D": 62, "D#": 63, "EB": 63, "E": 64,
        "F": 65, "F#": 66, "GB": 66, "G": 67, "G#": 68, "AB": 68,
        "A": 69, "A#": 70, "BB": 70, "B": 71,
    }

    for section in sections:
        bars = max(1, int(section.get("bars", 2)))
        chords = section.get("chords") or [tonic]
        motif = section.get("motif_degrees") or [1, 3, 5, 3]
        energy = max(1, min(5, int(section.get("energy", 3))))

        for bar in range(bars):
            chord_name = _safe_str(chords[bar % len(chords)], tonic).upper()
            root = chord_roots.get(chord_name, 60)

            _append_pad_bar(pad_track, root, energy, ticks_per_beat=480)
            _append_bass_bar(bass_track, root, energy, ticks_per_beat=480)
            _append_lead_bar(lead_track, scale_notes, motif, energy, ticks_per_beat=480)

            if drums_enabled:
                _append_drum_bar(drum_track, energy, ticks_per_beat=480)

    if output_path:
        midi_path = output_path
    else:
        tmp = NamedTemporaryFile(delete=False, suffix=".mid")
        midi_path = tmp.name
        tmp.close()

    mid.save(midi_path)
    midi_b64 = base64.b64encode(Path(midi_path).read_bytes()).decode("utf-8")
    return midi_path, midi_b64


def _build_scale(tonic: str, scale: str) -> List[int]:
    tonic_map = {
        "C": 60, "C#": 61, "DB": 61, "D": 62, "D#": 63, "EB": 63, "E": 64,
        "F": 65, "F#": 66, "GB": 66, "G": 67, "G#": 68, "AB": 68,
        "A": 69, "A#": 70, "BB": 70, "B": 71,
    }
    root = tonic_map.get(tonic.upper(), 60)
    intervals = [0, 2, 4, 5, 7, 9, 11] if scale != "minor" else [0, 2, 3, 5, 7, 8, 10]
    return [root + i for i in intervals]


def _degree_to_midi(scale_notes: List[int], degree: int, octave_shift: int = 0) -> int:
    if degree <= 0:
        degree = 1
    idx = (degree - 1) % len(scale_notes)
    octave = (degree - 1) // len(scale_notes)
    return scale_notes[idx] + (12 * octave) + (12 * octave_shift)


def _append_note(track, note: int, velocity: int, duration_beats: float, ticks_per_beat: int = 480):
    duration_ticks = int(duration_beats * ticks_per_beat)
    track.append(__import__("mido").Message("note_on", note=note, velocity=velocity, time=0))
    track.append(__import__("mido").Message("note_off", note=note, velocity=0, time=duration_ticks))


def _append_lead_bar(track, scale_notes: List[int], motif: List[int], energy: int, ticks_per_beat: int = 480):
    velocity = 70 + energy * 8
    motif = motif[:8] if motif else [1, 3, 5, 3]
    for degree in motif:
        note = _degree_to_midi(scale_notes, int(degree), octave_shift=1)
        _append_note(track, note, velocity, 0.5, ticks_per_beat=ticks_per_beat)


def _append_bass_bar(track, root: int, energy: int, ticks_per_beat: int = 480):
    velocity = 64 + energy * 6
    bass_root = root - 24
    for _ in range(4):
        _append_note(track, bass_root, velocity, 1.0, ticks_per_beat=ticks_per_beat)


def _append_pad_bar(track, root: int, energy: int, ticks_per_beat: int = 480):
    import mido
    velocity = 42 + energy * 4
    third = root + 4
    fifth = root + 7
    duration_ticks = int(4 * ticks_per_beat)

    for note in (root, third, fifth):
        track.append(mido.Message("note_on", note=note, velocity=velocity, time=0))
    track.append(mido.Message("note_off", note=root, velocity=0, time=duration_ticks))
    track.append(mido.Message("note_off", note=third, velocity=0, time=0))
    track.append(mido.Message("note_off", note=fifth, velocity=0, time=0))


def _append_drum_bar(track, energy: int, ticks_per_beat: int = 480):
    import mido
    kick = 36
    snare = 38
    hat = 42

    events = [
        (kick, 90, 0.0),
        (hat, 56 + energy * 4, 0.0),
        (hat, 56 + energy * 4, 0.5),
        (snare, 80, 1.0),
        (hat, 56 + energy * 4, 1.5),
        (kick, 90, 2.0),
        (hat, 56 + energy * 4, 2.0),
        (hat, 56 + energy * 4, 2.5),
        (snare, 82, 3.0),
        (hat, 56 + energy * 4, 3.5),
    ]

    current_ticks = 0
    for note, velocity, beat_pos in events:
        event_ticks = int(beat_pos * ticks_per_beat)
        delta = max(0, event_ticks - current_ticks)
        track.append(mido.Message("note_on", channel=9, note=note, velocity=velocity, time=delta))
        track.append(mido.Message("note_off", channel=9, note=note, velocity=0, time=int(0.08 * ticks_per_beat)))
        current_ticks = event_ticks + int(0.08 * ticks_per_beat)