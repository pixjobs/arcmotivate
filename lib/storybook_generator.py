import base64
import json
import logging
import math
import random
import struct
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from google import genai

logger = logging.getLogger(__name__)

TEXT_MODEL = "gemini-3.1-flash-lite-preview"
STRUCTURED_MODEL = "gemini-3-flash-preview"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

SAMPLE_RATE = 22050


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
    primary = superpowers.get("primary") or superpowers.get("archetype") or "Explorer"
    secondary = superpowers.get("secondary") or ""
    superpower = superpowers.get("superpower") or ""
    description = superpowers.get("description") or ""
    growth_nudge = superpowers.get("growth_nudge") or ""
    interests = ", ".join(_collect_interests(user_profile))
    return {
        "primary": str(primary),
        "secondary": str(secondary),
        "superpower": str(superpower),
        "description": str(description),
        "growth_nudge": str(growth_nudge),
        "interests": interests,
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _note_to_midi(note_name: str) -> int:
    mapping = {
        "C": 60, "C#": 61, "DB": 61, "D": 62, "D#": 63, "EB": 63, "E": 64,
        "F": 65, "F#": 66, "GB": 66, "G": 67, "G#": 68, "AB": 68,
        "A": 69, "A#": 70, "BB": 70, "B": 71,
    }
    return mapping.get(note_name.upper(), 60)


def _midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _build_scale(tonic: str, scale: str) -> List[int]:
    root = _note_to_midi(tonic)
    intervals = [0, 2, 4, 5, 7, 9, 11] if scale != "minor" else [0, 2, 3, 5, 7, 8, 10]
    return [root + i for i in intervals]


def _degree_to_midi(scale_notes: List[int], degree: int, octave_shift: int = 0) -> int:
    if degree <= 0:
        degree = 1
    idx = (degree - 1) % len(scale_notes)
    octave = (degree - 1) // len(scale_notes)
    return scale_notes[idx] + (12 * octave) + (12 * octave_shift)


# ============================================================
# HERO RECAP
# ============================================================

def generate_hero_recap(user_profile: Dict[str, Any]) -> str:
    client = get_client()
    profile = _profile_summary(user_profile)

    prompt = f"""
You are ArcMotivate.

Write a 2–3 sentence ending-screen message for a young person aged 8–18.
Tone: uplifting, modern, calm, grounded.
Do not use fantasy language, destiny language, or fake job titles.

Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Growth nudge: {profile['growth_nudge']}

Requirements:
- 2–3 sentences total
- Simple language
- Make it about exploration and growth, not choosing one forever path
- End with a small forward-looking line
""".strip()

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config={"temperature": 0.85},
        )
        text = (response.text or "").strip()
        return text or "You’re learning what fits you, one small signal at a time. Next level: try one small experiment this week."
    except Exception as exc:
        logger.error("Hero recap generation failed: %s", exc)
        return "You’re learning what fits you, one small signal at a time. Next level: try one small experiment this week."


# ============================================================
# PIXEL ART
# ============================================================

def generate_pixel_art_illustration(scene_description: str) -> str:
    """
    Generates a neon pixel-art image.
    Returns base64-encoded image bytes as a UTF-8 string; empty string on failure.
    """
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
                "image_config": {"aspect_ratio": "16:9"},
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
    client = get_client()
    profile = _profile_summary(user_profile)

    prompt = f"""
Create a custom avatar portrait for a young person's exploration profile.

Character signals:
- Core archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Latest conversational signal: {latest_signal}

Style requirements:
- polished digital illustration
- futuristic but grounded
- expressive, optimistic, thoughtful
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
# SONG SPEC -> WAV
# ============================================================

def generate_custom_song(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a custom calming audio spec, then renders it to WAV.

    Returns:
        {
            "title": str,
            "subtitle": str,
            "bpm": int,
            "audio_b64": str,
            "audio_path": str,
            "mime_type": "audio/wav",
            "spec": dict
        }
    """
    spec = generate_song_spec(user_profile, recent_chat)
    if not spec:
        spec = _fallback_song_spec(_profile_summary(user_profile))

    audio_path, audio_b64 = render_song_spec_to_wav(spec, output_path=output_path)

    return {
        "title": spec.get("title", "My Sound"),
        "subtitle": spec.get("subtitle", "A calm custom theme"),
        "bpm": int(spec.get("bpm", 68)),
        "audio_b64": audio_b64,
        "audio_path": audio_path,
        "mime_type": "audio/wav",
        "spec": spec,
    }


def generate_song_spec(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Uses Gemini structured output to create a compact, soothing audio blueprint.
    Strongly biased toward calm / ambient / subtle / reflective textures.
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
You are designing a short personal soundscape for a young person.

Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Growth nudge: {profile['growth_nudge']}

Recent conversation:
{history_text}

Create a compact, soothing, subtle, loopable personal theme.

The sound should feel:
- calm
- safe
- reflective
- quietly hopeful
- immersive without demanding attention

Allowed vibe references:
- ambient
- chillout
- soft cinematic
- dreamy
- gentle ASMR-like textures
- theta-wave-inspired calm focus
- warm, minimal synth pads
- soft glassy tones
- slow pulse

Not allowed:
- punk
- rock
- techno
- EDM drops
- aggressive drums
- distortion
- glitch chaos
- jump-scare transitions
- noisy, harsh, or overly busy arrangements

Musical constraints:
- keep it lightweight and subtle
- make it suitable for browser playback
- target 18 to 28 seconds
- use slow to medium-slow tempo
- drums should usually be false
- prefer soft pulse over obvious beat
- keep the melody sparse
- prioritize texture, breath, and emotional clarity over hooks

Return only structured JSON.
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
            "texture": {"type": "string"},
            "carrier_hz": {"type": "number"},
            "beat_hz": {"type": "number"},
            "shimmer": {"type": "number"},
            "noise_amount": {"type": "number"},
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
            "texture",
            "carrier_hz",
            "beat_hz",
            "shimmer",
            "noise_amount",
            "drums",
            "sections",
        ],
    }

    try:
        response = client.models.generate_content(
            model=STRUCTURED_MODEL,
            contents=prompt,
            config={
                "temperature": 0.65,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        return json.loads((response.text or "").strip())
    except Exception as exc:
        logger.error("Song spec generation failed: %s", exc)
        return _fallback_song_spec(profile)


def _fallback_song_spec(profile: Dict[str, str]) -> Dict[str, Any]:
    title = f"{profile['primary']} Drift"
    return {
        "title": title,
        "subtitle": "A calm custom theme",
        "bpm": 68,
        "tonic": "C",
        "scale": "major",
        "mood": "calm, reflective, quietly hopeful",
        "texture": "soft ambient pad with subtle shimmer",
        "carrier_hz": 140.0,
        "beat_hz": 5.0,
        "shimmer": 0.22,
        "noise_amount": 0.018,
        "drums": False,
        "sections": [
            {
                "name": "settle",
                "bars": 4,
                "chords": ["C", "Am", "F", "G"],
                "motif_degrees": [1, 3, 5, 3],
                "energy": 2,
            },
            {
                "name": "float",
                "bars": 4,
                "chords": ["Am", "F", "C", "G"],
                "motif_degrees": [3, 5, 6, 5],
                "energy": 3,
            },
        ],
    }


# ============================================================
# FAST WAV RENDERER
# ============================================================

def render_song_spec_to_wav(
    spec: Dict[str, Any],
    output_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Fast pure-Python WAV renderer.
    Produces browser-native audio with a subtle ambient / theta-inspired feel.

    Returns:
        (audio_path, audio_b64)
    """
    bpm = int(spec.get("bpm", 68))
    tonic = _safe_str(spec.get("tonic"), "C").upper()
    scale = _safe_str(spec.get("scale"), "major").lower()
    carrier_hz = float(spec.get("carrier_hz", 140.0))
    beat_hz = float(spec.get("beat_hz", 5.0))
    shimmer = float(spec.get("shimmer", 0.22))
    noise_amount = float(spec.get("noise_amount", 0.018))
    sections = spec.get("sections") or []

    bpm = max(52, min(84, bpm))
    carrier_hz = _clamp(carrier_hz, 90.0, 220.0)
    beat_hz = _clamp(beat_hz, 3.0, 7.0)
    shimmer = _clamp(shimmer, 0.0, 0.45)
    noise_amount = _clamp(noise_amount, 0.0, 0.05)

    scale_notes = _build_scale(tonic, scale)
    chord_roots = {
        "C": 60, "C#": 61, "DB": 61, "D": 62, "D#": 63, "EB": 63, "E": 64,
        "F": 65, "F#": 66, "GB": 66, "G": 67, "G#": 68, "AB": 68,
        "A": 69, "A#": 70, "BB": 70, "B": 71,
    }

    beat_seconds = 60.0 / bpm
    bar_seconds = 4.0 * beat_seconds

    arrangement: List[Dict[str, Any]] = []
    total_duration = 0.0

    for section in sections:
        bars = max(1, int(section.get("bars", 2)))
        chords = section.get("chords") or [tonic]
        motif = section.get("motif_degrees") or [1, 3, 5, 3]
        energy = max(1, min(5, int(section.get("energy", 2))))

        for bar in range(bars):
            chord_name = _safe_str(chords[bar % len(chords)], tonic).upper()
            root = chord_roots.get(chord_name, 60)
            arrangement.append(
                {
                    "root": root,
                    "motif": motif,
                    "energy": energy,
                    "start": total_duration,
                    "duration": bar_seconds,
                }
            )
            total_duration += bar_seconds

    total_duration = _clamp(total_duration, 18.0, 28.0)
    total_frames = int(total_duration * SAMPLE_RATE)

    if output_path:
        audio_path = output_path
    else:
        tmp = NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = tmp.name
        tmp.close()

    rng = random.Random(1337)

    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        frames = bytearray()

        for i in range(total_frames):
            t = i / SAMPLE_RATE

            section = arrangement[min(len(arrangement) - 1, int(t / bar_seconds))] if arrangement else {
                "root": 60,
                "motif": [1, 3, 5, 3],
                "energy": 2,
                "start": 0.0,
                "duration": bar_seconds,
            }

            root = int(section["root"])
            motif = section["motif"]
            energy = int(section["energy"])
            local_t = t - float(section["start"])
            local_phase = local_t / max(0.001, float(section["duration"]))

            # Soft chord bed
            third = root + (3 if scale == "minor" else 4)
            fifth = root + 7

            root_f = _midi_to_freq(root - 12)
            third_f = _midi_to_freq(third - 12)
            fifth_f = _midi_to_freq(fifth - 12)

            chord_env = _slow_env(local_phase)

            pad = (
                0.19 * math.sin(2 * math.pi * root_f * t) +
                0.14 * math.sin(2 * math.pi * third_f * t) +
                0.12 * math.sin(2 * math.pi * fifth_f * t)
            ) * chord_env

            # Sparse glassy motif
            motif_step = min(len(motif) - 1, int(local_phase * max(1, len(motif))))
            degree = int(motif[motif_step]) if motif else 1
            lead_note = _degree_to_midi(scale_notes, degree, octave_shift=1)
            lead_f = _midi_to_freq(lead_note)
            lead_env = _plucked_env((local_phase * len(motif)) % 1.0)
            lead = (
                0.06 * math.sin(2 * math.pi * lead_f * t) +
                0.025 * math.sin(2 * math.pi * lead_f * 2.0 * t)
            ) * lead_env * (0.6 + 0.08 * energy)

            # Theta-like carrier and subtle binaural separation
            left_carrier = 0.08 * math.sin(2 * math.pi * (carrier_hz - beat_hz / 2.0) * t)
            right_carrier = 0.08 * math.sin(2 * math.pi * (carrier_hz + beat_hz / 2.0) * t)

            # Soft shimmer
            shimmer_freq = root_f * 2.0
            shimmer_sig = shimmer * 0.035 * math.sin(2 * math.pi * shimmer_freq * t)

            # Breath / ASMR-like noise
            noise = (rng.uniform(-1.0, 1.0) * noise_amount) * _breath_env(local_phase)

            left = pad + lead + left_carrier + shimmer_sig + noise
            right = pad + lead + right_carrier + shimmer_sig + noise

            # Gentle master fade in/out
            master = _master_env(t, total_duration)
            left *= master
            right *= master

            left = _soft_clip(left * 0.9)
            right = _soft_clip(right * 0.9)

            left_i = int(_clamp(left, -1.0, 1.0) * 32767)
            right_i = int(_clamp(right, -1.0, 1.0) * 32767)

            frames.extend(struct.pack("<hh", left_i, right_i))

        wav_file.writeframes(frames)

    audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
    return audio_path, audio_b64


def _slow_env(phase: float) -> float:
    phase = _clamp(phase, 0.0, 1.0)
    return 0.55 + 0.45 * math.sin(math.pi * phase)


def _plucked_env(phase: float) -> float:
    phase = _clamp(phase, 0.0, 1.0)
    attack = min(1.0, phase / 0.08)
    decay = math.exp(-3.8 * phase)
    return attack * decay


def _breath_env(phase: float) -> float:
    phase = _clamp(phase, 0.0, 1.0)
    return 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * phase))


def _master_env(t: float, total_duration: float) -> float:
    fade_in = _clamp(t / 1.4, 0.0, 1.0)
    fade_out = _clamp((total_duration - t) / 1.8, 0.0, 1.0)
    return fade_in * fade_out


def _soft_clip(x: float) -> float:
    return math.tanh(x)