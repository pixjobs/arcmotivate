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


def _extract_text_from_msg(msg: Dict[str, Any]) -> str:
    text = msg.get("text")
    if text and isinstance(text, str):
        return text
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""


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
    Uses Gemini structured output to create a compact lofi chillout audio blueprint.
    Biased toward mellow / warm / jazzy / tape-warbled textures.
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    history_lines: List[str] = []
    for msg in recent_chat or []:
        role = str(msg.get("role", "user"))
        text = _extract_text_from_msg(msg)
        text = _safe_str(text)
        if text:
            history_lines.append(f"{role}: {text}")
    history_text = "\n".join(history_lines[-8:])

    prompt = f"""
You are designing a short personal lofi chillout soundtrack for a young person.

Profile:
- Archetype: {profile['primary']}
- Secondary signal: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}
- Growth nudge: {profile['growth_nudge']}

Recent conversation:
{history_text}

Create a compact, mellow, warm, loopable personal acoustic piano theme.

The sound should feel like:
- intimate, acoustic lofi piano
- warm vinyl crackling softly in the background
- late-night cozy reflection session
- calm rain on a window while you read
- mellow, drowsy, acoustic minimalism

Musical style guide:
- Use jazzy chord progressions with 7ths (Cmaj7, Am7, Dm7, Fmaj7, etc.)
- Very slow, laid-back tempo (50–70 BPM)
- Warm acoustic piano textures (no synth pads)
- Subtle tape wobble / pitch drift
- Soft vinyl dust / hiss (not distracting)
- Sparse, breathy melody — like a slow, distant piano line
- No drums or rhythm section
- Overall: ACOUSTIC and MELLOW above all else

Not allowed:
- harsh synths, shiny electric leads, or synth pads
- rock, EDM, techno, trap
- fast tempos or energetic buildups
- complex arrangements or busy melodies
- anything that pulls attention away from what the listener is doing

Personalization:
- Let the archetype influence the emotional color (e.g., "Builder" → warm and steady, "Explorer" → curious and drifty, "Creator" → playful and gentle)
- The title and subtitle should reflect their personality warmly

Musical constraints:
- target 20 to 28 seconds
- tempo 50 to 70 BPM
- drums must be false
- keep shimmer very low (0.01 to 0.05)
- use higher noise_amount for vinyl warmth (0.02 to 0.04)
- wobble_rate 0.3 to 0.8 Hz for tape drift feel
- prefer minor or mixolydian scales for that lofi color

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
            "wobble_rate": {"type": "number"},
            "vinyl_dust": {"type": "number"},
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
            "wobble_rate",
            "vinyl_dust",
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
    title = f"{profile['primary']} Lofi"
    return {
        "title": title,
        "subtitle": "A mellow acoustic piano theme just for you",
        "bpm": 60,
        "tonic": "C",
        "scale": "minor",
        "mood": "mellow, warm, acoustic chill",
        "texture": "warm lofi piano with vinyl dust",
        "carrier_hz": 130.0,
        "beat_hz": 4.5,
        "shimmer": 0.02,
        "noise_amount": 0.028,
        "wobble_rate": 0.5,
        "vinyl_dust": 0.015,
        "drums": False,
        "sections": [
            {
                "name": "settle",
                "bars": 4,
                "chords": ["Cm", "Ab", "Fm", "G"],
                "motif_degrees": [1, 3, 5, 3],
                "energy": 2,
            },
            {
                "name": "drift",
                "bars": 4,
                "chords": ["Ab", "Fm", "Cm", "G"],
                "motif_degrees": [3, 5, 6, 5],
                "energy": 2,
            },
        ],
    }


# ============================================================
# FAST WAV RENDERER (LOFI EDITION)
# ============================================================

def render_song_spec_to_wav(
    spec: Dict[str, Any],
    output_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Pure-Python WAV renderer tuned for lofi chillout aesthetics.
    Warm pads, jazzy voicings, tape wobble, vinyl dust.

    Returns:
        (audio_path, audio_b64)
    """
    bpm = int(spec.get("bpm", 68))
    tonic = _safe_str(spec.get("tonic"), "C").upper()
    scale = _safe_str(spec.get("scale"), "minor").lower()
    carrier_hz = float(spec.get("carrier_hz", 130.0))
    beat_hz = float(spec.get("beat_hz", 4.5))
    shimmer = float(spec.get("shimmer", 0.08))
    noise_amount = float(spec.get("noise_amount", 0.028))
    wobble_rate = float(spec.get("wobble_rate", 0.5))
    vinyl_dust = float(spec.get("vinyl_dust", 0.015))
    sections = spec.get("sections") or []

    # Lofi Acoustic constraints
    bpm = max(50, min(70, bpm))
    carrier_hz = _clamp(carrier_hz, 80.0, 180.0)
    beat_hz = _clamp(beat_hz, 2.5, 6.0)
    shimmer = _clamp(shimmer, 0.0, 0.05) # Much lower shimmer so it doesn't sound synthy
    noise_amount = _clamp(noise_amount, 0.01, 0.045)
    wobble_rate = _clamp(wobble_rate, 0.2, 1.0)
    vinyl_dust = _clamp(vinyl_dust, 0.005, 0.03)

    scale_notes = _build_scale(tonic, scale)
    chord_roots = {
        "C": 60, "C#": 61, "DB": 61, "D": 62, "D#": 63, "EB": 63, "E": 64,
        "F": 65, "F#": 66, "GB": 66, "G": 67, "G#": 68, "AB": 68,
        "A": 69, "A#": 70, "BB": 70, "B": 71,
        # Also map chord names with quality markers (strip them)
        "CM": 60, "DM": 62, "EM": 64, "FM": 65, "GM": 67, "AM": 69, "BM": 71,
    }

    beat_seconds = 60.0 / bpm
    bar_seconds = 4.0 * beat_seconds

    arrangement: List[Dict[str, Any]] = []
    total_duration = 0.0

    for section in sections:
        bars = max(1, int(section.get("bars", 2)))
        chords = section.get("chords") or [tonic]
        motif = section.get("motif_degrees") or [1, 3, 5, 3]
        energy = max(1, min(4, int(section.get("energy", 2))))  # Cap energy for lofi

        for bar in range(bars):
            raw_chord = _safe_str(chords[bar % len(chords)], tonic).upper()
            # Strip quality markers (MAJ7, M7, 7, etc.) for root lookup
            chord_clean = raw_chord.replace("MAJ7", "").replace("MIN7", "").replace("M7", "").replace("M", "").replace("7", "").strip()
            if not chord_clean:
                chord_clean = tonic
            root = chord_roots.get(chord_clean, chord_roots.get(chord_clean[0], 60))
            # Detect minor quality
            is_minor_chord = "M" in raw_chord and "MAJ" not in raw_chord
            arrangement.append(
                {
                    "root": root,
                    "motif": motif,
                    "energy": energy,
                    "start": total_duration,
                    "duration": bar_seconds,
                    "is_minor": is_minor_chord or scale in ("minor", "dorian"),
                }
            )
            total_duration += bar_seconds

    total_duration = _clamp(total_duration, 20.0, 28.0)
    total_frames = int(total_duration * SAMPLE_RATE)

    if output_path:
        audio_path = output_path
    else:
        tmp = NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = tmp.name
        tmp.close()

    rng = random.Random(42)

    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        frames = bytearray()

        phase_root = 0.0
        phase_third = 0.0
        phase_fifth = 0.0
        phase_seventh = 0.0
        phase_root_dt1 = 0.0
        phase_fifth_dt2 = 0.0
        phase_lead1 = 0.0
        phase_lead2 = 0.0
        phase_sub = 0.0
        phase_shimmer = 0.0

        for i in range(total_frames):
            t = i / SAMPLE_RATE

            section = arrangement[min(len(arrangement) - 1, int(t / bar_seconds))] if arrangement else {
                "root": 60,
                "motif": [1, 3, 5, 3],
                "energy": 2,
                "start": 0.0,
                "duration": bar_seconds,
                "is_minor": True,
            }

            root = int(section["root"])
            motif = section["motif"]
            energy = int(section["energy"])
            is_minor = bool(section.get("is_minor", True))
            local_t = t - float(section["start"])
            local_phase = local_t / max(0.001, float(section["duration"]))

            # ── Tape wobble (pitch drift) ──
            wobble = 1.0 + 0.002 * math.sin(2 * math.pi * wobble_rate * t)

            # ── Warm chord pad with 7th (jazzy lofi voicing) ──
            third = root + (3 if is_minor else 4)
            fifth = root + 7
            seventh = root + (10 if is_minor else 11)  # min7 or maj7

            root_f = _midi_to_freq(root - 12) * wobble
            third_f = _midi_to_freq(third - 12) * wobble
            fifth_f = _midi_to_freq(fifth - 12) * wobble
            seventh_f = _midi_to_freq(seventh - 12) * wobble

            chord_env = _piano_env(local_phase)

            phase_root = (phase_root + root_f / SAMPLE_RATE) % 1.0
            phase_third = (phase_third + third_f / SAMPLE_RATE) % 1.0
            phase_fifth = (phase_fifth + fifth_f / SAMPLE_RATE) % 1.0
            phase_seventh = (phase_seventh + seventh_f / SAMPLE_RATE) % 1.0

            pad = (
                _piano_note(phase_root, 0.8) +
                _piano_note(phase_third, 0.6) +
                _piano_note(phase_fifth, 0.5) +
                _piano_note(phase_seventh, 0.3)
            ) * chord_env * 0.15

            # ── Sparse muted-key motif (very soft, like distant Rhodes) ──
            motif_step = min(len(motif) - 1, int(local_phase * max(1, len(motif))))
            degree = int(motif[motif_step]) if motif else 1
            lead_note = _degree_to_midi(scale_notes, degree, octave_shift=1)
            lead_f = _midi_to_freq(lead_note) * wobble
            lead_env = _lofi_pluck_env((local_phase * len(motif)) % 1.0)
            
            phase_lead1 = (phase_lead1 + lead_f / SAMPLE_RATE) % 1.0

            lead = _piano_note(phase_lead1, 0.9) * lead_env * (0.35 + 0.05 * energy)

            # ── Warm sub-bass pulse (very subtle) ──
            sub_f = _midi_to_freq(root - 24) * wobble
            phase_sub = (phase_sub + sub_f / SAMPLE_RATE) % 1.0
            sub = 0.04 * math.sin(2 * math.pi * phase_sub) * _piano_env(local_phase)

            # ── Very soft shimmer (muted) ──
            shimmer_freq = root_f * 2.0
            phase_shimmer = (phase_shimmer + shimmer_freq / SAMPLE_RATE) % 1.0
            shimmer_sig = shimmer * 0.02 * math.sin(2 * math.pi * phase_shimmer)

            # ── Vinyl dust / crackle ──
            if rng.random() < 0.03:
                crackle = rng.uniform(-0.08, 0.08) * vinyl_dust * 10.0
            else:
                crackle = 0.0
            hiss = rng.uniform(-1.0, 1.0) * noise_amount

            left = pad + sub + lead + shimmer_sig + hiss + crackle
            right = pad + sub + lead + shimmer_sig + hiss + crackle * 0.7

            # Gentle master fade in/out
            master = _master_env(t, total_duration)
            left *= master
            right *= master

            # Soft saturation (warmer than hard clip)
            left = _soft_clip(left * 0.75)
            right = _soft_clip(right * 0.75)

            left_i = int(_clamp(left, -1.0, 1.0) * 32767)
            right_i = int(_clamp(right, -1.0, 1.0) * 32767)

            frames.extend(struct.pack("<hh", left_i, right_i))

        wav_file.writeframes(frames)

    audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
    return audio_path, audio_b64


def _slow_env(phase: float) -> float:
    """Gentle swell for lofi pads — slower attack, longer sustain."""
    phase = _clamp(phase, 0.0, 1.0)
    return 0.45 + 0.55 * math.sin(math.pi * phase)


def _piano_env(phase: float) -> float:
    """Acoustic piano envelope with sharp attack and long decay."""
    phase = _clamp(phase, 0.0, 1.0)
    attack_phase = 0.05
    if phase < attack_phase:
        return phase / attack_phase
    else:
        # Long exponential decay
        decay_phase = (phase - attack_phase) / (1.0 - attack_phase)
        return math.exp(-3.5 * decay_phase)


def _piano_note(phase: float, velocity: float) -> float:
    """Additive synthesis for a basic string/piano timbre using 4 harmonics."""
    return velocity * (
        1.00 * math.sin(2 * math.pi * phase) +
        0.35 * math.sin(4 * math.pi * phase) +
        0.15 * math.sin(6 * math.pi * phase) +
        0.05 * math.sin(8 * math.pi * phase)
    )


def _lofi_pluck_env(phase: float) -> float:
    """Soft muted-key envelope — slower attack, gentler decay than a pluck."""
    phase = _clamp(phase, 0.0, 1.0)
    attack = min(1.0, phase / 0.10)  # Slower attack for warmth
    decay = math.exp(-3.0 * phase)   # Gentler decay
    return attack * decay


def _breath_env(phase: float) -> float:
    phase = _clamp(phase, 0.0, 1.0)
    return 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * phase))


def _master_env(t: float, total_duration: float) -> float:
    fade_in = _clamp(t / 2.0, 0.0, 1.0)    # Slower fade-in
    fade_out = _clamp((total_duration - t) / 2.5, 0.0, 1.0)  # Slower fade-out
    return fade_in * fade_out


def _soft_clip(x: float) -> float:
    return math.tanh(x)


# ============================================================
# 3-PANEL IDENTITY COMIC
# ============================================================

def generate_identity_comic(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Generate a 3-panel identity comic.

    Returns:
        [{"caption": str, "image_b64": str}, ...]
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    history_lines: List[str] = []
    for msg in recent_chat or []:
        role = str(msg.get("role", "user"))
        text = _extract_text_from_msg(msg)
        text = _safe_str(text)
        if text:
            history_lines.append(f"{role}: {text}")
    history_text = "\n".join(history_lines[-6:])

    prompt = f"""
You are creating a 3-panel visual story for a young person aged 8–18.

Profile:
- Archetype: {profile['primary']}
- Style: {profile['secondary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}
- Interests: {profile['interests']}

Recent conversation:
{history_text}

Create exactly 3 panels that tell a visual micro-story of their exploration:

Panel 1 — "Curiosity Spark": The moment something caught their attention. Reference what they actually talked about.
Panel 2 — "Experimenting": Show them actively exploring, building, or trying something.
Panel 3 — "Growth Direction": A forward-looking scene showing where this curiosity could lead.

For each panel provide:
- caption: One short sentence (max 12 words). Simple, evocative.
- image_prompt: A vivid scene description for neon pixel-art. Include specific visual details.

Constraints:
- No fantasy, magic, or RPG language
- Grounded in real activities and real-world settings
- Each image_prompt should be visually distinct
""".strip()

    schema = {
        "type": "object",
        "properties": {
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "image_prompt": {"type": "string"},
                    },
                    "required": ["caption", "image_prompt"],
                },
            }
        },
        "required": ["panels"],
    }

    try:
        response = client.models.generate_content(
            model=STRUCTURED_MODEL,
            contents=prompt,
            config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        data = json.loads((response.text or "").strip())
        panels_raw = (data.get("panels") or [])[:3]
    except Exception as exc:
        logger.error("Comic panel spec generation failed: %s", exc)
        panels_raw = [
            {"caption": "Something sparked your curiosity.", "image_prompt": "A glowing spark floating above an open book in a neon-lit room"},
            {"caption": "You started experimenting.", "image_prompt": "Hands building something colorful on a workbench with neon tools"},
            {"caption": "The path keeps unfolding.", "image_prompt": "A figure walking toward a glowing horizon with scattered project ideas floating around"},
        ]

    result: List[Dict[str, str]] = []
    for panel in panels_raw:
        caption = _safe_str(panel.get("caption"), "…")
        image_prompt = _safe_str(panel.get("image_prompt"), "")
        image_b64 = ""
        if image_prompt:
            try:
                image_b64 = generate_pixel_art_illustration(image_prompt)
            except Exception:
                logger.exception("Comic panel image generation failed")
        result.append({"caption": caption, "image_b64": image_b64})

    return result


# ============================================================
# FUTURE POSTCARD
# ============================================================

def generate_future_postcard(
    user_profile: Dict[str, Any],
    recent_chat: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Generate a 'Postcard from Future You' for narrative closure.

    Returns:
        {"image_b64": str, "caption": str}
    """
    client = get_client()
    profile = _profile_summary(user_profile)

    # Generate caption
    caption_prompt = f"""
You are ArcMotivate. Write a single-sentence postcard message from a young person's future self.

Profile:
- Archetype: {profile['primary']}
- Superpower: {profile['superpower']}
- Description: {profile['description']}

Rules:
- One sentence only, max 18 words
- Warm, calm, forward-looking
- No career labels, no fantasy, no hype
- Start with "You're" or "You"
- Example: "You're still exploring — but now you know what energises you."
""".strip()

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=caption_prompt,
            config={"temperature": 0.8},
        )
        caption = (response.text or "").strip().strip('"')
    except Exception:
        logger.exception("Postcard caption failed")
        caption = "You're still exploring — but now you know what energises you."

    # Generate image
    image_prompt = (
        f"A wide cinematic pixel-art postcard scene: a young person standing at the edge of a glowing "
        f"futuristic landscape, looking forward with calm confidence. The scene reflects their archetype "
        f"as {profile['primary']}. Neon highlights, warm tones, expansive horizon, no text, no logos."
    )

    image_b64 = ""
    try:
        image_b64 = generate_pixel_art_illustration(image_prompt)
    except Exception:
        logger.exception("Postcard image failed")

    return {"image_b64": image_b64, "caption": caption}