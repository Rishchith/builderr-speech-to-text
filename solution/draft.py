"""
Advanced Streaming Dictation Engine — draft()
==============================================
This is the ONE function you implement for the streaming / live-dictation track.
The sealed harness calls draft() repeatedly as audio arrives, then once with
is_final=True when the user stops speaking.

HOW WE BEAT RambleFix (current benchmark: ~4.3s final latency):

  Target: final text lands ≤ 2.0s after user stops.

  Strategy:
    1. FAST PARTIAL (is_final=False):
       - Only re-decode when we have ≥ 1.5s of NEW audio since last decode.
       - Use small/int8 model for partials (speed > accuracy for drafts).
       - Commit the longest stable word prefix across two consecutive drafts.
       - This gives Time-To-First-Segment (TTFS) < 1.5s.

    2. FINAL DECODE (is_final=True):
       - Switch to medium model + Hinglish prompt for maximum faithfulness.
       - This is the text that actually gets scored on meaning accuracy.
       - Target: ≤ 2.0s for the full final decode.

    3. COMMITMENT STRATEGY (no revision churn):
       - stable_chars is non-decreasing (contract requirement).
       - We only commit a prefix once it appears identically in 2 consecutive
         partial decodes → very low churn rate.
       - On is_final=True, commit everything (stable_chars = len(text)).

SCORING LEVERS:
  • TTFS (time to first useful partial) → fast partial decode
  • End-to-final latency → medium model pre-warmed on first partial
  • Revision churn → conservative commit strategy
  • Meaning + faithfulness → medium model + Hinglish prompt on final
"""

from __future__ import annotations

import re
import time
import os
import threading

# ── Audio constants ───────────────────────────────────────────────────────────
_SR            = 16_000       # 16kHz mono PCM s16le
_BYTES_PER_SAMPLE = 2
_MIN_FIRST_BYTES  = int(_SR * 1.0) * _BYTES_PER_SAMPLE   # 1.0s before first decode
_REDECODE_BYTES   = int(_SR * 1.5) * _BYTES_PER_SAMPLE   # re-decode every 1.5s new audio

# ── Hinglish faithfulness prompt (same as transcribe.py) ─────────────────────
_HINGLISH_PROMPT = (
    "Yeh ek Hinglish conversation hai jisme Hindi aur English dono bolte hain. "
    "Transcript mein exactly wahi likho jo bola gaya, translate mat karo. "
    "For example: 'Aaj meeting mein kya discuss kiya?' not 'What was discussed?'"
)

# ── Per-clip state ────────────────────────────────────────────────────────────
_prev_text:       str   = ""
_committed:       str   = ""
_last_decode_at:  int   = 0    # byte offset of last partial decode
_decode_count:    int   = 0
_fast_model             = None
_mid_model              = None
_mid_model_loading      = False   # background warm-up flag

# ── Model accessors ───────────────────────────────────────────────────────────

def _get_fast_model():
    global _fast_model
    if _fast_model is None:
        from faster_whisper import WhisperModel
        _fast_model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=max(4, os.cpu_count() or 4),
        )
    return _fast_model


def _get_mid_model():
    global _mid_model
    if _mid_model is None:
        from faster_whisper import WhisperModel
        _mid_model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=max(4, os.cpu_count() or 4),
        )
    return _mid_model


def _warm_mid_model_background():
    """Start loading the medium model in a background thread on first partial."""
    global _mid_model_loading
    if _mid_model is None and not _mid_model_loading:
        _mid_model_loading = True
        t = threading.Thread(target=_get_mid_model, daemon=True)
        t.start()


# ── PCM → text ────────────────────────────────────────────────────────────────

def _decode_pcm(audio_buffer: bytes, use_mid: bool = False, is_final: bool = False) -> str:
    """
    Decode raw PCM bytes to text.
    use_mid=True  → medium model + Hinglish prompt (accuracy, ~1.8s/10s clip)
    use_mid=False → small model (speed, ~0.4s/10s clip)
    """
    try:
        import numpy as np
        audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""

        if use_mid:
            model = _get_mid_model()
            segments, _ = model.transcribe(
                audio,
                language=None,
                task="transcribe",
                initial_prompt=_HINGLISH_PROMPT,
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200},
                without_timestamps=True,
                condition_on_previous_text=True,
            )
        else:
            model = _get_fast_model()
            segments, _ = model.transcribe(
                audio,
                language=None,
                task="transcribe",
                beam_size=3,
                best_of=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                without_timestamps=True,
            )

        text = " ".join(s.text for s in segments).strip()
        return _remove_loops(text)

    except Exception:
        return ""


def _remove_loops(text: str) -> str:
    """Remove Whisper hallucination repetition loops."""
    if not text:
        return text
    words = text.split()
    if len(words) < 8:
        return text
    for n in (4, 3):
        for i in range(len(words) - n * 3):
            chunk = words[i:i+n]
            count = 0
            pos = i + n
            while pos + n <= len(words) and words[pos:pos+n] == chunk:
                count += 1
                pos += n
            if count >= 2:
                return ' '.join(words[:i+n])
    return text


# ── Stability / commitment logic ──────────────────────────────────────────────

def _common_word_prefix(a: str, b: str) -> str:
    """Longest common word prefix between two strings."""
    aw = _tokenise(a)
    bw = _tokenise(b)
    out: list[str] = []
    for wa, wb in zip(aw, bw):
        if wa.lower() != wb.lower():
            break
        out.append(wb)
    return " ".join(out)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[\w'\u0900-\u097F.-]+", text, flags=re.UNICODE)


def _prefix_char_len(prefix_words: str, full_text: str) -> int:
    """
    Return the character length in full_text corresponding to prefix_words.
    Ensures stable_chars points to a word boundary.
    """
    if not prefix_words:
        return 0
    # Find where prefix ends in full_text
    idx = full_text.lower().find(prefix_words.lower())
    if idx == -1:
        return 0
    return idx + len(prefix_words)


# ── Public API ────────────────────────────────────────────────────────────────

def draft_reset() -> None:
    """Called by the harness at the start of each clip. Clear per-clip state."""
    global _prev_text, _committed, _last_decode_at, _decode_count
    _prev_text      = ""
    _committed      = ""
    _last_decode_at = 0
    _decode_count   = 0


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """
    Core streaming function.

    Args:
        audio_buffer : ALL audio received so far — raw PCM s16le, mono, 16kHz.
        is_final     : True when user has stopped speaking (last call per clip).

    Returns:
        (text_so_far, stable_chars) where:
          - text_so_far  : best current transcript (Hindi+English faithful)
          - stable_chars : length of committed prefix (non-decreasing, never rewritten)
    """
    global _prev_text, _committed, _last_decode_at, _decode_count

    # ── FINAL decode — use medium model for maximum accuracy ──────────────────
    if is_final:
        # If medium model is already loaded (warmed by background thread), great.
        # Otherwise this call will load it — adds ~1s one-time cost.
        text = _decode_pcm(audio_buffer, use_mid=True, is_final=True)
        if not text:
            # Fallback: return whatever we have committed
            text = _committed or _prev_text or ""
        text = text or _committed
        _committed = text
        _prev_text = text
        return (text, len(text))

    # ── PARTIAL decode ────────────────────────────────────────────────────────

    # Don't decode until we have enough audio for a useful first partial
    if len(audio_buffer) < _MIN_FIRST_BYTES:
        return (_committed, len(_committed))

    # Only re-decode if we have enough NEW audio since last decode
    new_bytes = len(audio_buffer) - _last_decode_at
    if new_bytes < _REDECODE_BYTES and _decode_count > 0:
        return (_prev_text or _committed, len(_committed))

    # Trigger background warm-up of medium model on very first partial
    if _decode_count == 0:
        _warm_mid_model_background()

    # Decode with fast model
    text = _decode_pcm(audio_buffer, use_mid=False)
    _decode_count   += 1
    _last_decode_at  = len(audio_buffer)

    if not text:
        # Never blank-out committed prefix
        return (_committed, len(_committed))

    # ── Update commitment ─────────────────────────────────────────────────────
    # Commit the longest word prefix that was stable across prev and current decode
    stable_prefix = _common_word_prefix(_prev_text, text)
    new_commit_len = _prefix_char_len(stable_prefix, text)

    # stable_chars must be non-decreasing
    if new_commit_len >= len(_committed):
        _committed = text[:new_commit_len]

    _prev_text = text

    return (text, len(_committed))
