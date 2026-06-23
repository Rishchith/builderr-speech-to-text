"""
Streaming Dictation Engine — draft()
=====================================
Optimised after reading STREAMING_CONTRACT.md feedback from Soham:

  KEY SCORING DIMENSIONS (from contract):
    1. Faithfulness (40pts) — keep Hindi-English mix, no translation
    2. Final latency (30pts) — clean final within ~2s of user stopping
    3. Commit churn (20pts) — stable_chars must NOT go backwards, low rewrites
    4. TTFS (10pts) — first useful partial fast

  FIXES from v1 (based on Soham's feedback):
    ❌ OLD: waited for medium model before showing ANY partial
    ✅ NEW: show small-model partials immediately (TTFS < 1s)
           medium model warms in background from first call
           final decode uses medium (faithfulness) but starts ASAP

    ❌ OLD: committed only after 2 consecutive identical prefixes (too slow)
    ✅ NEW: commit aggressively after first decode — use a 1-word safety buffer
           so stable_chars advances every decode cycle

  TIMING TARGETS:
    TTFS            < 0.8s   (first partial text visible)
    Final latency   < 2.0s   (after is_final=True)
    Churn rate      < 5%     (stable_chars never decreases)
"""

from __future__ import annotations

import re
import os
import time
import threading
from typing import Optional

# ── Audio constants ───────────────────────────────────────────────────────────
_SR               = 16_000
_BYTES_PER_SAMPLE = 2
_MIN_FIRST_BYTES  = int(_SR * 0.8) * _BYTES_PER_SAMPLE   # first partial at 0.8s
_REDECODE_BYTES   = int(_SR * 1.0) * _BYTES_PER_SAMPLE   # re-decode every 1.0s new audio

# ── Hinglish faithfulness prompt ──────────────────────────────────────────────
_HINGLISH_PROMPT = (
    "Yeh ek Hinglish conversation hai jisme Hindi aur English dono bolte hain. "
    "Transcript mein exactly wahi likho jo bola gaya, translate mat karo. "
    "Example: 'Aaj meeting mein kya discuss kiya?' not 'What was discussed today?'"
)

# ── Per-clip state ────────────────────────────────────────────────────────────
_prev_text        : str   = ""
_stable_committed : str   = ""
_last_decode_at   : int   = 0
_decode_count     : int   = 0
_fast_model             = None
_mid_model              = None
_mid_loading      : bool  = False

# ── Model loaders ─────────────────────────────────────────────────────────────

def _get_fast():
    global _fast_model
    if _fast_model is None:
        from faster_whisper import WhisperModel
        _fast_model = WhisperModel(
            "small", device="cpu", compute_type="int8",
            num_workers=1,
            cpu_threads=max(4, os.cpu_count() or 4),
        )
    return _fast_model

def _get_mid():
    global _mid_model
    if _mid_model is None:
        from faster_whisper import WhisperModel
        _mid_model = WhisperModel(
            "medium", device="cpu", compute_type="int8",
            num_workers=1,
            cpu_threads=max(4, os.cpu_count() or 4),
        )
    return _mid_model

def _warm_mid():
    """Load medium model in background — called on very first partial."""
    global _mid_loading
    if _mid_model is None and not _mid_loading:
        _mid_loading = True
        threading.Thread(target=_get_mid, daemon=True).start()

# ── Decode helpers ────────────────────────────────────────────────────────────

def _decode(audio_bytes: bytes, use_mid: bool = False) -> str:
    try:
        import numpy as np
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        model = _get_mid() if use_mid else _get_fast()
        kwargs = dict(
            language=None,
            task="transcribe",       # NEVER translate
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
            without_timestamps=True,
        )
        if use_mid:
            kwargs.update(
                initial_prompt=_HINGLISH_PROMPT,
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4],
                condition_on_previous_text=True,
            )
        else:
            kwargs.update(beam_size=3, best_of=1)

        segs, _ = model.transcribe(audio, **kwargs)
        text = " ".join(s.text for s in segs).strip()
        return _remove_loops(text)
    except Exception:
        return ""


def _remove_loops(text: str) -> str:
    """Remove Whisper hallucination repetition loops."""
    words = text.split()
    if len(words) < 8:
        return text
    for n in (4, 3):
        for i in range(len(words) - n * 2):
            chunk = words[i:i+n]
            count, pos = 0, i + n
            while pos + n <= len(words) and words[pos:pos+n] == chunk:
                count += 1; pos += n
            if count >= 2:
                return ' '.join(words[:i+n])
    return text

# ── Commitment logic (aggressive but stable) ──────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[\w'\u0900-\u097F.-]+", text, flags=re.UNICODE)

def _safe_commit_prefix(text: str, safety_words: int = 1) -> str:
    """
    Commit everything EXCEPT the last `safety_words` words.
    This is aggressive (commits most of the text each decode) but leaves
    a small tail that can still be revised as more audio arrives.
    Guarantees low churn while maximising stable_chars advancement.
    """
    if not text:
        return ""
    words = _tokenise(text)
    if len(words) <= safety_words:
        return ""
    commit_words = words[:-safety_words]
    # Reconstruct from original text to preserve spacing/punctuation
    joined = " ".join(commit_words)
    idx = text.lower().find(commit_words[-1].lower())
    if idx != -1:
        end = idx + len(commit_words[-1])
        return text[:end]
    return joined

# ── Public API ────────────────────────────────────────────────────────────────

def draft_reset() -> None:
    """Called by harness at start of each new clip."""
    global _prev_text, _stable_committed, _last_decode_at, _decode_count
    _prev_text        = ""
    _stable_committed = ""
    _last_decode_at   = 0
    _decode_count     = 0


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """
    Streaming transcription function.

    Args:
        audio_buffer : ALL audio so far — PCM s16le, mono, 16kHz.
        is_final     : True on last call (user stopped speaking).

    Returns:
        (text, stable_chars)
          text         — best current transcript (Hinglish faithful)
          stable_chars — committed prefix length (non-decreasing, no rewrites)
    """
    global _prev_text, _stable_committed, _last_decode_at, _decode_count

    # ── FINAL: use medium model for max faithfulness ──────────────────────────
    if is_final:
        # Medium model should already be warm from background loading.
        # If not loaded yet, this call blocks once (~1s) then is fast after.
        text = _decode(audio_buffer, use_mid=True)
        if not text:
            text = _prev_text or _stable_committed or ""
        _prev_text        = text
        _stable_committed = text          # commit everything on final
        _decode_count    += 1
        return (text, len(text))

    # ── PARTIAL: use fast small model, commit aggressively ───────────────────

    # Not enough audio yet
    if len(audio_buffer) < _MIN_FIRST_BYTES:
        return (_stable_committed, len(_stable_committed))

    # Enough new audio since last decode?
    new_bytes = len(audio_buffer) - _last_decode_at
    if new_bytes < _REDECODE_BYTES and _decode_count > 0:
        return (_prev_text or _stable_committed, len(_stable_committed))

    # Kick off medium model warm-up in background on very first partial
    if _decode_count == 0:
        _warm_mid()

    # Decode with fast model
    text = _decode(audio_buffer, use_mid=False)
    _decode_count   += 1
    _last_decode_at  = len(audio_buffer)

    if not text:
        return (_stable_committed, len(_stable_committed))

    # Commit all but last 1 word (aggressive advance, low churn)
    new_commit = _safe_commit_prefix(text, safety_words=1)
    # stable_chars must never decrease
    if len(new_commit) > len(_stable_committed):
        _stable_committed = new_commit

    _prev_text = text
    return (text, len(_stable_committed))
