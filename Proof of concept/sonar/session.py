"""Reading and writing recorded sessions on disk.

A session directory contains:
    meta.json   metadata sent by the watch (sample rates, chirp parameters, label)
    audio.wav   mono 16-bit PCM exactly as captured by the watch microphone
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

META_FILE = "meta.json"
AUDIO_FILE = "audio.wav"

DEFAULT_CHIRP = {"f0_hz": 18000.0, "f1_hz": 21000.0, "samples": 600, "taper_samples": 30}


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono int16 samples to a WAV file."""
    pcm = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit WAV file. Returns (float64 samples in [-1, 1], sample rate).

    Multi-channel files are reduced to their first channel.
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV files are supported")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    return pcm.astype(np.float64) / 32768.0, sample_rate


def write_meta(session_dir: Path, meta: dict) -> None:
    (session_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_session(session_dir: Path) -> tuple[np.ndarray, dict]:
    """Load a session directory. Returns (audio, meta)."""
    session_dir = Path(session_dir)
    audio_path = session_dir / AUDIO_FILE
    meta_path = session_dir / META_FILE
    if not audio_path.exists():
        raise FileNotFoundError(f"{audio_path} not found (was the session finalized?)")
    audio, wav_rate = read_wav(audio_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return audio, complete_meta(meta, wav_rate)


def complete_meta(meta: dict, wav_sample_rate: int) -> dict:
    """Fill in defaults so the analyzer also works on a bare WAV file."""
    meta = dict(meta)
    meta.setdefault("input_sample_rate", wav_sample_rate)
    meta.setdefault("output_sample_rate", meta["input_sample_rate"])
    meta.setdefault("chirp", dict(DEFAULT_CHIRP))
    meta.setdefault("emission_delay_s", 0.0)
    meta.setdefault("label", "unknown")
    return meta
