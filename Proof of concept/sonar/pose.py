"""Hand-openness estimate from two calibration poses.

Each chirp period is turned into the magnitude spectrum of the received
chirp across its band. That spectrum is the frequency response of the path
speaker -> hand -> microphone: reflections from the hand interfere with the
direct sound and carve a pose-dependent pattern into it.

The spectrum is split into its overall level (how loud the chirp arrives)
and its shape (the pattern across frequency). Only the shape is used, so the
estimate does not depend on the speaker volume.

Openness is the position of a frame's spectral shape along the line that
joins the "closed" calibration shape (0) and the "open" one (1).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal

from .chirp import ChirpSpec

SETTLE_S = 0.3            # skipped after the speaker starts
EDGE_BINS = 1             # spectral lines dropped at each end of the band, inside the taper


def chirp_spectra(audio: np.ndarray, meta: dict) -> tuple[np.ndarray, float]:
    """Per-chirp magnitude spectra in dB.

    Returns (matrix of shape frames x spectral lines, frames per second).
    The chirp repeats exactly every `samples` samples, so the spectrum of one
    period is a set of lines `sample_rate / samples` apart and its magnitude
    does not depend on where the period is cut.
    """
    spec = ChirpSpec.from_meta(meta)
    fs_in = float(meta["input_sample_rate"])
    if fs_in != spec.sample_rate:
        ratio = Fraction(int(spec.sample_rate), int(fs_in))
        audio = signal.resample_poly(audio, ratio.numerator, ratio.denominator)
    start = int((float(meta.get("emission_delay_s", 0.0)) + SETTLE_S) * spec.sample_rate)
    segment = audio[start:]
    n = spec.samples
    frames = len(segment) // n
    if frames < 20:
        raise ValueError("recording too short to analyze")
    return frame_spectra(segment[: frames * n].reshape(frames, n), spec, fs_in), spec.sample_rate / n


def frame_spectra(frames: np.ndarray, spec: ChirpSpec, fs_in: float) -> np.ndarray:
    """Magnitude spectra in dB of whole chirp periods (one period per row)."""
    n = spec.samples
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) / (n / 2)
    first, last = _line_range(spec, fs_in)
    return 20.0 * np.log10(np.maximum(spectrum[:, first: last + 1], 1e-12))


def spectral_lines_hz(meta: dict) -> np.ndarray:
    """Frequencies of the columns returned by `chirp_spectra`."""
    spec = ChirpSpec.from_meta(meta)
    first, last = _line_range(spec, float(meta["input_sample_rate"]))
    return np.arange(first, last + 1) * spec.sample_rate / spec.samples


def _line_range(spec: ChirpSpec, fs_in: float) -> tuple[int, int]:
    line_hz = spec.sample_rate / spec.samples
    first = int(np.ceil(spec.f0_hz / line_hz)) + EDGE_BINS
    last = int(np.floor(min(spec.f1_hz, 0.49 * fs_in) / line_hz)) - EDGE_BINS
    return first, last


def split_level_shape(spectra_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate each frame into its mean level (dB) and its shape around that level."""
    level = spectra_db.mean(axis=1)
    return level, spectra_db - level[:, None]


@dataclass
class OpennessModel:
    """Two calibration shapes; scores frames between them."""

    closed_shape: np.ndarray
    open_shape: np.ndarray

    @classmethod
    def fit(cls, closed_db: np.ndarray, open_db: np.ndarray) -> "OpennessModel":
        _, closed = split_level_shape(closed_db)
        _, opened = split_level_shape(open_db)
        return cls(closed.mean(axis=0), opened.mean(axis=0))

    @property
    def axis(self) -> np.ndarray:
        return self.open_shape - self.closed_shape

    def score(self, spectra_db: np.ndarray) -> np.ndarray:
        """Openness per frame: 0 at the closed calibration, 1 at the open one."""
        _, shape = split_level_shape(spectra_db)
        axis = self.axis
        return (shape - self.closed_shape) @ axis / float(axis @ axis)


def smooth(values: np.ndarray, frames: int) -> np.ndarray:
    """Moving average; the result is shorter by `frames - 1`."""
    frames = max(1, int(frames))
    return np.convolve(values, np.ones(frames) / frames, mode="valid")


def dominant_frequency(values: np.ndarray, frame_rate: float,
                       lo_hz: float = 0.15, hi_hz: float = 5.0) -> tuple[float, float]:
    """Strongest rhythm in a signal.

    Returns (frequency in Hz, share of the lo-hi band power within 0.15 Hz of it).
    A share near 1 means one steady rhythm; a low share means no clear rhythm.
    """
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    freqs = np.fft.rfftfreq(len(centered), 1.0 / frame_rate)
    band = (freqs > lo_hz) & (freqs < hi_hz)
    if not np.any(band) or power[band].sum() <= 0:
        return 0.0, 0.0
    peak = freqs[np.argmax(power * band)]
    share = power[np.abs(freqs - peak) < 0.15].sum() / power[band].sum()
    return float(peak), float(share)


def evaluate(closed_db: np.ndarray, open_db: np.ndarray, moving_db: np.ndarray,
             frame_rate: float) -> dict:
    """Run the open/closed/moving test and return its metrics and traces."""
    # Separation, tested honestly: calibrate on the first half of each static
    # recording and score the second half, which the model has not seen.
    half_c, half_o = len(closed_db) // 2, len(open_db) // 2
    held_out_model = OpennessModel.fit(closed_db[:half_c], open_db[:half_o])
    closed_scores = held_out_model.score(closed_db[half_c:])
    open_scores = held_out_model.score(open_db[half_o:])
    pooled = np.sqrt((closed_scores.var() + open_scores.var()) / 2.0)
    separation = float((open_scores.mean() - closed_scores.mean()) / max(pooled, 1e-12))

    def rms(delta: np.ndarray) -> float:
        return float(np.sqrt(np.mean(delta ** 2)))

    _, closed_shape = split_level_shape(closed_db)
    _, open_shape = split_level_shape(open_db)
    drift_closed = rms(closed_shape[:half_c].mean(0) - closed_shape[half_c:].mean(0))
    drift_open = rms(open_shape[:half_o].mean(0) - open_shape[half_o:].mean(0))
    between = rms(closed_shape.mean(0) - open_shape.mean(0))

    model = OpennessModel.fit(closed_db, open_db)
    window = max(1, int(round(0.1 * frame_rate)))
    moving_scores = smooth(model.score(moving_db), window)
    frequency, share = dominant_frequency(moving_scores, frame_rate)

    # Is the main change during the movement the same change that separates the poses?
    _, moving_shape = split_level_shape(moving_db)
    centered = moving_shape - moving_shape.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    alignment = float(abs(components[0] @ model.axis) / np.linalg.norm(model.axis))
    chance = float(1.0 / np.sqrt(moving_shape.shape[1]))

    closed_level, _ = split_level_shape(closed_db)
    open_level, _ = split_level_shape(open_db)
    return {
        "metrics": {
            "held_out_closed_mean": float(closed_scores.mean()),
            "held_out_closed_std": float(closed_scores.std()),
            "held_out_open_mean": float(open_scores.mean()),
            "held_out_open_std": float(open_scores.std()),
            "separation_d_prime": separation,
            "shape_difference_between_poses_db": between,
            "shape_drift_within_closed_db": drift_closed,
            "shape_drift_within_open_db": drift_open,
            "level_closed_dbfs": float(closed_level.mean()),
            "level_open_dbfs": float(open_level.mean()),
            "moving_openness_p02": float(np.percentile(moving_scores, 2)),
            "moving_openness_p50": float(np.percentile(moving_scores, 50)),
            "moving_openness_p98": float(np.percentile(moving_scores, 98)),
            "moving_rhythm_hz": frequency,
            "moving_rhythm_share": share,
            "movement_alignment_with_pose_axis": alignment,
            "alignment_expected_by_chance": chance,
        },
        "traces": {
            "frame_rate": frame_rate,
            "closed_shape": model.closed_shape,
            "open_shape": model.open_shape,
            "closed_scores": smooth(closed_scores, window),
            "open_scores": smooth(open_scores, window),
            "moving_scores": moving_scores,
        },
    }
