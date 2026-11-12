"""Chirp definition shared by the watch app, the simulator and the analyzer.

The Swift implementation (watch-app/SonarProbe/ChirpGenerator.swift) must stay
sample-for-sample identical to `make_chirp`. If you change one, change both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChirpSpec:
    """A linear up-chirp that is repeated back to back with no gap."""

    f0_hz: float
    f1_hz: float
    samples: int
    sample_rate: float
    taper_samples: int = 30

    @property
    def duration_s(self) -> float:
        return self.samples / self.sample_rate

    @property
    def sweep_rate_hz_per_s(self) -> float:
        return (self.f1_hz - self.f0_hz) / self.duration_s

    @classmethod
    def from_meta(cls, meta: dict) -> "ChirpSpec":
        chirp = meta["chirp"]
        return cls(
            f0_hz=float(chirp["f0_hz"]),
            f1_hz=float(chirp["f1_hz"]),
            samples=int(chirp["samples"]),
            sample_rate=float(meta["output_sample_rate"]),
            taper_samples=int(chirp.get("taper_samples", 0)),
        )


def chirp_value(spec: ChirpSpec, position: np.ndarray) -> np.ndarray:
    """Evaluate one chirp period at fractional sample positions.

    `position` is measured in output samples from the start of the period and
    must lie in [0, spec.samples). Integer positions give exactly the samples
    the watch plays; fractional positions are used to simulate delayed echoes.
    """
    position = np.asarray(position, dtype=np.float64)
    t = position / spec.sample_rate
    phase = 2.0 * np.pi * (spec.f0_hz * t + 0.5 * spec.sweep_rate_hz_per_s * t * t)
    return _taper(spec, position) * np.sin(phase)


def make_chirp(spec: ChirpSpec) -> np.ndarray:
    """Return one chirp period as float64 samples in [-1, 1]."""
    return chirp_value(spec, np.arange(spec.samples))


def _taper(spec: ChirpSpec, position: np.ndarray) -> np.ndarray:
    """Raised-cosine fade at both ends, to soften the audible click at wrap-around."""
    window = np.ones_like(position)
    m = spec.taper_samples
    if m <= 0:
        return window
    head = position < m
    window[head] = 0.5 * (1.0 - np.cos(np.pi * (position[head] + 0.5) / m))
    tail = position > spec.samples - 1 - m
    window[tail] = 0.5 * (1.0 - np.cos(np.pi * (spec.samples - position[tail] - 0.5) / m))
    return np.clip(window, 0.0, 1.0)
