"""Synthetic watch recordings, used to test the receiver and the analyzer
before any real hardware is available.

The model is deliberately simple: the microphone hears the speaker directly
(the strongest path), a few static reflections (wrist, arm, body) and
optionally one weak moving reflection that stands in for the fingers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal

from .chirp import ChirpSpec, chirp_value


@dataclass
class Scenario:
    """Parameters of one simulated recording."""

    name: str = "good"
    label: str = "still"
    input_sample_rate: int = 48000
    output_sample_rate: int = 48000
    duration_s: float = 8.0
    emission_delay_s: float = 1.0
    f0_hz: float = 18000.0
    f1_hz: float = 21000.0
    chirp_samples: int = 600
    taper_samples: int = 30
    # Direct speaker-to-microphone path through the watch case.
    direct_amplitude: float = 0.25
    direct_delay_samples: float = 6.0
    # (delay in samples, amplitude) of reflections that do not move.
    static_echoes: list[tuple[float, float]] = field(
        default_factory=lambda: [(14.0, 0.02), (33.0, 0.01), (80.0, 0.004)]
    )
    # One reflection whose delay oscillates, standing in for moving fingers.
    moving: bool = False
    moving_amplitude: float = 0.006
    moving_delay_center: float = 30.0
    moving_delay_swing: float = 6.0
    moving_rate_hz: float = 1.5
    # Sensor noise (white) and ambient sound (below 4 kHz).
    noise_rms: float = 3e-4
    processing_noise_rms: float = 5e-5
    ambient_rms: float = 0.02
    # Optional impairments.
    speaker_rolloff_hz: float | None = None   # gentle loss of output above this frequency
    capture_lowpass_hz: float | None = None   # the OS filters the microphone above this frequency
    dropouts: list[tuple[float, int]] = field(default_factory=list)  # (time in s, samples lost)
    seed: int = 0

    def chirp_spec(self) -> ChirpSpec:
        return ChirpSpec(
            f0_hz=self.f0_hz,
            f1_hz=self.f1_hz,
            samples=self.chirp_samples,
            sample_rate=float(self.output_sample_rate),
            taper_samples=self.taper_samples,
        )

    def meta(self) -> dict:
        return {
            "protocol": 1,
            "session_id": f"sim-{self.name}",
            "label": self.label,
            "input_sample_rate": self.input_sample_rate,
            "input_channels": 1,
            "output_sample_rate": self.output_sample_rate,
            "sample_format": "int16le",
            "chirp": {
                "f0_hz": self.f0_hz,
                "f1_hz": self.f1_hz,
                "samples": self.chirp_samples,
                "taper_samples": self.taper_samples,
                "amplitude": 0.8,
            },
            "emission_delay_s": self.emission_delay_s,
            "audio_session_mode": "simulated",
            "device": {"model": "simulator", "system": "none"},
        }


SCENARIOS: dict[str, Scenario] = {
    "good-still": Scenario(name="good-still", label="still"),
    "good-moving": Scenario(name="good-moving", label="moving", moving=True),
    "rolloff": Scenario(name="rolloff", label="still", speaker_rolloff_hz=19000.0),
    "filtered-16k": Scenario(name="filtered-16k", label="still", capture_lowpass_hz=16000.0),
    "rate-16k": Scenario(name="rate-16k", label="still", input_sample_rate=16000),
    "rate-44k": Scenario(
        name="rate-44k", label="moving", moving=True,
        input_sample_rate=44100, output_sample_rate=44100, chirp_samples=551,
    ),
    "dropouts": Scenario(name="dropouts", label="still", dropouts=[(3.0, 137), (5.5, 412)]),
    "silent-speaker": Scenario(name="silent-speaker", label="still", direct_amplitude=0.0,
                               static_echoes=[]),
    # A stand-in for hand poses: one echo whose delay differs by less than half
    # a wavelength between "closed" and "open", so the spectrum changes smoothly.
    "pose-closed": Scenario(name="pose-closed", label="closed", moving=True, moving_amplitude=0.05,
                            moving_delay_center=30.0, moving_delay_swing=0.0),
    "pose-open": Scenario(name="pose-open", label="open", moving=True, moving_amplitude=0.05,
                          moving_delay_center=30.8, moving_delay_swing=0.0),
    "pose-live": Scenario(name="pose-live", label="live", moving=True, moving_amplitude=0.05,
                          moving_delay_center=30.4, moving_delay_swing=0.4, moving_rate_hz=0.5,
                          duration_s=20.0),
}


def synthesize(scenario: Scenario) -> np.ndarray:
    """Return the simulated microphone signal as int16 samples at the input rate."""
    rng = np.random.default_rng(scenario.seed)
    spec = scenario.chirp_spec()
    fs_out = scenario.output_sample_rate
    total = int(round(scenario.duration_s * fs_out))
    start = int(round(scenario.emission_delay_s * fs_out))
    n = np.arange(total - start, dtype=np.float64)

    emitted = np.zeros(total)
    paths = [(scenario.direct_delay_samples, scenario.direct_amplitude)]
    paths += list(scenario.static_echoes)
    for delay, amplitude in paths:
        emitted[start:] += amplitude * _delayed_chirp_train(spec, n, delay)
    if scenario.moving:
        t = n / fs_out
        delay = scenario.moving_delay_center + scenario.moving_delay_swing * np.sin(
            2.0 * np.pi * scenario.moving_rate_hz * t
        )
        emitted[start:] += scenario.moving_amplitude * _delayed_chirp_train(spec, n, delay)

    if scenario.speaker_rolloff_hz is not None:
        sos = signal.butter(4, scenario.speaker_rolloff_hz, btype="lowpass", fs=fs_out, output="sos")
        emitted = signal.sosfilt(sos, emitted)

    captured = emitted + scenario.noise_rms * rng.standard_normal(total)
    ambient_sos = signal.butter(4, 4000.0, btype="lowpass", fs=fs_out, output="sos")
    ambient = signal.sosfilt(ambient_sos, rng.standard_normal(total))
    captured += scenario.ambient_rms * ambient / (np.std(ambient) + 1e-12)

    if scenario.capture_lowpass_hz is not None:
        # A steep filter such as a voice-processing chain or a lossy codec would apply.
        sos = signal.ellip(10, 0.5, 90.0, scenario.capture_lowpass_hz, btype="lowpass",
                           fs=fs_out, output="sos")
        captured = signal.sosfilt(sos, captured)
    # Noise added after any OS processing (converter and codec noise).
    captured += scenario.processing_noise_rms * rng.standard_normal(total)

    if scenario.input_sample_rate != fs_out:
        ratio = Fraction(scenario.input_sample_rate, fs_out)
        captured = signal.resample_poly(captured, ratio.numerator, ratio.denominator)

    for time_s, lost in sorted(scenario.dropouts, reverse=True):
        at = int(time_s * scenario.input_sample_rate)
        captured = np.delete(captured, slice(at, at + lost))

    return np.clip(np.round(captured * 32767.0), -32768, 32767).astype(np.int16)


def _delayed_chirp_train(spec: ChirpSpec, n: np.ndarray, delay) -> np.ndarray:
    """The endlessly repeating chirp, delayed by a (possibly varying) number of samples."""
    return chirp_value(spec, np.mod(n - delay, spec.samples))
