"""Phase 0 feasibility analysis of a watch recording.

Answers three questions about one recording:
  1. Does the chirp band survive the speaker -> air -> microphone -> OS chain?
  2. Is the capture clean enough to build echo profiles (sharp direct-path
     peak, no dropped samples)?
  3. How much does the near-range echo change from chirp to chirp? Comparing
     this between a "still" and a "moving" recording tells whether finger
     motion is visible at all.

The thresholds below are engineering starting points. They have NOT been
calibrated on a real Apple Watch; revisit them once real recordings exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import fft, signal

from . import SPEED_OF_SOUND_M_S
from .chirp import ChirpSpec, make_chirp

# --- Thresholds -------------------------------------------------------------
BAND_SNR_PASS_DB = 20.0        # in-band power while emitting vs. silent baseline
BAND_SNR_MARGINAL_DB = 10.0
COVERAGE_BIN_DB = 10.0         # a frequency bin counts as usable above this SNR
COVERAGE_PASS = 0.6            # fraction of the chirp band that must be usable
COVERAGE_MARGINAL = 0.3
FLATNESS_PASS = 0.6            # effective bandwidth as a fraction of the nominal band
PEAK_PASS_DB = 20.0            # matched-filter direct-path peak over the profile floor
PEAK_MARGINAL_DB = 12.0
GLITCH_PASS = 0.02             # fraction of frames whose direct-path peak is misplaced
CLIP_LIMIT = 1e-3              # fraction of samples at full scale
MOTION_PASS_DB = 6.0           # "moving" vs "still" near-range differential energy
MOTION_MARGINAL_DB = 3.0

# --- Analysis geometry ------------------------------------------------------
SETTLE_S = 0.3                 # skipped after the speaker starts
NEAR_BINS = (8, 60)            # range bins after the direct path that fingers can reach
PEAK_TOLERANCE_BINS = 1        # allowed frame-to-frame wobble of the direct-path peak

PASS, MARGINAL, FAIL, INFO, SKIPPED = "pass", "marginal", "fail", "info", "skipped"


@dataclass
class Check:
    name: str
    status: str
    summary: str


@dataclass
class Report:
    label: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)   # intermediate data for the plots
    verdict: str = FAIL

    def add(self, name: str, status: str, summary: str) -> None:
        self.checks.append(Check(name, status, summary))

    def status_of(self, name: str) -> str | None:
        return next((c.status for c in self.checks if c.name == name), None)

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "verdict": self.verdict,
            "checks": [vars(c) for c in self.checks],
            "metrics": self.metrics,
        }


def analyze(audio: np.ndarray, meta: dict) -> Report:
    """Run every Phase 0 check on one recording."""
    report = Report(label=str(meta.get("label", "unknown")))
    spec = ChirpSpec.from_meta(meta)
    fs_in = float(meta["input_sample_rate"])
    delay_s = float(meta.get("emission_delay_s", 0.0))

    _check_levels(report, audio, fs_in)
    band_ok = _check_sample_rate(report, spec, fs_in)

    emitting = audio[int((delay_s + SETTLE_S) * fs_in): len(audio) - int(0.1 * fs_in)]
    if len(emitting) < fs_in:
        report.add("recording_length", FAIL,
                   "Less than 1 s of audio after the speaker starts; record for longer.")
        report.verdict = FAIL
        return report

    if band_ok:
        baseline = _baseline_segment(audio, fs_in, delay_s)
        _check_band(report, emitting, baseline, spec, fs_in)
        _check_echo_profile(report, audio, spec, fs_in, delay_s)
    report.verdict = _overall_verdict(report)
    return report


def compare_motion(still: Report, moving: Report) -> Check:
    """Is the near-range echo livelier when the fingers move than when they rest?"""
    key = "near_motion_db"
    if key not in still.metrics or key not in moving.metrics:
        return Check("finger_motion", SKIPPED,
                     "One of the recordings has no echo profile to compare.")
    delta = moving.metrics[key] - still.metrics[key]
    text = (f"Near-range echo change is {delta:+.1f} dB stronger with moving fingers "
            f"(moving {moving.metrics[key]:.1f} dB, still {still.metrics[key]:.1f} dB "
            f"relative to the direct path).")
    if delta >= MOTION_PASS_DB:
        return Check("finger_motion", PASS, text)
    if delta >= MOTION_MARGINAL_DB:
        return Check("finger_motion", MARGINAL, text)
    return Check("finger_motion", FAIL, text + " Finger motion is not visible in the echo.")


# --- Individual checks ------------------------------------------------------

def _check_levels(report: Report, audio: np.ndarray, fs_in: float) -> None:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    clipped = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0.0
    report.metrics.update(
        duration_s=len(audio) / fs_in,
        input_sample_rate=fs_in,
        peak_dbfs=_db20(peak),
        rms_dbfs=_db20(rms),
        clipped_fraction=clipped,
    )
    if peak == 0.0:
        report.add("levels", FAIL, "The recording is digital silence; the microphone delivered no data.")
    elif clipped > CLIP_LIMIT:
        report.add("levels", MARGINAL,
                   f"{clipped:.2%} of samples are clipped; lower the watch volume.")
    else:
        report.add("levels", PASS,
                   f"Peak {_db20(peak):.1f} dBFS, RMS {_db20(rms):.1f} dBFS, no clipping.")


def _check_sample_rate(report: Report, spec: ChirpSpec, fs_in: float) -> bool:
    nyquist = fs_in / 2.0
    if nyquist <= spec.f0_hz + 500.0:
        report.add("sample_rate", FAIL,
                   f"The microphone runs at {fs_in:.0f} Hz, so nothing above {nyquist:.0f} Hz is "
                   f"captured. The {spec.f0_hz:.0f}-{spec.f1_hz:.0f} Hz chirp cannot be recorded.")
        return False
    if nyquist < spec.f1_hz:
        report.add("sample_rate", MARGINAL,
                   f"The microphone runs at {fs_in:.0f} Hz; only {spec.f0_hz:.0f}-{nyquist:.0f} Hz "
                   f"of the chirp band can be captured.")
    else:
        report.add("sample_rate", PASS,
                   f"The microphone runs at {fs_in:.0f} Hz, enough for the "
                   f"{spec.f0_hz:.0f}-{spec.f1_hz:.0f} Hz chirp.")
    return True


def _baseline_segment(audio: np.ndarray, fs_in: float, delay_s: float) -> np.ndarray | None:
    """Audio recorded before the speaker starts, used as the noise reference."""
    lo, hi = int(0.2 * fs_in), int((delay_s - 0.15) * fs_in)
    return audio[lo:hi] if hi - lo >= int(0.25 * fs_in) else None


def _check_band(report: Report, emitting: np.ndarray, baseline: np.ndarray | None,
                spec: ChirpSpec, fs_in: float) -> None:
    nperseg = 4096
    freqs, psd_signal = signal.welch(emitting, fs=fs_in, nperseg=nperseg)
    band_hi = min(spec.f1_hz, 0.98 * fs_in / 2.0)
    band = (freqs >= spec.f0_hz) & (freqs <= band_hi)

    if baseline is not None:
        _, psd_noise = signal.welch(baseline, fs=fs_in, nperseg=min(nperseg, len(baseline)))
        if len(psd_noise) != len(psd_signal):
            psd_noise = np.interp(freqs, np.linspace(0, fs_in / 2, len(psd_noise)), psd_noise)
        reference = "silent baseline"
    else:
        # No silent lead-in: use the band just below the chirp as the noise estimate.
        below = (freqs >= spec.f0_hz - 3500.0) & (freqs <= spec.f0_hz - 500.0)
        psd_noise = np.full_like(psd_signal, np.mean(psd_signal[below]))
        reference = "band below the chirp"

    smooth = max(1, int(200.0 / (freqs[1] - freqs[0])))
    kernel = np.ones(smooth) / smooth
    signal_smooth = np.convolve(psd_signal, kernel, mode="same")
    noise_smooth = np.convolve(psd_noise, kernel, mode="same")
    ratio_db = _db10(signal_smooth / np.maximum(noise_smooth, 1e-30))

    snr_db = _db10(np.mean(psd_signal[band]) / max(np.mean(psd_noise[band]), 1e-30))
    usable = band & (ratio_db >= COVERAGE_BIN_DB)
    coverage = float(np.sum(usable) / max(np.sum(band), 1))
    usable_lo, usable_hi = _longest_run(freqs, usable)

    edge = 500.0
    low_edge = np.mean(psd_signal[(freqs >= spec.f0_hz) & (freqs <= spec.f0_hz + edge)])
    high_edge = np.mean(psd_signal[(freqs >= band_hi - edge) & (freqs <= band_hi)])
    rolloff_db = _db10(high_edge / max(low_edge, 1e-30))

    # Effective bandwidth: equals the full band for a flat response and shrinks
    # when the speaker or microphone favours one end. It sets the range resolution.
    excess = np.maximum(signal_smooth[band] - noise_smooth[band], 0.0)
    bin_hz = freqs[1] - freqs[0]
    effective_hz = float(bin_hz * np.sum(excess) ** 2 / max(np.sum(excess ** 2), 1e-60))
    nominal_hz = band_hi - spec.f0_hz
    resolution_cm = 100.0 * SPEED_OF_SOUND_M_S / (2.0 * max(effective_hz, 1.0))

    report.metrics.update(
        band_snr_db=snr_db, band_coverage=coverage, usable_band_hz=[usable_lo, usable_hi],
        band_rolloff_db=rolloff_db, noise_reference=reference,
        effective_bandwidth_hz=effective_hz, range_resolution_cm=resolution_cm,
    )
    report.arrays.update(freqs=freqs, psd_signal=signal_smooth, psd_noise=noise_smooth,
                         band=(spec.f0_hz, band_hi))

    text = f"Chirp band is {snr_db:.1f} dB above the {reference}."
    if snr_db >= BAND_SNR_PASS_DB:
        report.add("band_snr", PASS, text)
    elif snr_db >= BAND_SNR_MARGINAL_DB:
        report.add("band_snr", MARGINAL, text + " Weak: raise the volume or try a lower band.")
    else:
        report.add("band_snr", FAIL, text + " The chirp is not reaching the recording.")

    text = f"{coverage:.0%} of the band is usable ({usable_lo:.0f}-{usable_hi:.0f} Hz)."
    status = PASS if coverage >= COVERAGE_PASS else MARGINAL if coverage >= COVERAGE_MARGINAL else FAIL
    report.add("band_coverage", status, text)

    if snr_db >= BAND_SNR_MARGINAL_DB:
        text = (f"Effective bandwidth {effective_hz:.0f} Hz of {nominal_hz:.0f} Hz "
                f"(range resolution about {resolution_cm:.1f} cm); top of the band is "
                f"{rolloff_db:+.1f} dB relative to the bottom.")
        if effective_hz >= FLATNESS_PASS * nominal_hz:
            report.add("band_flatness", PASS, text)
        else:
            report.add("band_flatness", MARGINAL,
                       text + " The response is strongly tilted; the chirp will need "
                              "equalisation or a different band.")


def _check_echo_profile(report: Report, audio: np.ndarray, spec: ChirpSpec,
                        fs_in: float, delay_s: float) -> None:
    profile = echo_profile(audio, spec, fs_in, delay_s)
    if profile is None:
        report.add("direct_path", FAIL, "Too little audio to build an echo profile.")
        return
    frames, center = profile.shape[0], spec.samples // 2
    peak_index = np.argmax(profile, axis=1) - center
    peak_amplitude = np.max(profile, axis=1)
    good = np.abs(peak_index) <= PEAK_TOLERANCE_BINS
    mean_profile = np.median(profile[good] if np.any(good) else profile, axis=0)
    peak_db = _db20(mean_profile[center] / max(np.median(mean_profile), 1e-30))
    glitch_fraction = float(1.0 - np.mean(good))
    jumps = int(np.sum(np.abs(np.diff(peak_index)) > PEAK_TOLERANCE_BINS))
    direct_amplitude = float(np.median(peak_amplitude))
    amplitude_cv = float(np.std(peak_amplitude[good]) / direct_amplitude) if np.any(good) else float("nan")

    report.metrics.update(
        frames=frames, frame_rate_hz=spec.sample_rate / spec.samples,
        direct_peak_db=peak_db, direct_amplitude_dbfs=_db20(direct_amplitude),
        glitch_fraction=glitch_fraction, timing_jumps=jumps, direct_amplitude_cv=amplitude_cv,
    )
    report.arrays.update(profile=profile, mean_profile=mean_profile, peak_index=peak_index,
                         bin_cm=100.0 * SPEED_OF_SOUND_M_S / spec.sample_rate)

    text = f"Direct speaker-to-microphone peak is {peak_db:.1f} dB above the profile floor."
    if peak_db >= PEAK_PASS_DB:
        report.add("direct_path", PASS, text)
    elif peak_db >= PEAK_MARGINAL_DB:
        report.add("direct_path", MARGINAL, text + " The matched filter barely locks on.")
    else:
        report.add("direct_path", FAIL, text + " The matched filter does not find the chirp.")
        return

    text = (f"{glitch_fraction:.1%} of {frames} frames have a misplaced peak, "
            f"with {jumps} timing jump(s).")
    if glitch_fraction <= GLITCH_PASS:
        report.add("timing", PASS, text)
    else:
        report.add("timing", MARGINAL,
                   text + " Samples are being dropped or the audio clocks drift; "
                          "per-frame re-alignment will be needed.")

    _measure_motion(report, profile, good, direct_amplitude, spec)


def echo_profile(audio: np.ndarray, spec: ChirpSpec, fs_in: float,
                 delay_s: float) -> np.ndarray | None:
    """Matched-filter the recording into a (frames x range bins) envelope matrix.

    Each row is one chirp period. The direct speaker-to-microphone path is
    placed at the centre column, so columns to its right are later echoes.
    """
    fs_out = spec.sample_rate
    if fs_in != fs_out:
        ratio = Fraction(int(fs_out), int(fs_in))
        audio = signal.resample_poly(audio, ratio.numerator, ratio.denominator)
    band_hi = min(spec.f1_hz, 0.49 * min(fs_in, fs_out))
    sos = signal.butter(5, [spec.f0_hz, band_hi], btype="bandpass", fs=fs_out, output="sos")
    filtered = signal.sosfiltfilt(sos, audio)

    template = make_chirp(spec)
    segment = filtered[int((delay_s + SETTLE_S) * fs_out):]
    if len(segment) < 22 * spec.samples:
        return None
    correlation = signal.correlate(segment, template, mode="valid", method="fft")
    correlation /= np.sum(template ** 2)
    padded = fft.next_fast_len(len(correlation))
    envelope = np.abs(signal.hilbert(correlation, N=padded))[: len(correlation)]

    n = spec.samples
    rough = envelope[: (len(envelope) // n) * n].reshape(-1, n)
    # Align on the most common peak position, so that a recording with dropped
    # samples still lines up on its longest intact stretch.
    direct = int(np.argmax(np.bincount(np.argmax(rough, axis=1), minlength=n)))
    envelope = envelope[(direct - n // 2) % n:]
    return envelope[: (len(envelope) // n) * n].reshape(-1, n)


def _measure_motion(report: Report, profile: np.ndarray, good: np.ndarray,
                    direct_amplitude: float, spec: ChirpSpec) -> None:
    """Chirp-to-chirp change of the echo, near the watch and far from it."""
    center, n = spec.samples // 2, spec.samples
    differential = np.diff(profile, axis=0) / direct_amplitude
    # A frame next to a misplaced one may itself be partly corrupted, so drop it too.
    bad = ~good
    clean = ~(bad | np.roll(bad, 1) | np.roll(bad, -1))
    valid = clean[1:] & clean[:-1]
    near = slice(center + NEAR_BINS[0], min(center + NEAR_BINS[1] + 1, n))
    far = slice(center - int(0.43 * n), center - int(0.10 * n))
    near_power = np.mean(differential[:, near] ** 2, axis=1)
    far_power = np.mean(differential[:, far] ** 2, axis=1)
    if not np.any(valid):
        report.add("echo_change", SKIPPED, "No clean consecutive frames to compare.")
        return

    near_db = _db10(np.mean(near_power[valid]))
    far_db = _db10(np.mean(far_power[valid]))
    window = max(1, int(round(0.25 * spec.sample_rate / n)))
    smoothed = np.convolve(np.where(valid, near_power, 0.0), np.ones(window) / window, mode="same")
    smoothed_db = _db10(np.maximum(smoothed, 1e-30))
    smoothed_db[: window // 2] = np.nan          # the moving average is incomplete at the ends
    smoothed_db[len(smoothed_db) - window // 2:] = np.nan
    dynamic_range_db = float(np.nanpercentile(smoothed_db, 90) - np.nanpercentile(smoothed_db, 10))

    report.metrics.update(near_motion_db=near_db, far_motion_db=far_db,
                          motion_dynamic_range_db=dynamic_range_db)
    report.arrays.update(differential=differential, motion_db=smoothed_db, near=near)
    report.add("echo_change", INFO,
               f"Near-range echo changes by {near_db:.1f} dB relative to the direct path "
               f"(far range {far_db:.1f} dB). Compare a 'still' and a 'moving' recording "
               f"to judge finger sensitivity.")


def _overall_verdict(report: Report) -> str:
    blocking = ("levels", "sample_rate", "recording_length", "band_snr", "direct_path")
    statuses = {c.name: c.status for c in report.checks}
    if any(statuses.get(name) == FAIL for name in blocking):
        return FAIL
    if any(status in (MARGINAL, FAIL) for status in statuses.values()):
        return MARGINAL
    return PASS


# --- Helpers ----------------------------------------------------------------

def _longest_run(freqs: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Frequency limits of the longest contiguous run of True in `mask`."""
    best, start = (0, 0), None
    for i, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] == best[0]:
        return 0.0, 0.0
    return float(freqs[best[0]]), float(freqs[best[1] - 1])


def _db10(value) -> float:
    return float(10.0 * np.log10(np.maximum(value, 1e-30))) if np.ndim(value) == 0 \
        else 10.0 * np.log10(np.maximum(value, 1e-30))


def _db20(value) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-15)))
