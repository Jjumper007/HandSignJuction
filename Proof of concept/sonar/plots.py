"""Diagnostic figure for one analyzed recording."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal

from .analysis import NEAR_BINS, Report

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"
SERIES_ON = "#2a78d6"      # speaker emitting
SERIES_OFF = "#eb6834"     # silent baseline
MAGNITUDE = LinearSegmentedColormap.from_list("magnitude", [SURFACE, SERIES_ON, "#0b2a55"])
POLARITY = LinearSegmentedColormap.from_list("polarity", [SERIES_ON, "#e6e5e1", SERIES_OFF])

BINS_BEFORE, BINS_AFTER = 20, 80   # range bins shown around the direct path


def save_figure(report: Report, audio: np.ndarray, meta: dict, path: Path) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID, "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.12, wspace=0.05, hspace=0.05)
    fig.suptitle(f"Recording '{report.label}'   verdict: {report.verdict.upper()}",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    _spectrogram(axes[0, 0], audio, meta, report)
    _spectrum(axes[0, 1], report)
    _mean_profile(axes[0, 2], report)
    _profile_image(axes[1, 0], report)
    _differential_image(axes[1, 1], report)
    _motion(axes[1, 2], report)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def _spectrogram(ax, audio: np.ndarray, meta: dict, report: Report) -> None:
    ax.set_title("Spectrogram around the speaker start")
    fs = float(meta["input_sample_rate"])
    shown = audio[: int((float(meta.get("emission_delay_s", 0.0)) + 2.0) * fs)]
    if len(shown) < 2048:
        return _unavailable(ax)
    freqs, times, power = signal.spectrogram(shown, fs=fs, nperseg=1024, noverlap=768)
    level = 10.0 * np.log10(np.maximum(power, 1e-30))
    top = np.percentile(level, 99.5)
    mesh = ax.pcolormesh(times, freqs / 1000.0, level, cmap=MAGNITUDE,
                         vmin=top - 70.0, vmax=top, shading="auto", rasterized=True)
    f0, f1 = meta["chirp"]["f0_hz"], meta["chirp"]["f1_hz"]
    ax.set_ylim(max(0.0, (f0 - 4000.0) / 1000.0), min(fs / 2.0, f1 + 2000.0) / 1000.0)
    for edge in (f0, f1):
        ax.axhline(edge / 1000.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.grid(False)
    ax.figure.colorbar(mesh, ax=ax, label="Power (dB)", pad=0.01)


def _spectrum(ax, report: Report) -> None:
    ax.set_title("Spectrum: speaker on vs. off")
    arrays = report.arrays
    if "psd_signal" not in arrays:
        return _unavailable(ax)
    khz = arrays["freqs"] / 1000.0
    lo, hi = arrays["band"]
    ax.axvspan(lo / 1000.0, hi / 1000.0, color=GRID, alpha=0.6, linewidth=0, label="Chirp band")
    ax.plot(khz, 10.0 * np.log10(np.maximum(arrays["psd_noise"], 1e-30)),
            color=SERIES_OFF, linewidth=1.6, label="Speaker off (noise)")
    ax.plot(khz, 10.0 * np.log10(np.maximum(arrays["psd_signal"], 1e-30)),
            color=SERIES_ON, linewidth=1.6, label="Speaker on")
    ax.set_xlim(max(0.0, lo / 1000.0 - 6.0), khz[-1])
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power density (dB)")
    ax.legend(frameon=False, loc="upper left")


def _mean_profile(ax, report: Report) -> None:
    ax.set_title("Average echo profile")
    arrays = report.arrays
    if "mean_profile" not in arrays:
        return _unavailable(ax)
    profile = arrays["mean_profile"]
    center = len(profile) // 2
    bins = np.arange(-BINS_BEFORE, BINS_AFTER + 1)
    level = 20.0 * np.log10(np.maximum(profile[center + bins] / profile[center], 1e-9))
    ax.axvspan(NEAR_BINS[0] * arrays["bin_cm"], NEAR_BINS[1] * arrays["bin_cm"],
               color=GRID, alpha=0.6, linewidth=0, label="Finger range")
    ax.plot(bins * arrays["bin_cm"], level, color=SERIES_ON, linewidth=1.8)
    ax.set_xlabel("Extra path length after the direct path (cm)")
    ax.set_ylabel("Level relative to direct path (dB)")
    ax.legend(frameon=False, loc="upper right")


def _profile_image(ax, report: Report) -> None:
    ax.set_title("Echo profile over time")
    arrays = report.arrays
    if "profile" not in arrays:
        return _unavailable(ax)
    image, extent = _crop(arrays["profile"], arrays["bin_cm"], report)
    level = 20.0 * np.log10(np.maximum(image / np.max(image), 1e-9))
    mesh = ax.imshow(level.T, origin="lower", aspect="auto", extent=extent, cmap=MAGNITUDE,
                     vmin=-70.0, vmax=0.0, interpolation="nearest")
    ax.set_xlabel("Time since speaker start (s)")
    ax.set_ylabel("Extra path length (cm)")
    ax.grid(False)
    ax.figure.colorbar(mesh, ax=ax, label="Level (dB)", pad=0.01)


def _differential_image(ax, report: Report) -> None:
    ax.set_title("Chirp-to-chirp change (differential profile)")
    arrays = report.arrays
    if "differential" not in arrays:
        return _unavailable(ax)
    image, extent = _crop(arrays["differential"], arrays["bin_cm"], report)
    limit = max(float(np.percentile(np.abs(image), 99.5)), 1e-9)
    mesh = ax.imshow(image.T, origin="lower", aspect="auto", extent=extent, cmap=POLARITY,
                     vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xlabel("Time since speaker start (s)")
    ax.set_ylabel("Extra path length (cm)")
    ax.grid(False)
    ax.figure.colorbar(mesh, ax=ax, label="Change relative to direct path", pad=0.01)


def _motion(ax, report: Report) -> None:
    ax.set_title("Echo change in the finger range")
    arrays = report.arrays
    if "motion_db" not in arrays:
        return _unavailable(ax)
    seconds = np.arange(len(arrays["motion_db"])) / report.metrics["frame_rate_hz"]
    ax.plot(seconds, arrays["motion_db"], color=SERIES_ON, linewidth=1.6)
    ax.set_xlabel("Time since speaker start (s)")
    ax.set_ylabel("Change relative to direct path (dB)")


def _crop(matrix: np.ndarray, bin_cm: float, report: Report):
    center = matrix.shape[1] // 2
    image = matrix[:, center - BINS_BEFORE: center + BINS_AFTER + 1]
    duration = matrix.shape[0] / report.metrics["frame_rate_hz"]
    extent = (0.0, duration, -BINS_BEFORE * bin_cm, BINS_AFTER * bin_cm)
    return image, extent


def _unavailable(ax) -> None:
    ax.text(0.5, 0.5, "Not available for this recording", transform=ax.transAxes,
            ha="center", va="center", color=TEXT_SECONDARY)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
