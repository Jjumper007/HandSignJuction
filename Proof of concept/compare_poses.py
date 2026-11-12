"""Open hand vs. closed fist: can the two be told apart, and does a recording
of the hand opening and closing move between them?

    python compare_poses.py --closed recordings/<fist> --open recordings/<open hand> \
                            --moving recordings/<opening and closing>

Writes `poses.png` and `poses.json` (use --out to change the folder).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sonar.plots import GRID, SURFACE, TEXT, TEXT_SECONDARY
from sonar.pose import chirp_spectra, evaluate, spectral_lines_hz
from sonar.session import load_session

CLOSED_COLOR = "#2a78d6"
OPEN_COLOR = "#eb6834"
MOVING_COLOR = "#1baf7a"

SEPARATION_PASS = 3.0     # d' between held-out closed and open scores
SWING_PASS = 0.5          # openness swing (2nd to 98th percentile) in the moving recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closed", type=Path, required=True, help="session with a closed fist, still")
    parser.add_argument("--open", type=Path, required=True, help="session with an open hand, still")
    parser.add_argument("--moving", type=Path, required=True, help="session opening and closing the hand")
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    spectra = {}
    for name in ("closed", "open", "moving"):
        audio, meta = load_session(getattr(args, name))
        spectra[name], frame_rate = chirp_spectra(audio, meta)
    result = evaluate(spectra["closed"], spectra["open"], spectra["moving"], frame_rate)
    metrics = result["metrics"]

    print_summary(metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "poses.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    figure = args.out / "poses.png"
    save_figure(result, spectral_lines_hz(meta), figure)
    print(f"figure: {figure}")


def print_summary(m: dict) -> None:
    swing = m["moving_openness_p98"] - m["moving_openness_p02"]
    print("Static poses (calibrated on the first half of each recording, scored on the second)")
    print(f"  closed fist : openness {m['held_out_closed_mean']:+.2f} +/- {m['held_out_closed_std']:.2f}")
    print(f"  open hand   : openness {m['held_out_open_mean']:+.2f} +/- {m['held_out_open_std']:.2f}")
    print(f"  separation  : d' = {m['separation_d_prime']:.1f}   "
          f"[{'PASS' if m['separation_d_prime'] >= SEPARATION_PASS else 'FAIL'}]")
    print(f"  spectral shape differs by {m['shape_difference_between_poses_db']:.1f} dB between poses; "
          f"it drifts {m['shape_drift_within_closed_db']:.1f} dB within the closed recording and "
          f"{m['shape_drift_within_open_db']:.1f} dB within the open one")
    print("Moving recording")
    print(f"  openness range (2nd to 98th percentile): {m['moving_openness_p02']:+.2f} to "
          f"{m['moving_openness_p98']:+.2f}   [{'PASS' if swing >= SWING_PASS else 'FAIL'}]")
    print(f"  rhythm: {m['moving_rhythm_hz']:.2f} Hz, holding {m['moving_rhythm_share']:.0%} of the "
          f"slow-movement power")
    print(f"  main change during the movement lines up with the open/closed difference: "
          f"{m['movement_alignment_with_pose_axis']:.2f} (about {m['alignment_expected_by_chance']:.2f} "
          f"would be chance, 1.00 is identical)")


def save_figure(result: dict, lines_hz: np.ndarray, path: Path) -> None:
    traces, m = result["traces"], result["metrics"]
    rate = traces["frame_rate"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID, "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    fig.suptitle("Hand openness from the chirp spectrum", fontsize=13, fontweight="bold",
                 x=0.01, ha="left")

    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("Spectral shape of each static pose")
    ax.plot(lines_hz / 1000.0, traces["closed_shape"], color=CLOSED_COLOR, linewidth=1.8,
            label="Closed fist")
    ax.plot(lines_hz / 1000.0, traces["open_shape"], color=OPEN_COLOR, linewidth=1.8,
            label="Open hand")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Level relative to the frame mean (dB)")
    ax.legend(frameon=False, loc="lower left")

    ax = fig.add_subplot(grid[0, 1])
    ax.set_title(f"Static poses scored on unseen halves (d' = {m['separation_d_prime']:.1f})")
    for key, color, label in (("closed_scores", CLOSED_COLOR, "Closed fist"),
                              ("open_scores", OPEN_COLOR, "Open hand")):
        values = traces[key]
        ax.plot(np.arange(len(values)) / rate, values, color=color, linewidth=1.6, label=label)
    ax.set_xlabel("Time within the unseen half (s)")
    ax.set_ylabel("Openness (0 = closed, 1 = open)")
    ax.legend(frameon=False, loc="center right")

    ax = fig.add_subplot(grid[1, :])
    ax.set_title(f"Hand opening and closing (rhythm {m['moving_rhythm_hz']:.2f} Hz)")
    moving = traces["moving_scores"]
    seconds = np.arange(len(moving)) / rate
    for level, color, label in ((0.0, CLOSED_COLOR, "closed calibration"),
                                (1.0, OPEN_COLOR, "open calibration")):
        ax.axhline(level, color=color, linewidth=1.2, linestyle="--")
        ax.annotate(label, xy=(1.0, level), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points",
                    va="bottom", ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.plot(seconds, moving, color=MOVING_COLOR, linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Openness (0 = closed, 1 = open)")

    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
