"""Phase 0 verdict: can this watch act as a finger sonar?

    python analyze.py recordings/20261112-101500_still_ab12cd34
    python analyze.py --latest 2          # the two most recent recordings
    python analyze.py capture.wav --delay 1.0

Each recording gets an `analysis.png` figure and a `report.json` next to it.
When the recordings include one labelled "still" and one labelled "moving",
they are also compared to tell whether finger motion shows up in the echo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sonar.analysis import FAIL, Check, Report, analyze, compare_motion
from sonar.plots import save_figure
from sonar.session import DEFAULT_CHIRP, complete_meta, load_session, read_wav

STATUS_TAG = {"pass": "PASS", "marginal": "WARN", "fail": "FAIL", "info": "INFO", "skipped": "SKIP"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", nargs="*", type=Path,
                        help="session directories or WAV files")
    parser.add_argument("--latest", type=int, metavar="N",
                        help="analyze the N most recent sessions in --root")
    parser.add_argument("--root", type=Path, default=Path("recordings"))
    parser.add_argument("--no-plots", action="store_true")
    bare = parser.add_argument_group("for WAV files without meta.json")
    bare.add_argument("--f0", type=float, default=DEFAULT_CHIRP["f0_hz"])
    bare.add_argument("--f1", type=float, default=DEFAULT_CHIRP["f1_hz"])
    bare.add_argument("--chirp-samples", type=int, default=DEFAULT_CHIRP["samples"])
    bare.add_argument("--delay", type=float, default=0.0,
                      help="seconds of silence before the speaker starts")
    bare.add_argument("--label", default="unknown")
    args = parser.parse_args()

    targets = list(args.recordings)
    if args.latest:
        sessions = sorted((p for p in args.root.glob("*") if (p / "audio.wav").exists()),
                          key=lambda p: p.stat().st_mtime)
        targets += sessions[-args.latest:]
    if not targets:
        parser.error("give at least one recording, or use --latest N")

    reports: list[Report] = []
    for target in targets:
        audio, meta, out_dir, prefix = _load(target, args)
        report = analyze(audio, meta)
        reports.append(report)
        _print_report(target, report)
        (out_dir / f"{prefix}report.json").write_text(
            json.dumps(report.to_json(), indent=2), encoding="utf-8")
        if not args.no_plots:
            figure = out_dir / f"{prefix}analysis.png"
            save_figure(report, audio, meta, figure)
            print(f"  figure: {figure}")

    still = next((r for r in reports if r.label == "still"), None)
    moving = next((r for r in reports if r.label == "moving"), None)
    if still and moving:
        check = compare_motion(still, moving)
        print("\nFinger motion (moving vs. still)")
        _print_check(check)
    elif len(reports) == 1 or not (still or moving):
        print("\nTo test finger sensitivity, record one session labelled 'still' and one "
              "labelled 'moving', then analyze both together.")

    return 1 if any(r.verdict == FAIL for r in reports) else 0


def _load(target: Path, args: argparse.Namespace):
    """Returns (audio, meta, output directory, output file prefix)."""
    if target.is_dir():
        audio, meta = load_session(target)
        return audio, meta, target, ""
    audio, sample_rate = read_wav(target)
    meta = complete_meta({
        "label": args.label,
        "emission_delay_s": args.delay,
        "chirp": {"f0_hz": args.f0, "f1_hz": args.f1, "samples": args.chirp_samples,
                  "taper_samples": DEFAULT_CHIRP["taper_samples"]},
    }, sample_rate)
    return audio, meta, target.parent, f"{target.stem}_"


def _print_report(target: Path, report: Report) -> None:
    print(f"\n{target}  [label: {report.label}]")
    for check in report.checks:
        _print_check(check)
    print(f"  => signal chain verdict: {report.verdict.upper()}")


def _print_check(check: Check) -> None:
    print(f"  [{STATUS_TAG[check.status]}] {check.name}: {check.summary}")


if __name__ == "__main__":
    sys.exit(main())
