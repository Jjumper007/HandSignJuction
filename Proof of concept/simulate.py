"""Generate synthetic watch recordings to exercise the receiver and the analyzer.

    python simulate.py --list
    python simulate.py good-still good-moving             # write session folders
    python simulate.py good-moving --send 127.0.0.1:8765  # act as a fake watch

The scenarios are idealised models, not predictions of what an Apple Watch
will do. They exist to check that the tooling reports each situation correctly.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from sonar.session import write_meta, write_wav
from sonar.synth import SCENARIOS, Scenario, synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="*", help="scenario names (see --list)")
    parser.add_argument("--list", action="store_true", help="list the available scenarios")
    parser.add_argument("--out", type=Path, default=Path("recordings"),
                        help="where session folders are written")
    parser.add_argument("--send", metavar="HOST:PORT",
                        help="upload to a running receiver instead of writing files")
    parser.add_argument("--realtime", action="store_true",
                        help="with --send, pace the chunks like a real watch")
    args = parser.parse_args()

    if args.list or not args.scenarios:
        for name, scenario in SCENARIOS.items():
            print(f"{name:16s} label={scenario.label}")
        return

    for name in args.scenarios:
        if name not in SCENARIOS:
            parser.error(f"unknown scenario '{name}' (see --list)")
        scenario = SCENARIOS[name]
        samples = synthesize(scenario)
        if args.send:
            send_session(args.send, scenario, samples, realtime=args.realtime)
            print(f"sent '{name}' to {args.send}")
        else:
            directory = write_session(args.out, scenario, samples)
            print(f"wrote {directory}")


def write_session(root: Path, scenario: Scenario, samples: np.ndarray) -> Path:
    directory = root / f"sim_{scenario.name}"
    directory.mkdir(parents=True, exist_ok=True)
    write_wav(directory / "audio.wav", samples, scenario.input_sample_rate)
    write_meta(directory, scenario.meta())
    return directory


def send_session(address: str, scenario: Scenario, samples: np.ndarray,
                 realtime: bool = False) -> None:
    """Upload a recording exactly the way the watch app does."""
    base = f"http://{address}"
    meta = scenario.meta()
    session_id = meta["session_id"]
    _post(f"{base}/session/start", json.dumps(meta).encode("utf-8"),
          {"Content-Type": "application/json"})
    chunk = scenario.input_sample_rate   # one second per chunk
    pcm = samples.astype("<i2")
    count = 0
    for index, offset in enumerate(range(0, len(pcm), chunk)):
        _post(f"{base}/chunk", pcm[offset: offset + chunk].tobytes(), {
            "Content-Type": "application/octet-stream",
            "X-Session-Id": session_id,
            "X-Chunk-Index": str(index),
            "X-Sample-Rate": str(scenario.input_sample_rate),
        })
        count += 1
        if realtime:
            time.sleep(1.0)
    end = {"session_id": session_id, "total_chunks": count, "total_frames": int(len(pcm))}
    _post(f"{base}/session/end", json.dumps(end).encode("utf-8"),
          {"Content-Type": "application/json"})


def _post(url: str, body: bytes, headers: dict) -> None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


if __name__ == "__main__":
    main()
