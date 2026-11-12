"""Generate synthetic watch recordings to exercise the receiver and the analyzer.

    python simulate.py --list
    python simulate.py good-still good-moving             # write session folders
    python simulate.py good-moving --send 127.0.0.1:8765  # act as a fake watch
    python simulate.py --replay recordings/<session> --send 127.0.0.1:8765 --realtime

The scenarios are idealised models, not predictions of what an Apple Watch
will do. They exist to check that the tooling reports each situation correctly.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from sonar.session import load_session, write_meta, write_wav
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
    parser.add_argument("--replay", type=Path, metavar="SESSION",
                        help="with --send, upload this saved session instead of a scenario")
    parser.add_argument("--label", help="with --replay, label to send instead of the saved one")
    args = parser.parse_args()

    if args.replay:
        if not args.send:
            parser.error("--replay needs --send HOST:PORT")
        replay(args.send, args.replay, args.label, args.realtime)
        print(f"replayed '{args.replay}' to {args.send}")
        return

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
    """Upload a synthetic recording exactly the way the watch app does."""
    send_recording(address, scenario.meta(), samples, realtime=realtime)


def send_recording(address: str, meta: dict, samples: np.ndarray, realtime: bool = False,
                   chunk_seconds: float = 0.2) -> None:
    """Upload int16 samples with their metadata, in chunks, like the watch app."""
    base = f"http://{address}"
    session_id = meta["session_id"]
    sample_rate = int(meta["input_sample_rate"])
    _post(f"{base}/session/start", json.dumps(meta).encode("utf-8"),
          {"Content-Type": "application/json"})
    chunk = max(1, int(chunk_seconds * sample_rate))
    pcm = np.asarray(samples).astype("<i2")
    for index, offset in enumerate(range(0, len(pcm), chunk)):
        _post(f"{base}/chunk", pcm[offset: offset + chunk].tobytes(), {
            "Content-Type": "application/octet-stream",
            "X-Session-Id": session_id,
            "X-Chunk-Index": str(index),
            "X-Sample-Rate": str(sample_rate),
        })
        if realtime:
            time.sleep(chunk_seconds)
    end = {"session_id": session_id, "total_frames": int(len(pcm))}
    _post(f"{base}/session/end", json.dumps(end).encode("utf-8"),
          {"Content-Type": "application/json"})


def replay(address: str, directory: Path, label: str | None, realtime: bool) -> None:
    """Send a saved session again, as if the watch were recording it now."""
    audio, meta = load_session(directory)
    meta = {key: value for key, value in meta.items() if key not in ("received", "end")}
    meta["session_id"] = f"replay-{uuid.uuid4().hex[:12]}"
    if label:
        meta["label"] = label
    samples = np.round(audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    send_recording(address, meta, samples, realtime=realtime)


def _post(url: str, body: bytes, headers: dict) -> None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


if __name__ == "__main__":
    main()
