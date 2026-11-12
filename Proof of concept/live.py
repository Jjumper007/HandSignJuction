"""Receiver with a live view of hand openness.

    python live.py                 # then open http://localhost:8765/live

It does everything `receiver.py` does (every session is still saved as a WAV
file) and also estimates how open the hand is while the audio arrives.

Calibration comes from two saved sessions, one labelled "closed" and one
labelled "open". The most recent of each is used, and recording a new one on
the watch recalibrates immediately. To calibrate from sessions that carry
another label:

    python live.py --closed recordings/<fist session> --open recordings/<open hand session>
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from receiver import Handler, SessionStore, local_addresses, make_server
from sonar.chirp import ChirpSpec
from sonar.pose import SETTLE_S, OpennessModel, chirp_spectra, frame_spectra
from sonar.session import complete_meta, load_session

HISTORY_S = 20.0          # how much of the trace the page shows
SMOOTH_S = 0.15           # moving average applied to the openness
POINTS_PER_SECOND = 20    # trace resolution sent to the page
PAGE = Path(__file__).parent / "sonar" / "live.html"


@dataclass
class Calibration:
    model: OpennessModel
    closed_name: str
    open_name: str

    @property
    def lines(self) -> int:
        return len(self.model.closed_shape)


def load_calibration(closed_dir: Path, open_dir: Path) -> Calibration:
    closed_audio, closed_meta = load_session(closed_dir)
    open_audio, open_meta = load_session(open_dir)
    closed_db, _ = chirp_spectra(closed_audio, closed_meta)
    open_db, _ = chirp_spectra(open_audio, open_meta)
    if closed_db.shape[1] != open_db.shape[1]:
        raise ValueError("the two calibration sessions use different chirp settings")
    return Calibration(OpennessModel.fit(closed_db, open_db), closed_dir.name, open_dir.name)


def latest_session(root: Path, label: str) -> Path | None:
    """Most recent saved session with this label (folder names start with a timestamp)."""
    matches = []
    for directory in root.iterdir() if root.exists() else []:
        meta_path = directory / "meta.json"
        if not (directory / "audio.wav").exists() or not meta_path.exists():
            continue
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")).get("label") == label:
                matches.append(directory)
        except (OSError, json.JSONDecodeError):
            continue
    return max(matches, key=lambda d: d.name) if matches else None


class Stream:
    """One session while it is being received: audio in, openness points out."""

    def __init__(self, meta: dict):
        meta = complete_meta(meta, int(meta.get("input_sample_rate", 48000)))
        self.session_id = str(meta.get("session_id", ""))
        self.label = str(meta.get("label", "unknown"))
        self.spec = ChirpSpec.from_meta(meta)
        self.sample_rate = float(meta["input_sample_rate"])
        # Resampling a stream is not supported; on the watch both rates match.
        self.supported = self.sample_rate == self.spec.sample_rate
        self.to_skip = int((float(meta.get("emission_delay_s", 0.0)) + SETTLE_S) * self.sample_rate)
        self.pending = np.empty(0, dtype=np.float64)
        self.next_index = 0
        self.samples_received = 0
        self.frames_done = 0
        self.frame_rate = self.spec.sample_rate / self.spec.samples
        self.every = max(1, round(self.frame_rate / POINTS_PER_SECOND))
        self.recent: deque[float] = deque(maxlen=max(1, round(SMOOTH_S * self.frame_rate)))
        self.points: deque[tuple[float, float]] = deque()
        self.started_wall = time.time()
        self.active = True

    @property
    def seconds_received(self) -> float:
        return self.samples_received / self.sample_rate

    def feed(self, index: int, data: bytes, calibration: Calibration | None) -> None:
        if index < self.next_index:
            return                                   # a retried chunk we already have
        self.next_index = index + 1
        samples = np.frombuffer(data[: len(data) - (len(data) % 2)], dtype="<i2")
        self.samples_received += len(samples)
        if not self.supported:
            return
        if self.to_skip:
            dropped = min(self.to_skip, len(samples))
            self.to_skip -= dropped
            samples = samples[dropped:]
        self.pending = np.concatenate([self.pending, samples.astype(np.float64) / 32768.0])
        n = self.spec.samples
        frames = len(self.pending) // n
        if frames == 0:
            return
        block = self.pending[: frames * n].reshape(frames, n)
        self.pending = self.pending[frames * n:]
        spectra = frame_spectra(block, self.spec, self.sample_rate)
        usable = calibration is not None and calibration.lines == spectra.shape[1]
        scores = calibration.model.score(spectra) if usable else None
        block_end = self.seconds_received - len(self.pending) / self.sample_rate
        for i in range(frames):
            self.frames_done += 1
            if scores is None:
                continue
            self.recent.append(float(scores[i]))
            if self.frames_done % self.every == 0:
                at = block_end - (frames - 1 - i) * n / self.sample_rate
                self.points.append((at, sum(self.recent) / len(self.recent)))
        while self.points and self.points[0][0] < self.seconds_received - HISTORY_S:
            self.points.popleft()


class LiveState:
    """Everything the page shows. Safe to use from the server's worker threads."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._calibration: Calibration | None = None
        self._stream: Stream | None = None
        self._note = ""
        self._pose_dirs: dict[str, Path | None] = {"closed": None, "open": None}

    def calibrate(self, closed_dir: Path | None = None, open_dir: Path | None = None) -> None:
        """Rebuild the calibration. A given session replaces the one in use for
        that pose; a pose with no session yet falls back to the latest labelled one."""
        for label, given in (("closed", closed_dir), ("open", open_dir)):
            if given is not None:
                self._pose_dirs[label] = given
            elif self._pose_dirs[label] is None:
                self._pose_dirs[label] = latest_session(self.root, label)
        closed_dir, open_dir = self._pose_dirs["closed"], self._pose_dirs["open"]
        if closed_dir is None or open_dir is None:
            missing = [name for name, d in (("Closed fist", closed_dir), ("Open hand", open_dir))
                       if d is None]
            note, calibration = f"Not calibrated. Record on the watch: {' and '.join(missing)}.", None
        else:
            try:
                calibration, note = load_calibration(closed_dir, open_dir), ""
            except (OSError, ValueError) as error:
                calibration, note = None, f"Calibration failed: {error}"
        with self._lock:
            self._calibration, self._note = calibration, note
        if calibration is not None:
            print(f"[calibrated] closed = {calibration.closed_name}, open = {calibration.open_name}")
        else:
            print(f"[calibration] {note}")

    def session_started(self, meta: dict) -> None:
        with self._lock:
            self._stream = Stream(meta)

    def chunk(self, session_id: str, index: int, data: bytes) -> None:
        with self._lock:
            stream = self._stream
            if stream is not None and stream.session_id == session_id:
                stream.feed(index, data, self._calibration)

    def session_saved(self, directory: Path, meta: dict) -> None:
        with self._lock:
            stream = self._stream
            if stream is not None and stream.session_id == meta.get("session_id"):
                stream.active = False
        if meta.get("label") == "closed":
            self.calibrate(closed_dir=directory)
        elif meta.get("label") == "open":
            self.calibrate(open_dir=directory)

    def snapshot(self) -> dict:
        with self._lock:
            calibration, stream, note = self._calibration, self._stream, self._note
            session, points, openness = None, [], None
            if stream is not None:
                points = [[round(t, 3), round(v, 4)] for t, v in stream.points]
                openness = points[-1][1] if points else None
                delay = time.time() - stream.started_wall - stream.seconds_received
                session = {
                    "label": stream.label,
                    "active": stream.active,
                    "seconds": round(stream.seconds_received, 2),
                    "delay_s": round(max(delay, 0.0), 2) if stream.active else None,
                    "supported": stream.supported,
                }
                if calibration is not None and stream.frames_done and not points:
                    note = "The calibration does not match this session's chirp band."
        return {
            "calibrated": calibration is not None,
            "calibration": None if calibration is None else
            {"closed": calibration.closed_name, "open": calibration.open_name},
            "note": note,
            "session": session,
            "openness": openness,
            "points": points,
            "history_s": HISTORY_S,
        }


class LiveStore(SessionStore):
    """A session store that also feeds the live view."""

    def __init__(self, root: Path, live: LiveState):
        super().__init__(root)
        self.live = live
        self.on_saved = live.session_saved

    def start(self, meta: dict) -> Path:
        directory = super().start(meta)
        self.live.session_started(meta)
        return directory

    def add_chunk(self, session_id: str, index: int, sample_rate: int, data: bytes):
        session = super().add_chunk(session_id, index, sample_rate, data)
        self.live.chunk(session_id, index, data)
        return session


class LiveHandler(Handler):
    live: LiveState

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/live"):
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif path == "/live/state":
            self._send(200, "application/json", json.dumps(self.live.snapshot()).encode("utf-8"))
        else:
            super().do_GET()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_live_server(host: str, port: int, out_dir: Path):
    """Returns (server, live state)."""
    live = LiveState(out_dir)
    handler = type("BoundLiveHandler", (LiveHandler,), {"live": live})
    server = make_server(host, port, out_dir, handler_class=handler, store=LiveStore(out_dir, live))
    return server, live


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, default=Path("recordings"))
    parser.add_argument("--closed", type=Path, help="session to use as the closed-fist calibration")
    parser.add_argument("--open", type=Path, help="session to use as the open-hand calibration")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    server, live = make_live_server(args.host, args.port, args.out)
    store: SessionStore = server.RequestHandlerClass.store
    live.calibrate(args.closed, args.open)

    def watchdog() -> None:
        while True:
            time.sleep(5.0)
            store.finalize_idle()

    threading.Thread(target=watchdog, daemon=True).start()

    url = f"http://localhost:{args.port}/live"
    print(f"Live view: {url}")
    for address in local_addresses():
        print(f"  Enter this in the watch app:  {address}:{args.port}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        store.finalize_all()
        server.server_close()


if __name__ == "__main__":
    main()
