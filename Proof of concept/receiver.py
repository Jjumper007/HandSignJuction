"""Receives audio from the watch app over HTTP and stores each session as a WAV file.

Uses only the Python standard library, so it runs on any machine:

    python receiver.py                # listens on port 8765
    python receiver.py --port 9000 --out recordings

Protocol (version 1), all requests are plain HTTP:
    GET  /ping            health check, used by the watch's "Test connection" button
    POST /session/start   JSON metadata; must contain "session_id"
    POST /chunk           raw mono int16 little-endian PCM
                          headers: X-Session-Id, X-Chunk-Index, X-Sample-Rate
    POST /session/end     JSON; must contain "session_id"

Security note: there is no authentication. Run it only on a network you trust
(a phone hotspot is ideal) and stop it when you are done.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import sys
import threading
import time
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sonar import PROTOCOL_VERSION

MAX_BODY_BYTES = 8 * 1024 * 1024
IDLE_TIMEOUT_S = 20.0
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class Session:
    def __init__(self, directory: Path, meta: dict):
        self.directory = directory
        self.meta = meta
        self.chunk_dir = directory / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.indices: set[int] = set()
        self.byte_count = 0
        self.last_activity = time.monotonic()


class SessionStore:
    """Keeps open sessions on disk and turns them into WAV files when they end."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        # Optional hook, called with (directory, metadata) after a session is saved.
        self.on_saved = None

    def start(self, meta: dict) -> Path:
        session_id = _valid_session_id(meta.get("session_id"))
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._create(session_id, meta)
            else:
                session.meta.update(meta)
            _write_json(session.directory / "meta.json", session.meta)
            return session.directory

    def add_chunk(self, session_id: str, index: int, sample_rate: int, data: bytes) -> Session:
        session_id = _valid_session_id(session_id)
        if index < 0:
            raise ValueError("invalid or missing chunk index")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                # The start request was lost; keep the audio anyway.
                session = self._create(session_id, {"session_id": session_id,
                                                    "input_sample_rate": sample_rate,
                                                    "label": "recovered"})
                _write_json(session.directory / "meta.json", session.meta)
            if index not in session.indices:
                (session.chunk_dir / f"{index:06d}.pcm").write_bytes(data)
                session.indices.add(index)
                session.byte_count += len(data)
            session.last_activity = time.monotonic()
            return session

    def end(self, session_id: str, summary: dict) -> Path | None:
        session_id = _valid_session_id(session_id)
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.meta["end"] = summary
        return self._save(session)

    def finalize_idle(self, timeout_s: float = IDLE_TIMEOUT_S) -> list[Path]:
        now = time.monotonic()
        with self._lock:
            idle = [sid for sid, s in self._sessions.items() if now - s.last_activity > timeout_s]
            sessions = [self._sessions.pop(sid) for sid in idle]
        return [self._save(session) for session in sessions]

    def _save(self, session: Session) -> Path:
        directory = _finalize(session)
        if self.on_saved is not None:
            self.on_saved(directory, session.meta)
        return directory

    def finalize_all(self) -> list[Path]:
        return self.finalize_idle(timeout_s=-1.0)

    def _create(self, session_id: str, meta: dict) -> Session:
        label = re.sub(r"[^a-z0-9_-]", "", str(meta.get("label", "unlabeled")).lower())[:24]
        name = f"{datetime.now():%Y%m%d-%H%M%S}_{label or 'unlabeled'}_{session_id[:8]}"
        session = Session(self.root / name, dict(meta))
        self._sessions[session_id] = session
        return session


def _finalize(session: Session) -> Path:
    """Join the chunks in order into audio.wav and record what was received."""
    sample_rate = int(session.meta.get("input_sample_rate", 48000))
    indices = sorted(session.indices)
    missing = sorted(set(range(indices[-1] + 1)) - session.indices) if indices else []
    frames = 0
    with wave.open(str(session.directory / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in indices:
            data = (session.chunk_dir / f"{index:06d}.pcm").read_bytes()
            data = data[: len(data) - (len(data) % 2)]
            wav.writeframes(data)
            frames += len(data) // 2
    session.meta["received"] = {
        "chunks": len(indices),
        "missing_chunks": missing,
        "frames": frames,
        "duration_s": round(frames / sample_rate, 3),
    }
    _write_json(session.directory / "meta.json", session.meta)
    shutil.rmtree(session.chunk_dir, ignore_errors=True)
    print(f"[saved] {session.directory}  ({frames / sample_rate:.1f} s at {sample_rate} Hz, "
          f"{len(indices)} chunks, {len(missing)} missing)")
    print(f"        analyze with:  python analyze.py \"{session.directory}\"")
    return session.directory


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    store: SessionStore

    def do_GET(self) -> None:
        if self.path == "/ping":
            self._reply(200, {"status": "ok", "server": "sonar-receiver",
                              "protocol": PROTOCOL_VERSION})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            body = self._read_body()
            if self.path == "/session/start":
                directory = self.store.start(_parse_json(body))
                print(f"[start] {directory.name}")
                self._reply(200, {"status": "ok"})
            elif self.path == "/chunk":
                session = self.store.add_chunk(
                    self.headers.get("X-Session-Id", ""),
                    int(self.headers.get("X-Chunk-Index", "-1")),
                    int(self.headers.get("X-Sample-Rate", "48000")),
                    body,
                )
                self._reply(200, {"status": "ok", "chunks": len(session.indices)})
            elif self.path == "/session/end":
                summary = _parse_json(body)
                directory = self.store.end(summary.get("session_id", ""), summary)
                self._reply(200, {"status": "ok", "saved": directory is not None})
            else:
                self._reply(404, {"error": "not found"})
        except ValueError as error:
            self._reply(400, {"error": str(error)})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("body too large")
        return self.rfile.read(length)

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # keep the console readable
        pass


def _valid_session_id(value) -> str:
    if not isinstance(value, str) or not SESSION_ID_PATTERN.match(value):
        raise ValueError("invalid or missing session id")
    return value


def _parse_json(body: bytes) -> dict:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def local_addresses() -> list[str]:
    """IPv4 addresses of this machine that the watch could reach."""
    addresses: set[str] = set()
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))   # no packet is sent; this only picks a route
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(a for a in addresses if not a.startswith("127."))


def make_server(host: str, port: int, out_dir: Path, handler_class: type = Handler,
                store: SessionStore | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (handler_class,), {"store": store or SessionStore(out_dir)})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive sonar recordings from the watch app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, default=Path("recordings"))
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # show progress even when output is piped

    server = make_server(args.host, args.port, args.out)
    store: SessionStore = server.RequestHandlerClass.store

    def watchdog() -> None:
        while True:
            time.sleep(5.0)
            store.finalize_idle()

    threading.Thread(target=watchdog, daemon=True).start()

    print(f"Sonar receiver listening on port {args.port}; recordings go to '{args.out}'.")
    for address in local_addresses():
        print(f"  Enter this in the watch app:  {address}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        store.finalize_all()
        server.server_close()


if __name__ == "__main__":
    main()
