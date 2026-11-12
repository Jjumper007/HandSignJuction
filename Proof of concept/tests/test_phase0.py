"""Tests for the Phase 0 tooling, run on synthetic recordings.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receiver import make_server  # noqa: E402
from simulate import send_session  # noqa: E402
from sonar.analysis import FAIL, MARGINAL, PASS, analyze, compare_motion  # noqa: E402
from sonar.chirp import ChirpSpec, make_chirp  # noqa: E402
from sonar.session import load_session  # noqa: E402
from sonar.synth import SCENARIOS, synthesize  # noqa: E402


def run_scenario(name: str):
    scenario = SCENARIOS[name]
    audio = synthesize(scenario).astype(np.float64) / 32768.0
    return analyze(audio, scenario.meta())


class ChirpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ChirpSpec(f0_hz=18000.0, f1_hz=21000.0, samples=600, sample_rate=48000.0)

    def test_length_and_range(self) -> None:
        chirp = make_chirp(self.spec)
        self.assertEqual(len(chirp), 600)
        self.assertLessEqual(float(np.max(np.abs(chirp))), 1.0)

    def test_energy_is_inside_the_band(self) -> None:
        spectrum = np.abs(np.fft.rfft(np.tile(make_chirp(self.spec), 40))) ** 2
        freqs = np.fft.rfftfreq(600 * 40, d=1.0 / 48000.0)
        inside = spectrum[(freqs >= 17800.0) & (freqs <= 21200.0)].sum()
        self.assertGreater(inside / spectrum.sum(), 0.98)

    def test_taper_starts_and_ends_near_zero(self) -> None:
        chirp = make_chirp(self.spec)
        self.assertLess(abs(chirp[0]), 0.01)
        self.assertLess(abs(chirp[-1]), 0.01)


class AnalyzerTests(unittest.TestCase):
    def test_clean_recording_passes(self) -> None:
        report = run_scenario("good-still")
        self.assertEqual(report.verdict, PASS)
        self.assertGreater(report.metrics["band_snr_db"], 40.0)
        self.assertEqual(report.metrics["timing_jumps"], 0)

    def test_native_44k_recording_passes(self) -> None:
        self.assertEqual(run_scenario("rate-44k").verdict, PASS)

    def test_low_sample_rate_fails(self) -> None:
        report = run_scenario("rate-16k")
        self.assertEqual(report.verdict, FAIL)
        self.assertEqual(report.status_of("sample_rate"), FAIL)

    def test_filtered_microphone_fails(self) -> None:
        report = run_scenario("filtered-16k")
        self.assertEqual(report.verdict, FAIL)
        self.assertEqual(report.status_of("band_snr"), FAIL)

    def test_silent_speaker_fails(self) -> None:
        report = run_scenario("silent-speaker")
        self.assertEqual(report.verdict, FAIL)
        self.assertEqual(report.status_of("direct_path"), FAIL)

    def test_tilted_response_is_flagged(self) -> None:
        report = run_scenario("rolloff")
        self.assertEqual(report.status_of("band_flatness"), MARGINAL)
        self.assertEqual(report.verdict, MARGINAL)

    def test_dropped_samples_are_flagged(self) -> None:
        report = run_scenario("dropouts")
        self.assertEqual(report.status_of("timing"), MARGINAL)
        self.assertEqual(report.metrics["timing_jumps"], 2)
        self.assertEqual(report.status_of("direct_path"), PASS)

    def test_moving_fingers_are_detected(self) -> None:
        check = compare_motion(run_scenario("good-still"), run_scenario("good-moving"))
        self.assertEqual(check.status, PASS)

    def test_two_still_recordings_show_no_motion(self) -> None:
        check = compare_motion(run_scenario("good-still"), run_scenario("good-still"))
        self.assertEqual(check.status, FAIL)


class ReceiverTests(unittest.TestCase):
    def test_upload_round_trip(self) -> None:
        scenario = SCENARIOS["good-moving"]
        samples = synthesize(scenario)
        with tempfile.TemporaryDirectory() as temp:
            server = make_server("127.0.0.1", 0, Path(temp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                send_session(f"127.0.0.1:{server.server_address[1]}", scenario, samples)
            finally:
                server.shutdown()
                server.server_close()
            sessions = [p for p in Path(temp).iterdir() if p.is_dir()]
            self.assertEqual(len(sessions), 1)
            audio, meta = load_session(sessions[0])
            self.assertEqual(meta["label"], "moving")
            self.assertEqual(meta["received"]["missing_chunks"], [])
            np.testing.assert_array_equal(np.round(audio * 32768.0).astype(np.int16), samples)


if __name__ == "__main__":
    unittest.main()
