# Finger sonar on a smartwatch: Phase 0

Phase 0 answers one question before anything else is built:
**can an Apple Watch emit an 18-21 kHz chirp from its speaker and record it,
unprocessed, with its own microphone at the same time?**

| Piece | What it does | Where it runs |
|---|---|---|
| `watch-app/Sources/` | watchOS app: plays the chirp, records the microphone, uploads the audio | Apple Watch (built on a Mac) |
| `receiver.py` | HTTP server that stores each recording as a WAV file | Mac or PC |
| `analyze.py` | Verdict and diagnostic figure for each recording | Mac or PC |
| `simulate.py` | Synthetic recordings, to test the tooling without hardware | Mac or PC |

The Python side is tested (`python -m unittest discover -s tests`).
**The watch app has never been compiled or run**: it was written on Windows.
Expect to fix a few build errors on the first try.

## 1. Python setup

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulate.py good-still good-moving
python analyze.py --latest 2       # should print PASS twice and a finger-motion PASS
```

## 2. Build the watch app (Mac with Xcode 26 or newer)

You need the iPhone that is paired with the watch, its cable, and an Apple
Account signed in under Xcode > Settings > Apple Accounts (a free one works;
the app then expires after 7 days).

1. **Create the project.** File > New > Project > watchOS > App. Product name
   `SonarProbe`, your personal team, **Watch-only App**, no tests. Save it
   inside `watch-app/`.
2. **Add the sources.** In Finder, copy the nine files from `watch-app/Sources/`
   into `watch-app/SonarProbe/SonarProbe Watch App/`, replacing the template's
   `ContentView.swift` and `SonarProbeApp.swift`. If they do not show up in
   Xcode, drag them into the "SonarProbe Watch App" group and tick the target.
3. **Target settings** (select the project, then the "SonarProbe Watch App" target):
   - *Info* tab, add two keys:
     - `Privacy - Microphone Usage Description` = `Records the echo of the emitted chirp.`
     - `App Transport Security Settings` (dictionary) containing `Allow Arbitrary Loads` = `YES`
   - *Signing & Capabilities* > `+ Capability` > **Background Modes** > tick **Audio**.
     (Equivalent: an Info key `UIBackgroundModes`, an array holding `audio`.)
   - *Build Settings*, search "isolation": set **Default Actor Isolation** to
     `nonisolated`. If the setting does not exist, skip this.
4. **Set the receiver address** in `AppConfig.swift` (`defaultServerAddress`),
   so you do not have to type it on the watch.
5. **Connect the devices.** Plug the iPhone into the Mac, unlock it and tap
   Trust. Turn on Developer Mode on the iPhone and on the watch (Settings >
   Privacy & Security > Developer Mode; it restarts the device and only
   appears after Xcode has seen the device). Keep the watch unlocked, on the
   wrist and on the same Wi-Fi as the Mac.
6. **Run.** Choose the watch as the run destination and press Cmd+R. The first
   connection can take a long time while Xcode copies symbols from the watch.

## 3. Run the test

1. Start the receiver: `python receiver.py`. It prints the address to use.
2. On the watch, disconnect any Bluetooth headphones and turn off Silent Mode.
   Open **Volume** in the app and turn the Digital Crown to the maximum.
3. **Settings > Test connection** should say "Connected".
4. Record three sessions of about 15 seconds each, keeping the wrist raised so
   the screen stays on:
   - **Hand still**: watch on the wrist, hand relaxed and motionless.
   - **Fingers moving**: open and close the fingers continuously; keep the wrist still.
   - **Watch off wrist** (optional): watch lying on a table.
5. Analyze: `python analyze.py --latest 3`

While recording, the watch itself shows the first answer under **Signal**:
`Chirp: Detected / Weak / Not detected` and the gain in dB, plus the real
microphone sample rate under **Hardware**. These need no network.

## 4. Reading the result

| Check | Meaning if it fails |
|---|---|
| `sample_rate` | The microphone runs below 44.1 kHz; the band cannot be recorded in this audio mode |
| `band_snr` | The chirp does not reach the recording: speaker too weak, or the OS filters it out |
| `band_coverage`, `band_flatness` | Only part of the band survives; range resolution suffers |
| `direct_path` | The matched filter cannot lock onto the chirp |
| `timing` | Samples are dropped or clocks drift; echo profiles will need re-alignment |
| `finger_motion` | Compares "moving" against "still": is finger movement visible in the echo? |

The thresholds are engineering starting points, not values calibrated on real hardware.

If the chirp is not detected, try in this order: check the volume, switch
**Measurement mode** off in Settings, then try the **16-19 kHz** band (audible
to some people).

## 5. If the upload does not work

Apple states that a watchOS app reaching a host on the local network may not
work by default, so this is the most likely thing to fail. In order:

1. Check the Mac firewall and that the Mac and the iPhone share a Wi-Fi network.
   A network that isolates clients (common at events) will not work; use the
   iPhone's hotspot with the Mac connected to it.
2. Let the watch use Wi-Fi directly: on the iPhone, turn off Bluetooth and
   Wi-Fi in the **Settings app** (not Control Center), then retry.
3. Use a public HTTPS tunnel, which the watch reaches like any website:
   `cloudflared tunnel --url http://localhost:8765`, then enter the printed
   `https://....trycloudflare.com` address in the app. The audio, including
   room sound, then passes through a third-party service, and the address is
   public while the tunnel runs.
4. With no network at all, the numbers on the watch screen already answer the
   first question. Write them down.

## Not verified yet

- Whether the watch speaker and microphone respond at 18-21 kHz.
- Whether measurement mode really bypasses system processing on watchOS.
- Whether the in-app volume control affects this audio session.
- Whether recording continues when the wrist drops.
- Whether a lone watch-only target built from a free account installs without issues.
