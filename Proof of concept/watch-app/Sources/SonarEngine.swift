import AVFoundation
import WatchKit

enum SonarError: LocalizedError {
    case noMicrophone
    case noSpeaker
    case chirpBufferFailed

    var errorDescription: String? {
        switch self {
        case .noMicrophone: return "The microphone reported no usable audio format."
        case .noSpeaker: return "The speaker reported no usable audio format."
        case .chirpBufferFailed: return "Could not build the chirp buffer."
        }
    }
}

/// Plays the chirp on the speaker while capturing raw microphone audio.
///
/// `start`, `stop` and the session methods must be called on the main thread.
/// Audio arrives on a background thread, where chunks are handed to the
/// uploader; `onStatus` is always called on the main thread.
///
/// State shared with the audio thread is guarded by `lock`, hence `@unchecked Sendable`.
final class SonarEngine: @unchecked Sendable {
    /// What the hardware actually gave us for this session.
    struct SessionInfo {
        let inputSampleRate: Double
        let inputChannels: Int
        let outputSampleRate: Double
        let audioSessionMode: String
        let outputVolume: Float
        let chirp: ChirpSpec

        /// Metadata in the format `receiver.py` and `analyze.py` expect.
        func metadata(sessionID: String, label: RecordingLabel) -> [String: Any] {
            let device = WKInterfaceDevice.current()
            return [
                "protocol": AppConfig.protocolVersion,
                "session_id": sessionID,
                "label": label.rawValue,
                "input_sample_rate": Int(inputSampleRate),
                "input_channels": inputChannels,
                "output_sample_rate": Int(outputSampleRate),
                "sample_format": "int16le",
                "chirp": [
                    "f0_hz": chirp.startFrequency,
                    "f1_hz": chirp.endFrequency,
                    "samples": chirp.samples,
                    "taper_samples": chirp.taperSamples,
                    "amplitude": Double(AppConfig.chirpAmplitude),
                ] as [String: Any],
                "emission_delay_s": AppConfig.emissionDelaySeconds,
                "audio_session_mode": audioSessionMode,
                "output_volume": Double(outputVolume),
                "device": [
                    "model": device.model,
                    "system": "\(device.systemName) \(device.systemVersion)",
                ],
                "started_at": ISO8601DateFormatter().string(from: Date()),
            ]
        }
    }

    /// Live levels shown on the watch. All levels are in dB relative to full scale.
    struct Status {
        var elapsedSeconds = 0.0
        var totalLevel = -120.0
        var bandLevel: Double?
        var baselineBandLevel: Double?
        var isEmitting = false

        /// How far the chirp band rose when the speaker started.
        var chirpGain: Double? {
            guard isEmitting, let band = bandLevel, let baseline = baselineBandLevel else { return nil }
            return band - baseline
        }
    }

    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var generation = 0

    // Shared with the audio thread; guarded by `lock`.
    private let lock = NSLock()
    private var pending: [Int16] = []
    private var chunkFrames = 0
    private var capturedFrames = 0
    private var inputSampleRate = 0.0
    private var meter: BandLevelMeter?
    private var baselinePower = 0.0
    private var baselineFrames = 0
    private var emissionStartFrame: Int?
    private var emittingPower: Double?
    private var uploader: ChunkUploader?
    private var onStatus: ((Status) -> Void)?

    /// Asks for microphone permission. `completion` runs on the main thread.
    static func requestMicrophoneAccess(completion: @escaping (Bool) -> Void) {
        AVAudioApplication.requestRecordPermission { granted in
            DispatchQueue.main.async { completion(granted) }
        }
    }

    /// Activates the audio session without recording.
    ///
    /// watchOS restricts networking for ordinary apps and relaxes it while an
    /// audio session is active, so the session is held for connection tests
    /// and until the last chunk of a recording has been uploaded.
    func holdSession(prefersMeasurement: Bool) {
        let session = AVAudioSession.sharedInstance()
        _ = try? configureSession(session, prefersMeasurement: prefersMeasurement)
        try? session.setActive(true)
    }

    func releaseSession() {
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Starts capturing and, after a short silence, emitting. Captured audio is
    /// handed to `uploader` in chunks.
    func start(settings: RecordingSettings,
               uploader: ChunkUploader,
               onStatus: @escaping (Status) -> Void) throws -> SessionInfo {
        let session = AVAudioSession.sharedInstance()
        let mode = try configureSession(session, prefersMeasurement: settings.prefersMeasurementMode)
        try session.setActive(true)

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw SonarError.noMicrophone
        }
        var outputRate = engine.outputNode.outputFormat(forBus: 0).sampleRate
        if outputRate <= 0 { outputRate = session.sampleRate }
        guard outputRate > 0,
              let playerFormat = AVAudioFormat(standardFormatWithSampleRate: outputRate, channels: 1) else {
            throw SonarError.noSpeaker
        }

        let chirp = ChirpSpec(band: settings.band, sampleRate: outputRate)
        let periods = max(1, Int(outputRate) / chirp.samples)   // about one second of chirps
        guard let loopBuffer = ChirpGenerator.makeLoopBuffer(spec: chirp,
                                                             amplitude: AppConfig.chirpAmplitude,
                                                             periods: periods,
                                                             format: playerFormat) else {
            throw SonarError.chirpBufferFailed
        }

        let player = AVAudioPlayerNode()
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: playerFormat)

        lock.lock()
        pending.removeAll(keepingCapacity: true)
        chunkFrames = max(1, Int(AppConfig.chunkSeconds * inputFormat.sampleRate))
        capturedFrames = 0
        inputSampleRate = inputFormat.sampleRate
        meter = BandLevelMeter(cutoffFrequency: settings.band.startFrequency - 1_000,
                               sampleRate: inputFormat.sampleRate)
        baselinePower = 0
        baselineFrames = 0
        emissionStartFrame = nil
        emittingPower = nil
        self.uploader = uploader
        self.onStatus = onStatus
        lock.unlock()

        // The system only honours tap buffers of 100 to 400 ms. The closure is
        // marked @Sendable so it never inherits main-thread isolation: it runs
        // on the audio thread.
        let tapFrames = AVAudioFrameCount(0.1 * inputFormat.sampleRate)
        input.installTap(onBus: 0, bufferSize: tapFrames, format: inputFormat) { @Sendable [weak self] buffer, _ in
            self?.process(buffer)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        self.engine = engine
        self.player = player

        // Record a stretch of silence first, then start the speaker.
        generation += 1
        let startedGeneration = generation
        player.scheduleBuffer(loopBuffer, at: nil, options: .loops, completionHandler: nil)
        DispatchQueue.main.asyncAfter(deadline: .now() + AppConfig.emissionDelaySeconds) { [weak self] in
            guard let self = self, self.generation == startedGeneration else { return }
            self.player?.play()
            self.lock.lock()
            self.emissionStartFrame = self.capturedFrames
            self.lock.unlock()
        }

        return SessionInfo(inputSampleRate: inputFormat.sampleRate,
                           inputChannels: Int(inputFormat.channelCount),
                           outputSampleRate: outputRate,
                           audioSessionMode: mode,
                           outputVolume: session.outputVolume,
                           chirp: chirp)
    }

    /// Stops playback and capture, hands the last partial chunk to the uploader
    /// and returns the number of frames captured. The audio session stays
    /// active; call `releaseSession` once the upload is over.
    @discardableResult
    func stop() -> Int {
        generation += 1
        engine?.inputNode.removeTap(onBus: 0)
        player?.stop()
        engine?.stop()
        engine = nil
        player = nil

        lock.lock()
        let remainder = pending
        pending.removeAll()
        let totalFrames = capturedFrames
        let sampleRate = Int(inputSampleRate)
        if !remainder.isEmpty {
            uploader?.enqueueChunk(SonarEngine.pcmData(remainder), sampleRate: sampleRate)
        }
        uploader = nil
        onStatus = nil
        lock.unlock()
        return totalFrames
    }

    /// Measurement mode asks the system for as little signal processing as
    /// possible. If the watch refuses it, fall back to the default mode.
    private func configureSession(_ session: AVAudioSession, prefersMeasurement: Bool) throws -> String {
        if prefersMeasurement {
            do {
                try session.setCategory(.playAndRecord, mode: .measurement, options: [])
                return "measurement"
            } catch {
                // Fall through to the default mode.
            }
        }
        try session.setCategory(.playAndRecord, mode: .default, options: [])
        return "default"
    }

    /// Runs on the audio thread for every captured buffer.
    private func process(_ buffer: AVAudioPCMBuffer) {
        let samples = SonarEngine.firstChannel(of: buffer)
        guard !samples.isEmpty else { return }

        var chunks: [Data] = []
        var status = Status()

        lock.lock()
        let levels = meter?.process(samples) ?? (total: 0, band: 0)
        let bandIsUsable = meter?.isUsable ?? false
        let middleSeconds = (Double(capturedFrames) + Double(samples.count) / 2) / inputSampleRate
        capturedFrames += samples.count

        // The stretch before the speaker starts is the noise reference.
        if middleSeconds > 0.2, middleSeconds < AppConfig.emissionDelaySeconds - 0.15 {
            baselinePower += levels.band * Double(samples.count)
            baselineFrames += samples.count
        }
        if let startFrame = emissionStartFrame {
            status.isEmitting = true
            let sinceStart = Double(capturedFrames - startFrame) / inputSampleRate
            if sinceStart > 0.3 {
                emittingPower = emittingPower.map { 0.8 * $0 + 0.2 * levels.band } ?? levels.band
            }
        }

        status.elapsedSeconds = Double(capturedFrames) / inputSampleRate
        status.totalLevel = BandLevelMeter.decibels(levels.total)
        if bandIsUsable {
            status.bandLevel = BandLevelMeter.decibels(emittingPower ?? levels.band)
            if baselineFrames > 0 {
                status.baselineBandLevel = BandLevelMeter.decibels(baselinePower / Double(baselineFrames))
            }
        }

        pending.append(contentsOf: samples)
        while pending.count >= chunkFrames {
            chunks.append(SonarEngine.pcmData(Array(pending.prefix(chunkFrames))))
            pending.removeFirst(chunkFrames)
        }
        // Hand the chunks over while still holding the lock, so `stop` can never
        // slip its final partial chunk in ahead of them. This only queues work.
        let sampleRate = Int(inputSampleRate)
        for chunk in chunks {
            uploader?.enqueueChunk(chunk, sampleRate: sampleRate)
        }
        let deliverStatus = onStatus
        lock.unlock()
        DispatchQueue.main.async {
            deliverStatus?(status)
        }
    }

    /// Converts the first channel of a buffer to 16-bit samples.
    private static func firstChannel(of buffer: AVAudioPCMBuffer) -> [Int16] {
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return [] }
        var samples = [Int16](repeating: 0, count: frames)
        if let floatData = buffer.floatChannelData {
            let channel = floatData[0]
            for index in 0..<frames {
                let value = channel[index]
                let clamped = value.isFinite ? max(-1.0, min(1.0, value)) : 0.0
                samples[index] = Int16((clamped * 32767.0).rounded())
            }
        } else if let intData = buffer.int16ChannelData {
            let channel = intData[0]
            for index in 0..<frames {
                samples[index] = channel[index]
            }
        } else {
            return []
        }
        return samples
    }

    /// Raw little-endian bytes (the watch CPU is little-endian).
    private static func pcmData(_ samples: [Int16]) -> Data {
        samples.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
