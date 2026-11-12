import Foundation

/// Compile-time defaults. Edit `defaultServerAddress` before building so you
/// do not have to type an IP address on the watch.
enum AppConfig {
    /// Where `receiver.py` can be reached: "host:port" for plain HTTP on the
    /// local network, or a full URL such as "https://name.trycloudflare.com".
    static let defaultServerAddress = "10.133.1.70:8765"

    static let protocolVersion = 1

    /// One chirp lasts 12.5 ms (600 samples at 48 kHz), repeated with no gap.
    static let chirpDurationSeconds = 0.0125
    /// Raised-cosine fade at each end of the chirp, to soften the audible click.
    static let chirpTaperSamples = 30
    static let chirpAmplitude: Float = 0.8

    /// Silence recorded before the speaker starts; it is the noise reference.
    static let emissionDelaySeconds = 1.0
    /// Audio is handed to the uploader in pieces of this length. The uploader
    /// merges pieces when the network is slower than that.
    static let chunkSeconds = 0.2
    /// Recording stops by itself after this long. Kept short on purpose: the
    /// watch may withdraw network access from a long-running audio session.
    static let maxRecordingSeconds = 30.0
    /// The live view runs for longer; it still stops by itself to save battery.
    static let maxLiveSeconds = 300.0
}

/// Frequency band swept by the chirp.
enum ChirpBand: String, CaseIterable, Identifiable {
    case high
    case low

    var id: String { rawValue }

    var title: String {
        switch self {
        case .high: return "18-21 kHz"
        case .low: return "16-19 kHz"
        }
    }

    var startFrequency: Double {
        switch self {
        case .high: return 18_000
        case .low: return 16_000
        }
    }

    var endFrequency: Double { startFrequency + 3_000 }
}

/// What the hand is doing during the recording. The raw value is stored with
/// the audio, so pose recordings double as labelled training data.
enum RecordingLabel: String, CaseIterable, Identifiable {
    // Hand poses, held still. "closed" and "open" calibrate the live view.
    case closed
    case open
    case half
    // Continuous stream for the live view on the computer.
    case live
    // Signal checks.
    case still
    case moving
    case offWrist = "off-wrist"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .closed: return "Closed fist"
        case .open: return "Open hand"
        case .half: return "Half open"
        case .live: return "Live view"
        case .still: return "Hand still"
        case .moving: return "Fingers moving"
        case .offWrist: return "Watch off wrist"
        }
    }

    var maxRecordingSeconds: Double {
        self == .live ? AppConfig.maxLiveSeconds : AppConfig.maxRecordingSeconds
    }
}

/// Everything the user can change before a recording.
struct RecordingSettings {
    var serverAddress: String
    var band: ChirpBand
    var label: RecordingLabel
    var prefersMeasurementMode: Bool
}
