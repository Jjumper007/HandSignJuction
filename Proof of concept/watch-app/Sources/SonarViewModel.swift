import SwiftUI

/// Connects the screens to the audio engine and the uploader.
///
/// Everything here runs on the main thread. The engine and the uploader only
/// ever call the closures created here on the main thread as well.
@MainActor
final class SonarViewModel: ObservableObject {
    enum Phase {
        case idle
        case recording
        case uploading
        case uploadFailed
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var headline = "Ready"
    @Published private(set) var sessionInfo: SonarEngine.SessionInfo?
    @Published private(set) var status: SonarEngine.Status?
    @Published private(set) var upload = ChunkUploader.Progress()
    @Published private(set) var connectionResult = ""

    private let engine = SonarEngine()
    private var uploader: ChunkUploader?
    private var activeSessionID = ""
    private var activeLabel = RecordingLabel.still

    func start(settings: RecordingSettings) {
        guard phase == .idle else { return }
        guard let baseURL = ChunkUploader.makeBaseURL(from: settings.serverAddress) else {
            headline = "Invalid server address"
            return
        }
        headline = "Requesting microphone"
        SonarEngine.requestMicrophoneAccess { [weak self] granted in
            guard let self = self else { return }
            guard granted else {
                self.headline = "Microphone access denied"
                return
            }
            self.beginSession(settings: settings, baseURL: baseURL)
        }
    }

    func stop() {
        guard phase == .recording else { return }
        let totalFrames = engine.stop()
        uploader?.finish(totalFrames: totalFrames)
        phase = upload.hasFailed ? .uploadFailed : .uploading
        headline = upload.hasFailed ? "Upload failed" : "Uploading"
    }

    func retryUpload() {
        guard phase == .uploadFailed else { return }
        phase = .uploading
        headline = "Uploading"
        uploader?.retry()
    }

    /// Gives up on a recording that could not be uploaded.
    func discardUpload() {
        guard phase == .uploadFailed else { return }
        finishSession(headline: "Recording discarded")
    }

    func testConnection(address: String, prefersMeasurementMode: Bool) {
        guard phase == .idle else { return }
        guard let baseURL = ChunkUploader.makeBaseURL(from: address) else {
            connectionResult = "Invalid server address"
            return
        }
        connectionResult = "Testing"
        // Test under the same conditions as an upload: with the audio session active.
        engine.holdSession(prefersMeasurement: prefersMeasurementMode)
        ChunkUploader.ping(baseURL: baseURL) { [weak self] result in
            guard let self = self else { return }
            self.connectionResult = result
            if self.phase == .idle {
                self.engine.releaseSession()
            }
        }
    }

    private func beginSession(settings: RecordingSettings, baseURL: URL) {
        guard phase == .idle else { return }
        let sessionID = UUID().uuidString.lowercased()
        let uploader = ChunkUploader(baseURL: baseURL, sessionID: sessionID) { [weak self] progress in
            self?.uploadDidProgress(progress, sessionID: sessionID)
        }
        do {
            let info = try engine.start(settings: settings, uploader: uploader) { [weak self] status in
                self?.statusDidChange(status)
            }
            uploader.begin(metadata: info.metadata(sessionID: sessionID, label: settings.label))
            self.uploader = uploader
            activeSessionID = sessionID
            activeLabel = settings.label
            sessionInfo = info
            status = nil
            upload = ChunkUploader.Progress()
            phase = .recording
            headline = "Recording"
        } catch {
            engine.stop()
            engine.releaseSession()
            headline = "Audio error: \(error.localizedDescription)"
        }
    }

    private func statusDidChange(_ newStatus: SonarEngine.Status) {
        guard phase == .recording else { return }
        status = newStatus
        if newStatus.elapsedSeconds >= activeLabel.maxRecordingSeconds {
            stop()
        }
    }

    private func uploadDidProgress(_ progress: ChunkUploader.Progress, sessionID: String) {
        guard sessionID == activeSessionID else { return }
        upload = progress
        if phase == .recording {
            // A live stream with nowhere to go would only pile up in memory.
            if progress.hasFailed, activeLabel == .live {
                stop()
            }
            return
        }
        if progress.isFinished {
            finishSession(headline: "Saved on the laptop")
        } else if progress.hasFailed {
            phase = .uploadFailed
            headline = "Upload failed"
        }
    }

    private func finishSession(headline newHeadline: String) {
        uploader = nil
        activeSessionID = ""
        engine.releaseSession()
        phase = .idle
        headline = newHeadline
    }
}
