import Foundation

/// Sends one recording session to the receiver over plain HTTP.
///
/// Requests go out strictly one after another, in order: session start, the
/// audio, session end. Audio that piles up while a request is in flight is
/// merged into the next request, so the delay adapts to the network: a fast
/// link sends small pieces often, a slow one sends bigger pieces less often.
/// A failed request is retried a few times; after that the queue pauses until
/// `retry()` is called, and nothing is lost.
///
/// Plain HTTP requests through `URLSession` are the only networking that
/// watchOS allows every app to use, which is why this is not a socket stream.
///
/// All mutable state is confined to `queue`, hence `@unchecked Sendable`.
final class ChunkUploader: @unchecked Sendable {
    struct Progress {
        var sentSeconds = 0.0
        var pendingSeconds = 0.0
        var hasFailed = false
        var isFinished = false
        var lastError: String?
    }

    /// Work that has not been turned into a request yet.
    private enum Pending {
        case metadata(Data)
        case audio(Data)
        case end(totalFrames: Int)
    }

    /// A request that has been built. It is resent unchanged on retry, so the
    /// receiver can recognise a repeated chunk by its index.
    private struct Outgoing {
        let request: URLRequest
        let body: Data
        let audioBytes: Int
        let isLast: Bool
    }

    private static let maxAttempts = 5
    /// Upper bound for one request: about ten seconds of audio.
    private static let maxRequestBytes = 1_000_000

    private let baseURL: URL
    private let sessionID: String
    private let onProgress: (Progress) -> Void
    private let queue = DispatchQueue(label: "sonar.chunk-uploader")
    private let urlSession: URLSession

    // Everything below is touched only on `queue`.
    private var pending: [Pending] = []
    private var outgoing: Outgoing?
    private var isSending = false
    private var attempts = 0
    private var nextChunkIndex = 0
    private var sampleRate = 48_000
    private var sentAudioBytes = 0
    private var hasFailed = false
    private var isFinished = false
    private var lastError: String?

    /// `onProgress` is called on the main thread.
    init(baseURL: URL, sessionID: String, onProgress: @escaping (Progress) -> Void) {
        self.baseURL = baseURL
        self.sessionID = sessionID
        self.onProgress = onProgress
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 10
        configuration.waitsForConnectivity = false
        urlSession = URLSession(configuration: configuration)
    }

    /// Accepts "192.168.1.10:8765" (plain HTTP on the local network) or a full
    /// URL such as "https://example.trycloudflare.com". Returns nil when malformed.
    static func makeBaseURL(from address: String) -> URL? {
        let trimmed = address.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let text = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
        guard let url = URL(string: text), url.host != nil else { return nil }
        return url
    }

    /// Asks the receiver whether it is reachable. `completion` runs on the main thread.
    static func ping(baseURL: URL, completion: @escaping (String) -> Void) {
        var request = URLRequest(url: baseURL.appendingPathComponent("ping"))
        request.timeoutInterval = 5
        URLSession.shared.dataTask(with: request) { _, response, error in
            let result: String
            if let error = error {
                result = "Failed: \(error.localizedDescription)"
            } else if let status = (response as? HTTPURLResponse)?.statusCode, status == 200 {
                result = "Connected"
            } else {
                result = "Unexpected reply from the server"
            }
            DispatchQueue.main.async {
                completion(result)
            }
        }.resume()
    }

    /// Queues the session metadata. It is always sent before any audio.
    func begin(metadata: [String: Any]) {
        guard let body = try? JSONSerialization.data(withJSONObject: metadata) else { return }
        queue.async {
            self.pending.insert(.metadata(body), at: 0)
            self.publishAndSend()
        }
    }

    /// Queues a piece of mono 16-bit little-endian PCM.
    func enqueueChunk(_ pcm: Data, sampleRate: Int) {
        queue.async {
            self.sampleRate = max(1, sampleRate)
            self.pending.append(.audio(pcm))
            self.publishAndSend()
        }
    }

    /// Queues the end-of-session marker; the receiver then writes the WAV file.
    func finish(totalFrames: Int) {
        queue.async {
            self.pending.append(.end(totalFrames: totalFrames))
            self.publishAndSend()
        }
    }

    /// Resumes a queue that paused after repeated failures.
    func retry() {
        queue.async {
            self.hasFailed = false
            self.attempts = 0
            self.publishAndSend()
        }
    }

    private func publishAndSend() {
        publish()
        sendNext()
    }

    private func publish() {
        var waitingBytes = outgoing?.audioBytes ?? 0
        for item in pending {
            if case .audio(let pcm) = item {
                waitingBytes += pcm.count
            }
        }
        let bytesPerSecond = Double(2 * sampleRate)
        var snapshot = Progress()
        snapshot.sentSeconds = Double(sentAudioBytes) / bytesPerSecond
        snapshot.pendingSeconds = Double(waitingBytes) / bytesPerSecond
        snapshot.hasFailed = hasFailed
        snapshot.isFinished = isFinished
        snapshot.lastError = lastError
        let report = onProgress
        DispatchQueue.main.async {
            report(snapshot)
        }
    }

    private func sendNext() {
        guard !isSending, !hasFailed else { return }
        if outgoing == nil {
            outgoing = makeOutgoing()
        }
        guard let current = outgoing else { return }
        isSending = true
        let task = urlSession.uploadTask(with: current.request, from: current.body) { [weak self] _, response, error in
            guard let self = self else { return }
            self.queue.async {
                self.handleResult(response: response, error: error)
            }
        }
        task.resume()
    }

    /// Builds the next request from the head of the queue.
    private func makeOutgoing() -> Outgoing? {
        guard let first = pending.first else { return nil }
        switch first {
        case .metadata(let body):
            pending.removeFirst()
            return Outgoing(request: jsonRequest(path: "session/start"), body: body,
                            audioBytes: 0, isLast: false)

        case .audio:
            // Take every piece of audio that is waiting, up to the size limit.
            var body = Data()
            while let next = pending.first {
                guard case .audio(let pcm) = next else { break }
                if !body.isEmpty && body.count + pcm.count > ChunkUploader.maxRequestBytes { break }
                body.append(pcm)
                pending.removeFirst()
            }
            var request = URLRequest(url: baseURL.appendingPathComponent("chunk"))
            request.httpMethod = "POST"
            request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            request.setValue(sessionID, forHTTPHeaderField: "X-Session-Id")
            request.setValue(String(nextChunkIndex), forHTTPHeaderField: "X-Chunk-Index")
            request.setValue(String(sampleRate), forHTTPHeaderField: "X-Sample-Rate")
            nextChunkIndex += 1
            return Outgoing(request: request, body: body, audioBytes: body.count, isLast: false)

        case .end(let totalFrames):
            pending.removeFirst()
            let summary: [String: Any] = ["session_id": sessionID, "total_frames": totalFrames]
            let body = (try? JSONSerialization.data(withJSONObject: summary))
                ?? Data("{\"session_id\":\"\(sessionID)\"}".utf8)
            return Outgoing(request: jsonRequest(path: "session/end"), body: body,
                            audioBytes: 0, isLast: true)
        }
    }

    private func jsonRequest(path: String) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    private func handleResult(response: URLResponse?, error: Error?) {
        isSending = false
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if error == nil, (200..<300).contains(status) {
            if let sent = outgoing {
                sentAudioBytes += sent.audioBytes
                if sent.isLast { isFinished = true }
            }
            outgoing = nil
            attempts = 0
            lastError = nil
            publishAndSend()
            return
        }

        attempts += 1
        lastError = error?.localizedDescription ?? "HTTP \(status)"
        if attempts >= ChunkUploader.maxAttempts {
            hasFailed = true
            publish()
        } else {
            publish()
            queue.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                self?.sendNext()
            }
        }
    }
}
